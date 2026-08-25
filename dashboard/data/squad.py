"""Live current-squad query for the dashboard: real FPL picks (public
endpoint, ground truth) joined to internal player rows and xPts projections,
falling back to the bot's own last logged lineup if the public endpoint has
nothing yet (e.g. between seasons)."""

from __future__ import annotations

import json
import logging

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from dashboard.fpl_public import get_picks
from projection.pipeline import _get_current_and_next_gw, get_latest_projections

logger = logging.getLogger(__name__)


def _players_by_fpl_id(db: Session) -> pd.DataFrame:
    """One row per fpl_id -- the CURRENT season's row.

    ``fpl_id`` is reused across seasons: 975 player rows carry only 756
    distinct fpl_ids on the live DB, because FPL renumbers its elements every
    summer and rows for departed players keep the id they had. Without the
    dedupe below, the caller's merge on ``fpl_id`` matched BOTH rows and
    returned a 19-player "15-man squad" -- Setford alongside Gabriel (both
    fpl_id 4), Roerslev alongside Van Hecke (112), Darwin alongside Semenyo
    (397), Lascelles alongside Gibbs-White (480). That reached the published
    site and the dashboard's Squad page; preflight's "site squad matches
    decision_log" check is what caught it (2026-08-25).

    ``updated_at`` is the discriminator rather than ``status``: run_full_ingest
    refreshes every player FPL currently serves on each run, so the live row is
    always the freshly-touched one, while a departed player's row keeps the
    timestamp of the last ingest that still saw him. ``status`` looks tempting
    -- the four collisions above were all status 'u' against 'a' -- but a
    current player can legitimately be 'u' after a mid-season transfer out,
    which would then drop the RIGHT row.
    """
    query = text("""
        SELECT player_id, fpl_id, web_name, position, now_cost, team_short
        FROM (
            SELECT p.id AS player_id, p.fpl_id, p.web_name, p.position,
                   p.now_cost, t.short_name AS team_short,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.fpl_id
                       ORDER BY p.updated_at DESC, p.id DESC
                   ) AS rn
            FROM players p
            JOIN teams t ON t.id = p.team_id
        )
        WHERE rn = 1
    """)
    return pd.read_sql(query, db.bind)


def _fallback_lineup(
    db: Session,
) -> tuple[list[int], list[int], int | None, int | None, int, float]:
    """Latest logged lineup: (squad_ids, starting_ids, captain_id,
    vice_captain_id, gameweek, projected_gain). ``projected_gain`` is the
    total XI xPts recorded at decision time -- the only xPts figure
    available for a true cold-start squad, since the cold-start build
    doesn't persist per-player projections to ``player_projections``."""
    row = db.execute(
        text(
            "SELECT details, gameweek, projected_gain FROM decision_log "
            "WHERE decision_type = 'lineup' ORDER BY created_at DESC LIMIT 1"
        )
    ).fetchone()
    if not row:
        return [], [], None, None, 0, 0.0
    details = json.loads(row[0])
    return (
        details.get("squad_ids", []),
        details.get("starting_ids", []),
        details.get("captain_id"),
        details.get("vice_captain_id"),
        row[1],
        row[2],
    )


def get_current_squad(db: Session, team_id: int) -> pd.DataFrame:
    """Columns: player_id, fpl_id, web_name, position, now_cost, team_short,
    is_starting, multiplier, is_captain, is_vice_captain, xpts, gameweek.

    ``xpts`` is NaN (not 0.0) when no per-player projection exists yet --
    e.g. a true pre-season cold-start squad, where ``player_projections``
    has no rows at all. ``squad.attrs["fallback_projected_total"]`` carries
    the decision's own recorded total XI xPts in that case, since that
    number does exist even though the per-player breakdown doesn't."""
    players = _players_by_fpl_id(db)
    if players.empty:
        return pd.DataFrame()

    current_gw, next_gw = _get_current_and_next_gw()
    payload: dict = {}
    gw_used = next_gw
    for gw in (next_gw, current_gw):
        payload = get_picks(team_id, gw)
        if payload.get("picks"):
            gw_used = gw
            break

    fallback_total: float | None = None
    if payload.get("picks"):
        picks = pd.DataFrame(payload["picks"]).rename(columns={"position": "squad_slot"})
        squad = players.merge(picks, left_on="fpl_id", right_on="element", how="inner")
        squad["is_starting"] = squad["multiplier"] > 0
    else:
        logger.info("No live FPL picks for team=%s; falling back to decision_log", team_id)
        squad_ids, starting_ids, captain_id, vice_captain_id, gw_used, projected_gain = (
            _fallback_lineup(db)
        )
        if not squad_ids:
            return pd.DataFrame()
        squad = players[players["player_id"].isin(squad_ids)].copy()
        squad["is_starting"] = squad["player_id"].isin(starting_ids)
        squad["multiplier"] = squad["player_id"].apply(lambda pid: 2 if pid == captain_id else 1)
        squad["is_captain"] = squad["player_id"] == captain_id
        squad["is_vice_captain"] = squad["player_id"] == vice_captain_id
        fallback_total = projected_gain

    projections = get_latest_projections(gw_used)
    if not projections.empty:
        cols = ["player_id", "xpts"]
        if "xpts_var" in projections.columns:
            cols.append("xpts_var")
        squad = squad.merge(projections[cols], on="player_id", how="left")
    else:
        squad["xpts"] = float("nan")
        squad["xpts_var"] = float("nan")
    squad["gameweek"] = gw_used
    squad.attrs["fallback_projected_total"] = (
        fallback_total if squad["xpts"].isna().all() else None
    )
    return squad
