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
    query = text("""
        SELECT p.id AS player_id, p.fpl_id, p.web_name, p.position, p.now_cost,
               t.short_name AS team_short
        FROM players p
        JOIN teams t ON t.id = p.team_id
    """)
    return pd.read_sql(query, db.bind)


def _fallback_lineup(db: Session) -> tuple[list[int], int | None, int | None, int]:
    """Latest logged lineup: (starting_ids, captain_id, vice_captain_id, gameweek)."""
    row = db.execute(
        text(
            "SELECT details, gameweek FROM decision_log WHERE decision_type = 'lineup' "
            "ORDER BY created_at DESC LIMIT 1"
        )
    ).fetchone()
    if not row:
        return [], None, None, 0
    details = json.loads(row[0])
    return (
        details.get("starting_ids", []),
        details.get("captain_id"),
        details.get("vice_captain_id"),
        row[1],
    )


def get_current_squad(db: Session, team_id: int) -> pd.DataFrame:
    """Columns: player_id, fpl_id, web_name, position, now_cost, team_short,
    is_starting, multiplier, is_captain, is_vice_captain, xpts, gameweek."""
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

    if payload.get("picks"):
        picks = pd.DataFrame(payload["picks"]).rename(columns={"position": "squad_slot"})
        squad = players.merge(picks, left_on="fpl_id", right_on="element", how="inner")
        squad["is_starting"] = squad["multiplier"] > 0
    else:
        logger.info("No live FPL picks for team=%s; falling back to decision_log", team_id)
        starting_ids, captain_id, vice_captain_id, gw_used = _fallback_lineup(db)
        if not starting_ids:
            return pd.DataFrame()
        squad = players[players["player_id"].isin(starting_ids)].copy()
        squad["is_starting"] = True
        squad["multiplier"] = squad["player_id"].apply(lambda pid: 2 if pid == captain_id else 1)
        squad["is_captain"] = squad["player_id"] == captain_id
        squad["is_vice_captain"] = squad["player_id"] == vice_captain_id

    projections = get_latest_projections(gw_used)
    if not projections.empty:
        squad = squad.merge(projections[["player_id", "xpts"]], on="player_id", how="left")
    else:
        squad["xpts"] = 0.0
    squad["xpts"] = squad["xpts"].fillna(0.0)
    squad["gameweek"] = gw_used
    return squad
