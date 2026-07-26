"""whoscored.py — WhoScored raw-event adapter for player_match_events (P10 follow-up).

FBref's free match-summary table structurally lacks clearances/blocks/
recoveries (confirmed by inspecting raw cached HTML — plan/phase-2-xpts-engine.md
P10) — this was the root cause of both the DefCon underestimate and the
GK-bonus miscalibration (the same crippled CBI/CBIRT channel). WhoScored's
soccerdata reader exposes the raw Opta event stream (one row per action)
instead of a pre-aggregated summary, and DOES carry these as distinct event
types: BallRecovery, Clearance, Tackle, Interception, BlockedPass, plus
TakeOn (dribbles) — confirmed against a real probed match (2026-07-26).

This module aggregates that stream into counts and PATCHES them onto the
EXISTING player_match_events rows FBref already wrote — it does not re-derive
goals/assists/xG/key_passes (Understat already covers those, see
understat_xg.py) or minutes (player_gw_stats/FPL API already covers that).

Cross-source note: WhoScored's own game_id is a different ID space than
FBref's — rather than crosswalk match IDs, this aggregates by (player,
gameweek) via kickoff-date -> gameweek assignment (same approach
understat_xg.py already uses) and UPDATEs existing rows keyed on (player_id,
season, gameweek). For a rare DGW gameweek this puts the same 2-match sum on
both of that week's existing rows — an accepted simplification (P12 already
defers precise per-team DGW handling elsewhere in this codebase).
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from data.db import get_session
from data.ingestors.fbref import _build_name_map, _match_player
from scripts.backfill_odds import assign_gameweek

logger = logging.getLogger(__name__)

WHOSCORED_LEAGUE = "ENG-Premier League"
SEASON_MAP = {
    "2021-22": "2021-2022", "2022-23": "2022-2023", "2023-24": "2023-2024",
    "2024-25": "2024-2025", "2025-26": "2025-2026", "2026-27": "2026-2027",
}

# WhoScored event `type` -> our player_match_events count field. Limited to
# the fields FBref's summary table structurally can't provide (verified
# absent from real cached match HTML — see the P10 plan entry) plus dribbles
# (also absent from FBref's summary table).
EVENT_TYPE_FIELD = {
    "Tackle": "tackles",
    "Interception": "interceptions",
    "Clearance": "clearances",
    "BlockedPass": "blocks",
    "BallRecovery": "recoveries",
}
DRIBBLE_TYPE = "TakeOn"
UPDATE_FIELDS = (*EVENT_TYPE_FIELD.values(), "dribbles")


def aggregate_match_events(events: pd.DataFrame) -> pd.DataFrame:
    """Raw WhoScored event rows (one per action, from ``read_events``) -> one
    row per (game_id, player_id, player) with our count fields. Pure — no
    DB/network. A dribble only counts if the TakeOn was won (``outcome_type``
    == "Successful"); every other field is a plain per-type row count.
    """
    empty = pd.DataFrame(columns=["game_id", "player_id", "player", *UPDATE_FIELDS])
    if events.empty:
        return empty

    df = events.reset_index()
    df = df[df["player_id"].notna()].copy()
    if df.empty:
        return empty
    df["player_id"] = df["player_id"].astype(int)

    df["_agg_field"] = df["type"].map(EVENT_TYPE_FIELD)
    is_dribble = (df["type"] == DRIBBLE_TYPE) & (df["outcome_type"] == "Successful")
    df.loc[is_dribble, "_agg_field"] = "dribbles"
    df = df.dropna(subset=["_agg_field"])
    if df.empty:
        return empty

    pivot = (
        df.groupby(["game_id", "player_id", "_agg_field"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(UPDATE_FIELDS), fill_value=0)
        .reset_index()
    )
    names = (
        events.reset_index()[["game_id", "player_id", "player"]]
        .dropna(subset=["player_id"])
        .assign(player_id=lambda d: d["player_id"].astype(int))
        .drop_duplicates(subset=["game_id", "player_id"])
    )
    return pivot.merge(names, on=["game_id", "player_id"], how="left")


def sum_by_gameweek(
    agg: pd.DataFrame,
    kickoff_of: dict,
    deadlines: dict[int, datetime],
    name_map: dict[str, int],
) -> tuple[dict[tuple[int, int], dict[str, int]], int]:
    """(game_id, player) counts -> summed (player_id, gameweek) counts (DGW-safe,
    same simplification understat_xg.py uses). Also resolves WhoScored's raw
    player name to our players.id via fuzzy name match. Pure given its inputs
    (no DB/network — deadlines/name_map are pre-fetched by the caller)."""
    totals: dict[tuple[int, int], dict[str, int]] = {}
    unmatched = 0
    for _, row in agg.iterrows():
        pid = _match_player(str(row.get("player", "")), name_map)
        kickoff = kickoff_of.get(row["game_id"])
        gw = assign_gameweek(kickoff, deadlines) if kickoff is not None else None
        if not pid or gw is None:
            unmatched += 1
            continue
        key = (int(pid), int(gw))
        bucket = totals.setdefault(key, dict.fromkeys(UPDATE_FIELDS, 0))
        for field in UPDATE_FIELDS:
            bucket[field] += int(row.get(field, 0))
    return totals, unmatched


def _load_deadlines(season: str) -> dict[int, datetime]:
    from data.models import Gameweek
    db = get_session()
    try:
        rows = db.query(Gameweek.id, Gameweek.deadline_time).filter(
            Gameweek.season == season
        ).all()
        return {gw: dl for gw, dl in rows if dl is not None}
    finally:
        db.close()


def _write_defensive_counts(season: str, totals: dict[tuple[int, int], dict[str, int]]) -> int:
    """UPDATE existing player_match_events rows (written by fbref.py) with the
    richer WhoScored counts. Never INSERTs — a player/gameweek with no
    existing row (no FBref match report) is skipped, the same fallback
    posture the rest of this pipeline uses (never invents an event row)."""
    db = get_session()
    written = 0
    try:
        for (pid, gw), counts in totals.items():
            result = db.execute(
                text("""
                    UPDATE player_match_events
                    SET tackles = :tackles, interceptions = :interceptions,
                        clearances = :clearances, blocks = :blocks,
                        recoveries = :recoveries, dribbles = :dribbles
                    WHERE player_id = :pid AND season = :season AND gameweek = :gw
                """),
                {**counts, "pid": pid, "season": season, "gw": gw},
            )
            written += result.rowcount or 0
        db.commit()
    finally:
        db.close()
    return written


def ingest_whoscored_season(  # pragma: no cover - live network + browser only
    season: str,
    *,
    no_cache: bool = False,
    path_to_browser: str | None = None,
    headless: bool = False,
) -> tuple[int, int]:
    """Scrape one PL season's WhoScored event stream and patch clearances/
    blocks/interceptions/tackles/recoveries/dribbles onto the EXISTING
    player_match_events rows (written by ``fbref.ingest_fbref_season`` — run
    that FIRST, this only updates rows it already created). Returns
    ``(rows_updated, unmatched player-games)``.
    """
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError(
            "whoscored ingest needs soccerdata (+ a browser), intentionally not "
            "a core dependency. Run with soccerdata layered on for that run:\n"
            "    WHOSCORED_HEADED=1 uv run --with soccerdata "
            "python scripts/scrape_whoscored.py 2025-26"
        ) from exc

    sd_season = SEASON_MAP.get(season)
    if not sd_season:
        raise ValueError(f"No WhoScored season mapping for {season!r}")

    ws_kwargs: dict = {
        "leagues": WHOSCORED_LEAGUE, "seasons": sd_season,
        "no_cache": no_cache, "headless": headless,
    }
    if path_to_browser:
        ws_kwargs["path_to_browser"] = path_to_browser
    ws = sd.WhoScored(**ws_kwargs)

    schedule = ws.read_schedule().reset_index()
    kickoff_of = dict(zip(schedule["game_id"], schedule["date"], strict=False))

    events = ws.read_events(output_fmt="events")
    agg = aggregate_match_events(events)

    deadlines = _load_deadlines(season)
    name_map = _build_name_map()
    totals, unmatched = sum_by_gameweek(agg, kickoff_of, deadlines, name_map)

    written = _write_defensive_counts(season, totals)
    logger.info(
        "WhoScored %s: %d player-GW rows updated, %d unmatched",
        season, written, unmatched,
    )
    return written, unmatched
