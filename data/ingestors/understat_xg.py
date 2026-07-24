"""understat_xg.py — per-match xG/xA/key-passes from Understat → player_xg_stats.

The FREE solution to the xG gap: soccerdata's Understat reader returns real
per-player-per-match ``xg``/``xa``/``key_passes``/``shots`` (TLS-client, no
browser, no API key), which is exactly what P3 (goals) and P4 (assists) want.
This replaces the shots-only interim: P3's weight becomes real xG, P4's becomes
real xA/key-passes.

Player rows are matched to FPL ids by name; each match is assigned to a
gameweek via the per-season deadlines (T3a). Pure parsers are unit-tested; only
``ingest_understat_xg_season`` needs soccerdata + network.

Note: this feed's ``xg`` is total (incl. penalty xG); it has no separate npxG,
so ``npxg`` is stored equal to ``xg`` (a small over-count for penalty takers,
documented). ``xa`` is Understat expected-assists.
"""

from __future__ import annotations

import logging
from datetime import datetime

from data.ingestors.fbref import (
    _build_name_map,
    _match_player,
    _write_xg_rows,
    aggregate_xg_rows,
)
from scripts.backfill_odds import assign_gameweek

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "ENG-Premier League"
SEASON_MAP = {
    "2021-22": "2021", "2022-23": "2022", "2023-24": "2023",
    "2024-25": "2024", "2025-26": "2025", "2026-27": "2026",
}


def parse_game_date(game: str) -> datetime | None:
    """Understat ``game`` label starts 'YYYY-MM-DD ...' → date (fallback when a
    kickoff time isn't joined from the schedule)."""
    if not game:
        return None
    token = str(game).strip().split(" ")[0]
    try:
        return datetime.strptime(token, "%Y-%m-%d")
    except ValueError:
        return None


def understat_row_to_xg(row: dict) -> dict:
    """One Understat player-match row → player_xg_stats field dict (pure).
    npxg = xg (no penalty split in this feed)."""
    xg = round(float(row.get("xg", 0.0) or 0.0), 4)
    return {
        "xg": xg,
        "npxg": xg,
        "xa": round(float(row.get("xa", 0.0) or 0.0), 4),
        "shots": int(row.get("shots", 0) or 0),
        "key_passes": int(row.get("key_passes", 0) or 0),
    }


def _load_deadlines(season: str) -> dict[int, datetime]:
    from data.db import get_session
    from data.models import Gameweek
    db = get_session()
    try:
        rows = db.query(Gameweek.id, Gameweek.deadline_time).filter(
            Gameweek.season == season
        ).all()
        return {gw: dl for gw, dl in rows if dl is not None}
    finally:
        db.close()


def ingest_understat_xg_season(  # pragma: no cover - live network (no browser)
    season: str,
    *,
    no_cache: bool = False,
) -> tuple[int, int]:
    """Scrape one PL season of Understat per-match xG → player_xg_stats.
    Returns (rows_written, unmatched). Browserless (TLS client)."""
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError(
            "understat xg ingest needs soccerdata: "
            "`uv run --with soccerdata python scripts/scrape_understat_xg.py 2025-26`"
        ) from exc

    yr = SEASON_MAP.get(season)
    if not yr:
        raise ValueError(f"No Understat season mapping for {season!r}")

    us = sd.Understat(leagues=UNDERSTAT_LEAGUE, seasons=yr, no_cache=no_cache)
    pm = us.read_player_match_stats().reset_index()
    schedule = us.read_schedule().reset_index()
    kickoff_of = dict(zip(schedule["game_id"], schedule["date"], strict=False))

    deadlines = _load_deadlines(season)
    name_map = _build_name_map()

    per_match: list[tuple[int, int, dict]] = []
    unmatched = 0
    for rec in pm.to_dict("records"):
        player_id = _match_player(str(rec.get("player", "")), name_map)
        kickoff = kickoff_of.get(rec.get("game_id"))
        if kickoff is None:
            kickoff = parse_game_date(rec.get("game", ""))
        gw = assign_gameweek(kickoff, deadlines) if kickoff is not None else None
        if not player_id or gw is None:
            unmatched += 1
            continue
        per_match.append((player_id, int(gw), understat_row_to_xg(rec)))

    written = _write_xg_rows(season, aggregate_xg_rows(per_match))
    logger.info(
        "Understat xg %s: %d player-GW rows written, %d unmatched",
        season, written, unmatched,
    )
    return written, unmatched
