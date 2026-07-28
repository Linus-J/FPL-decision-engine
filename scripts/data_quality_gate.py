#!/usr/bin/env python
"""data_quality_gate.py — run the reusable checks in data/quality_checks.py
against live data. Meant to be run periodically (e.g. alongside the normal
ingestion cadence) so the bug classes found in the 2026-07-28
data-completeness audit get caught automatically instead of surfacing later
as a captaincy-monopoly or a suspiciously-low backtest number.

    DB_PATH=fpl_bot_v2.db uv run --extra events python scripts/data_quality_gate.py

Exits 1 if any check reports an "error"-severity issue, 0 otherwise
(warnings are printed but don't fail the run).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def run_team_id_freshness_check() -> list:
    """Any player still in today's live FPL feed whose stored team_id
    disagrees with it right now (the Penders/Anselmino/Garnacho/Targett
    staleness found during the audit -- fixed by re-running the normal
    ingestion pipeline, not a code change)."""
    import httpx

    from data.db import get_session
    from data.models import Player
    from data.quality_checks import check_team_id_matches_live

    live = httpx.get(FPL_BOOTSTRAP_URL, timeout=15).json()
    live_player_team = {e["code"]: e["team"] for e in live["elements"]}

    db = get_session()
    try:
        rows = (
            db.query(Player.code, Player.web_name, Player.team_id)
            .filter(Player.code.isnot(None))
            .all()
        )
    finally:
        db.close()

    player_team_ids = {code: (web, tid) for code, web, tid in rows}
    return check_team_id_matches_live(player_team_ids, live_player_team)


def run_understat_coverage_check(season: str = "2025-26") -> list:
    """Name-match coverage against Understat's per-match feed for `season`.
    Would have caught the season-wide xG gap (only 14/524 players had any
    nonzero xg) immediately instead of it surfacing later as a captaincy
    monopoly. Best-effort: skipped (with a warning) if soccerdata isn't
    installed or the network call fails."""
    from data.quality_checks import check_name_match_coverage

    try:
        import soccerdata as sd

        from data.ingestors.fbref import _build_name_map, _match_player
        from data.ingestors.understat_xg import SEASON_MAP
    except ImportError:
        logger.warning("soccerdata not installed -- skipping Understat coverage check")
        return []

    yr = SEASON_MAP.get(season)
    if not yr:
        logger.warning("No Understat season mapping for %r -- skipping", season)
        return []

    try:
        name_map = _build_name_map()
        us = sd.Understat(leagues="ENG-Premier League", seasons=yr)
        names = us.read_player_match_stats().reset_index()["player"].unique()
    except Exception as exc:  # pragma: no cover - live network
        logger.warning("Understat coverage check failed to fetch live data: %s", exc)
        return []

    matched = sum(1 for n in names if _match_player(str(n), name_map) is not None)
    return check_name_match_coverage(f"understat/{season}", matched, len(names))


def main() -> int:
    issues = []
    issues += run_team_id_freshness_check()
    issues += run_understat_coverage_check()

    if not issues:
        logger.info("data quality gate: all checks passed")
        return 0

    has_error = False
    for issue in issues:
        level = logging.ERROR if issue.severity == "error" else logging.WARNING
        logger.log(level, "[%s] %s", issue.check, issue.message)
        has_error = has_error or issue.severity == "error"

    logger.info(
        "data quality gate: %d issue(s) found (%s)",
        len(issues),
        "FAILING" if has_error else "warnings only",
    )
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
