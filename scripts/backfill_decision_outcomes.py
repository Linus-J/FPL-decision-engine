#!/usr/bin/env python
"""Backfill ``DecisionLog.actual_outcome`` for past 'lineup' decisions once
their gameweek has finished. Purely additive -- never touches ``dry_run`` or
any decision-making logic. See plan/dashboard-v1.md for the design rationale
and known simplification (no autosub modelling, since ``bench_order`` isn't
persisted on lineup decisions today).
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from data.db import get_session
from optimiser.chips import Chip
from scripts.backtest import _score_squad

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _actual_points_by_player(db, season: str, gameweek: int) -> dict[int, int]:
    """Sums across rows per player -- a genuine DGW player has two rows
    (same gameweek, different opponent; see PlayerGameweekStats)."""
    rows = db.execute(
        text(
            "SELECT player_id, SUM(total_points) FROM player_gw_stats "
            "WHERE season = :season AND gameweek = :gw GROUP BY player_id"
        ),
        {"season": season, "gw": gameweek},
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _gw_finished(db, season: str, gameweek: int) -> bool:
    row = db.execute(
        text("SELECT finished FROM gameweeks WHERE season = :season AND id = :gw"),
        {"season": season, "gw": gameweek},
    ).fetchone()
    return bool(row and row[0])


def backfill(season: str = "2026-27") -> int:
    """Returns the number of DecisionLog rows updated."""
    db = get_session()
    updated = 0
    try:
        pending = db.execute(
            text(
                "SELECT id, gameweek, details FROM decision_log "
                "WHERE decision_type = 'lineup' AND actual_outcome IS NULL"
            )
        ).fetchall()
        if not pending:
            return 0

        chip_by_gw: dict[int, str | None] = {}
        for gw, details_json in db.execute(
            text("SELECT gameweek, details FROM decision_log WHERE decision_type = 'chip'")
        ).fetchall():
            chip_by_gw[gw] = json.loads(details_json).get("chip")

        for log_id, gameweek, details_json in pending:
            if not _gw_finished(db, season, gameweek):
                continue
            details = json.loads(details_json)
            squad_ids = details.get("squad_ids", [])
            starting_ids = details.get("starting_ids", [])
            captain_id = details.get("captain_id")
            vice_captain_id = details.get("vice_captain_id")
            if not squad_ids or not starting_ids or captain_id is None:
                logger.warning(
                    "Skipping decision %d (GW%d): incomplete lineup details", log_id, gameweek
                )
                continue

            actual_points = _actual_points_by_player(db, season, gameweek)
            chip = chip_by_gw.get(gameweek)
            actual = _score_squad(
                squad_ids=squad_ids,
                starting_ids=starting_ids,
                captain_id=captain_id,
                actual_points=actual_points,
                bench_boost=(chip == Chip.BENCH_BOOST.value),
                triple_captain=(chip == Chip.TRIPLE_CAPTAIN.value),
                vice_captain_id=vice_captain_id,
            )
            db.execute(
                text("UPDATE decision_log SET actual_outcome = :pts WHERE id = :id"),
                {"pts": actual, "id": log_id},
            )
            updated += 1
            logger.info("GW%d decision %d: actual_outcome=%d", gameweek, log_id, actual)

        db.commit()
    finally:
        db.close()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    args = parser.parse_args()
    n = backfill(args.season)
    logger.info("Backfilled %d decision(s)", n)


if __name__ == "__main__":
    main()
