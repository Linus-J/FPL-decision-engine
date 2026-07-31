#!/usr/bin/env python
"""Backfill ``actual_outcome`` for past 'lineup' decisions once their
gameweek has finished -- both the real bot's ``decision_log`` and every
simulated persona's ``sim_decision_log`` (plan/simulation-engine-v1.md).
Purely additive -- never touches ``dry_run`` or any decision-making logic.
See plan/dashboard-v1.md for the design rationale and known simplification
(no autosub modelling, since ``bench_order`` isn't persisted on lineup
decisions today).
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

_VALID_TABLES = ("decision_log", "sim_decision_log")


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


def _backfill_table(
    db, season: str, table: str, sim_manager_id: int | None = None
) -> int:
    """``table`` is always one of ``_VALID_TABLES`` -- never user input --
    so the f-string below is safe; asserted for defense in depth.
    ``sim_manager_id`` scopes both the pending rows and the chip lookup to
    one persona (required, and only meaningful, for ``sim_decision_log``:
    each persona has its own independent chip-usage history)."""
    assert table in _VALID_TABLES
    sim_filter = " AND sim_manager_id = :sim_manager_id" if sim_manager_id is not None else ""
    params = {"sim_manager_id": sim_manager_id} if sim_manager_id is not None else {}

    pending = db.execute(
        text(
            f"SELECT id, gameweek, details FROM {table} "
            f"WHERE decision_type = 'lineup' AND actual_outcome IS NULL{sim_filter}"
        ),
        params,
    ).fetchall()
    if not pending:
        return 0

    chip_by_gw: dict[int, str | None] = {}
    for gw, details_json in db.execute(
        text(f"SELECT gameweek, details FROM {table} WHERE decision_type = 'chip'{sim_filter}"),
        params,
    ).fetchall():
        chip_by_gw[gw] = json.loads(details_json).get("chip")

    updated = 0
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
                "Skipping %s decision %d (GW%d): incomplete lineup details",
                table, log_id, gameweek,
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
            text(f"UPDATE {table} SET actual_outcome = :pts WHERE id = :id"),
            {"pts": actual, "id": log_id},
        )
        updated += 1
        logger.info("%s GW%d decision %d: actual_outcome=%d", table, gameweek, log_id, actual)

    return updated


def backfill(season: str = "2026-27") -> int:
    """Returns the total number of rows updated across the real
    ``decision_log`` and every persona's ``sim_decision_log``."""
    db = get_session()
    try:
        total = _backfill_table(db, season, "decision_log")

        sim_manager_ids = [
            row[0] for row in db.execute(
                text("SELECT id FROM sim_managers WHERE season = :season"), {"season": season}
            ).fetchall()
        ]
        for sim_manager_id in sim_manager_ids:
            total += _backfill_table(db, season, "sim_decision_log", sim_manager_id=sim_manager_id)

        db.commit()
        return total
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    args = parser.parse_args()
    n = backfill(args.season)
    logger.info("Backfilled %d decision(s)", n)


if __name__ == "__main__":
    main()
