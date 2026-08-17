#!/usr/bin/env python
"""Backfill ``actual_outcome`` for past 'lineup' decisions once their
gameweek has finished -- both the real bot's ``decision_log`` and every
simulated persona's ``sim_decision_log`` (plan/simulation-engine-v1.md).
Purely additive -- never touches ``dry_run`` or any decision-making logic.

``actual_outcome`` is the NET points a manager holding that squad would
really have scored: auto-substitutions applied, vice-captain promoted if the
captain blanked, and any hits deducted. The first two used to be impossible
here (``bench_order`` and positions were not persisted on lineup decisions)
and the third was simply missed, so every recorded outcome understated the
decision -- and understated it hardest exactly where the bench mattered.
Fixed 2026-08-16, P2.1/P2.2; decisions recorded before that lack the fields
and are still scored the old way, with a warning, rather than being silently
rescored onto a different basis.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from config.strategy import TRANSFERS
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


def _minutes_by_player(db, season: str, gameweek: int) -> dict[int, int]:
    """Minutes played per player in a finished gameweek (P2.1) -- summed
    across a double gameweek's rows, same convention as the points above.

    This is what makes auto-substitution modelling possible: ``_score_squad``
    needs ``minutes`` to know which starters blanked, and it silently keeps
    the old no-autosub behaviour when any of its three optional arguments is
    missing."""
    rows = db.execute(
        text(
            "SELECT player_id, SUM(minutes) FROM player_gw_stats "
            "WHERE season = :season AND gameweek = :gw GROUP BY player_id"
        ),
        {"season": season, "gw": gameweek},
    ).fetchall()
    return {r[0]: int(r[1] or 0) for r in rows}


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
    # Enforced rather than merely documented: the latest-per-gameweek subquery
    # below takes MAX(created_at) within the given scope, so an unscoped call
    # on sim_decision_log would resolve to the single globally-latest row and
    # silently score ONE persona out of ninety.
    if table == "sim_decision_log" and sim_manager_id is None:
        raise ValueError(
            "sim_decision_log must be scored one persona at a time; an "
            "unscoped call would score only the globally-latest row"
        )
    sim_filter = " AND sim_manager_id = :sim_manager_id" if sim_manager_id is not None else ""
    params = {"sim_manager_id": sim_manager_id} if sim_manager_id is not None else {}

    # Only the LATEST lineup per gameweek is scored -- the decision that
    # actually stood at the deadline.
    #
    # Re-running a gameweek is documented as safe, and appends a new lineup row
    # each time rather than replacing the old one (every read elsewhere takes
    # ORDER BY created_at DESC LIMIT 1, so that is correct storage). But this
    # scorer selected EVERY unscored lineup row, so seven re-runs of GW1 on
    # 2026-08-17 left 8 rows in decision_log and 689 in sim_decision_log for 90
    # real (persona, gameweek) pairs -- a 7.7x inflation. simulation.analysis
    # groups on gameweek, so those would have counted as independent
    # observations and skewed both the persona ranking and the calibration
    # sample, from superseded decisions that were never live.
    #
    # Superseded rows are simply left unscored: actual_outcome IS NULL already
    # means "not scored", and every consumer filters on it.
    scope = " AND t2.sim_manager_id = :sim_manager_id" if sim_manager_id is not None else ""
    pending = db.execute(
        text(
            f"SELECT id, gameweek, details FROM {table} AS t1 "
            f"WHERE decision_type = 'lineup' AND actual_outcome IS NULL{sim_filter} "
            f"AND created_at = ("
            f"    SELECT MAX(t2.created_at) FROM {table} AS t2 "
            f"    WHERE t2.decision_type = 'lineup' AND t2.gameweek = t1.gameweek{scope}"
            f")"
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
        # P2.1: positions/bench_order are recorded on the lineup decision as
        # of 2026-08-16 (agent/decision_engine.py::_lineup_shape). Decisions
        # written before that carry neither, and `_score_squad` requires all
        # THREE of minutes/positions/bench_order together -- so those older
        # rows keep their existing no-autosub scoring rather than being
        # silently rescored on a different basis.
        positions = {int(k): v for k, v in (details.get("positions") or {}).items()}
        bench_order = {int(k): int(v) for k, v in (details.get("bench_order") or {}).items()}
        autosub_kwargs: dict = {}
        if positions and bench_order:
            autosub_kwargs = {
                "minutes": _minutes_by_player(db, season, gameweek),
                "positions": positions,
                "bench_order": bench_order,
            }
        else:
            logger.warning(
                "%s decision %d (GW%d): no positions/bench_order recorded — "
                "scoring WITHOUT auto-substitutions, which understates a squad "
                "whose starters blanked",
                table, log_id, gameweek,
            )

        actual = _score_squad(
            squad_ids=squad_ids,
            starting_ids=starting_ids,
            captain_id=captain_id,
            actual_points=actual_points,
            bench_boost=(chip == Chip.BENCH_BOOST.value),
            triple_captain=(chip == Chip.TRIPLE_CAPTAIN.value),
            vice_captain_id=vice_captain_id,
            **autosub_kwargs,
        )
        # P2.2: hits are a real cost of the decision and were never netted
        # off -- they are booked on the separate `transfers` row, so the
        # lineup row recorded gross points. `actual_outcome` is what the
        # season analysis compares personas on, so it has to be net.
        hits = int(details.get("hits_taken") or 0)
        actual -= hits * abs(TRANSFERS.hit_cost_points)
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
