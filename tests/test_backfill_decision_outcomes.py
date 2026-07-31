"""scripts/backfill_decision_outcomes.py

Covers: a normal finished GW gets actual_outcome written (captain doubled,
bench excluded); a genuine double-gameweek player's two player_gw_stats rows
are summed; an unfinished GW and a row that already has actual_outcome are
both left untouched; a bench_boost GW includes bench points."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import scripts.backfill_decision_outcomes as backfill_module
from data.models import Base, DecisionLog, Gameweek, PlayerGameweekStats, SimDecisionLog, SimManager


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'backfill.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(backfill_module, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _gw(session, gw_id: int, finished: bool) -> None:
    session.add(Gameweek(id=gw_id, season="2026-27", name=f"GW{gw_id}",
                          deadline_time=datetime(2026, 9, 1), finished=finished))
    session.commit()


def _stats(session, player_id: int, gw: int, points: int, opponent: int = 1) -> None:
    session.add(PlayerGameweekStats(
        player_id=player_id, gameweek=gw, season="2026-27",
        total_points=points, opponent_team_id=opponent,
    ))
    session.commit()


def _lineup(session, gw: int, squad_ids, starting_ids, captain_id, vice_captain_id=None) -> int:
    entry = DecisionLog(
        gameweek=gw, decision_type="lineup",
        details=json.dumps({
            "squad_ids": squad_ids, "starting_ids": starting_ids,
            "captain_id": captain_id, "vice_captain_id": vice_captain_id,
        }),
        projected_gain=0.0, dry_run=True,
    )
    session.add(entry)
    session.commit()
    return entry.id


def test_unfinished_gw_is_skipped(session):
    _gw(session, 10, finished=False)
    _lineup(session, 10, [1, 2], [1], captain_id=1)
    n = backfill_module.backfill("2026-27")
    assert n == 0
    row = session.query(DecisionLog).one()
    assert row.actual_outcome is None


def test_finished_gw_computes_captain_doubled_starting_only(session):
    _gw(session, 11, finished=True)
    _stats(session, 1, 11, points=10)  # captain -> doubled
    _stats(session, 2, 11, points=4)   # starting
    _stats(session, 3, 11, points=99)  # bench, should not count
    _lineup(session, 11, squad_ids=[1, 2, 3], starting_ids=[1, 2], captain_id=1)

    n = backfill_module.backfill("2026-27")

    assert n == 1
    row = session.query(DecisionLog).one()
    assert row.actual_outcome == 24  # 10*2 + 4


def test_dgw_player_stats_rows_are_summed(session):
    _gw(session, 12, finished=True)
    _stats(session, 1, 12, points=5, opponent=1)
    _stats(session, 1, 12, points=8, opponent=2)  # same player, 2nd fixture (DGW)
    _lineup(session, 12, squad_ids=[1], starting_ids=[1], captain_id=1)

    backfill_module.backfill("2026-27")

    row = session.query(DecisionLog).one()
    assert row.actual_outcome == 26  # (5 + 8) * 2 (captain)


def test_already_backfilled_row_is_left_untouched(session):
    _gw(session, 13, finished=True)
    _stats(session, 1, 13, points=10)
    entry_id = _lineup(session, 13, squad_ids=[1], starting_ids=[1], captain_id=1)
    session.query(DecisionLog).filter_by(id=entry_id).update({"actual_outcome": 999})
    session.commit()

    n = backfill_module.backfill("2026-27")

    assert n == 0
    row = session.query(DecisionLog).one()
    assert row.actual_outcome == 999


def test_bench_boost_includes_bench_points(session):
    _gw(session, 14, finished=True)
    _stats(session, 1, 14, points=6)
    _stats(session, 2, 14, points=3)  # bench, counted because of bench boost
    session.add(DecisionLog(
        gameweek=14, decision_type="chip",
        details=json.dumps({"chip": "bboost", "reason": "test"}),
        projected_gain=0.0, dry_run=True,
    ))
    session.commit()
    _lineup(session, 14, squad_ids=[1, 2], starting_ids=[1], captain_id=1)

    backfill_module.backfill("2026-27")

    row = session.query(DecisionLog).filter_by(decision_type="lineup").one()
    assert row.actual_outcome == 15  # 6*2 (captain) + 3 (bench, boosted)


def _add_sim_manager(session, manager_id: int) -> None:
    session.add(SimManager(
        id=manager_id, season="2026-27", label=f"sim-{manager_id}",
        risk_mode="balanced", variance_weight=0.0, max_ownership_differential=0.5,
        chip_aggressiveness=1.0,
    ))
    session.commit()


def _sim_lineup(session, sim_manager_id: int, gw: int, squad_ids, starting_ids, captain_id) -> int:
    entry = SimDecisionLog(
        sim_manager_id=sim_manager_id, gameweek=gw, decision_type="lineup",
        details=json.dumps({
            "squad_ids": squad_ids, "starting_ids": starting_ids, "captain_id": captain_id,
        }),
        projected_gain=0.0,
    )
    session.add(entry)
    session.commit()
    return entry.id


def test_sim_decision_log_gets_backfilled(session):
    _gw(session, 15, finished=True)
    _add_sim_manager(session, 1)
    _stats(session, 1, 15, points=7)
    _stats(session, 2, 15, points=2)
    _sim_lineup(session, 1, 15, squad_ids=[1, 2], starting_ids=[1, 2], captain_id=1)

    n = backfill_module.backfill("2026-27")

    assert n == 1
    row = session.query(SimDecisionLog).one()
    assert row.actual_outcome == 16  # 7*2 (captain) + 2


def test_sim_decision_log_chip_history_is_isolated_per_manager(session):
    """Manager 1 played bench boost this GW; manager 2 did not -- manager
    2's bench points must NOT be counted even though both share a
    gameweek."""
    _gw(session, 16, finished=True)
    _add_sim_manager(session, 1)
    _add_sim_manager(session, 2)
    _stats(session, 1, 16, points=5)
    _stats(session, 2, 16, points=9)  # bench for both managers

    session.add(SimDecisionLog(
        sim_manager_id=1, gameweek=16, decision_type="chip",
        details=json.dumps({"chip": "bboost", "reason": "test"}), projected_gain=0.0,
    ))
    session.commit()

    _sim_lineup(session, 1, 16, squad_ids=[1, 2], starting_ids=[1], captain_id=1)
    _sim_lineup(session, 2, 16, squad_ids=[1, 2], starting_ids=[1], captain_id=1)

    backfill_module.backfill("2026-27")

    manager1_row = session.query(SimDecisionLog).filter_by(
        sim_manager_id=1, decision_type="lineup"
    ).one()
    manager2_row = session.query(SimDecisionLog).filter_by(
        sim_manager_id=2, decision_type="lineup"
    ).one()
    assert manager1_row.actual_outcome == 19  # 5*2 (captain) + 9 (bench, boosted)
    assert manager2_row.actual_outcome == 10  # 5*2 (captain) only -- no boost


def test_backfill_covers_both_real_and_sim_logs_in_one_call(session):
    _gw(session, 17, finished=True)
    _add_sim_manager(session, 1)
    _stats(session, 1, 17, points=4)
    _lineup(session, 17, squad_ids=[1], starting_ids=[1], captain_id=1)
    _sim_lineup(session, 1, 17, squad_ids=[1], starting_ids=[1], captain_id=1)

    n = backfill_module.backfill("2026-27")

    assert n == 2
    assert session.query(DecisionLog).filter_by(decision_type="lineup").one().actual_outcome == 8
    assert session.query(SimDecisionLog).one().actual_outcome == 8
