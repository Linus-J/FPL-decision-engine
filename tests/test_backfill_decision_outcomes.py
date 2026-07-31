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
from data.models import Base, DecisionLog, Gameweek, PlayerGameweekStats


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
