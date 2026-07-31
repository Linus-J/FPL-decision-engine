"""dashboard/data/simulations.py"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dashboard.data.simulations import get_leaderboard, get_real_squad_cumulative_actual
from data.models import Base, DecisionLog, SimDecisionLog, SimManager


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'sim_leaderboard.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def _manager(session, manager_id: int, label: str, season: str = "2026-27") -> None:
    session.add(SimManager(
        id=manager_id, season=season, label=label, risk_mode="balanced",
        variance_weight=0.0, max_ownership_differential=0.5, chip_aggressiveness=1.0,
    ))
    session.commit()


def test_leaderboard_empty_when_no_personas(session):
    df = get_leaderboard(session, season="2026-27")
    assert df.empty


def test_leaderboard_ranks_by_cumulative_actual_descending(session):
    _manager(session, 1, "sim-001")
    _manager(session, 2, "sim-002")
    session.add_all([
        SimDecisionLog(sim_manager_id=1, gameweek=1, decision_type="lineup",
                        details="{}", actual_outcome=40),
        SimDecisionLog(sim_manager_id=1, gameweek=2, decision_type="lineup",
                        details="{}", actual_outcome=50),
        SimDecisionLog(sim_manager_id=2, gameweek=1, decision_type="lineup",
                        details="{}", actual_outcome=100),
    ])
    session.commit()

    df = get_leaderboard(session, season="2026-27")

    assert list(df.sort_values("rank")["label"]) == ["sim-002", "sim-001"]
    row2 = df[df["label"] == "sim-002"].iloc[0]
    row1 = df[df["label"] == "sim-001"].iloc[0]
    assert row2["cumulative_actual"] == 100
    assert row1["cumulative_actual"] == 90
    assert row1["gws_scored"] == 2
    assert row2["gws_scored"] == 1


def test_leaderboard_persona_with_no_scored_gws_shows_zero(session):
    _manager(session, 1, "sim-001")
    df = get_leaderboard(session, season="2026-27")
    assert df.iloc[0]["cumulative_actual"] == 0
    assert df.iloc[0]["gws_scored"] == 0


def test_leaderboard_scoped_per_season(session):
    _manager(session, 1, "sim-2627", season="2026-27")
    _manager(session, 2, "sim-2728", season="2027-28")
    df = get_leaderboard(session, season="2026-27")
    assert list(df["label"]) == ["sim-2627"]


def test_real_squad_cumulative_actual(session):
    assert get_real_squad_cumulative_actual(session) == 0.0
    session.add_all([
        DecisionLog(gameweek=1, decision_type="lineup", details="{}", actual_outcome=45),
        DecisionLog(gameweek=2, decision_type="lineup", details="{}", actual_outcome=55),
        DecisionLog(gameweek=2, decision_type="transfers", details="{}", actual_outcome=None),
    ])
    session.commit()
    assert get_real_squad_cumulative_actual(session) == 100.0
