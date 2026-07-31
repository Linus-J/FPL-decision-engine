"""agent/decision_engine.py's sim-aware storage helpers
(_load_squad_state, _load_own_decision_log, _record_decision).

Core property under test: a `sim_manager_id` completely isolates a
persona's squad/decision history in `sim_decision_log` from the real bot's
`decision_log` and from every OTHER persona's rows -- the simulation
engine's safety story depends on this being airtight.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import agent.decision_engine as decision_engine
from config.strategy import OPTIMISER
from data.models import Base, DecisionLog, SimDecisionLog, SimManager


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'sim_storage.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(decision_engine, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _add_sim_manager(session, manager_id: int) -> None:
    session.add(SimManager(
        id=manager_id, season="2026-27", label=f"sim-{manager_id}",
        risk_mode="balanced", variance_weight=0.0, max_ownership_differential=0.5,
        chip_aggressiveness=1.0,
    ))
    session.commit()


def test_load_squad_state_real_reads_decision_log(session):
    session.add(DecisionLog(
        gameweek=5, decision_type="lineup",
        details=json.dumps({"squad_ids": [1, 2, 3], "budget": 99.5, "free_transfers": 2}),
        projected_gain=10.0, dry_run=True,
    ))
    session.commit()
    squad_ids, budget, free_transfers = decision_engine._load_squad_state(
        None, team_id=12345, config=OPTIMISER
    )
    assert squad_ids == [1, 2, 3]
    assert budget == pytest.approx(99.5)
    assert free_transfers == 2


def test_load_squad_state_sim_reads_only_its_own_manager(session):
    _add_sim_manager(session, 1)
    _add_sim_manager(session, 2)
    session.add(SimDecisionLog(
        sim_manager_id=1, gameweek=5, decision_type="lineup",
        details=json.dumps({"squad_ids": [10, 11], "budget": 90.0, "free_transfers": 1}),
        projected_gain=5.0,
    ))
    session.add(SimDecisionLog(
        sim_manager_id=2, gameweek=5, decision_type="lineup",
        details=json.dumps({"squad_ids": [20, 21], "budget": 80.0, "free_transfers": 3}),
        projected_gain=8.0,
    ))
    session.add(DecisionLog(
        gameweek=5, decision_type="lineup",
        details=json.dumps({"squad_ids": [99], "budget": 50.0, "free_transfers": 5}),
        projected_gain=1.0, dry_run=True,
    ))
    session.commit()

    squad_ids, budget, free_transfers = decision_engine._load_squad_state(
        1, team_id=0, config=OPTIMISER
    )
    assert squad_ids == [10, 11]
    assert budget == pytest.approx(90.0)
    assert free_transfers == 1


def test_load_squad_state_defaults_when_nothing_logged(session):
    squad_ids, budget, free_transfers = decision_engine._load_squad_state(
        None, team_id=1, config=OPTIMISER
    )
    assert squad_ids == []
    assert budget == 100.0
    assert free_transfers == 1

    _add_sim_manager(session, 7)
    squad_ids, budget, free_transfers = decision_engine._load_squad_state(
        7, team_id=1, config=OPTIMISER
    )
    assert squad_ids == []
    assert budget == 100.0
    assert free_transfers == 1


def test_record_decision_real_writes_decision_log_not_sim(session):
    decision_engine._record_decision(
        None, gameweek=3, decision_type="chip",
        details={"chip": "wildcard", "reason": "test"}, projected_gain=12.0, dry_run=True,
    )
    assert session.query(DecisionLog).count() == 1
    assert session.query(SimDecisionLog).count() == 0
    row = session.query(DecisionLog).one()
    assert row.dry_run is True
    assert json.loads(row.details)["chip"] == "wildcard"


def test_record_decision_sim_writes_sim_decision_log_not_real(session):
    _add_sim_manager(session, 4)
    decision_engine._record_decision(
        4, gameweek=3, decision_type="chip",
        details={"chip": "bboost", "reason": "test"}, projected_gain=7.0, dry_run=True,
    )
    assert session.query(SimDecisionLog).count() == 1
    assert session.query(DecisionLog).count() == 0
    row = session.query(SimDecisionLog).one()
    assert row.sim_manager_id == 4
    assert json.loads(row.details)["chip"] == "bboost"


def test_run_for_persona_never_refreshes_projections(session, monkeypatch):
    """Real gap found while building the simulation engine: run_for_persona
    must NOT call run_projections(persist=True) -- scripts/run_simulations.py
    runs right after scripts/run_agent.py in the same scheduled job, which
    has already refreshed this gameweek's projections. Calling it again per
    persona would silently redo the same computation up to 100x and write
    near-duplicate rows to player_projections every single run.

    Uses the early "squad already exists but projections are empty ->
    abort" branch (a pre-existing SimDecisionLog lineup row) so the test
    stays a pure unit check on the refresh-projections wiring, without
    needing a full players/PuLP fixture for a real cold-start build."""
    import pandas as pd

    _add_sim_manager(session, 1)
    persona = session.query(SimManager).filter_by(id=1).one()
    decision_engine._record_decision(
        1, gameweek=1, decision_type="lineup",
        details={"squad_ids": [1, 2, 3], "budget": 100.0, "free_transfers": 1},
        projected_gain=10.0,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("run_projections must not be called by run_for_persona")

    monkeypatch.setattr(decision_engine, "run_projections", _boom)
    monkeypatch.setattr(decision_engine, "_get_current_and_next_gw", lambda: (1, 1))
    monkeypatch.setattr(
        decision_engine, "get_latest_projections",
        lambda: pd.DataFrame(columns=["player_id", "gameweek", "xpts"]),
    )

    result = decision_engine.run_for_persona(persona, season="2026-27")

    assert result == {"error": "no_projections"}


def test_load_own_decision_log_isolates_personas(session):
    _add_sim_manager(session, 1)
    _add_sim_manager(session, 2)
    decision_engine._record_decision(1, gameweek=1, decision_type="lineup", details={"a": 1})
    decision_engine._record_decision(2, gameweek=1, decision_type="lineup", details={"a": 2})
    decision_engine._record_decision(None, gameweek=1, decision_type="lineup", details={"a": 3})

    df1 = decision_engine._load_own_decision_log(1)
    df2 = decision_engine._load_own_decision_log(2)
    df_real = decision_engine._load_own_decision_log(None)

    assert len(df1) == 1 and json.loads(df1.iloc[0]["details"])["a"] == 1
    assert len(df2) == 1 and json.loads(df2.iloc[0]["details"])["a"] == 2
    assert len(df_real) == 1 and json.loads(df_real.iloc[0]["details"])["a"] == 3
