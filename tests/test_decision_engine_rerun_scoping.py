"""Re-running the SAME gameweek must not bank a free transfer (2026-08-28).

Live bug, found on the GW2 deadline day. ``_load_squad_state`` took the most
recent ``decision_log`` lineup row with no gameweek filter, but the
``free_transfers`` field on that row is NEXT gameweek's allowance -- it is
written by ``roll_forward_free_transfers``, which adds the weekly +1. So the
second run of a gameweek read the first run's roll-forward as *this* week's
allowance.

Observed: GW1 cold start stored free_transfers=1; the 2026-08-25 GW2 run read
1, used 0, stored 2; the 2026-08-28 GW2 re-run read that 2 as its own
allowance and made two transfers for zero hits. The real allowance was 1, so
the plan should have cost -4 -- and because the ILP prices hits in its
objective, it might not have chosen two transfers at all.

The squad/bank/purchase-price fields are idempotent under a re-run; only the
free-transfer ledger advances a week each time. The compounding case is worse:
a re-run after a run that DID transfer starts from the post-transfer squad and
stacks more transfers on top of it.
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
    engine = create_engine(f"sqlite:///{tmp_path / 'rerun_scoping.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(decision_engine, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _lineup(gameweek: int, free_transfers: int, squad_ids: list[int]) -> dict:
    return {
        "squad_ids": squad_ids,
        "budget": 100.0,
        "bank": 0.0,
        "free_transfers": free_transfers,
        "purchase_prices": {},
    }


def test_rerunning_same_gameweek_does_not_bank_a_free_transfer(session):
    """The GW2 re-run must read GW1's roll-forward (1), not GW2's own (2)."""
    session.add(DecisionLog(
        gameweek=1, decision_type="lineup",
        details=json.dumps(_lineup(1, free_transfers=1, squad_ids=[1, 2, 3])),
        projected_gain=60.0, dry_run=True,
    ))
    session.commit()
    session.add(DecisionLog(
        gameweek=2, decision_type="lineup",
        details=json.dumps(_lineup(2, free_transfers=2, squad_ids=[1, 2, 3])),
        projected_gain=70.0, dry_run=True,
    ))
    session.commit()

    state = decision_engine._load_squad_state(
        None, team_id=504618, config=OPTIMISER, decided_gw=2
    )

    assert state.free_transfers == 1, (
        "read its own previous GW2 run's roll-forward as this week's allowance"
    )


def test_scoping_prefers_the_latest_run_of_an_earlier_gameweek(session):
    """Two runs of GW1: the LATER one is still the right starting point."""
    for ft, squad in ((1, [1, 2, 3]), (1, [4, 5, 6])):
        session.add(DecisionLog(
            gameweek=1, decision_type="lineup",
            details=json.dumps(_lineup(1, free_transfers=ft, squad_ids=squad)),
            projected_gain=60.0, dry_run=True,
        ))
        session.commit()

    state = decision_engine._load_squad_state(
        None, team_id=504618, config=OPTIMISER, decided_gw=2
    )

    assert state.squad_ids == [4, 5, 6]


def test_no_prior_gameweek_falls_back_to_cold_start(session):
    """Only same-gameweek rows exist -> nothing to carry, cold start."""
    session.add(DecisionLog(
        gameweek=1, decision_type="lineup",
        details=json.dumps(_lineup(1, free_transfers=2, squad_ids=[1, 2, 3])),
        projected_gain=60.0, dry_run=True,
    ))
    session.commit()

    state = decision_engine._load_squad_state(
        None, team_id=504618, config=OPTIMISER, decided_gw=1
    )

    assert state.squad_ids == []


def test_sim_personas_are_scoped_too(session):
    """The persona path shares the defect -- it filters only on manager id."""
    session.add(SimManager(
        id=7, season="2026-27", label="sim-7",
        risk_level=0.0, max_ownership_differential=0.5, chip_aggressiveness=1.0,
    ))
    session.commit()
    session.add(SimDecisionLog(
        sim_manager_id=7, gameweek=1, decision_type="lineup",
        details=json.dumps(_lineup(1, free_transfers=1, squad_ids=[1, 2, 3])),
        projected_gain=60.0,
    ))
    session.commit()
    session.add(SimDecisionLog(
        sim_manager_id=7, gameweek=2, decision_type="lineup",
        details=json.dumps(_lineup(2, free_transfers=2, squad_ids=[1, 2, 3])),
        projected_gain=70.0,
    ))
    session.commit()

    state = decision_engine._load_squad_state(
        7, team_id=504618, config=OPTIMISER, decided_gw=2
    )

    assert state.free_transfers == 1
