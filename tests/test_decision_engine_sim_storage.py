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
from config.strategy import CHIP_TIMING, OPTIMISER
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
        risk_level=0.0, max_ownership_differential=0.5,
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
    state = decision_engine._load_squad_state(
        None, team_id=12345, config=OPTIMISER
    )
    assert state.squad_ids == [1, 2, 3]
    assert state.budget == pytest.approx(99.5)
    assert state.free_transfers == 2


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

    state = decision_engine._load_squad_state(
        1, team_id=0, config=OPTIMISER
    )
    assert state.squad_ids == [10, 11]
    assert state.budget == pytest.approx(90.0)
    assert state.free_transfers == 1


def test_load_squad_state_defaults_when_nothing_logged(session):
    state = decision_engine._load_squad_state(
        None, team_id=1, config=OPTIMISER
    )
    assert state.squad_ids == []
    assert state.budget == 100.0
    assert state.free_transfers == 1

    _add_sim_manager(session, 7)
    state = decision_engine._load_squad_state(
        7, team_id=1, config=OPTIMISER
    )
    assert state.squad_ids == []
    assert state.budget == 100.0
    assert state.free_transfers == 1


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
        lambda **_: pd.DataFrame(columns=["player_id", "gameweek", "xpts"]),
    )
    # Mid-season signal, so the empty-projections branch takes the real
    # "abort" path rather than the cold-start path -- see
    # test_run_decision_cycle_reruns_cold_start_when_still_preseason for the
    # complementary case.
    monkeypatch.setattr(decision_engine, "season_has_played_history", lambda season: True)

    result = decision_engine.run_for_persona(persona, season="2026-27")

    assert result == {"error": "no_projections"}


def test_run_decision_cycle_reruns_cold_start_when_still_preseason(session, monkeypatch):
    """Real bug found 2026-08-09 (the user's own manual --dry-run rerun):
    gating the abort-vs-cold-start branch on `squad_ids` alone meant that
    once the FIRST cold-start run recorded a squad, every later rerun during
    the same still-pre-season window aborted with no_projections instead of
    rebuilding -- the user could never re-run to refine the initial squad as
    new signal data came in. A squad already existing must NOT force the
    abort branch while the season genuinely has no played-gameweek history
    yet."""
    import pandas as pd

    decision_engine._record_decision(
        None, gameweek=1, decision_type="lineup",
        details={"squad_ids": [1, 2, 3], "budget": 100.0, "free_transfers": 1},
        projected_gain=10.0,
    )

    fake_squad = pd.DataFrame({
        "id": [1, 2, 3], "web_name": ["A", "B", "C"], "position": ["GKP", "DEF", "FWD"],
        "now_cost": [5.0, 5.0, 5.0], "is_starting": [True, True, True],
        "is_captain": [True, False, False], "is_vice_captain": [False, True, False],
        "bench_order": [None, None, None],
    })
    fake_solution = type("Solution", (), {"squad": fake_squad, "total_cost": 15.0})()
    fake_xi = type("XI", (), {
        "squad": fake_squad, "starting_xi": fake_squad,
        "captain_id": 1, "vice_captain_id": 2, "total_xpts": 12.5,
    })()

    monkeypatch.setattr(decision_engine, "run_projections", lambda **kwargs: None)
    monkeypatch.setattr(decision_engine, "_get_current_and_next_gw", lambda: (1, 1))
    monkeypatch.setattr(
        decision_engine, "get_latest_projections",
        lambda **_: pd.DataFrame(columns=["player_id", "gameweek", "xpts"]),
    )
    monkeypatch.setattr(decision_engine, "season_has_played_history", lambda season: False)
    monkeypatch.setattr(decision_engine, "_load_players", lambda: pd.DataFrame())
    monkeypatch.setattr(
        decision_engine.cold_start, "build_initial_squad",
        lambda season, players, config: (fake_solution, pd.DataFrame()),
    )
    monkeypatch.setattr(decision_engine, "optimise_starting_xi", lambda *a, **kw: fake_xi)

    result = decision_engine._run_decision_cycle(
        season="2026-27", dry_run=True, force_chip=None, config=OPTIMISER,
        chip_timing=decision_engine.CHIP_TIMING, team_id=1, sim_manager_id=None,
    )

    assert result["cold_start"] is True
    assert "error" not in result


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


def test_cold_start_branches_on_the_season_not_on_persisted_projections(monkeypatch, session):
    """Found 2026-08-16 by auditing the live decision log, which contained a
    Triple Captain recorded as PLAYED in GW1 before a ball was kicked.

    The branch tested `projections.empty`, a fact about what happens to be
    persisted rather than about the season. The moment the cold start began
    persisting its own projections — so the site and dashboard had numbers to
    show — the frame stopped being empty pre-season and the engine took the
    IN-SEASON path, ran recommend_chip, and burned a chip.
    """
    import pandas as pd

    from agent import decision_engine as de

    # Pre-season, but WITH projections persisted — the exact state that broke.
    monkeypatch.setattr(de, "season_has_played_history", lambda season: False)
    monkeypatch.setattr(de, "_get_current_and_next_gw", lambda: (1, 1))
    monkeypatch.setattr(de, "run_projections", lambda **k: None)
    monkeypatch.setattr(
        de, "get_latest_projections",
        lambda **_: pd.DataFrame([
            {"player_id": 1, "gameweek": 1, "xpts": 5.0, "start_probability": 0.9},
        ]),
    )
    monkeypatch.setattr(de, "apply_departure_discount", lambda proj, ov: proj)
    monkeypatch.setattr(de, "load_latest_ownership", lambda: pd.DataFrame())
    monkeypatch.setattr(de, "load_p_leave_overrides", lambda: {})
    monkeypatch.setattr(de, "_load_players", lambda: pd.DataFrame())
    monkeypatch.setattr(de, "persist_projections", lambda df: None)
    monkeypatch.setattr(de, "_record_decision", lambda *a, **k: None)
    monkeypatch.setattr(de, "log_rumoured_squad_members", lambda *a, **k: None)

    chip_calls = []
    monkeypatch.setattr(
        de, "recommend_chip",
        lambda **k: chip_calls.append(k) or de.ChipRecommendation(None, "", 0.0),
    )

    called = {}

    def _fake_cold_start(season, players=None, config=None):
        called["cold_start"] = True
        raise RuntimeError("stop here — reaching the cold start is the assertion")

    monkeypatch.setattr(de.cold_start, "build_initial_squad", _fake_cold_start)

    with pytest.raises(RuntimeError):
        de._run_decision_cycle(
            season="2026-27", dry_run=True, force_chip=None,
            config=OPTIMISER, chip_timing=CHIP_TIMING, team_id=None,
            sim_manager_id=None, refresh_projections=False,
        )

    assert called.get("cold_start"), "pre-season must cold start regardless of persisted rows"
    assert not chip_calls, "a chip must never be evaluated before the season starts"
