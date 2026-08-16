"""simulation/engine.py::run_all_personas"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import agent.decision_engine as decision_engine
import simulation.engine as engine_module
from data.models import Base
from simulation.personas import SIM_COUNT


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'engine.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(engine_module, "get_session", lambda: Local())
    return Local


def test_run_all_personas_records_a_result_per_persona(session_factory, monkeypatch):
    monkeypatch.setattr(
        decision_engine, "run_for_persona",
        lambda persona, season: {"gameweek": 1, "persona_id": persona.id},
    )
    results = engine_module.run_all_personas("2026-27")
    assert len(results) == SIM_COUNT
    for pid, result in results.items():
        assert result == {"gameweek": 1, "persona_id": pid}


def test_run_all_personas_isolates_a_failing_persona(session_factory, monkeypatch):
    def _flaky(persona, season):
        if persona.id == 3:
            raise RuntimeError("boom")
        return {"ok": True}

    monkeypatch.setattr(decision_engine, "run_for_persona", _flaky)
    results = engine_module.run_all_personas("2026-27")

    assert results[3] == {"error": "exception"}
    others = [r for pid, r in results.items() if pid != 3]
    assert all(r == {"ok": True} for r in others)
    assert len(others) == SIM_COUNT - 1


def test_run_all_personas_reuses_persisted_personas_across_calls(session_factory, monkeypatch):
    seen_persona_ids: list[int] = []
    monkeypatch.setattr(
        decision_engine, "run_for_persona",
        lambda persona, season: seen_persona_ids.append(persona.id) or {"ok": True},
    )
    first_results = engine_module.run_all_personas("2026-27")
    first_ids_seen = sorted(seen_persona_ids)

    seen_persona_ids.clear()
    second_results = engine_module.run_all_personas("2026-27")
    second_ids_seen = sorted(seen_persona_ids)

    assert set(first_results) == set(second_results)
    assert first_ids_seen == second_ids_seen
