"""Weekly simulation batch (plan/simulation-engine-v1.md): steps every
persisted persona forward one gameweek via the same decision logic the
real bot uses (``agent.decision_engine.run_for_persona``). Each persona is
isolated by try/except so one failure can never abort the batch or affect
another persona's result.
"""

from __future__ import annotations

import logging

from data.db import get_session
from simulation.personas import load_or_create_personas

logger = logging.getLogger(__name__)


def run_all_personas(season: str) -> dict[int, dict]:
    """Returns ``{sim_manager_id: result_dict}`` for every persona --
    ``{"error": "exception"}`` for any persona whose step raised."""
    import agent.decision_engine as decision_engine

    db = get_session()
    try:
        personas = load_or_create_personas(db, season)
        results: dict[int, dict] = {}
        for persona in personas:
            try:
                results[persona.id] = decision_engine.run_for_persona(persona, season)
            except Exception:
                logger.exception(
                    "Simulation failed for persona id=%s label=%s", persona.id, persona.label
                )
                results[persona.id] = {"error": "exception"}
        return results
    finally:
        db.close()
