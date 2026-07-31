"""Persona generation for the live simulation engine
(plan/simulation-engine-v1.md). Personas are generated once per season
with a fixed seed and persisted to ``sim_managers`` -- a given persona's
parameters never change mid-season. Exact ranges below are tuning
constants, not load-bearing design; adjust after observing a season's
worth of behaviour.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from data.models import SimManager

SIM_COUNT = 100
# Fixed, arbitrary constant -- reproducibility (every season regenerates
# the exact same 100 personas if sim_managers is ever rebuilt from empty),
# not something to vary per run.
SIM_SEED = 20260731

_RISK_LEVEL_RANGE = (-1.0, 1.0)
_MAX_OWNERSHIP_DIFFERENTIAL_RANGE = (0.0, 1.0)
_CHIP_AGGRESSIVENESS_RANGE = (0.5, 1.5)


def generate_personas(season: str, n: int = SIM_COUNT, seed: int = SIM_SEED) -> list[dict]:
    """Deterministic: the same ``(season, n, seed)`` always produces
    byte-identical persona parameters. Returns plain dicts (not ORM rows)
    so this stays testable without a DB.

    ``risk_level`` (continuous, plan/risk-aware-cold-start-v1.md,
    2026-07-31) now drives BOTH the differential-chasing (lambda) and
    variance-awareness (mu) axes together via a single dial per persona --
    superseding the old separate categorical ``risk_mode`` + independent
    ``variance_weight`` pair."""
    rng = np.random.default_rng(seed)
    personas = []
    for i in range(n):
        risk_level = round(float(rng.uniform(*_RISK_LEVEL_RANGE)), 4)
        max_ownership_differential = round(
            float(rng.uniform(*_MAX_OWNERSHIP_DIFFERENTIAL_RANGE)), 4
        )
        chip_aggressiveness = round(float(rng.uniform(*_CHIP_AGGRESSIVENESS_RANGE)), 4)
        personas.append({
            "season": season,
            "label": f"sim-{i:03d}-{risk_level:+.2f}",
            "risk_level": risk_level,
            "max_ownership_differential": max_ownership_differential,
            "chip_aggressiveness": chip_aggressiveness,
        })
    return personas


def load_or_create_personas(
    db: Session, season: str, n: int = SIM_COUNT, seed: int = SIM_SEED
) -> list[SimManager]:
    """This season's persisted persona rows, generating + persisting them
    only on the season's first-ever call. Never regenerates once any rows
    exist for the season -- a persona's identity/config stays stable for
    the whole season."""
    existing = (
        db.query(SimManager)
        .filter(SimManager.season == season)
        .order_by(SimManager.id)
        .all()
    )
    if existing:
        return existing

    rows = [SimManager(**p) for p in generate_personas(season, n, seed)]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows
