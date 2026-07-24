"""saves.py — P6 goalkeeper saves component.

A keeper's save volume is driven mostly by how much the opponent shoots, so we
anchor on the opponent's expected goals λ (from team_goals.py): shots-on-target
faced ≈ λ / conversion, and saves ≈ SoT × (1 − conversion) (the on-target shots
that don't score). FPL awards 1 pt per 3 saves. GK-only.

Deliberately fixture-anchored (a reasonable v1); a per-keeper shot-stopping
skill term could refine it later. Pure + deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from config.strategy import SCORING

# League goals-per-shot-on-target (~0.3): SoT_faced = λ_opp / conversion.
LEAGUE_CONVERSION = 0.30
_MAX_SAVES = 40


def expected_saves(lam_opponent: float, p_play: float) -> float:
    """Expected GK saves = on-target shots faced × (share that don't score),
    scaled by chance of playing. Anchored on the opponent's expected goals."""
    sot_faced = max(0.0, lam_opponent) / LEAGUE_CONVERSION
    return sot_faced * (1.0 - LEAGUE_CONVERSION) * max(0.0, p_play)


def expected_save_points(exp_saves: float) -> float:
    """E[⌊saves / 3⌋] × 1pt over Poisson(exp_saves) — the FPL 1-per-3 rule as an
    expectation (not exp_saves/3, which over-counts the sub-3 mass)."""
    k = np.arange(_MAX_SAVES + 1)
    units = np.floor(k / SCORING.saves_per_bonus_point)
    expected_units = float(np.sum(units * poisson.pmf(k, max(0.0, exp_saves))))
    return expected_units * SCORING.points_save_bonus


def sample_save_points(rng: np.random.Generator, exp_saves: float) -> int:
    """One MC draw: saves ~ Poisson(exp_saves) → ⌊saves/3⌋ points (P10)."""
    saves = int(rng.poisson(max(0.0, exp_saves)))
    return (saves // SCORING.saves_per_bonus_point) * SCORING.points_save_bonus
