"""clean_sheets.py — P5 clean-sheet + goals-conceded points.

Odds-anchored: the opponent's expected goals λ (from team_goals.py) gives the
clean-sheet probability exp(−λ) — replacing v1's capped 1X2 heuristic. FPL only
awards the CS bonus if the player is on for 60+ minutes, so both terms are
conditioned on P(60+) from the P1 minutes model. GK/DEF also lose 1 pt per 2
goals conceded (in-play), the expected value of which is computed from the same
Poisson λ. Pure + deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from config.strategy import SCORING

CS_POINTS = {
    "GK": SCORING.points_cs_gk, "GKP": SCORING.points_cs_gk,
    "DEF": SCORING.points_cs_def,
    "MID": SCORING.points_cs_mid,
    "FWD": SCORING.points_cs_fwd,
}
_CS_ELIGIBLE = {"GK", "GKP", "DEF", "MID", "FWD"}
_CONCEDE_POSITIONS = {"GK", "GKP", "DEF"}
_MAX_GOALS = 12


def clean_sheet_prob(lam_opponent: float) -> float:
    """P(opponent scores 0) = Poisson(0; λ) = exp(−λ)."""
    return float(np.exp(-max(0.0, lam_opponent)))


def expected_cs_points(lam_opponent: float, p60: float, position: str) -> float:
    """Expected clean-sheet points = P(CS) × P(60+) × position CS value. A CS
    only scores if the player lasts 60 minutes (hence ``p60`` from P1)."""
    if position not in _CS_ELIGIBLE:
        return 0.0
    return clean_sheet_prob(lam_opponent) * max(0.0, p60) * CS_POINTS.get(position, 0)


def expected_concede_points(lam_opponent: float, p_play: float, position: str) -> float:
    """Expected goals-conceded penalty for GK/DEF: −1 per 2 conceded, i.e.
    ``E[floor(X/2)] × −1`` for X~Poisson(λ), scaled by the chance of playing."""
    if position not in _CONCEDE_POSITIONS:
        return 0.0
    k = np.arange(_MAX_GOALS + 1)
    units = np.floor(k / SCORING.goals_conceded_per_penalty)
    expected_units = float(np.sum(units * poisson.pmf(k, max(0.0, lam_opponent))))
    return expected_units * SCORING.points_goals_conceded_penalty * max(0.0, p_play)


def sample_clean_sheet_points(
    rng: np.random.Generator, lam_opponent: float, played_60: bool, position: str
) -> int:
    """One MC draw of CS + concede points for a player (P10). Draws the
    opponent's goals once; CS bonus needs a clean sheet AND 60+ minutes."""
    if position not in _CS_ELIGIBLE:
        return 0
    conceded = int(rng.poisson(max(0.0, lam_opponent)))
    pts = 0
    if played_60 and conceded == 0:
        pts += CS_POINTS.get(position, 0)
    if position in _CONCEDE_POSITIONS:
        units = conceded // SCORING.goals_conceded_per_penalty
        pts += units * SCORING.points_goals_conceded_penalty
    return pts
