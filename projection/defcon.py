"""defcon.py — P7 Defensive Contribution component (26/27 DefCon points).

DefCon awards 2 pts when a player's defensive actions clear a positional
threshold (DEF: CBIT ≥ 10; MID/FWD: CBIRT ≥ 12; GK: none — config.strategy
DEFCON). Given a player's expected per-match defensive-action count (CBIT or
CBIRT, from their rolling `player_match_events` rate), the DefCon points are
P(actions ≥ threshold) × 2, modelling per-match actions as Poisson.

Pure + deterministic. The rate itself is computed at assembly time (P10) from
`player_match_events`; this component maps rate + position → points.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from config.strategy import DEFCON

_CBIT_POSITIONS = {"DEF"}
_CBIRT_POSITIONS = {"MID", "FWD"}


def defcon_threshold(position: str) -> int | None:
    """Threshold for the position (None for GK, which earns no DefCon)."""
    if position in _CBIT_POSITIONS:
        return DEFCON.def_threshold
    if position in _CBIRT_POSITIONS:
        return DEFCON.mid_fwd_threshold
    return None


def p_hits_threshold(expected_actions: float, threshold: int | None) -> float:
    """P(Poisson(expected_actions) ≥ threshold)."""
    if threshold is None:
        return 0.0
    return float(poisson.sf(threshold - 1, max(0.0, expected_actions)))


def expected_defcon_points(expected_actions: float, position: str, p_play: float) -> float:
    """Expected DefCon points = P(hit threshold) × 2 × chance of playing."""
    thr = defcon_threshold(position)
    if thr is None:
        return 0.0
    return p_hits_threshold(expected_actions, thr) * DEFCON.points * max(0.0, p_play)


def sample_defcon_points(
    rng: np.random.Generator, expected_actions: float, position: str, played: bool
) -> int:
    """One MC draw: actions ~ Poisson → 2 pts if ≥ threshold (P10)."""
    thr = defcon_threshold(position)
    if thr is None or not played:
        return 0
    return DEFCON.points if rng.poisson(max(0.0, expected_actions)) >= thr else 0
