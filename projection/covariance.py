"""covariance.py — P-COV joint per-fixture Monte-Carlo sampling (finding C2).

Summing independent per-player marginal samples gives ≈0 team covariance — the
structure P10 exists to produce. Two defenders on the same team either both
keep a clean sheet or both don't; that shared fate is invisible if each player
draws their own independent Poisson for the opponent's goals. This module
draws the team-level latents (goals scored, goals conceded) ONCE per fixture
per scenario, and every player on that team conditions their sample on the
shared draw instead of redrawing it.

Pure + deterministic given an RNG. No DB or network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def sample_team_goals(rng: np.random.Generator, team_lambda: float) -> int:
    """The ONE shared draw for a team's goals this scenario (goals scored, if
    ``team_lambda`` is the team's own λ; goals conceded, if it's the
    opponent's λ). Callers pass the SAME returned value to every player on
    that team for that scenario — this is what produces the covariance."""
    return int(rng.poisson(max(0.0, team_lambda)))


def split_multinomial(
    rng: np.random.Generator,
    team_total: int,
    players: Sequence[Mapping],
) -> dict[int, int]:
    """Assign a shared team-level count (goals or assists, already drawn once
    via ``sample_team_goals``) across players by weight × minutes_frac — the
    same anchor as ``goals.distribute_team_goals``, but splitting an actual
    drawn integer via a multinomial instead of each player independently
    redrawing their own Poisson. All players see the same ``team_total`` in a
    given scenario, so a big scenario for the team is a big scenario for
    everyone who might have scored — the source of within-team correlation on
    the attacking side (conditional on the shared total, individual shares are
    a zero-sum split, same as one player's goal "taking away" from another's).
    """
    ids = [int(p["player_id"]) for p in players]
    if not ids:
        return {}
    if team_total <= 0:
        return dict.fromkeys(ids, 0)
    weights = np.array(
        [max(0.0, float(p.get("weight", 0.0))) * max(0.0, float(p.get("minutes_frac", 0.0)))
         for p in players],
        dtype=float,
    )
    total_w = weights.sum()
    if total_w <= 0:
        weights = np.array([max(0.0, float(p.get("minutes_frac", 0.0))) for p in players])
        total_w = weights.sum()
        if total_w <= 0:
            # no one expected to feature — assign nothing rather than an
            # arbitrary uniform split among players who won't play
            return dict.fromkeys(ids, 0)
    probs = weights / total_w
    counts = rng.multinomial(team_total, probs)
    return dict(zip(ids, counts.tolist(), strict=True))
