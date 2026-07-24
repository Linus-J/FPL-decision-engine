"""assists.py — P4 assists component.

Every goal has ~0.75 chance of a recorded assist, so a team's expected assists
≈ its expected goals (team λ, from the odds anchor) × an assisted-goal fraction.
Those are distributed among players by a creativity **weight** × minutes_frac —
the same anchor-conserving split as goals (reused from goals.py).

The `weight` is generic: ideally per-90 xA / key-passes, but those aren't in the
free feed yet (see the xG note in plan/phase-2-xpts-engine.md), so the caller
passes the best available creativity signal (interim: rolling actual assists);
swapping in xA when the SportMonks feed lands is a drop-in change. FPL assists ≈
Opta *primary* assists — ASSIST_FRACTION is the tunable calibration knob.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from config.strategy import SCORING
from projection.goals import distribute_team_goals

# Share of goals that yield a recorded (primary) assist. Calibrate FPL-vs-Opta
# against realised assist rates (P4 gate); ~0.75 is a reasonable prior.
ASSIST_FRACTION = 0.75


def distribute_team_assists(
    players: Sequence[Mapping],
    team_lambda: float,
    assist_fraction: float = ASSIST_FRACTION,
) -> dict[int, float]:
    """Expected assists per player: split (team λ × assist_fraction) by each
    player's creativity weight × minutes_frac. Anchor-conserving (Σ = team
    assists). Reuses the goals distributor — mathematically the same split."""
    return distribute_team_goals(players, team_lambda * assist_fraction)


def expected_assist_points(expected_assists: float) -> float:
    return expected_assists * SCORING.points_assist


def sample_assists(rng: np.random.Generator, expected_assists: float) -> int:
    return int(rng.poisson(max(0.0, expected_assists)))
