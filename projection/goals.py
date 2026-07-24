"""goals.py — P3 goals component: distribute odds-implied team goals to players.

The odds anchor (team_goals.py) gives each side's expected goals λ. This splits
λ among a team's players by their attacking *weight* (shot rate × expected
minutes), so Σ player goals == team λ — the odds carry finishing/quality at the
team level, the weight only allocates *who* scores.

`weight` is deliberately generic: today it's per-90 shots (the real per-GW signal
we have), but swapping in per-90 npxG later is a drop-in change — the component
and the P10 assembly are unaffected. Pure + deterministic (MC sampling takes an
explicit RNG).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from config.strategy import SCORING

# FPL points per goal by position (26/27 = SCORING).
GOAL_POINTS = {
    "GK": SCORING.points_goal_gk, "GKP": SCORING.points_goal_gk,
    "DEF": SCORING.points_goal_def,
    "MID": SCORING.points_goal_mid,
    "FWD": SCORING.points_goal_fwd,
}


def distribute_team_goals(
    players: Sequence[Mapping],
    team_lambda: float,
) -> dict[int, float]:
    """Split a team's expected goals (``team_lambda``) among its players.

    ``players``: each mapping has ``player_id``, ``weight`` (per-90 shots or
    npxG), and ``minutes_frac`` (E[minutes]/90, from P1 — a benched player gets
    ~none of the goals). Returns ``{player_id: expected_goals}`` summing to
    ``team_lambda`` (anchor-conserving). If no player carries any weight, the
    goals are spread evenly across those expected to play.
    """
    contrib = {
        int(p["player_id"]): max(0.0, float(p.get("weight", 0.0)))
        * max(0.0, float(p.get("minutes_frac", 0.0)))
        for p in players
    }
    total = sum(contrib.values())
    if total <= 0:
        # degenerate: no shot data — spread over players expected to feature
        playing = {int(p["player_id"]): max(0.0, float(p.get("minutes_frac", 0.0)))
                   for p in players}
        tot_play = sum(playing.values())
        if tot_play <= 0:
            return {pid: 0.0 for pid in contrib}
        return {pid: team_lambda * w / tot_play for pid, w in playing.items()}
    return {pid: team_lambda * w / total for pid, w in contrib.items()}


def expected_goal_points(expected_goals: float, position: str) -> float:
    """Expected points from goals for a player (E[goals] × points-per-goal)."""
    return expected_goals * GOAL_POINTS.get(position, SCORING.points_goal_mid)


def sample_goals(rng: np.random.Generator, expected_goals: float) -> int:
    """One Monte-Carlo goal draw ~ Poisson(E[goals]) (P10 assembly)."""
    return int(rng.poisson(max(0.0, expected_goals)))
