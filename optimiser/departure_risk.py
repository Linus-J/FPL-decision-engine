"""departure_risk.py — squad-construction gate (v2-build-plan §6.5).

xPts and the optimiser otherwise assume a player stays a PL player for the
whole planning horizon. A confirmed departure (FPL status='u' or dropped from
bootstrap entirely) is ground truth, needs no model — `confirmed_p_leave`
maps it straight to `p_leave=1.0`. A pre-confirmation RUMOUR needs the
Phase-4 news layer's LLM credibility grading (NOT built yet — this module
only provides the deterministic fusion mechanism `apply_departure_discount`
so that once a real `p_leave` source exists, wiring it in is a one-line
change, not a redesign).

Graduated handling (the plan's own wording):
  p_leave >= hard_exclude_p_leave  -> hard-exclude (never picked; force-sold
                                       immediately if already owned)
  rumour_floor <= p_leave < hard_exclude -> discount horizon xPts by
                                       P(stays) = 1 - p_leave
  p_leave < rumour_floor           -> no effect (too uncertain to act on)

Pure + deterministic. No DB/network.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from config.strategy import DEPARTURE_RISK, DepartureRiskRules

CONFIRMED_DEPARTURE_STATUSES = frozenset({"u"})


def confirmed_p_leave(status: str) -> float:
    """FPL ground truth -> p_leave. Only `status='u'` is currently modelled
    (an element dropped entirely from bootstrap never reaches this function
    at all, since it wouldn't be in our candidate pool to begin with —
    that case degrades safely, not silently: a removed player simply can't
    be selected, the same practical effect as p_leave=1.0 would produce)."""
    return 1.0 if status in CONFIRMED_DEPARTURE_STATUSES else 0.0


def is_hard_excluded(p_leave: float, rules: DepartureRiskRules = DEPARTURE_RISK) -> bool:
    return p_leave >= rules.hard_exclude_p_leave


def stay_probability_multiplier(
    p_leave: float, rules: DepartureRiskRules = DEPARTURE_RISK
) -> float:
    """The xPts multiplier for the RUMOUR tier: 1.0 (no effect) below the
    rumour floor, P(stays) = 1 - p_leave once a rumour is credible enough to
    act on, and 0.0 once it crosses into hard-exclude territory (handled by
    candidate-pool filtering instead, but included here so this function
    alone is a safe, total discount to apply)."""
    if p_leave >= rules.hard_exclude_p_leave:
        return 0.0
    if p_leave < rules.rumour_floor_p_leave:
        return 1.0
    return max(0.0, 1.0 - p_leave)


def apply_departure_discount(
    projections: pd.DataFrame,
    p_leave_by_player: Mapping[int, float],
    rules: DepartureRiskRules = DEPARTURE_RISK,
) -> pd.DataFrame:
    """Scales ``xpts``/``xpts_mean`` for every horizon GW of a player with a
    known ``p_leave`` (rumour tier only — hard-excludes are the candidate-
    pool filter's job, not a discount). Returns a copy; players not in
    ``p_leave_by_player`` are untouched. With an empty ``p_leave_by_player``
    (the current reality — no rumour source exists yet, Phase 4 unbuilt),
    this is a no-op."""
    if not p_leave_by_player or projections.empty:
        return projections
    out = projections.copy()
    for pid, p_leave in p_leave_by_player.items():
        mult = stay_probability_multiplier(p_leave, rules)
        mask = out["player_id"] == pid
        for col in ("xpts", "xpts_mean"):
            if col in out.columns:
                out.loc[mask, col] = out.loc[mask, col] * mult
    return out


def hard_excluded_ids(
    players: pd.DataFrame, rules: DepartureRiskRules = DEPARTURE_RISK
) -> set[int]:
    """Player ids to hard-exclude from a candidate pool, from ``players``'
    current ``status`` column. Ground-truth confirmed departures only today
    (the rumour tier needs a real p_leave source — Phase 4 — to ever reach
    this threshold from a non-ground-truth signal)."""
    if "status" not in players.columns:
        return set()
    p_leave = players["status"].map(confirmed_p_leave)
    return set(players.loc[p_leave >= rules.hard_exclude_p_leave, "id"].astype(int))
