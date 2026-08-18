"""A bench is an ordered queue, not a set.

FPL's automatic substitutions promote the first eligible bench player when a
starter does not appear, so bench slot 1 is worth P(at least one starter
blanks) and slot 3 needs three simultaneous absences. Weighting all four
equally underpays the slot that gets used and overpays the two that do not,
which is what buys four non-playing £4.0m enablers instead of one real
substitute.

Measured on the live GW1 XI (mean P(start) 0.93): P(>=1 outfield starter
misses) = 0.53, P(>=2) = 0.15, P(>=3) = 0.03. AIrsenal reaches the same shape
with hand-tuned constants (GK 0.03, outfield 0.65/0.30/0.10).
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from config.strategy import OPTIMISER
from optimiser.squad import optimise_squad

GWS = (1, 2, 3)


BUDGET = 88.0


def _pool_with_a_bench_defender_choice():
    """A pool where the fifth defender is a real trade, not a free upgrade.

    The budget binds: the cheap bench defender leaves the strongest possible
    XI at exactly £87.0m, and buying the £6.0m substitute who actually plays
    forces one midfielder down to a cheaper, worse one. So the choice costs
    1.0 xPts a gameweek of XI strength to gain 3.0 xPts a gameweek of bench —
    worth taking at a slot-1 weight of 0.53, not worth it at a flat 0.15.
    Without a binding budget both tests would pass for the wrong reason.
    """
    rows, proj = [], []
    pid = 0

    def add(position: str, cost: float, xpts: float) -> int:
        nonlocal pid
        pid += 1
        rows.append({
            "id": pid, "web_name": f"p{pid}", "position": position,
            "now_cost": cost, "team_id": pid, "status": "a",
            "start_probability": 0.9,
        })
        for gw in GWS:
            proj.append({
                "player_id": pid, "gameweek": gw, "xpts": xpts,
                "xpts_var": 1.0, "start_probability": 0.9,
            })
        return pid

    # The XI: 1 GK, 4 DEF, 5 MID, 1 FWD.
    add("GKP", 5.0, 5.0)
    for _ in range(4):
        add("DEF", 6.0, 5.0)
    for _ in range(5):
        add("MID", 7.0, 6.0)
    add("FWD", 7.0, 6.0)

    # Cheaper, worse midfielders — the only way to fund a better bench.
    for _ in range(2):
        add("MID", 5.0, 5.0)

    # Filler that never competes for a starting place.
    add("GKP", 4.0, 1.0)
    for _ in range(2):
        add("FWD", 4.0, 1.0)

    # The choice for the fifth defender, who will be the first substitute.
    decent = add("DEF", 6.0, 4.0)
    fodder = add("DEF", 4.0, 1.0)
    return pd.DataFrame(rows), pd.DataFrame(proj), decent, fodder


def test_slot_weighting_buys_a_first_substitute_who_actually_plays():
    players, projections, decent, fodder = _pool_with_a_bench_defender_choice()
    solution = optimise_squad(projections, players, budget=BUDGET, horizon=len(GWS))
    selected = set(solution.squad["id"])

    assert decent in selected, "the first bench slot is reached ~half of all gameweeks"
    assert fodder not in selected


def test_a_flat_bench_weight_takes_the_fodder_instead():
    """The behaviour this replaced. At a uniform 0.15 the first slot is
    underpaid by more than three times, so the solver banks the £2.0m."""
    players, projections, decent, fodder = _pool_with_a_bench_defender_choice()
    flat = dataclasses.replace(
        OPTIMISER, bench_slot_weights=(0.15, 0.15, 0.15), bench_gk_weight=0.15,
        bench_value_weight=1.0,
    )
    solution = optimise_squad(
        projections, players, budget=BUDGET, horizon=len(GWS), config=flat
    )
    selected = set(solution.squad["id"])

    assert fodder in selected
    assert decent not in selected


def test_bench_value_weight_zero_ignores_the_bench_entirely():
    players, projections, decent, fodder = _pool_with_a_bench_defender_choice()
    off = dataclasses.replace(OPTIMISER, bench_value_weight=0.0)
    solution = optimise_squad(
        projections, players, budget=BUDGET, horizon=len(GWS), config=off
    )

    assert fodder in set(solution.squad["id"])


def test_slot_weights_are_strictly_decreasing():
    """The solver relies on this to order the bench without an explicit
    constraint: with decreasing weights, putting the best substitute anywhere
    but slot 1 is dominated. If these were ever made equal or increasing, the
    assignment would become arbitrary and bench_order meaningless."""
    weights = OPTIMISER.bench_slot_weights
    assert list(weights) == sorted(weights, reverse=True)
    assert len(set(weights)) == len(weights)
    assert OPTIMISER.bench_gk_weight < weights[0]


def test_the_best_substitute_is_placed_in_the_first_bench_slot():
    players, projections, decent, _ = _pool_with_a_bench_defender_choice()
    solution = optimise_squad(projections, players, budget=BUDGET, horizon=len(GWS))

    bench = solution.squad[~solution.squad["is_starting"]]
    outfield = bench[bench["position"] != "GKP"].sort_values("bench_order")
    assert int(outfield.iloc[0]["id"]) == decent
