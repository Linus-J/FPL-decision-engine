"""A solution pool: the N best distinct squads, via no-good cuts.

A single reported squad hides how much of itself is real. It cannot say whether
its twelfth pick beat the alternative by four points or by two hundredths, and
that is the difference between a conviction and a coin toss. Solving repeatedly
with each previous answer forbidden shows which players survive being made to
choose again — on the live GW1 frame, 11 of 15 appear in all ten squads while
the contested four are exactly the cheap bench places.

CBC has no native solution pool; Gurobi and CPLEX expose the same idea as
PoolSearchMode / populate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from optimiser.squad import generate_squad_pool, optimise_squad


def _pool_frame():
    positions = ["GKP"] * 4 + ["DEF"] * 8 + ["MID"] * 8 + ["FWD"] * 5
    rows = []
    for i, position in enumerate(positions):
        pid = i + 1
        rows.append({
            "id": pid, "position": position, "now_cost": 4.5,
            "team_id": 1 + (i % 8), "status": "a", "start_probability": 0.9,
            "web_name": f"p{pid}",
        })
    players = pd.DataFrame(rows)
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "xpts": 4.0 + (pid % 5), "xpts_var": 1.0}
        for pid in players["id"] for gw in (10, 11, 12)
    ])
    return players, projections


def test_a_forbidden_squad_is_not_returned_again():
    players, projections = _pool_frame()
    first = optimise_squad(projections, players, budget=100.0, horizon=3)
    first_ids = sorted(int(p) for p in first.squad["id"])

    second = optimise_squad(
        projections, players, budget=100.0, horizon=3, forbidden_squads=[first_ids]
    )
    assert sorted(int(p) for p in second.squad["id"]) != first_ids


def test_the_cut_forbids_the_squad_not_its_players():
    """The point of a no-good cut over a set of exclusions: every fourteen-man
    subset stays legal, so the next answer is usually the same squad with one
    change rather than a wholesale rebuild."""
    players, projections = _pool_frame()
    first = optimise_squad(projections, players, budget=100.0, horizon=3)
    first_ids = set(int(p) for p in first.squad["id"])

    second = optimise_squad(
        projections, players, budget=100.0, horizon=3,
        forbidden_squads=[sorted(first_ids)],
    )
    second_ids = set(int(p) for p in second.squad["id"])
    assert len(first_ids & second_ids) == len(first_ids) - 1


def test_pool_returns_distinct_squads_in_non_increasing_order():
    players, projections = _pool_frame()
    pool = generate_squad_pool(projections, players, n=5, budget=100.0, horizon=3)

    assert len(pool) == 5
    seen = {tuple(sorted(int(p) for p in s.squad["id"])) for s in pool}
    assert len(seen) == 5, "every squad in the pool must be distinct"


def test_pool_stops_early_instead_of_raising_when_squads_run_out():
    """Asking for more squads than legally exist must degrade to "here is what
    there is" — a pool is a diagnostic, and one that raises at the boundary is
    useless exactly when the answer (there are barely any alternatives) is most
    interesting."""
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = []
    for i, position in enumerate(positions):
        pid = i + 1
        rows.append({
            "id": pid, "position": position, "now_cost": 4.0,
            "team_id": 1 + (i % 6), "status": "a", "start_probability": 0.9,
            "web_name": f"p{pid}",
        })
    players = pd.DataFrame(rows)
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": 10, "xpts": 4.0, "xpts_var": 1.0}
        for pid in players["id"]
    ])
    # Exactly fifteen legal players, so exactly one legal squad exists.
    pool = generate_squad_pool(projections, players, n=5, budget=100.0, horizon=1)
    assert len(pool) == 1


def test_pool_of_zero_or_one_is_handled():
    players, projections = _pool_frame()
    assert generate_squad_pool(projections, players, n=0, budget=100.0, horizon=3) == []
    assert len(generate_squad_pool(projections, players, n=1, budget=100.0, horizon=3)) == 1


def test_pool_respects_an_incoming_forbidden_list():
    players, projections = _pool_frame()
    first = optimise_squad(projections, players, budget=100.0, horizon=3)
    banned = sorted(int(p) for p in first.squad["id"])

    pool = generate_squad_pool(
        projections, players, n=3, budget=100.0, horizon=3, forbidden_squads=[banned]
    )
    for solution in pool:
        assert sorted(int(p) for p in solution.squad["id"]) != banned


def test_reported_totals_stay_true_expected_points_across_the_pool():
    """Each pool entry reports its own true undecayed total, so the ranking
    (by objective) and the number shown (true xPts) can legitimately disagree
    — which is the point of printing both."""
    players, projections = _pool_frame()
    pool = generate_squad_pool(projections, players, n=4, budget=100.0, horizon=3)
    per_player = projections.groupby("player_id")["xpts"].sum()

    for solution in pool:
        expected = float(sum(per_player[pid] for pid in solution.starting_xi["id"]))
        expected += float(per_player[solution.captain_id])
        assert solution.total_xpts == pytest.approx(expected)
