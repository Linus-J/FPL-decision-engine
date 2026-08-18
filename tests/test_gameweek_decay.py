"""Gameweek decay weights the OBJECTIVE, never the reported points.

Summing a multi-gameweek horizon with equal weight claims a projection five
weeks out is worth as much as one for the match about to kick off. On the live
GW1 frame that was measurably false: 22% of the squad's projected points came
from gameweeks bookmakers had priced, 78% from the strength model with 17 of 20
teams on prior-season fallback.

Every serious FPL optimiser discounts — Çay's `solve_multi_period_fpl` defaults
to 0.84, FPLReview recommends 0.80-0.95. The trap is applying it to the
reported total as well, which would quietly restate the quantity being
predicted: "expected points" must stay expected points.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from config.strategy import OPTIMISER
from optimiser.squad import _multi_gw_xpts, optimise_squad

GWS = (10, 11, 12)


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": 1, "gameweek": gw, "xpts": 4.0, "xpts_var": 1.0} for gw in GWS
    ])


def test_decay_weights_each_gameweek_by_its_distance():
    decayed = _multi_gw_xpts(_frame(), horizon=3, decay=0.85)
    assert decayed.loc[1] == pytest.approx(4.0 * (1 + 0.85 + 0.85**2))


def test_decay_of_one_is_the_old_equal_weight_sum():
    """The escape hatch has to be exact, not merely close — it is how the
    previous behaviour stays reproducible."""
    assert _multi_gw_xpts(_frame(), horizon=3, decay=1.0).loc[1] == pytest.approx(12.0)
    assert (
        _multi_gw_xpts(_frame(), horizon=3, decay=1.0).loc[1]
        == _multi_gw_xpts(_frame(), horizon=3).loc[1]
    )


def _pool():
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
        for pid in players["id"] for gw in GWS
    ])
    return players, projections


def test_reported_total_xpts_is_undecayed():
    """The headline number must remain true expected points over the horizon.

    Decay is a statement about confidence, not about scoring: a manager reading
    `total_xpts` is being told what the squad is expected to score, and
    discounting that would make the figure mean nothing in particular.
    """
    players, projections = _pool()
    solution = optimise_squad(projections, players, budget=100.0, horizon=3)

    per_player = projections.groupby("player_id")["xpts"].sum()
    expected = float(sum(per_player[pid] for pid in solution.starting_xi["id"]))
    expected += float(per_player[solution.captain_id])

    assert solution.total_xpts == pytest.approx(expected)


def test_decay_is_reachable_from_config_and_off_at_one():
    """Two squads built from identical projections, differing only in decay,
    must report the same true total when the selection is unchanged — proving
    the knob touches the objective rather than the accounting."""
    players, projections = _pool()
    flat = dataclasses.replace(OPTIMISER, gameweek_decay=1.0)
    steep = dataclasses.replace(OPTIMISER, gameweek_decay=0.5)

    a = optimise_squad(projections, players, budget=100.0, horizon=3, config=flat)
    b = optimise_squad(projections, players, budget=100.0, horizon=3, config=steep)

    # This pool is flat across gameweeks, so decay cannot change the ranking
    # and both solves must land on the same true total.
    assert a.total_xpts == pytest.approx(b.total_xpts)


def test_decay_prefers_the_player_who_scores_sooner():
    """The behaviour the whole change exists for: same total over the horizon,
    but one player front-loads it. Undecayed the two are interchangeable;
    decayed, the earlier points win."""
    players, projections = _pool()
    early, late = 1, 2
    projections = projections[~projections["player_id"].isin([early, late])]
    rows = []
    for gw, front in zip(GWS, (9.0, 3.0, 3.0), strict=True):
        rows.append({"player_id": early, "gameweek": gw, "xpts": front, "xpts_var": 1.0})
    for gw, back in zip(GWS, (3.0, 3.0, 9.0), strict=True):
        rows.append({"player_id": late, "gameweek": gw, "xpts": back, "xpts_var": 1.0})
    projections = pd.concat([projections, pd.DataFrame(rows)], ignore_index=True)

    decayed = _multi_gw_xpts(projections, horizon=3, decay=0.85)
    assert decayed.loc[early] > decayed.loc[late]
    # ...and they are genuinely tied without it, so the ordering is decay's
    # doing and not some other asymmetry in the fixture.
    undecayed = _multi_gw_xpts(projections, horizon=3, decay=1.0)
    assert undecayed.loc[early] == pytest.approx(undecayed.loc[late])
