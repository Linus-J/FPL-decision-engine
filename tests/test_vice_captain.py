"""The vice-captaincy has to be worth something in the objective.

`vice` was constrained -- exactly one, must be a starter, must not be the
captain -- and appeared nowhere in the objective, so every legal choice tied
and the solver returned whichever it happened to branch on. On the live GW1
frame that was Raya at 5.53 xPts while Gabriel sat in the same XI on 7.43.

The armband passes to the vice only when the captain does not feature, so his
expected contribution is P(captain blanks) x his own score. Measured on this
engine's own start probabilities, the top ten by GW1 xPts average 0.827, so
that probability is about 0.17 -- not a rounding error.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from config.strategy import OPTIMISER
from optimiser.squad import optimise_squad, optimise_starting_xi

GWS = (10, 11, 12)


def _pool():
    """A pool with an unambiguous best and second-best in the XI."""
    positions = ["GKP"] * 4 + ["DEF"] * 8 + ["MID"] * 8 + ["FWD"] * 5
    rows, proj = [], []
    for i, position in enumerate(positions):
        pid = i + 1
        rows.append({
            "id": pid, "position": position, "now_cost": 4.5,
            "team_id": 1 + (i % 8), "status": "a", "start_probability": 0.9,
            "web_name": f"p{pid}",
        })
        # Two clear standouts, both outfield so neither is forced by position.
        xpts = 9.0 if pid == 13 else (8.0 if pid == 14 else 3.0 + (pid % 3))
        for gw in GWS:
            proj.append({
                "player_id": pid, "gameweek": gw, "xpts": xpts, "xpts_var": 1.0,
            })
    return pd.DataFrame(rows), pd.DataFrame(proj)


def test_the_vice_is_the_best_starter_who_is_not_the_captain():
    players, projections = _pool()
    solution = optimise_squad(projections, players, budget=100.0, horizon=3)

    assert solution.captain_id == 13
    assert solution.vice_captain_id == 14, (
        "the vice must be the next-best starter, not an arbitrary tie-break"
    )


def test_the_vice_is_never_the_captain():
    players, projections = _pool()
    solution = optimise_squad(projections, players, budget=100.0, horizon=3)
    assert solution.captain_id != solution.vice_captain_id


def test_the_vice_is_always_a_starter():
    players, projections = _pool()
    solution = optimise_squad(projections, players, budget=100.0, horizon=3)
    assert solution.vice_captain_id in set(solution.starting_xi["id"])


def test_starting_xi_picks_the_vice_on_merit_too():
    """optimise_starting_xi has its own solve and had the same gap; the weekly
    lineup call is where the choice actually gets made most often."""
    players, projections = _pool()
    squad = optimise_squad(projections, players, budget=100.0, horizon=3).squad

    xi = optimise_starting_xi(squad, projections, GWS[0])
    assert xi.captain_id == 13
    assert xi.vice_captain_id == 14


def test_the_weight_does_not_buy_squad_places():
    """The vice term must break ties, not bid for selection. Zeroing it may not
    change the reported total, because total_xpts counts the XI and the captain
    and never the vice."""
    players, projections = _pool()
    off = dataclasses.replace(OPTIMISER, vice_captain_weight=0.0)

    with_weight = optimise_squad(projections, players, budget=100.0, horizon=3)
    without = optimise_squad(projections, players, budget=100.0, horizon=3, config=off)

    assert set(with_weight.squad["id"]) == set(without.squad["id"])
    assert with_weight.total_xpts == without.total_xpts


def test_the_weight_is_small_enough_to_stay_a_tie_break():
    """A vice worth more than a starting place would distort selection. It
    represents P(captain blanks), which is well under a half by construction."""
    assert 0.0 < OPTIMISER.vice_captain_weight < 0.5
