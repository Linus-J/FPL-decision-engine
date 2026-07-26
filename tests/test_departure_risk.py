"""Departure-risk gate (v2-build-plan §6.5) — the squad-construction gate
that hard-excludes/force-sells confirmed departures.

Pure functions (confirmed_p_leave, stay_probability_multiplier,
apply_departure_discount, hard_excluded_ids) plus the real bug fix in
evaluate_transfers: an already-OWNED player who becomes status='u' used to
be silently dropped from the ILP's variable set entirely (no `tout`
variable existed for them at all), so they never appeared in the reported
transfers_out even though the squad-size constraint happened to force a
replacement in. Now they're forced out explicitly (tout==1) and correctly
reported.
"""

from __future__ import annotations

import pandas as pd
import pytest

from optimiser.departure_risk import (
    apply_departure_discount,
    confirmed_p_leave,
    hard_excluded_ids,
    is_hard_excluded,
    stay_probability_multiplier,
)
from optimiser.transfers import evaluate_transfers


def test_confirmed_p_leave_only_status_u():
    assert confirmed_p_leave("u") == 1.0
    for status in ("a", "d", "i", "s", "n"):
        assert confirmed_p_leave(status) == 0.0


def test_is_hard_excluded_threshold():
    assert is_hard_excluded(0.7) is True
    assert is_hard_excluded(0.69) is False
    assert is_hard_excluded(1.0) is True


def test_stay_probability_multiplier_three_tiers():
    assert stay_probability_multiplier(0.1) == 1.0        # below rumour floor -> no effect
    assert stay_probability_multiplier(0.5) == pytest.approx(0.5)  # rumour tier -> 1 - p_leave
    assert stay_probability_multiplier(0.9) == 0.0         # hard-exclude tier -> zeroed


def test_apply_departure_discount_scales_only_named_players():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 5, "xpts": 10.0, "xpts_mean": 10.0},
        {"player_id": 2, "gameweek": 5, "xpts": 8.0, "xpts_mean": 8.0},
    ])
    out = apply_departure_discount(projections, {1: 0.5})
    assert out.loc[out["player_id"] == 1, "xpts"].iloc[0] == pytest.approx(5.0)
    assert out.loc[out["player_id"] == 2, "xpts"].iloc[0] == 8.0  # untouched


def test_apply_departure_discount_empty_map_is_noop():
    projections = pd.DataFrame([{"player_id": 1, "gameweek": 5, "xpts": 10.0}])
    out = apply_departure_discount(projections, {})
    pd.testing.assert_frame_equal(out, projections)


def test_hard_excluded_ids_from_status_column():
    players = pd.DataFrame([
        {"id": 1, "status": "a"}, {"id": 2, "status": "u"}, {"id": 3, "status": "d"},
    ])
    assert hard_excluded_ids(players) == {2}


def test_hard_excluded_ids_missing_status_column_is_safe():
    assert hard_excluded_ids(pd.DataFrame([{"id": 1}])) == set()


# --- evaluate_transfers: the real bug fix ----------------------------------

def _owned_squad():
    # 2 GKP, 5 DEF, 5 MID, 3 FWD across 5 teams, round-robin so each team
    # gets exactly 3 players (respects max_players_per_club=3)
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    costs = {"GKP": 4.5, "DEF": 4.5, "MID": 5.0, "FWD": 5.5}
    rows = []
    for i, position in enumerate(positions):
        pid = i + 1
        rows.append({
            "id": pid, "position": position, "now_cost": costs[position],
            "team_id": 1 + (i % 5), "status": "a", "web_name": f"p{pid}",
        })
    return pd.DataFrame(rows)


def _replacement_candidates(start_id: int):
    # extra DEF options on teams NOT already at the 3-per-club cap
    rows = []
    for i, team in enumerate((6, 7, 8)):
        rows.append({"id": start_id + i, "position": "DEF", "now_cost": 4.5,
                     "team_id": team, "status": "a", "web_name": f"repl{i}"})
    return pd.DataFrame(rows)


def test_evaluate_transfers_force_sells_confirmed_departure():
    owned = _owned_squad()
    departed_id = owned[owned["position"] == "DEF"]["id"].iloc[0]
    owned.loc[owned["id"] == departed_id, "status"] = "u"  # confirmed departure

    replacements = _replacement_candidates(start_id=100)
    players = pd.concat([owned, replacements], ignore_index=True)
    squad_ids = owned["id"].tolist()

    gws = [10, 11, 12]
    proj_rows = []
    for pid in players["id"]:
        base = 8.0 if pid in replacements["id"].values else 4.0
        base = 0.0 if pid == departed_id else base
        for gw in gws:
            proj_rows.append({
                "player_id": pid, "gameweek": gw, "xpts": base,
                "start_probability": 0.9,
            })
    projections = pd.DataFrame(proj_rows)

    plan = evaluate_transfers(
        current_squad_ids=squad_ids,
        projections=projections,
        players=players,
        free_transfers=1,
        available_budget=100.0,
    )

    out_ids = {t["player_id"] for t in plan.transfers_out}
    in_ids = {t["player_id"] for t in plan.transfers_in}
    assert departed_id in out_ids, "confirmed departure must be reported as sold"
    assert len(in_ids) >= 1, "a replacement must be bought to refill the DEF slot"
    # exactly one DEF slot needed refilling from this scenario
    new_squad = [pid for pid in squad_ids if pid not in out_ids] + list(in_ids)
    assert len(new_squad) == 15
    assert players[players["id"].isin(new_squad)]["position"].value_counts().to_dict() == {
        "DEF": 5, "MID": 5, "FWD": 3, "GKP": 2,
    }


def test_evaluate_transfers_no_departure_is_unaffected():
    owned = _owned_squad()
    replacements = _replacement_candidates(start_id=100)
    players = pd.concat([owned, replacements], ignore_index=True)
    squad_ids = owned["id"].tolist()

    gws = [10, 11, 12]
    proj_rows = [
        {"player_id": pid, "gameweek": gw, "xpts": 4.0, "start_probability": 0.9}
        for pid in players["id"] for gw in gws
    ]
    projections = pd.DataFrame(proj_rows)

    plan = evaluate_transfers(
        current_squad_ids=squad_ids, projections=projections, players=players,
        free_transfers=1, available_budget=100.0,
    )
    # no forced sale needed -- a sensible plan makes 0 transfers when nothing
    # is meaningfully better (all replacement candidates project identically)
    assert plan.hits_taken == 0
