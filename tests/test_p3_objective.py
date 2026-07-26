"""P3-3: the risk-adjusted objective wired into optimise_squad/
optimise_starting_xi/evaluate_transfers.

Two things matter here: (1) with the DEFAULT config (risk_mode="balanced",
ownership=None — today's actual live state pre-GW1), every one of these
functions must behave EXACTLY as before P3-3 existed — same picks, and
`total_xpts`/`xpts_gain` must report TRUE expected points, not the
risk-adjusted score (which happens to equal xpts when balanced, but the
reporting must be correct even when it doesn't). (2) with EO data +
risk_mode != balanced, EO must actually change captain/selection choices
when it's the only thing distinguishing two options.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.strategy import OptimiserConfig
from optimiser.squad import optimise_squad, optimise_starting_xi
from optimiser.transfers import evaluate_transfers


def _minimal_squad_for_xi():
    # 2 GKP, 5 DEF, 5 MID, 3 FWD -- exactly one valid starting XI shape,
    # with two MID candidates tied on xpts but very different EO
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = []
    for i, position in enumerate(positions):
        pid = i + 1
        rows.append({
            "id": pid, "position": position, "now_cost": 5.0,
            "team_id": 1 + (i % 5), "web_name": f"p{pid}",
        })
    return pd.DataFrame(rows)


def test_optimise_starting_xi_default_config_ignores_ownership_entirely():
    squad = _minimal_squad_for_xi()
    gw = 10
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "xpts": 5.0 + (pid % 3), "xpts_var": 2.0}
        for pid in squad["id"]
    ])
    ownership = pd.DataFrame([
        {"player_id": pid, "top10k_selected_pct": 90.0} for pid in squad["id"]
    ])
    without_eo = optimise_starting_xi(squad, projections, gw)
    with_eo_but_balanced = optimise_starting_xi(squad, projections, gw, ownership=ownership)
    # risk_mode defaults to "balanced" -> lam=mu=0 -> EO must have zero effect
    assert without_eo.captain_id == with_eo_but_balanced.captain_id
    assert without_eo.total_xpts == pytest.approx(with_eo_but_balanced.total_xpts)


def test_optimise_starting_xi_reports_true_xpts_not_risk_adjusted_score():
    squad = _minimal_squad_for_xi()
    gw = 10
    # every player identical xpts=5.0 so total_xpts is trivially checkable:
    # 11 starters * 5.0 + 1 captain bonus * 5.0 = 60.0, regardless of any
    # internal risk-adjustment happening to the ILP's objective
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "xpts": 5.0, "xpts_var": 3.0}
        for pid in squad["id"]
    ])
    solution = optimise_starting_xi(squad, projections, gw)
    assert solution.total_xpts == pytest.approx(60.0)


def test_optimise_starting_xi_aggressive_mode_prefers_low_eo_captain(monkeypatch):
    import optimiser.squad as squad_mod

    aggressive = OptimiserConfig(
        risk_mode="aggressive", max_ownership_differential=0.5, variance_weight=0.0
    )
    monkeypatch.setattr(squad_mod, "OPTIMISER", aggressive)

    squad = _minimal_squad_for_xi()
    gw = 10
    # everyone tied at xpts=5.0 EXCEPT two MIDs (pid 8 low-owned, pid 9
    # high-owned) both bumped to the clear best xpts=8.0 -- one of them
    # will be captain; which one is decided purely by EO under aggressive mode
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw,
         "xpts": 8.0 if pid in (8, 9) else 5.0, "xpts_var": 0.0}
        for pid in squad["id"]
    ])
    ownership = pd.DataFrame([
        {"player_id": 8, "top10k_selected_pct": 5.0},   # differential
        {"player_id": 9, "top10k_selected_pct": 90.0},  # template
    ])
    solution = optimise_starting_xi(squad, projections, gw, ownership=ownership)
    assert solution.captain_id == 8, "aggressive mode should captain the low-EO differential"


def test_optimise_squad_default_config_matches_pre_p3_3_behaviour():
    # a simple from-scratch build (no current_squad_ids) -- default config
    # (balanced, no ownership) must pick purely by raw xpts, as before
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
    gws = [10, 11, 12]
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "xpts": 4.0 + (pid % 5), "xpts_var": 1.0}
        for pid in players["id"] for gw in gws
    ])
    solution = optimise_squad(projections, players, budget=100.0, horizon=3)
    assert len(solution.squad) == 15
    # total_xpts must be the TRUE sum, not inflated/deflated by any risk term
    expected = float(
        solution.starting_xi.merge(
            projections[projections["gameweek"].isin(gws)].groupby("player_id")["xpts"].sum(),
            left_on="id", right_index=True,
        )["xpts"].sum()
    )
    captain_extra = float(
        projections[
            (projections["player_id"] == solution.captain_id)
            & (projections["gameweek"].isin(gws))
        ]["xpts"].sum()
    )
    assert solution.total_xpts == pytest.approx(expected + captain_extra)


def _owned_squad_round_robin():
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = []
    for i, position in enumerate(positions):
        pid = i + 1
        rows.append({
            "id": pid, "position": position, "now_cost": 5.0,
            "team_id": 1 + (i % 5), "status": "a", "web_name": f"p{pid}",
        })
    return pd.DataFrame(rows)


def test_evaluate_transfers_default_config_ignores_ownership_entirely():
    owned = _owned_squad_round_robin()
    squad_ids = owned["id"].tolist()
    gws = [10, 11, 12]
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "xpts": 4.0, "start_probability": 0.9}
        for pid in owned["id"] for gw in gws
    ])
    ownership = pd.DataFrame([
        {"player_id": pid, "top10k_selected_pct": 80.0} for pid in owned["id"]
    ])
    without_eo = evaluate_transfers(
        squad_ids, projections, owned, free_transfers=1, available_budget=100.0
    )
    with_eo = evaluate_transfers(
        squad_ids, projections, owned, free_transfers=1,
        available_budget=100.0, ownership=ownership,
    )
    assert without_eo.hits_taken == with_eo.hits_taken == 0
    assert without_eo.net_xpts_gain == pytest.approx(with_eo.net_xpts_gain)
