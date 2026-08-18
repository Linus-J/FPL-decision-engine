"""P3-3: the risk-adjusted objective wired into optimise_squad/
optimise_starting_xi/evaluate_transfers.

Two things matter here: (1) with the DEFAULT config (risk_level=0.0,
ownership=None — today's actual live state pre-GW1), ownership/EO must
have EXACTLY zero effect (lambda=0 at risk_level=0), and `total_xpts`/
`xpts_gain` must report TRUE expected points, not the risk-adjusted score
— true regardless of ``mu`` (see plan/risk-aware-cold-start-v1.md: mu is
no longer 0 by default, but that only affects the variance term, never
ownership/EO or the true-xpts reporting). (2) with EO data + risk_level
!= 0, EO must actually change captain/selection choices when it's the
only thing distinguishing two options.
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
    # risk_level defaults to 0.0 -> lam=0 (mu is non-zero by default now,
    # but mu only touches the variance term, never EO) -> EO must have
    # zero effect either way
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
        risk_level=1.0, max_ownership_differential=0.5, mu_baseline=0.0, mu_range=0.0
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


def test_optimise_starting_xi_config_param_overrides_without_monkeypatch():
    """Simulation-engine entry point: passing ``config=`` directly (no
    monkeypatch of the module global) must have the same effect as
    ``test_optimise_starting_xi_aggressive_mode_prefers_low_eo_captain``'s
    monkeypatch-based override -- proves the explicit-parameter path (used
    by ``run_for_persona``) actually takes effect, not just that it's
    harmless when omitted (already covered by the untouched suite)."""
    aggressive = OptimiserConfig(
        risk_level=1.0, max_ownership_differential=0.5, mu_baseline=0.0, mu_range=0.0
    )
    squad = _minimal_squad_for_xi()
    gw = 10
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw,
         "xpts": 8.0 if pid in (8, 9) else 5.0, "xpts_var": 0.0}
        for pid in squad["id"]
    ])
    ownership = pd.DataFrame([
        {"player_id": 8, "top10k_selected_pct": 5.0},
        {"player_id": 9, "top10k_selected_pct": 90.0},
    ])
    solution = optimise_starting_xi(squad, projections, gw, ownership=ownership, config=aggressive)
    assert solution.captain_id == 8, "explicit config= override should behave like the global one"


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


def test_evaluate_transfers_transfer_rules_param_overrides_without_monkeypatch():
    """Simulation-engine entry point: passing ``transfer_rules=`` directly
    must behave like monkeypatching the module's ``TRANSFERS`` global (see
    ``test_transfer_switching_cost_disabled_allows_a_gain_the_default_cost_blocks``)."""
    import dataclasses

    from config.strategy import TRANSFERS as real_transfers

    # +0.5 pts/GW * 3 GWs = 1.5 total gain: blocked under the default 1.5pt
    # switching cost, allowed once it's disabled via the explicit parameter.
    players, squad_ids, projections = _squad_with_one_candidate_upgrade(0.5)
    blocked_plan = evaluate_transfers(
        squad_ids, projections, players, free_transfers=1, available_budget=100.0
    )
    assert blocked_plan.transfers_in == []

    allowed_plan = evaluate_transfers(
        squad_ids, projections, players, free_transfers=1, available_budget=100.0,
        transfer_rules=dataclasses.replace(real_transfers, transfer_switching_cost=0.0),
    )
    assert any(t["player_id"] == 16 for t in allowed_plan.transfers_in)


# --- transfer_switching_cost (2026-07-29, real case: Bruno Fernandes sold at
# GW10 for Gakpo despite 4 straight solidly-scoring gameweeks) -------------

def _squad_with_one_candidate_upgrade(candidate_gain_per_gw: float):
    """15-player owned squad (all flat 4.0 xpts/GW) + one extra FWD
    candidate projecting `candidate_gain_per_gw` more than the weakest
    owned FWD, over a 3-GW horizon."""
    owned = _owned_squad_round_robin()
    candidate = pd.DataFrame([{
        "id": 16, "position": "FWD", "now_cost": 5.0,
        "team_id": 6, "status": "a", "web_name": "candidate",
    }])
    players = pd.concat([owned, candidate], ignore_index=True)
    gws = [10, 11, 12]
    rows = []
    for pid in owned["id"]:
        for gw in gws:
            rows.append({
                "player_id": pid, "gameweek": gw, "xpts": 4.0, "start_probability": 0.9,
            })
    for gw in gws:
        rows.append({
            "player_id": 16, "gameweek": gw,
            "xpts": 4.0 + candidate_gain_per_gw, "start_probability": 0.9,
        })
    projections = pd.DataFrame(rows)
    return players, owned["id"].tolist(), projections


def test_transfer_switching_cost_blocks_a_marginal_upgrade():
    # +0.3 pts/GW * 3 GWs = 0.9 total gain -- well under the 1.5-point
    # switching cost (see the "disabled" test below for the isolated,
    # cost-specific threshold). Should NOT transfer for a noise-sized edge.
    players, squad_ids, projections = _squad_with_one_candidate_upgrade(0.3)
    plan = evaluate_transfers(
        squad_ids, projections, players, free_transfers=1, available_budget=100.0
    )
    assert plan.transfers_in == []


def test_transfer_switching_cost_allows_a_substantial_upgrade():
    # +5 pts/GW * 3 GWs = 15 total gain -- comfortably clears the cost.
    players, squad_ids, projections = _squad_with_one_candidate_upgrade(5.0)
    plan = evaluate_transfers(
        squad_ids, projections, players, free_transfers=1, available_budget=100.0
    )
    assert any(t["player_id"] == 16 for t in plan.transfers_in)


def test_transfer_switching_cost_disabled_allows_a_gain_the_default_cost_blocks(monkeypatch):
    import dataclasses

    from optimiser import transfers as transfers_module
    from optimiser.transfers import TRANSFERS as real_transfers

    # +0.5 pts/GW * 3 GWs = 1.5 total gain: confirmed blocked under the
    # default 1.5-point switching cost (right at the margin), but transfers
    # cleanly once the cost is disabled -- isolates the cost's own effect
    # from any other pre-existing friction in the multi-period ILP (e.g.
    # ft_terminal_value), which also makes very small gains a wash on their
    # own regardless of this setting.
    players, squad_ids, projections = _squad_with_one_candidate_upgrade(0.5)
    blocked_plan = evaluate_transfers(
        squad_ids, projections, players, free_transfers=1, available_budget=100.0
    )
    assert blocked_plan.transfers_in == []

    monkeypatch.setattr(
        transfers_module, "TRANSFERS",
        dataclasses.replace(real_transfers, transfer_switching_cost=0.0),
    )
    allowed_plan = evaluate_transfers(
        squad_ids, projections, players, free_transfers=1, available_budget=100.0
    )
    assert any(t["player_id"] == 16 for t in allowed_plan.transfers_in)


def _pool_with_a_backup_gk_choice():
    """2 GKP (a fixed starter + a choice of backup), 6 DEF, 6 MID, 4 FWD --
    enough real cost/xPts variety elsewhere that spending slack budget has a
    genuine opportunity cost, isolating the backup-GK decision specifically.
    Empirically found (2026-07-30) at budget=100: with no bench-value
    weight, the solver leaves ~£2m unspent rather than upgrade a bench slot
    that contributes nothing to the objective; with a real weight, that
    slack is spent on the better backup instead."""
    rows: list[dict] = []
    pid_counter = [1]

    def add(position: str, cost: float, xpts: float, n: int = 1) -> None:
        for _ in range(n):
            pid = pid_counter[0]
            rows.append({
                "id": pid, "position": position, "now_cost": cost, "xpts": xpts,
                "team_id": 1 + (pid % 8), "status": "a", "start_probability": 0.9,
                "web_name": f"p{pid}",
            })
            pid_counter[0] += 1

    add("GKP", 8.0, 8.0)  # the starter, never in question
    weak_id = pid_counter[0]
    add("GKP", 4.0, 0.5)  # cheap, weak backup
    decent_id = pid_counter[0]
    add("GKP", 5.5, 3.0)  # pricier, meaningfully better backup
    add("DEF", 4.0, 4.0, n=3)
    add("DEF", 6.0, 7.0, n=3)
    add("MID", 5.0, 5.0, n=3)
    add("MID", 9.0, 9.0, n=3)
    add("FWD", 6.0, 6.0, n=2)
    add("FWD", 11.0, 10.0, n=2)

    players = pd.DataFrame(rows)
    gws = [10, 11, 12]
    projections = pd.DataFrame([
        {"player_id": r["id"], "gameweek": gw, "xpts": r["xpts"]}
        for r in rows for gw in gws
    ])
    return players, projections, weak_id, decent_id


def test_bench_gk_weight_zero_picks_the_cheapest_backup_gk(monkeypatch):
    import dataclasses

    from optimiser import squad as squad_module
    from optimiser.squad import OPTIMISER as real_optimiser

    monkeypatch.setattr(
        squad_module, "OPTIMISER",
        dataclasses.replace(real_optimiser, bench_gk_weight=0.0),
    )
    players, projections, weak_id, decent_id = _pool_with_a_backup_gk_choice()
    solution = optimise_squad(projections, players, budget=100.0, horizon=3)
    selected = set(solution.squad["id"])
    assert weak_id in selected
    assert decent_id not in selected


def test_bench_gk_weight_upgrades_the_backup_gk_when_enabled(monkeypatch):
    import dataclasses

    from optimiser import squad as squad_module
    from optimiser.squad import OPTIMISER as real_optimiser

    players, projections, weak_id, decent_id = _pool_with_a_backup_gk_choice()
    monkeypatch.setattr(
        squad_module, "OPTIMISER",
        dataclasses.replace(real_optimiser, bench_gk_weight=0.15),
    )
    solution = optimise_squad(projections, players, budget=100.0, horizon=3)
    selected = set(solution.squad["id"])
    assert decent_id in selected
    assert weak_id not in selected
    # the upgrade is a genuinely benched player, not accidentally promoted
    # into the starting XI
    assert decent_id not in solution.starting_xi["id"].tolist()
