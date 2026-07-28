import logging
from dataclasses import dataclass

import pandas as pd
import pulp

from config.strategy import OPTIMISER, SQUAD, TRANSFERS
from optimiser.departure_risk import confirmed_p_leave, is_hard_excluded
from optimiser.scoring import lambda_mu_for_risk_mode, risk_adjusted_score

logger = logging.getLogger(__name__)

SQUAD_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
STARTING_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}


@dataclass
class TransferPlan:
    transfers_in: list[dict]
    transfers_out: list[dict]
    hits_taken: int
    xpts_gain: float
    net_xpts_gain: float


def _squad_xpts(squad_ids: list[int], projections: pd.DataFrame, horizon: int) -> float:
    gws = sorted(projections["gameweek"].unique())[:horizon]
    subset = projections[
        projections["gameweek"].isin(gws) & projections["player_id"].isin(squad_ids)
    ]
    total = 0.0
    for gw in gws:
        gw_xpts = subset[subset["gameweek"] == gw]["xpts"].nlargest(11)
        captain_bonus = float(gw_xpts.max()) if len(gw_xpts) > 0 else 0.0
        total += float(gw_xpts.sum()) + captain_bonus
    return total


def evaluate_transfers(
    current_squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    free_transfers: int = 1,
    available_budget: float | None = None,
    wildcard_active: bool = False,
    dgw_gws: set[int] | None = None,
    solver_time_limit: int | None = None,
    ownership: pd.DataFrame | None = None,
) -> TransferPlan:
    """``ownership`` (P3-3, optional, default None): see ``optimise_squad``'s
    docstring — ``None`` (the current live reality — EO sampling can't
    produce real data pre-GW1) makes EO a uniform 0% for every candidate, a
    constant rescale that doesn't change which transfers get made."""
    horizon = OPTIMISER.transfer_planning_horizon_gws
    current_squad = players[players["id"].isin(current_squad_ids)].copy()
    budget = available_budget or float(current_squad["now_cost"].sum())

    gws = sorted(projections["gameweek"].unique())[:horizon]
    H = len(gws)
    if H == 0:
        return TransferPlan([], [], 0, 0.0, 0.0)

    candidate_pids = set(current_squad_ids)
    if "start_probability" in projections.columns:
        candidate_pids |= set(
            projections[projections["start_probability"] >= OPTIMISER.min_start_probability]["player_id"]
        )
    else:
        candidate_pids |= set(projections["player_id"].unique())

    df = players[players["id"].isin(candidate_pids)].copy()
    # Status filter applies to NOT-currently-owned candidates only (never buy
    # an unavailable/departed player). An OWNED player is kept regardless of
    # status so a confirmed departure (status='u') can be explicitly modelled
    # as a forced sale below, rather than silently dropped from the ILP's
    # variable set entirely — the latter left the squad-size constraint to
    # coincidentally force a replacement transfer in, but with no `tout`
    # variable for the departed player at all, so they never appeared in the
    # reported `transfers_out` (v2-build-plan §6.5 departure-risk gate fix).
    owned_mask = df["id"].isin(current_squad_ids)
    df = df[owned_mask | df["status"].isin(["a", "d"])]
    df = df.reset_index(drop=True)

    pid_list = df["id"].tolist()
    pid_set = set(pid_list)
    N = len(pid_list)
    pid_to_i = {pid: i for i, pid in enumerate(pid_list)}

    costs = df["now_cost"].tolist()
    positions = df["position"].tolist()
    teams = df["team_id"].tolist()
    statuses = df["status"].tolist()
    in_current = [1 if pid in set(current_squad_ids) else 0 for pid in pid_list]
    confirmed_departure_ids = {
        pid for pid, status, owned in zip(pid_list, statuses, in_current, strict=True)
        if owned and is_hard_excluded(confirmed_p_leave(status))
    }

    lam, mu = lambda_mu_for_risk_mode(
        OPTIMISER.risk_mode, OPTIMISER.max_ownership_differential, OPTIMISER.variance_weight
    )
    # P3-6: always-on optimiser's-curse discount for the weekly transfer
    # search, independent of risk_mode (see TransferRules.transfer_variance_penalty).
    mu -= TRANSFERS.transfer_variance_penalty
    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        eo_by_pid = {pid: float(eo_map.get(pid, 0.0)) for pid in pid_list}
    else:
        eo_by_pid = dict.fromkeys(pid_list, 0.0)
    has_var = "xpts_var" in projections.columns

    # xpts_gain/net_xpts_gain reporting reads straight from `projections` via
    # _squad_xpts (true expected points) -- only the ILP's own objective
    # coefficients need the risk-adjusted score.
    scores_pw: dict[tuple[int, int], float] = {}
    for w, gw in enumerate(gws):
        gw_df = projections[projections["gameweek"] == gw].set_index("player_id")
        gw_proj = gw_df["xpts"]
        gw_var = gw_df["xpts_var"] if has_var else None
        for pid in pid_list:
            x = float(gw_proj.get(pid, 0.0))
            v = float(gw_var.get(pid, 0.0)) if gw_var is not None else 0.0
            scores_pw[(pid, w)] = risk_adjusted_score(x, v, eo_by_pid[pid], lam, mu)

    prob = pulp.LpProblem("mp_transfers", pulp.LpMaximize)

    squad = {
        (pid, w): pulp.LpVariable(f"sq_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    starting = {
        (pid, w): pulp.LpVariable(f"st_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    tin = {
        (pid, w): pulp.LpVariable(f"ti_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    tout = {
        (pid, w): pulp.LpVariable(f"to_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    captain = {
        (pid, w): pulp.LpVariable(f"cp_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    ft = {w: pulp.LpVariable(f"ft_{w}", lowBound=1, upBound=5, cat="Integer") for w in range(H + 1)}
    hit = {w: pulp.LpVariable(f"hit_{w}", lowBound=0, upBound=15, cat="Integer") for w in range(H)}
    n_trans = {w: pulp.LpVariable(f"nt_{w}", lowBound=0, upBound=15, cat="Integer") for w in range(H)}

    if wildcard_active:
        prob += ft[0] == 15
    else:
        prob += ft[0] == free_transfers

    # Departure-risk gate (§6.5): a confirmed departure (status='u') already
    # owned must be sold immediately, not merely made ineligible to buy back
    # (the generic "can't tin an owned player at w=0" constraint below
    # already prevents re-buying) — forcing tout==1 here (rather than
    # omitting them from the model, the prior behaviour) means they
    # correctly show up in the reported transfers_out and go through the
    # normal hit/FT accounting.
    for pid in confirmed_departure_ids:
        prob += tout[(pid, 0)] == 1

    for w in range(H):
        for i, pid in enumerate(pid_list):
            if w == 0:
                prob += squad[(pid, w)] == in_current[i] + tin[(pid, w)] - tout[(pid, w)]
            else:
                prob += squad[(pid, w)] == squad[(pid, w - 1)] + tin[(pid, w)] - tout[(pid, w)]

            prob += tin[(pid, w)] + tout[(pid, w)] <= 1

            if in_current[i] == 0 and w == 0:
                prob += tout[(pid, w)] == 0
            if in_current[i] == 1 and w == 0:
                prob += tin[(pid, w)] == 0

            prob += captain[(pid, w)] <= starting[(pid, w)]
            prob += starting[(pid, w)] <= squad[(pid, w)]

        prob += pulp.lpSum(squad[(pid, w)] for pid in pid_list) == SQUAD.squad_size
        prob += pulp.lpSum(starting[(pid, w)] for pid in pid_list) == 11
        prob += pulp.lpSum(costs[i] * squad[(pid_list[i], w)] for i in range(N)) <= budget

        for pos, count in SQUAD_COUNTS.items():
            pos_pids = [pid for pid, p in zip(pid_list, positions) if p == pos]
            prob += pulp.lpSum(squad[(pid, w)] for pid in pos_pids) == count
            prob += pulp.lpSum(starting[(pid, w)] for pid in pos_pids) >= STARTING_MIN[pos]
            prob += pulp.lpSum(starting[(pid, w)] for pid in pos_pids) <= STARTING_MAX[pos]

        team_ids = list(set(teams))
        for tid in team_ids:
            tid_pids = [pid for pid, t in zip(pid_list, teams) if t == tid]
            prob += pulp.lpSum(squad[(pid, w)] for pid in tid_pids) <= SQUAD.max_players_per_club

        prob += pulp.lpSum(tin[(pid, w)] for pid in pid_list) == pulp.lpSum(tout[(pid, w)] for pid in pid_list)
        prob += n_trans[w] == pulp.lpSum(tin[(pid, w)] for pid in pid_list)
        prob += pulp.lpSum(captain[(pid, w)] for pid in pid_list) == 1

        prob += hit[w] >= n_trans[w] - ft[w]

        if wildcard_active and w == 0:
            prob += hit[0] == 0
        else:
            prob += hit[w] <= TRANSFERS.max_hits_per_gw

        prob += ft[w + 1] <= ft[w] - n_trans[w] + 1
        prob += ft[w + 1] <= TRANSFERS.max_banked_free_transfers
        prob += ft[w + 1] >= 1

    prob += pulp.lpSum(
        scores_pw[(pid, w)] * starting[(pid, w)]
        + scores_pw[(pid, w)] * captain[(pid, w)]
        - hit[w] * abs(TRANSFERS.hit_cost_points)
        for pid in pid_list
        for w in range(H)
    ) + TRANSFERS.ft_terminal_value * ft[H]

    solver_args = {"msg": False}
    if solver_time_limit:
        solver_args["timeLimit"] = solver_time_limit
    prob.solve(pulp.PULP_CBC_CMD(**solver_args))

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Not Solved"):
        logger.warning("Multi-period ILP status: %s — returning no transfers", status)
        return TransferPlan([], [], 0, 0.0, 0.0)

    gw0_in = [pid for pid in pid_list if (pulp.value(tin[(pid, 0)]) or 0) > 0.5]
    gw0_out = [pid for pid in pid_list if (pulp.value(tout[(pid, 0)]) or 0) > 0.5]

    actual_hits = max(0, len(gw0_in) - (free_transfers if not wildcard_active else 15))

    new_squad_ids = [pid for pid in current_squad_ids if pid not in set(gw0_out)] + gw0_in
    xpts_after = _squad_xpts(new_squad_ids, projections, horizon)
    xpts_before = _squad_xpts(current_squad_ids, projections, horizon)
    xpts_gain = xpts_after - xpts_before
    net_xpts_gain = xpts_gain + actual_hits * TRANSFERS.hit_cost_points

    def _player_info(pid: int) -> dict:
        row = players[players["id"] == pid]
        return {
            "player_id": pid,
            "web_name": row["web_name"].values[0] if len(row) else str(pid),
            "cost": float(row["now_cost"].values[0]) if len(row) else 0.0,
        }

    plan = TransferPlan(
        transfers_in=[_player_info(pid) for pid in gw0_in],
        transfers_out=[_player_info(pid) for pid in gw0_out],
        hits_taken=actual_hits,
        xpts_gain=xpts_gain,
        net_xpts_gain=net_xpts_gain,
    )

    logger.info(
        "Transfer plan: %d in / %d out, %d hits, net gain %.2f xPts [ILP status: %s]",
        len(plan.transfers_in), len(plan.transfers_out),
        plan.hits_taken, plan.net_xpts_gain, status,
    )
    return plan


def should_take_hit(plan: TransferPlan) -> bool:
    if plan.hits_taken == 0:
        return False
    return plan.net_xpts_gain > 0
