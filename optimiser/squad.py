import logging
from dataclasses import dataclass

import pandas as pd
import pulp

from config.strategy import OPTIMISER, SQUAD, TRANSFERS
from optimiser.captaincy import scenario_based_captain
from optimiser.scoring import lambda_mu_for_risk_mode, risk_adjusted_score

logger = logging.getLogger(__name__)

POSITIONS = ("GKP", "DEF", "MID", "FWD")

SQUAD_COUNTS = {
    "GKP": SQUAD.gk_count,
    "DEF": SQUAD.def_count,
    "MID": SQUAD.mid_count,
    "FWD": SQUAD.fwd_count,
}

STARTING_MIN = {
    "GKP": SQUAD.starting_gk,
    "DEF": SQUAD.starting_def_min,
    "MID": SQUAD.starting_mid_min,
    "FWD": SQUAD.starting_fwd_min,
}

STARTING_MAX = {
    "GKP": SQUAD.starting_gk,
    "DEF": SQUAD.starting_def_max,
    "MID": SQUAD.starting_mid_max,
    "FWD": SQUAD.starting_fwd_max,
}


@dataclass
class SquadSolution:
    squad: pd.DataFrame
    starting_xi: pd.DataFrame
    captain_id: int
    vice_captain_id: int
    total_xpts: float
    total_cost: float
    hits_taken: int


def _multi_gw_xpts(projections: pd.DataFrame, horizon: int) -> pd.Series:
    gws = sorted(projections["gameweek"].unique())[:horizon]
    subset = projections[projections["gameweek"].isin(gws)]
    return subset.groupby("player_id")["xpts"].sum()


def _multi_gw_var(projections: pd.DataFrame, horizon: int) -> pd.Series:
    """P3-3: per-player summed xpts_var over the horizon (own-variance only —
    see optimiser/scoring.py for why teammate covariance isn't here). Empty
    Series (not an error) if the caller's projections predate P10's
    distributional columns."""
    if "xpts_var" not in projections.columns:
        return pd.Series(dtype=float)
    gws = sorted(projections["gameweek"].unique())[:horizon]
    subset = projections[projections["gameweek"].isin(gws)]
    return subset.groupby("player_id")["xpts_var"].sum()


def optimise_squad(
    projections: pd.DataFrame,
    players: pd.DataFrame,
    budget: float = SQUAD.budget_total,
    horizon: int | None = None,
    current_squad_ids: list[int] | None = None,
    free_transfers: int = 1,
    max_transfers: int | None = None,
    force_include_ids: list[int] | None = None,
    force_exclude_ids: list[int] | None = None,
    ownership: pd.DataFrame | None = None,
    season: str | None = None,
) -> SquadSolution:
    """``ownership`` (P3-3, optional): a ``(player_id, top10k_selected_pct)``
    frame (P3-2) feeding the risk-adjusted objective's differential term.
    ``None`` (the current live reality — EO sampling can't produce real data
    pre-GW1) makes every player's EO 0%, which is a uniform rescale of the
    objective, not a ranking change — behaviour is identical to before
    this parameter existed.

    ``season`` (P3-4, optional): enables scenario-based captaincy (see
    ``optimiser/captaincy.py``) for the EARLIEST gameweek in the horizon —
    the one whose captain choice is actually about to be locked in.
    ``None`` (most callers — this function builds/rebuilds a SQUAD; final
    captaincy is usually decided later, per-GW, by ``optimise_starting_xi``)
    keeps the plain linear-argmax pick."""
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
    force_include_ids = set(force_include_ids or [])
    force_exclude_ids = set(force_exclude_ids or [])
    horizon_gws = sorted(projections["gameweek"].unique())[:horizon]
    target_gw = horizon_gws[0] if horizon_gws else None

    xpts_by_player = _multi_gw_xpts(projections, horizon)
    var_by_player = _multi_gw_var(projections, horizon)

    df = players.copy()
    df = df[df["status"].isin(["a", "d"])]
    df = df[df["start_probability"] >= OPTIMISER.min_start_probability] if "start_probability" in df.columns else df
    df = df[~df["id"].isin(force_exclude_ids)]

    df["xpts_total"] = df["id"].map(xpts_by_player).fillna(0.0)
    df["var_total"] = df["id"].map(var_by_player).fillna(0.0)

    lam, mu = lambda_mu_for_risk_mode(
        OPTIMISER.risk_mode, OPTIMISER.max_ownership_differential, OPTIMISER.variance_weight
    )
    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        df["eo_pct"] = df["id"].map(eo_map).fillna(0.0)
    else:
        df["eo_pct"] = 0.0
    df["effective_score"] = [
        risk_adjusted_score(x, v, e, lam, mu)
        for x, v, e in zip(df["xpts_total"], df["var_total"], df["eo_pct"], strict=True)
    ]

    if current_squad_ids:
        transfer_cost_per_extra = abs(TRANSFERS.hit_cost_points)
        in_squad = set(current_squad_ids)
    else:
        in_squad = set()
        transfer_cost_per_extra = 0

    player_ids = df["id"].tolist()
    idx = {pid: i for i, pid in enumerate(player_ids)}
    n = len(player_ids)

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

    selected = [pulp.LpVariable(f"sel_{i}", cat="Binary") for i in range(n)]
    starting = [pulp.LpVariable(f"sta_{i}", cat="Binary") for i in range(n)]
    captain = [pulp.LpVariable(f"cap_{i}", cat="Binary") for i in range(n)]
    vice = [pulp.LpVariable(f"vic_{i}", cat="Binary") for i in range(n)]

    scores = df["effective_score"].tolist()   # P3-3 risk-adjusted (objective);
    # true xpts_total for reporting is read straight off `df`/`starting_xi`
    costs = df["now_cost"].tolist()
    positions = df["position"].tolist()
    teams = df["team_id"].tolist()

    # 2026-07-30: a bench player used to contribute nothing to the objective
    # (only starting[i]/captain[i] did), so the solver had no reason to pick
    # anything but the cheapest feasible fodder once the starting XI was
    # set. `selected[i] - starting[i]` is 1 exactly when a player is on the
    # bench, so this adds a fractional (bench_value_weight) share of their
    # own score — real insurance value against an unpredicted blank in the
    # XI — without letting bench quality compete with the starting XI for
    # budget on equal terms.
    prob += pulp.lpSum(
        scores[i] * (starting[i] + captain[i])
        + OPTIMISER.bench_value_weight * scores[i] * (selected[i] - starting[i])
        for i in range(n)
    )

    prob += pulp.lpSum(selected) == SQUAD.squad_size
    prob += pulp.lpSum(costs[i] * selected[i] for i in range(n)) <= budget
    prob += pulp.lpSum(starting) == 11
    prob += pulp.lpSum(captain) == 1
    prob += pulp.lpSum(vice) == 1

    for pos in POSITIONS:
        pos_idx = [i for i, p in enumerate(positions) if p == pos]
        prob += pulp.lpSum(selected[i] for i in pos_idx) == SQUAD_COUNTS[pos]
        prob += pulp.lpSum(starting[i] for i in pos_idx) >= STARTING_MIN[pos]
        prob += pulp.lpSum(starting[i] for i in pos_idx) <= STARTING_MAX[pos]

    team_ids = list(set(teams))
    for tid in team_ids:
        team_idx = [i for i, t in enumerate(teams) if t == tid]
        prob += pulp.lpSum(selected[i] for i in team_idx) <= SQUAD.max_players_per_club

    for i in range(n):
        prob += starting[i] <= selected[i]
        prob += captain[i] <= starting[i]
        prob += vice[i] <= starting[i]
        prob += captain[i] + vice[i] <= 1

    for pid in force_include_ids:
        if pid in idx:
            prob += selected[idx[pid]] == 1

    if current_squad_ids and max_transfers is not None:
        new_player = [pulp.LpVariable(f"new_{i}", cat="Binary") for i in range(n)]
        for i, pid in enumerate(player_ids):
            if pid in in_squad:
                prob += new_player[i] == 0
            else:
                prob += new_player[i] == selected[i]
        prob += pulp.lpSum(new_player) <= max_transfers

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal" and current_squad_ids and max_transfers is not None:
        prob2 = pulp.LpProblem("fpl_squad_fallback", pulp.LpMaximize)
        selected2 = [pulp.LpVariable(f"sel2_{i}", cat="Binary") for i in range(n)]
        starting2 = [pulp.LpVariable(f"sta2_{i}", cat="Binary") for i in range(n)]
        captain2 = [pulp.LpVariable(f"cap2_{i}", cat="Binary") for i in range(n)]
        vice2 = [pulp.LpVariable(f"vic2_{i}", cat="Binary") for i in range(n)]
        prob2 += pulp.lpSum(
            scores[i] * (starting2[i] + captain2[i])
            + OPTIMISER.bench_value_weight * scores[i] * (selected2[i] - starting2[i])
            for i in range(n)
        )
        prob2 += pulp.lpSum(selected2) == SQUAD.squad_size
        prob2 += pulp.lpSum(costs[i] * selected2[i] for i in range(n)) <= budget
        prob2 += pulp.lpSum(starting2) == 11
        prob2 += pulp.lpSum(captain2) == 1
        prob2 += pulp.lpSum(vice2) == 1
        for pos in POSITIONS:
            pos_idx = [i for i, p in enumerate(positions) if p == pos]
            prob2 += pulp.lpSum(selected2[i] for i in pos_idx) == SQUAD_COUNTS[pos]
            prob2 += pulp.lpSum(starting2[i] for i in pos_idx) >= STARTING_MIN[pos]
            prob2 += pulp.lpSum(starting2[i] for i in pos_idx) <= STARTING_MAX[pos]
        for tid in list(set(teams)):
            team_idx = [i for i, t in enumerate(teams) if t == tid]
            prob2 += pulp.lpSum(selected2[i] for i in team_idx) <= SQUAD.max_players_per_club
        for i in range(n):
            prob2 += starting2[i] <= selected2[i]
            prob2 += captain2[i] <= starting2[i]
            prob2 += vice2[i] <= starting2[i]
            prob2 += captain2[i] + vice2[i] <= 1
        for pid in force_include_ids:
            if pid in idx:
                prob2 += selected2[idx[pid]] == 1
        prob2.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob2.status] == "Optimal":
            logger.warning("max_transfers=%d infeasible; falling back to unconstrained squad", max_transfers)
            selected = selected2
            starting = starting2
            captain = captain2
            vice = vice2
            prob = prob2

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"ILP solver did not find optimal solution: {pulp.LpStatus[prob.status]}")

    selected_ids = {player_ids[i] for i in range(n) if pulp.value(selected[i]) > 0.5}
    starting_ids = {player_ids[i] for i in range(n) if pulp.value(starting[i]) > 0.5}
    captain_id = next(player_ids[i] for i in range(n) if pulp.value(captain[i]) > 0.5)
    vice_id = next(player_ids[i] for i in range(n) if pulp.value(vice[i]) > 0.5)

    if season is not None and target_gw is not None:
        xpts_by_id = dict(zip(df["id"], df["xpts_total"], strict=True))
        var_by_id = dict(zip(df["id"], df["var_total"], strict=True))
        captain_id = scenario_based_captain(
            season, target_gw, list(starting_ids), xpts_by_id, var_by_id, mu
        )
        if captain_id == vice_id:
            remaining = [pid for pid in starting_ids if pid != captain_id]
            vice_id = max(remaining, key=lambda pid: xpts_by_id.get(pid, 0.0))

    squad_df = df[df["id"].isin(selected_ids)].copy()
    squad_df["is_starting"] = squad_df["id"].isin(starting_ids)
    squad_df["is_captain"] = squad_df["id"] == captain_id
    squad_df["is_vice_captain"] = squad_df["id"] == vice_id

    bench = squad_df[~squad_df["is_starting"]].copy()
    bench = bench.sort_values(
        ["position", "xpts_total"],
        key=lambda s: s.map({"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}) if s.name == "position" else -s,
        ascending=[True, True],
    )
    bench_order = {pid: i for i, pid in enumerate(bench["id"])}
    squad_df["bench_order"] = squad_df["id"].map(bench_order).fillna(-1).astype(int)

    starting_xi = squad_df[squad_df["is_starting"]].copy()

    if current_squad_ids:
        incoming = selected_ids - in_squad
        outgoing = in_squad - selected_ids
        transfers_made = len(incoming)
        hits = max(0, transfers_made - free_transfers)
    else:
        hits = 0

    # TRUE expected points (P3-3: the ILP's own objective value is now the
    # risk-adjusted `scores`, not real xpts — report the real figure,
    # computed straight from the starting XI + captain bonus).
    total_xpts = float(
        starting_xi["xpts_total"].sum()
        + starting_xi.loc[starting_xi["id"] == captain_id, "xpts_total"].sum()
    )
    total_cost = float(sum(
        df.loc[df["id"] == pid, "now_cost"].values[0]
        for pid in selected_ids
    ))

    logger.info(
        "Squad optimised: xPts=%.2f cost=£%.1fm hits=%d captain=%s",
        total_xpts, total_cost, hits,
        df.loc[df["id"] == captain_id, "web_name"].values[0],
    )

    return SquadSolution(
        squad=squad_df,
        starting_xi=starting_xi,
        captain_id=captain_id,
        vice_captain_id=vice_id,
        total_xpts=total_xpts,
        total_cost=total_cost,
        hits_taken=hits,
    )


def optimise_starting_xi(
    squad: pd.DataFrame,
    projections: pd.DataFrame,
    gw: int,
    ownership: pd.DataFrame | None = None,
    season: str | None = None,
) -> SquadSolution:
    """``ownership`` (P3-3, optional, default None): see ``optimise_squad``'s
    docstring — ``None`` (every call site today, including the P-XI backtest
    harness) makes EO a uniform 0% for every candidate, which is a constant
    rescale of the objective and changes NEITHER the captain pick NOR the
    starting XI versus the pre-P3-3 pure-argmax behaviour — the P-XI exit
    gate's already-reported numbers stay reproducible byte-for-byte.

    ``season`` (P3-4, optional, default None): enables scenario-based
    captaincy for this ``gw`` (see ``optimiser/captaincy.py``) — real joint
    MC samples over the additive own-variance approximation, where P3-1 has
    persisted them. At ``risk_mode="balanced"`` (mu=0, today's default) this
    is a no-op regardless of ``season`` — the P-XI gate stays byte-for-byte
    reproducible whether or not ``season`` is passed."""
    gw_proj = projections[projections["gameweek"] == gw][["player_id", "xpts"]].copy()
    if "xpts_var" in projections.columns:
        gw_var = projections[projections["gameweek"] == gw][["player_id", "xpts_var"]]
        gw_proj = gw_proj.merge(gw_var, on="player_id", how="left")
    df = squad.merge(gw_proj, left_on="id", right_on="player_id", how="left")
    df["xpts"] = df["xpts"].fillna(0.0)
    df["xpts_var"] = df["xpts_var"].fillna(0.0) if "xpts_var" in df.columns else 0.0

    lam, mu = lambda_mu_for_risk_mode(
        OPTIMISER.risk_mode, OPTIMISER.max_ownership_differential, OPTIMISER.variance_weight
    )
    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        df["eo_pct"] = df["id"].map(eo_map).fillna(0.0)
    else:
        df["eo_pct"] = 0.0
    df["effective_score"] = [
        risk_adjusted_score(x, v, e, lam, mu)
        for x, v, e in zip(df["xpts"], df["xpts_var"], df["eo_pct"], strict=True)
    ]

    player_ids = df["id"].tolist()
    n = len(player_ids)
    idx = {pid: i for i, pid in enumerate(player_ids)}
    positions = df["position"].tolist()
    scores = df["effective_score"].tolist()  # P3-3 risk-adjusted (objective);
    # true xpts for reporting is read straight off `df`/`starting_xi_df`

    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)

    starting = [pulp.LpVariable(f"sta_{i}", cat="Binary") for i in range(n)]
    captain = [pulp.LpVariable(f"cap_{i}", cat="Binary") for i in range(n)]
    vice = [pulp.LpVariable(f"vic_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(scores[i] * (starting[i] + captain[i]) for i in range(n))

    prob += pulp.lpSum(starting) == 11
    prob += pulp.lpSum(captain) == 1
    prob += pulp.lpSum(vice) == 1

    for pos in POSITIONS:
        pos_idx = [i for i, p in enumerate(positions) if p == pos]
        prob += pulp.lpSum(starting[i] for i in pos_idx) >= STARTING_MIN[pos]
        prob += pulp.lpSum(starting[i] for i in pos_idx) <= STARTING_MAX[pos]

    for i in range(n):
        prob += captain[i] <= starting[i]
        prob += vice[i] <= starting[i]
        prob += captain[i] + vice[i] <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Starting XI solver failed: {pulp.LpStatus[prob.status]}")

    starting_ids = {player_ids[i] for i in range(n) if pulp.value(starting[i]) > 0.5}
    captain_id = next(player_ids[i] for i in range(n) if pulp.value(captain[i]) > 0.5)
    vice_id = next(player_ids[i] for i in range(n) if pulp.value(vice[i]) > 0.5)

    if season is not None:
        xpts_by_id = dict(zip(df["id"], df["xpts"], strict=True))
        var_by_id = dict(zip(df["id"], df["xpts_var"], strict=True))
        captain_id = scenario_based_captain(
            season, gw, list(starting_ids), xpts_by_id, var_by_id, mu
        )
        if captain_id == vice_id:
            remaining = [pid for pid in starting_ids if pid != captain_id]
            vice_id = max(remaining, key=lambda pid: xpts_by_id.get(pid, 0.0))

    squad_out = df.copy()
    squad_out["is_starting"] = squad_out["id"].isin(starting_ids)
    squad_out["is_captain"] = squad_out["id"] == captain_id
    squad_out["is_vice_captain"] = squad_out["id"] == vice_id

    bench = squad_out[~squad_out["is_starting"]].copy()
    bench = bench.sort_values(
        ["position", "xpts"],
        key=lambda s: s.map({"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}) if s.name == "position" else -s,
        ascending=[True, True],
    )
    bench_order = {pid: i for i, pid in enumerate(bench["id"])}
    squad_out["bench_order"] = squad_out["id"].map(bench_order).fillna(-1).astype(int)

    starting_xi_df = squad_out[squad_out["is_starting"]]
    total_xpts = float(
        starting_xi_df["xpts"].sum()
        + starting_xi_df.loc[starting_xi_df["id"] == captain_id, "xpts"].sum()
    )

    return SquadSolution(
        squad=squad_out,
        starting_xi=starting_xi_df,
        captain_id=captain_id,
        vice_captain_id=vice_id,
        total_xpts=total_xpts,
        total_cost=float(squad["now_cost"].sum()),
        hits_taken=0,
    )
