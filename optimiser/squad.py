import logging
from dataclasses import dataclass

import pandas as pd
import pulp

from config.strategy import OPTIMISER, SQUAD, TRANSFERS

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


def optimise_squad(
    projections: pd.DataFrame,
    players: pd.DataFrame,
    budget: float = SQUAD.budget_total,
    horizon: int | None = None,
    current_squad_ids: list[int] | None = None,
    free_transfers: int = 1,
    force_include_ids: list[int] | None = None,
    force_exclude_ids: list[int] | None = None,
) -> SquadSolution:
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
    force_include_ids = set(force_include_ids or [])
    force_exclude_ids = set(force_exclude_ids or [])

    xpts_by_player = _multi_gw_xpts(projections, horizon)

    df = players.copy()
    df = df[df["status"].isin(["a", "d"])]
    df = df[df["start_probability"] >= OPTIMISER.min_start_probability] if "start_probability" in df.columns else df
    df = df[~df["id"].isin(force_exclude_ids)]

    df["xpts_total"] = df["id"].map(xpts_by_player).fillna(0.0)

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

    xpts = df["xpts_total"].tolist()
    costs = df["now_cost"].tolist()
    positions = df["position"].tolist()
    teams = df["team_id"].tolist()

    prob += pulp.lpSum(
        xpts[i] * (starting[i] + captain[i])
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

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"ILP solver did not find optimal solution: {pulp.LpStatus[prob.status]}")

    selected_ids = {player_ids[i] for i in range(n) if pulp.value(selected[i]) > 0.5}
    starting_ids = {player_ids[i] for i in range(n) if pulp.value(starting[i]) > 0.5}
    captain_id = next(player_ids[i] for i in range(n) if pulp.value(captain[i]) > 0.5)
    vice_id = next(player_ids[i] for i in range(n) if pulp.value(vice[i]) > 0.5)

    squad_df = df[df["id"].isin(selected_ids)].copy()
    squad_df["is_starting"] = squad_df["id"].isin(starting_ids)
    squad_df["is_captain"] = squad_df["id"] == captain_id
    squad_df["is_vice_captain"] = squad_df["id"] == vice_id

    starting_xi = squad_df[squad_df["is_starting"]].copy()

    if current_squad_ids:
        incoming = selected_ids - in_squad
        outgoing = in_squad - selected_ids
        transfers_made = len(incoming)
        hits = max(0, transfers_made - free_transfers)
    else:
        hits = 0

    total_xpts = float(pulp.value(prob.objective))
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


def optimise_starting_xi(squad: pd.DataFrame, projections: pd.DataFrame, gw: int) -> SquadSolution:
    gw_proj = projections[projections["gameweek"] == gw][["player_id", "xpts"]].copy()
    df = squad.merge(gw_proj, left_on="id", right_on="player_id", how="left")
    df["xpts"] = df["xpts"].fillna(0.0)

    player_ids = df["id"].tolist()
    n = len(player_ids)
    idx = {pid: i for i, pid in enumerate(player_ids)}
    positions = df["position"].tolist()
    xpts = df["xpts"].tolist()

    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)

    starting = [pulp.LpVariable(f"sta_{i}", cat="Binary") for i in range(n)]
    captain = [pulp.LpVariable(f"cap_{i}", cat="Binary") for i in range(n)]
    vice = [pulp.LpVariable(f"vic_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(xpts[i] * (starting[i] + captain[i]) for i in range(n))

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

    squad_out = squad.copy()
    squad_out["is_starting"] = squad_out["id"].isin(starting_ids)
    squad_out["is_captain"] = squad_out["id"] == captain_id
    squad_out["is_vice_captain"] = squad_out["id"] == vice_id

    return SquadSolution(
        squad=squad_out,
        starting_xi=squad_out[squad_out["is_starting"]],
        captain_id=captain_id,
        vice_captain_id=vice_id,
        total_xpts=float(pulp.value(prob.objective)),
        total_cost=float(squad["now_cost"].sum()),
        hits_taken=0,
    )
