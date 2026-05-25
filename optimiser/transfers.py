import logging
from dataclasses import dataclass

import pandas as pd

from config.strategy import DGW, OPTIMISER, TRANSFERS
from optimiser.squad import SquadSolution, optimise_squad

logger = logging.getLogger(__name__)


@dataclass
class TransferPlan:
    transfers_in: list[dict]
    transfers_out: list[dict]
    hits_taken: int
    xpts_gain: float
    net_xpts_gain: float


def _current_squad_df(current_squad_ids: list[int], players: pd.DataFrame) -> pd.DataFrame:
    return players[players["id"].isin(current_squad_ids)].copy()


def _squad_xpts(squad_ids: list[int], projections: pd.DataFrame, horizon: int) -> float:
    gws = sorted(projections["gameweek"].unique())[:horizon]
    subset = projections[
        projections["gameweek"].isin(gws) & projections["player_id"].isin(squad_ids)
    ]
    return float(subset["xpts"].sum())


def _xpts_with_transfers(
    solution: SquadSolution,
    projections: pd.DataFrame,
    horizon: int,
    hits: int,
) -> float:
    squad_ids = solution.squad["id"].tolist()
    raw = _squad_xpts(squad_ids, projections, horizon)
    return raw + hits * TRANSFERS.hit_cost_points


def evaluate_transfers(
    current_squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    free_transfers: int = 1,
    available_budget: float | None = None,
    wildcard_active: bool = False,
    dgw_gws: set[int] | None = None,
) -> TransferPlan:
    dgw_gws = dgw_gws or set()
    horizon = OPTIMISER.transfer_planning_horizon_gws

    current_squad = _current_squad_df(current_squad_ids, players)
    budget = available_budget or current_squad["now_cost"].sum()

    current_xpts = _squad_xpts(current_squad_ids, projections, horizon)

    best_plan: TransferPlan | None = None
    max_hits = TRANSFERS.max_hits_per_gw if not wildcard_active else 15

    if wildcard_active:
        hit_range = [0]
    else:
        hit_range = range(0, max_hits + 1)

    for extra_hits in hit_range:
        effective_ft = free_transfers + extra_hits * 0 if wildcard_active else free_transfers
        transfers_allowed = effective_ft + extra_hits if not wildcard_active else 15

        solution = optimise_squad(
            projections=projections,
            players=players,
            budget=budget,
            horizon=horizon,
            current_squad_ids=current_squad_ids,
            free_transfers=effective_ft,
        )

        new_squad_ids = solution.squad["id"].tolist()
        incoming = [pid for pid in new_squad_ids if pid not in set(current_squad_ids)]
        outgoing = [pid for pid in current_squad_ids if pid not in set(new_squad_ids)]

        actual_hits = max(0, len(incoming) - free_transfers)
        if actual_hits > max_hits and not wildcard_active:
            continue

        net_gain = _xpts_with_transfers(solution, projections, horizon, actual_hits) - current_xpts

        if best_plan is None or net_gain > best_plan.net_xpts_gain:
            best_plan = TransferPlan(
                transfers_in=[
                    {
                        "player_id": pid,
                        "web_name": players.loc[players["id"] == pid, "web_name"].values[0]
                        if pid in players["id"].values else str(pid),
                        "cost": players.loc[players["id"] == pid, "now_cost"].values[0]
                        if pid in players["id"].values else 0.0,
                    }
                    for pid in incoming
                ],
                transfers_out=[
                    {
                        "player_id": pid,
                        "web_name": players.loc[players["id"] == pid, "web_name"].values[0]
                        if pid in players["id"].values else str(pid),
                        "cost": players.loc[players["id"] == pid, "now_cost"].values[0]
                        if pid in players["id"].values else 0.0,
                    }
                    for pid in outgoing
                ],
                hits_taken=actual_hits,
                xpts_gain=solution.total_xpts - current_xpts,
                net_xpts_gain=net_gain,
            )

    if best_plan is None:
        best_plan = TransferPlan(
            transfers_in=[],
            transfers_out=[],
            hits_taken=0,
            xpts_gain=0.0,
            net_xpts_gain=0.0,
        )

    logger.info(
        "Transfer plan: %d in / %d out, %d hits, net gain %.2f xPts",
        len(best_plan.transfers_in),
        len(best_plan.transfers_out),
        best_plan.hits_taken,
        best_plan.net_xpts_gain,
    )
    return best_plan


def should_take_hit(plan: TransferPlan) -> bool:
    if plan.hits_taken == 0:
        return False
    breakeven = abs(TRANSFERS.hit_cost_points) * plan.hits_taken
    return plan.xpts_gain >= breakeven


def get_dgw_coverage(squad_ids: list[int], players: pd.DataFrame, dgw_gws: set[int], projections: pd.DataFrame) -> int:
    if not dgw_gws:
        return 0
    dgw_proj = projections[projections["gameweek"].isin(dgw_gws)]
    dgw_players = dgw_proj[dgw_proj["player_id"].isin(squad_ids)]
    eligible = dgw_players[dgw_players["xpts"] > 0]["player_id"].nunique()
    return int(eligible)
