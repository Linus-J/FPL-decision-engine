import json
import logging
from dataclasses import asdict
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from config.settings import settings
from config.strategy import OPTIMISER, TRANSFERS
from data.db import get_session
from data.models import DecisionLog
from optimiser.chips import Chip, ChipRecommendation, chips_used_this_season, recommend_chip
from optimiser.squad import SquadSolution, optimise_squad, optimise_starting_xi
from optimiser.transfers import TransferPlan, evaluate_transfers, get_dgw_coverage, should_take_hit
from projection.pipeline import (
    _get_current_and_next_gw,
    _get_dgw_gameweeks,
    _get_bgw_gameweeks,
    get_latest_projections,
    run_projections,
)

logger = logging.getLogger(__name__)


def _load_players() -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT id, fpl_id, web_name, position, team_id, now_cost,
                   status, chance_of_playing_next_round, selected_by_percent,
                   form, ict_index, influence, creativity, threat
            FROM players
        """)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()


def _load_my_squad(team_id: int) -> tuple[list[int], float, int]:
    db = get_session()
    try:
        query = text("""
            SELECT dl.details
            FROM decision_log dl
            WHERE dl.decision_type = 'lineup'
            ORDER BY dl.created_at DESC
            LIMIT 1
        """)
        row = db.execute(query).fetchone()
        if row:
            details = json.loads(row[0])
            squad_ids = details.get("squad_ids", [])
            budget = details.get("budget", OPTIMISER.transfer_planning_horizon_gws)
            free_transfers = details.get("free_transfers", 1)
            return squad_ids, float(budget), int(free_transfers)
        return [], 100.0, 1
    finally:
        db.close()


def _load_decision_log() -> pd.DataFrame:
    db = get_session()
    try:
        query = text("SELECT * FROM decision_log ORDER BY created_at DESC")
        return pd.read_sql(query, db.bind)
    finally:
        db.close()


def _log_decision(
    gameweek: int,
    decision_type: str,
    details: dict,
    projected_gain: float = 0.0,
    dry_run: bool = True,
) -> None:
    db = get_session()
    try:
        entry = DecisionLog(
            gameweek=gameweek,
            decision_type=decision_type,
            details=json.dumps(details),
            projected_gain=projected_gain,
            dry_run=dry_run,
            created_at=datetime.utcnow(),
        )
        db.add(entry)
        db.commit()
    finally:
        db.close()


def _bench_xpts(squad_ids: list[int], projections: pd.DataFrame, gw: int) -> float:
    gw_proj = projections[
        (projections["gameweek"] == gw) & projections["player_id"].isin(squad_ids)
    ].sort_values("xpts", ascending=False)
    if len(gw_proj) <= 11:
        return 0.0
    return float(gw_proj.iloc[11:]["xpts"].sum())


def run(
    season: str = "2026-27",
    force_chip: Chip | None = None,
    dry_run: bool | None = None,
) -> dict:
    dry_run = settings.dry_run if dry_run is None else dry_run
    current_gw, next_gw = _get_current_and_next_gw()

    logger.info("Decision engine starting: current_gw=%d next_gw=%d dry_run=%s", current_gw, next_gw, dry_run)

    logger.info("Running projection pipeline...")
    run_projections(season=season, persist=True)
    projections = get_latest_projections()

    if projections.empty:
        logger.error("No projections available — aborting")
        return {"error": "no_projections"}

    players = _load_players()
    players = players.merge(
        projections[projections["gameweek"] == next_gw][["player_id", "start_probability"]],
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    squad_ids, available_budget, free_transfers = _load_my_squad(settings.fpl_team_id)
    if not squad_ids:
        logger.warning("No saved squad found — running full squad optimisation (season start / first run)")
        available_budget = 100.0
        free_transfers = 15

    dgw_gws = _get_dgw_gameweeks(OPTIMISER.transfer_planning_horizon_gws)
    bgw_gws = _get_bgw_gameweeks(OPTIMISER.transfer_planning_horizon_gws)
    bgw_affected = sum(
        1 for pid in squad_ids
        if any(
            projections[
                (projections["gameweek"] == gw) & (projections["player_id"] == pid)
            ]["xpts"].sum() == 0
            for gw in bgw_gws
        )
    ) if bgw_gws and squad_ids else 0

    decision_log = _load_decision_log()
    chips_used = chips_used_this_season(decision_log)

    bench_pts = _bench_xpts(squad_ids, projections, next_gw) if squad_ids else 0.0

    chip_rec: ChipRecommendation
    if force_chip:
        chip_rec = ChipRecommendation(force_chip, "forced by operator", 0.0)
    else:
        chip_rec = recommend_chip(
            current_gw=next_gw,
            current_squad_ids=squad_ids,
            projections=projections,
            players=players,
            available_budget=available_budget,
            free_transfers=free_transfers,
            chips_used=chips_used,
            bench_xpts=bench_pts,
            dgw_gws=dgw_gws,
            bgw_affected_count=bgw_affected,
        )

    wildcard_active = chip_rec.chip == Chip.WILDCARD
    free_hit_active = chip_rec.chip == Chip.FREE_HIT

    if free_hit_active:
        transfer_plan = TransferPlan(
            transfers_in=[], transfers_out=[], hits_taken=0, xpts_gain=0.0, net_xpts_gain=0.0
        )
        squad_solution = optimise_squad(
            projections=projections,
            players=players,
            budget=available_budget,
            horizon=1,
        )
    else:
        transfer_plan = evaluate_transfers(
            current_squad_ids=squad_ids,
            projections=projections,
            players=players,
            free_transfers=free_transfers,
            available_budget=available_budget,
            wildcard_active=wildcard_active,
            dgw_gws=dgw_gws,
        )

        new_squad_ids = (
            [t["player_id"] for t in transfer_plan.transfers_in]
            + [pid for pid in squad_ids if pid not in {t["player_id"] for t in transfer_plan.transfers_out}]
        ) if transfer_plan.transfers_in else squad_ids

        squad_df = players[players["id"].isin(new_squad_ids)].copy()
        squad_solution = optimise_starting_xi(squad_df, projections, next_gw)

    xi_solution = squad_solution

    dgw_coverage = get_dgw_coverage(
        squad_solution.squad["id"].tolist(), players, dgw_gws, projections
    )

    result = {
        "gameweek": next_gw,
        "dry_run": dry_run,
        "chip": chip_rec.chip.value if chip_rec.chip else None,
        "chip_reason": chip_rec.reason,
        "transfers_in": transfer_plan.transfers_in,
        "transfers_out": transfer_plan.transfers_out,
        "hits_taken": transfer_plan.hits_taken,
        "net_xpts_gain": round(transfer_plan.net_xpts_gain, 2),
        "squad": squad_solution.squad[["id", "web_name", "position", "now_cost", "is_starting", "is_captain", "is_vice_captain", "bench_order"]].to_dict("records"),
        "captain_id": xi_solution.captain_id,
        "vice_captain_id": xi_solution.vice_captain_id,
        "total_xpts": round(xi_solution.total_xpts, 2),
        "total_cost": round(squad_solution.total_cost, 1),
        "dgw_coverage": dgw_coverage,
    }

    _log_decision(
        gameweek=next_gw,
        decision_type="transfers",
        details={
            "transfers_in": transfer_plan.transfers_in,
            "transfers_out": transfer_plan.transfers_out,
            "hits_taken": transfer_plan.hits_taken,
        },
        projected_gain=transfer_plan.net_xpts_gain,
        dry_run=dry_run,
    )

    _log_decision(
        gameweek=next_gw,
        decision_type="lineup",
        details={
            "squad_ids": squad_solution.squad["id"].tolist(),
            "starting_ids": xi_solution.starting_xi["id"].tolist(),
            "captain_id": xi_solution.captain_id,
            "vice_captain_id": xi_solution.vice_captain_id,
            "budget": available_budget,
            "free_transfers": max(0, free_transfers - len(transfer_plan.transfers_in)),
        },
        projected_gain=xi_solution.total_xpts,
        dry_run=dry_run,
    )

    if chip_rec.chip:
        _log_decision(
            gameweek=next_gw,
            decision_type="chip",
            details={"chip": chip_rec.chip.value, "reason": chip_rec.reason},
            projected_gain=chip_rec.expected_gain,
            dry_run=dry_run,
        )

    logger.info(
        "Decision complete: chip=%s transfers=%d→%d hits=%d xPts=%.2f captain=%s",
        chip_rec.chip,
        len(transfer_plan.transfers_out),
        len(transfer_plan.transfers_in),
        transfer_plan.hits_taken,
        xi_solution.total_xpts,
        squad_solution.squad.loc[squad_solution.squad["id"] == xi_solution.captain_id, "web_name"].values[0]
        if xi_solution.captain_id else "?",
    )

    return result
