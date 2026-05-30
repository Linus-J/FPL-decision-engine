#!/usr/bin/env python
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import pandas as pd
from sqlalchemy import text

from config.strategy import OPTIMISER, SQUAD, TRANSFERS
from data.db import get_session
from optimiser.chips import Chip, recommend_chip
from optimiser.squad import optimise_squad, optimise_starting_xi
from optimiser.transfers import evaluate_transfers
from projection.minutes_model import train as train_minutes, predict_batch as minutes_batch
from projection.points_model import train as train_points, predict_batch as points_batch


logger = logging.getLogger(__name__)


def _load_all_stats(season: str) -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT
                s.player_id, s.gameweek, s.season,
                s.minutes, s.total_points, s.goals_scored, s.assists,
                s.clean_sheets, s.goals_conceded, s.saves,
                s.yellow_cards, s.red_cards, s.bonus, s.bps,
                s.value AS now_cost,
                p.position, p.team_id,
                p.ict_index, p.influence, p.creativity, p.threat,
                p.form, p.selected_by_percent, p.status,
                p.chance_of_playing_next_round,
                COALESCE(x.xg, 0) AS xg, COALESCE(x.xa, 0) AS xa,
                COALESCE(x.xgi, 0) AS xgi, COALESCE(x.npxg, 0) AS npxg,
                COALESCE(x.shots, 0) AS shots, COALESCE(x.key_passes, 0) AS key_passes
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            LEFT JOIN player_xg_stats x
                ON x.player_id = s.player_id
                AND x.gameweek = s.gameweek
                AND x.season = s.season
            WHERE s.season = :season
            ORDER BY s.player_id, s.gameweek
        """)
        return pd.read_sql(query, db.bind, params={"season": season})
    finally:
        db.close()


def _load_players_snapshot(season: str, as_of_gw: int) -> pd.DataFrame:
    db = get_session()
    try:
        for gw in range(as_of_gw, as_of_gw - 5, -1):
            if gw < 1:
                break
            query = text("""
                SELECT DISTINCT
                    p.id, p.fpl_id, p.web_name, p.position, p.team_id,
                    s.value AS now_cost,
                    p.status, p.chance_of_playing_next_round,
                    p.selected_by_percent, p.form,
                    p.ict_index, p.influence, p.creativity, p.threat
                FROM players p
                JOIN player_gw_stats s ON s.player_id = p.id
                WHERE s.season = :season AND s.gameweek = :gw
            """)
            df = pd.read_sql(query, db.bind, params={"season": season, "gw": gw})
            if not df.empty:
                return df
        return pd.DataFrame()
    finally:
        db.close()


def _actual_gw_points(all_stats: pd.DataFrame, gw: int) -> dict[int, int]:
    subset = all_stats[all_stats["gameweek"] == gw]
    return dict(zip(subset["player_id"], subset["total_points"]))


def _build_gw_projections(
    history: pd.DataFrame,
    players: pd.DataFrame,
    minutes_model,
    points_model,
    target_gw: int,
    horizon: int,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    sp_by_player = minutes_batch(history, minutes_model)
    xp_by_player = points_batch(history, points_model)

    rows = []
    for _, player in players.iterrows():
        pid = int(player["id"])
        if player.get("status") in ("i", "u", "s"):
            sp, xp = 0.0, 0.0
        else:
            sp = float(sp_by_player.get(pid, 0.5))
            xp = float(xp_by_player.get(pid, 0.0))

        for gw_offset in range(horizon):
            rows.append({
                "player_id": pid,
                "gameweek": target_gw + gw_offset,
                "xpts": max(0.0, xp),
                "start_probability": sp,
            })

    return pd.DataFrame(rows)


def _score_squad(
    squad_ids: list[int],
    starting_ids: list[int],
    captain_id: int,
    actual_points: dict[int, int],
    bench_boost: bool = False,
    triple_captain: bool = False,
) -> int:
    captain_multiplier = 3 if triple_captain else 2
    total = 0
    for pid in starting_ids:
        pts = actual_points.get(pid, 0)
        if pid == captain_id:
            pts *= captain_multiplier
        total += pts
    if bench_boost:
        bench = [pid for pid in squad_ids if pid not in set(starting_ids)]
        total += sum(actual_points.get(pid, 0) for pid in bench)
    return total


def run_backtest(
    season: str = "2024-25",
    start_gw: int = 6,
    end_gw: int = 38,
    horizon: int | None = None,
    budget: float = SQUAD.budget_total,
) -> pd.DataFrame:
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
    all_stats = _load_all_stats(season)
    available_gws = sorted(all_stats["gameweek"].unique())

    results = []
    current_squad_ids: list[int] = []
    free_transfers = 15
    chips_used: set[Chip] = set()
    free_hit_active = False
    pre_free_hit_squad: list[int] = []
    squad_age_gws = 0

    for gw in available_gws:
        if gw < start_gw or gw > end_gw:
            continue

        history = all_stats[all_stats["gameweek"] < gw].copy()
        if history.empty:
            logger.info("GW%d: not enough history, skipping", gw)
            continue

        players = _load_players_snapshot(season, gw - 1)
        if players.empty:
            players = _load_players_snapshot(season, gw)
        if players.empty:
            logger.warning("GW%d: no player snapshot, skipping", gw)
            continue

        if len(history) < 50 or history["minutes"].nunique() < 2:
            logger.info("GW%d: insufficient training data (%d rows), skipping", gw, len(history))
            continue

        logger.info("GW%d: training models on %d rows...", gw, len(history))
        minutes_model = train_minutes(df_override=history, save=False, fast=True)
        points_model = train_points(df_override=history, save=False, fast=True)

        projections = _build_gw_projections(
            history=history,
            players=players,
            minutes_model=minutes_model,
            points_model=points_model,
            target_gw=gw,
            horizon=horizon,
        )

        if projections.empty:
            logger.warning("GW%d: no projections generated, skipping", gw)
            continue

        players = players.merge(
            projections[projections["gameweek"] == gw][["player_id", "start_probability"]],
            left_on="id", right_on="player_id", how="left",
        ).drop(columns=["player_id"], errors="ignore")
        players["start_probability"] = players["start_probability"].fillna(0.5)

        try:
            if free_hit_active:
                current_squad_ids = pre_free_hit_squad
                free_hit_active = False
                pre_free_hit_squad = []

            if not current_squad_ids:
                solution = optimise_squad(
                    projections=projections,
                    players=players,
                    budget=budget,
                    horizon=horizon,
                )
                new_squad_ids = solution.squad["id"].tolist()
                transfers_made = 0
                hits = 0
                squad_df = solution.squad
                chip_played: Chip | None = None
            else:
                in_snapshot = players[players["id"].isin(current_squad_ids)]
                missing = len(current_squad_ids) - len(in_snapshot)
                if missing > 2:
                    current_cost = SQUAD.budget_total
                else:
                    current_cost = in_snapshot["now_cost"].sum() + missing * 5.0

                bench_xpts_val = None
                try:
                    bench_ids = [pid for pid in current_squad_ids if pid not in
                                 optimise_starting_xi(players[players["id"].isin(current_squad_ids)].copy(), projections, gw).starting_xi["id"].tolist()]
                    bench_xpts_val = float(projections[
                        (projections["gameweek"] == gw) & projections["player_id"].isin(bench_ids)
                    ]["xpts"].sum())
                except Exception:
                    pass

                chip_rec = recommend_chip(
                    current_gw=gw,
                    current_squad_ids=current_squad_ids,
                    projections=projections,
                    players=players,
                    available_budget=current_cost,
                    free_transfers=free_transfers,
                    chips_used=chips_used,
                    bench_xpts=bench_xpts_val,
                    squad_age_gws=squad_age_gws,
                )
                chip_played = chip_rec.chip

                if chip_played == Chip.WILDCARD:
                    solution = optimise_squad(
                        projections=projections,
                        players=players,
                        budget=current_cost,
                        horizon=horizon,
                    )
                    new_squad_ids = solution.squad["id"].tolist()
                    transfers_made = len([p for p in new_squad_ids if p not in set(current_squad_ids)])
                    hits = 0
                    squad_df = solution.squad
                    free_transfers = 1
                    chips_used.add(Chip.WILDCARD)
                    logger.info("GW%d: WILDCARD played — gain=%.1f xPts", gw, chip_rec.expected_gain)

                elif chip_played == Chip.FREE_HIT:
                    fh_solution = optimise_squad(
                        projections=projections,
                        players=players,
                        budget=current_cost,
                        horizon=1,
                    )
                    pre_free_hit_squad = current_squad_ids[:]
                    new_squad_ids = fh_solution.squad["id"].tolist()
                    transfers_made = 0
                    hits = 0
                    squad_df = fh_solution.squad
                    free_hit_active = True
                    chips_used.add(Chip.FREE_HIT)
                    logger.info("GW%d: FREE HIT played — gain=%.1f xPts", gw, chip_rec.expected_gain)

                else:
                    if chip_played in (Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN):
                        chips_used.add(chip_played)
                        logger.info("GW%d: %s played — gain=%.1f xPts", gw, chip_played.value, chip_rec.expected_gain)

                    transfer_plan = evaluate_transfers(
                        current_squad_ids=current_squad_ids,
                        projections=projections,
                        players=players,
                        free_transfers=free_transfers,
                        available_budget=current_cost,
                    )
                    incoming = {t["player_id"] for t in transfer_plan.transfers_in}
                    outgoing = {t["player_id"] for t in transfer_plan.transfers_out}
                    new_squad_ids = [
                        pid for pid in current_squad_ids if pid not in outgoing
                    ] + list(incoming)
                    transfers_made = len(incoming)
                    hits = transfer_plan.hits_taken
                    squad_df = players[players["id"].isin(new_squad_ids)].copy()
                    expected_pos = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
                    actual_counts = squad_df["position"].value_counts().to_dict() if "position" in squad_df.columns else {}
                    if len(squad_df) != 15 or actual_counts != expected_pos:
                        logger.warning(
                            "GW%d: squad_df has %d players (pos=%s) — rebuilding from scratch",
                            gw, len(squad_df), actual_counts,
                        )
                        solution = optimise_squad(
                            projections=projections,
                            players=players,
                            budget=budget,
                            horizon=horizon,
                        )
                        new_squad_ids = solution.squad["id"].tolist()
                        transfers_made = 0
                        hits = 0
                        squad_df = solution.squad
        except Exception as e:
            logger.error("GW%d: optimiser failed — %s", gw, e)
            continue

        try:
            xi_solution = optimise_starting_xi(squad_df, projections, gw)
        except RuntimeError as e:
            pos_counts = squad_df["position"].value_counts().to_dict() if "position" in squad_df.columns else {}
            logger.error("GW%d: starting XI infeasible — squad size=%d pos=%s — %s", gw, len(squad_df), pos_counts, e)
            continue
        starting_ids = xi_solution.starting_xi["id"].tolist()
        captain_id = xi_solution.captain_id

        actual = _actual_gw_points(all_stats, gw)
        actual_pts = _score_squad(
            new_squad_ids, starting_ids, captain_id, actual,
            bench_boost=(chip_played == Chip.BENCH_BOOST),
            triple_captain=(chip_played == Chip.TRIPLE_CAPTAIN),
        )
        hit_penalty = hits * abs(TRANSFERS.hit_cost_points)
        net_pts = actual_pts - hit_penalty

        captain_name = squad_df.loc[squad_df["id"] == captain_id, "web_name"].values
        captain_name = captain_name[0] if len(captain_name) else "?"

        results.append({
            "gameweek": gw,
            "predicted_xpts": round(xi_solution.total_xpts, 2),
            "actual_pts": actual_pts,
            "hits": hits,
            "hit_penalty": hit_penalty,
            "net_pts": net_pts,
            "captain": captain_name,
            "squad_cost": round(squad_df["now_cost"].sum() if "now_cost" in squad_df.columns else 0, 1),
            "transfers_made": transfers_made,
            "chip_played": chip_played.value if chip_played else None,
            "free_transfers_start": free_transfers,
        })

        logger.info(
            "GW%d: predicted=%.1f actual=%d hits=%d net=%d captain=%s%s",
            gw, xi_solution.total_xpts, actual_pts, hits, net_pts, captain_name,
            f" [{chip_played.value}]" if chip_played else "",
        )

        current_squad_ids = new_squad_ids
        if chip_played == Chip.WILDCARD:
            free_transfers = 1
            squad_age_gws = 0
        elif free_hit_active:
            pass
        else:
            free_transfers = min(
                TRANSFERS.max_banked_free_transfers,
                max(1, free_transfers - transfers_made + TRANSFERS.free_transfers_per_gw),
            )
        if not free_hit_active:
            squad_age_gws += 1

    df = pd.DataFrame(results)
    if not df.empty:
        logger.info(
            "Backtest complete: GW%d–%d | avg actual=%.1f | total=%d | avg xPts=%.1f",
            df["gameweek"].min(), df["gameweek"].max(),
            df["actual_pts"].mean(),
            df["net_pts"].sum(),
            df["predicted_xpts"].mean(),
        )
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FPL backtest")
    p.add_argument("--season", default="2024-25")
    p.add_argument("--start-gw", type=int, default=6)
    p.add_argument("--end-gw", type=int, default=38)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="Save results CSV to this path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_backtest(
        season=args.season,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        horizon=args.horizon,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        logger.info("Results saved to %s", args.out)
    else:
        print(results.to_string())
