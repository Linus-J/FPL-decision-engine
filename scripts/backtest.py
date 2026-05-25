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
from optimiser.squad import optimise_squad, optimise_starting_xi
from projection.minutes_model import train as train_minutes, predict_start_probabilities
from projection.points_model import train as train_points, predict_points, _build_features, FEATURE_COLS


logger = logging.getLogger(__name__)


def _load_all_stats(season: str) -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT
                s.player_id, s.gameweek, s.season,
                s.minutes, s.total_points, s.goals_scored, s.assists,
                s.clean_sheets, s.goals_conceded, s.saves,
                s.yellow_cards, s.red_cards, s.bonus, s.bps, s.value,
                p.position, p.team_id, p.now_cost,
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


def _actual_gw_points(season: str, gw: int) -> dict[int, int]:
    db = get_session()
    try:
        rows = db.execute(text(
            "SELECT player_id, total_points FROM player_gw_stats WHERE season=:s AND gameweek=:g"
        ), {"s": season, "g": gw}).fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        db.close()


def _build_gw_projections(
    history: pd.DataFrame,
    players: pd.DataFrame,
    minutes_model,
    points_model,
    target_gw: int,
    horizon: int,
) -> pd.DataFrame:
    rows = []
    for _, player in players.iterrows():
        pid = int(player["id"])
        pstats = history[history["player_id"] == pid].copy()

        if pstats.empty:
            continue

        try:
            sp_series = predict_start_probabilities(pstats, minutes_model)
            sp = float(sp_series.iloc[-1]) if not sp_series.empty else 0.5
            xp_series = predict_points(pstats, points_model)
            xp = float(xp_series.iloc[-1]) if not xp_series.empty else 0.0
        except Exception:
            sp, xp = 0.5, 0.0

        if player.get("status") in ("i", "u", "s"):
            sp, xp = 0.0, 0.0

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
) -> int:
    total = 0
    for pid in starting_ids:
        pts = actual_points.get(pid, 0)
        if pid == captain_id:
            pts *= 2
        total += pts
    bench = [pid for pid in squad_ids if pid not in set(starting_ids)]
    bench_pts = [actual_points.get(pid, 0) for pid in bench]
    bench_pts.sort(reverse=True)
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
            solution = optimise_squad(
                projections=projections,
                players=players,
                budget=budget,
                horizon=horizon,
                current_squad_ids=current_squad_ids if current_squad_ids else None,
                free_transfers=free_transfers,
            )
        except Exception as e:
            logger.error("GW%d: optimiser failed — %s", gw, e)
            continue

        new_squad_ids = solution.squad["id"].tolist()
        transfers_made = len([pid for pid in new_squad_ids if pid not in set(current_squad_ids)]) if current_squad_ids else 0
        hits = max(0, transfers_made - free_transfers) if current_squad_ids else 0

        xi_solution = optimise_starting_xi(solution.squad, projections, gw)
        starting_ids = xi_solution.starting_xi["id"].tolist()
        captain_id = xi_solution.captain_id

        actual = _actual_gw_points(season, gw)
        actual_pts = _score_squad(new_squad_ids, starting_ids, captain_id, actual)
        hit_penalty = hits * abs(TRANSFERS.hit_cost_points)
        net_pts = actual_pts - hit_penalty

        captain_name = solution.squad.loc[solution.squad["id"] == captain_id, "web_name"].values
        captain_name = captain_name[0] if len(captain_name) else "?"

        results.append({
            "gameweek": gw,
            "predicted_xpts": round(xi_solution.total_xpts, 2),
            "actual_pts": actual_pts,
            "hits": hits,
            "hit_penalty": hit_penalty,
            "net_pts": net_pts,
            "captain": captain_name,
            "squad_cost": round(solution.total_cost, 1),
            "transfers_made": transfers_made,
        })

        logger.info(
            "GW%d: predicted=%.1f actual=%d hits=%d net=%d captain=%s",
            gw, xi_solution.total_xpts, actual_pts, hits, net_pts, captain_name,
        )

        current_squad_ids = new_squad_ids
        free_transfers = min(TRANSFERS.max_banked_free_transfers, max(1, free_transfers - transfers_made + TRANSFERS.free_transfers_per_gw))

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
