#!/usr/bin/env python
import argparse
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
from projection.fixture_adjust import fixture_multiplier
from projection.minutes_model import predict_batch as minutes_batch
from projection.minutes_model import train as train_minutes
from projection.points_model import predict_batch as points_batch
from projection.points_model import train as train_points

logger = logging.getLogger(__name__)


def _load_all_stats(season: str) -> pd.DataFrame:
    db = get_session()
    try:
        # Dynamic player attributes (ict/influence/creativity/threat/form/
        # selected_by_percent/status/chance) come from the point-in-time
        # snapshot as-of the gameweek deadline — NOT the mutable players.*
        # columns, which would leak the latest value into historical training
        # rows (Phase-1 leaks L1/L2). player_gw_stats/xg_stats stay as-is
        # (already point-in-time). Static columns (position, team_id) come from
        # players.
        query = text("""
            SELECT
                s.player_id, s.gameweek, s.season,
                s.minutes, s.total_points, s.goals_scored, s.assists,
                s.clean_sheets, s.goals_conceded, s.saves,
                s.yellow_cards, s.red_cards, s.bonus, s.bps,
                s.value AS now_cost,
                s.team_id_season, s.opponent_team_id, s.was_home,
                p.position, p.team_id,
                COALESCE(ps.ict_index, 0) AS ict_index,
                COALESCE(ps.influence, 0) AS influence,
                COALESCE(ps.creativity, 0) AS creativity,
                COALESCE(ps.threat, 0) AS threat,
                COALESCE(ps.form, 0) AS form,
                COALESCE(ps.selected_by_percent, 0) AS selected_by_percent,
                COALESCE(ps.status, 'a') AS status,
                ps.chance_of_playing_next_round AS chance_of_playing_next_round,
                COALESCE(x.xg, 0) AS xg, COALESCE(x.xa, 0) AS xa,
                COALESCE(x.xgi, 0) AS xgi, COALESCE(x.npxg, 0) AS npxg,
                COALESCE(x.shots, 0) AS shots, COALESCE(x.key_passes, 0) AS key_passes
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            JOIN gameweeks g ON g.id = s.gameweek AND g.season = s.season
            LEFT JOIN player_state_snapshots ps ON ps.id = (
                SELECT ps2.id FROM player_state_snapshots ps2
                WHERE ps2.player_id = s.player_id
                    AND ps2.season = s.season
                    AND ps2.snapshot_ts < g.deadline_time
                ORDER BY ps2.snapshot_ts DESC LIMIT 1
            )
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


def _load_players_snapshot(season: str, target_gw: int) -> pd.DataFrame:
    """Player state available when deciding for `target_gw`: the latest snapshot
    per player with snapshot_ts < deadline(season, target_gw). All dynamic
    columns (cost/status/ownership/form/ICT) are as-of — no leak of the current
    players.* row (Phase-1 leak L1)."""
    db = get_session()
    try:
        query = text("""
            SELECT
                p.id, p.fpl_id, p.web_name, p.position, p.team_id,
                ps.now_cost,
                ps.status, ps.chance_of_playing_next_round,
                ps.selected_by_percent, ps.form,
                ps.ict_index, ps.influence, ps.creativity, ps.threat
            FROM players p
            JOIN gameweeks g ON g.id = :gw AND g.season = :season
            JOIN player_state_snapshots ps ON ps.id = (
                SELECT ps2.id FROM player_state_snapshots ps2
                WHERE ps2.player_id = p.id
                    AND ps2.season = :season
                    AND ps2.snapshot_ts < g.deadline_time
                ORDER BY ps2.snapshot_ts DESC LIMIT 1
            )
        """)
        # int(): available_gws come from numpy (int64); SQLite won't match a
        # numpy int against gameweeks.id, silently returning no rows.
        return pd.read_sql(query, db.bind, params={"season": season, "gw": int(target_gw)})
    finally:
        db.close()


def _actual_gw_points(all_stats: pd.DataFrame, gw: int) -> dict[int, int]:
    subset = all_stats[all_stats["gameweek"] == gw]
    return dict(zip(subset["player_id"], subset["total_points"]))


def _opponent_context(
    season: str, all_stats: pd.DataFrame
) -> dict[tuple[int, int], tuple[float | None, bool | None]]:
    """Per (player_id, gameweek) → (opponent defence strength, was_home) for the
    season, from the point-in-time fixture context on the stat row (T3b) joined
    to TeamSeasonStrength. Feeds the per-GW fixture multiplier. The opponent
    defends away when the player is home, and vice-versa."""
    db = get_session()
    try:
        rows = db.execute(
            text("""SELECT team_id, strength_defence_home, strength_defence_away
                    FROM team_season_strength WHERE season = :s"""),
            {"s": season},
        ).fetchall()
    finally:
        db.close()
    defence = {int(r[0]): (r[1], r[2]) for r in rows}

    ctx: dict[tuple[int, int], tuple[float | None, bool | None]] = {}
    for r in all_stats.itertuples():
        opp = getattr(r, "opponent_team_id", None)
        if opp is None or pd.isna(opp):
            continue
        was_home = getattr(r, "was_home", None)
        was_home = None if (was_home is None or pd.isna(was_home)) else bool(was_home)
        def_home, def_away = defence.get(int(opp), (None, None))
        opp_def = def_away if was_home else def_home
        ctx[(int(r.player_id), int(r.gameweek))] = (opp_def, was_home)
    return ctx


def _build_gw_projections(
    history: pd.DataFrame,
    players: pd.DataFrame,
    minutes_model,
    points_model,
    target_gw: int,
    horizon: int,
    opp_ctx: dict[tuple[int, int], tuple[float | None, bool | None]] | None = None,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    opp_ctx = opp_ctx or {}
    sp_by_player = minutes_batch(history, minutes_model)
    xp_by_player = points_batch(history, points_model)

    rows = []
    for _, player in players.iterrows():
        pid = int(player["id"])
        if player.get("status") in ("i", "u", "s"):
            sp, base_xp = 0.0, 0.0
        else:
            sp = float(sp_by_player.get(pid, 0.5))
            base_xp = float(xp_by_player.get(pid, 0.0))

        # Per-GW fixture conditioning (fixes D3): each horizon GW is scaled by
        # its own opponent, so projections differ by fixture instead of the base
        # xp being broadcast flat across the horizon.
        for gw_offset in range(horizon):
            gw = target_gw + gw_offset
            opp_def, was_home = opp_ctx.get((pid, gw), (None, None))
            mult = fixture_multiplier(opp_def, was_home)
            xp_gw = max(0.0, base_xp * mult)
            rows.append({
                "player_id": pid,
                "gameweek": gw,
                "xpts": xp_gw,
                "xpts_mean": xp_gw,
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
    opp_ctx = _opponent_context(season, all_stats)

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

        # State as-of the target GW's deadline (snapshot for GW g is stamped
        # deadline(g) − ε, so it carries cumulative-through-(g-1) stats).
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
            opp_ctx=opp_ctx,
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
