#!/usr/bin/env python
import argparse
import dataclasses
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

from config.strategy import OPTIMISER, SQUAD, TRANSFERS, OptimiserConfig
from data.db import get_session
from optimiser.chips import Chip, recommend_chip
from optimiser.squad import (
    STARTING_MAX,
    STARTING_MIN,
    optimise_squad,
    optimise_squad_joint,
    optimise_starting_xi,
)
from optimiser.transfers import evaluate_transfers, roll_forward_free_transfers
from projection import assemble, cold_start
from projection.minutes_model import train as train_minutes
from projection.rescore import load_bonus_2627_map, rescore_actuals, rescore_coverage_relevant

logger = logging.getLogger(__name__)

# Pinned to the pre-risk-aware-cold-start scoring (plan/risk-aware-cold-
# start-v1.md, 2026-07-31): OPTIMISER's global default gained a genuine
# nonzero mu_baseline so "medium" risk carries real variance-awareness
# live. This harness's whole purpose is to be a stable, comparable
# yardstick across changes (the 52.48 pts/GW gate) -- letting it silently
# pick up an untuned new mu_baseline would make that number incomparable
# for reasons unrelated to any real model change. Cold-start's improved
# MEAN estimates (real peer-bucket data replacing the old synthetic linear
# prior) still flow through unpinned -- that's a projection-accuracy fix,
# not a risk-preference one, and should affect the backtest too.
_BACKTEST_CONFIG = dataclasses.replace(OPTIMISER, risk_level=0.0, mu_baseline=0.0, mu_range=0.0)

_load_all_stats = assemble.load_all_stats  # moved to assemble.py (P3-0) — shared with pipeline.py


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


def _fixture_count_by_gw(all_stats: pd.DataFrame) -> dict[int, int]:
    """Real distinct-match count per gameweek (home-perspective rows only,
    so each match counts once), derived from already-loaded historical
    fixture context (``team_id_season``/``opponent_team_id``/``was_home``).

    Real bug found 2026-07-29: ``run_backtest`` never passed ``dgw_gws``/
    ``bgw_affected_count`` to ``recommend_chip`` at all (both silently
    defaulted to "no DGW, no BGW blanks", always) — Bench Boost's and Free
    Hit's entire logic is gated behind those being non-empty/nonzero, so
    neither could ever even be EVALUATED during backtesting, regardless of
    threshold calibration. The obvious fix (read ``Gameweek.is_dgw``/
    ``is_bgw``) doesn't work for historical seasons: those columns are only
    ever populated by the LIVE ``fpl_api.py::upsert_fixtures`` path for the
    current season — confirmed live, all 38 gameweeks of the 2025-26 season
    show ``is_dgw=0, is_bgw=0``. This derives it instead from data the
    backtest already has loaded, the same historical stat rows the P12 DGW
    fix (``projection/assemble.py``) already relies on for the same reason."""
    home = all_stats.loc[
        all_stats["was_home"].astype(bool),
        ["gameweek", "team_id_season", "opponent_team_id"],
    ].dropna(subset=["opponent_team_id"]).drop_duplicates()
    return home.groupby("gameweek").size().to_dict()


def _dgw_bgw_gws_in_window(
    fixture_counts: dict[int, int], start_gw: int, horizon: int
) -> tuple[set[int], set[int]]:
    """(dgw_gws, bgw_gws) within [start_gw, start_gw + horizon) — a normal
    round is 10 real matches; more means some teams play twice (DGW), fewer
    means some teams don't play at all (BGW). A gameweek missing from
    ``fixture_counts`` entirely (shouldn't happen for real historical data,
    but possible near either edge of the loaded window) is treated as
    normal rather than guessed."""
    window = range(start_gw, start_gw + horizon)
    dgw_gws = {gw for gw in window if fixture_counts.get(gw, 10) > 10}
    bgw_gws = {gw for gw in window if fixture_counts.get(gw, 10) < 10}
    return dgw_gws, bgw_gws


def _bgw_affected_count(
    squad_ids: list[int], bgw_gws: set[int], projections: pd.DataFrame
) -> int:
    """How many of the squad's own players have zero projected points in
    ANY blank gameweek in the horizon — mirrors
    ``agent/decision_engine.py``'s live computation of the same signal, so
    the backtest exercises the identical Free-Hit-eligibility path the live
    agent would use."""
    if not bgw_gws or not squad_ids:
        return 0
    return sum(
        1 for pid in squad_ids
        if any(
            projections[
                (projections["gameweek"] == gw) & (projections["player_id"] == pid)
            ]["xpts"].sum() == 0
            for gw in bgw_gws
        )
    )


def _actual_gw_points(all_stats: pd.DataFrame, gw: int, score_2627: bool = False) -> dict[int, int]:
    """Actual points for a GW. ``score_2627`` reads the ``total_points_2627``
    column (P-RS) — required for a like-for-like comparison against a
    26/27-scored prediction (the exit gate); default stays old-rules
    ``total_points`` for backward compatibility.

    Real bug found 2026-07-29 (P12 double-gameweek class, in the actual-
    scoring path this time): a genuine DGW player has TWO rows here (same
    gameweek, different opponent — see ``data/models.py::PlayerGameweekStats``).
    ``dict(zip(...))`` silently kept whichever row happened to come last,
    discarding the other fixture's real points entirely. Groups and sums
    instead, so a DGW player's actual score reflects both matches."""
    subset = all_stats[all_stats["gameweek"] == gw]
    use_2627 = score_2627 and "total_points_2627" in subset.columns
    col = "total_points_2627" if use_2627 else "total_points"
    return subset.groupby("player_id")[col].sum().to_dict()


def _actual_gw_minutes(all_stats: pd.DataFrame, gw: int) -> dict[int, int]:
    """Actual minutes played for a GW, summed per player (same DGW-summing
    rationale as ``_actual_gw_points`` — a double-gameweek player's minutes
    across both fixtures, used by ``_apply_autosubs`` to decide who blanked)."""
    subset = all_stats[all_stats["gameweek"] == gw]
    return subset.groupby("player_id")["minutes"].sum().to_dict()


def _build_gw_projections(
    history: pd.DataFrame,
    players: pd.DataFrame,
    minutes_model,
    target_gw: int,
    horizon: int,
    all_stats: pd.DataFrame,
    match_odds: pd.DataFrame,
    defcon_events: pd.DataFrame,
    defcon_field_shares: dict,
    n_scenarios: int = assemble.DEFAULT_N_SCENARIOS,
    seed: int = 42,
    season: str | None = None,
    sample_sink: list | None = None,
) -> pd.DataFrame:
    """P10: MC-assembled per-fixture projections (real odds-implied λ per
    horizon GW — superseding P0's ``fixture_multiplier`` heuristic, D3's
    original fix — via ``assemble.assemble_gw_projections``)."""
    if history.empty:
        return pd.DataFrame()

    # `strength_rel` (§2) is deliberately NOT passed here. It only feeds
    # fixtures the bookmakers never priced, and the backtest runs over
    # completed seasons whose `historical_fixture_odds` coverage is 380/380 —
    # so the strength path cannot fire, and passing it would change nothing
    # except to make a historical run depend on strengths published later.
    # The live gap it exists to close (odds thin out past the next week or two)
    # has no analogue in a finished season.
    proj = assemble.assemble_gw_projections(
        history, all_stats, minutes_model, target_gw, horizon,
        match_odds, defcon_events, defcon_field_shares,
        n_scenarios=n_scenarios, seed=seed,
        season=season, sample_sink=sample_sink,
    )
    if proj.empty:
        return proj

    # Freshness override: `players` is the LIVE snapshot as-of target_gw's
    # deadline; assemble's minutes bands already apply the availability
    # override from `history`'s OWN status column (as-of the last COMPLETED
    # gw), which can be stale by up to a week if status changed since — this
    # is the same belt-and-suspenders double-check the old code had.
    unavailable = {
        int(pid) for pid, status in zip(players["id"], players.get("status", []))
        if status in ("i", "u", "s")
    }
    if unavailable:
        mask = proj["player_id"].isin(unavailable)
        proj.loc[mask, ["xpts", "xpts_mean", "xpts_var", "start_probability"]] = 0.0

    # Optimiser's-curse correction (2026-07-28): shrink xpts toward the
    # (gameweek, position) group mean before anything downstream picks off
    # the raw values — after the unavailable-player zeroing above, so a
    # genuinely unavailable player (xpts=0) doesn't get shrunk back up
    # toward its group mean. See assemble.apply_curse_shrinkage.
    if OPTIMISER.curse_shrinkage_enabled:
        proj = assemble.apply_curse_shrinkage(proj, players)

    return proj


def _apply_autosubs(
    squad_ids: list[int],
    starting_ids: list[int],
    positions: dict[int, str],
    bench_order: dict[int, int],
    minutes: dict[int, int],
) -> list[int]:
    """Real bug found 2026-07-30 (user's own report review): the backtest
    never modelled FPL's auto-substitution rule, so a starting pick that
    ended up not playing simply scored 0 — understating every benchmark's
    realistic total versus what an actual manager holding that squad would
    get. Mirrors FPL's real algorithm: the bench GK comes on unconditionally
    if the starting GK gets 0 minutes (keeper-for-keeper, no formation
    constraint); each outfield bench player who DID play is then tried in
    ``bench_order`` priority against the blanking (0-minute) starters,
    swapped in only if the resulting XI still satisfies ``STARTING_MIN``/
    ``STARTING_MAX`` per position (same-position blanks are tried first,
    since that swap is always formation-neutral). A bench player who also
    has 0 minutes can never be subbed on, same as real FPL."""
    starting = list(starting_ids)
    bench = [pid for pid in squad_ids if pid not in set(starting_ids)]
    bench.sort(key=lambda pid: bench_order.get(pid, 99))

    starting_gks = [pid for pid in starting if positions.get(pid) == "GKP"]
    if starting_gks and minutes.get(starting_gks[0], 0) == 0:
        bench_gk = next((pid for pid in bench if positions.get(pid) == "GKP"), None)
        if bench_gk is not None and minutes.get(bench_gk, 0) > 0:
            starting = [bench_gk if pid == starting_gks[0] else pid for pid in starting]
            bench.remove(bench_gk)

    def _position_counts(ids: list[int]) -> dict[str, int]:
        counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for pid in ids:
            counts[positions.get(pid, "MID")] += 1
        return counts

    for sub_id in [pid for pid in bench if positions.get(pid) != "GKP"]:
        if minutes.get(sub_id, 0) <= 0:
            continue
        blanks = [
            pid for pid in starting
            if minutes.get(pid, 0) == 0 and positions.get(pid) != "GKP"
        ]
        if not blanks:
            continue
        sub_pos = positions.get(sub_id)
        blanks.sort(key=lambda pid: 0 if positions.get(pid) == sub_pos else 1)
        for blank_id in blanks:
            trial = [sub_id if pid == blank_id else pid for pid in starting]
            counts = _position_counts(trial)
            if all(STARTING_MIN[p] <= counts[p] <= STARTING_MAX[p] for p in STARTING_MIN):
                starting = trial
                break
    return starting


def _score_squad(
    squad_ids: list[int],
    starting_ids: list[int],
    captain_id: int,
    actual_points: dict[int, int],
    bench_boost: bool = False,
    triple_captain: bool = False,
    vice_captain_id: int | None = None,
    minutes: dict[int, int] | None = None,
    positions: dict[int, str] | None = None,
    bench_order: dict[int, int] | None = None,
) -> int:
    """``minutes``/``positions``/``bench_order`` are optional (all three
    required together) so existing call sites without them keep the old,
    no-autosub behaviour rather than silently changing under them."""
    effective_starting = starting_ids
    effective_captain = captain_id
    if minutes is not None and positions is not None and bench_order is not None:
        effective_starting = _apply_autosubs(
            squad_ids, starting_ids, positions, bench_order, minutes
        )
        if (
            minutes.get(captain_id, 1) == 0
            and vice_captain_id is not None
            and minutes.get(vice_captain_id, 0) > 0
        ):
            effective_captain = vice_captain_id

    captain_multiplier = 3 if triple_captain else 2
    total = 0
    for pid in effective_starting:
        pts = actual_points.get(pid, 0)
        if pid == effective_captain:
            pts *= captain_multiplier
        total += pts
    if bench_boost:
        bench = [pid for pid in squad_ids if pid not in set(effective_starting)]
        total += sum(actual_points.get(pid, 0) for pid in bench)
    return total


def _record_trace_gw(
    trace: list[dict],
    *,
    gw: int,
    xi_solution,
    actual: dict[int, int],
    new_squad_ids: list[int],
    prev_squad_ids: list[int],
    players: pd.DataFrame,
    chip_played,
    hits: int,
    predicted_xpts: float,
    actual_pts: int,
    net_pts: int,
    squad_cost,
) -> None:
    """One rich per-gameweek record for scripts/render_squad_trace.py. Reads
    ``xi_solution.squad`` (the full 15, already carrying position/cost/xpts/
    is_starting/is_captain/is_vice_captain/bench_order from optimise_starting_xi)
    rather than re-deriving any of that. Transfers are a plain id-set diff
    against ``prev_squad_ids`` (still the PRE-update squad at this call site,
    regardless of which code path built ``new_squad_ids`` — initial build,
    wildcard, free hit, or a normal transfer), so this doesn't need its own
    copy of that branching logic."""
    squad_full = xi_solution.squad.copy()
    squad_full["actual_pts"] = squad_full["id"].map(actual).fillna(0).astype(int)

    name_by_id = dict(zip(players["id"], players.get("web_name", players["id"]), strict=False))
    cost_by_id = dict(
        zip(players["id"], players.get("now_cost", [None] * len(players)), strict=False)
    )

    def _named(pid: int) -> dict:
        return {
            "id": int(pid),
            "web_name": name_by_id.get(pid, str(pid)),
            "cost": cost_by_id.get(pid),
        }

    prev_set = set(prev_squad_ids)
    new_set = set(new_squad_ids)

    trace.append({
        "gameweek": gw,
        "chip": chip_played.value if chip_played else None,
        "hits": hits,
        "transfers_in": [_named(pid) for pid in new_squad_ids if pid not in prev_set],
        "transfers_out": [_named(pid) for pid in prev_squad_ids if pid not in new_set],
        "squad": squad_full[[
            "id", "web_name", "position", "now_cost", "xpts",
            "is_starting", "is_captain", "is_vice_captain", "bench_order", "actual_pts",
        ]].to_dict("records"),
        "predicted_xpts": round(float(predicted_xpts), 2),
        "actual_pts": int(actual_pts),
        "net_pts": int(net_pts),
        "squad_cost": round(float(squad_cost), 1) if squad_cost is not None else None,
    })


def run_backtest(
    season: str = "2024-25",
    start_gw: int = 6,
    end_gw: int = 38,
    horizon: int | None = None,
    budget: float = SQUAD.budget_total,
    score_2627: bool = False,
    trace: list[dict] | None = None,
) -> pd.DataFrame:
    """``score_2627`` (P-RS): score the ACTUAL side under 26/27 rules (swap the
    as-played bonus for the recomputed_bonus.bonus_2627 sum — standard scoring
    and DefCon are unchanged 25/26->26/27) so the exit gate compares predicted
    and actual on one basis (finding C1). Player-GWs with no event coverage
    keep their as-played total (never invents a 26/27 bonus).

    ``trace`` (2026-07-28, human-readable squad-evolution audit): if given a
    list, one rich dict per gameweek gets appended to it (full 15-man squad
    with position/cost/xpts/actual points/starting-bench-captain flags,
    named transfers in/out, chip, predicted/actual/net) — for
    scripts/render_squad_trace.py. ``None`` (default) skips this entirely;
    existing callers/tests see zero behaviour change."""
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
    all_stats = _load_all_stats(season)
    available_gws = sorted(all_stats["gameweek"].unique())
    match_odds = assemble.load_match_odds(season)
    defcon_events = assemble.load_defcon_events(season)
    defcon_field_shares = assemble.compute_defcon_field_shares(season)
    fixture_counts = _fixture_count_by_gw(all_stats)

    if score_2627:
        db = get_session()
        try:
            bonus_map = load_bonus_2627_map(db, season)
        finally:
            db.close()
        all_stats = rescore_actuals(all_stats, bonus_map)
        coverage = rescore_coverage_relevant(all_stats, bonus_map)
        logger.info(
            "P-RS: 26/27 re-score coverage %.1f%% of bonus-earning player-GWs "
            "(%d bonus_2627 entries)",
            100 * coverage, len(bonus_map),
        )

    results = []
    current_squad_ids: list[int] = []
    free_transfers = 15
    chips_used: list[tuple[Chip, int]] = []
    free_hit_active = False
    pre_free_hit_squad: list[int] = []
    squad_age_gws = 0

    cold_start_prior: pd.DataFrame | None = None

    for gw in available_gws:
        if gw < start_gw or gw > end_gw:
            continue

        history = all_stats[all_stats["gameweek"] < gw].copy()

        # State as-of the target GW's deadline (snapshot for GW g is stamped
        # deadline(g) − ε, so it carries cumulative-through-(g-1) stats).
        players = _load_players_snapshot(season, gw)
        if players.empty:
            logger.warning("GW%d: no player snapshot, skipping", gw)
            continue

        use_cold_start = (
            history.empty
            or history["gameweek"].nunique() < 2  # lag features need a prior GW to shift from
            or len(history) < 50
            or history["minutes"].nunique() < 2
        )
        if use_cold_start:
            # 2026-07-30 (user's own request: "we need to have and test a
            # method to start from GW1... for the realtime 26/27 season
            # which is approaching"). Early gameweeks with no/thin current-
            # season history used to be skipped outright — a start_gw=1 run
            # silently lost every early gameweek's decision AND its actual
            # points, not just its model training. Falls back to the same
            # prior-season-carryover projection cold_start.py already uses
            # for the live GW1 squad build (T7), repeated flat across the
            # horizon window (no fixture-specific signal exists this early
            # either way) so the SAME decision pipeline below (initial
            # build / transfer evaluation / chip gating / scoring) runs
            # completely unchanged regardless of projection source — GW6+
            # behaviour (already gate-tested) is byte-identical, since
            # `use_cold_start` is always False once real history exists.
            if cold_start_prior is None:
                cold_start_prior = cold_start.load_prior_season_features(
                    cold_start.prior_season_of(season)
                )
            base_proj = cold_start.project_cold_start(players, cold_start_prior, target_gw=gw)
            projections = pd.concat(
                [base_proj.assign(gameweek=g) for g in range(gw, gw + horizon)],
                ignore_index=True,
            )
            unavailable = {
                int(pid) for pid, status in zip(players["id"], players.get("status", []))
                if status in ("i", "u", "s")
            }
            if unavailable:
                mask = projections["player_id"].isin(unavailable)
                projections.loc[mask, ["xpts", "start_probability"]] = 0.0
            logger.info(
                "GW%d: cold-start projections (%d current-season history rows)", gw, len(history)
            )
        else:
            logger.info("GW%d: training minutes model on %d rows...", gw, len(history))
            minutes_model = train_minutes(df_override=history, save=False, fast=True)

            projections = _build_gw_projections(
                history=history,
                players=players,
                minutes_model=minutes_model,
                target_gw=gw,
                horizon=horizon,
                all_stats=all_stats,
                match_odds=match_odds,
                defcon_events=defcon_events,
                defcon_field_shares=defcon_field_shares,
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
                    season=season,
                    config=_BACKTEST_CONFIG,
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
                                 optimise_starting_xi(
                                     players[players["id"].isin(current_squad_ids)].copy(),
                                     projections, gw, config=_BACKTEST_CONFIG,
                                 ).starting_xi["id"].tolist()]
                    bench_xpts_val = float(projections[
                        (projections["gameweek"] == gw) & projections["player_id"].isin(bench_ids)
                    ]["xpts"].sum())
                except Exception:
                    pass

                dgw_gws, bgw_gws = _dgw_bgw_gws_in_window(fixture_counts, gw, horizon)
                # P1.5: only THIS gameweek's blanks justify playing a Free Hit
                # this gameweek -- mirrors agent/decision_engine.py, which is
                # the behaviour the live agent uses.
                bgw_affected_count = _bgw_affected_count(
                    current_squad_ids, {g for g in bgw_gws if g == gw}, projections
                )
                chip_rec = recommend_chip(
                    current_gw=gw,
                    current_squad_ids=current_squad_ids,
                    projections=projections,
                    players=players,
                    available_budget=current_cost,
                    free_transfers=free_transfers,
                    chips_used=chips_used,
                    bench_xpts=bench_xpts_val,
                    dgw_gws=dgw_gws,
                    bgw_affected_count=bgw_affected_count,
                    squad_age_gws=squad_age_gws,
                    season=season,
                    config=_BACKTEST_CONFIG,
                )
                chip_played = chip_rec.chip

                if chip_played == Chip.WILDCARD:
                    solution = optimise_squad(
                        projections=projections,
                        players=players,
                        budget=current_cost,
                        horizon=horizon,
                        season=season,
                        config=_BACKTEST_CONFIG,
                    )
                    new_squad_ids = solution.squad["id"].tolist()
                    transfers_made = len(
                        [p for p in new_squad_ids if p not in set(current_squad_ids)]
                    )
                    hits = 0
                    squad_df = solution.squad
                    free_transfers = 1
                    chips_used.append((Chip.WILDCARD, gw))
                    logger.info(
                        "GW%d: WILDCARD played — gain=%.1f xPts",
                        gw, chip_rec.expected_gain,
                    )

                elif chip_played == Chip.FREE_HIT:
                    fh_solution = optimise_squad(
                        projections=projections,
                        players=players,
                        budget=current_cost,
                        horizon=1,
                        season=season,
                        config=_BACKTEST_CONFIG,
                    )
                    pre_free_hit_squad = current_squad_ids[:]
                    new_squad_ids = fh_solution.squad["id"].tolist()
                    transfers_made = 0
                    hits = 0
                    squad_df = fh_solution.squad
                    free_hit_active = True
                    chips_used.append((Chip.FREE_HIT, gw))
                    logger.info(
                        "GW%d: FREE HIT played — gain=%.1f xPts",
                        gw, chip_rec.expected_gain,
                    )

                else:
                    if chip_played in (Chip.BENCH_BOOST, Chip.TRIPLE_CAPTAIN):
                        chips_used.append((chip_played, gw))
                        logger.info(
                            "GW%d: %s played — gain=%.1f xPts",
                            gw, chip_played.value, chip_rec.expected_gain,
                        )

                    transfer_plan = evaluate_transfers(
                        current_squad_ids=current_squad_ids,
                        projections=projections,
                        players=players,
                        free_transfers=free_transfers,
                        available_budget=current_cost,
                        config=_BACKTEST_CONFIG,
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
                    actual_counts = (
                        squad_df["position"].value_counts().to_dict()
                        if "position" in squad_df.columns else {}
                    )
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
                            season=season,
                            config=_BACKTEST_CONFIG,
                        )
                        new_squad_ids = solution.squad["id"].tolist()
                        transfers_made = 0
                        hits = 0
                        squad_df = solution.squad
        except Exception as e:
            logger.error("GW%d: optimiser failed — %s", gw, e)
            continue

        try:
            xi_solution = optimise_starting_xi(
                squad_df, projections, gw, season=season, config=_BACKTEST_CONFIG
            )
        except RuntimeError as e:
            pos_counts = (
                squad_df["position"].value_counts().to_dict()
                if "position" in squad_df.columns else {}
            )
            logger.error(
                "GW%d: starting XI infeasible — squad size=%d pos=%s — %s",
                gw, len(squad_df), pos_counts, e,
            )
            continue
        starting_ids = xi_solution.starting_xi["id"].tolist()
        captain_id = xi_solution.captain_id

        # Own-variance-only team-total variance (P3-3-level approximation, no
        # teammate covariance) + the captain-doubling correction
        # (Var(2X)=4*Var(X) = Var(X) + 3*Var(X) already-counted-once) --
        # feeds the walk-forward gate's per-GW MC simulation
        # (scripts/walk_forward_gate.py). Not present pre-P10 projections.
        if "xpts_var" in xi_solution.starting_xi.columns:
            starting_var = xi_solution.starting_xi["xpts_var"].fillna(0.0)
            captain_var = float(
                xi_solution.starting_xi.loc[
                    xi_solution.starting_xi["id"] == captain_id, "xpts_var"
                ].sum()
            )
            predicted_var = float(starting_var.sum() + 3 * captain_var)
        else:
            predicted_var = 0.0

        actual = _actual_gw_points(all_stats, gw, score_2627=score_2627)
        actual_minutes = _actual_gw_minutes(all_stats, gw)
        positions = dict(zip(xi_solution.squad["id"], xi_solution.squad["position"], strict=False))
        bench_order_map = dict(
            zip(xi_solution.squad["id"], xi_solution.squad["bench_order"], strict=False)
        )
        actual_pts = _score_squad(
            new_squad_ids, starting_ids, captain_id, actual,
            bench_boost=(chip_played == Chip.BENCH_BOOST),
            triple_captain=(chip_played == Chip.TRIPLE_CAPTAIN),
            vice_captain_id=xi_solution.vice_captain_id,
            minutes=actual_minutes, positions=positions, bench_order=bench_order_map,
        )
        hit_penalty = hits * abs(TRANSFERS.hit_cost_points)
        net_pts = actual_pts - hit_penalty

        captain_name = squad_df.loc[squad_df["id"] == captain_id, "web_name"].values
        captain_name = captain_name[0] if len(captain_name) else "?"

        if trace is not None:
            has_cost_col = "now_cost" in squad_df.columns
            _record_trace_gw(
                trace, gw=gw, xi_solution=xi_solution, actual=actual,
                new_squad_ids=new_squad_ids, prev_squad_ids=current_squad_ids,
                players=players, chip_played=chip_played, hits=hits,
                predicted_xpts=xi_solution.total_xpts, actual_pts=actual_pts,
                net_pts=net_pts,
                squad_cost=squad_df["now_cost"].sum() if has_cost_col else None,
            )

        results.append({
            "gameweek": gw,
            "predicted_xpts": round(xi_solution.total_xpts, 2),
            "predicted_var": round(predicted_var, 4),
            "actual_pts": actual_pts,
            "hits": hits,
            "hit_penalty": hit_penalty,
            "net_pts": net_pts,
            "captain": captain_name,
            "squad_cost": round(
                squad_df["now_cost"].sum() if "now_cost" in squad_df.columns else 0, 1
            ),
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
        # P1.2 (2026-08-16): this roll-forward used to be inline here and
        # WRONG in agent/decision_engine.py, which is how the live agent
        # silently stopped transferring after its first transfer. Both now
        # call the same function. Behaviour change for the Free Hit branch,
        # which used to `pass` (keeping the count flat): a Free Hit reverts
        # the squad, so its transfers never spend the allowance, but the
        # weekly allowance still accrues.
        free_transfers = roll_forward_free_transfers(
            free_transfers,
            transfers_made,
            wildcard_played=chip_played == Chip.WILDCARD,
            free_hit_played=bool(free_hit_active),
        )
        if chip_played == Chip.WILDCARD:
            squad_age_gws = 0
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


def _merge_squad_dynamic(
    squad_static: pd.DataFrame, players: pd.DataFrame, squad_ids: list[int]
) -> pd.DataFrame:
    """A fixed squad's static identity (id/position/team_id/web_name) + this
    GW's dynamic price/start-prob from the player snapshot. Only the dynamic
    columns are merged in — merging the full snapshot would duplicate
    position/team_id/web_name as pandas `_x`/`_y` suffixes (both frames carry
    them), silently breaking any code that reads the bare column name."""
    dynamic_cols = ["id", "now_cost", "start_probability"]
    merged = squad_static.merge(
        players.loc[players["id"].isin(squad_ids), dynamic_cols], on="id", how="left",
    )
    merged["now_cost"] = merged["now_cost"].fillna(0.0)
    merged["start_probability"] = merged["start_probability"].fillna(0.5)
    return merged


def run_naive_xi_backtest(
    season: str = "2024-25",
    start_gw: int = 6,
    end_gw: int = 38,
    horizon: int | None = None,
    budget: float = SQUAD.budget_total,
    score_2627: bool = False,
) -> pd.DataFrame:
    """P-XI: the Phase-2 EXIT-GATE harness (finding M2). Precisely: a **fixed**
    initial 15 built ONCE at ``start_gw`` (via ``optimise_squad`` — the same
    mechanism the Phase-1 baseline used, so re-running this with
    ``score_2627=False`` reproduces 40.2 like-for-like), then each GW just
    re-optimises the legal starting XI + captain from that fixed squad
    (``optimise_starting_xi`` — ILP-optimal: captain = the highest-projected
    starter, since doubling any other starter's xPts cannot beat that).
    **No transfers, no chips, no hits** — this isolates the projection quality
    from the decision layer (Phase 3).

    Uses the same real FPL auto-substitution rule as ``run_backtest``'s
    ``_score_squad`` (2026-07-30 fix) — a blanking starter is replaced by the
    highest-priority bench player who played, subject to formation legality
    — so the two numbers stay comparable on a like-for-like scoring basis.
    DGW handling is whatever ``player_gw_stats``/projections already carry
    (P12 refines the DGW xPts multiplier upstream).

    ``score_2627=True`` scores the actual side under 26/27 rules (P-RS) — the
    exit-gate call. ``score_2627=False`` reproduces the Phase-1 baseline.
    """
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
    all_stats = _load_all_stats(season)
    available_gws = sorted(all_stats["gameweek"].unique())
    match_odds = assemble.load_match_odds(season)
    defcon_events = assemble.load_defcon_events(season)
    defcon_field_shares = assemble.compute_defcon_field_shares(season)

    if score_2627:
        db = get_session()
        try:
            bonus_map = load_bonus_2627_map(db, season)
        finally:
            db.close()
        all_stats = rescore_actuals(all_stats, bonus_map)
        logger.info(
            "P-XI: 26/27 re-score coverage %.1f%% of bonus-earning player-GWs",
            100 * rescore_coverage_relevant(all_stats, bonus_map),
        )

    squad_ids: list[int] = []
    squad_static: pd.DataFrame | None = None  # position/team_id — fixed once built
    results = []

    for gw in available_gws:
        if gw < start_gw or gw > end_gw:
            continue

        history = all_stats[all_stats["gameweek"] < gw].copy()
        if history.empty or len(history) < 50 or history["minutes"].nunique() < 2:
            logger.info("GW%d: insufficient history, skipping", gw)
            continue

        players = _load_players_snapshot(season, gw)
        if players.empty:
            logger.warning("GW%d: no player snapshot, skipping", gw)
            continue

        minutes_model = train_minutes(df_override=history, save=False, fast=True)
        projections = _build_gw_projections(
            history=history, players=players,
            minutes_model=minutes_model,
            target_gw=gw, horizon=horizon,
            all_stats=all_stats, match_odds=match_odds,
            defcon_events=defcon_events, defcon_field_shares=defcon_field_shares,
        )
        if projections.empty:
            logger.warning("GW%d: no projections, skipping", gw)
            continue

        players = players.merge(
            projections[projections["gameweek"] == gw][["player_id", "start_probability"]],
            left_on="id", right_on="player_id", how="left",
        ).drop(columns=["player_id"], errors="ignore")
        players["start_probability"] = players["start_probability"].fillna(0.5)

        if not squad_ids:
            try:
                solution = optimise_squad(projections=projections, players=players,
                                          budget=budget, horizon=horizon, season=season,
                                          config=_BACKTEST_CONFIG)
            except Exception as e:
                logger.error("GW%d: initial squad build failed — %s", gw, e)
                continue
            squad_ids = solution.squad["id"].tolist()
            squad_static = solution.squad[["id", "position", "team_id", "web_name"]].copy()
            logger.info("GW%d: fixed initial 15 built (£%.1fm)", gw, solution.total_cost)

        squad_df = _merge_squad_dynamic(squad_static, players, squad_ids)

        try:
            xi_solution = optimise_starting_xi(
                squad_df, projections, gw, season=season, config=_BACKTEST_CONFIG
            )
        except RuntimeError as e:
            logger.error("GW%d: starting XI infeasible — %s", gw, e)
            continue

        starting_ids = xi_solution.starting_xi["id"].tolist()
        actual = _actual_gw_points(all_stats, gw, score_2627=score_2627)
        actual_minutes = _actual_gw_minutes(all_stats, gw)
        positions = dict(zip(xi_solution.squad["id"], xi_solution.squad["position"], strict=False))
        bench_order_map = dict(
            zip(xi_solution.squad["id"], xi_solution.squad["bench_order"], strict=False)
        )
        actual_pts = _score_squad(
            squad_ids, starting_ids, xi_solution.captain_id, actual,
            vice_captain_id=xi_solution.vice_captain_id,
            minutes=actual_minutes, positions=positions, bench_order=bench_order_map,
        )

        captain_name = squad_df.loc[squad_df["id"] == xi_solution.captain_id, "web_name"]
        results.append({
            "gameweek": gw,
            "predicted_xpts": round(xi_solution.total_xpts, 2),
            "actual_pts": actual_pts,
            "captain": captain_name.values[0] if len(captain_name) else "?",
        })
        logger.info("GW%d: predicted=%.1f actual=%d captain=%s",
                    gw, xi_solution.total_xpts, actual_pts,
                    captain_name.values[0] if len(captain_name) else "?")

    df = pd.DataFrame(results)
    if not df.empty:
        logger.info(
            "Naive-XI backtest complete: GW%d–%d | avg actual=%.1f | avg xPts=%.1f",
            df["gameweek"].min(), df["gameweek"].max(),
            df["actual_pts"].mean(), df["predicted_xpts"].mean(),
        )
    return df


def run_rebuild_backtest(
    season: str = "2025-26",
    start_gw: int = 6,
    end_gw: int = 38,
    horizon: int | None = None,
    budget: float = SQUAD.budget_total,
    score_2627: bool = True,
    config: OptimiserConfig | None = None,
) -> pd.DataFrame:
    """The calibration instrument for covariance-aware selection (Objective v2).

    Every gameweek, a fresh 15 is built at ``budget`` from that gameweek's
    projections and scored on actual points. No transfers, no chips, no hits,
    no carry-over -- each gameweek is an INDEPENDENT squad-selection
    observation, which is exactly what a squad-level re-ranker changes.

    ``run_naive_xi_backtest`` cannot serve this purpose: it fixes the initial
    15 and re-optimises only the XI, so the re-ranker's candidate pool has
    exactly one member there and a mu sweep over it would measure nothing.

    Scoring goes through the same ``_score_squad`` the other two harnesses use,
    including FPL's real auto-substitution rule, so the numbers stay
    comparable.

    Also reports ``n_clubs_at_cap`` -- clubs filled to
    ``SQUAD.max_players_per_club`` -- because pricing that concentration is
    half of what the joint measure is meant to buy, and a mean-points column
    alone would not show it moving.
    """
    cfg = config or _BACKTEST_CONFIG
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
    all_stats = _load_all_stats(season)
    available_gws = sorted(all_stats["gameweek"].unique())
    match_odds = assemble.load_match_odds(season)
    defcon_events = assemble.load_defcon_events(season)
    defcon_field_shares = assemble.compute_defcon_field_shares(season)

    if score_2627:
        db = get_session()
        try:
            bonus_map = load_bonus_2627_map(db, season)
        finally:
            db.close()
        all_stats = rescore_actuals(all_stats, bonus_map)

    results = []
    for gw in available_gws:
        if gw < start_gw or gw > end_gw:
            continue

        history = all_stats[all_stats["gameweek"] < gw].copy()
        if history.empty or len(history) < 50 or history["minutes"].nunique() < 2:
            logger.info("GW%d: insufficient history, skipping", gw)
            continue

        players = _load_players_snapshot(season, gw)
        if players.empty:
            logger.warning("GW%d: no player snapshot, skipping", gw)
            continue

        minutes_model = train_minutes(df_override=history, save=False, fast=True)
        sample_rows: list = []
        projections = _build_gw_projections(
            history=history, players=players, minutes_model=minutes_model,
            target_gw=gw, horizon=horizon, all_stats=all_stats,
            match_odds=match_odds, defcon_events=defcon_events,
            defcon_field_shares=defcon_field_shares,
            season=season, sample_sink=sample_rows,
        )
        if projections.empty:
            logger.warning("GW%d: no projections, skipping", gw)
            continue

        players = players.merge(
            projections[projections["gameweek"] == gw][["player_id", "start_probability"]],
            left_on="id", right_on="player_id", how="left",
        ).drop(columns=["player_id"], errors="ignore")
        players["start_probability"] = players["start_probability"].fillna(0.5)

        try:
            solution = optimise_squad_joint(
                projections, players,
                season=season, gameweek=gw, sample_rows=sample_rows,
                budget=budget, horizon=horizon, config=cfg,
            )
        except (RuntimeError, ValueError) as e:
            logger.error("GW%d: squad build failed — %s", gw, e)
            continue

        try:
            xi_solution = optimise_starting_xi(
                solution.squad, projections, gw, season=season, config=cfg
            )
        except RuntimeError as e:
            logger.error("GW%d: starting XI infeasible — %s", gw, e)
            continue

        squad_ids = [int(i) for i in solution.squad["id"]]
        starting_ids = xi_solution.starting_xi["id"].tolist()
        actual = _actual_gw_points(all_stats, gw, score_2627=score_2627)
        actual_minutes = _actual_gw_minutes(all_stats, gw)
        positions = dict(
            zip(xi_solution.squad["id"], xi_solution.squad["position"], strict=False)
        )
        bench_order_map = dict(
            zip(xi_solution.squad["id"], xi_solution.squad["bench_order"], strict=False)
        )
        actual_pts = _score_squad(
            squad_ids, starting_ids, xi_solution.captain_id, actual,
            vice_captain_id=xi_solution.vice_captain_id,
            minutes=actual_minutes, positions=positions, bench_order=bench_order_map,
        )

        club_counts = solution.squad["team_id"].value_counts()
        results.append({
            "gameweek": gw,
            "actual_pts": actual_pts,
            "predicted_xpts": round(xi_solution.total_xpts, 2),
            "total_cost": solution.total_cost,
            "n_clubs_at_cap": int((club_counts >= SQUAD.max_players_per_club).sum()),
        })
        logger.info(
            "GW%d: rebuilt 15 (£%.1fm), scored %s actual pts, %d club(s) at cap",
            gw, solution.total_cost, actual_pts, results[-1]["n_clubs_at_cap"],
        )

    df = pd.DataFrame(results)
    if not df.empty:
        logger.info(
            "Rebuild backtest complete: GW%d–%d | avg actual=%.1f | avg clubs at cap=%.2f",
            df["gameweek"].min(), df["gameweek"].max(),
            df["actual_pts"].mean(), df["n_clubs_at_cap"].mean(),
        )
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FPL backtest")
    p.add_argument("--season", default="2024-25")
    p.add_argument("--start-gw", type=int, default=6)
    p.add_argument("--end-gw", type=int, default=38)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="Save results CSV to this path")
    p.add_argument(
        "--score-2627", action="store_true",
        help="Score actuals under 26/27 rules (P-RS): swap in recomputed_bonus.bonus_2627",
    )
    p.add_argument(
        "--naive-xi", action="store_true",
        help="P-XI exit-gate harness: fixed initial 15, no transfers/chips (run_naive_xi_backtest)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = run_naive_xi_backtest if args.naive_xi else run_backtest
    results = runner(
        season=args.season,
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        horizon=args.horizon,
        score_2627=args.score_2627,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        logger.info("Results saved to %s", args.out)
    else:
        print(results.to_string())
