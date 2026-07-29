#!/usr/bin/env python
"""walk_forward_gate.py — Phase-3 walk-forward gate.

(v2-build-plan.md §5 / plan/phase-3-decision-layer.md "Gate": walk-forward vs
benchmarks — avg manager ~55-57, frozen template, v1, top-10k ~63 — reporting
a **simulated final-rank distribution**, not mean points, since a
risk-seeking bot can have a slightly lower mean with a fatter right tail and
still be the better rank-optimising choice.)

**Benchmarks, and how each is actually computed here** (no paid data, no
scraped percentile table exists — see "Population model" below):

- **v2 bot** — this project's full Phase-3 decision engine
  (``scripts.backtest.run_backtest``): transfers, chips (P3-5 scenario-EV
  gating), scenario-based captaincy (P3-4), walk-forward, 26/27-scored.
- **v1 bot** — this project's naive-XI harness
  (``scripts.backtest.run_naive_xi_backtest``, the Phase-2 exit-gate
  harness): a squad fixed at ``start_gw``, re-optimising ONLY the legal
  starting XI + captain each week (no transfers, no chips, no hits). This is
  "v1" in the sense of predating Phase 3's decision layer, not a revival of
  the separate, incompatible pre-rebuild codebase on ``master`` (a
  3-commit, ~11k-line-smaller snapshot from before this v2 rebuild — running
  it against the current DB schema would be its own project, not a
  reasonable scope here).
- **Frozen template** — ``run_frozen_template`` (this module): a squad AND
  its starting XI AND its captain, all picked ONCE at ``start_gw`` and never
  touched again — a stricter, more passive baseline than v1 (v1 still
  re-optimises lineup/captain weekly for rotation/injury; this doesn't even
  do that). Stand-in for "picked a squad on day one and never logged back
  in all season."
- **Avg manager / top-10k pace** — NOT computed; they're the plan's own
  approximate reference constants (``AVG_MANAGER_PTS_PER_GW``,
  ``TOP_10K_PTS_PER_GW``), used only as the two calibration anchors for the
  population model below. No free source of the real FPL manager-population
  score distribution exists (this project has consistently used only free
  data sources — see project memory); a wrong number here would be
  indistinguishable from a right one, so treat both as labelled
  approximations, not measurements.

**Population model (for turning a points total into a rank):** a Normal
distribution over season-total points, fit through exactly the two anchor
points above: mean = avg-manager pace, and the point ``(1 - 10,000 /
ASSUMED_POPULATION_SIZE)`` quantile = top-10k pace. ``ASSUMED_POPULATION_SIZE``
(9,000,000) is a documented round-number assumption (FPL's real
season-to-season entrant count is public but not pinned down here), not a
measurement — the resulting rank numbers are directional, not exact.

**Rank SIMULATION for the v2 bot** (the actual point of this module, per the
plan's "report the distribution, not the mean" framing): ``run_backtest``
now also returns ``predicted_var`` per GW — the own-variance-only team-total
variance (P3-3-level approximation, no teammate covariance) plus the
captain-doubling correction (``Var(2X) = 4*Var(X)``). Monte-Carlo redrawing
each GW's score from ``Normal(predicted_xpts - hit_penalty, predicted_var)``
and summing across GWs (independence across GWs assumed — no cross-GW
correlation modelled) answers "if we replayed this exact season with the
SAME weekly decisions but different random luck, what range of season
totals / ranks might we have gotten" — capturing the bot's own predictive
uncertainty, not uncertainty about which decisions to make. v1 and the
frozen template only get a single realised point estimate each (they carry
no persisted per-GW variance in the same form), so they're plotted as
reference points against the v2 bot's distribution, not distributions of
their own.

**Known limitation, not silently absorbed:** ``predicted_xpts``/
``predicted_var`` on a chip-played GW (bench boost, triple captain) don't
reflect the chip's actual scoring rule — ``optimise_starting_xi`` has no
notion of chips. Chips are rare (at most 4-5 GWs of ~33), so this is a
second-order effect on the season-total distribution, not corrected here.
"""

from __future__ import annotations

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

import numpy as np
import pandas as pd
from scipy import stats

from config.strategy import OPTIMISER, SQUAD
from data.db import get_session
from optimiser.squad import optimise_squad, optimise_starting_xi
from projection import assemble
from projection.minutes_model import train as train_minutes
from projection.rescore import load_bonus_2627_map, rescore_actuals
from scripts.backtest import (
    _actual_gw_minutes,
    _actual_gw_points,
    _build_gw_projections,
    _load_all_stats,
    _load_players_snapshot,
    _score_squad,
    run_backtest,
    run_naive_xi_backtest,
)

logger = logging.getLogger(__name__)

# Plan's own approximate benchmark constants (phase-3-decision-layer.md
# "Gate" / v2-build-plan.md §5) -- calibration anchors, not measurements.
AVG_MANAGER_PTS_PER_GW = 56.0   # midpoint of the plan's "~55-57"
TOP_10K_PTS_PER_GW = 63.0       # the plan's "~63"
ASSUMED_POPULATION_SIZE = 9_000_000  # documented assumption, not scraped
N_MC_TRIALS = 20_000


def run_frozen_template(
    season: str,
    start_gw: int,
    end_gw: int,
    horizon: int | None = None,
    budget: float = SQUAD.budget_total,
    score_2627: bool = True,
) -> pd.DataFrame:
    """Squad + starting XI + captain, all picked ONCE at ``start_gw``, never
    touched again. See module docstring for why this is a stricter baseline
    than ``run_naive_xi_backtest`` (v1), which still re-optimises lineup and
    captain weekly."""
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
    all_stats = _load_all_stats(season)

    if score_2627:
        db = get_session()
        try:
            bonus_map = load_bonus_2627_map(db, season)
        finally:
            db.close()
        all_stats = rescore_actuals(all_stats, bonus_map)

    match_odds = assemble.load_match_odds(season)
    defcon_events = assemble.load_defcon_events(season)
    defcon_field_shares = assemble.compute_defcon_field_shares(season)

    history = all_stats[all_stats["gameweek"] < start_gw].copy()
    players = _load_players_snapshot(season, start_gw)
    minutes_model = train_minutes(df_override=history, save=False, fast=True)
    projections = _build_gw_projections(
        history=history, players=players, minutes_model=minutes_model,
        target_gw=start_gw, horizon=horizon, all_stats=all_stats,
        match_odds=match_odds, defcon_events=defcon_events,
        defcon_field_shares=defcon_field_shares,
    )
    solution = optimise_squad(
        projections=projections, players=players, budget=budget,
        horizon=horizon, season=season,
    )
    xi_solution = optimise_starting_xi(solution.squad, projections, start_gw, season=season)
    squad_ids = solution.squad["id"].tolist()
    starting_ids = xi_solution.starting_xi["id"].tolist()
    captain_id = xi_solution.captain_id
    vice_captain_id = xi_solution.vice_captain_id
    positions = dict(zip(xi_solution.squad["id"], xi_solution.squad["position"], strict=False))
    bench_order_map = dict(
        zip(xi_solution.squad["id"], xi_solution.squad["bench_order"], strict=False)
    )
    logger.info(
        "Frozen template built at GW%d: £%.1fm, captain=%s (never revisited)",
        start_gw, solution.total_cost,
        solution.squad.loc[solution.squad["id"] == captain_id, "web_name"].values[0]
        if captain_id else "?",
    )

    results = []
    for gw in sorted(all_stats["gameweek"].unique()):
        if gw < start_gw or gw > end_gw:
            continue
        actual = _actual_gw_points(all_stats, gw, score_2627=score_2627)
        actual_minutes = _actual_gw_minutes(all_stats, gw)
        pts = _score_squad(
            squad_ids, starting_ids, captain_id, actual,
            vice_captain_id=vice_captain_id,
            minutes=actual_minutes, positions=positions, bench_order=bench_order_map,
        )
        results.append({"gameweek": gw, "actual_pts": pts})

    df = pd.DataFrame(results)
    if not df.empty:
        logger.info(
            "Frozen template complete: GW%d-%d | total=%d | avg=%.1f",
            df["gameweek"].min(), df["gameweek"].max(),
            df["actual_pts"].sum(), df["actual_pts"].mean(),
        )
    return df


def _fit_population_model(n_gws: int) -> tuple[float, float]:
    """Normal(mu, sigma) over season-TOTAL points across the window, fit
    through the two plan-given anchor points. See module docstring."""
    mu = AVG_MANAGER_PTS_PER_GW * n_gws
    top10k_percentile = 1.0 - 10_000 / ASSUMED_POPULATION_SIZE
    z = stats.norm.ppf(top10k_percentile)
    sigma = (TOP_10K_PTS_PER_GW * n_gws - mu) / z
    return mu, sigma


def _percentile_and_rank(season_total: float, mu: float, sigma: float) -> tuple[float, float]:
    percentile = float(stats.norm.cdf(season_total, mu, sigma))
    rank = max(1.0, (1.0 - percentile) * ASSUMED_POPULATION_SIZE)
    return percentile, rank


def simulate_rank_distribution(
    per_gw_mean: np.ndarray,
    per_gw_var: np.ndarray,
    mu: float,
    sigma: float,
    n_trials: int = N_MC_TRIALS,
    seed: int = 42,
) -> pd.DataFrame:
    """MC over the bot's OWN per-GW predictive uncertainty (see module
    docstring) -- independent Normal draw per GW, summed to a season total,
    then mapped through the population model to a percentile/rank. Returns
    one row per trial: ``season_total``, ``percentile``, ``rank``."""
    rng = np.random.default_rng(seed)
    std = np.sqrt(np.clip(per_gw_var, 0.0, None))
    draws = rng.normal(per_gw_mean, std, size=(n_trials, len(per_gw_mean)))
    season_totals = draws.sum(axis=1)
    percentiles = stats.norm.cdf(season_totals, mu, sigma)
    ranks = np.clip((1.0 - percentiles) * ASSUMED_POPULATION_SIZE, 1.0, None)
    return pd.DataFrame(
        {"season_total": season_totals, "percentile": percentiles, "rank": ranks}
    )


def run_gate(season: str = "2025-26", start_gw: int = 6, end_gw: int = 38) -> dict:
    n_gws = end_gw - start_gw + 1
    mu, sigma = _fit_population_model(n_gws)

    logger.info("=== v2 bot (full decision engine) ===")
    bot_df = run_backtest(season=season, start_gw=start_gw, end_gw=end_gw, score_2627=True)

    logger.info("=== v1 bot (naive-XI harness) ===")
    v1_df = run_naive_xi_backtest(season=season, start_gw=start_gw, end_gw=end_gw, score_2627=True)

    logger.info("=== frozen template ===")
    frozen_df = run_frozen_template(season=season, start_gw=start_gw, end_gw=end_gw)

    bot_actual_total = float(bot_df["net_pts"].sum())
    v1_actual_total = float(v1_df["actual_pts"].sum())
    frozen_actual_total = float(frozen_df["actual_pts"].sum())

    bot_pct, bot_rank = _percentile_and_rank(bot_actual_total, mu, sigma)
    v1_pct, v1_rank = _percentile_and_rank(v1_actual_total, mu, sigma)
    frozen_pct, frozen_rank = _percentile_and_rank(frozen_actual_total, mu, sigma)

    has_var = "predicted_var" in bot_df.columns and bot_df["predicted_var"].notna().any()
    sim = None
    if has_var:
        per_gw_mean = (bot_df["predicted_xpts"] - bot_df["hit_penalty"]).to_numpy()
        per_gw_var = bot_df["predicted_var"].to_numpy()
        sim = simulate_rank_distribution(per_gw_mean, per_gw_var, mu, sigma)

    result = {
        "season": season, "start_gw": start_gw, "end_gw": end_gw, "n_gws": n_gws,
        "population_model": {"mu": mu, "sigma": sigma,
                              "assumed_population_size": ASSUMED_POPULATION_SIZE},
        "v2_bot": {"actual_total": bot_actual_total, "percentile": bot_pct, "rank": bot_rank},
        "v1_bot": {"actual_total": v1_actual_total, "percentile": v1_pct, "rank": v1_rank},
        "frozen_template": {"actual_total": frozen_actual_total,
                             "percentile": frozen_pct, "rank": frozen_rank},
        "avg_manager_reference_total": AVG_MANAGER_PTS_PER_GW * n_gws,
        "top_10k_reference_total": TOP_10K_PTS_PER_GW * n_gws,
    }

    if sim is not None:
        q = sim["rank"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        pq = sim["season_total"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        result["v2_bot_simulated_rank_distribution"] = {
            "p05": float(q.loc[0.05]), "p25": float(q.loc[0.25]), "p50": float(q.loc[0.50]),
            "p75": float(q.loc[0.75]), "p95": float(q.loc[0.95]),
        }
        result["v2_bot_simulated_points_distribution"] = {
            "p05": float(pq.loc[0.05]), "p25": float(pq.loc[0.25]), "p50": float(pq.loc[0.50]),
            "p75": float(pq.loc[0.75]), "p95": float(pq.loc[0.95]),
        }

    logger.info("=== WALK-FORWARD GATE RESULT (%s, GW%d-%d) ===", season, start_gw, end_gw)
    logger.info(
        "Frozen template: %d pts (rank ~%.0f, %.1f%%ile)",
        frozen_actual_total, frozen_rank, frozen_pct * 100,
    )
    logger.info(
        "v1 bot (naive-XI): %d pts (rank ~%.0f, %.1f%%ile)", v1_actual_total, v1_rank, v1_pct * 100,
    )
    logger.info(
        "v2 bot (this project): %d pts (rank ~%.0f, %.1f%%ile)",
        bot_actual_total, bot_rank, bot_pct * 100,
    )
    if sim is not None:
        logger.info(
            "v2 bot SIMULATED rank distribution: p05=%.0f p25=%.0f p50=%.0f p75=%.0f p95=%.0f",
            result["v2_bot_simulated_rank_distribution"]["p05"],
            result["v2_bot_simulated_rank_distribution"]["p25"],
            result["v2_bot_simulated_rank_distribution"]["p50"],
            result["v2_bot_simulated_rank_distribution"]["p75"],
            result["v2_bot_simulated_rank_distribution"]["p95"],
        )
    logger.info(
        "Reference: avg manager ~%.0f pts, top-10k pace ~%.0f pts",
        result["avg_manager_reference_total"], result["top_10k_reference_total"],
    )

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-3 walk-forward gate")
    p.add_argument("--season", default="2025-26")
    p.add_argument("--start-gw", type=int, default=6)
    p.add_argument("--end-gw", type=int, default=38)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_gate(season=args.season, start_gw=args.start_gw, end_gw=args.end_gw)
