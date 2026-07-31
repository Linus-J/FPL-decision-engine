#!/usr/bin/env python
"""Sweeps mu_baseline (the "medium risk" variance-awareness constant,
plan/risk-aware-cold-start-v1.md) against the Phase-2 exit-gate harness to
find a value better-grounded than the untuned starting guess.

Deliberately does NOT calibrate mu_range or chip-timing thresholds here:

- mu_range only has any effect at a non-zero risk_level, and the harness's
  metric (mean actual points per GW) has no way to reward "a fatter
  right tail at a similar mean" -- walk_forward_gate.py's simulated
  final-rank distribution exists specifically for that question and is a
  separate, heavier evaluation than this script attempts.
- Chip-timing thresholds gate rare events (each chip fires ~1-2 times per
  half-season) -- a short backtest window won't exercise enough chip
  decisions for a reliable signal either way.

Runs run_naive_xi_backtest (the actual Phase-2 exit-gate function, not the
full transfers+chips harness) so the measured effect isolates the risk
scoring change from transfer/chip decision noise. Uses a REDUCED GW window
by default for speed -- this is a directional first pass, not a final
gate validation (re-run the full GW6-38 window on whatever value wins here
before trusting it as a real gate number).

Usage:
    uv run python scripts/calibrate_risk_constants.py
    uv run python scripts/calibrate_risk_constants.py --start-gw 6 --end-gw 38
"""
import argparse
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

import pandas as pd

import scripts.backtest as bt
from config.strategy import OPTIMISER

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = [-0.05, 0.0, 0.05, 0.1, 0.15, 0.2]


def sweep_mu_baseline(
    season: str, start_gw: int, end_gw: int, candidates: list[float]
) -> pd.DataFrame:
    """Runs the naive-XI exit-gate harness once per candidate, holding
    risk_level=0.0 (the real squad's setting) and mu_range=0.0 (irrelevant
    at risk_level=0 -- mu = mu_baseline + 0*mu_range = mu_baseline exactly).
    Mutates scripts.backtest._BACKTEST_CONFIG directly between runs --
    fine for a one-off calibration script, not something a test or the
    real decision path should ever do."""
    original_config = bt._BACKTEST_CONFIG
    rows = []
    try:
        for mu_baseline in candidates:
            bt._BACKTEST_CONFIG = dataclasses.replace(
                OPTIMISER, risk_level=0.0, mu_baseline=mu_baseline, mu_range=0.0
            )
            logger.info(
                "Running naive-XI backtest GW%d-%d, mu_baseline=%.3f ...",
                start_gw, end_gw, mu_baseline,
            )
            df = bt.run_naive_xi_backtest(
                season=season, start_gw=start_gw, end_gw=end_gw, score_2627=True
            )
            # run_naive_xi_backtest has no transfers/chips/hits (fixed initial 15,
            # re-optimised starting XI only) so its actual_pts column IS the net
            # outcome for this harness -- unlike run_backtest there's no separate
            # net_pts column to prefer over it.
            avg_actual = float(df["actual_pts"].mean()) if not df.empty else float("nan")
            rows.append({
                "mu_baseline": mu_baseline,
                "avg_actual_pts_per_gw": round(avg_actual, 3),
                "n_gws": len(df),
            })
            logger.info(
                "mu_baseline=%.3f -> avg %.2f actual pts/GW over %d GWs",
                mu_baseline, avg_actual, len(df),
            )
    finally:
        bt._BACKTEST_CONFIG = original_config
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--start-gw", type=int, default=6)
    parser.add_argument("--end-gw", type=int, default=20)
    parser.add_argument(
        "--candidates", type=float, nargs="+", default=DEFAULT_CANDIDATES,
    )
    parser.add_argument("--out", type=Path, default=Path("results/mu_baseline_calibration.csv"))
    args = parser.parse_args()

    results = sweep_mu_baseline(args.season, args.start_gw, args.end_gw, args.candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    best = results.loc[results["avg_actual_pts_per_gw"].idxmax()]
    logger.info("Results written to %s", args.out)
    logger.info(
        "Best in this sweep: mu_baseline=%.3f (%.2f actual pts/GW). "
        "Today's default is %.3f. Re-run the full GW6-38 window on the "
        "winner before changing config/strategy.py -- this was a reduced "
        "window for speed, not a final gate validation.",
        best["mu_baseline"], best["avg_actual_pts_per_gw"], OPTIMISER.mu_baseline,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
