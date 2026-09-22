#!/usr/bin/env python
"""Rebuild-path validation for `derive_bench_weights_per_solve`
(docs/superpowers/specs/2026-09-22-bench-weight-rebuild-path-validation-design.md).

Runs `run_rebuild_backtest` with the flag off vs. on across both 2024-25 and
2025-26 (GW6-38), and reports a per-season and pooled paired Wilcoxon
comparison on realized, bench-inclusive points (`_score_squad`/
`_apply_autosubs` already replace a blanking starter from the bench using
real minutes, mirroring FPL's own auto-substitution rule).

Validates the REBUILD shape only (wildcard/free-hit/cold-start, via
`optimise_squad`'s derive-and-resolve step) -- not the weekly-hoist shape
(`agent/decision_engine.py::derive_bench_config`), which no backtest harness
exercises. See the design doc for why that split is deliberate.

Usage:
    uv run python scripts/calibrate_bench_weights.py
    uv run python scripts/calibrate_bench_weights.py --seasons 2025-26
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
from scipy.stats import wilcoxon

import scripts.backtest as bt

logger = logging.getLogger(__name__)

DEFAULT_SEASONS = ("2024-25", "2025-26")


def _paired_diffs(raw: pd.DataFrame, season: str | None = None) -> pd.Series:
    """Per-(season, gameweek) `on - off` actual-points differences.

    Pooling (``season=None``) concatenates every season's paired differences
    rather than joining across seasons -- a gameweek number is not a join key
    between two different seasons, only within one.
    """
    subset = raw if season is None else raw[raw["season"] == season]
    off = subset[~subset["derive_bench_weights"]][["season", "gameweek", "actual_pts"]]
    on = subset[subset["derive_bench_weights"]][["season", "gameweek", "actual_pts"]]
    merged = off.merge(on, on=["season", "gameweek"], suffixes=("_off", "_on"))
    return merged["actual_pts_on"] - merged["actual_pts_off"]


def _wilcoxon_or_nan(diffs: pd.Series) -> tuple[float, float]:
    if len(diffs) < 2 or (diffs == diffs.iloc[0]).all():
        return float("nan"), float("nan")
    try:
        result = wilcoxon(diffs)
    except ValueError:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def _season_row(raw: pd.DataFrame, season: str | None) -> dict:
    subset = raw if season is None else raw[raw["season"] == season]
    off = subset[~subset["derive_bench_weights"]]["actual_pts"]
    on = subset[subset["derive_bench_weights"]]["actual_pts"]
    diffs = _paired_diffs(raw, season=season)
    stat, p = _wilcoxon_or_nan(diffs)
    return {
        "season": season if season is not None else "pooled",
        "off_mean_actual_pts": float(off.mean()) if len(off) else float("nan"),
        "on_mean_actual_pts": float(on.mean()) if len(on) else float("nan"),
        "diff_mean": float(diffs.mean()) if len(diffs) else float("nan"),
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "n_gws": int(len(diffs)),
    }


def summarize(raw: pd.DataFrame, seasons) -> pd.DataFrame:
    """One row per season plus a final pooled row (spec: pooling mirrors the
    aggregate approach `b03a985` already used on this harness, so a weak or
    non-replicating result surfaces the same way it did there)."""
    rows = [_season_row(raw, season) for season in seasons]
    rows.append(_season_row(raw, None))
    return pd.DataFrame(rows)


def recommend(summary: pd.DataFrame, seasons) -> str:
    """The pre-committed bar (design doc, Decision criterion): flip only if
    "on" beats "off" on mean actual pts/GW in EVERY season individually, and
    the pooled paired Wilcoxon test gives p < 0.05. Never a judgement call."""
    per_season = summary[summary["season"].isin(seasons)]
    wins_every_season = bool(
        (per_season["on_mean_actual_pts"] > per_season["off_mean_actual_pts"]).all()
    )
    pooled = summary[summary["season"] == "pooled"].iloc[0]
    pooled_significant = bool(pooled["wilcoxon_p"] < 0.05)

    if wins_every_season and pooled_significant:
        return (
            f"RECOMMENDATION: flip derive_bench_weights_per_solve to True. "
            f"Beat static weights in every season checked "
            f"({', '.join(seasons)}); pooled Wilcoxon p={pooled['wilcoxon_p']:.4f}."
        )
    reasons = []
    if not wins_every_season:
        reasons.append("did not beat static weights in every season")
    if not pooled_significant:
        reasons.append(f"pooled Wilcoxon p={pooled['wilcoxon_p']:.4f} is not < 0.05")
    return (
        "RECOMMENDATION: leave off derive_bench_weights_per_solve. "
        + "; ".join(reasons) + "."
    )


def collect_raw(seasons, start_gw: int, end_gw: int) -> pd.DataFrame:
    """One independent `run_rebuild_backtest` call per (season, flag) cell.

    Unlike the `mu_candidates` sweep in `run_rebuild_backtest`, these cannot
    share one pool: the flag changes `generate_squad_pool`'s own per-candidate
    solves, so the pool itself differs between the off and on arms.
    """
    frames = []
    for season in seasons:
        for derive in (False, True):
            cfg = dataclasses.replace(
                bt._BACKTEST_CONFIG, derive_bench_weights_per_solve=derive
            )
            logger.info(
                "Running rebuild backtest season=%s derive_bench_weights=%s "
                "GW%d-%d ...", season, derive, start_gw, end_gw,
            )
            df = bt.run_rebuild_backtest(
                season=season, start_gw=start_gw, end_gw=end_gw, config=cfg
            )
            if df.empty:
                continue
            df = df.copy()
            df["season"] = season
            df["derive_bench_weights"] = derive
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    p.add_argument("--start-gw", type=int, default=6)
    p.add_argument("--end-gw", type=int, default=38)
    p.add_argument(
        "--raw-out", type=Path,
        default=Path("results/bench_weight_backtest_raw.csv"),
    )
    p.add_argument(
        "--out", type=Path, default=Path("results/bench_weight_calibration.csv"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    raw = collect_raw(args.seasons, args.start_gw, args.end_gw)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.raw_out, index=False)
    logger.info("Per-(season, gameweek, flag) rows written to %s", args.raw_out)

    summary = summarize(raw, args.seasons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    logger.info("Summary written to %s", args.out)
    print(summary.to_string(index=False))

    logger.info(recommend(summary, args.seasons))


if __name__ == "__main__":
    main()
