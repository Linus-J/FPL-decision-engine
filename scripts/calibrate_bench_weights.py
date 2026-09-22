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
