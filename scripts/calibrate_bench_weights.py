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
