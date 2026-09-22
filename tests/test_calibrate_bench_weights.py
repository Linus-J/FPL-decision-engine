"""Bench-weight rebuild-path validation (scripts/calibrate_bench_weights.py)."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import scripts.calibrate_bench_weights as calib


def _raw_row(season, gw, derive, actual_pts):
    return {
        "season": season, "gameweek": gw, "derive_bench_weights": derive,
        "actual_pts": actual_pts, "predicted_xpts": 0.0, "total_cost": 100.0,
        "n_clubs_at_cap": 0,
    }


def test_paired_diffs_joins_on_gameweek_within_one_season():
    raw = pd.DataFrame([
        _raw_row("2025-26", 6, False, 50.0),
        _raw_row("2025-26", 6, True, 55.0),
        _raw_row("2025-26", 7, False, 40.0),
        _raw_row("2025-26", 7, True, 38.0),
    ])
    diffs = calib._paired_diffs(raw, season="2025-26")
    assert sorted(diffs.tolist()) == [-2.0, 5.0]


def test_paired_diffs_pools_across_seasons_when_season_is_none():
    raw = pd.DataFrame([
        _raw_row("2024-25", 10, False, 30.0),
        _raw_row("2024-25", 10, True, 33.0),
        _raw_row("2025-26", 6, False, 50.0),
        _raw_row("2025-26", 6, True, 55.0),
    ])
    diffs = calib._paired_diffs(raw, season=None)
    assert sorted(diffs.tolist()) == [3.0, 5.0]
