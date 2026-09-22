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


def test_summarize_reports_one_row_per_season_plus_pooled():
    raw = pd.DataFrame([
        _raw_row("2024-25", 6, False, 40.0),
        _raw_row("2024-25", 6, True, 42.0),
        _raw_row("2024-25", 7, False, 30.0),
        _raw_row("2024-25", 7, True, 34.0),
        _raw_row("2025-26", 6, False, 50.0),
        _raw_row("2025-26", 6, True, 55.0),
        _raw_row("2025-26", 7, False, 45.0),
        _raw_row("2025-26", 7, True, 50.0),
    ])
    summary = calib.summarize(raw, seasons=["2024-25", "2025-26"])

    assert list(summary["season"]) == ["2024-25", "2025-26", "pooled"]
    row = summary[summary["season"] == "2024-25"].iloc[0]
    assert row["off_mean_actual_pts"] == pytest.approx(35.0)
    assert row["on_mean_actual_pts"] == pytest.approx(38.0)
    assert row["diff_mean"] == pytest.approx(3.0)
    assert row["n_gws"] == 2
    pooled = summary[summary["season"] == "pooled"].iloc[0]
    assert pooled["n_gws"] == 4


def test_summarize_does_not_crash_on_zero_variance_diffs():
    raw = pd.DataFrame([
        _raw_row("2025-26", 6, False, 50.0),
        _raw_row("2025-26", 6, True, 50.0),
        _raw_row("2025-26", 7, False, 50.0),
        _raw_row("2025-26", 7, True, 50.0),
    ])
    summary = calib.summarize(raw, seasons=["2025-26"])
    row = summary[summary["season"] == "2025-26"].iloc[0]
    assert pd.isna(row["wilcoxon_p"])


def test_summarize_handles_a_season_with_no_rows():
    raw = pd.DataFrame([
        _raw_row("2025-26", 6, False, 50.0),
        _raw_row("2025-26", 6, True, 55.0),
    ])
    summary = calib.summarize(raw, seasons=["2024-25", "2025-26"])
    row = summary[summary["season"] == "2024-25"].iloc[0]
    assert row["n_gws"] == 0
    assert pd.isna(row["wilcoxon_p"])


def _summary_row(season, off_mean, on_mean, p):
    return {
        "season": season, "off_mean_actual_pts": off_mean,
        "on_mean_actual_pts": on_mean, "diff_mean": on_mean - off_mean,
        "wilcoxon_stat": 0.0, "wilcoxon_p": p, "n_gws": 33,
    }


def test_recommend_flips_only_when_both_seasons_win_and_pooled_is_significant():
    summary = pd.DataFrame([
        _summary_row("2024-25", 50.0, 51.0, 0.2),
        _summary_row("2025-26", 48.0, 50.0, 0.2),
        _summary_row("pooled", 49.0, 50.5, 0.01),
    ])
    rec = calib.recommend(summary, seasons=["2024-25", "2025-26"])
    assert "flip" in rec.lower()
    assert "leave off" not in rec.lower()


def test_recommend_holds_off_when_pooled_p_is_not_significant():
    summary = pd.DataFrame([
        _summary_row("2024-25", 50.0, 51.0, 0.2),
        _summary_row("2025-26", 48.0, 50.0, 0.2),
        _summary_row("pooled", 49.0, 50.5, 0.19),
    ])
    rec = calib.recommend(summary, seasons=["2024-25", "2025-26"])
    assert "leave off" in rec.lower()


def test_recommend_holds_off_when_one_season_loses():
    summary = pd.DataFrame([
        _summary_row("2024-25", 50.0, 49.0, 0.2),
        _summary_row("2025-26", 48.0, 50.0, 0.2),
        _summary_row("pooled", 49.0, 49.5, 0.01),
    ])
    rec = calib.recommend(summary, seasons=["2024-25", "2025-26"])
    assert "leave off" in rec.lower()
