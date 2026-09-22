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


def test_collect_raw_runs_four_independent_cells_and_tags_them(monkeypatch):
    seen_configs = []

    def fake_rebuild(season, start_gw, end_gw, config=None, **kwargs):
        seen_configs.append((season, config.derive_bench_weights_per_solve))
        return pd.DataFrame([
            {
                "gameweek": 6, "actual_pts": 50.0, "predicted_xpts": 40.0,
                "total_cost": 100.0, "n_clubs_at_cap": 0,
            },
        ])

    monkeypatch.setattr(calib.bt, "run_rebuild_backtest", fake_rebuild)

    raw = calib.collect_raw(["2024-25", "2025-26"], start_gw=6, end_gw=6)

    assert seen_configs == [
        ("2024-25", False), ("2024-25", True),
        ("2025-26", False), ("2025-26", True),
    ], "one off cell and one on cell per season, off called before on"
    assert set(raw["season"]) == {"2024-25", "2025-26"}
    assert set(raw["derive_bench_weights"]) == {False, True}
    assert len(raw) == 4


def test_collect_raw_passes_the_backtest_config_baseline(monkeypatch):
    """Only the flag should differ from `_BACKTEST_CONFIG` -- risk_level,
    mu_baseline, mu_range stay pinned to the backtest's stable baseline."""
    seen = []
    monkeypatch.setattr(
        calib.bt, "run_rebuild_backtest",
        lambda season, start_gw, end_gw, config=None, **k: (
            seen.append(config), pd.DataFrame()
        )[1],
    )
    calib.collect_raw(["2025-26"], start_gw=6, end_gw=6)
    off_cfg, on_cfg = seen
    baseline = calib.bt._BACKTEST_CONFIG
    assert off_cfg == dataclasses.replace(baseline, derive_bench_weights_per_solve=False)
    assert on_cfg == dataclasses.replace(baseline, derive_bench_weights_per_solve=True)


def test_main_writes_both_csvs_and_logs_a_recommendation(monkeypatch, tmp_path, caplog):
    def fake_rebuild(season, start_gw, end_gw, config=None, **kwargs):
        bump = 5.0 if config.derive_bench_weights_per_solve else 0.0
        return pd.DataFrame([
            {
                "gameweek": gw, "actual_pts": 50.0 + bump, "predicted_xpts": 40.0,
                "total_cost": 100.0, "n_clubs_at_cap": 0,
            }
            for gw in (6, 7)
        ])

    monkeypatch.setattr(calib.bt, "run_rebuild_backtest", fake_rebuild)

    raw_out = tmp_path / "raw.csv"
    summary_out = tmp_path / "summary.csv"
    with caplog.at_level("INFO"):
        calib.main([
            "--seasons", "2024-25", "2025-26",
            "--start-gw", "6", "--end-gw", "7",
            "--raw-out", str(raw_out), "--out", str(summary_out),
        ])

    assert raw_out.exists()
    assert summary_out.exists()
    raw = pd.read_csv(raw_out)
    assert len(raw) == 8  # 2 seasons x 2 flags x 2 gameweeks
    summary = pd.read_csv(summary_out)
    assert list(summary["season"]) == ["2024-25", "2025-26", "pooled"]
    assert any("RECOMMENDATION" in rec for rec in caplog.messages)
