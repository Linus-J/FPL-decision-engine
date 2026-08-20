"""Sweeping mu for Objective v2 (scripts/calibrate_risk_constants.py)."""

from __future__ import annotations

import pandas as pd

import scripts.calibrate_risk_constants as calib


def test_rebuild_harness_is_selectable(monkeypatch):
    seen = []

    def fake_rebuild(season, start_gw, end_gw, config=None, **kwargs):
        seen.append(config.mu_baseline)
        return pd.DataFrame([
            {"gameweek": 6, "actual_pts": 50.0, "n_clubs_at_cap": 1},
        ])

    monkeypatch.setattr(calib.bt, "run_rebuild_backtest", fake_rebuild)

    df = calib.sweep_mu_baseline(
        "2025-26", 6, 6, candidates=[0.0, -0.5], harness="rebuild"
    )

    assert seen == [0.0, -0.5], "each candidate mu must reach the harness"
    assert list(df["mu_baseline"]) == [0.0, -0.5]
    assert "avg_clubs_at_cap" in df.columns


def test_sweep_passes_config_rather_than_mutating_module_state(monkeypatch):
    """The naive-XI sweep mutates bt._BACKTEST_CONFIG; the rebuild harness
    takes config as an argument so repeated runs cannot interfere."""
    original = calib.bt._BACKTEST_CONFIG

    monkeypatch.setattr(
        calib.bt, "run_rebuild_backtest",
        lambda season, start_gw, end_gw, config=None, **k: pd.DataFrame(
            [{"gameweek": 6, "actual_pts": 1.0, "n_clubs_at_cap": 0}]
        ),
    )
    calib.sweep_mu_baseline("2025-26", 6, 6, candidates=[0.3], harness="rebuild")

    assert calib.bt._BACKTEST_CONFIG is original


def test_rebuild_sweep_holds_risk_level_and_mu_range_at_zero(monkeypatch):
    """mu = mu_baseline + risk_level * mu_range, so the sweep only isolates
    mu_baseline if the other two are pinned."""
    seen = []
    monkeypatch.setattr(
        calib.bt, "run_rebuild_backtest",
        lambda season, start_gw, end_gw, config=None, **k: (
            seen.append((config.risk_level, config.mu_range)),
            pd.DataFrame([{"gameweek": 6, "actual_pts": 1.0, "n_clubs_at_cap": 0}]),
        )[1],
    )
    calib.sweep_mu_baseline("2025-26", 6, 6, candidates=[-0.5], harness="rebuild")

    assert seen == [(0.0, 0.0)]


def test_naive_xi_harness_is_still_the_default(monkeypatch):
    called = {}

    def fake_naive(**kwargs):
        called["naive"] = True
        return pd.DataFrame([{"gameweek": 6, "actual_pts": 1.0}])

    monkeypatch.setattr(calib.bt, "run_naive_xi_backtest", fake_naive)
    calib.sweep_mu_baseline("2025-26", 6, 6, candidates=[0.0])
    assert called.get("naive") is True


def test_empty_result_frame_does_not_crash_the_sweep(monkeypatch):
    monkeypatch.setattr(
        calib.bt, "run_rebuild_backtest",
        lambda season, start_gw, end_gw, config=None, **k: pd.DataFrame(),
    )
    df = calib.sweep_mu_baseline(
        "2025-26", 6, 6, candidates=[0.0], harness="rebuild"
    )
    assert df["n_gws"].iloc[0] == 0
    assert pd.isna(df["avg_actual_pts_per_gw"].iloc[0])
