"""Sweeping mu for Objective v2 (scripts/calibrate_risk_constants.py)."""

from __future__ import annotations

import pandas as pd

import scripts.calibrate_risk_constants as calib


def test_rebuild_harness_is_selectable(monkeypatch):
    seen = {}

    def fake_rebuild(season, start_gw, end_gw, mu_candidates=None, **kwargs):
        seen["mu_candidates"] = mu_candidates
        seen["calls"] = seen.get("calls", 0) + 1
        return pd.DataFrame([
            {"gameweek": 6, "actual_pts": 50.0, "n_clubs_at_cap": 1, "mu_baseline": 0.0},
            {"gameweek": 6, "actual_pts": 44.0, "n_clubs_at_cap": 0, "mu_baseline": -0.5},
        ])

    monkeypatch.setattr(calib.bt, "run_rebuild_backtest", fake_rebuild)

    df = calib.sweep_mu_baseline(
        "2025-26", 6, 6, candidates=[0.0, -0.5], harness="rebuild"
    )

    assert seen["mu_candidates"] == [0.0, -0.5], "every candidate reaches the harness"
    assert seen["calls"] == 1, "ONE pass, so all candidates share one pool"
    assert list(df["mu_baseline"]) == [0.0, -0.5]
    assert list(df["avg_actual_pts_per_gw"]) == [50.0, 44.0]
    assert list(df["avg_clubs_at_cap"]) == [1.0, 0.0]


def test_rebuild_sweep_compares_candidates_on_identical_draws(monkeypatch):
    """Running the harness once per candidate would give each its own Monte
    Carlo draws, so a difference could be sampling noise rather than mu."""
    calls = []
    monkeypatch.setattr(
        calib.bt, "run_rebuild_backtest",
        lambda season, start_gw, end_gw, mu_candidates=None, **k: (
            calls.append(mu_candidates),
            pd.DataFrame([
                {"gameweek": 6, "actual_pts": 1.0, "n_clubs_at_cap": 0, "mu_baseline": m}
                for m in mu_candidates
            ]),
        )[1],
    )
    calib.sweep_mu_baseline(
        "2025-26", 6, 6, candidates=[-1.0, 0.0, 0.5], harness="rebuild"
    )
    assert calls == [[-1.0, 0.0, 0.5]]


def test_sweep_does_not_mutate_module_state_on_the_rebuild_path(monkeypatch):
    """The naive-XI sweep mutates bt._BACKTEST_CONFIG; the rebuild path must
    not, so repeated or nested runs cannot interfere."""
    original = calib.bt._BACKTEST_CONFIG

    monkeypatch.setattr(
        calib.bt, "run_rebuild_backtest",
        lambda season, start_gw, end_gw, mu_candidates=None, **k: pd.DataFrame(
            [{"gameweek": 6, "actual_pts": 1.0, "n_clubs_at_cap": 0, "mu_baseline": 0.3}]
        ),
    )
    calib.sweep_mu_baseline("2025-26", 6, 6, candidates=[0.3], harness="rebuild")

    assert calib.bt._BACKTEST_CONFIG is original


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
        lambda season, start_gw, end_gw, mu_candidates=None, **k: pd.DataFrame(),
    )
    df = calib.sweep_mu_baseline(
        "2025-26", 6, 6, candidates=[0.0], harness="rebuild"
    )
    assert df["n_gws"].iloc[0] == 0
    assert pd.isna(df["avg_actual_pts_per_gw"].iloc[0])


def test_raw_out_writes_the_per_gameweek_rows(monkeypatch, tmp_path):
    """The aggregate means cannot distinguish a real effect from one lucky
    gameweek. The paired test that can needs the individual rows kept."""
    monkeypatch.setattr(
        calib.bt, "run_rebuild_backtest",
        lambda season, start_gw, end_gw, mu_candidates=None, **k: pd.DataFrame([
            {"gameweek": 6, "actual_pts": 50.0, "n_clubs_at_cap": 1, "mu_baseline": 0.0},
            {"gameweek": 7, "actual_pts": 60.0, "n_clubs_at_cap": 2, "mu_baseline": 0.0},
        ]),
    )
    out = tmp_path / "nested" / "raw.csv"
    calib.sweep_mu_baseline(
        "2025-26", 6, 7, candidates=[0.0], harness="rebuild", raw_out=out
    )

    assert out.exists()
    written = pd.read_csv(out)
    assert list(written["gameweek"]) == [6, 7]
    assert "mu_baseline" in written.columns
