"""P3-3 objective v1 — the linear EO/variance reweighting (optimiser/scoring.py).

See the module docstring for why this is a multiplicative EO reweighting +
linear own-variance term rather than the plan's literal differential_value
formula (confirmed algebraically to be a no-op for the mean objective).
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.strategy import OptimiserConfig
from optimiser.scoring import (
    add_effective_score,
    differential_multiplier,
    lambda_mu_for_risk_mode,
    risk_adjusted_score,
)


def test_lambda_mu_balanced_is_always_zero():
    assert lambda_mu_for_risk_mode("balanced", 0.5, 0.1) == (0.0, 0.0)


def test_lambda_mu_safe_is_negative():
    lam, mu = lambda_mu_for_risk_mode("safe", 0.5, 0.1)
    assert lam == pytest.approx(-0.5)
    assert mu == pytest.approx(-0.1)


def test_lambda_mu_aggressive_is_positive():
    lam, mu = lambda_mu_for_risk_mode("aggressive", 0.5, 0.1)
    assert lam == pytest.approx(0.5)
    assert mu == pytest.approx(0.1)


def test_lambda_mu_unknown_mode_defaults_to_balanced():
    assert lambda_mu_for_risk_mode("typo-mode", 0.5, 0.1) == (0.0, 0.0)


def test_differential_multiplier_zero_lambda_is_always_one():
    assert differential_multiplier(0.0, 0.0) == 1.0
    assert differential_multiplier(100.0, 0.0) == 1.0
    assert differential_multiplier(37.0, 0.0) == 1.0


def test_differential_multiplier_positive_lambda_favours_low_ownership():
    zero_owned = differential_multiplier(0.0, 0.5)
    fully_owned = differential_multiplier(100.0, 0.5)
    assert zero_owned == pytest.approx(1.5)
    assert fully_owned == pytest.approx(1.0)
    assert zero_owned > fully_owned


def test_differential_multiplier_negative_lambda_favours_high_ownership():
    zero_owned = differential_multiplier(0.0, -0.5)
    fully_owned = differential_multiplier(100.0, -0.5)
    assert zero_owned == pytest.approx(0.5)
    assert fully_owned == pytest.approx(1.0)
    assert fully_owned > zero_owned


def test_differential_multiplier_clamped_at_zero():
    assert differential_multiplier(0.0, -5.0) == 0.0


def test_risk_adjusted_score_balanced_reduces_to_raw_xpts():
    # lam=0, mu=0 (balanced) -> effective score == raw xpts exactly
    assert risk_adjusted_score(xpts=6.0, xpts_var=4.0, eo_pct=50.0, lam=0.0, mu=0.0) == 6.0


def test_risk_adjusted_score_variance_term():
    aggressive = risk_adjusted_score(xpts=6.0, xpts_var=4.0, eo_pct=50.0, lam=0.0, mu=0.5)
    safe = risk_adjusted_score(xpts=6.0, xpts_var=4.0, eo_pct=50.0, lam=0.0, mu=-0.5)
    assert aggressive == pytest.approx(8.0)   # 6 + 0.5*4
    assert safe == pytest.approx(4.0)         # 6 - 0.5*4
    assert aggressive > safe


def test_add_effective_score_no_ownership_is_constant_rescale():
    # EO unavailable (current live reality, pre-GW1) -> eo_pct=0 for everyone
    # -> effective_score is a UNIFORM rescale, ranking is unchanged from raw
    # xpts (the "degrades to pre-P3-3 behaviour" guarantee)
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 5, "xpts": 6.0, "xpts_var": 4.0},
        {"player_id": 2, "gameweek": 5, "xpts": 4.0, "xpts_var": 1.0},
    ])
    config = OptimiserConfig(
        risk_mode="aggressive", max_ownership_differential=0.5, variance_weight=0.0
    )
    out = add_effective_score(projections, ownership=None, config=config)
    # both get the SAME multiplier (eo=0 for both) -> ranking preserved
    assert out.loc[out["player_id"] == 1, "effective_score"].iloc[0] > \
        out.loc[out["player_id"] == 2, "effective_score"].iloc[0]
    # exact expected value: 6.0 * (1 + 0.5*(1-0)) = 9.0
    assert out.loc[out["player_id"] == 1, "effective_score"].iloc[0] == pytest.approx(9.0)


def test_add_effective_score_balanced_mode_ignores_ownership():
    projections = pd.DataFrame([{"player_id": 1, "gameweek": 5, "xpts": 6.0, "xpts_var": 4.0}])
    ownership = pd.DataFrame([{"player_id": 1, "top10k_selected_pct": 80.0}])
    config = OptimiserConfig(risk_mode="balanced")
    out = add_effective_score(projections, ownership=ownership, config=config)
    assert out["effective_score"].iloc[0] == 6.0


def test_add_effective_score_missing_ownership_row_defaults_to_zero_eo():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 5, "xpts": 6.0, "xpts_var": 0.0},
        {"player_id": 2, "gameweek": 5, "xpts": 6.0, "xpts_var": 0.0},
    ])
    ownership = pd.DataFrame([{"player_id": 1, "top10k_selected_pct": 100.0}])
    config = OptimiserConfig(risk_mode="aggressive", max_ownership_differential=0.5)
    out = add_effective_score(projections, ownership=ownership, config=config)
    p1 = out.loc[out["player_id"] == 1, "effective_score"].iloc[0]
    p2 = out.loc[out["player_id"] == 2, "effective_score"].iloc[0]
    # player 2 (no ownership row -> eo=0, max differential) scores higher
    # than player 1 (100% owned) under aggressive/differential-seeking mode
    assert p2 > p1


def test_add_effective_score_missing_xpts_var_column_defaults_to_zero():
    projections = pd.DataFrame([{"player_id": 1, "gameweek": 5, "xpts": 6.0}])
    out = add_effective_score(projections)
    assert out["effective_score"].iloc[0] == 6.0
