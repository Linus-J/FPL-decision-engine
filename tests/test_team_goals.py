"""P3/P5 — odds → expected team goals (double-Poisson solver)."""

from __future__ import annotations

import pytest

from projection.team_goals import (
    clean_sheet_prob,
    outcome_probs,
    team_goals_from_odds,
)


@pytest.mark.parametrize("lam_h,lam_a", [(1.6, 1.1), (2.2, 0.7), (0.9, 0.9), (1.2, 1.8)])
def test_recovers_known_lambdas_roundtrip(lam_h, lam_a):
    # generate the exact odds a fixture with these rates would imply, then
    # confirm the solver recovers the rates
    h, d, a, o = outcome_probs(lam_h, lam_a)
    rec_h, rec_a = team_goals_from_odds(h, d, a, o)
    assert rec_h == pytest.approx(lam_h, abs=0.06)
    assert rec_a == pytest.approx(lam_a, abs=0.06)


def test_outcome_probs_normalise():
    h, d, a, o = outcome_probs(1.5, 1.2)
    assert h + d + a == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= o <= 1.0


def test_strong_favourite_has_higher_home_rate():
    # heavy home favourite → λ_home > λ_away
    lam_h, lam_a = team_goals_from_odds(0.75, 0.17, 0.08, 0.55)
    assert lam_h > lam_a


def test_clean_sheet_prob_monotonic():
    assert clean_sheet_prob(0.0) == pytest.approx(1.0)
    assert clean_sheet_prob(1.0) > clean_sheet_prob(2.0) > clean_sheet_prob(3.0)
    # a strong defence (low opponent λ) keeps more clean sheets
    assert clean_sheet_prob(0.5) == pytest.approx(0.6065, abs=1e-3)


def test_degenerate_odds_fall_back():
    assert team_goals_from_odds(0.0, 0.0, 0.0) == (1.35, 1.15)
    assert team_goals_from_odds(0.5, 0.5, 0.0) == (1.35, 1.15)  # no away mass
