"""P3/P5 — odds → expected team goals (double-Poisson solver)."""

from __future__ import annotations

import pytest

from projection.team_goals import (
    NEUTRAL_LAMBDA_AWAY,
    NEUTRAL_LAMBDA_HOME,
    clean_sheet_prob,
    outcome_probs,
    team_goals_from_odds,
    team_goals_from_strength,
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
    neutral = (NEUTRAL_LAMBDA_HOME, NEUTRAL_LAMBDA_AWAY)
    assert team_goals_from_odds(0.0, 0.0, 0.0) == neutral
    assert team_goals_from_odds(0.5, 0.5, 0.0) == neutral  # no away mass


# --- strength-based lambda (engine review §2) --------------------------------

def test_neutral_teams_give_the_league_average_fixture():
    """Both sides exactly average -> the fitted bases, so the strength path and
    the no-information path agree at the centre instead of disagreeing."""
    lam_h, lam_a = team_goals_from_strength(1.0, 1.0, 1.0, 1.0)
    assert lam_h == pytest.approx(NEUTRAL_LAMBDA_HOME)
    assert lam_a == pytest.approx(NEUTRAL_LAMBDA_AWAY)


def test_strong_attack_raises_and_strong_defence_suppresses():
    """FPL's defence scale runs the opposite way to the goal rate: a HIGHER
    strength_defence means a BETTER defence, so it must reduce the opponent's
    lambda. Getting that sign wrong would invert every unpriced fixture."""
    base_h, base_a = team_goals_from_strength(1.0, 1.0, 1.0, 1.0)

    strong_home_attack, _ = team_goals_from_strength(1.3, 1.0, 1.0, 1.0)
    assert strong_home_attack > base_h

    weak_away_defence, _ = team_goals_from_strength(1.0, 1.0, 1.0, 0.8)
    assert weak_away_defence > base_h

    strong_away_defence, _ = team_goals_from_strength(1.0, 1.0, 1.0, 1.3)
    assert strong_away_defence < base_h

    _, away_vs_strong_home_defence = team_goals_from_strength(1.0, 1.3, 1.0, 1.0)
    assert away_vs_strong_home_defence < base_a


def test_missing_strengths_degrade_one_term_at_a_time():
    """A promoted side with no prior top-flight rating must not drag the whole
    fixture to neutral -- the half we DO know still differentiates it."""
    known_only = team_goals_from_strength(1.4, 1.4, None, None)
    assert known_only[0] == pytest.approx(
        team_goals_from_strength(1.4, 1.4, 1.0, 1.0)[0]
    )
    # A fully unknown fixture is exactly the league average.
    assert team_goals_from_strength(None, None, None, None) == pytest.approx(
        (NEUTRAL_LAMBDA_HOME, NEUTRAL_LAMBDA_AWAY)
    )


def test_strength_lambdas_stay_inside_the_solver_bounds():
    """Extreme ratios are clipped to the same range the odds solver uses, so a
    freak strength ratio cannot produce a nonsensical Poisson rate."""
    lam_h, lam_a = team_goals_from_strength(5.0, 0.1, 5.0, 0.1)
    assert 0.05 <= lam_h <= 6.0
    assert 0.05 <= lam_a <= 6.0
