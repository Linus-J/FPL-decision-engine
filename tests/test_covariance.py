"""P-COV: joint per-fixture sampling produces real teammate covariance (C2)."""

from __future__ import annotations

import numpy as np

import projection.clean_sheets as CS
from projection.covariance import sample_team_goals, split_multinomial


def _players():
    return [
        {"player_id": 1, "weight": 4.0, "minutes_frac": 1.0},
        {"player_id": 2, "weight": 2.0, "minutes_frac": 1.0},
        {"player_id": 3, "weight": 0.0, "minutes_frac": 0.5},
    ]


def test_split_multinomial_conserves_team_total():
    rng = np.random.default_rng(1)
    for _ in range(50):
        team_total = int(rng.integers(0, 5))
        out = split_multinomial(rng, team_total, _players())
        assert sum(out.values()) == team_total


def test_split_multinomial_zero_total_is_all_zero():
    rng = np.random.default_rng(2)
    assert split_multinomial(rng, 0, _players()) == {1: 0, 2: 0, 3: 0}


def test_split_multinomial_no_weight_or_minutes_is_all_zero():
    rng = np.random.default_rng(3)
    players = [{"player_id": 9, "weight": 0.0, "minutes_frac": 0.0}]
    assert split_multinomial(rng, 3, players) == {9: 0}


def test_split_multinomial_empty_players():
    rng = np.random.default_rng(4)
    assert split_multinomial(rng, 3, []) == {}


def test_sample_team_goals_mean_matches_lambda():
    rng = np.random.default_rng(5)
    draws = [sample_team_goals(rng, 1.4) for _ in range(20000)]
    assert abs(np.mean(draws) - 1.4) < 0.05


def test_shared_latent_gives_correlated_cs_points_vs_independent_baseline():
    """The C2 gate: corr(CS points) between two same-team defenders > 0.5
    when scored against a shared team-conceded latent, vs ≈0 when each
    independently redraws their own opponent-goals Poisson (the defect a
    summed-marginal model has)."""
    rng = np.random.default_rng(42)
    lam_opp = 1.1
    p60 = 0.85
    n = 8000

    shared_a, shared_b, indep_a, indep_b = [], [], [], []
    for _ in range(n):
        conceded = sample_team_goals(rng, lam_opp)
        played_a = rng.random() < p60
        played_b = rng.random() < p60
        shared_a.append(
            CS.sample_clean_sheet_points(rng, lam_opp, played_a, "DEF", conceded=conceded)
        )
        shared_b.append(
            CS.sample_clean_sheet_points(rng, lam_opp, played_b, "DEF", conceded=conceded)
        )

        played_a2 = rng.random() < p60
        played_b2 = rng.random() < p60
        indep_a.append(CS.sample_clean_sheet_points(rng, lam_opp, played_a2, "DEF"))
        indep_b.append(CS.sample_clean_sheet_points(rng, lam_opp, played_b2, "DEF"))

    corr_shared = float(np.corrcoef(shared_a, shared_b)[0, 1])
    corr_indep = float(np.corrcoef(indep_a, indep_b)[0, 1])
    assert corr_shared > 0.5, f"shared-latent corr too low: {corr_shared}"
    assert abs(corr_indep) < 0.15, f"independent-draw corr should be ≈0: {corr_indep}"
