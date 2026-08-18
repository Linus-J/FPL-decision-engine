"""Team strength fitted to results, for the gameweeks bookmakers have not priced.

Measured walk-forward against odds-implied lambda over four held-out seasons
(scripts/benchmark_strength_models.py): the published-strength power law wins
when it has the current season's ratings (MAE 0.249 vs 0.262) and loses clearly
when it is running on last season's, which is what it actually does at GW1
(0.319 vs 0.262). That is the case carrying 78% of the initial squad's
projected points, which is why this exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projection.dixon_coles import DixonColesFit, _tau, fit_dixon_coles


def _synthetic_matches(
    seed: int = 0, seasons=("2023-24", "2024-25"), repeats: int = 8
) -> pd.DataFrame:
    """Three tiers of team, simulated forward from known parameters.

    Recovering an ordering the data was built with is the only honest test of a
    fit: asserting on hand-picked coefficients would just pin whatever the
    optimiser happened to do on the day.

    ``repeats`` matters more than it looks. A single round robin over six teams
    is 30 matches a season, and at that size the fit is noise — a first version
    of this fixture had a mid-tier side come out stronger than a strong one, and
    a test written against it was measuring the seed.
    """
    rng = np.random.default_rng(seed)
    strong, mid, weak = [101, 102], [103, 104], [105, 106]
    attack = {**dict.fromkeys(strong, 0.45), **dict.fromkeys(mid, 0.0),
              **dict.fromkeys(weak, -0.45)}
    concede = {**dict.fromkeys(strong, -0.35), **dict.fromkeys(mid, 0.0),
               **dict.fromkeys(weak, 0.35)}
    codes = strong + mid + weak

    rows = []
    for season in seasons:
        gw = 0
        for _ in range(repeats):
            for home in codes:
                for away in codes:
                    if home == away:
                        continue
                    gw += 1
                    lam_h = np.exp(0.35 + attack[home] + concede[away])
                    lam_a = np.exp(0.15 + attack[away] + concede[home])
                    rows.append({
                        "season": season, "gameweek": (gw % 38) + 1,
                        "home_code": home, "away_code": away,
                        "home_goals": int(rng.poisson(lam_h)),
                        "away_goals": int(rng.poisson(lam_a)),
                    })
    return pd.DataFrame(rows)


def test_fit_recovers_the_ordering_it_was_generated_with():
    fit = fit_dixon_coles(_synthetic_matches())
    assert min(fit.attack[c] for c in (101, 102)) > max(fit.attack[c] for c in (105, 106))
    # `concede` is defensive WEAKNESS, so the strong sides must sit lower.
    assert max(fit.concede[c] for c in (101, 102)) < min(fit.concede[c] for c in (105, 106))


def test_a_strong_side_at_home_outscores_a_weak_one():
    fit = fit_dixon_coles(_synthetic_matches())
    strong_home, _ = fit.lambdas(101, 105)
    weak_home, _ = fit.lambdas(105, 101)
    assert strong_home > weak_home


def test_home_advantage_is_positive():
    fit = fit_dixon_coles(_synthetic_matches())
    assert fit.intercept_home > fit.intercept_away
    # Same two sides, reversed: the home team scores more either way round.
    home_a, away_a = fit.lambdas(103, 104)
    home_b, away_b = fit.lambdas(104, 103)
    assert home_a > away_b and home_b > away_a


def test_an_unknown_team_gets_the_promoted_prior_not_league_average():
    """A newly promoted side is not an average Premier League team, and a plain
    shrinkage-to-zero prior says exactly that. Teams appearing for the first
    time partway through the window are the closest observable analogue."""
    matches = _synthetic_matches()
    # A weak side that only exists in the later season, i.e. a newcomer.
    late = matches[matches["season"] == "2024-25"].copy()
    late["home_code"] = late["home_code"].replace({105: 999})
    late["away_code"] = late["away_code"].replace({105: 999})
    matches = pd.concat(
        [matches[matches["season"] == "2023-24"], late], ignore_index=True
    )
    fit = fit_dixon_coles(matches)

    # Compared against the fitted league average rather than a nominated team,
    # so the claim under test is "a newcomer is rated below average" and not
    # "a newcomer is rated below whichever side the fixture happened to name".
    mean_attack = float(np.mean(list(fit.attack.values())))
    mean_concede = float(np.mean(list(fit.concede.values())))
    assert fit.promoted_attack < mean_attack, "a promoted side must not attack like an average one"
    assert fit.promoted_concede > mean_concede, "...nor defend like one"


def test_lambdas_are_clipped_to_a_sane_range():
    fit = DixonColesFit(
        attack={1: 50.0}, concede={2: 50.0},
        intercept_home=10.0, intercept_away=10.0, rho=0.0,
    )
    lam_home, lam_away = fit.lambdas(1, 2)
    assert 0.0 < lam_home <= 6.0
    assert 0.0 < lam_away <= 6.0


def test_tau_only_touches_the_four_low_scorelines():
    """Dixon and Coles' correction applies to 0-0, 1-0, 0-1 and 1-1 and leaves
    everything else exactly as independent Poissons had it."""
    home = np.array([0, 1, 0, 1, 2, 3])
    away = np.array([0, 0, 1, 1, 1, 2])
    lam_h = np.full(6, 1.4)
    lam_a = np.full(6, 1.1)
    out = _tau(home, away, lam_h, lam_a, rho=0.1)

    assert not np.isclose(out[:4], 1.0).any(), "the four low scorelines must be adjusted"
    assert np.allclose(out[4:], 1.0), "everything else must be left alone"


def test_tau_is_neutral_at_rho_zero():
    home = np.array([0, 1, 0, 1, 2])
    away = np.array([0, 0, 1, 1, 3])
    out = _tau(home, away, np.full(5, 1.3), np.full(5, 1.0), rho=0.0)
    assert np.allclose(out, 1.0)


def test_fitting_nothing_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        fit_dixon_coles(pd.DataFrame())


def test_recent_results_outweigh_old_ones():
    """Recency has to actually bite, or the fit is just a five-year average and
    a team that has transformed is mispriced all season."""
    rows = []
    for gw in range(1, 20):  # long ago: 100 thrashes 200
        rows.append({"season": "2021-22", "gameweek": gw, "home_code": 100,
                     "away_code": 200, "home_goals": 5, "away_goals": 0})
    for gw in range(1, 20):  # recently: the reverse
        rows.append({"season": "2025-26", "gameweek": gw, "home_code": 100,
                     "away_code": 200, "home_goals": 0, "away_goals": 5})
    fit = fit_dixon_coles(pd.DataFrame(rows), half_life_gws=20.0)
    assert fit.attack[200] > fit.attack[100]
