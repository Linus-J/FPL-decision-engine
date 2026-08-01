"""P11 — cross-league translation-factor calibration (config lookup + pure
hold-out/factor-computation math). No live scrape needed."""

from __future__ import annotations

from config.strategy import PRIOR_LEAGUE


def test_prior_league_rules_covers_all_five_leagues():
    leagues = ["ENG-Championship", "ESP-La Liga", "ITA-Serie A",
               "GER-Bundesliga", "FRA-Ligue 1"]
    for league in leagues:
        assert PRIOR_LEAGUE.translation_factor(league) > 0
        assert PRIOR_LEAGUE.translation_variance(league) > 0


def test_championship_factor_discounted_below_top5():
    # the plan's own literature-style prior: Championship output doesn't
    # translate 1:1 to the PL, top-5 leagues roughly do.
    assert PRIOR_LEAGUE.translation_factor("ENG-Championship") < 1.0
    assert PRIOR_LEAGUE.translation_factor("ESP-La Liga") == 1.0
