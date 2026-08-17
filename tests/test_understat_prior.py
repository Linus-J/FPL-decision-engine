"""understat_prior.py — expected-goals metrics for the prior-league tier.

The live ingest is network-bound and excluded from coverage; what is tested
here is the arithmetic and, more importantly, the guard that stops one
player's xG being written onto another.
"""

from __future__ import annotations

from data.ingestors.understat_prior import (
    MIN_MINUTES_FOR_CHECK,
    UNDERSTAT_PRIOR_LEAGUES,
    minutes_agree,
    per90,
)


def test_per90_converts_season_totals():
    npxg90, xa90 = per90(minutes=1800, np_xg=18.0, xa=9.0)
    assert npxg90 == 0.9
    assert xa90 == 0.45


def test_per90_zero_minutes_is_zero_not_a_crash():
    assert per90(0, 5.0, 5.0) == (0.0, 0.0)


def test_minutes_agreement_accepts_the_same_player():
    """The two sources time substitutions slightly differently; that must not
    reject a genuine match."""
    assert minutes_agree(1479, 1479)
    assert minutes_agree(1479, 1450)
    assert minutes_agree(2000, 1800)


def test_minutes_agreement_rejects_a_probable_name_collision():
    """The point of the guard. Matching Understat names onto FPL codes is a
    new entity-resolution surface, and this project has already had a
    set-piece duty land on the wrong player who shared a first name. Two
    sources disagreeing wildly about how long someone played means the name
    match cannot be trusted to carry xG.
    """
    assert not minutes_agree(2500, 300)
    assert not minutes_agree(1800, 0)


def test_low_minute_players_skip_the_check_rather_than_being_rejected():
    """Below the threshold the relative test is meaningless -- 10 vs 25
    minutes is a 150% gap between two players who barely featured. Those rows
    carry no signal and are excluded from the cold start anyway."""
    assert minutes_agree(MIN_MINUTES_FOR_CHECK - 1, 5)
    assert minutes_agree(0, 0)


def test_championship_is_not_claimed_as_covered():
    """Understat does not cover the Championship, so those rows must keep
    falling back to npg90 rather than silently appearing to have xG."""
    assert "ENG-Championship" not in UNDERSTAT_PRIOR_LEAGUES
    assert set(UNDERSTAT_PRIOR_LEAGUES) == {
        "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1",
    }
