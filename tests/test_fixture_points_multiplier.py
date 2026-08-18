"""Odds-anchored, position-aware fixture adjustment for the cold start.

2026-08-18. The cold start scored fixtures with ``fixture_multiplier``, a
strength ratio its own docstring calls "a deliberately simple, bounded
scaffold: the real fixture-conditioning comes from the component models (P3/P5
anchor on odds-implied team goals)". The cold start never reaches those models,
so the scaffold became the permanent answer for the season's biggest decision --
and it is far too flat to be that.

Measured on GW1 2026-27: the scaffold spans 0.89 to 1.10 across all twenty
sides; the bookmakers span 0.46 to 1.72. Six of GW1's ten fixtures had real
odds sitting unused in `fixture_odds` on the day the initial squad was built.
"""

from __future__ import annotations

import pytest

from projection.fixture_adjust import (
    APPEARANCE_POINTS,
    fixture_points_multiplier,
)

# A comfortably-above-appearance player, so the floor does not dominate.
PPA = 6.0

# Arsenal at home to a promoted side, from the real GW1 odds.
BIG_HOME = dict(lam_for=2.61, lam_against=0.46, is_home=True)
# The reverse fixture, from the same prices.
HARD_AWAY = dict(lam_for=0.46, lam_against=2.61, is_home=False)


def _mult(position, **kw):
    return fixture_points_multiplier(per_appearance_points=PPA, position=position, **kw)


def test_a_good_fixture_lifts_every_position_and_a_bad_one_lowers_them():
    for pos in ("GKP", "DEF", "MID", "FWD"):
        assert _mult(pos, **BIG_HOME) > 1.0
        assert _mult(pos, **HARD_AWAY) < 1.0


def test_the_clean_sheet_channel_is_what_moves_defenders():
    """The defect this exists to prevent: scaling a defender by his own team's
    expected GOALS. A side that scores freely and concedes freely is a good
    fixture for its forward and a bad one for its keeper, and applying one
    attacking multiplier to both had the engine captaining a defender off a
    number that belonged to the attack.
    """
    leaky = dict(lam_for=2.2, lam_against=2.0, is_home=True)
    assert _mult("FWD", **leaky) > 1.0, "the forward of a free-scoring side gains"
    assert _mult("GKP", **leaky) < 1.0, "its keeper does not"
    # Ordered by how much of each position's return depends on a clean sheet.
    assert _mult("GKP", **leaky) < _mult("DEF", **leaky) < _mult("MID", **leaky)


def test_a_shut_out_opponent_helps_the_keeper_most():
    assert _mult("GKP", **BIG_HOME) > _mult("FWD", **BIG_HOME)


def test_appearance_points_are_never_scaled():
    """A player banks appearance points for turning out regardless of opponent.
    Scaling a whole average by 1.7 credits 3.4 points for showing up; scaling it
    by 0.4 implies a player can be worth almost nothing for playing 90 minutes.
    """
    # Someone who earns nothing beyond turning up is unaffected by any fixture.
    for fixture in (BIG_HOME, HARD_AWAY):
        flat = fixture_points_multiplier(
            per_appearance_points=APPEARANCE_POINTS, position="MID", **fixture
        )
        assert flat == pytest.approx(1.0)

    # And a hard fixture can never take a player below their appearance floor.
    worst = fixture_points_multiplier(
        lam_for=0.05, lam_against=6.0, is_home=False,
        per_appearance_points=PPA, position="FWD",
    )
    assert worst * PPA >= APPEARANCE_POINTS


def test_cheap_players_are_less_fixture_sensitive_than_premiums():
    """More of a low scorer's average is the fixture-independent appearance
    floor, so the same fixture moves them proportionally less. That is a real
    property, not an artefact."""
    premium = fixture_points_multiplier(
        per_appearance_points=8.0, position="FWD", **BIG_HOME
    )
    cheap = fixture_points_multiplier(
        per_appearance_points=2.5, position="FWD", **BIG_HOME
    )
    assert premium > cheap


def test_missing_odds_degrade_to_neutral_rather_than_guessing():
    assert fixture_points_multiplier(
        None, None, True, PPA, "MID"
    ) == pytest.approx(1.0)


def test_extreme_prices_are_clamped():
    """A freak quote must not treble a projection on its own."""
    m = fixture_points_multiplier(
        lam_for=25.0, lam_against=0.001, is_home=True,
        per_appearance_points=PPA, position="FWD",
    )
    assert m * PPA < 4 * PPA
