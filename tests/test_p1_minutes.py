"""Phase-2 P1 — 3-way minutes model helpers + deterministic availability override.

Pure logic (network-free): band bucketing, the M1 availability override (incl.
the non-'a' injection the aggregate backfill can't exercise), band-vector mapping
tolerant of an absent class, and expected appearance points. The trained model is
smoke-tested on the live DB separately.
"""

from __future__ import annotations

import pandas as pd
import pytest

from projection import minutes_model as mm


def test_minutes_band_bucketing():
    assert mm.minutes_band(0) == mm.BAND_DNP
    assert mm.minutes_band(None) == mm.BAND_DNP
    assert mm.minutes_band(1) == mm.BAND_CAMEO
    assert mm.minutes_band(59) == mm.BAND_CAMEO
    assert mm.minutes_band(60) == mm.BAND_START
    assert mm.minutes_band(90) == mm.BAND_START


def test_bands_from_proba_handles_absent_class():
    # classifier only saw classes 0 and 2 → band 1 must fill to 0
    assert mm._bands_from_proba([0.3, 0.7], [0, 2]) == (0.3, 0.0, 0.7)
    assert mm._bands_from_proba([0.2, 0.5, 0.3], [0, 1, 2]) == (0.2, 0.5, 0.3)


def test_availability_override_injured_is_certain_dnp():
    # M1: the non-'a' path the constant-'a' backfill can't test
    for st in ("i", "u", "s"):
        assert mm.apply_availability_override(0.1, 0.2, 0.7, st, None) == (1.0, 0.0, 0.0)


def test_availability_override_doubtful_scales_by_cop():
    # doubtful, 50% chance of playing → half the playing mass moves to DNP
    p0, p1, p2 = mm.apply_availability_override(0.2, 0.1, 0.7, "d", 0.5)
    assert p1 == pytest.approx(0.05)
    assert p2 == pytest.approx(0.35)
    assert p0 == pytest.approx(1 - 0.5 * (0.1 + 0.7))
    assert sum((p0, p1, p2)) == pytest.approx(1.0)


def test_availability_override_available_unchanged():
    assert mm.apply_availability_override(0.2, 0.1, 0.7, "a", None) == (0.2, 0.1, 0.7)
    assert mm.apply_availability_override(0.2, 0.1, 0.7, None, None) == (0.2, 0.1, 0.7)


def test_expected_appearance_points():
    assert mm.expected_appearance_points(0.1, 0.7) == pytest.approx(1.5)   # 0.1*1 + 0.7*2
    assert mm.expected_appearance_points(0.0, 1.0) == pytest.approx(2.0)


def test_availability_features_removed_from_model():
    # M1: is_available/cop_next are no longer learned features
    assert "is_available" not in mm.FEATURE_COLS
    assert "cop_next" not in mm.FEATURE_COLS


# --- Real bug found 2026-07-28 (user-driven squad-trace review): status is
# constant ('a') throughout backfilled history, so apply_availability_override
# never fires during backtesting -- genuinely-injured players (real minutes=0
# for several straight gameweeks) kept getting captained/started. dnp_streak
# is a real, leakage-free signal from already-played history instead. -------


def test_trailing_dnp_streak_counts_consecutive_zeros():
    minutes = pd.Series([90, 90, 0, 0, 0, 90, 45])
    streak = mm._trailing_dnp_streak(minutes)
    assert streak.tolist() == [0, 0, 1, 2, 3, 0, 0]


def test_trailing_dnp_streak_resets_per_player_group():
    df = pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 2],
        "season": ["2025-26"] * 6,
        "minutes": [90, 0, 0, 0, 0, 90],
    })
    streak = df.groupby(["player_id", "season"])["minutes"].transform(mm._trailing_dnp_streak)
    assert streak.tolist() == [0, 1, 2, 1, 2, 0]


def test_apply_recent_absence_override_no_streak_unchanged():
    assert mm.apply_recent_absence_override(0.1, 0.2, 0.7, 0) == (0.1, 0.2, 0.7)


def test_apply_recent_absence_override_one_blank_halves_playing_mass():
    p0, p1, p2 = mm.apply_recent_absence_override(0.1, 0.2, 0.7, 1)
    assert p1 == pytest.approx(0.1)
    assert p2 == pytest.approx(0.35)
    assert sum((p0, p1, p2)) == pytest.approx(1.0)


def test_apply_recent_absence_override_two_plus_blanks_heavily_discounts():
    p0, p1, p2 = mm.apply_recent_absence_override(0.1, 0.2, 0.7, 2)
    assert p2 == pytest.approx(0.7 * mm._DNP_STREAK_RETENTION_2PLUS)
    assert p0 > 0.8  # the real Ekitiké case: should be treated as very unlikely to start
    # a longer streak (e.g. 5 straight blanks) is discounted exactly as hard,
    # not progressively harder -- 2+ is already the floor.
    assert mm.apply_recent_absence_override(0.1, 0.2, 0.7, 5) == (p0, p1, p2)
