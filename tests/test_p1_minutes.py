"""Phase-2 P1 — 3-way minutes model helpers + deterministic availability override.

Pure logic (network-free): band bucketing, the M1 availability override (incl.
the non-'a' injection the aggregate backfill can't exercise), band-vector mapping
tolerant of an absent class, and expected appearance points. The trained model is
smoke-tested on the live DB separately.
"""

from __future__ import annotations

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
