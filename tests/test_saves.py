"""P6 saves component — fixture-anchored GK saves."""

from __future__ import annotations

import numpy as np
import pytest

from projection import saves as SV


def test_expected_saves_scales_with_opponent_and_minutes():
    # more attacking opponent → more saves; no minutes → none
    assert SV.expected_saves(1.5, 1.0) > SV.expected_saves(0.5, 1.0) > 0
    assert SV.expected_saves(1.5, 0.0) == 0.0
    # λ=1.2, full play: SoT=4.0, saves=4.0*0.7=2.8
    assert SV.expected_saves(1.2, 1.0) == pytest.approx(2.8, abs=1e-6)


def test_expected_save_points_is_one_per_three():
    # negligible for tiny volume, rising with load; strictly below exp_saves/3-ish
    assert SV.expected_save_points(0.3) < 0.2
    assert SV.expected_save_points(3.0) > SV.expected_save_points(1.0) > 0


def test_sample_save_points_mean_matches_expectation():
    rng = np.random.default_rng(3)
    exp = SV.expected_saves(1.4, 1.0)
    draws = [SV.sample_save_points(rng, exp) for _ in range(30000)]
    assert np.mean(draws) == pytest.approx(SV.expected_save_points(exp), abs=0.03)
