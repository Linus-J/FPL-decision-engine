"""P7 DefCon component."""

from __future__ import annotations

import numpy as np
import pytest

from config.strategy import DEFCON
from projection import defcon as DC


def test_thresholds_by_position():
    assert DC.defcon_threshold("DEF") == DEFCON.def_threshold      # 10 (CBIT)
    assert DC.defcon_threshold("MID") == DEFCON.mid_fwd_threshold  # 12 (CBIRT)
    assert DC.defcon_threshold("FWD") == DEFCON.mid_fwd_threshold
    assert DC.defcon_threshold("GK") is None                       # no DefCon


def test_p_hits_threshold_monotonic():
    # a higher expected action count clears the threshold more often
    assert DC.p_hits_threshold(12, 10) > DC.p_hits_threshold(8, 10)
    assert 0.0 <= DC.p_hits_threshold(10, 10) <= 1.0


def test_expected_defcon_points():
    # a high-CBIT defender expects close to the 2-pt award
    high = DC.expected_defcon_points(14, "DEF", 1.0)
    low = DC.expected_defcon_points(5, "DEF", 1.0)
    assert high > low
    assert high <= DEFCON.points
    assert DC.expected_defcon_points(20, "GK", 1.0) == 0.0     # GK never
    assert DC.expected_defcon_points(14, "DEF", 0.0) == 0.0    # no minutes


def test_sample_defcon_points_mean_matches_expectation():
    rng = np.random.default_rng(7)
    draws = [DC.sample_defcon_points(rng, 11.0, "DEF", True) for _ in range(30000)]
    assert np.mean(draws) == pytest.approx(DC.expected_defcon_points(11.0, "DEF", 1.0), abs=0.05)
    # not played → 0
    assert DC.sample_defcon_points(rng, 20.0, "DEF", False) == 0
