import dataclasses

import pytest

from config.strategy import CHIP_TIMING, OPTIMISER, assert_horizons_consistent


def test_comparison_ships_live_with_bars_seeded_from_the_legacy_thresholds():
    assert CHIP_TIMING.chip_comparison_enabled is True
    assert CHIP_TIMING.chip_comparison_horizon_gws == CHIP_TIMING.wildcard_eval_horizon_gws
    assert CHIP_TIMING.free_hit_comparison_margin == CHIP_TIMING.free_hit_single_gw_gain_threshold
    assert CHIP_TIMING.wildcard_comparison_margin == CHIP_TIMING.wildcard_pts_gain_threshold


def test_comparison_horizon_is_registered_as_a_consumer():
    """A horizon longer than the persisted projection frame must RAISE.

    Registration is the whole point: an unregistered consumer silently gets
    fewer gameweeks than its bar assumes.
    """
    too_long = dataclasses.replace(
        CHIP_TIMING, chip_comparison_horizon_gws=OPTIMISER.projection_horizon_gws + 1
    )
    with pytest.raises(ValueError, match="chip_comparison_horizon_gws"):
        assert_horizons_consistent(OPTIMISER, too_long)
