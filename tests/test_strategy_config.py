"""Tests for OptimiserConfig fields."""


def test_cold_start_lookahead_gws_default():
    from config.strategy import OptimiserConfig

    assert OptimiserConfig().cold_start_lookahead_gws == 5


def test_cold_start_lookahead_gws_overridable():
    from config.strategy import OptimiserConfig

    assert OptimiserConfig(cold_start_lookahead_gws=1).cold_start_lookahead_gws == 1
