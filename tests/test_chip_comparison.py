import dataclasses

import pandas as pd
import pytest

from config.strategy import CHIP_TIMING, OPTIMISER, assert_horizons_consistent
from optimiser.chip_comparison import build_no_chip_option
from optimiser.transfers import TransferPlan


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


def _projections(player_ids, gws, xpts=5.0):
    rows = []
    for gw in gws:
        for pid in player_ids:
            rows.append({"player_id": pid, "gameweek": gw, "xpts": xpts})
    return pd.DataFrame(rows)


def test_no_chip_option_is_base_plus_its_plan_gain(monkeypatch):
    """horizon_xpts is a TOTAL, so every option is on one scale."""
    squad = list(range(1, 16))
    proj = _projections(squad, [3, 4, 5], xpts=2.0)
    plan = TransferPlan(
        transfers_in=[], transfers_out=[], hits_taken=0, xpts_gain=9.0, net_xpts_gain=9.0
    )
    monkeypatch.setattr(
        "optimiser.chip_comparison.evaluate_transfers", lambda *a, **k: plan
    )
    option = build_no_chip_option(
        current_squad_ids=squad,
        projections=proj,
        players=pd.DataFrame({"id": squad}),
        free_transfers=1,
        horizon=3,
    )
    # 24 xpts a week (11 starters at 2.0 = 22, plus a 2.0 captain bonus),
    # across 3 gameweeks = 72, plus the plan's own 9.0 gain.
    assert option.horizon_xpts == pytest.approx(72.0 + 9.0)
    assert option.chip is None
    assert option.plan is plan


def test_a_failed_solve_yields_no_option_rather_than_a_zero(monkeypatch):
    """Scoring an unknown as 0 would rank it below every real option."""
    def _boom(*a, **k):
        raise ValueError("squad build failed — infeasible")

    monkeypatch.setattr("optimiser.chip_comparison.evaluate_transfers", _boom)
    option = build_no_chip_option(
        current_squad_ids=list(range(1, 16)),
        projections=_projections(list(range(1, 16)), [3]),
        players=pd.DataFrame({"id": list(range(1, 16))}),
        free_transfers=1,
        horizon=1,
    )
    assert option is None
