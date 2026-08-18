"""Hand-entered ceilings on start probability.

The minutes model projects from a player's own history, so a summer signing
carries his previous club's status wholesale: Elliot Anderson arrived at
Manchester City on 37 Nottingham Forest starts and was handed
``start_probability = 0.97`` with no Manchester City minutes in evidence
anywhere. Competition for a place is a fact about a squad, not about a
player's record, and nothing in the pipeline can see it.

Deliberately not a blanket new-signing rule, which the data does not support:
across 1,149 player-seasons, prior-season regulars who changed club retained
95.6-97.2% of the minutes share stayers retained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimiser.rotation_risk import apply_rotation_risk


def _projections() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": 1, "gameweek": gw, "start_probability": 0.97,
         "xpts": 5.0, "xpts_var": 4.0, "upside": 2.0, "downside": 1.5}
        for gw in (1, 2, 3)
    ] + [
        {"player_id": 2, "gameweek": gw, "start_probability": 0.60,
         "xpts": 3.0, "xpts_var": 2.0, "upside": 1.0, "downside": 0.8}
        for gw in (1, 2, 3)
    ])


def test_cap_lowers_start_probability_and_the_points_that_follow():
    out = apply_rotation_risk(_projections(), {1: 0.70})
    capped = out[out["player_id"] == 1]

    assert np.allclose(capped["start_probability"], 0.70)
    # Expected points are near-proportional to playing at all, so they must
    # move with it or the frame contradicts itself and the optimiser reads
    # the half that did not move.
    assert np.allclose(capped["xpts"], 5.0 * 0.70 / 0.97)


def test_the_spread_scales_with_the_points():
    """A week the player does not feature contributes neither points nor
    spread, so the risk columns take the same scaling the mean does."""
    out = apply_rotation_risk(_projections(), {1: 0.70})
    capped = out[out["player_id"] == 1].iloc[0]
    ratio = 0.70 / 0.97

    assert capped["xpts_var"] == pytest.approx(4.0 * ratio)
    assert capped["upside"] == pytest.approx(2.0 * ratio)
    assert capped["downside"] == pytest.approx(1.5 * ratio)


def test_a_cap_only_ever_lowers():
    """A ceiling above the model's own estimate must do nothing. An override
    that could promote a player would turn a safety mechanism into a way to
    talk the engine into a hunch."""
    out = apply_rotation_risk(_projections(), {2: 0.90})
    untouched = out[out["player_id"] == 2]

    assert np.allclose(untouched["start_probability"], 0.60)
    assert np.allclose(untouched["xpts"], 3.0)


def test_players_without_an_override_are_untouched():
    out = apply_rotation_risk(_projections(), {1: 0.70})
    other = out[out["player_id"] == 2]

    assert np.allclose(other["start_probability"], 0.60)
    assert np.allclose(other["xpts"], 3.0)


def test_no_overrides_is_a_no_op():
    frame = _projections()
    assert apply_rotation_risk(frame, {}) is frame


def test_an_empty_frame_is_a_no_op():
    empty = pd.DataFrame()
    assert apply_rotation_risk(empty, {1: 0.5}).empty


def test_an_unknown_player_id_is_ignored_not_an_error():
    """The override file is hand-edited and outlives any given squad; a stale
    entry for a departed player must not take the pipeline down."""
    out = apply_rotation_risk(_projections(), {9999: 0.5})
    assert np.allclose(out["xpts"], _projections()["xpts"])


def test_a_frame_without_start_probability_is_left_alone():
    frame = pd.DataFrame([{"player_id": 1, "gameweek": 1, "xpts": 5.0}])
    out = apply_rotation_risk(frame, {1: 0.5})
    assert np.allclose(out["xpts"], 5.0)


def test_a_zero_start_probability_does_not_divide_by_zero():
    """An already-zeroed player — unavailable, or a confirmed departure — must
    stay at zero rather than producing an infinity."""
    frame = pd.DataFrame([{
        "player_id": 1, "gameweek": 1, "start_probability": 0.0, "xpts": 0.0,
    }])
    out = apply_rotation_risk(frame, {1: 0.5})
    assert out["xpts"].notna().all()
    assert out["start_probability"].iloc[0] == pytest.approx(0.0)


def test_cap_is_applied_across_every_gameweek_of_the_horizon():
    out = apply_rotation_risk(_projections(), {1: 0.50})
    capped = out[out["player_id"] == 1]
    assert len(capped) == 3
    assert np.allclose(capped["start_probability"], 0.50)
