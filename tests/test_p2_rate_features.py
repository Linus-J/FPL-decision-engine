"""Phase-2 P2 — rate-feature contract (defect D4).

Model features must be rates, not season-cumulative volume, and identical on the
train/serve paths. These lock that the banned cumulative/proxy columns are gone
from every component's FEATURE_COLS and can't creep back.

points_model.py's own guard was removed 2026-08-01 along with the file itself
(confirmed dead in the live and backtest paths -- superseded by
projection/assemble.py's P10 MC assembly). minutes_model.py is still live
(assemble.py's 3-way minutes-band predictor), so its guard stays.
"""

from __future__ import annotations

import pytest

from projection import minutes_model
from projection.features import CUMULATIVE_BANNED_FEATURES, assert_rate_only


def test_minutes_model_feature_cols_are_rate_only():
    assert CUMULATIVE_BANNED_FEATURES.isdisjoint(minutes_model.FEATURE_COLS)


def test_rolling_rate_features_still_present():
    # the rate signal that replaced cumulative ICT/form must still be there
    assert "avg_minutes_5gw" in minutes_model.FEATURE_COLS


def test_assert_rate_only_guard():
    assert_rate_only(["avg_xg_5gw", "now_cost", "pos_MID"])   # clean → no raise
    for banned in ("ict_index", "influence", "creativity", "threat", "form"):
        with pytest.raises(ValueError, match="banned"):
            assert_rate_only(["avg_xg_5gw", banned])
