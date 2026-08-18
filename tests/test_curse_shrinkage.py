"""Optimiser's-curse shrinkage (projection/assemble.py::apply_curse_shrinkage),
added 2026-07-28 after the walk-forward gate showed v2's own predicted-vs-
actual points had a +12.6 pt/GW bias with only 0.33 correlation, traced to
squad-building/starting-XI/captaincy selecting from the top of the
projected-xpts distribution with zero bias correction (only the weekly
transfer ILP had one, via the now-superseded transfer_variance_penalty).

A first version shrunk each player by an amount proportional to their own
xpts_var relative to the group's between-player variance (textbook
James-Stein empirical-Bayes shrinkage) — reverted after a live gate run
showed it collapsing `predicted` to a near-constant value regardless of
squad, because xpts_var is OUTCOME variance (how spiky a player's returns
are), not ESTIMATION uncertainty about the mean, and the two are not
interchangeable here. These tests cover the simpler, uniform-strength
replacement."""

from __future__ import annotations

import pandas as pd
import pytest

from projection.assemble import CURSE_SHRINKAGE_STRENGTH, apply_curse_shrinkage


def _players(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"id": pid, "position": pos} for pid, pos in rows])


def test_shrinks_toward_the_group_mean_by_the_fixed_strength():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 3.0},
        {"player_id": 2, "gameweek": 10, "xpts": 3.0, "xpts_mean": 3.0, "xpts_var": 1.0},
        {"player_id": 3, "gameweek": 10, "xpts": 1.0, "xpts_mean": 1.0, "xpts_var": 1.0},
    ])
    players = _players([(1, "MID"), (2, "MID"), (3, "MID")])
    out = apply_curse_shrinkage(projections, players)

    group_mean = projections["xpts"].mean()  # 4.0
    expected_top = group_mean + (1.0 - CURSE_SHRINKAGE_STRENGTH) * (8.0 - group_mean)
    shrunk_top = out[out["player_id"] == 1]["xpts"].iloc[0]
    assert shrunk_top == pytest.approx(expected_top)
    assert shrunk_top < 8.0  # pulled down toward the mean


def test_xpts_var_does_not_affect_shrinkage_amount():
    # Two players with identical raw xpts but wildly different xpts_var
    # (outcome variance, e.g. a spiky explosive forward vs a steady
    # grinder) must shrink by the SAME amount -- xpts_var is not an
    # estimation-uncertainty signal here, see the module docstring.
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 0.1},
        {"player_id": 2, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 50.0},
        {"player_id": 3, "gameweek": 10, "xpts": 1.0, "xpts_mean": 1.0, "xpts_var": 1.0},
    ])
    players = _players([(1, "MID"), (2, "MID"), (3, "MID")])
    out = apply_curse_shrinkage(projections, players)
    a = out[out["player_id"] == 1]["xpts"].iloc[0]
    b = out[out["player_id"] == 2]["xpts"].iloc[0]
    assert a == b


def test_xpts_raw_preserves_original_value():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 3.0},
        {"player_id": 2, "gameweek": 10, "xpts": 3.0, "xpts_mean": 3.0, "xpts_var": 1.0},
        {"player_id": 3, "gameweek": 10, "xpts": 2.5, "xpts_mean": 2.5, "xpts_var": 1.0},
    ])
    players = _players([(1, "MID"), (2, "MID"), (3, "MID")])
    out = apply_curse_shrinkage(projections, players)
    assert out.set_index("player_id")["xpts_raw"].to_dict() == {1: 8.0, 2: 3.0, 3: 2.5}
    assert (out["xpts"] != out["xpts_raw"]).any()


def test_xpts_mean_and_xpts_var_are_left_untouched():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 3.0},
        {"player_id": 2, "gameweek": 10, "xpts": 3.0, "xpts_mean": 3.0, "xpts_var": 1.0},
        {"player_id": 3, "gameweek": 10, "xpts": 2.5, "xpts_mean": 2.5, "xpts_var": 1.0},
    ])
    players = _players([(1, "MID"), (2, "MID"), (3, "MID")])
    out = apply_curse_shrinkage(projections, players)
    assert out["xpts_mean"].tolist() == [8.0, 3.0, 2.5]
    assert out["xpts_var"].tolist() == [3.0, 1.0, 1.0]


def test_group_below_minimum_size_is_left_unshrunk():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 5.0},
        {"player_id": 2, "gameweek": 10, "xpts": 2.0, "xpts_mean": 2.0, "xpts_var": 5.0},
    ])
    players = _players([(1, "MID"), (2, "MID")])
    out = apply_curse_shrinkage(projections, players)
    assert out["xpts"].tolist() == [8.0, 2.0]


def test_identical_group_values_is_a_no_op():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 4.0, "xpts_mean": 4.0, "xpts_var": 2.0},
        {"player_id": 2, "gameweek": 10, "xpts": 4.0, "xpts_mean": 4.0, "xpts_var": 2.0},
        {"player_id": 3, "gameweek": 10, "xpts": 4.0, "xpts_mean": 4.0, "xpts_var": 2.0},
    ])
    players = _players([(1, "MID"), (2, "MID"), (3, "MID")])
    out = apply_curse_shrinkage(projections, players)
    assert out["xpts"].tolist() == [4.0, 4.0, 4.0]


def test_positions_are_shrunk_independently():
    # A FWD's raw xpts (8.0) sits well above the MID group's mean (~2.75)
    # but should shrink toward the FWD group's own mean, not get pulled
    # down by unrelated MIDs.
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 3.0},
        {"player_id": 2, "gameweek": 10, "xpts": 7.0, "xpts_mean": 7.0, "xpts_var": 3.0},
        {"player_id": 3, "gameweek": 10, "xpts": 6.0, "xpts_mean": 6.0, "xpts_var": 3.0},
        {"player_id": 10, "gameweek": 10, "xpts": 3.0, "xpts_mean": 3.0, "xpts_var": 1.0},
        {"player_id": 11, "gameweek": 10, "xpts": 2.5, "xpts_mean": 2.5, "xpts_var": 1.0},
        {"player_id": 12, "gameweek": 10, "xpts": 2.0, "xpts_mean": 2.0, "xpts_var": 1.0},
    ])
    players = _players([(1, "FWD"), (2, "FWD"), (3, "FWD"), (10, "MID"), (11, "MID"), (12, "MID")])
    out = apply_curse_shrinkage(projections, players)
    fwd_shrunk = out[out["player_id"] == 1]["xpts"].iloc[0]
    assert fwd_shrunk > 5.0  # nowhere near the MID group's ~2.5 mean


def test_gameweeks_are_shrunk_independently():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 3.0},
        {"player_id": 2, "gameweek": 10, "xpts": 3.0, "xpts_mean": 3.0, "xpts_var": 1.0},
        {"player_id": 3, "gameweek": 10, "xpts": 2.5, "xpts_mean": 2.5, "xpts_var": 1.0},
        {"player_id": 1, "gameweek": 11, "xpts": 1.0, "xpts_mean": 1.0, "xpts_var": 3.0},
        {"player_id": 2, "gameweek": 11, "xpts": 3.0, "xpts_mean": 3.0, "xpts_var": 1.0},
        {"player_id": 3, "gameweek": 11, "xpts": 2.5, "xpts_mean": 2.5, "xpts_var": 1.0},
    ])
    players = _players([(1, "MID"), (2, "MID"), (3, "MID")])
    out = apply_curse_shrinkage(projections, players)
    gw10 = out[(out["player_id"] == 1) & (out["gameweek"] == 10)]["xpts"].iloc[0]
    gw11 = out[(out["player_id"] == 1) & (out["gameweek"] == 11)]["xpts"].iloc[0]
    assert gw10 > gw11  # unaffected by the other gameweek's very different pool


def test_missing_position_column_is_a_no_op():
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0, "xpts_mean": 8.0, "xpts_var": 3.0},
    ])
    players = pd.DataFrame([{"id": 1, "status": "a"}])
    out = apply_curse_shrinkage(projections, players)
    assert "xpts_raw" not in out.columns
    assert out["xpts"].tolist() == [8.0]


def test_empty_projections_is_a_no_op():
    projections = pd.DataFrame(columns=["player_id", "gameweek", "xpts", "xpts_mean", "xpts_var"])
    players = _players([(1, "MID")])
    out = apply_curse_shrinkage(projections, players)
    assert out.empty


def test_a_zeroed_player_is_never_resurrected_by_shrinkage():
    """Regression, 2026-08-18 (engine review §3).

    A zero here is not a low estimate — it is a statement that the player will
    not feature. Unavailable players are zeroed and confirmed departures
    discounted to 0.0 BEFORE shrinkage runs. Shrinking them toward a positive
    group mean handed them points back: at the default strength a zeroed
    player in a group averaging 4.0 came out at 0.60 xpts, so a leaver the
    departure gate had just eliminated became selectable again.
    """
    projections = pd.DataFrame([
        {"player_id": 1, "gameweek": 10, "xpts": 8.0},
        {"player_id": 2, "gameweek": 10, "xpts": 3.0},
        {"player_id": 3, "gameweek": 10, "xpts": 1.0},
        {"player_id": 4, "gameweek": 10, "xpts": 0.0},   # departed / unavailable
    ])
    players = _players([(1, "MID"), (2, "MID"), (3, "MID"), (4, "MID")])
    out = apply_curse_shrinkage(projections, players)

    assert out[out["player_id"] == 4]["xpts"].iloc[0] == 0.0


def test_the_group_mean_ignores_players_who_will_not_feature():
    """Regression, 2026-08-18 (engine review §3), the subtler half.

    Non-participants used to drag the group mean down, so how hard a real
    player was shrunk depended on how many irrelevant players happened to be
    in that week's frame — the correction for a premium should not move
    because a fringe player got injured.

    Both frames below hold the same three real players; the second merely adds
    non-participants. The shrunk values must be identical.
    """
    real = [
        {"player_id": 1, "gameweek": 10, "xpts": 8.0},
        {"player_id": 2, "gameweek": 10, "xpts": 3.0},
        {"player_id": 3, "gameweek": 10, "xpts": 1.0},
    ]
    padding = [
        {"player_id": pid, "gameweek": 10, "xpts": 0.0} for pid in range(4, 30)
    ]
    roster = [(pid, "MID") for pid in range(1, 30)]

    without = apply_curse_shrinkage(pd.DataFrame(real), _players(roster))
    with_padding = apply_curse_shrinkage(pd.DataFrame(real + padding), _players(roster))

    for pid in (1, 2, 3):
        assert without[without["player_id"] == pid]["xpts"].iloc[0] == pytest.approx(
            with_padding[with_padding["player_id"] == pid]["xpts"].iloc[0]
        )
