"""Regression test for a real bug found in data/ingestors/fpl_api.py during
the 2026-07-28 data-completeness audit: a genuine double-gameweek player's
FPL history has two entries with the same ``round``, and the old
``ingest_player_history`` wrote each with its own on_conflict_do_update,
so the second fixture's stats silently overwrote (rather than summed with)
the first -- the same P12 double-gameweek bug class already fixed in
projection/assemble.py, reproduced unfixed in this ingestor."""

from __future__ import annotations

from data.ingestors.fpl_api import _accumulate_gw_history


def test_single_gameweek_entry_passes_through():
    history = [{"round": 10, "total_points": 6, "minutes": 90, "selected": 500, "value": 55}]
    result = _accumulate_gw_history(history)
    assert result[10]["total_points"] == 6
    assert result[10]["minutes"] == 90
    assert result[10]["selected"] == 500
    assert result[10]["value"] == 5.5


def test_double_gameweek_sums_cumulative_stats_not_overwrites():
    history = [
        {"round": 12, "total_points": 6, "minutes": 90, "goals_scored": 1, "bps": 30,
         "selected": 500, "value": 55},
        {"round": 12, "total_points": 2, "minutes": 90, "goals_scored": 0, "bps": 10,
         "selected": 500, "value": 55},
    ]
    result = _accumulate_gw_history(history)
    assert result[12]["total_points"] == 8
    assert result[12]["minutes"] == 180
    assert result[12]["goals_scored"] == 1
    assert result[12]["bps"] == 40


def test_double_gameweek_keeps_latest_snapshot_fields_not_summed():
    # selected/value are point-in-time snapshots, not per-fixture stats --
    # must NOT be summed across the two DGW entries.
    history = [
        {"round": 12, "total_points": 6, "selected": 500, "value": 55},
        {"round": 12, "total_points": 2, "selected": 510, "value": 56},
    ]
    result = _accumulate_gw_history(history)
    assert result[12]["selected"] == 510
    assert result[12]["value"] == 5.6


def test_missing_round_is_skipped():
    history = [{"round": None, "total_points": 6}, {"round": 5, "total_points": 3}]
    result = _accumulate_gw_history(history)
    assert list(result.keys()) == [5]
