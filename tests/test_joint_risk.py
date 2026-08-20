"""Objective v2 — covariance-aware squad selection (optimiser/joint_risk.py)."""

from __future__ import annotations

import numpy as np

from optimiser import joint_risk


def test_matrix_from_rows_aligns_players_to_columns():
    rows = [
        {"player_id": 7, "gameweek": 1, "season": "2026-27", "scenario_id": 0, "xpts": 1.0},
        {"player_id": 7, "gameweek": 1, "season": "2026-27", "scenario_id": 1, "xpts": 3.0},
        {"player_id": 9, "gameweek": 1, "season": "2026-27", "scenario_id": 0, "xpts": 5.0},
        {"player_id": 9, "gameweek": 1, "season": "2026-27", "scenario_id": 1, "xpts": 7.0},
    ]
    m = joint_risk.matrix_from_rows(rows, gameweek=1)

    assert m.values.shape == (2, 2)
    assert np.allclose(m.values[:, m.column_index[7]], [1.0, 3.0])
    assert np.allclose(m.values[:, m.column_index[9]], [5.0, 7.0])


def test_matrix_from_rows_filters_by_gameweek():
    rows = [
        {"player_id": 7, "gameweek": 1, "season": "2026-27", "scenario_id": 0, "xpts": 1.0},
        {"player_id": 7, "gameweek": 2, "season": "2026-27", "scenario_id": 0, "xpts": 99.0},
    ]
    m = joint_risk.matrix_from_rows(rows, gameweek=1)

    assert m.values.shape == (1, 1)
    assert m.values[0, m.column_index[7]] == 1.0


def test_matrix_from_rows_is_empty_when_no_rows_match():
    m = joint_risk.matrix_from_rows([], gameweek=1)
    assert m.column_index == {}
    assert m.values.size == 0


def test_players_with_no_draws_are_absent_not_zero():
    """A missing player must be detectable, never silently worth zero points."""
    rows = [{"player_id": 7, "gameweek": 1, "season": "2026-27", "scenario_id": 0, "xpts": 1.0}]
    m = joint_risk.matrix_from_rows(rows, gameweek=1)

    assert 7 in m.column_index
    assert 8 not in m.column_index


def test_disjoint_scenario_ranges_from_different_fixtures_are_stacked():
    """assemble.py gives each FIXTURE its own disjoint scenario_id range, so a
    naive pivot leaves NaN where two fixtures' ranges do not overlap. Those
    NaNs are not missing data and must not become rows of zeros."""
    rows = (
        [{"player_id": 1, "gameweek": 1, "season": "s", "scenario_id": i, "xpts": 1.0}
         for i in range(4)]
        + [{"player_id": 2, "gameweek": 1, "season": "s", "scenario_id": 4 + i, "xpts": 2.0}
           for i in range(4)]
    )
    m = joint_risk.matrix_from_rows(rows, gameweek=1)

    assert m.values.shape == (4, 2), "two fixtures of 4 scenarios each -> 4 rows, not 8"
    assert not np.isnan(m.values).any()
    assert np.allclose(m.values[:, m.column_index[1]], 1.0)
    assert np.allclose(m.values[:, m.column_index[2]], 2.0)


def test_matrices_from_rows_keys_by_gameweek_and_drops_empty_ones():
    """The pool is ranked on a decayed multi-GW objective, so the re-rank needs
    one matrix per horizon gameweek, not just the target one."""
    rows = [
        {"player_id": 7, "gameweek": 1, "season": "2026-27", "scenario_id": 0, "xpts": 1.0},
        {"player_id": 7, "gameweek": 2, "season": "2026-27", "scenario_id": 0, "xpts": 4.0},
    ]
    out = joint_risk.matrices_from_rows(rows, gameweeks=[1, 2, 3])

    assert set(out) == {1, 2}, "GW3 has no draws and must be dropped, not empty-padded"
    assert out[2].values[0, out[2].column_index[7]] == 4.0
