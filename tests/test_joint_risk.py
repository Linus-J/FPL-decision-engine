"""Objective v2 — covariance-aware squad selection (optimiser/joint_risk.py)."""

from __future__ import annotations

import numpy as np
import pytest

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


# --- pure core -----------------------------------------------------------


def _matrix(columns: dict[int, list[float]]) -> joint_risk.ScenarioMatrix:
    ids = list(columns)
    return joint_risk.ScenarioMatrix(
        values=np.column_stack([np.array(columns[i], dtype=float) for i in ids]),
        column_index={pid: i for i, pid in enumerate(ids)},
    )


def test_semideviation_is_one_sided():
    # sqrt(E[max(0, x - mean)^2]) over ALL observations, matching
    # risk_adjusted_score's definition: mean 5, two deviations of +5, so
    # sqrt((25 + 25) / 4) -- not the mean of the upper half alone.
    totals = np.array([0.0, 0.0, 10.0, 10.0])
    assert joint_risk.semideviation(totals, upper=True) == pytest.approx(np.sqrt(12.5))
    # Symmetric input, so both sides agree.
    assert joint_risk.semideviation(totals, upper=False) == pytest.approx(np.sqrt(12.5))

    skewed = np.array([4.0, 4.0, 4.0, 16.0])
    assert joint_risk.semideviation(skewed, upper=True) > joint_risk.semideviation(
        skewed, upper=False
    )


def test_captain_is_counted_twice():
    m = _matrix({1: [2.0, 4.0], 2: [1.0, 1.0]})
    totals = joint_risk.squad_totals(
        m, xi_ids=[1, 2], captain_id=1, bench_ids=[],
        bench_weights=(), gk_bench_weight=0.0,
    )
    assert np.allclose(totals, [5.0, 9.0])


def test_joint_mean_equals_the_linear_objective_at_mu_zero():
    """At mu=0 the joint score must reduce to the plain expected total, so
    switching the objective on cannot move a pick until mu moves."""
    m = _matrix({1: [2.0, 4.0], 2: [1.0, 3.0]})
    score = joint_risk.joint_score(
        {1: m}, xi_ids=[1, 2], captain_id=1, bench_ids=[], mu=0.0,
        bench_weights=(), gk_bench_weight=0.0,
    )
    assert score == pytest.approx(8.0)  # mean of [5.0, 11.0], captain doubled


def test_horizon_totals_apply_decay_like_the_linear_objective():
    """The pool is ranked on decayed multi-GW points, so the re-rank must use
    the same weights or mu=0 would silently reorder it."""
    near = _matrix({1: [10.0, 10.0]})
    far = _matrix({1: [10.0, 10.0]})
    totals = joint_risk.horizon_totals(
        {1: near, 2: far}, xi_ids=[1], captain_id=1, bench_ids=[],
        bench_weights=(), gk_bench_weight=0.0, decay=0.5,
    )
    # GW1 at weight 1.0 and GW2 at weight 0.5, captain doubling each.
    assert np.allclose(totals, [20.0 + 0.5 * 20.0] * 2)


def test_horizon_totals_ignore_gameweeks_with_no_matrix():
    m = _matrix({1: [4.0, 4.0]})
    totals = joint_risk.horizon_totals(
        {1: m}, xi_ids=[1], captain_id=1, bench_ids=[],
        bench_weights=(), gk_bench_weight=0.0, decay=1.0,
    )
    assert np.allclose(totals, [8.0, 8.0])


def test_correlated_teammates_are_penalised_where_the_old_scorer_was_blind():
    """THE test for this feature.

    Three players with IDENTICAL marginals: same mean, same variance, same
    semi-deviation taken one at a time. Players 1 and 2 move together (a keeper
    and his own centre-back sharing a clean sheet); player 3 moves against them.

    Summing per-player risk -- whether as variances or, as scoring.py actually
    does, as semi-deviations -- cannot tell these two squads apart, because
    every per-player input is the same. The joint measure can: the diversified
    pair has a materially thinner downside.
    """
    together = [6.0, 0.0, 6.0, 0.0]
    against = [0.0, 6.0, 0.0, 6.0]
    m = _matrix({1: together, 2: together, 3: against})

    for pid in (1, 2, 3):
        col = m.values[:, m.column_index[pid]]
        assert col.mean() == pytest.approx(3.0)
        assert col.std() == pytest.approx(3.0)

    correlated = joint_risk.joint_score(
        {1: m}, xi_ids=[1, 2], captain_id=1, bench_ids=[], mu=-1.0,
        bench_weights=(), gk_bench_weight=0.0,
    )
    diversified = joint_risk.joint_score(
        {1: m}, xi_ids=[1, 3], captain_id=1, bench_ids=[], mu=-1.0,
        bench_weights=(), gk_bench_weight=0.0,
    )

    assert diversified > correlated, (
        "the joint measure must prefer splitting correlated teammates at mu<0"
    )


def test_bench_slots_are_weighted_by_slot_order():
    m = _matrix({1: [10.0, 10.0], 2: [4.0, 4.0], 3: [4.0, 4.0]})
    first = joint_risk.squad_totals(
        m, xi_ids=[1], captain_id=1, bench_ids=[2, 3],
        bench_weights=(0.5, 0.1), gk_bench_weight=0.0,
    )
    assert np.allclose(first, [20.0 + 0.5 * 4.0 + 0.1 * 4.0] * 2)


def test_missing_player_raises_rather_than_scoring_zero():
    m = _matrix({1: [1.0, 2.0]})
    with pytest.raises(KeyError):
        joint_risk.squad_totals(
            m, xi_ids=[1, 999], captain_id=1, bench_ids=[],
            bench_weights=(), gk_bench_weight=0.0,
        )
