"""Objective v2 — covariance-aware squad selection (optimiser/joint_risk.py)."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest

from config.strategy import OPTIMISER
from optimiser import joint_risk
from optimiser.scoring import lambda_mu_for_risk_level
from optimiser.squad import SquadSolution


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


# --- orchestrator --------------------------------------------------------


def _solution(squad_ids: list[int], xi_ids: list[int], captain_id: int) -> SquadSolution:
    squad = pd.DataFrame({
        "id": squad_ids,
        "position": ["MID"] * len(squad_ids),
        "now_cost": [5.0] * len(squad_ids),
    })
    return SquadSolution(
        squad=squad,
        starting_xi=squad[squad["id"].isin(xi_ids)].copy(),
        captain_id=captain_id,
        vice_captain_id=xi_ids[-1],
        total_xpts=0.0,
        total_cost=float(len(squad_ids) * 5),
        hits_taken=0,
    )


def test_mu_zero_returns_pool_head_without_touching_the_db(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("mu == 0 must not open a database session")

    monkeypatch.setattr(joint_risk, "load_scenario_matrices", explode)
    pool = [_solution([1, 2], [1, 2], 1), _solution([1, 3], [1, 3], 1)]

    chosen = joint_risk.covariance_aware_squad(
        pool, mu=0.0, cfg=OPTIMISER, season="2026-27", gameweek=1
    )
    assert chosen is pool[0]


def test_empty_samples_fall_back_to_pool_head(caplog):
    pool = [_solution([1, 2], [1, 2], 1), _solution([1, 3], [1, 3], 1)]
    with caplog.at_level(logging.WARNING):
        chosen = joint_risk.covariance_aware_squad(
            pool, mu=-1.0, cfg=OPTIMISER, matrices={}
        )
    assert chosen is pool[0]
    assert "no scenario samples" in caplog.text.lower()


def test_rerank_prefers_the_diversified_squad():
    together, against = [6.0, 0.0, 6.0, 0.0], [0.0, 6.0, 0.0, 6.0]
    m = _matrix({1: together, 2: together, 3: against})
    pool = [_solution([1, 2], [1, 2], 1), _solution([1, 3], [1, 3], 1)]

    chosen = joint_risk.covariance_aware_squad(
        pool, mu=-1.0, cfg=OPTIMISER, matrices={1: m}
    )
    assert chosen is pool[1], "the joint re-rank must overturn the mean-ordered pool"


def test_a_candidate_with_missing_draws_is_skipped_not_crashed():
    m = _matrix({1: [1.0, 2.0], 2: [1.0, 2.0]})
    pool = [_solution([1, 999], [1, 999], 1), _solution([1, 2], [1, 2], 1)]

    chosen = joint_risk.covariance_aware_squad(
        pool, mu=-1.0, cfg=OPTIMISER, matrices={1: m}
    )
    assert chosen is pool[1]


def test_all_candidates_unscoreable_falls_back_to_pool_head():
    m = _matrix({4: [1.0, 2.0]})
    pool = [_solution([1, 2], [1, 2], 1)]
    chosen = joint_risk.covariance_aware_squad(
        pool, mu=-1.0, cfg=OPTIMISER, matrices={1: m}
    )
    assert chosen is pool[0]


def test_empty_pool_is_an_error_not_a_silent_none():
    with pytest.raises(ValueError, match="non-empty"):
        joint_risk.covariance_aware_squad([], mu=-1.0, cfg=OPTIMISER, matrices={})


# --- squad.py entry point ------------------------------------------------


def test_optimise_squad_joint_generates_the_pool_at_mu_zero(monkeypatch):
    """The pool must always come from the pure-mean objective, so one pool per
    gameweek can be reused across every mu in a sweep."""
    from optimiser import squad as squad_module

    seen = {}

    def fake_pool(projections, players, n=10, **kwargs):
        seen["config"] = kwargs.get("config")
        seen["n"] = n
        return [_solution([1, 2], [1, 2], 1)]

    monkeypatch.setattr(squad_module, "generate_squad_pool", fake_pool)

    cfg = dataclasses.replace(OPTIMISER, risk_level=-1.0, mu_baseline=0.0, mu_range=1.0)
    squad_module.optimise_squad_joint(
        pd.DataFrame(), pd.DataFrame(), config=cfg, pool_size=7, matrices={},
    )

    assert seen["n"] == 7
    assert seen["config"].mu_baseline == 0.0
    assert seen["config"].risk_level == 0.0, "pool must be built at mu=0"
    assert seen["config"].mu_range == 0.0


def test_optimise_squad_joint_builds_matrices_from_sample_rows(monkeypatch):
    """The backtest route: samples arrive in memory, never via the DB."""
    from optimiser import squad as squad_module

    monkeypatch.setattr(
        squad_module, "generate_squad_pool",
        lambda projections, players, n=10, **k: [
            _solution([1, 2], [1, 2], 1), _solution([1, 3], [1, 3], 1)
        ],
    )
    monkeypatch.setattr(
        joint_risk, "load_scenario_matrices",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not hit the DB")),
    )

    rows = (
        [{"player_id": p, "gameweek": 1, "season": "s", "scenario_id": i,
          "xpts": v[i]} for p, v in
         {1: [6.0, 0.0], 2: [6.0, 0.0], 3: [0.0, 6.0]}.items() for i in range(2)]
    )
    cfg = dataclasses.replace(OPTIMISER, risk_level=-1.0, mu_baseline=0.0, mu_range=1.0)

    chosen = squad_module.optimise_squad_joint(
        pd.DataFrame(), pd.DataFrame(), gameweek=1, sample_rows=rows,
        config=cfg, horizon=1,
    )
    assert chosen.captain_id == 1
    assert set(chosen.squad["id"]) == {1, 3}, "diversified squad wins at mu<0"


def test_optimise_squad_joint_raises_when_no_squad_is_feasible(monkeypatch):
    from optimiser import squad as squad_module

    monkeypatch.setattr(
        squad_module, "generate_squad_pool", lambda *a, **k: []
    )
    with pytest.raises(RuntimeError, match="no feasible squad"):
        squad_module.optimise_squad_joint(pd.DataFrame(), pd.DataFrame(), matrices={})


# --- live wiring ---------------------------------------------------------


def test_live_free_hit_path_uses_the_joint_optimiser():
    """A calibrated mu only reaches the real bot if the decision engine calls
    the joint entry point. It called optimise_squad directly until 2026-08-20,
    which would have left the whole feature dormant live even at mu != 0."""
    import inspect

    from agent import decision_engine

    src = inspect.getsource(decision_engine)
    assert "optimise_squad_joint(" in src
    assert "optimise_squad(" not in src.replace("optimise_squad_joint(", "")


def test_joint_optimiser_at_mu_zero_returns_the_plain_optimum(monkeypatch):
    """The mu=0 short-circuit: at exactly zero the joint optimiser cannot
    change a pick, and must not even reach the DB to establish that.

    This was the SHIPPED-DEFAULT safety property until 2026-08-25, when
    mu_baseline moved 0.0 -> -0.25 and the short-circuit stopped firing on the
    live path (see config/strategy.py and the sibling test below). The
    mechanism still has to work, because mu returning to 0 is the documented
    way to switch the re-ranker back off -- so the test now pins mu=0
    explicitly instead of relying on the default being zero.
    """
    from optimiser import squad as squad_module

    mu_zero = dataclasses.replace(OPTIMISER, mu_baseline=0.0, risk_level=0.0)
    head = _solution([1, 2], [1, 2], 1)
    monkeypatch.setattr(
        squad_module, "generate_squad_pool",
        lambda projections, players, n=10, **k: [head, _solution([1, 3], [1, 3], 1)],
    )
    monkeypatch.setattr(
        joint_risk, "load_scenario_matrices",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not hit the DB")),
    )

    chosen = squad_module.optimise_squad_joint(
        pd.DataFrame(), pd.DataFrame(), config=mu_zero, season="2026-27", gameweek=1,
    )
    assert chosen is head


def test_joint_reranker_is_live_on_the_shipped_defaults():
    """The inverse of the test above, and the reason it had to change.

    mu_baseline is -0.25 as of 2026-08-25, so the re-ranker is NOT dormant on
    the live path: every real squad decision now generates a pool and scores
    it against the scenario matrices. If someone sets mu back to 0 without
    meaning to, this fails and says so -- which is the failure that would
    otherwise be silent, since a dormant re-ranker still returns a legal
    squad, just the mean-optimal one.
    """
    lam, mu = lambda_mu_for_risk_level(
        OPTIMISER.risk_level,
        OPTIMISER.max_ownership_differential,
        OPTIMISER.mu_baseline,
        OPTIMISER.mu_range,
    )
    assert mu != 0.0, "mu_baseline is 0 -- the joint re-ranker is dormant"
    assert lam == 0.0, "lambda should still be 0 at the default risk_level of 0"
