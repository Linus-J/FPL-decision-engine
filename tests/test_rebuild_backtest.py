"""The calibration instrument for Objective v2 (scripts/backtest.py).

run_naive_xi_backtest fixes the initial 15, so a squad-level re-ranker has a
pool of one there and a mu sweep over it measures nothing. This harness
rebuilds the 15 every gameweek instead, making each gameweek an independent
squad-selection observation.
"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.backtest as bt
from optimiser.squad import SquadSolution


def _stub_solution(squad_ids=(1, 2, 3)):
    squad = pd.DataFrame({
        "id": list(squad_ids),
        "position": ["MID"] * len(squad_ids),
        "team_id": [10] * len(squad_ids),
        "web_name": [f"p{i}" for i in squad_ids],
        "now_cost": [5.0] * len(squad_ids),
        "bench_order": [0] * len(squad_ids),
    })
    return SquadSolution(
        squad=squad, starting_xi=squad.copy(), captain_id=squad_ids[0],
        vice_captain_id=squad_ids[-1], total_xpts=10.0,
        total_cost=5.0 * len(squad_ids), hits_taken=0,
    )


def _stub_backtest_io(monkeypatch, gws):
    """Stub every I/O boundary run_rebuild_backtest touches, so the test
    exercises the loop's structure and nothing else."""
    # The harness skips a gameweek with under 50 prior rows, so the stub needs
    # real history behind the first gameweek under test, not just the gameweeks
    # being scored.
    all_gws = list(range(1, max(gws) + 1))
    stats = pd.DataFrame({
        "gameweek": [g for g in all_gws for _ in range(60)],
        "player_id": [1] * 60 * len(all_gws),
        "minutes": [90, 0] * (30 * len(all_gws)),
    })
    monkeypatch.setattr(bt, "_load_all_stats", lambda season: stats)
    monkeypatch.setattr(bt.assemble, "load_match_odds", lambda season: pd.DataFrame())
    monkeypatch.setattr(bt.assemble, "load_defcon_events", lambda season: pd.DataFrame())
    monkeypatch.setattr(
        bt.assemble, "compute_defcon_field_shares",
        lambda season: {"DEF": {}, "MID_FWD": {}},
    )
    monkeypatch.setattr(
        bt, "_load_players_snapshot",
        lambda season, gw: pd.DataFrame({
            "id": [1, 2, 3], "position": ["MID"] * 3, "team_id": [10] * 3,
            "web_name": ["p1", "p2", "p3"], "now_cost": [5.0] * 3,
        }),
    )
    monkeypatch.setattr(bt, "train_minutes", lambda **k: None)
    monkeypatch.setattr(
        bt, "_build_gw_projections",
        lambda **k: pd.DataFrame({
            "player_id": [1, 2, 3], "gameweek": [k["target_gw"]] * 3,
            "xpts": [5.0, 4.0, 3.0], "start_probability": [1.0] * 3,
        }),
    )
    monkeypatch.setattr(
        bt, "optimise_starting_xi", lambda squad, proj, gw, **k: _stub_solution()
    )
    monkeypatch.setattr(bt, "_actual_gw_points", lambda *a, **k: {1: 5.0, 2: 4.0, 3: 3.0})
    monkeypatch.setattr(bt, "_actual_gw_minutes", lambda *a, **k: {1: 90, 2: 90, 3: 90})
    monkeypatch.setattr(bt, "_score_squad", lambda *a, **k: 42)
    # The harness generates the pool itself now, so the sweep can reuse one
    # pool across every candidate mu instead of rebuilding it per candidate.
    monkeypatch.setattr(
        bt, "generate_squad_pool",
        lambda projections, players, n=10, **k: [_stub_solution()],
    )


def test_rebuild_harness_rebuilds_every_gameweek(monkeypatch):
    builds = []

    def fake_joint(projections, players, **kwargs):
        builds.append(kwargs.get("gameweek"))
        return _stub_solution()

    monkeypatch.setattr(bt, "optimise_squad_joint", fake_joint)
    _stub_backtest_io(monkeypatch, gws=[6, 7, 8])

    df = bt.run_rebuild_backtest(
        season="2025-26", start_gw=6, end_gw=8, score_2627=False
    )

    assert builds == [6, 7, 8], "the 15 must be rebuilt from scratch each GW"
    assert list(df["gameweek"]) == [6, 7, 8]
    assert list(df["actual_pts"]) == [42, 42, 42]


def test_rebuild_harness_carries_no_squad_between_gameweeks(monkeypatch):
    """No transfers, no chips, no carry-over -- each GW is independent."""
    seen_kwargs = []

    def fake_joint(projections, players, **kwargs):
        seen_kwargs.append(kwargs)
        return _stub_solution()

    monkeypatch.setattr(bt, "optimise_squad_joint", fake_joint)
    _stub_backtest_io(monkeypatch, gws=[6, 7])

    bt.run_rebuild_backtest(season="2025-26", start_gw=6, end_gw=7, score_2627=False)

    assert len(seen_kwargs) == 2
    for kwargs in seen_kwargs:
        assert "current_squad" not in kwargs
        assert "max_transfers" not in kwargs


def test_rebuild_harness_reports_club_concentration(monkeypatch):
    """Pricing concentration is half of what the joint measure should buy, so
    the harness has to expose it rather than only mean points."""
    monkeypatch.setattr(
        bt, "optimise_squad_joint",
        lambda projections, players, **k: _stub_solution(squad_ids=(1, 2, 3)),
    )
    _stub_backtest_io(monkeypatch, gws=[6])

    df = bt.run_rebuild_backtest(
        season="2025-26", start_gw=6, end_gw=6, score_2627=False
    )

    # All three stub players share team 10, which is at the 3-per-club cap.
    assert df["n_clubs_at_cap"].iloc[0] == 1


def test_rebuild_harness_threads_samples_into_the_re_ranker(monkeypatch):
    """The whole point of sample_sink: the joint draws must reach the
    re-ranker in memory, never via projection_samples."""
    seen = {}

    def fake_joint(projections, players, **kwargs):
        seen["sample_rows"] = kwargs.get("sample_rows")
        return _stub_solution()

    monkeypatch.setattr(bt, "optimise_squad_joint", fake_joint)
    _stub_backtest_io(monkeypatch, gws=[6])

    def fake_projections(**k):
        sink = k.get("sample_sink")
        assert sink is not None, "the harness must pass a sample_sink"
        sink.append(
            {"player_id": 1, "gameweek": 6, "season": "2025-26",
             "scenario_id": 0, "xpts": 5.0}
        )
        return pd.DataFrame({
            "player_id": [1, 2, 3], "gameweek": [6] * 3,
            "xpts": [5.0, 4.0, 3.0], "start_probability": [1.0] * 3,
        })

    monkeypatch.setattr(bt, "_build_gw_projections", fake_projections)

    bt.run_rebuild_backtest(
        season="2025-26", start_gw=6, end_gw=6, score_2627=False
    )

    assert seen["sample_rows"], "sample rows must reach optimise_squad_joint"
    assert seen["sample_rows"][0]["player_id"] == 1


def test_rebuild_harness_skips_a_gameweek_it_cannot_build(monkeypatch):
    def fake_joint(projections, players, **kwargs):
        if kwargs.get("gameweek") == 7:
            raise RuntimeError("infeasible")
        return _stub_solution()

    monkeypatch.setattr(bt, "optimise_squad_joint", fake_joint)
    _stub_backtest_io(monkeypatch, gws=[6, 7, 8])

    df = bt.run_rebuild_backtest(
        season="2025-26", start_gw=6, end_gw=8, score_2627=False
    )

    assert list(df["gameweek"]) == [6, 8], "a failed GW is skipped, not fatal"


@pytest.mark.parametrize("harness", ["rebuild"])
def test_rebuild_harness_returns_empty_frame_when_no_gameweeks_qualify(
    monkeypatch, harness
):
    monkeypatch.setattr(bt, "optimise_squad_joint", lambda *a, **k: _stub_solution())
    _stub_backtest_io(monkeypatch, gws=[6])

    df = bt.run_rebuild_backtest(
        season="2025-26", start_gw=20, end_gw=25, score_2627=False
    )
    assert df.empty


def test_mu_candidates_reuse_one_pool_per_gameweek(monkeypatch):
    """The whole reason mu_candidates exists. The pool depends only on the
    pure-mean objective, so regenerating it per candidate would repeat every
    MILP solve and every Monte-Carlo assembly once per candidate."""
    pools_built = []
    projections_built = []

    monkeypatch.setattr(bt, "optimise_squad_joint", lambda p, pl, **k: _stub_solution())
    _stub_backtest_io(monkeypatch, gws=[6, 7])

    def counting_pool(projections, players, n=10, **k):
        pools_built.append(n)
        return [_stub_solution()]

    def counting_projections(**k):
        projections_built.append(k["target_gw"])
        return pd.DataFrame({
            "player_id": [1, 2, 3], "gameweek": [k["target_gw"]] * 3,
            "xpts": [5.0, 4.0, 3.0], "start_probability": [1.0] * 3,
        })

    monkeypatch.setattr(bt, "generate_squad_pool", counting_pool)
    monkeypatch.setattr(bt, "_build_gw_projections", counting_projections)

    df = bt.run_rebuild_backtest(
        season="2025-26", start_gw=6, end_gw=7, score_2627=False,
        mu_candidates=[-1.0, -0.5, 0.0],
    )

    assert len(pools_built) == 2, "one pool per gameweek, not one per (gw, mu)"
    assert len(projections_built) == 2, "one MC assembly per gameweek"
    assert len(df) == 6, "3 mus x 2 gameweeks of results"
    assert sorted(df["mu_baseline"].unique()) == [-1.0, -0.5, 0.0]


def test_single_mu_run_has_no_mu_column(monkeypatch):
    """Without mu_candidates the harness behaves as before, one row per GW."""
    monkeypatch.setattr(bt, "optimise_squad_joint", lambda p, pl, **k: _stub_solution())
    _stub_backtest_io(monkeypatch, gws=[6])

    df = bt.run_rebuild_backtest(
        season="2025-26", start_gw=6, end_gw=6, score_2627=False
    )
    assert "mu_baseline" not in df.columns
    assert len(df) == 1
