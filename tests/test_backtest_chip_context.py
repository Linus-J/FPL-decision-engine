"""Regression tests for a real bug found 2026-07-29 (user's own review):
scripts/backtest.py::run_backtest never passed dgw_gws/bgw_affected_count
to recommend_chip at all, so Bench Boost and Free Hit -- both gated behind
those being non-empty/nonzero -- could never even be EVALUATED during
backtesting, regardless of threshold calibration. Gameweek.is_dgw/is_bgw
can't fix this either: those columns are only ever populated by the live
fpl_api.py ingestion path, confirmed empty (0) for all 38 gameweeks of the
2025-26 season. These pure helpers derive DGW/BGW status instead from
already-loaded historical fixture context."""

from __future__ import annotations

import pandas as pd

from scripts.backtest import (
    _bgw_affected_count,
    _dgw_bgw_gws_in_window,
    _fixture_count_by_gw,
)


def _all_stats_row(gw, team, opp, was_home):
    return {"gameweek": gw, "team_id_season": team, "opponent_team_id": opp, "was_home": was_home}


def test_fixture_count_by_gw_normal_round_counts_home_matches_once():
    # 3 real matches (6 team-perspective rows: 3 home + 3 away)
    rows = []
    for gw in (5,):
        for home, away in [(1, 2), (3, 4), (5, 6)]:
            rows.append(_all_stats_row(gw, home, away, True))
            rows.append(_all_stats_row(gw, away, home, False))
    all_stats = pd.DataFrame(rows)
    assert _fixture_count_by_gw(all_stats) == {5: 3}


def test_dgw_bgw_gws_in_window_detects_both():
    fixture_counts = {10: 10, 11: 13, 12: 10, 13: 6}
    dgw, bgw = _dgw_bgw_gws_in_window(fixture_counts, start_gw=10, horizon=4)
    assert dgw == {11}
    assert bgw == {13}


def test_dgw_bgw_gws_in_window_missing_gw_treated_as_normal():
    dgw, bgw = _dgw_bgw_gws_in_window({}, start_gw=10, horizon=2)
    assert dgw == set()
    assert bgw == set()


def test_dgw_bgw_gws_in_window_only_considers_the_lookahead():
    fixture_counts = {10: 10, 11: 13}  # GW11's DGW is outside a horizon=1 window
    dgw, bgw = _dgw_bgw_gws_in_window(fixture_counts, start_gw=10, horizon=1)
    assert dgw == set()


def test_bgw_affected_count_counts_squad_players_blanking():
    projections = pd.DataFrame([
        {"gameweek": 13, "player_id": 1, "xpts": 0.0},
        {"gameweek": 13, "player_id": 2, "xpts": 4.0},
        {"gameweek": 13, "player_id": 3, "xpts": 0.0},
    ])
    assert _bgw_affected_count([1, 2, 3], {13}, projections) == 2


def test_bgw_affected_count_no_bgw_gws_is_zero():
    projections = pd.DataFrame([{"gameweek": 13, "player_id": 1, "xpts": 0.0}])
    assert _bgw_affected_count([1], set(), projections) == 0


def test_bgw_affected_count_empty_squad_is_zero():
    projections = pd.DataFrame([{"gameweek": 13, "player_id": 1, "xpts": 0.0}])
    assert _bgw_affected_count([], {13}, projections) == 0
