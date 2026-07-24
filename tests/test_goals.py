"""P3 goals component — odds-anchored team-goal distribution by shot share."""

from __future__ import annotations

import numpy as np
import pytest

from projection.goals import (
    GOAL_POINTS,
    distribute_team_goals,
    expected_goal_points,
    sample_goals,
)


def _players():
    return [
        {"player_id": 1, "weight": 3.0, "minutes_frac": 1.0},   # heavy shooter, plays
        {"player_id": 2, "weight": 1.0, "minutes_frac": 1.0},
        {"player_id": 3, "weight": 2.0, "minutes_frac": 0.0},   # shooter but benched
    ]


def test_distribution_conserves_team_lambda():
    out = distribute_team_goals(_players(), team_lambda=1.8)
    assert sum(out.values()) == pytest.approx(1.8)


def test_heavier_shooter_gets_more_and_benched_gets_none():
    out = distribute_team_goals(_players(), team_lambda=1.8)
    assert out[1] > out[2] > 0
    assert out[3] == pytest.approx(0.0)          # minutes_frac 0 → no goals
    assert out[1] == pytest.approx(1.8 * 3 / 4)  # weights 3 vs 1 among players who play


def test_no_shot_data_spreads_over_players_expected_to_play():
    players = [{"player_id": 1, "weight": 0.0, "minutes_frac": 1.0},
               {"player_id": 2, "weight": 0.0, "minutes_frac": 0.5},
               {"player_id": 3, "weight": 0.0, "minutes_frac": 0.0}]
    out = distribute_team_goals(players, team_lambda=1.5)
    assert sum(out.values()) == pytest.approx(1.5)
    assert out[1] == pytest.approx(1.5 * 1.0 / 1.5)
    assert out[3] == pytest.approx(0.0)


def test_nobody_playing_is_all_zero():
    players = [{"player_id": 1, "weight": 2.0, "minutes_frac": 0.0}]
    assert distribute_team_goals(players, 1.5) == {1: 0.0}


def test_expected_goal_points_by_position():
    assert expected_goal_points(0.5, "FWD") == pytest.approx(0.5 * GOAL_POINTS["FWD"])
    assert expected_goal_points(0.5, "DEF") == pytest.approx(0.5 * 6)
    assert expected_goal_points(0.5, "GK") == pytest.approx(0.5 * 10)


def test_sample_goals_mean_matches_lambda():
    rng = np.random.default_rng(42)
    draws = [sample_goals(rng, 0.8) for _ in range(20000)]
    assert np.mean(draws) == pytest.approx(0.8, abs=0.03)
