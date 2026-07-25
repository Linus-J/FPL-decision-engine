"""P5 clean sheets + P4 assists components."""

from __future__ import annotations

import numpy as np
import pytest

from projection import assists as A
from projection import clean_sheets as CS


# --- P5 clean sheets --------------------------------------------------------
def test_clean_sheet_prob():
    assert CS.clean_sheet_prob(0.0) == pytest.approx(1.0)
    assert CS.clean_sheet_prob(0.5) == pytest.approx(0.6065, abs=1e-3)
    assert CS.clean_sheet_prob(1.5) < CS.clean_sheet_prob(0.5)


def test_expected_cs_points_by_position_and_minutes():
    lam = 0.7
    base = CS.clean_sheet_prob(lam) * 0.9
    assert CS.expected_cs_points(lam, 0.9, "GK") == pytest.approx(base * 4)
    assert CS.expected_cs_points(lam, 0.9, "DEF") == pytest.approx(base * 4)
    assert CS.expected_cs_points(lam, 0.9, "MID") == pytest.approx(base * 1)
    assert CS.expected_cs_points(lam, 0.9, "FWD") == 0.0        # FWD CS = 0 pts
    # no minutes → no CS points even for a strong defence
    assert CS.expected_cs_points(0.2, 0.0, "DEF") == 0.0


def test_expected_concede_points_gk_def_only_and_negative():
    assert CS.expected_concede_points(2.0, 1.0, "MID") == 0.0    # only GK/DEF
    d2 = CS.expected_concede_points(2.0, 1.0, "DEF")
    d05 = CS.expected_concede_points(0.5, 1.0, "DEF")
    assert d2 < d05 < 0.0                                        # more goals → more negative
    assert CS.expected_concede_points(0.0, 1.0, "DEF") == pytest.approx(0.0)


def test_sample_clean_sheet_points_matches_expectation():
    rng = np.random.default_rng(0)
    lam = 0.7
    ref = CS.clean_sheet_prob(lam) * 4 + CS.expected_concede_points(lam, 1.0, "GK")
    draws = [CS.sample_clean_sheet_points(rng, lam, True, "GK") for _ in range(30000)]
    assert np.mean(draws) == pytest.approx(ref, abs=0.05)


def test_sample_clean_sheet_points_gates_concede_penalty_on_played_any():
    # a certain-DNP defender (played_60=False, played_any=False) must never
    # be docked for goals conceded while they weren't on the pitch (P10 bug:
    # the concede-penalty branch had no play-time gate at all).
    rng = np.random.default_rng(0)
    draws = [
        CS.sample_clean_sheet_points(rng, 2.0, False, "DEF", conceded=3, played_any=False)
        for _ in range(50)
    ]
    assert all(d == 0 for d in draws)
    # played_any=True (a sub who came on) still gets docked normally
    docked = CS.sample_clean_sheet_points(rng, 2.0, False, "DEF", conceded=4, played_any=True)
    assert docked < 0


# --- P4 assists -------------------------------------------------------------
def _players():
    return [
        {"player_id": 1, "weight": 3.0, "minutes_frac": 1.0},
        {"player_id": 2, "weight": 1.0, "minutes_frac": 1.0},
    ]


def test_team_assists_conserve_and_scale_by_fraction():
    out = A.distribute_team_assists(_players(), team_lambda=2.0, assist_fraction=0.75)
    assert sum(out.values()) == pytest.approx(2.0 * 0.75)
    assert out[1] > out[2]


def test_expected_assist_points():
    assert A.expected_assist_points(0.4) == pytest.approx(0.4 * 3)


def test_sample_assists_mean():
    rng = np.random.default_rng(1)
    draws = [A.sample_assists(rng, 0.5) for _ in range(20000)]
    assert np.mean(draws) == pytest.approx(0.5, abs=0.03)
