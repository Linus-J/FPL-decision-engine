"""P10 — MC assembly core (projection/assemble.py). Covers the pure
per-fixture sampler; the DB loaders are thin passthroughs exercised by the
live P-XI gate run instead of mocked here.
"""

from __future__ import annotations

import numpy as np
import pytest

from projection import assemble

SHARES = {
    "DEF": {"clearances": 0.3, "blocks": 0.15, "interceptions": 0.2, "tackles": 0.35},
    "MID_FWD": {"clearances": 0.15, "blocks": 0.1, "interceptions": 0.15,
                "tackles": 0.3, "recoveries": 0.3},
}


def _player(pid, position, **overrides):
    base = {
        "player_id": pid, "position": position,
        "goal_weight": 0.0, "assist_weight": 0.0, "key_pass_rate": 0.0,
        "yellow_rate": 0.0, "red_rate": 0.0, "defcon_rate": 0.0,
        "p0": 0.1, "p1": 0.1, "p2": 0.8,
    }
    base.update(overrides)
    return base


def test_minutes_scale_bands():
    assert assemble._minutes_scale(0) == 0.0
    assert assemble._minutes_scale(1) == assemble.CAMEO_MINUTES_FRAC
    assert assemble._minutes_scale(2) == 1.0


def test_draw_band_deterministic_certain_dnp():
    rng = np.random.default_rng(0)
    draws = [assemble._draw_band(rng, 1.0, 0.0, 0.0) for _ in range(50)]
    assert all(b == 0 for b in draws)


def test_draw_band_deterministic_certain_start():
    rng = np.random.default_rng(0)
    draws = [assemble._draw_band(rng, 0.0, 0.0, 1.0) for _ in range(50)]
    assert all(b == 2 for b in draws)


def test_draw_band_normalizes_unnormalized_probs():
    # doesn't sum to 1 -- must not raise, must stay in {0,1,2}
    rng = np.random.default_rng(0)
    draws = [assemble._draw_band(rng, 2.0, 2.0, 4.0) for _ in range(200)]
    assert set(draws) <= {0, 1, 2}


def test_defcon_split_conserves_and_zero_stays_zero():
    rng = np.random.default_rng(0)
    out = assemble._defcon_split(rng, 0, "DEF", SHARES)
    assert out == dict.fromkeys(assemble._DEF_CBIT_FIELDS, 0)

    out2 = assemble._defcon_split(rng, 10, "DEF", SHARES)
    assert sum(out2.values()) == 10
    assert set(out2) == set(assemble._DEF_CBIT_FIELDS)

    out3 = assemble._defcon_split(rng, 12, "MID", SHARES)
    assert sum(out3.values()) == 12
    assert set(out3) == set(assemble._MID_FWD_CBIRT_FIELDS)


def test_sample_fixture_returns_all_players_right_shape():
    home = [_player(1, "FWD"), _player(2, "DEF")]
    away = [_player(11, "FWD"), _player(12, "DEF")]
    rng = np.random.default_rng(1)
    out = assemble.sample_fixture(rng, home, away, 1.5, 1.2, 100, SHARES)
    assert set(out) == {1, 2, 11, 12}
    for arr in out.values():
        assert arr.shape == (100,)


def test_sample_fixture_certain_dnp_player_scores_zero():
    never_plays = _player(1, "FWD", p0=1.0, p1=0.0, p2=0.0, goal_weight=5.0)
    home = [never_plays, _player(2, "DEF", p0=1.0, p1=0.0, p2=0.0)]
    away = [_player(11, "FWD"), _player(12, "DEF")]
    rng = np.random.default_rng(2)
    out = assemble.sample_fixture(rng, home, away, 1.8, 1.8, 200, SHARES)
    assert (out[1] == 0).all()
    assert (out[2] == 0).all()


def test_sample_fixture_certain_starter_gets_appearance_points_every_scenario():
    always_plays = _player(1, "MID", p0=0.0, p1=0.0, p2=1.0)
    home = [always_plays]
    away = [_player(11, "MID", p0=0.0, p1=0.0, p2=1.0)]
    rng = np.random.default_rng(3)
    out = assemble.sample_fixture(rng, home, away, 0.0, 0.0, 200, SHARES)
    # with lam=0 (no goals ever) and zero weights, points floor is the
    # appearance points (2) -- bonus/cards/defcon only add, never subtract
    # below that here since yellow/red rates are 0.
    assert (out[1] >= 2).all()


def test_sample_fixture_prolific_striker_outscores_bench_warmer_on_average():
    striker = _player(1, "FWD", p2=0.95, p1=0.05, p0=0.0, goal_weight=0.6, assist_weight=0.1)
    fringe = _player(2, "FWD", p2=0.02, p1=0.03, p0=0.95, goal_weight=0.6, assist_weight=0.1)
    home = [striker, fringe]
    away = [_player(11, "DEF")]
    rng = np.random.default_rng(4)
    out = assemble.sample_fixture(rng, home, away, 2.2, 1.0, 3000, SHARES)
    assert out[1].mean() > out[2].mean() * 3


@pytest.mark.parametrize("n", [1, 2])
def test_sample_fixture_handles_tiny_scenario_counts(n):
    home = [_player(1, "FWD")]
    away = [_player(11, "DEF")]
    rng = np.random.default_rng(5)
    out = assemble.sample_fixture(rng, home, away, 1.0, 1.0, n, SHARES)
    assert out[1].shape == (n,)
