"""P12 — per-team DGW assembly (projection/assemble.py).

Real bug: ``assemble_gw_projections`` deduped its fixture-row scaffold on
just ``(player_id, gameweek)``, so a genuine double-gameweek player's SECOND
real fixture row (same gameweek, different opponent) was silently dropped
before any sampling ever happened — DGW players were projected as if they
only played once. Fixed by deduping on the fixture's own identity and
merging same-gameweek fixture contributions per player into one summed row
(so `xpts`/`xpts_mean`/`xpts_var` reflect BOTH fixtures) rather than either
truncating to one or emitting duplicate rows for the same player+gameweek
(which would corrupt any 1-row-per-player downstream merge, e.g.
``optimise_starting_xi``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from data.models import Base, ProjectionSample
from projection import assemble

VALUE_BY_PID = {1: 5.0, 2: 7.0, 3: 9.0}


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'dgw.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(assemble, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _history():
    rows = []
    for pid, team, pos in [(1, 10, "FWD"), (2, 20, "DEF"), (3, 30, "FWD")]:
        rows.append({
            "player_id": pid, "gameweek": 1, "season": "2099-00",
            "position": pos, "team_id_season": team,
            "xg": 0.3, "xa": 0.1, "key_passes": 1, "yellow_cards": 0, "red_cards": 0,
        })
    return pd.DataFrame(rows)


def _all_stats_dgw():
    """Team 10 plays TWICE in gameweek 2 (vs 20, then vs 30) -- a genuine
    double gameweek for player 1. Teams 20/30 each play once."""
    rows = [
        # fixture A: team 10 (home) vs team 20 (away)
        {"player_id": 1, "gameweek": 2, "team_id_season": 10,
         "opponent_team_id": 20, "was_home": True},
        {"player_id": 2, "gameweek": 2, "team_id_season": 20,
         "opponent_team_id": 10, "was_home": False},
        # fixture B: team 30 (home) vs team 10 (away) -- player 1's SECOND fixture this gw
        {"player_id": 3, "gameweek": 2, "team_id_season": 30,
         "opponent_team_id": 10, "was_home": True},
        {"player_id": 1, "gameweek": 2, "team_id_season": 10,
         "opponent_team_id": 30, "was_home": False},
    ]
    return pd.DataFrame(rows)


def _match_odds_dgw():
    return pd.DataFrame([
        {"gameweek": 2, "home_team_id": 10, "away_team_id": 20,
         "home_win_prob": 0.5, "draw_prob": 0.25, "away_win_prob": 0.25, "over25_prob": 0.5},
        {"gameweek": 2, "home_team_id": 30, "away_team_id": 10,
         "home_win_prob": 0.4, "draw_prob": 0.3, "away_win_prob": 0.3, "over25_prob": 0.5},
    ])


def _fake_sample_fixture(rng, home_players, away_players, lam_home, lam_away, n, shares):
    return {
        p["player_id"]: np.full(n, VALUE_BY_PID[p["player_id"]])
        for p in [*home_players, *away_players]
    }


def _patch_minutes_and_sampling(monkeypatch):
    monkeypatch.setattr(
        assemble, "predict_minutes_bands",
        lambda history, model: dict.fromkeys(range(1, 4), (0.0, 0.0, 1.0)),
    )
    monkeypatch.setattr(assemble, "sample_fixture", _fake_sample_fixture)


def test_dgw_player_gets_both_fixtures_summed_into_one_row(monkeypatch):
    _patch_minutes_and_sampling(monkeypatch)
    out = assemble.assemble_gw_projections(
        history=_history(), all_stats=_all_stats_dgw(), minutes_model=None,
        target_gw=2, horizon=1, match_odds=_match_odds_dgw(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1,
    )
    # exactly ONE row per player per gameweek -- no duplication, no truncation
    assert out["player_id"].tolist().count(1) == 1
    assert out["player_id"].tolist().count(2) == 1
    assert out["player_id"].tolist().count(3) == 1

    p1 = out[out["player_id"] == 1].iloc[0]
    p2 = out[out["player_id"] == 2].iloc[0]
    p3 = out[out["player_id"] == 3].iloc[0]

    # player 1 played TWO fixtures this gw -> doubled total (5.0 + 5.0)
    assert p1["xpts"] == pytest.approx(2 * VALUE_BY_PID[1])
    assert p1["xpts_mean"] == pytest.approx(2 * VALUE_BY_PID[1])
    # players 2 and 3 played only their single fixture -> untouched
    assert p2["xpts"] == pytest.approx(VALUE_BY_PID[2])
    assert p3["xpts"] == pytest.approx(VALUE_BY_PID[3])


def test_dgw_player_start_probability_is_p_at_least_one(monkeypatch):
    monkeypatch.setattr(
        assemble, "predict_minutes_bands",
        lambda history, model: {1: (0.4, 0.0, 0.6), 2: (0.0, 0.0, 1.0), 3: (0.0, 0.0, 1.0)},
    )
    monkeypatch.setattr(assemble, "sample_fixture", _fake_sample_fixture)
    out = assemble.assemble_gw_projections(
        history=_history(), all_stats=_all_stats_dgw(), minutes_model=None,
        target_gw=2, horizon=1, match_odds=_match_odds_dgw(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1,
    )
    p1 = out[out["player_id"] == 1].iloc[0]
    # P(starts >= 1 of 2 fixtures) = 1 - (1-0.6)(1-0.6) = 0.84, not a bare 0.6
    assert p1["start_probability"] == pytest.approx(1.0 - (1.0 - 0.6) * (1.0 - 0.6))


def test_dgw_player_persists_disjoint_scenario_ranges_per_fixture(session, monkeypatch):
    _patch_minutes_and_sampling(monkeypatch)
    n_scenarios = 4
    assemble.assemble_gw_projections(
        history=_history(), all_stats=_all_stats_dgw(), minutes_model=None,
        target_gw=2, horizon=1, match_odds=_match_odds_dgw(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=n_scenarios, seed=1, persist_samples=True, season="2099-00",
    )
    rows = session.execute(select(ProjectionSample)).scalars().all()
    p1_scenario_ids = sorted(r.scenario_id for r in rows if r.player_id == 1)
    # player 1 has samples from BOTH fixtures persisted -- 2*n_scenarios rows,
    # under the two fixtures' own disjoint ranges (unmerged at the raw-sample
    # level, only the reporting row above is summed)
    assert len(p1_scenario_ids) == 2 * n_scenarios
    assert p1_scenario_ids == list(range(n_scenarios)) + list(range(n_scenarios, 2 * n_scenarios))


def test_single_gameweek_player_unaffected(monkeypatch):
    """Regression guard: a normal (non-DGW) player's projection is byte-
    identical to the pre-fix single-fixture behaviour."""
    _patch_minutes_and_sampling(monkeypatch)
    out = assemble.assemble_gw_projections(
        history=_history(), all_stats=_all_stats_dgw(), minutes_model=None,
        target_gw=2, horizon=1, match_odds=_match_odds_dgw(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1,
    )
    p2 = out[out["player_id"] == 2].iloc[0]
    assert p2["xpts"] == pytest.approx(VALUE_BY_PID[2])
    assert p2["start_probability"] == pytest.approx(1.0)


# --- clean-sheet probability (2026-08-16) --------------------------------
#
# player_projections.cs_probability was 0.0 on every row ever written.
# assemble.py had each player's per-scenario clean sheet internally (it feeds
# the BPS simulator) but never surfaced it, so scripts/plot_analysis.py's
# clean-sheet-by-team chart was permanently blank — not a pre-season
# artefact, as it appeared. sample_team_goals draws Poisson(lambda), so
# P(concede 0) is exactly exp(-lambda_opponent) and needs no reduction over
# scenarios. These call the real assembler rather than restating the formula.


def _run(monkeypatch, bands=None):
    monkeypatch.setattr(
        assemble, "predict_minutes_bands",
        lambda history, model: bands or dict.fromkeys(range(1, 4), (0.0, 0.0, 1.0)),
    )
    monkeypatch.setattr(assemble, "sample_fixture", _fake_sample_fixture)
    return assemble.assemble_gw_projections(
        history=_history(), all_stats=_all_stats_dgw(), minutes_model=None,
        target_gw=2, horizon=1, match_odds=_match_odds_dgw(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1,
    )


def test_cs_probability_is_populated_at_all(monkeypatch):
    """The regression itself: this column was 0.0 for every row, always."""
    out = _run(monkeypatch)
    assert "cs_probability" in out.columns
    assert (out["cs_probability"] > 0).all()
    assert (out["cs_probability"] <= 1).all()


def test_cs_probability_matches_the_poisson_closed_form(monkeypatch):
    """Player 2 is on team 20, away to team 10 in a single fixture, so their
    clean-sheet chance is exp(-lambda of team 10) — read from the same
    odds-derived lambda the sampler draws goals with, not re-derived here."""
    import math

    from projection.team_goals import team_goals_from_odds

    lam_home, _ = team_goals_from_odds(0.5, 0.25, 0.25, 0.5)  # team 10 vs 20
    out = _run(monkeypatch)
    p2 = out[out["player_id"] == 2].iloc[0]
    assert p2["cs_probability"] == pytest.approx(math.exp(-lam_home))


def test_cs_probability_is_scaled_by_the_sixty_minute_chance(monkeypatch):
    """FPL awards a clean sheet only to a player who reaches 60 minutes, so a
    rotation risk on a watertight defence is not a certain clean sheet."""
    nailed = _run(monkeypatch)
    rotated = _run(monkeypatch, bands={1: (0.0, 0.0, 1.0), 2: (0.5, 0.0, 0.5), 3: (0.0, 0.0, 1.0)})

    nailed_p2 = nailed[nailed["player_id"] == 2].iloc[0]["cs_probability"]
    rotated_p2 = rotated[rotated["player_id"] == 2].iloc[0]["cs_probability"]
    assert rotated_p2 == pytest.approx(nailed_p2 * 0.5)


def test_cs_probability_combines_across_a_double_gameweek(monkeypatch):
    """Player 1 plays TWICE in gameweek 2, so they get two chances at a clean
    sheet — combined as P(at least one), matching how start_probability is
    merged for the same player."""
    out = _run(monkeypatch)
    p1 = out[out["player_id"] == 1].iloc[0]
    single_fixture_max = out[out["player_id"] != 1]["cs_probability"].max()
    assert p1["cs_probability"] > single_fixture_max, (
        "two fixtures must beat any single-fixture chance"
    )
    assert p1["cs_probability"] < 1.0
