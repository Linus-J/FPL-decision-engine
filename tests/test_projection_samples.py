"""P3-1: persisting P-COV scenario draws to ProjectionSample.

Monkeypatches predict_minutes_bands and sample_fixture so this exercises
just assemble_gw_projections's persistence/scenario-offset wiring, not the
MC sampling itself (already covered by tests/test_assemble.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from data.models import Base, ProjectionSample
from projection import assemble


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'samples.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(assemble, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _history():
    # two players per team, four teams -- just enough for _build_rolling_features
    rows = []
    for pid, team, pos in [
        (1, 10, "FWD"), (2, 10, "DEF"), (3, 20, "FWD"), (4, 20, "DEF"),
        (5, 30, "FWD"), (6, 40, "DEF"),
    ]:
        rows.append({
            "player_id": pid, "gameweek": 1, "season": "2099-00",
            "position": pos, "team_id_season": team,
            "xg": 0.3, "xa": 0.1, "key_passes": 1, "yellow_cards": 0, "red_cards": 0,
        })
    return pd.DataFrame(rows)


def _all_stats():
    # fixture A: team 10 (home) vs team 20 (away); fixture B: team 30 vs 40
    rows = []
    for pid, team, opp, home in [
        (1, 10, 20, True), (2, 10, 20, True), (3, 20, 10, False), (4, 20, 10, False),
        (5, 30, 40, True), (6, 40, 30, False),
    ]:
        rows.append({
            "player_id": pid, "gameweek": 2, "team_id_season": team,
            "opponent_team_id": opp, "was_home": home,
        })
    return pd.DataFrame(rows)


def _match_odds():
    return pd.DataFrame([
        {"gameweek": 2, "home_team_id": 10, "away_team_id": 20,
         "home_win_prob": 0.5, "draw_prob": 0.25, "away_win_prob": 0.25, "over25_prob": 0.5},
        {"gameweek": 2, "home_team_id": 30, "away_team_id": 40,
         "home_win_prob": 0.4, "draw_prob": 0.3, "away_win_prob": 0.3, "over25_prob": 0.5},
    ])


def _assemble_with(assemble_module, monkeypatch, **kwargs):
    """Same call as the persist_samples test, with only the sink/persist
    flags differing, so the two routes cannot drift apart."""
    monkeypatch.setattr(
        assemble_module, "predict_minutes_bands",
        lambda history, model: dict.fromkeys(range(1, 7), (0.0, 0.0, 1.0)),
    )

    def fake_sample_fixture(rng, home_players, away_players, lam_home, lam_away, n, shares):
        ids = [p["player_id"] for p in home_players] + [p["player_id"] for p in away_players]
        return {pid: np.arange(n, dtype=float) + pid * 100 for pid in ids}

    monkeypatch.setattr(assemble_module, "sample_fixture", fake_sample_fixture)

    return assemble_module.assemble_gw_projections(
        history=_history(), all_stats=_all_stats(), minutes_model=None,
        target_gw=2, horizon=1, match_odds=_match_odds(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1, **kwargs,
    )


def test_sample_sink_collects_rows_without_touching_the_db(session, monkeypatch):
    """sample_sink is the backtest's route to the raw draws: same rows
    persist_samples would write, but returned in memory and never inserted."""
    calls = []
    monkeypatch.setattr(assemble, "_write_projection_samples", lambda rows: calls.append(rows))

    sink = []
    df = _assemble_with(assemble, monkeypatch, sample_sink=sink, season="2099-00")

    assert not df.empty
    assert calls == [], "sample_sink must not write to the database"
    assert sink, "sample_sink must be populated"
    assert set(sink[0]) == {"player_id", "gameweek", "season", "scenario_id", "xpts"}
    assert len(sink) == 6 * 4
    assert session.execute(select(ProjectionSample)).scalars().all() == []


def test_sample_sink_and_persist_samples_agree_row_for_row(session, monkeypatch):
    """The two routes must not drift: one is the live path, one the backtest."""
    written = []
    monkeypatch.setattr(assemble, "_write_projection_samples", lambda rows: written.extend(rows))
    _assemble_with(assemble, monkeypatch, persist_samples=True, season="2099-00")

    sink = []
    _assemble_with(assemble, monkeypatch, sample_sink=sink, season="2099-00")

    assert sink == written


def test_sample_sink_requires_season(monkeypatch):
    with pytest.raises(ValueError, match="season"):
        _assemble_with(assemble, monkeypatch, sample_sink=[])


def test_persist_samples_writes_disjoint_scenario_ranges_per_fixture(session, monkeypatch):
    n_scenarios = 4
    monkeypatch.setattr(
        assemble, "predict_minutes_bands",
        lambda history, model: dict.fromkeys(range(1, 7), (0.0, 0.0, 1.0)),
    )

    def fake_sample_fixture(rng, home_players, away_players, lam_home, lam_away, n, shares):
        ids = [p["player_id"] for p in home_players] + [p["player_id"] for p in away_players]
        return {pid: np.arange(n, dtype=float) + pid * 100 for pid in ids}

    monkeypatch.setattr(assemble, "sample_fixture", fake_sample_fixture)

    out = assemble.assemble_gw_projections(
        history=_history(), all_stats=_all_stats(), minutes_model=None,
        target_gw=2, horizon=1, match_odds=_match_odds(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=n_scenarios, seed=1,
        persist_samples=True, season="2099-00",
    )
    assert set(out["player_id"]) == {1, 2, 3, 4, 5, 6}

    rows = session.execute(select(ProjectionSample)).scalars().all()
    assert len(rows) == 6 * n_scenarios

    by_player = {}
    for r in rows:
        by_player.setdefault(r.player_id, []).append(r.scenario_id)

    # fixture A (players 1-4) assembled first -> scenario_id range [0, n)
    for pid in (1, 2, 3, 4):
        assert sorted(by_player[pid]) == list(range(n_scenarios))
    # fixture B (players 5-6) assembled second -> disjoint range [n, 2n)
    for pid in (5, 6):
        assert sorted(by_player[pid]) == list(range(n_scenarios, 2 * n_scenarios))


def test_persist_samples_off_by_default_writes_nothing(session, monkeypatch):
    monkeypatch.setattr(
        assemble, "predict_minutes_bands",
        lambda history, model: dict.fromkeys(range(1, 7), (0.0, 0.0, 1.0)),
    )
    monkeypatch.setattr(
        assemble, "sample_fixture",
        lambda rng, home, away, lh, la, n, shares: {
            p["player_id"]: np.ones(n) for p in [*home, *away]
        },
    )
    assemble.assemble_gw_projections(
        history=_history(), all_stats=_all_stats(), minutes_model=None,
        target_gw=2, horizon=1, match_odds=_match_odds(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=3, seed=1,
    )
    assert session.execute(select(ProjectionSample)).scalars().all() == []


def test_persist_samples_requires_season():
    with pytest.raises(ValueError, match="season"):
        assemble.assemble_gw_projections(
            history=_history(), all_stats=_all_stats(), minutes_model=None,
            target_gw=2, horizon=1, match_odds=_match_odds(),
            defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
            persist_samples=True,
        )
