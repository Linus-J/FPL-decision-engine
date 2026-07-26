"""P3-0: live-serving projections via the P10 MC assembly.

Covers the two new live-only loaders (fixture context + match odds — the
backtest path gets these from played player_gw_stats rows, which don't
exist yet for a live, unplayed fixture) and run_projections's cold-start
fallback. The full assemble.py MC path itself is already covered by
tests/test_assemble.py and the live backtest harness; these tests are
about the live-specific plumbing around it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Fixture, FixtureOdds, Gameweek, Player, Team
from projection import assemble, pipeline


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'pipeline.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(pipeline, "get_session", lambda: Local())
    monkeypatch.setattr(assemble, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _team(id_, name):
    return Team(id=id_, name=name, short_name=name[:3].upper())


def _player(id_, team_id, position="MID"):
    return Player(
        id=id_, fpl_id=id_, code=id_, first_name="P", second_name=str(id_),
        web_name=f"P{id_}", team_id=team_id, position=position, now_cost=5.0,
    )


def test_build_live_fixture_context_resolves_opponent_and_home_away(session):
    session.add_all([_team(1, "Home"), _team(2, "Away")])
    session.add_all([_player(10, team_id=1), _player(20, team_id=2)])
    session.add(Fixture(id=1, fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
    session.commit()

    out = pipeline._build_live_fixture_context("2026-27", [1])
    home_row = out[out["player_id"] == 10].iloc[0]
    away_row = out[out["player_id"] == 20].iloc[0]
    assert bool(home_row["was_home"]) is True
    assert home_row["opponent_team_id"] == 2
    assert bool(away_row["was_home"]) is False
    assert away_row["opponent_team_id"] == 1


def test_build_live_fixture_context_empty_gws_returns_empty_shape(session):
    out = pipeline._build_live_fixture_context("2026-27", [])
    assert out.empty
    assert list(out.columns) == [
        "player_id", "gameweek", "team_id_season", "opponent_team_id", "was_home",
    ]


def test_build_live_fixture_context_no_fixture_for_gw_is_empty(session):
    session.add_all([_team(1, "Home"), _team(2, "Away")])
    session.add(_player(10, team_id=1))
    session.commit()
    out = pipeline._build_live_fixture_context("2026-27", [5])
    assert out.empty


def test_load_live_match_odds_uses_latest_fetch_before_deadline(session):
    session.add_all([_team(1, "Home"), _team(2, "Away")])
    session.add(Gameweek(id=1, season="2026-27", name="GW1", deadline_time=datetime(2026, 8, 15)))
    session.add(Fixture(id=1, fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
    session.commit()
    # a stale early fetch, a good pre-deadline fetch, and a leaky post-deadline fetch
    session.add(FixtureOdds(
        fixture_id=1, home_win_prob=0.40, draw_prob=0.30, away_win_prob=0.30,
        over25_prob=0.50, fetched_at=datetime(2026, 8, 10),
    ))
    session.add(FixtureOdds(
        fixture_id=1, home_win_prob=0.55, draw_prob=0.25, away_win_prob=0.20,
        over25_prob=0.60, fetched_at=datetime(2026, 8, 14),
    ))
    session.add(FixtureOdds(
        fixture_id=1, home_win_prob=0.99, draw_prob=0.005, away_win_prob=0.005,
        over25_prob=0.99, fetched_at=datetime(2026, 8, 15) + timedelta(minutes=1),
    ))
    session.commit()

    out = pipeline._load_live_match_odds("2026-27", [1])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["home_win_prob"] == pytest.approx(0.55)  # the latest BEFORE the deadline


def test_load_live_match_odds_empty_gws_returns_empty_shape(session):
    out = pipeline._load_live_match_odds("2026-27", [])
    assert out.empty
    assert "home_win_prob" in out.columns


def test_run_projections_cold_start_returns_empty_not_crash(session):
    # no player_gw_stats rows for the season at all (GW1, season hasn't
    # started) -- assemble.load_all_stats returns empty, run_projections
    # must return gracefully, not crash
    session.add(Gameweek(id=1, season="2026-27", name="GW1", is_next=True,
                         deadline_time=datetime(2026, 8, 15)))
    session.commit()
    out = pipeline.run_projections(season="2026-27", horizon=1, persist=False)
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert set(out.columns) == {
        "player_id", "gameweek", "xpts", "xpts_mean", "xpts_var", "start_probability",
    }
