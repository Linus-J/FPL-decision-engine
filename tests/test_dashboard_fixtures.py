"""dashboard/data/fixtures.py"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import dashboard.data.fixtures as fixtures_module
from data.models import Base, Fixture, Player, Team


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fixtures.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def test_get_upcoming_fixtures_filters_by_window(session, monkeypatch):
    session.add_all([
        Team(id=1, name="Team A", short_name="TMA",
             strength_overall_home=4, strength_overall_away=5),
        Team(id=2, name="Team B", short_name="TMB",
             strength_overall_home=2, strength_overall_away=3),
    ])
    session.add_all([
        Fixture(fpl_id=1, season="2026-27", gameweek=6, team_h_id=1, team_a_id=2,
                kickoff_time=datetime(2026, 9, 1)),
        Fixture(fpl_id=2, season="2026-27", gameweek=20, team_h_id=1, team_a_id=2,
                kickoff_time=datetime(2027, 1, 1)),
    ])
    session.commit()

    monkeypatch.setattr(fixtures_module, "_get_current_and_next_gw", lambda: (5, 6))
    monkeypatch.setattr(fixtures_module, "_get_current_season", lambda: "2026-27")

    df = fixtures_module.get_upcoming_fixtures(session, lookahead_gws=8)

    assert list(df["gameweek"]) == [6]
    assert df.iloc[0]["home"] == "TMA"
    assert df.iloc[0]["away"] == "TMB"
    assert df.iloc[0]["home_fdr"] == 3  # TMB's strength_overall_away
    assert df.iloc[0]["away_fdr"] == 4  # TMA's strength_overall_home
    assert pd.api.types.is_datetime64_any_dtype(df["kickoff_time"])


def test_get_squad_dgw_exposure_empty_without_squad(session):
    assert fixtures_module.get_squad_dgw_exposure(session, []) == {}


def test_get_squad_dgw_exposure_delegates_to_coverage_helper(session, monkeypatch):
    session.add(Team(id=1, name="Team A", short_name="TMA"))
    session.add(Player(
        id=1, fpl_id=101, code=101, first_name="A", second_name="One", web_name="One",
        team_id=1, position="MID", now_cost=6.0,
    ))
    session.commit()

    monkeypatch.setattr(fixtures_module, "_get_dgw_gameweeks", lambda lookahead: {12})
    monkeypatch.setattr(
        fixtures_module, "get_latest_projections",
        lambda **_: pd.DataFrame({"player_id": [1], "gameweek": [12], "xpts": [8.0]}),
    )

    captured = {}

    def fake_coverage(squad_ids, players, dgw_gws, projections):
        captured["squad_ids"] = squad_ids
        captured["dgw_gws"] = dgw_gws
        return {12: {"squad_players_involved": 1, "combined_xpts": 8.0}}

    monkeypatch.setattr(fixtures_module, "get_dgw_coverage", fake_coverage)

    result = fixtures_module.get_squad_dgw_exposure(session, [1])

    assert result == {12: {"squad_players_involved": 1, "combined_xpts": 8.0}}
    assert captured["squad_ids"] == [1]
    assert captured["dgw_gws"] == {12}
