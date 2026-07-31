"""dashboard/data/news.py::get_injury_news"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dashboard.data.news import get_injury_news
from data.models import Base, Player, Team


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'news.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def _team(session, team_id: int = 1) -> Team:
    team = Team(id=team_id, name="Team A", short_name="TMA")
    session.add(team)
    session.commit()
    return team


def _player(session, player_id: int, team_id: int, status: str = "a", news: str = "") -> Player:
    p = Player(
        id=player_id, fpl_id=player_id, code=player_id,
        first_name="First", second_name="Last", web_name=f"Player{player_id}",
        team_id=team_id, position="MID", now_cost=6.0,
        status=status, news=news,
    )
    session.add(p)
    session.commit()
    return p


def test_no_news_returns_empty(session):
    _team(session)
    _player(session, 1, 1, status="a", news="")
    df = get_injury_news(session)
    assert df.empty


def test_injured_player_is_returned(session):
    _team(session)
    _player(session, 1, 1, status="i", news="Knee injury - Expected back 01 Sep")
    df = get_injury_news(session)
    assert len(df) == 1
    assert df.iloc[0]["web_name"] == "Player1"
    assert df.iloc[0]["in_squad"] is False or df.iloc[0]["in_squad"] == False  # noqa: E712


def test_in_squad_flag_set_for_given_ids(session):
    _team(session)
    _player(session, 1, 1, status="d", news="75% chance of playing")
    _player(session, 2, 1, status="i", news="Hamstring injury")
    df = get_injury_news(session, squad_ids=[1])
    row1 = df[df["player_id"] == 1].iloc[0]
    row2 = df[df["player_id"] == 2].iloc[0]
    assert bool(row1["in_squad"]) is True
    assert bool(row2["in_squad"]) is False
