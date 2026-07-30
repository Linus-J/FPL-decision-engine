"""data/ingestors/fpl_api.py::upsert_players -- marking players absent from
the live bootstrap as departed.

Real bug found 2026-07-30 (the user's own review of a drafted initial
squad: "goalkeepers, Akinmboni, Casemiro, Abdullahi, Fraser are not in
this season"). Verified live: Casemiro left Manchester United on a free
transfer to Inter Miami CF when his contract expired -- genuinely gone
from the Premier League. upsert_players only ever added/updated players
PRESENT in the current bootstrap; a player who left the league entirely
just vanished from FPL's own "elements" list, but their row sat at
whatever stale status ('a') they had at their last successful sync,
forever, since nothing ever revisited a row absent from every subsequent
fetch.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import data.ingestors.fpl_api as fpl_api
from data.models import Base, Player


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'upsert_players.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fpl_api, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _element(code: int, web_name: str, team: int = 1, status: str = "a") -> dict:
    return {
        "id": code, "code": code, "first_name": "A", "second_name": web_name,
        "web_name": web_name, "team": team, "element_type": 3, "now_cost": 50,
        "cost_change_start": 0, "status": status, "news": "",
        "selected_by_percent": "0", "form": "0", "total_points": 0, "minutes": 0,
        "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 0,
        "saves": 0, "yellow_cards": 0, "red_cards": 0, "bonus": 0, "bps": 0,
        "ict_index": "0", "influence": "0", "creativity": "0", "threat": "0",
        "chance_of_playing_next_round": None, "chance_of_playing_this_round": None,
        "transfers_in_event": 0, "transfers_out_event": 0,
    }


def test_upsert_players_marks_a_player_missing_from_the_bootstrap_as_departed(session):
    fpl_api.upsert_players({"elements": [_element(1, "Stays"), _element(2, "Leaves")]})
    # Next sync: "Leaves" (code=2) no longer appears in the bootstrap at all.
    fpl_api.upsert_players({"elements": [_element(1, "Stays")]})

    stays = session.query(Player).filter_by(code=1).one()
    leaves = session.query(Player).filter_by(code=2).one()
    assert stays.status == "a"
    assert leaves.status == "u"
    assert "departed" in leaves.news.lower()


def test_upsert_players_does_not_touch_players_still_present(session):
    fpl_api.upsert_players({"elements": [_element(1, "Stays", status="a")]})
    fpl_api.upsert_players({"elements": [_element(1, "Stays", status="a")]})
    stays = session.query(Player).filter_by(code=1).one()
    assert stays.status == "a"


def test_upsert_players_does_not_downgrade_an_already_departed_player_twice(session):
    fpl_api.upsert_players({"elements": [_element(1, "Stays"), _element(2, "Leaves")]})
    fpl_api.upsert_players({"elements": [_element(1, "Stays")]})
    left_at = session.query(Player).filter_by(code=2).one().updated_at

    # A later sync where they're STILL absent must not keep bumping updated_at
    # every single run (only the transition a->u should log/touch the row).
    fpl_api.upsert_players({"elements": [_element(1, "Stays")]})
    still_left_at = session.query(Player).filter_by(code=2).one().updated_at
    assert left_at == still_left_at
