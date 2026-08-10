"""transfermarkt.py — auto-fill confirmed transfers + rumour candidates
from Transfermarkt (plan 2026-08-10). Fetch layer, club-name resolution,
and player-name matching -- Task 1 of the plan. Parsers (Task 2/3) and
YAML sync (Task 4/5) are tested separately.
"""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.ingestors import transfermarkt as tm
from data.models import Base, Player, Team, TeamSeasonStrength


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'tm.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(tm, "get_session", lambda: Local())
    return Local


def test_fetch_returns_html_on_success(monkeypatch):
    class _FakeResponse:
        text = "<html>ok</html>"
        def raise_for_status(self):
            pass

    class _FakeClient:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get(self, url):
            return _FakeResponse()

    monkeypatch.setattr(tm.httpx, "Client", _FakeClient)
    assert tm._fetch("https://example.invalid") == "<html>ok</html>"


def test_fetch_returns_empty_string_on_network_failure(monkeypatch, caplog):
    class _FakeClient:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get(self, url):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(tm.httpx, "Client", _FakeClient)
    import logging
    with caplog.at_level(logging.WARNING):
        result = tm._fetch("https://example.invalid")
    assert result == ""
    assert "boom" in caplog.text or "failed" in caplog.text.lower()


def test_tm_club_name_to_short_name_covers_all_current_clubs(temp_session):
    """Every club name this module maps must resolve to a short_name that
    actually exists in the live teams table -- catches a typo in either
    the hand-curated dict or a club rename before it silently drops
    matches at runtime."""
    s = temp_session()
    try:
        for short_name in set(tm._TM_CLUB_NAME_TO_SHORT_NAME.values()):
            s.add(Team(name=short_name, short_name=short_name))
        s.commit()
        db_short_names = {row[0] for row in s.execute(
            __import__("sqlalchemy").text("SELECT short_name FROM teams")
        )}
    finally:
        s.close()
    assert set(tm._TM_CLUB_NAME_TO_SHORT_NAME.values()) <= db_short_names


def test_resolve_pl_team_ids_scopes_to_season(temp_session):
    s = temp_session()
    try:
        s.add(Team(id=1, name="Arsenal", short_name="ARS"))
        s.add(Team(id=2, name="Leeds", short_name="LEE"))
        # id=2 (Leeds) is NOT in the current season's TeamSeasonStrength --
        # simulates a team that existed in a prior season but isn't in the
        # PL this season (or vice versa: a team ingested historically that
        # shouldn't be treated as a current destination).
        s.add(TeamSeasonStrength(season="2026-27", team_id=1, code=11))
        s.commit()
    finally:
        s.close()

    result = tm.resolve_pl_team_ids("2026-27")
    assert result == {"ARS": 1}


def test_resolve_pl_team_ids_empty_when_no_current_season_rows(temp_session):
    assert tm.resolve_pl_team_ids("2026-27") == {}


def test_build_player_name_map_matches_all_three_variants(temp_session):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=100, first_name="Bruno", second_name="Guimarães",
                     web_name="B.Guimarães", team_id=1, position="MID", now_cost=6.5,
                     status="a"))
        s.commit()
    finally:
        s.close()

    name_map = tm._build_player_name_map()
    assert name_map["b.guimarães"] == 100
    assert name_map["guimarães"] == 100
    assert name_map["bruno guimarães"] == 100


def test_build_player_name_map_drops_ambiguous_names(temp_session):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=100, first_name="A", second_name="Gabriel",
                     web_name="Gabriel", team_id=1, position="DEF", now_cost=5.0,
                     status="a"))
        s.add(Player(fpl_id=2, code=200, first_name="B", second_name="Gabriel",
                     web_name="Gabriel", team_id=2, position="MID", now_cost=6.0,
                     status="a"))
        s.commit()
    finally:
        s.close()

    name_map = tm._build_player_name_map()
    assert "gabriel" not in name_map
