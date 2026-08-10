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


def test_tm_club_name_to_short_name_covers_all_current_clubs():
    """Independent check (NOT derived from the dict under test -- a prior
    version of this test seeded its fixture DB from
    _TM_CLUB_NAME_TO_SHORT_NAME.values() itself, making it tautological
    and unable to catch a wrong entry). This hardcodes the 20 real
    short_names this project's own live DB actually uses for the current
    season (verified directly against team_season_strength, 2026-08-10),
    independent of anything in transfermarkt.py."""
    expected_short_names = {
        "ARS", "AVL", "BOU", "BRE", "BHA", "CHE", "COV", "CRY", "EVE", "FUL",
        "HUL", "IPS", "LEE", "LIV", "MCI", "MUN", "NEW", "NFO", "SUN", "TOT",
    }
    assert set(tm._TM_CLUB_NAME_TO_SHORT_NAME.values()) == expected_short_names
    # every TM display name maps to a DISTINCT short_name -- no two club
    # names collapsing onto the same team by mistake
    assert len(tm._TM_CLUB_NAME_TO_SHORT_NAME) == len(expected_short_names)


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


_TRANSFERS_FIXTURE_HTML = """
<html><body>
<div class="box">
  <h2 class="content-box-headline" id="to-11">Arsenal FC</h2>
  <div class="responsive-table">
    <table>
      <thead><tr><th>In</th><th>Age</th><th>Nat.</th><th>Position</th><th>Pos</th>
      <th>Market value</th><th>Left</th><th>Fee</th></tr></thead>
      <tbody>
        <tr>
          <td><div class="di nowrap"><span class="hide-for-small">
            <a href="/bruno-guimaraes/profil/spieler/520624"
               title="Bruno Guimarães">Bruno Guimarães</a>
          </span></div></td>
          <td>28</td><td></td><td>Central Midfield</td><td>CM</td>
          <td>&euro;70.00m</td><td>Newcastle</td><td>&euro;87.50m</td>
        </tr>
        <tr>
          <td><div class="di nowrap"><span class="hide-for-small">
            <a href="/unknown-player/profil/spieler/1" title="Unknown Newbie">Unknown Newbie</a>
          </span></div></td>
          <td>19</td><td></td><td>Forward</td><td>FW</td>
          <td>&euro;1.00m</td><td>Some Lower League Club</td><td>&euro;0.50m</td>
        </tr>
      </tbody>
    </table>
  </div>
  <div class="responsive-table">
    <table>
      <thead><tr><th>Out</th><th>Age</th><th>Nat.</th><th>Position</th><th>Pos</th>
      <th>Market value</th><th>Joined</th><th>Fee</th></tr></thead>
      <tbody>
        <tr>
          <td><div class="di nowrap"><span class="hide-for-small">
            <a href="/some-outgoing/profil/spieler/2" title="Some Outgoing">Some Outgoing</a>
          </span></div></td>
          <td>30</td><td></td><td>Midfielder</td><td>MF</td>
          <td>&euro;5.00m</td><td>Chelsea FC</td><td>&euro;3.00m</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
</body></html>
"""


def test_scrape_confirmed_transfers_matches_and_resolves():
    name_map = {"bruno guimarães": 100}  # "Unknown Newbie" deliberately absent
    pl_team_ids = {"ARS": 1}
    result = tm.scrape_confirmed_transfers(_TRANSFERS_FIXTURE_HTML, name_map, pl_team_ids)
    assert result == [
        {
            "code": 100,
            "team_id": 1,
            "reason": "Transfermarkt: transferred to Arsenal FC",
            "as_of": tm._today_str(),
        }
    ]


def test_scrape_confirmed_transfers_skips_unmatched_player_name():
    # "Unknown Newbie" has no entry in name_map -- must be silently skipped,
    # not raise, and must not appear in the result at all.
    name_map = {}
    pl_team_ids = {"ARS": 1}
    result = tm.scrape_confirmed_transfers(_TRANSFERS_FIXTURE_HTML, name_map, pl_team_ids)
    assert result == []


def test_scrape_confirmed_transfers_skips_unresolvable_club():
    # pl_team_ids has no "ARS" entry -- Arsenal's whole box must be skipped.
    name_map = {"bruno guimarães": 100}
    pl_team_ids = {}
    result = tm.scrape_confirmed_transfers(_TRANSFERS_FIXTURE_HTML, name_map, pl_team_ids)
    assert result == []


def test_scrape_confirmed_transfers_ignores_out_table():
    # "Some Outgoing" is only in the "Out" table -- must never appear,
    # even if somehow matchable (Out-table transfers are covered by the
    # DESTINATION club's own "In" table instead).
    name_map = {"bruno guimarães": 100, "some outgoing": 999}
    pl_team_ids = {"ARS": 1}
    result = tm.scrape_confirmed_transfers(_TRANSFERS_FIXTURE_HTML, name_map, pl_team_ids)
    codes = {r["code"] for r in result}
    assert 999 not in codes


def test_scrape_confirmed_transfers_empty_html_returns_empty_list():
    assert tm.scrape_confirmed_transfers("", {}, {}) == []
