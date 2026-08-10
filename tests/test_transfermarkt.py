"""transfermarkt.py — auto-fill confirmed transfers + rumour candidates
from Transfermarkt (plan 2026-08-10). Fetch layer, club-name resolution,
and player-name matching -- Task 1 of the plan. Parsers (Task 2/3) and
YAML sync (Task 4/5) are tested separately.
"""

from __future__ import annotations

import httpx
import pytest
import yaml
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


_RUMOURS_FIXTURE_HTML = """
<html><body>
<table class="items">
<thead><tr><th>Player</th><th>Nation</th><th>Age</th><th>Club</th>
<th>Interested club</th><th>Most recent source from</th><th>Assessment</th></tr></thead>
<tbody>
<tr>
  <td><table class="inline-table"><tr><td rowspan="2">
    <img alt="Bradley Barcola"/></td><td class="hauptlink">
    <a href="/bradley-barcola/profil/spieler/708265" title="Bradley Barcola">Bradley Barcola</a>
  </td></tr><tr><td>Left Winger</td></tr></table></td>
  <td></td>
  <td>23</td>
  <td><table class="inline-table"><tr><td rowspan="2"><a
    href="/fc-liverpool/startseite/verein/31" title="Liverpool FC"><img alt="Liverpool"/></a></td>
    <td class="hauptlink"><a href="/fc-liverpool/startseite/verein/31"
    title="Liverpool FC">Liverpool</a></td></tr></table></td>
  <td><table class="inline-table"><tr><td rowspan="2"><a
    href="/psg/startseite/verein/583" title="Paris Saint-Germain"><img alt="PSG"/></a></td>
    <td class="hauptlink"><a href="/psg/startseite/verein/583"
    title="Paris Saint-Germain">Paris Saint-Germain</a></td></tr></table></td>
  <td>10/08/2026</td>
  <td class="rechts hauptlink">71 %</td>
</tr>
<tr>
  <td><table class="inline-table"><tr><td rowspan="2">
    <img alt="No Assessment Player"/></td><td class="hauptlink">
    <a href="/no-assessment-player/profil/spieler/999"
    title="No Assessment Player">No Assessment Player</a>
  </td></tr><tr><td>Striker</td></tr></table></td>
  <td></td>
  <td>25</td>
  <td><table class="inline-table"><tr><td rowspan="2"><a
    href="/chelsea/startseite/verein/631" title="Chelsea FC"><img alt="Chelsea"/></a></td>
    <td class="hauptlink"><a href="/chelsea/startseite/verein/631"
    title="Chelsea FC">Chelsea</a></td></tr></table></td>
  <td></td>
  <td>09/08/2026</td>
  <td class="rechts hauptlink">-</td>
</tr>
<tr>
  <td><table class="inline-table"><tr><td rowspan="2">
    <img alt="Below Threshold Player"/></td><td class="hauptlink">
    <a href="/below-threshold/profil/spieler/998"
    title="Below Threshold Player">Below Threshold Player</a>
  </td></tr><tr><td>Defender</td></tr></table></td>
  <td></td>
  <td>27</td>
  <td><table class="inline-table"><tr><td rowspan="2"><a
    href="/everton/startseite/verein/29" title="Everton FC"><img alt="Everton"/></a></td>
    <td class="hauptlink"><a href="/everton/startseite/verein/29"
    title="Everton FC">Everton</a></td></tr></table></td>
  <td></td>
  <td>08/08/2026</td>
  <td class="rechts hauptlink">25 %</td>
</tr>
<tr>
  <td><table class="inline-table"><tr><td rowspan="2">
    <img alt="Non PL Player"/></td><td class="hauptlink">
    <a href="/non-pl-player/profil/spieler/997" title="Non PL Player">Non PL Player</a>
  </td></tr><tr><td>Midfielder</td></tr></table></td>
  <td></td>
  <td>24</td>
  <td><table class="inline-table"><tr><td rowspan="2"><a
    href="/bayern/startseite/verein/27" title="Bayern Munich"><img alt="Bayern"/></a></td>
    <td class="hauptlink"><a href="/bayern/startseite/verein/27"
    title="Bayern Munich">Bayern Munich</a></td></tr></table></td>
  <td></td>
  <td>07/08/2026</td>
  <td class="rechts hauptlink">90 %</td>
</tr>
</tbody>
</table>
</body></html>
"""


def test_scrape_rumours_matches_and_maps_assessment_to_p_leave():
    name_map = {"bradley barcola": 200}
    result = tm.scrape_rumours(_RUMOURS_FIXTURE_HTML, name_map)
    assert result == [
        {
            "code": 200,
            "p_leave": 0.71,
            "reason": "Transfermarkt rumour: Liverpool FC -> Paris Saint-Germain",
            "as_of": tm._today_str(),
        }
    ]


def test_scrape_rumours_skips_unrated_row():
    name_map = {"bradley barcola": 200, "no assessment player": 201}
    result = tm.scrape_rumours(_RUMOURS_FIXTURE_HTML, name_map)
    codes = {r["code"] for r in result}
    assert 201 not in codes  # "-" assessment, no credibility score at all


def test_scrape_rumours_drops_below_threshold():
    name_map = {"bradley barcola": 200, "below threshold player": 202}
    result = tm.scrape_rumours(_RUMOURS_FIXTURE_HTML, name_map, min_assessment_pct=40)
    codes = {r["code"] for r in result}
    assert 202 not in codes  # 25% < 40% floor


def test_scrape_rumours_drops_non_pl_current_club():
    # "Non PL Player" plays for Bayern Munich, not a Premier League club --
    # not a departure risk to an existing FPL squad, out of scope entirely.
    name_map = {"bradley barcola": 200, "non pl player": 203}
    result = tm.scrape_rumours(_RUMOURS_FIXTURE_HTML, name_map)
    codes = {r["code"] for r in result}
    assert 203 not in codes


def test_scrape_rumours_sorted_by_p_leave_descending():
    name_map = {"bradley barcola": 200, "below threshold player": 202}
    result = tm.scrape_rumours(_RUMOURS_FIXTURE_HTML, name_map, min_assessment_pct=0)
    p_leaves = [r["p_leave"] for r in result]
    assert p_leaves == sorted(p_leaves, reverse=True)


def test_scrape_rumours_unmatched_player_name_skipped():
    result = tm.scrape_rumours(_RUMOURS_FIXTURE_HTML, {})
    assert result == []


def test_scrape_rumours_empty_html_returns_empty_list():
    assert tm.scrape_rumours("", {}) == []


@pytest.fixture
def overrides_file(tmp_path, monkeypatch):
    path = tmp_path / "transfer_overrides.yaml"
    path.write_text("confirmed: []\nrumoured: []\n")
    monkeypatch.setattr(tm, "OVERRIDES_PATH", path)
    return path


def test_sync_confirmed_overrides_writes_new_entry_with_source_tag(overrides_file):
    candidates = [{"code": 100, "team_id": 1, "reason": "x", "as_of": "2026-08-10"}]
    tm.sync_confirmed_overrides(candidates, current_team_ids={100: 5})  # DB not caught up yet
    data = yaml.safe_load(overrides_file.read_text())
    assert data["confirmed"] == [
        {"code": 100, "team_id": 1, "reason": "x", "as_of": "2026-08-10", "source": "transfermarkt"}
    ]


def test_sync_confirmed_overrides_is_idempotent(overrides_file):
    candidates = [{"code": 100, "team_id": 1, "reason": "x", "as_of": "2026-08-10"}]
    tm.sync_confirmed_overrides(candidates, current_team_ids={100: 5})
    first = overrides_file.read_text()
    tm.sync_confirmed_overrides(candidates, current_team_ids={100: 5})
    second = overrides_file.read_text()
    assert first == second


def test_sync_confirmed_overrides_updates_existing_source_entry(overrides_file):
    tm.sync_confirmed_overrides(
        [{"code": 100, "team_id": 1, "reason": "old", "as_of": "2026-08-01"}],
        current_team_ids={100: 5},
    )
    tm.sync_confirmed_overrides(
        [{"code": 100, "team_id": 2, "reason": "new", "as_of": "2026-08-10"}],
        current_team_ids={100: 5},
    )
    data = yaml.safe_load(overrides_file.read_text())
    assert len(data["confirmed"]) == 1
    assert data["confirmed"][0]["team_id"] == 2
    assert data["confirmed"][0]["reason"] == "new"


def test_sync_confirmed_overrides_removes_entry_once_fpl_catches_up(overrides_file):
    tm.sync_confirmed_overrides(
        [{"code": 100, "team_id": 1, "reason": "x", "as_of": "2026-08-10"}],
        current_team_ids={100: 5},  # FPL not caught up -- entry needed
    )
    data = yaml.safe_load(overrides_file.read_text())
    assert len(data["confirmed"]) == 1

    # Rerun with NO new candidates, but FPL's own team_id now agrees (5 -> 1
    # matches what the override already corrected it to).
    tm.sync_confirmed_overrides([], current_team_ids={100: 1})
    data = yaml.safe_load(overrides_file.read_text())
    assert data["confirmed"] == []


def test_sync_confirmed_overrides_never_touches_hand_written_entry(overrides_file):
    overrides_file.write_text(yaml.safe_dump({
        "confirmed": [
            {"code": 999, "team_id": 7, "reason": "manually added", "as_of": "2026-07-01"},
        ],
        "rumoured": [],
    }))
    # Even though code=999's team_id (7) doesn't match "live" data (10),
    # a hand-written entry (no `source` field) must never be removed or
    # modified by the self-cleanup logic.
    tm.sync_confirmed_overrides([], current_team_ids={999: 10})
    data = yaml.safe_load(overrides_file.read_text())
    assert data["confirmed"] == [
        {"code": 999, "team_id": 7, "reason": "manually added", "as_of": "2026-07-01"}
    ]


def test_sync_confirmed_overrides_preserves_rumoured_list_untouched(overrides_file):
    overrides_file.write_text(yaml.safe_dump({
        "confirmed": [],
        "rumoured": [{"code": 555, "p_leave": 0.3, "reason": "r", "as_of": "2026-08-01"}],
    }))
    tm.sync_confirmed_overrides(
        [{"code": 100, "team_id": 1, "reason": "x", "as_of": "2026-08-10"}],
        current_team_ids={100: 5},
    )
    data = yaml.safe_load(overrides_file.read_text())
    assert data["rumoured"] == [
        {"code": 555, "p_leave": 0.3, "reason": "r", "as_of": "2026-08-01"}
    ]
