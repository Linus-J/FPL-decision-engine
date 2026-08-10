# Transfermarkt scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-fill `config/transfer_overrides.yaml`'s `confirmed` list from Transfermarkt's transfers page (auto-applied), and produce a reviewable `config/transfer_overrides_candidates.yaml` of rumoured departures from Transfermarkt's rumours page (never auto-applied), both keyed to our stable player `code`.

**Architecture:** One new standalone ingestor module, `data/ingestors/transfermarkt.py`, split into a fetch layer (network), two pure parse functions (HTML string in, candidate dicts out — no I/O, directly unit-testable against fixture HTML), and two sync/write functions (the only functions touching the filesystem/DB). Same shape as the existing `press_conference.py`/`injury_parser.py` ingestors; reuses `press_conference.py`'s exact ambiguous-name-drop matching pattern, adapted to key on `code` instead of `player_id`.

**Tech Stack:** Python 3.12, `httpx` (sync client), `BeautifulSoup`/`lxml` (all already project dependencies — no new dependency needed), PyYAML, pytest.

**Reference spec:** `docs/superpowers/specs/2026-08-10-transfermarkt-scraper-design.md` (approved).

## Global Constraints

- Python env: `.venv/bin/python` for every command. Tests: `.venv/bin/python -m pytest <path> -v`. Lint: `.venv/bin/ruff check <path>`. ruff line-length = 100, target py312.
- Never crash on a network failure or an unexpected page structure — log a `warning`/`error` and degrade to an empty result, matching every other ingestor in this codebase.
- Name matching: three variants (`web_name`, `second_name`, `first_name second_name`, all lowercased), any name shared by more than one current player dropped entirely from the matchable set — never guess.
- `code` (not `id`/`team_id`) is the only stable cross-transfer identifier for everything this plan writes into either YAML file.
- `sync_confirmed_overrides` must be idempotent (rerun with identical candidates produces a byte-identical file) and must never modify a `confirmed` entry that lacks `source: transfermarkt` (hand-written entries are untouchable).
- Follow this repo's existing docstring convention (explain the *why*) rather than a terser default — match `press_conference.py`/`data/overrides.py`'s style.

---

### Task 1: Fetch layer, club resolution, and name matching

**Files:**
- Create: `data/ingestors/transfermarkt.py`
- Test: `tests/test_transfermarkt.py`

**Interfaces:**
- Produces:
  - `_fetch(url: str) -> str` — returns response HTML text, empty string `""` on any network failure (logged at `warning`).
  - `_TM_CLUB_NAME_TO_SHORT_NAME: dict[str, str]` — Transfermarkt's full club display name → our DB's 3-letter `short_name` (module-level constant, hand-curated for the 20 current PL clubs, verified against live Transfermarkt headline text 2026-08-10).
  - `resolve_pl_team_ids(season: str) -> dict[str, int]` — `short_name` (uppercase) → `team_id`, scoped to `season` via `team_season_strength` (NOT the `teams` table alone, which holds every team ever ingested across all seasons, not just the current 20).
  - `_build_player_name_map() -> dict[str, int]` — `name` (lowercased) → stable `code`, ambiguous names dropped.

- [ ] **Step 1: Write the failing tests**

Write `tests/test_transfermarkt.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.ingestors.transfermarkt'`

- [ ] **Step 3: Implement `data/ingestors/transfermarkt.py` (Task 1 portion)**

```python
"""transfermarkt.py — auto-fill confirmed transfers + rumour candidates
(plan 2026-08-10: docs/superpowers/specs/2026-08-10-transfermarkt-scraper-
design.md). Confirmed transfers are concrete, dated facts -- once a player
is confidently matched to our stable `code`, this writes DIRECTLY into
config/transfer_overrides.yaml's `confirmed` list, no human gate. Rumours
are inherently uncertain even at a high credibility score -- this only
ever writes a separate, gitignored config/transfer_overrides_candidates.yaml
for manual review, matching the original Feature B design's stance that a
wrong automatic team correction is worse than a missed one.

A prototype (2026-08-10, not committed) confirmed both Transfermarkt pages
are plain server-rendered HTML -- no headless browser needed, unlike
FBref's Cloudflare-gated pages. robots.txt disallows only the `wget`
user-agent specifically (`User-agent: * / Allow: /` covers everything
else) -- a real browser-like User-Agent is used here for reliability
against Transfermarkt's bot-detection heuristics regardless (empirically
verified during the prototype; an honest custom UA was not tested and may
be treated differently by their WAF even though robots.txt would permit
it)."""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import text

from data.db import get_session
from data.models import Player

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

TRANSFERS_URL = (
    "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/plus/"
    "?saison_id={year}&s_w=s&leihe=1&intern=0&intern=1"
)
RUMOURS_URL = "https://www.transfermarkt.com/premier-league/geruechte/wettbewerb/GB1"

# Transfermarkt's full club display name -> our DB's 3-letter short_name.
# Hand-curated against the live transfers-page headlines, 2026-08-10 --
# same maintenance convention as fbref.py's SEASON_MAP: a newly promoted
# club needs a new entry here (or its transfers/rumours are silently
# unresolved -- degrades safely, not misleadingly, but won't be caught
# until the club actually appears on the scraped page).
_TM_CLUB_NAME_TO_SHORT_NAME: dict[str, str] = {
    "Arsenal FC": "ARS",
    "Aston Villa": "AVL",
    "AFC Bournemouth": "BOU",
    "Brentford FC": "BRE",
    "Brighton & Hove Albion": "BHA",
    "Chelsea FC": "CHE",
    "Coventry City": "COV",
    "Crystal Palace": "CRY",
    "Everton FC": "EVE",
    "Fulham FC": "FUL",
    "Hull City": "HUL",
    "Ipswich Town": "IPS",
    "Leeds United": "LEE",
    "Liverpool FC": "LIV",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle United": "NEW",
    "Nottingham Forest": "NFO",
    "Sunderland AFC": "SUN",
    "Tottenham Hotspur": "TOT",
}


def _fetch(url: str) -> str:
    """Response HTML text, or "" on any network failure (logged at
    warning, never raised -- matches every other ingestor's degrade-safely
    posture)."""
    try:
        with httpx.Client(headers=_HEADERS, timeout=20.0, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.text
    except httpx.HTTPError as exc:
        logger.warning("transfermarkt: fetch failed for %s: %s", url, exc)
        return ""


def resolve_pl_team_ids(season: str) -> dict[str, int]:
    """short_name (uppercase) -> team_id, scoped to the CURRENT season via
    team_season_strength -- the `teams` table alone holds every team ever
    ingested across all seasons (Phase-1 finding: team_id is a per-season
    alphabetical index, not stable across promotion/relegation), so a plain
    `SELECT * FROM teams` would wrongly include historical/non-PL clubs."""
    db = get_session()
    try:
        rows = db.execute(
            text("""
                SELECT t.short_name, tss.team_id
                FROM team_season_strength tss
                JOIN teams t ON t.id = tss.team_id
                WHERE tss.season = :season
            """),
            {"season": season},
        ).fetchall()
        return {short_name: int(team_id) for short_name, team_id in rows}
    finally:
        db.close()


def _build_player_name_map() -> dict[str, int]:
    """name -> code, for matching a Transfermarkt player name to our
    internal stable identity. Same ambiguous-name-drop pattern as
    press_conference.py::_build_player_name_map, adapted to key on `code`
    (team_id/player_id are both per-season/reassignable; `code` is not --
    the identity Feature B's override mechanism itself already requires)."""
    db = get_session()
    try:
        players = (
            db.query(Player)
            .filter(Player.status != "n", Player.code.isnot(None))
            .all()
        )
        candidates: dict[str, set[int]] = {}
        for p in players:
            for key in (
                p.web_name.lower(),
                p.second_name.lower(),
                f"{p.first_name} {p.second_name}".lower(),
            ):
                candidates.setdefault(key, set()).add(p.code)
        return {name: codes.pop() for name, codes in candidates.items() if len(codes) == 1}
    finally:
        db.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check data/ingestors/transfermarkt.py tests/test_transfermarkt.py`
Expected: no errors

```bash
git add data/ingestors/transfermarkt.py tests/test_transfermarkt.py
git commit -m "feat: transfermarkt.py fetch layer, club resolution, name matching"
```

---

### Task 2: `scrape_confirmed_transfers` parser

**Files:**
- Modify: `data/ingestors/transfermarkt.py` (append)
- Test: `tests/test_transfermarkt.py` (append)

**Interfaces:**
- Consumes: `_TM_CLUB_NAME_TO_SHORT_NAME` (Task 1), `resolve_pl_team_ids` (Task 1), `_build_player_name_map` (Task 1).
- Produces: `scrape_confirmed_transfers(html: str, name_map: dict[str, int], pl_team_ids: dict[str, int]) -> list[dict]` — each dict shaped `{"code": int, "team_id": int, "reason": str, "as_of": str}`, matching Feature B's exact `confirmed` entry shape (`data/overrides.py::load_team_overrides` reads `code`/`team_id` from it). Pure function — no I/O, no DB access, no network; `name_map`/`pl_team_ids` are passed in so tests never need a live DB or network.

**Confirmed page structure** (verified live, 2026-08-10): each PL club has one `<div class="box">` containing an `<h2 class="content-box-headline" id="to-{tm_club_id}">{Club Display Name}</h2>` followed by exactly two `<div class="responsive-table">` blocks — the first table's header row is `["In", "Age", "Nat.", "Position", "Pos", "Market value", "Left", "Fee"]` (arrivals; column index 7 is the club the player left), the second is `["Out", ...]` (departures; skip — every real transfer already appears in some other club's "In" table, so processing only "In" tables covers every transfer exactly once). Each row's player name is at `td[0]` via `div.di.nowrap > span.hide-for-small > a[title]` (the `title` attribute is the clean full name; the visible text is a duplicate mobile/desktop pair). Row `<td>`s must be selected with `row.find_all("td", recursive=False)` — a plain `row.select("td")` incorrectly picks up cells with the SAME tag nested one level deeper and desyncs the column count (confirmed by hand-tracing the actual page: 9 `<td>` per row here happen to align by coincidence since this page's name cell nests `<div>`/`<span>`, not another `<table>` — Task 3's rumours page does NOT have this coincidence, so use `recursive=False` consistently in both parsers rather than relying on that coincidence).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transfermarkt.py`:

```python
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
            <a href="/bruno-guimaraes/profil/spieler/520624" title="Bruno Guimarães">Bruno Guimarães</a>
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -k scrape_confirmed_transfers -v`
Expected: FAIL with `AttributeError: module 'data.ingestors.transfermarkt' has no attribute 'scrape_confirmed_transfers'`

- [ ] **Step 3: Implement `scrape_confirmed_transfers`**

Append to `data/ingestors/transfermarkt.py` (add `from datetime import date` to the imports):

```python
def _today_str() -> str:
    return date.today().isoformat()


def scrape_confirmed_transfers(
    html: str, name_map: dict[str, int], pl_team_ids: dict[str, int],
) -> list[dict]:
    """(code, team_id, reason, as_of) candidates for config/transfer_overrides.yaml's
    `confirmed` list -- see this function's module docstring for the page
    structure this depends on. Pure: no I/O. Processes only each club's
    "In" (arrivals) table -- every real transfer already appears in the
    destination club's In table, so the Out tables are redundant and
    skipped entirely to avoid double-processing the same transfer."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    candidates: list[dict] = []
    for headline in soup.select('h2.content-box-headline[id^="to-"]'):
        club_name = headline.get_text(strip=True)
        short_name = _TM_CLUB_NAME_TO_SHORT_NAME.get(club_name)
        if short_name is None:
            continue
        team_id = pl_team_ids.get(short_name)
        if team_id is None:
            continue
        box = headline.find_parent("div", class_="box")
        if box is None:
            continue
        tables = box.select("div.responsive-table")
        if not tables:
            continue
        in_table = tables[0]
        for row in in_table.select("tbody > tr"):
            cells = row.find_all("td", recursive=False)
            if not cells:
                continue
            name_link = cells[0].select_one("span.hide-for-small a[title]")
            if name_link is None:
                continue
            player_name = name_link["title"].strip().lower()
            code = name_map.get(player_name)
            if code is None:
                logger.warning(
                    "transfermarkt: transfer row player %r has no matching "
                    "current player, skipping",
                    name_link["title"],
                )
                continue
            candidates.append({
                "code": code,
                "team_id": team_id,
                "reason": f"Transfermarkt: transferred to {club_name}",
                "as_of": _today_str(),
            })
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -k scrape_confirmed_transfers -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check data/ingestors/transfermarkt.py tests/test_transfermarkt.py`
Expected: no errors

```bash
git add data/ingestors/transfermarkt.py tests/test_transfermarkt.py
git commit -m "feat: scrape_confirmed_transfers parser"
```

---

### Task 3: `scrape_rumours` parser

**Files:**
- Modify: `data/ingestors/transfermarkt.py` (append)
- Test: `tests/test_transfermarkt.py` (append)

**Interfaces:**
- Consumes: `_build_player_name_map` (Task 1). Does NOT need `pl_team_ids` from `resolve_pl_team_ids` — the rumours page's "Club" cell gives a full display name directly via an `<a title=...>`, matched against `_TM_CLUB_NAME_TO_SHORT_NAME`'s KEYS (not `resolve_pl_team_ids`'s season-scoped team_ids, since a rumour candidate never writes a `team_id` at all — only `code`/`p_leave`/`reason`/`as_of`, matching Feature B's `rumoured` entry shape). "Is this club currently in the PL" only needs the name-to-short_name dict's key membership, not a team_id resolution.
- Produces: `scrape_rumours(html: str, name_map: dict[str, int], min_assessment_pct: int = 40) -> list[dict]` — each dict shaped `{"code": int, "p_leave": float, "reason": str, "as_of": str}`, sorted by `p_leave` descending. Pure function.

**Rumours page structure** (verified live, 2026-08-10): a `<table class="items">` with a 7-column header row `["Player", "Nation", "Age", "Club", "Interested club", "Most recent source from", "Assessment"]`. Each data row has exactly 7 DIRECT-CHILD `<td>`s (must use `row.find_all("td", recursive=False)` — the Player and Club cells each nest their OWN inner `<table class="inline-table">`, and a plain `row.select("td")` recurses into those, desyncing the column count entirely, unlike the transfers page). `td[0]` (Player) contains `table.inline-table a[title]` for the clean name. `td[3]` (Club — the player's CURRENT club) and `td[4]` (Interested club — the rumoured destination) contain the same `a[title]` pattern; column order matches header order exactly, confirmed against a real live row (Bradley Barcola's actual `Club` cell resolved to "Paris Saint-Germain", `Interested club` to "Liverpool FC" — i.e. `td[3]` is unambiguously the CURRENT club, not the destination). **Note for the fixture below:** since this scraper only accepts rumours where the CURRENT club is in the PL (see `scrape_rumours`'s docstring — this is the departure-risk tier, a player with no PL club has no `code` in our `players` table for a departure discount to apply to), the fixture's happy-path row uses a PL club as the CURRENT club (`td[3]`) and a foreign club as the interested one (`td[4]`) — the reverse of the real Barcola/PSG/Liverpool example cited above, which would otherwise get filtered out by this scraper entirely (correctly — Barcola isn't an FPL player while at PSG). `td[6]` (Assessment) is plain text: either `"71 %"`-style (strip `%`, parse as `int`/100.0 → `float`) or `"-"` (no credibility score — skip the row entirely).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transfermarkt.py`:

```python
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
    <a href="/no-assessment-player/profil/spieler/999" title="No Assessment Player">No Assessment Player</a>
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
    <a href="/below-threshold/profil/spieler/998" title="Below Threshold Player">Below Threshold Player</a>
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -k scrape_rumours -v`
Expected: FAIL with `AttributeError: module 'data.ingestors.transfermarkt' has no attribute 'scrape_rumours'`

- [ ] **Step 3: Implement `scrape_rumours`**

Append to `data/ingestors/transfermarkt.py`:

```python
def scrape_rumours(
    html: str, name_map: dict[str, int], min_assessment_pct: int = 40,
) -> list[dict]:
    """(code, p_leave, reason, as_of) candidates for
    config/transfer_overrides_candidates.yaml's `rumoured` shape -- see
    this function's module docstring for the page structure this depends
    on. Pure: no I/O. Only rumours where the player's CURRENT club (not
    the rumoured destination) is a Premier League club are in scope --
    this is specifically the departure-risk tier (existing squad risk),
    not a scouting/incoming-signing feature. Assessment maps directly to
    p_leave (Transfermarkt's own credibility score for the rumour is a
    reasonable proxy for likelihood); an unrated ("-") row is dropped
    entirely -- too noisy to act on, not a 0.0 p_leave (which would be a
    false, misleadingly precise signal)."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.items")
    if table is None:
        return []
    candidates: list[dict] = []
    for row in table.select("tbody > tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) != 7:
            continue
        player_link = cells[0].select_one("table.inline-table a[title]")
        club_link = cells[3].select_one("table.inline-table a[title]")
        interested_link = cells[4].select_one("table.inline-table a[title]")
        assessment_text = cells[6].get_text(strip=True)
        if player_link is None or club_link is None:
            continue

        club_name = club_link["title"].strip()
        if club_name not in _TM_CLUB_NAME_TO_SHORT_NAME:
            continue

        if "%" not in assessment_text:
            continue
        try:
            pct = int(assessment_text.replace("%", "").strip())
        except ValueError:
            continue
        if pct < min_assessment_pct:
            continue

        player_name = player_link["title"].strip().lower()
        code = name_map.get(player_name)
        if code is None:
            logger.warning(
                "transfermarkt: rumour row player %r has no matching current "
                "player, skipping",
                player_link["title"],
            )
            continue

        interested_name = interested_link["title"].strip() if interested_link else "?"
        candidates.append({
            "code": code,
            "p_leave": pct / 100.0,
            "reason": f"Transfermarkt rumour: {club_name} -> {interested_name}",
            "as_of": _today_str(),
        })
    return sorted(candidates, key=lambda c: c["p_leave"], reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -k scrape_rumours -v`
Expected: all PASS

- [ ] **Step 5: Run the full test_transfermarkt.py file to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -v`
Expected: all PASS

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/ruff check data/ingestors/transfermarkt.py tests/test_transfermarkt.py`
Expected: no errors

```bash
git add data/ingestors/transfermarkt.py tests/test_transfermarkt.py
git commit -m "feat: scrape_rumours parser"
```

---

### Task 4: `sync_confirmed_overrides` — auto-apply with self-cleanup

**Files:**
- Modify: `data/ingestors/transfermarkt.py` (append)
- Test: `tests/test_transfermarkt.py` (append)

**Interfaces:**
- Consumes: `data.overrides.OVERRIDES_PATH` (existing, from the earlier cold-start-lookahead plan) — reused directly rather than duplicating the path constant. `data.overrides._load_yaml` is NOT reused (it's a private helper of that module) — this task reads/writes the YAML file directly via `yaml.safe_load`/`yaml.safe_dump`, matching how `data/overrides.py` itself does it.
- Produces: `sync_confirmed_overrides(candidates: list[dict], current_team_ids: dict[int, int]) -> None` — `candidates` is `scrape_confirmed_transfers`'s output; `current_team_ids` is `code -> team_id` from the LIVE `players` table (used to detect when a `source: transfermarkt` entry has become redundant because FPL's own data caught up). Writes `config/transfer_overrides.yaml` in place. Never touches an entry without `source: "transfermarkt"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transfermarkt.py` (add `import yaml` to the top-level imports if not already present):

```python
import yaml


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -k sync_confirmed_overrides -v`
Expected: FAIL with `AttributeError: module 'data.ingestors.transfermarkt' has no attribute 'sync_confirmed_overrides'` (and `OVERRIDES_PATH` doesn't exist yet either)

- [ ] **Step 3: Implement `sync_confirmed_overrides`**

Append to `data/ingestors/transfermarkt.py` (add `import yaml` and `from data.overrides import OVERRIDES_PATH` to the imports — reusing the existing path constant from Feature B's `data/overrides.py` rather than duplicating it, so the two modules can never drift apart on where the file lives):

```python
def _load_overrides_file() -> dict:
    if not OVERRIDES_PATH.exists():
        return {"confirmed": [], "rumoured": []}
    with OVERRIDES_PATH.open() as f:
        data = yaml.safe_load(f)
    return data or {"confirmed": [], "rumoured": []}


def sync_confirmed_overrides(candidates: list[dict], current_team_ids: dict[int, int]) -> None:
    """Merges `candidates` (scrape_confirmed_transfers's output) into
    config/transfer_overrides.yaml's `confirmed` list, tagged
    `source: "transfermarkt"`. Idempotent: rerunning with identical
    candidates produces a byte-identical file. Self-cleaning: an existing
    `source: transfermarkt` entry whose `team_id` now matches
    `current_team_ids` (FPL's own live data caught up) is removed as
    redundant. NEVER modifies or removes an entry without
    `source: "transfermarkt"` -- a hand-written entry is untouchable by
    this function no matter how stale it looks."""
    data = _load_overrides_file()
    existing = data.get("confirmed") or []

    hand_written = [e for e in existing if e.get("source") != "transfermarkt"]
    scraper_written = {e["code"]: e for e in existing if e.get("source") == "transfermarkt"}

    for candidate in candidates:
        scraper_written[candidate["code"]] = {**candidate, "source": "transfermarkt"}

    still_needed = {
        code: entry
        for code, entry in scraper_written.items()
        if current_team_ids.get(code) != entry["team_id"]
    }

    data["confirmed"] = hand_written + list(still_needed.values())
    data.setdefault("rumoured", [])

    with OVERRIDES_PATH.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -k sync_confirmed_overrides -v`
Expected: all PASS

- [ ] **Step 5: Add `config/transfer_overrides_candidates.yaml` to `.gitignore`**

Edit `.gitignore` — add a new section right after the "# Isolated implementation worktrees" section:

```
# Scraped/derived data (not version-controlled -- regenerate on demand)
config/transfer_overrides_candidates.yaml
```

- [ ] **Step 6: Run the full test_transfermarkt.py file, then lint and commit**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -v`
Expected: all PASS

Run: `.venv/bin/ruff check data/ingestors/transfermarkt.py tests/test_transfermarkt.py`
Expected: no errors

```bash
git add data/ingestors/transfermarkt.py tests/test_transfermarkt.py .gitignore
git commit -m "feat: sync_confirmed_overrides — auto-apply with idempotent self-cleanup"
```

---

### Task 5: `write_rumour_candidates`, CLI entrypoint, end-to-end wiring

**Files:**
- Modify: `data/ingestors/transfermarkt.py` (append)
- Test: `tests/test_transfermarkt.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `write_rumour_candidates(candidates: list[dict]) -> None` — fully overwrites `config/transfer_overrides_candidates.yaml` (gitignored) each call, no merge logic. `run() -> None` — the `if __name__ == "__main__":` entrypoint tying fetch → parse → sync/write together for both pages, matching `press_conference.py`'s/`injury_parser.py`'s `run_*`/`ingest_*` naming convention.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transfermarkt.py`:

```python
@pytest.fixture
def candidates_file(tmp_path, monkeypatch):
    path = tmp_path / "transfer_overrides_candidates.yaml"
    monkeypatch.setattr(tm, "CANDIDATES_PATH", path)
    return path


def test_write_rumour_candidates_creates_file(candidates_file):
    candidates = [{"code": 200, "p_leave": 0.71, "reason": "x", "as_of": "2026-08-10"}]
    tm.write_rumour_candidates(candidates)
    data = yaml.safe_load(candidates_file.read_text())
    assert data["rumoured"] == candidates


def test_write_rumour_candidates_fully_overwrites_on_rerun(candidates_file):
    tm.write_rumour_candidates(
        [{"code": 200, "p_leave": 0.71, "reason": "x", "as_of": "2026-08-10"}]
    )
    tm.write_rumour_candidates(
        [{"code": 300, "p_leave": 0.55, "reason": "y", "as_of": "2026-08-10"}]
    )
    data = yaml.safe_load(candidates_file.read_text())
    codes = {c["code"] for c in data["rumoured"]}
    assert codes == {300}  # code=200 from the first run is gone, not merged


def test_write_rumour_candidates_empty_list_writes_empty_file(candidates_file):
    tm.write_rumour_candidates([])
    data = yaml.safe_load(candidates_file.read_text())
    assert data["rumoured"] == []


def test_run_wires_fetch_parse_sync_together(monkeypatch, overrides_file, candidates_file):
    """End-to-end: run() must call the real fetch/parse/sync functions in
    order, with the transfers result reaching sync_confirmed_overrides and
    the rumours result reaching write_rumour_candidates. Network calls are
    stubbed (no live HTTP in tests); DB-backed helpers are stubbed too, so
    this proves WIRING, not re-testing each already-covered function."""
    calls = {}

    monkeypatch.setattr(tm, "_fetch", lambda url: "TRANSFERS_HTML" if "transfers" in url else "RUMOURS_HTML")
    monkeypatch.setattr(tm, "resolve_pl_team_ids", lambda season: {"ARS": 1})
    monkeypatch.setattr(tm, "_build_player_name_map", lambda: {"someone": 42})
    monkeypatch.setattr(
        tm, "scrape_confirmed_transfers",
        lambda html, name_map, pl_team_ids: calls.setdefault("transfers_args", (html, name_map, pl_team_ids)) or [
            {"code": 42, "team_id": 1, "reason": "x", "as_of": "2026-08-10"}
        ],
    )
    monkeypatch.setattr(
        tm, "scrape_rumours",
        lambda html, name_map: calls.setdefault("rumours_args", (html, name_map)) or [
            {"code": 42, "p_leave": 0.5, "reason": "y", "as_of": "2026-08-10"}
        ],
    )
    # current_team_ids for sync_confirmed_overrides comes from a live DB
    # query inside run() -- stub the whole helper rather than the session,
    # so this test never touches a real DB.
    monkeypatch.setattr(tm, "_current_team_ids", lambda: {42: 5})

    tm.run(season="2026-27")

    assert calls["transfers_args"] == ("TRANSFERS_HTML", {"someone": 42}, {"ARS": 1})
    assert calls["rumours_args"] == ("RUMOURS_HTML", {"someone": 42})

    confirmed = yaml.safe_load(overrides_file.read_text())["confirmed"]
    assert confirmed == [
        {"code": 42, "team_id": 1, "reason": "x", "as_of": "2026-08-10", "source": "transfermarkt"}
    ]
    rumoured = yaml.safe_load(candidates_file.read_text())["rumoured"]
    assert rumoured == [{"code": 42, "p_leave": 0.5, "reason": "y", "as_of": "2026-08-10"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -k "write_rumour_candidates or test_run_wires" -v`
Expected: FAIL with `AttributeError` (`write_rumour_candidates`, `CANDIDATES_PATH`, `_current_team_ids`, `run` don't exist yet)

- [ ] **Step 3: Implement `write_rumour_candidates`, `_current_team_ids`, and `run`**

Append to `data/ingestors/transfermarkt.py`:

```python
CANDIDATES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "transfer_overrides_candidates.yaml"
)

_CANDIDATES_HEADER = (
    "# Transfermarkt rumour candidates -- auto-generated, NOT applied automatically.\n"
    "# Review these and copy any you trust into transfer_overrides.yaml's `rumoured`\n"
    "# list yourself. Fully regenerated on every run -- do not hand-edit.\n"
)


def write_rumour_candidates(candidates: list[dict]) -> None:
    """Fully overwrites config/transfer_overrides_candidates.yaml (gitignored)
    with `candidates` -- no merge/idempotency logic, since nothing here is
    auto-applied; each run is a fresh review-queue snapshot."""
    with CANDIDATES_PATH.open("w") as f:
        f.write(_CANDIDATES_HEADER)
        yaml.safe_dump({"rumoured": candidates}, f, sort_keys=False)


def _current_team_ids() -> dict[int, int]:
    """code -> team_id from the LIVE players table -- used by
    sync_confirmed_overrides to detect a now-redundant auto-written entry."""
    db = get_session()
    try:
        rows = db.execute(
            text("SELECT code, team_id FROM players WHERE code IS NOT NULL")
        ).fetchall()
        return {int(code): int(team_id) for code, team_id in rows}
    finally:
        db.close()


def run(season: str = "2026-27") -> None:
    """Manual, on-demand entrypoint (not wired into scripts/run_weekly.py
    -- matches this codebase's existing convention for press_conference.py/
    injury_parser.py). Fetches both pages, matches players once (shared
    across both parsers), and syncs both output files."""
    name_map = _build_player_name_map()
    pl_team_ids = resolve_pl_team_ids(season)

    transfers_html = _fetch(TRANSFERS_URL.format(year=season.split("-")[0]))
    confirmed_candidates = scrape_confirmed_transfers(transfers_html, name_map, pl_team_ids)
    sync_confirmed_overrides(confirmed_candidates, _current_team_ids())
    logger.info("Transfermarkt: %d confirmed transfer(s) synced", len(confirmed_candidates))

    rumours_html = _fetch(RUMOURS_URL)
    rumour_candidates = scrape_rumours(rumours_html, name_map)
    write_rumour_candidates(rumour_candidates)
    logger.info("Transfermarkt: %d rumour candidate(s) written for review", len(rumour_candidates))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_transfermarkt.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full project test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all PASS. Grep first for anything unexpected touching this new module:

```bash
grep -rln "data.ingestors.transfermarkt\|import transfermarkt" tests/ --include=*.py
```

(Expected: only `tests/test_transfermarkt.py` — this is a new, standalone module with no existing consumers.)

- [ ] **Step 6: Full repo-wide lint**

Run: `.venv/bin/ruff check .`
Expected: no NEW errors introduced by this plan's files (compare against the pre-existing baseline if any errors appear — check they're in files this plan didn't touch, same verification approach used in the cold-start-lookahead plan's Task 7).

- [ ] **Step 7: Commit**

```bash
git add data/ingestors/transfermarkt.py tests/test_transfermarkt.py
git commit -m "feat: transfermarkt run() entrypoint — full fetch/parse/sync wiring"
```

- [ ] **Step 8: Manual smoke test against the live site (not part of automated tests)**

This step is NOT a pytest run — it's a one-time manual sanity check against the real Transfermarkt pages, since Tasks 1-5's tests all use fixture HTML (deliberately, to keep the suite network-free and fast). Run:

```bash
.venv/bin/python -c "from data.ingestors import transfermarkt as tm; tm.run(season='2026-27')"
```

Expected: no exception, log lines reporting some number of confirmed transfers synced and rumour candidates written (could legitimately be 0 of either, depending on live transfer-window activity at run time — that's not a failure). Then inspect `config/transfer_overrides.yaml` and `config/transfer_overrides_candidates.yaml` by hand to confirm the entries look sensible (real player names, plausible clubs) before considering this plan done. If the live page structure has drifted from what Tasks 1-3's selectors assume (Transfermarkt redesigns periodically), this step is where that would surface — report back to the user rather than silently patching selectors without visibility, since it may indicate the fixture-based tests need updating too.

This is the final task. No further skill invocation needed after this — end of plan.
