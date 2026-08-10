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
from datetime import date
from pathlib import Path

import httpx
import yaml
from bs4 import BeautifulSoup
from sqlalchemy import text

from data.db import get_session
from data.models import Player
from data.overrides import OVERRIDES_PATH

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


def _today_str() -> str:
    return date.today().isoformat()


def scrape_confirmed_transfers(
    html: str, name_map: dict[str, int], pl_team_ids: dict[str, int],
) -> list[dict]:
    """(code, team_id, reason, as_of) candidates for config/transfer_overrides.yaml's
    `confirmed` list -- see this module's docstring for the page
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
        header_cell = in_table.select_one("th")
        header_text = header_cell.get_text(strip=True) if header_cell else None
        if header_text != "In":
            logger.warning(
                "transfermarkt: expected the first responsive-table block for "
                "%r to be the 'In' (arrivals) table, got header %r instead -- "
                "skipping this club's transfers entirely rather than risk "
                "reading departures as arrivals",
                club_name, header_text,
            )
            continue
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


def scrape_rumours(
    html: str, name_map: dict[str, int], min_assessment_pct: int = 40,
) -> list[dict]:
    """(code, p_leave, reason, as_of) candidates for
    config/transfer_overrides_candidates.yaml's `rumoured` shape -- see
    this module's docstring for the page structure this depends
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
        # Plain `a[title]` (not scoped to `table.inline-table`) -- the live
        # page wraps Player/Interested club in that inner table but NOT the
        # current Club cell, an inconsistency confirmed directly against the
        # live site 2026-08-10. A bare descendant search matches either way.
        player_link = cells[0].select_one("a[title]")
        club_link = cells[3].select_one("a[title]")
        interested_link = cells[4].select_one("a[title]")
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
    hand_written_codes = {e["code"] for e in hand_written}
    scraper_written = {e["code"]: e for e in existing if e.get("source") == "transfermarkt"}

    for candidate in candidates:
        if candidate["code"] in hand_written_codes:
            logger.warning(
                "transfermarkt: code %s already has a hand-written confirmed "
                "override -- skipping the scraped candidate so it never "
                "silently overrides a manual correction",
                candidate["code"],
            )
            continue
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
