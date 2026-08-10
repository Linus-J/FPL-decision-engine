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
