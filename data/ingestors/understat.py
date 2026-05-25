import asyncio
import json
import logging
import re
from datetime import datetime

import aiohttp
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.models import Player, PlayerXGStats

logger = logging.getLogger(__name__)

UNDERSTAT_BASE = "https://understat.com"
UNDERSTAT_LEAGUE = "EPL"

SEASON_MAP = {
    "2022-23": "2022",
    "2023-24": "2023",
    "2024-25": "2024",
    "2026-27": "2026",
}


def _extract_json(html: str, var_name: str) -> list | dict | None:
    pattern = re.compile(
        rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)", re.DOTALL
    )
    match = pattern.search(html)
    if not match:
        return None
    raw = match.group(1).encode("utf-8").decode("unicode_escape")
    return json.loads(raw)


async def _fetch_league_players(
    session: aiohttp.ClientSession, season: str
) -> list[dict]:
    url = f"{UNDERSTAT_BASE}/league/{UNDERSTAT_LEAGUE}/{season}"
    async with session.get(url) as resp:
        resp.raise_for_status()
        html = await resp.text()
    data = _extract_json(html, "playersData")
    return data if isinstance(data, list) else []


async def _fetch_player_matches(
    session: aiohttp.ClientSession, player_id: str, season: str
) -> list[dict]:
    url = f"{UNDERSTAT_BASE}/player/{player_id}"
    async with session.get(url) as resp:
        resp.raise_for_status()
        html = await resp.text()
    data = _extract_json(html, "matchesData")
    if not isinstance(data, list):
        return []
    return [m for m in data if m.get("season") == season]


def _build_fpl_name_map() -> dict[str, int]:
    db = get_session()
    try:
        players = db.query(Player).all()
        name_map: dict[str, int] = {}
        for p in players:
            full = f"{p.first_name} {p.second_name}".lower()
            web = p.web_name.lower()
            name_map[full] = p.id
            name_map[web] = p.id
        return name_map
    finally:
        db.close()


def _match_player(understat_name: str, name_map: dict[str, int]) -> int | None:
    normalised = understat_name.strip().lower()
    if normalised in name_map:
        return name_map[normalised]
    for key, player_id in name_map.items():
        if normalised in key or key in normalised:
            return player_id
    return None


def _gw_from_date(match_date: str, understat_season: str) -> int | None:
    # Understat match objects carry no GW field. We derive an approximate GW
    # from the calendar date using known season-start anchors.
    # Precision of +/-1 GW is fine -- xG feeds rolling aggregates, not exact per-GW lookups.
    season_starts: dict[str, datetime] = {
        "2022": datetime(2022, 8, 5),
        "2023": datetime(2023, 8, 11),
        "2024": datetime(2024, 8, 16),
        "2026": datetime(2026, 8, 14),
    }
    start = season_starts.get(understat_season)
    if not start:
        return None
    try:
        match_dt = datetime.fromisoformat(match_date)
    except (ValueError, TypeError):
        return None
    days = (match_dt - start).days
    if days < 0:
        return None
    return min(max((days // 7) + 1, 1), 38)


async def ingest_understat_season(season_label: str, db_season: str) -> None:
    name_map = _build_fpl_name_map()
    inserted = 0
    skipped = 0

    connector = aiohttp.TCPConnector(limit=3)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; fpl-bot/1.0)"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        players = await _fetch_league_players(session, season_label)
        logger.info("Found %d players on Understat for %s", len(players), season_label)

        for player_data in players:
            player_name = player_data.get("player_name", "")
            db_player_id = _match_player(player_name, name_map)
            if not db_player_id:
                skipped += 1
                continue

            understat_id = player_data.get("id")
            if not understat_id:
                continue

            try:
                matches = await _fetch_player_matches(session, str(understat_id), season_label)
                await asyncio.sleep(0.3)
            except Exception as exc:
                logger.warning("Could not fetch matches for %s: %s", player_name, exc)
                continue

            db = get_session()
            try:
                for match in matches:
                    gw = _gw_from_date(match.get("date", ""), season_label)
                    if not gw:
                        continue
                    stmt = (
                        insert(PlayerXGStats)
                        .values(
                            player_id=db_player_id,
                            gameweek=gw,
                            season=db_season,
                            xg=float(match.get("xG", 0) or 0),
                            xa=float(match.get("xA", 0) or 0),
                            xgi=float(match.get("xG", 0) or 0) + float(match.get("xA", 0) or 0),
                            npxg=float(match.get("npxG", 0) or 0),
                            shots=int(match.get("shots", 0) or 0),
                            key_passes=int(match.get("key_passes", 0) or 0),
                        )
                        .on_conflict_do_nothing()
                    )
                    db.execute(stmt)
                    inserted += 1
                db.commit()
            finally:
                db.close()

    logger.info(
        "Understat %s: inserted %d xG rows, skipped %d unmatched",
        db_season, inserted, skipped,
    )


async def run_understat_ingest(seasons: list[str] | None = None) -> None:
    targets = seasons or list(SEASON_MAP.keys())
    for db_season in targets:
        understat_year = SEASON_MAP.get(db_season)
        if not understat_year:
            logger.warning("No understat mapping for season %s", db_season)
            continue
        logger.info("Ingesting Understat xG for %s", db_season)
        await ingest_understat_season(understat_year, db_season)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_understat_ingest())
