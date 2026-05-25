import asyncio
import logging
from datetime import datetime

import aiohttp
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import text

from config.settings import settings
from data.db import get_session
from data.models import Fixture, FixtureOdds

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"
REGIONS = "uk"
MARKETS = "h2h,totals"
ODDS_FORMAT = "decimal"


async def _fetch_odds(session: aiohttp.ClientSession) -> list[dict]:
    url = f"{BASE_URL}/sports/{SPORT}/odds/"
    params = {
        "apiKey": settings.the_odds_api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": "iso",
    }
    async with session.get(url, params=params) as resp:
        if resp.status == 401:
            raise RuntimeError("The Odds API: invalid API key")
        if resp.status == 422:
            raise RuntimeError("The Odds API: invalid parameters")
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        logger.info("Odds API: %s requests used, %s remaining", used, remaining)
        return await resp.json()


def _implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def _normalise(home: float, draw: float, away: float) -> tuple[float, float, float]:
    total = home + draw + away
    if total == 0:
        return 0.33, 0.33, 0.33
    return home / total, draw / total, away / total


def _extract_h2h(bookmakers: list[dict]) -> tuple[float, float, float]:
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] == "h2h":
                outcomes = {o["name"]: _implied_prob(o["price"]) for o in market["outcomes"]}
                home_teams = [k for k in outcomes if k not in ("Draw",)]
                if len(home_teams) >= 2:
                    names = sorted(home_teams)
                    return _normalise(
                        outcomes.get(names[0], 0.33),
                        outcomes.get("Draw", 0.33),
                        outcomes.get(names[1], 0.33),
                    )
    return 0.33, 0.33, 0.33


def _cs_from_h2h(home_win: float, draw: float, away_win: float) -> tuple[float, float]:
    home_cs = draw + away_win * 0.3
    away_cs = draw + home_win * 0.3
    return round(min(home_cs, 0.6), 3), round(min(away_cs, 0.6), 3)


def _match_fixture(
    home_team_name: str,
    away_team_name: str,
    commence_time: str,
    db_fixtures: list[dict],
) -> int | None:
    name_lower = home_team_name.lower()
    for fix in db_fixtures:
        if (
            fix["team_h_name"].lower() in name_lower
            or name_lower in fix["team_h_name"].lower()
        ):
            return fix["id"]
    return None


async def ingest_odds() -> int:
    if not settings.the_odds_api_key:
        logger.warning("THE_ODDS_API_KEY not set — skipping odds ingest")
        return 0

    db = get_session()
    try:
        rows = db.execute(text("""
            SELECT f.id, t_h.name AS team_h_name, t_a.name AS team_a_name, f.kickoff_time
            FROM fixtures f
            JOIN teams t_h ON t_h.id = f.team_h_id
            JOIN teams t_a ON t_a.id = f.team_a_id
            WHERE f.finished = 0
        """)).fetchall()
        db_fixtures = [
            {"id": r[0], "team_h_name": r[1], "team_a_name": r[2], "kickoff_time": r[3]}
            for r in rows
        ]
    finally:
        db.close()

    async with aiohttp.ClientSession() as session:
        odds_data = await _fetch_odds(session)

    upserted = 0
    db = get_session()
    try:
        for event in odds_data:
            fixture_id = _match_fixture(
                event.get("home_team", ""),
                event.get("away_team", ""),
                event.get("commence_time", ""),
                db_fixtures,
            )
            if fixture_id is None:
                logger.debug("No fixture match for %s vs %s", event.get("home_team"), event.get("away_team"))
                continue

            home_win, draw, away_win = _extract_h2h(event.get("bookmakers", []))
            home_cs, away_cs = _cs_from_h2h(home_win, draw, away_win)

            stmt = sqlite_insert(FixtureOdds).values(
                fixture_id=fixture_id,
                home_win_prob=round(home_win, 3),
                draw_prob=round(draw, 3),
                away_win_prob=round(away_win, 3),
                btts_prob=0.0,
                home_cs_prob=home_cs,
                away_cs_prob=away_cs,
                fetched_at=datetime.utcnow(),
            ).on_conflict_do_update(
                index_elements=["fixture_id"],
                set_={
                    "home_win_prob": round(home_win, 3),
                    "draw_prob": round(draw, 3),
                    "away_win_prob": round(away_win, 3),
                    "home_cs_prob": home_cs,
                    "away_cs_prob": away_cs,
                    "fetched_at": datetime.utcnow(),
                },
            )
            db.execute(stmt)
            upserted += 1

        db.commit()
        logger.info("Odds ingest complete: %d fixtures updated", upserted)
        return upserted
    finally:
        db.close()


def ingest_odds_sync() -> int:
    return asyncio.run(ingest_odds())
