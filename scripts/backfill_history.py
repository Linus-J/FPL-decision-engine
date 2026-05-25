"""
Pulls 3 seasons of historical FPL GW data from vaastav/Fantasy-Premier-League.

Run once before training the projection models:
    python -m scripts.backfill_history

Data lands in player_gw_stats table, tagged with the correct season string.
"""

import asyncio
import io
import logging

import pandas as pd
import httpx
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session, init_db
from data.models import PlayerGameweekStats, Player

logger = logging.getLogger(__name__)

VAASTAV_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

SEASONS = [
    ("2022-23", "2022-23"),
    ("2023-24", "2023-24"),
    ("2024-25", "2024-25"),
]

GW_CSV_URL = "{base}/{season}/gws/merged_gw.csv"


async def _fetch_csv(client: httpx.AsyncClient, url: str) -> pd.DataFrame | None:
    try:
        resp = await client.get(url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        logger.warning("Could not fetch %s: %s", url, exc)
        return None


def _build_player_name_map() -> dict[str, int]:
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


def _fuzzy_match(name: str, name_map: dict[str, int]) -> int | None:
    normalised = name.strip().lower()
    if normalised in name_map:
        return name_map[normalised]

    for key, player_id in name_map.items():
        if normalised in key or key in normalised:
            return player_id

    return None


def _ingest_dataframe(df: pd.DataFrame, season: str, name_map: dict[str, int]) -> int:
    db = get_session()
    inserted = 0
    try:
        for _, row in df.iterrows():
            player_name = str(row.get("name", ""))
            player_id = _fuzzy_match(player_name, name_map)
            if not player_id:
                continue

            gw = int(row.get("GW", 0))
            if not gw:
                continue

            stmt = (
                insert(PlayerGameweekStats)
                .values(
                    player_id=player_id,
                    gameweek=gw,
                    season=season,
                    total_points=int(row.get("total_points", 0) or 0),
                    minutes=int(row.get("minutes", 0) or 0),
                    goals_scored=int(row.get("goals_scored", 0) or 0),
                    assists=int(row.get("assists", 0) or 0),
                    clean_sheets=int(row.get("clean_sheets", 0) or 0),
                    goals_conceded=int(row.get("goals_conceded", 0) or 0),
                    saves=int(row.get("saves", 0) or 0),
                    yellow_cards=int(row.get("yellow_cards", 0) or 0),
                    red_cards=int(row.get("red_cards", 0) or 0),
                    bonus=int(row.get("bonus", 0) or 0),
                    bps=int(row.get("bps", 0) or 0),
                    selected=int(row.get("selected", 0) or 0),
                    transfers_in=int(row.get("transfers_in", 0) or 0),
                    transfers_out=int(row.get("transfers_out", 0) or 0),
                    value=float(row.get("value", 0) or 0) / 10.0,
                )
                .on_conflict_do_nothing()
            )
            db.execute(stmt)
            inserted += 1

        db.commit()
    finally:
        db.close()

    return inserted


async def backfill() -> None:
    init_db()
    name_map = _build_player_name_map()
    logger.info("Built name map with %d player entries", len(name_map))

    async with httpx.AsyncClient() as client:
        for vaastav_season, db_season in SEASONS:
            url = GW_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            logger.info("Fetching %s ...", url)
            df = await _fetch_csv(client, url)
            if df is None:
                logger.warning("Skipping season %s — no data", vaastav_season)
                continue

            count = _ingest_dataframe(df, db_season, name_map)
            logger.info("Season %s: inserted %d GW rows", db_season, count)

    logger.info("Historical backfill complete")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(backfill())


if __name__ == "__main__":
    main()
