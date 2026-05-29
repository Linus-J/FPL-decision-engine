#!/usr/bin/env python
import asyncio
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import httpx
import pandas as pd
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session, init_db
from data.models import PlayerGameweekStats

logger = logging.getLogger(__name__)

VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

SEASONS = [
    ("2021-22", "2021-22"),
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


def _build_fpl_id_map() -> dict[int, int]:
    db = get_session()
    try:
        from sqlalchemy import text
        rows = db.execute(text("SELECT id, fpl_id FROM players")).fetchall()
        return {int(fpl_id): int(db_id) for db_id, fpl_id in rows}
    finally:
        db.close()


def _ingest_dataframe(df: pd.DataFrame, season: str, fpl_id_map: dict[int, int]) -> tuple[int, int]:
    df = df.rename(columns={"round": "GW"}) if "GW" not in df.columns else df

    required = {"element", "GW", "minutes", "total_points"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("Season %s CSV missing columns: %s — skipping", season, missing)
        return 0, 0

    db = get_session()
    inserted = skipped = 0
    try:
        for _, row in df.iterrows():
            fpl_id = int(row.get("element", 0) or 0)
            player_id = fpl_id_map.get(fpl_id)
            if not player_id:
                skipped += 1
                continue

            gw = int(row.get("GW", 0) or 0)
            if not gw:
                skipped += 1
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

    return inserted, skipped


async def backfill() -> None:
    init_db()
    fpl_id_map = _build_fpl_id_map()
    logger.info("FPL ID map: %d players in DB", len(fpl_id_map))

    async with httpx.AsyncClient() as client:
        for vaastav_season, db_season in SEASONS:
            url = GW_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            logger.info("Fetching %s ...", url)
            df = await _fetch_csv(client, url)
            if df is None:
                logger.warning("Skipping season %s — no data", vaastav_season)
                continue

            total_rows = len(df)
            inserted, skipped = _ingest_dataframe(df, db_season, fpl_id_map)
            match_rate = inserted / total_rows * 100 if total_rows else 0
            logger.info(
                "Season %s: %d rows → %d inserted (%.0f%% match rate), %d unmatched",
                db_season, total_rows, inserted, match_rate, skipped,
            )

    logger.info("Historical backfill complete")


def main() -> None:
    asyncio.run(backfill())


if __name__ == "__main__":
    main()
