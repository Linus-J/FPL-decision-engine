#!/usr/bin/env python
import asyncio
import io
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import httpx
import pandas as pd
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session, init_db
from data.models import Gameweek, PlayerGameweekStats

logger = logging.getLogger(__name__)

VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

SEASONS = [
    ("2021-22", "2021-22"),
    ("2022-23", "2022-23"),
    ("2023-24", "2023-24"),
    ("2024-25", "2024-25"),
]

GW_CSV_URL = "{base}/{season}/gws/merged_gw.csv"
FIXTURES_CSV_URL = "{base}/{season}/fixtures.csv"
PLAYERS_RAW_CSV_URL = "{base}/{season}/players_raw.csv"

# FPL deadline is ~90 minutes before the first kickoff of the gameweek.
DEADLINE_LEAD = timedelta(minutes=90)


async def _fetch_csv(client: httpx.AsyncClient, url: str) -> pd.DataFrame | None:
    try:
        resp = await client.get(url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        logger.warning("Could not fetch %s: %s", url, exc)
        return None


# --- Per-season gameweek deadlines (T3a; fixes M1/M2 for the as-of boundary) ---

def compute_gw_deadlines(fixtures_df: pd.DataFrame) -> dict[int, datetime]:
    """Earliest kickoff per gameweek − 90 min, as naive UTC.

    Pure/testable. Rows with a missing event or kickoff_time are ignored
    (postponed/TBC fixtures). The result is the FPL-deadline proxy the
    leakage-free backtest reads against.
    """
    if not {"event", "kickoff_time"}.issubset(fixtures_df.columns):
        return {}
    df = fixtures_df.dropna(subset=["event", "kickoff_time"]).copy()
    if df.empty:
        return {}
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["kickoff_time"])
    deadlines: dict[int, datetime] = {}
    for gw, grp in df.groupby("event"):
        first_kickoff = grp["kickoff_time"].min()
        # store naive UTC to match the rest of the codebase (datetime.utcnow)
        deadline = first_kickoff.tz_convert("UTC").tz_localize(None).to_pydatetime()
        deadlines[int(gw)] = deadline - DEADLINE_LEAD
    return deadlines


def upsert_gameweek_deadlines(season: str, deadlines: dict[int, datetime]) -> int:
    db = get_session()
    written = 0
    try:
        for gw, deadline in deadlines.items():
            stmt = (
                insert(Gameweek)
                .values(
                    id=gw,
                    season=season,
                    name=f"Gameweek {gw}",
                    deadline_time=deadline,
                    finished=True,
                )
                .on_conflict_do_update(
                    index_elements=["id", "season"],
                    set_={"deadline_time": deadline},
                )
            )
            db.execute(stmt)
            written += 1
        db.commit()
    finally:
        db.close()
    return written


# --- Cross-season player mapping via stable code (T3a; fixes M3) ---

def element_code_map(players_raw_df: pd.DataFrame) -> dict[int, int]:
    """Per-season vaastav element id -> FPL cross-season `code`. Pure/testable."""
    if not {"id", "code"}.issubset(players_raw_df.columns):
        return {}
    out: dict[int, int] = {}
    for element, code in zip(players_raw_df["id"], players_raw_df["code"], strict=False):
        if pd.notna(element) and pd.notna(code):
            out[int(element)] = int(code)
    return out


def build_code_to_dbid_map() -> dict[int, int]:
    from sqlalchemy import text

    db = get_session()
    try:
        rows = db.execute(
            text("SELECT id, code FROM players WHERE code IS NOT NULL")
        ).fetchall()
        return {int(code): int(db_id) for db_id, code in rows}
    finally:
        db.close()


def resolve_player_id(
    element: int,
    elem_to_code: dict[int, int],
    code_to_dbid: dict[int, int],
) -> int | None:
    """vaastav element -> code -> players.id. None if the player is not in the
    current squad (e.g. left the league) — those rows are skipped, not misjoined."""
    code = elem_to_code.get(element)
    if code is None:
        return None
    return code_to_dbid.get(code)


def _ingest_dataframe(
    df: pd.DataFrame,
    season: str,
    elem_to_code: dict[int, int],
    code_to_dbid: dict[int, int],
) -> tuple[int, int]:
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
            element = int(row.get("element", 0) or 0)
            player_id = resolve_player_id(element, elem_to_code, code_to_dbid)
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
    code_to_dbid = build_code_to_dbid_map()
    logger.info("Code→DB-id map: %d players with a stable code", len(code_to_dbid))

    async with httpx.AsyncClient() as client:
        for vaastav_season, db_season in SEASONS:
            # 1. Per-season gameweek deadlines (the as-of boundary for T3/T4).
            fixtures = await _fetch_csv(
                client, FIXTURES_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            )
            if fixtures is not None:
                deadlines = compute_gw_deadlines(fixtures)
                n = upsert_gameweek_deadlines(db_season, deadlines)
                logger.info("Season %s: %d gameweek deadlines written", db_season, n)
            else:
                logger.warning("Season %s: no fixtures.csv — deadlines skipped", db_season)

            # 2. Element→code crosswalk (fixes the season-unstable id join, M3).
            players_raw = await _fetch_csv(
                client, PLAYERS_RAW_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            )
            elem_to_code = element_code_map(players_raw) if players_raw is not None else {}
            logger.info("Season %s: %d element→code entries", db_season, len(elem_to_code))

            # 3. Per-GW player stats, mapped via code (not the reassigned element id).
            df = await _fetch_csv(
                client, GW_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            )
            if df is None:
                logger.warning("Skipping season %s — no merged_gw.csv", vaastav_season)
                continue

            total_rows = len(df)
            inserted, skipped = _ingest_dataframe(df, db_season, elem_to_code, code_to_dbid)
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
