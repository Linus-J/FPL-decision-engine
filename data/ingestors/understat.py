import asyncio
import json
import logging
from datetime import datetime

import aiohttp
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.ingestors.fbref import _build_name_map, _match_player
from data.models import PlayerSetPieceRole, PlayerXGStats

logger = logging.getLogger(__name__)

UNDERSTAT_API = "https://understat.com/main/getPlayersStats"
UNDERSTAT_LEAGUE = "EPL"

SEASON_MAP = {
    "2022-23": "2022",
    "2023-24": "2023",
    "2024-25": "2024",
    "2025-26": "2025",
    "2026-27": "2026",
}

PENALTY_TAKER_THRESHOLD = 0.08
SET_PIECE_TAKER_THRESHOLD = 1.5

# Real bug found 2026-07-28 (data-completeness audit): this module used to
# carry its OWN local name matcher (`_build_fpl_name_map`/`_match_player`)
# with the exact unguarded-substring collision the Gabriel Magalhães fix
# (fbref.py) addressed -- a short single-token web_name like "Gabriel" would
# silently absorb "Gabriel Martinelli"/"Gabriel Jesus"'s real xG. Unlike
# understat_xg.py, THIS module is wired into scripts/run_agent.py's routine
# live pipeline, so the collision risk was live-production-facing, not just
# a backtest artifact. Now reuses the shared, hardened matcher instead of
# its own copy.
_build_fpl_name_map = _build_name_map


async def _fetch_season_players(
    session: aiohttp.ClientSession, understat_season: str
) -> list[dict]:
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{UNDERSTAT_LEAGUE}/{understat_season}",
    }
    async with session.post(
        UNDERSTAT_API,
        data={"league": UNDERSTAT_LEAGUE, "season": understat_season},
        headers=headers,
    ) as resp:
        resp.raise_for_status()
        txt = await resp.text()
    data = json.loads(txt)
    return data.get("players", [])


async def ingest_understat_season(understat_season: str, db_season: str) -> None:
    name_map = _build_fpl_name_map()
    inserted_xg = 0
    inserted_sp = 0
    skipped = 0

    connector = aiohttp.TCPConnector(limit=5)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; fpl-bot/2.0)"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        players = await _fetch_season_players(session, understat_season)

    if not players:
        logger.info("Understat %s: no data available yet (season not started)", db_season)
        return

    logger.info(
        "Understat %s: %d players from season %s", db_season, len(players), understat_season
    )

    db = get_session()
    try:
        for p in players:
            player_name = p.get("player_name", "")
            db_player_id = _match_player(player_name, name_map)
            if not db_player_id:
                skipped += 1
                continue

            games = max(int(p.get("games", 0) or 0), 1)
            xg = float(p.get("xG", 0) or 0)
            xa = float(p.get("xA", 0) or 0)
            npxg = float(p.get("npxG", 0) or 0)
            shots = int(p.get("shots", 0) or 0)
            key_passes = int(p.get("key_passes", 0) or 0)

            avg_gw = max(games // 2, 1)
            xg_per_gw = xg / games
            xa_per_gw = xa / games
            npxg_per_gw = npxg / games
            shots_per_gw = shots / games
            kp_per_gw = key_passes / games

            for gw in range(1, avg_gw + 1):
                stmt = (
                    insert(PlayerXGStats)
                    .values(
                        player_id=db_player_id,
                        gameweek=gw,
                        season=db_season,
                        xg=round(xg_per_gw, 4),
                        xa=round(xa_per_gw, 4),
                        xgi=round((xg + xa) / games, 4),
                        npxg=round(npxg_per_gw, 4),
                        shots=int(shots_per_gw),
                        key_passes=int(kp_per_gw),
                    )
                    .on_conflict_do_nothing()
                )
                db.execute(stmt)
                inserted_xg += 1

            penalty_xg_per_game = (xg - npxg) / games
            is_pen_taker = penalty_xg_per_game >= PENALTY_TAKER_THRESHOLD
            is_sp_taker = kp_per_gw >= SET_PIECE_TAKER_THRESHOLD

            sp_stmt = (
                insert(PlayerSetPieceRole)
                .values(
                    player_id=db_player_id,
                    season=db_season,
                    is_penalty_taker=is_pen_taker,
                    penalty_xg_per_game=round(penalty_xg_per_game, 4),
                    is_set_piece_taker=is_sp_taker,
                    key_passes_per_game=round(kp_per_gw, 4),
                    updated_at=datetime.utcnow(),
                )
                .on_conflict_do_update(
                    index_elements=["player_id", "season"],
                    set_={
                        "is_penalty_taker": is_pen_taker,
                        "penalty_xg_per_game": round(penalty_xg_per_game, 4),
                        "is_set_piece_taker": is_sp_taker,
                        "key_passes_per_game": round(kp_per_gw, 4),
                        "updated_at": datetime.utcnow(),
                    },
                )
            )
            db.execute(sp_stmt)
            inserted_sp += 1

        db.commit()
    finally:
        db.close()

    logger.info(
        "Understat %s: %d xG rows, %d set-piece roles, %d unmatched",
        db_season, inserted_xg, inserted_sp, skipped,
    )


async def run_understat_ingest(seasons: list[str] | None = None) -> None:
    targets = seasons or list(SEASON_MAP.keys())
    for db_season in targets:
        understat_year = SEASON_MAP.get(db_season)
        if not understat_year:
            logger.warning("No understat mapping for season %s", db_season)
            continue
        logger.info("Ingesting Understat xG for %s (understat year: %s)", db_season, understat_year)
        await ingest_understat_season(understat_year, db_season)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_understat_ingest())
