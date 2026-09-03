import asyncio
import json
import logging

import aiohttp
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.ingestors.fbref import _build_name_map, _match_player
from data.ingestors.setpiece import players_with_published_roles, write_setpiece_roles
from data.models import PlayerXGStats

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


def setpiece_role_from_understat(
    player_id: int,
    *,
    xg: float,
    npxg: float,
    key_passes_per_game: float,
    games: int,
    is_published: bool,
) -> dict:
    """One ``write_setpiece_roles`` role dict from a player's season totals.

    Source precedence (2026-09-03). This ingest runs on EVERY weekly run
    (scripts/run_agent.py -> scripts/run_weekly.py) and, until today, wrote
    all four role columns unconditionally -- with no equivalent of the
    deference ``ingest_setpiece_roles`` was given in 0ef8f75. So every week it
    overwrote the published depth chart with two proxies measured over a
    handful of played gameweeks:

      * ``is_penalty_taker``/``penalty_xg_per_game`` from realised penalty xG
        (``xg - npxg``), which says only "has he taken one YET". Two games
        into 2026-27 that made Szoboszlai -- Liverpool's SECOND-choice taker,
        who happened to take one -- carry 0.3806 penalty xG per game, against
        the 0.01138 his depth-chart order implies, while Haaland, Saka,
        Palmer, Isak, Mateta and ten other published first-choice takers were
        written back to 0.0. ``load_penalty_duty`` reads that column straight
        into ``goal_weight``, so at 5 points a goal it handed Szoboszlai
        roughly +1.85 xPts a game he had not earned and took ~0.36 off
        Haaland: enough to flip the GW3 captaincy between them.
      * ``is_set_piece_taker`` from key passes per game, which is a
        creativity proxy rather than a duty at all, and had written 43
        published corner/free-kick takers back to False.

    Withholding a key rather than writing a False is the whole mechanism:
    ``write_setpiece_roles`` updates only the keys it is given, so a withheld
    one keeps the depth chart's value. ``key_passes_per_game`` is always
    offered because a taker list has no opinion on it -- the same split the
    FBref path already makes.

    Realised penalty xG is still the better answer LATER in a season, once
    it is measured over enough games to beat a pre-season list. Nothing here
    encodes that crossover; the depth chart simply wins until it is reloaded
    with a newer one.
    """
    role: dict = {
        "player_id": player_id,
        "key_passes_per_game": round(key_passes_per_game, 4),
    }
    if is_published:
        return role

    penalty_xg_per_game = (xg - npxg) / games
    role["is_penalty_taker"] = penalty_xg_per_game >= PENALTY_TAKER_THRESHOLD
    role["penalty_xg_per_game"] = round(penalty_xg_per_game, 4)
    role["is_set_piece_taker"] = key_passes_per_game >= SET_PIECE_TAKER_THRESHOLD
    return role


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

    # Read BEFORE the loop: one query, and the set must not shift underneath
    # a partially-written pass.
    published = players_with_published_roles(db_season)

    roles: list[dict] = []
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

            roles.append(setpiece_role_from_understat(
                db_player_id,
                xg=xg, npxg=npxg, key_passes_per_game=kp_per_gw, games=games,
                is_published=db_player_id in published,
            ))

        db.commit()
    finally:
        db.close()

    # Written through write_setpiece_roles rather than this module's own
    # upsert (2026-09-03) so the PARTIAL-update contract applies here too:
    # the keys setpiece_role_from_understat withheld keep whatever the
    # published depth chart wrote, instead of being overwritten with a proxy.
    inserted_sp = write_setpiece_roles(db_season, roles)
    deferred = sum(1 for r in roles if "is_penalty_taker" not in r)

    logger.info(
        "Understat %s: %d xG rows, %d set-piece roles, %d unmatched, "
        "%d deferred to a published depth chart",
        db_season, inserted_xg, inserted_sp, skipped, deferred,
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
