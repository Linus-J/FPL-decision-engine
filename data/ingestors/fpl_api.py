import asyncio
import logging
from datetime import datetime

import aiohttp
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.models import (
    Fixture,
    Gameweek,
    Player,
    PlayerGameweekStats,
    PlayerStateSnapshot,
    Team,
    TeamSeasonStrength,
)

logger = logging.getLogger(__name__)

FPL_BASE = "https://fantasy.premierleague.com/api"

_STRENGTH_COLS = (
    "strength_overall_home", "strength_overall_away",
    "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
)


async def _get(session: aiohttp.ClientSession, path: str) -> dict | list:
    url = f"{FPL_BASE}{path}"
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_bootstrap() -> dict:
    async with aiohttp.ClientSession() as session:
        return await _get(session, "/bootstrap-static/")


async def fetch_fixtures() -> list:
    async with aiohttp.ClientSession() as session:
        return await _get(session, "/fixtures/")


async def fetch_player_summary(player_fpl_id: int) -> dict:
    async with aiohttp.ClientSession() as session:
        return await _get(session, f"/element-summary/{player_fpl_id}/")


async def fetch_live_gameweek(gw: int) -> dict:
    async with aiohttp.ClientSession() as session:
        return await _get(session, f"/event/{gw}/live/")


def _position_name(element_type: int) -> str:
    return {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(element_type, "UNK")


def upsert_teams(bootstrap: dict, season: str = "2026-27") -> None:
    db = get_session()
    try:
        for t in bootstrap["teams"]:
            stmt = (
                insert(Team)
                .values(
                    id=t["id"],
                    name=t["name"],
                    short_name=t["short_name"],
                    strength_overall_home=t["strength_overall_home"],
                    strength_overall_away=t["strength_overall_away"],
                    strength_attack_home=t["strength_attack_home"],
                    strength_attack_away=t["strength_attack_away"],
                    strength_defence_home=t["strength_defence_home"],
                    strength_defence_away=t["strength_defence_away"],
                    updated_at=datetime.utcnow(),
                )
                .on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "name": t["name"],
                        "short_name": t["short_name"],
                        "strength_overall_home": t["strength_overall_home"],
                        "strength_overall_away": t["strength_overall_away"],
                        "strength_attack_home": t["strength_attack_home"],
                        "strength_attack_away": t["strength_attack_away"],
                        "strength_defence_home": t["strength_defence_home"],
                        "strength_defence_away": t["strength_defence_away"],
                        "updated_at": datetime.utcnow(),
                    },
                )
            )
            db.execute(stmt)
        db.commit()
        logger.info("Upserted %d teams", len(bootstrap["teams"]))
    finally:
        db.close()

    # Real gap found 2026-07-30 (the user's own live-smoke-test follow-up):
    # `team_season_strength` (the season-scoped table projection/features.py's
    # FDR joins actually read) is only ever written by
    # scripts/backfill_history.py, which targets the fixed list of PAST
    # seasons it backfills -- nothing in the live sync path ever wrote a row
    # for the CURRENT season, so every live FDR feature silently fell back
    # to the neutral 1200 default, permanently, for the whole live-serving
    # path. `teams` (this function, above) already gets the real live
    # strength values every sync; this just also copies them into the
    # season-scoped table under the current season.
    db = get_session()
    try:
        for t in bootstrap["teams"]:
            row = {col: t[col] for col in _STRENGTH_COLS}
            stmt = (
                insert(TeamSeasonStrength)
                .values(season=season, team_id=t["id"], code=t.get("code"), **row)
                .on_conflict_do_update(
                    index_elements=["season", "team_id"],
                    set_=row,
                )
            )
            db.execute(stmt)
        db.commit()
        logger.info("Upserted %d team_season_strength rows for %s", len(bootstrap["teams"]), season)
    finally:
        db.close()


def upsert_gameweeks(bootstrap: dict, season: str = "2026-27") -> None:
    db = get_session()
    try:
        for gw in bootstrap["events"]:
            deadline = datetime.fromisoformat(gw["deadline_time"].replace("Z", "+00:00"))
            stmt = (
                insert(Gameweek)
                .values(
                    id=gw["id"],
                    season=season,
                    name=gw["name"],
                    deadline_time=deadline,
                    finished=gw["finished"],
                    is_current=gw["is_current"],
                    is_next=gw["is_next"],
                    average_entry_score=gw.get("average_entry_score") or 0,
                    highest_score=gw.get("highest_score") or 0,
                    is_dgw=False,
                    is_bgw=False,
                )
                .on_conflict_do_update(
                    index_elements=["id", "season"],
                    set_={
                        "name": gw["name"],
                        "deadline_time": deadline,
                        "finished": gw["finished"],
                        "is_current": gw["is_current"],
                        "is_next": gw["is_next"],
                        "average_entry_score": gw.get("average_entry_score") or 0,
                        "highest_score": gw.get("highest_score") or 0,
                    },
                )
            )
            db.execute(stmt)
        db.commit()
        logger.info("Upserted %d gameweeks", len(bootstrap["events"]))
    finally:
        db.close()


def upsert_players(bootstrap: dict) -> None:
    db = get_session()
    try:
        seen_codes: set[int] = set()
        for p in bootstrap["elements"]:
            if p.get("code") is not None:
                seen_codes.add(p["code"])
            news_added = None
            if p.get("news_added"):
                news_added = datetime.fromisoformat(p["news_added"].replace("Z", "+00:00"))

            stmt = (
                insert(Player)
                .values(
                    fpl_id=p["id"],
                    code=p.get("code"),
                    first_name=p["first_name"],
                    second_name=p["second_name"],
                    web_name=p["web_name"],
                    team_id=p["team"],
                    position=_position_name(p["element_type"]),
                    now_cost=p["now_cost"] / 10.0,
                    cost_change_start=p.get("cost_change_start", 0) / 10.0,
                    status=p.get("status", "a"),
                    news=p.get("news", ""),
                    news_added=news_added,
                    selected_by_percent=float(p.get("selected_by_percent", 0) or 0),
                    form=float(p.get("form", 0) or 0),
                    total_points=p.get("total_points", 0),
                    minutes=p.get("minutes", 0),
                    goals_scored=p.get("goals_scored", 0),
                    assists=p.get("assists", 0),
                    clean_sheets=p.get("clean_sheets", 0),
                    goals_conceded=p.get("goals_conceded", 0),
                    saves=p.get("saves", 0),
                    yellow_cards=p.get("yellow_cards", 0),
                    red_cards=p.get("red_cards", 0),
                    bonus=p.get("bonus", 0),
                    bps=p.get("bps", 0),
                    ict_index=float(p.get("ict_index", 0) or 0),
                    influence=float(p.get("influence", 0) or 0),
                    creativity=float(p.get("creativity", 0) or 0),
                    threat=float(p.get("threat", 0) or 0),
                    chance_of_playing_next_round=p.get("chance_of_playing_next_round"),
                    chance_of_playing_this_round=p.get("chance_of_playing_this_round"),
                    transfers_in_event=p.get("transfers_in_event", 0),
                    transfers_out_event=p.get("transfers_out_event", 0),
                    updated_at=datetime.utcnow(),
                )
                # Upsert on the stable `code`, not the season-reassigned fpl_id
                # (M3). fpl_id itself is updated to the current season's value.
                .on_conflict_do_update(
                    index_elements=["code"],
                    set_={
                        "fpl_id": p["id"],
                        "first_name": p["first_name"],
                        "second_name": p["second_name"],
                        "web_name": p["web_name"],
                        "team_id": p["team"],
                        "now_cost": p["now_cost"] / 10.0,
                        "cost_change_start": p.get("cost_change_start", 0) / 10.0,
                        "status": p.get("status", "a"),
                        "news": p.get("news", ""),
                        "news_added": news_added,
                        "selected_by_percent": float(p.get("selected_by_percent", 0) or 0),
                        "form": float(p.get("form", 0) or 0),
                        "total_points": p.get("total_points", 0),
                        "minutes": p.get("minutes", 0),
                        "goals_scored": p.get("goals_scored", 0),
                        "assists": p.get("assists", 0),
                        "clean_sheets": p.get("clean_sheets", 0),
                        "goals_conceded": p.get("goals_conceded", 0),
                        "saves": p.get("saves", 0),
                        "yellow_cards": p.get("yellow_cards", 0),
                        "red_cards": p.get("red_cards", 0),
                        "bonus": p.get("bonus", 0),
                        "bps": p.get("bps", 0),
                        "ict_index": float(p.get("ict_index", 0) or 0),
                        "influence": float(p.get("influence", 0) or 0),
                        "creativity": float(p.get("creativity", 0) or 0),
                        "threat": float(p.get("threat", 0) or 0),
                        "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
                        "chance_of_playing_this_round": p.get("chance_of_playing_this_round"),
                        "transfers_in_event": p.get("transfers_in_event", 0),
                        "transfers_out_event": p.get("transfers_out_event", 0),
                        "updated_at": datetime.utcnow(),
                    },
                )
            )
            db.execute(stmt)
        db.commit()
        logger.info("Upserted %d players", len(bootstrap["elements"]))

        # Real bug found 2026-07-30 (the user's own review of a drafted
        # initial squad: "goalkeepers, Akinmboni, Casemiro, Abdullahi,
        # Fraser are not in this season"). Verified live: Casemiro left
        # Manchester United on a free transfer to Inter Miami CF when his
        # contract expired -- confirmed genuinely gone from the Premier
        # League. Upserting only ever adds/updates players PRESENT in the
        # current bootstrap; a player who has left the league entirely
        # just vanishes from FPL's own "elements" list, but their row here
        # sat at whatever stale status ('a') they had at their LAST
        # successful sync, forever, since nothing ever revisited a row
        # absent from every SUBSEQUENT fetch (all six were last touched
        # 2026-07-23, a full week stale). Being completely dropped from
        # the live bootstrap is an even stronger departure signal than
        # status='u' -- mark them 'u' so the existing departure gate
        # (optimiser/departure_risk.py's hard-exclude,
        # projection/cold_start.py's apply_departure_gate) correctly
        # excludes them like any other confirmed departure.
        if seen_codes:
            stale_players = (
                db.query(Player)
                .filter(Player.code.isnot(None), ~Player.code.in_(seen_codes))
                .all()
            )
            for sp in stale_players:
                if sp.status != "u":
                    sp.status = "u"
                    sp.news = "No longer in the live FPL bootstrap — presumed departed."
                    sp.updated_at = datetime.utcnow()
            if stale_players:
                db.commit()
                logger.info(
                    "Marked %d players absent from the bootstrap as departed (status=u)",
                    len(stale_players),
                )
    finally:
        db.close()


def _snapshot_gameweek_context(bootstrap: dict) -> int | None:
    """The GW this snapshot informs — the upcoming (is_next) GW, else current."""
    events = bootstrap.get("events", [])
    for ev in events:
        if ev.get("is_next"):
            return ev["id"]
    for ev in events:
        if ev.get("is_current"):
            return ev["id"]
    return None


def write_player_snapshots(
    bootstrap: dict,
    snapshot_ts: datetime,
    season: str = "2026-27",
) -> int:
    """Append one point-in-time PlayerStateSnapshot per player (never UPDATE).

    This is the source of truth for leakage-free feature reads. Idempotent on
    the (player_id, snapshot_ts) unique key: re-running with the same timestamp
    inserts nothing. ``upsert_players`` must have run first so the FPL-id -> DB-id
    map is populated.
    """
    db = get_session()
    try:
        id_map = {p.fpl_id: p.id for p in db.query(Player.fpl_id, Player.id).all()}
        gw_context = _snapshot_gameweek_context(bootstrap)
        written = 0
        for p in bootstrap["elements"]:
            player_db_id = id_map.get(p["id"])
            if player_db_id is None:
                logger.warning("Snapshot skipped — no DB row for FPL id %d", p["id"])
                continue

            news_added = None
            if p.get("news_added"):
                news_added = datetime.fromisoformat(p["news_added"].replace("Z", "+00:00"))

            stmt = (
                insert(PlayerStateSnapshot)
                .values(
                    player_id=player_db_id,
                    snapshot_ts=snapshot_ts,
                    season=season,
                    gameweek_context=gw_context,
                    now_cost=p["now_cost"] / 10.0,
                    status=p.get("status", "a"),
                    chance_of_playing_this_round=p.get("chance_of_playing_this_round"),
                    chance_of_playing_next_round=p.get("chance_of_playing_next_round"),
                    selected_by_percent=float(p.get("selected_by_percent", 0) or 0),
                    form=float(p.get("form", 0) or 0),
                    ict_index=float(p.get("ict_index", 0) or 0),
                    influence=float(p.get("influence", 0) or 0),
                    creativity=float(p.get("creativity", 0) or 0),
                    threat=float(p.get("threat", 0) or 0),
                    news=p.get("news", ""),
                    news_added=news_added,
                    transfers_in_event=p.get("transfers_in_event", 0),
                    transfers_out_event=p.get("transfers_out_event", 0),
                )
                .on_conflict_do_nothing(index_elements=["player_id", "snapshot_ts"])
            )
            result = db.execute(stmt)
            written += result.rowcount or 0
        db.commit()
        logger.info(
            "Wrote %d player snapshots at %s (gw_context=%s)",
            written,
            snapshot_ts.isoformat(),
            gw_context,
        )
        return written
    finally:
        db.close()


def upsert_fixtures(raw_fixtures: list, season: str = "2026-27") -> None:
    db = get_session()
    try:
        gw_fixture_counts: dict[int, int] = {}
        for f in raw_fixtures:
            gw = f.get("event")
            if gw:
                gw_fixture_counts[gw] = gw_fixture_counts.get(gw, 0) + 1

        for f in raw_fixtures:
            kickoff = None
            if f.get("kickoff_time"):
                kickoff = datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))

            gw = f.get("event")
            is_dgw = bool(gw and gw_fixture_counts.get(gw, 0) > 10)

            stmt = (
                insert(Fixture)
                .values(
                    fpl_id=f["id"],
                    season=season,
                    gameweek=gw,
                    team_h_id=f["team_h"],
                    team_a_id=f["team_a"],
                    team_h_score=f.get("team_h_score"),
                    team_a_score=f.get("team_a_score"),
                    finished=f.get("finished", False),
                    kickoff_time=kickoff,
                    is_dgw=is_dgw,
                )
                .on_conflict_do_update(
                    index_elements=["season", "fpl_id"],
                    set_={
                        "gameweek": gw,
                        "team_h_score": f.get("team_h_score"),
                        "team_a_score": f.get("team_a_score"),
                        "finished": f.get("finished", False),
                        "kickoff_time": kickoff,
                        "is_dgw": is_dgw,
                    },
                )
            )
            db.execute(stmt)

        for gw_id, count in gw_fixture_counts.items():
            if count > 10:
                db.query(Gameweek).filter(
                    Gameweek.id == gw_id, Gameweek.season == season
                ).update({"is_dgw": True})
            elif count < 10:
                db.query(Gameweek).filter(
                    Gameweek.id == gw_id, Gameweek.season == season
                ).update({"is_bgw": True})

        db.commit()
        logger.info("Upserted %d fixtures", len(raw_fixtures))
    finally:
        db.close()


# Genuinely cumulative per-fixture stats -- summed across a DGW's two
# fixtures rather than the second overwriting the first (see
# _accumulate_gw_history below).
_SUM_HISTORY_FIELDS = (
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "yellow_cards", "red_cards", "bonus", "bps",
    "transfers_in", "transfers_out",
)


def _accumulate_gw_history(history: list[dict]) -> dict[int, dict]:
    """FPL's per-element ``history`` -> one row per gameweek. Pure/testable.

    Real bug found 2026-07-28 (data-completeness audit): the unique key on
    ``PlayerGameweekStats`` is ``(player_id, gameweek, season)`` with no
    fixture/opponent component, but a genuine double-gameweek player's
    history has TWO entries with the same ``round`` -- the old caller wrote
    each entry with its own ``on_conflict_do_update``, so the second
    fixture's stat line silently overwrote (not summed with) the first,
    destroying one match's entire contribution. ``selected``/``value`` are
    point-in-time squad-value/ownership snapshots, not per-fixture stats, so
    the LATEST entry's value is kept rather than summed.
    """
    by_gw: dict[int, dict] = {}
    for entry in history:
        gw = entry.get("round")
        if not gw:
            continue
        acc = by_gw.setdefault(gw, dict.fromkeys(_SUM_HISTORY_FIELDS, 0))
        for field in _SUM_HISTORY_FIELDS:
            acc[field] += entry.get(field, 0) or 0
        acc["selected"] = entry.get("selected", 0)
        acc["value"] = entry.get("value", 0) / 10.0
    return by_gw


# Real bug found 2026-07-30 (a regression from THIS session's own earlier
# fix): PlayerGameweekStats' unique constraint was widened from
# (player_id, gameweek, season) to include opponent_team_id, so both of a
# DGW's fixtures could be stored (see data/models.py). This function
# deliberately pre-sums a DGW into ONE row (_accumulate_gw_history, a
# different, already-correct fix for the SAME underlying DGW bug class,
# from earlier still) and never set opponent_team_id at all -- meaning
# every row here got the column's NULL default, and SQLite never treats
# two NULLs as equal for uniqueness, so `on_conflict_do_update` targeting
# the OLD 3-column shape no longer matches any real constraint, and even a
# fixed 4-column target would insert a fresh duplicate row every single
# re-run instead of updating. A fixed sentinel (0 -- never a real FPL team
# id) keeps this path's existing summed-row design and its tests intact
# while making the conflict target real again.
_NO_OPPONENT_SENTINEL = 0


async def ingest_player_history(
    player_fpl_id: int, player_db_id: int, season: str = "2026-27"
) -> None:
    db = get_session()
    try:
        data = await fetch_player_summary(player_fpl_id)
        by_gw = _accumulate_gw_history(data.get("history", []))
        for gw, vals in by_gw.items():
            vals = {**vals, "opponent_team_id": _NO_OPPONENT_SENTINEL}
            stmt = (
                insert(PlayerGameweekStats)
                .values(player_id=player_db_id, gameweek=gw, season=season, **vals)
                .on_conflict_do_update(
                    index_elements=["player_id", "gameweek", "season", "opponent_team_id"],
                    set_=vals,
                )
            )
            db.execute(stmt)
        db.commit()
    finally:
        db.close()


async def run_full_ingest(season: str = "2026-27") -> None:
    logger.info("Starting full FPL ingest for season %s", season)

    bootstrap = await fetch_bootstrap()
    upsert_teams(bootstrap, season)
    upsert_gameweeks(bootstrap, season)
    upsert_players(bootstrap)
    # Append-only point-in-time capture (source of truth for leakage-free reads).
    write_player_snapshots(bootstrap, datetime.utcnow(), season)

    raw_fixtures = await fetch_fixtures()
    upsert_fixtures(raw_fixtures, season)

    db = get_session()
    try:
        players = db.query(Player).all()
        # status 'u' = confirmed departed (not a PL player right now, same
        # convention as cold_start.py's _DEPARTED_STATUS) -- their OLD
        # fpl_id's element-summary endpoint is gone from FPL's API once
        # they leave, so this is a guaranteed 404 every single sync, not a
        # transient failure. Real bug found 2026-08-01 (live-testing on the
        # user's machine): 169/733 players failed this way on one run,
        # spamming warnings for data that was never coming back -- their
        # history from when they WERE active is already in our DB.
        player_map = {p.fpl_id: p.id for p in players if p.status != "u"}
    finally:
        db.close()

    semaphore = asyncio.Semaphore(5)

    async def _ingest_with_semaphore(fpl_id: int, db_id: int) -> None:
        async with semaphore:
            try:
                await ingest_player_history(fpl_id, db_id, season)
            except Exception as exc:
                logger.warning("Failed to ingest history for player %d: %s", fpl_id, exc)

    tasks = [
        _ingest_with_semaphore(fpl_id, db_id)
        for fpl_id, db_id in player_map.items()
    ]
    await asyncio.gather(*tasks)

    logger.info("Full FPL ingest complete — %d players processed", len(tasks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data.db import init_db
    init_db()
    asyncio.run(run_full_ingest())
