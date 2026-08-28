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

# Real published strengths sit around 900-1400. Anything this small is not a
# strength on that scale. Mirrored by projection.features._PLAUSIBLE_STRENGTH_FLOOR,
# which is where the convention is actually enforced on read.
_PLAUSIBLE_STRENGTH_FLOOR = 100


def _is_placeholder_strength(row: dict) -> bool:
    """Is this FPL's PRE-SEASON placeholder rather than a real strength?

    Before a season starts the bootstrap returns ``strength_attack_*`` and
    ``strength_defence_*`` as 0 and ``strength_overall_*`` on the 1-5 tier
    scale -- Arsenal came back as overall_home=4, attack_home=0 on 2026-08-17.
    Copied verbatim, that put the live season's FDR features on a completely
    different scale from the five seasons the model trains on (975-1390), with
    four of the six pinned to zero. ``load_fixture_difficulty`` could not
    rescue it either: its COALESCE only fires for a MISSING row, and a row full
    of zeros is present.

    Writing the neutral value instead costs nothing real -- a placeholder
    carries no signal either way -- and keeps the column on the scale the model
    understands. It self-corrects the moment FPL publishes, because real values
    are not placeholders.
    """
    overall = [row[c] for c in ("strength_overall_home", "strength_overall_away")]
    others = [
        row[c] for c in (
            "strength_attack_home", "strength_attack_away",
            "strength_defence_home", "strength_defence_away",
        )
    ]
    if any(v is None for v in overall + others):
        return True
    return all(v == 0 for v in others) or any(
        v < _PLAUSIBLE_STRENGTH_FLOOR for v in overall
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
    placeholders = 0
    try:
        for t in bootstrap["teams"]:
            row = {col: t[col] for col in _STRENGTH_COLS}
            if _is_placeholder_strength(row):
                placeholders += 1
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
        if placeholders:
            logger.warning(
                "%d/%d %s teams have PLACEHOLDER strengths in the bootstrap "
                "(attack/defence 0, overall on FPL's 1-5 tier). Stored as-is: "
                "0 is this project's established 'not published' signal and "
                "cold_start.load_current_defence_strength relies on it to fall "
                "back to PRIOR-season strengths. Consumers must treat values "
                "below %d as absent, not as data.",
                placeholders, len(bootstrap["teams"]), season,
                _PLAUSIBLE_STRENGTH_FLOOR,
            )
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
                    # §8: FPL's "data is final" flag. Defaults False when the
                    # key is absent so an older/partial payload is treated as
                    # NOT final rather than silently final -- the safe side.
                    data_checked=gw.get("data_checked", False),
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
                        "data_checked": gw.get("data_checked", False),
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


# Never NULL. SQLite does not treat two NULLs as equal for uniqueness, so a
# NULL ``opponent_team_id`` makes ``on_conflict_do_update`` match nothing and
# insert a fresh duplicate row on every single re-run. 0 is never a real FPL
# team id, so it is a safe stand-in for "this entry named no opponent".
_NO_OPPONENT_SENTINEL = 0


def _history_rows(history: list[dict]) -> dict[tuple[int, int], dict]:
    """FPL's per-element ``history`` -> one row per (gameweek, opponent). Pure.

    Keyed to match ``PlayerGameweekStats``' unique constraint
    ``(player_id, gameweek, season, opponent_team_id)``, which is also the
    shape ``scripts/backfill_history.py`` writes. Both of a double gameweek's
    fixtures therefore survive as their own rows, carrying their own opponent
    and home/away flag.

    This used to pre-sum a gameweek into ONE row with the sentinel opponent --
    a workaround for the older 3-column key, under which the second fixture's
    stat line silently overwrote the first instead of adding to it. The key
    has since been widened, so the summing is redundant, and it was actively
    harmful: it dropped ``opponent_team``/``was_home`` on the floor, which is
    what every odds and FDR join in ``projection/features.py`` matches on.
    Found 2026-08-28 -- all 571 live-season rows had my_cs_prob/opp_cs_prob
    pinned at 0.2 and over25_prob at 0.5 while the model was fitted on real
    variation.

    ``selected``/``value`` are point-in-time squad-value/ownership snapshots
    rather than per-fixture stats, so each row keeps its own rather than
    summing. Two entries sharing one (round, opponent) -- which real fixture
    data should never produce -- are summed, because the unique key cannot
    hold them separately.
    """
    by_key: dict[tuple[int, int], dict] = {}
    for entry in history:
        gw = entry.get("round")
        if not gw:
            continue
        opponent = entry.get("opponent_team")
        key = (int(gw), int(opponent) if opponent else _NO_OPPONENT_SENTINEL)
        acc = by_key.get(key)
        if acc is None:
            acc = dict.fromkeys(_SUM_HISTORY_FIELDS, 0)
            acc["opponent_team_id"] = key[1]
            acc["was_home"] = None
            acc["fpl_fixture_id"] = None
            by_key[key] = acc
        for field in _SUM_HISTORY_FIELDS:
            acc[field] += entry.get(field, 0) or 0
        acc["selected"] = entry.get("selected", 0)
        acc["value"] = entry.get("value", 0) / 10.0
        if entry.get("was_home") is not None:
            acc["was_home"] = bool(entry["was_home"])
        if entry.get("fixture") is not None:
            acc["fpl_fixture_id"] = int(entry["fixture"])
    return by_key


def _resolve_team_id_season(
    fixture_sides: dict[int, tuple[int, int]],
    fpl_fixture_id: int | None,
    was_home: bool | None,
) -> int | None:
    """The player's OWN team id for this fixture, from the fixture's two sides.

    Derived from the fixture rather than read off ``players.team_id`` so a
    mid-season transfer does not retroactively relabel the player's earlier
    gameweeks with his new club. ``None`` when the fixture has not been
    ingested or the entry carries no home/away flag -- honest about not
    knowing, rather than guessing a side.
    """
    if fpl_fixture_id is None or was_home is None:
        return None
    sides = fixture_sides.get(fpl_fixture_id)
    if not sides:
        return None
    home_id, away_id = sides
    return home_id if was_home else away_id


def _load_fixture_sides(season: str) -> dict[int, tuple[int, int]]:
    """``{FPL fixture id: (team_h_id, team_a_id)}`` for one season.

    Loaded once per ingest and threaded through rather than queried per
    player -- ``run_full_ingest`` fans out over ~700 of them.
    """
    db = get_session()
    try:
        rows = db.query(
            Fixture.fpl_id, Fixture.team_h_id, Fixture.team_a_id
        ).filter(Fixture.season == season).all()
        return {
            int(fpl_id): (int(h), int(a))
            for fpl_id, h, a in rows
            if fpl_id is not None and h is not None and a is not None
        }
    finally:
        db.close()


async def ingest_player_history(
    player_fpl_id: int,
    player_db_id: int,
    season: str = "2026-27",
    fixture_sides: dict[int, tuple[int, int]] | None = None,
) -> None:
    """One ``player_gw_stats`` row per (gameweek, opponent) for this player.

    ``fixture_sides`` resolves each row's ``team_id_season``; pass the map
    from ``_load_fixture_sides`` to avoid re-querying it per player.
    """
    if fixture_sides is None:
        fixture_sides = _load_fixture_sides(season)
    db = get_session()
    try:
        data = await fetch_player_summary(player_fpl_id)
        rows = _history_rows(data.get("history", []))
        for (gw, _opponent), row in rows.items():
            vals = dict(row)
            # Carried only to resolve the player's own side; not a column.
            fpl_fixture_id = vals.pop("fpl_fixture_id", None)
            vals["team_id_season"] = _resolve_team_id_season(
                fixture_sides, fpl_fixture_id, vals["was_home"]
            )
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

    # Loaded once and threaded through: this fans out over ~700 players, and
    # every one of them needs the same fixture -> (home, away) map to resolve
    # its own side.
    fixture_sides = _load_fixture_sides(season)

    semaphore = asyncio.Semaphore(5)

    async def _ingest_with_semaphore(fpl_id: int, db_id: int) -> None:
        async with semaphore:
            try:
                await ingest_player_history(fpl_id, db_id, season, fixture_sides)
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
