import asyncio
import logging
from datetime import datetime, timedelta

import aiohttp
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.settings import settings
from data.db import get_session
from data.models import FixtureOdds

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


def _extract_h2h(
    bookmakers: list[dict], home_team_name: str, away_team_name: str
) -> tuple[float, float, float]:
    """Real bug found 2026-07-28 (data-completeness audit): this used to sort
    the two non-Draw outcome names ALPHABETICALLY and assign the first to
    home, the second to away -- with no reference to which team is actually
    home. Whenever the away team's name sorted before the home team's
    (roughly half of all fixtures), home_win_prob/away_win_prob (and the
    home_cs_prob/away_cs_prob derived from them) were silently swapped, with
    no error or log line. Now keyed directly by the real home/away team
    names from the same odds-API event payload."""
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] == "h2h":
                outcomes = {o["name"]: _implied_prob(o["price"]) for o in market["outcomes"]}
                if home_team_name not in outcomes or away_team_name not in outcomes:
                    continue
                return _normalise(
                    outcomes[home_team_name],
                    outcomes.get("Draw", 0.33),
                    outcomes[away_team_name],
                )
    return 0.33, 0.33, 0.33


def _cs_from_h2h(
    home_win: float, draw: float, away_win: float, over25: float = 0.0
) -> tuple[float, float]:
    """(P(home keeps a clean sheet), P(away keeps a clean sheet)).

    A clean sheet belongs to the DEFENCE — ``home_cs`` is the away side
    failing to score. The previous heuristic had it inverted:

        home_cs = draw + away_win * 0.3

    which is P(the HOME team fails to score), attributing home clean sheets
    to away wins. Against a fixture priced at 80.6%/13.2%/6.2% it returned
    home_cs=0.151 for the dominant side and 0.374 for the underdog — exactly
    backwards, and each side was handed the other's number.
    ``projection/features.py`` feeds these to the minutes model as
    ``my_cs_prob``/``opp_cs_prob``, so it has been reading them inverted.

    Replaced by the model's own Poisson rather than a re-swapped heuristic:
    goals are drawn as Poisson(lambda) everywhere else in this project
    (``projection/covariance.py::sample_team_goals``), so P(concede 0) is
    exactly exp(-lambda_opponent) — the same closed form
    ``projection/assemble.py`` uses. That makes this column consistent with
    the rest of the model instead of a parallel approximation.

    ``over25`` is what lets the total be split from the result; without it
    ``team_goals_from_odds`` cannot separate a 1-0 from a 3-2, so the caller
    should pass it whenever the totals market was available.

    Delegates to ``projection.team_goals`` so this (the live path) and
    ``scripts.backfill_odds`` (the training path) cannot drift apart — see that
    function's docstring for why they must not.
    """
    from projection.team_goals import clean_sheet_probs_from_odds

    return clean_sheet_probs_from_odds(home_win, draw, away_win, over25)


def _extract_over25(bookmakers: list[dict]) -> float:
    """Implied P(over 2.5 goals) from the totals market, de-vigged against the
    paired under. Returns 0.0 if the 2.5 line is not offered."""
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] != "totals":
                continue
            over = under = None
            for o in market.get("outcomes", []):
                if abs(float(o.get("point", 0)) - 2.5) > 1e-9:
                    continue
                if o["name"].lower() == "over":
                    over = _implied_prob(o["price"])
                elif o["name"].lower() == "under":
                    under = _implied_prob(o["price"])
            if over and under:
                return round(over / (over + under), 3)
    return 0.0


def _match_fixture(
    home_team_name: str,
    away_team_name: str,
    commence_time: str,
    db_fixtures: list[dict],
) -> int | None:
    """Real bug found 2026-07-28 (data-completeness audit): this used to
    match on the HOME team name alone, against every one of that team's
    still-unfinished fixtures, with no away-team check and no ordering --
    a team with more than one unfinished home fixture in the response
    window got all its odds attached to whichever fixture the (unordered)
    query happened to return first, silently leaving its other fixture(s)
    with zero odds coverage. Now requires BOTH team names to match, and
    breaks ties (a genuine same-week double-header) by nearest kickoff
    time to the odds event's own ``commence_time``."""
    home_lower = home_team_name.lower()
    away_lower = away_team_name.lower()

    candidates = [
        fix
        for fix in db_fixtures
        if (
            fix["team_h_name"].lower() in home_lower
            or home_lower in fix["team_h_name"].lower()
        )
        and (
            fix["team_a_name"].lower() in away_lower
            or away_lower in fix["team_a_name"].lower()
        )
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]["id"]

    try:
        commence = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return candidates[0]["id"]

    def _kickoff_delta(fix: dict) -> timedelta:
        ko = fix.get("kickoff_time")
        if ko is None:
            return timedelta.max
        if isinstance(ko, str):
            ko = datetime.fromisoformat(ko)
        if ko.tzinfo is not None:
            ko = ko.replace(tzinfo=None)
        return abs(ko - commence)

    return min(candidates, key=_kickoff_delta)["id"]


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
                logger.debug(
                    "No fixture match for %s vs %s", event.get("home_team"), event.get("away_team")
                )
                continue

            bookmakers = event.get("bookmakers", [])
            home_win, draw, away_win = _extract_h2h(
                bookmakers, event.get("home_team", ""), event.get("away_team", "")
            )
            over25 = _extract_over25(bookmakers)
            home_cs, away_cs = _cs_from_h2h(home_win, draw, away_win, over25)

            # Append-only (finding L4): one row per fetch, keyed
            # (fixture_id, fetched_at). Never UPDATE — the as-of read
            # (features.load_live_odds_asof) picks the latest ≤ deadline.
            # btts_prob is left NULL (MARKETS never requests a BTTS market —
            # see FixtureOdds.btts_prob) rather than a fake 0.0.
            stmt = sqlite_insert(FixtureOdds).values(
                fixture_id=fixture_id,
                home_win_prob=round(home_win, 3),
                draw_prob=round(draw, 3),
                away_win_prob=round(away_win, 3),
                over25_prob=over25,
                btts_prob=None,
                home_cs_prob=home_cs,
                away_cs_prob=away_cs,
                fetched_at=datetime.utcnow(),
            ).on_conflict_do_nothing(index_elements=["fixture_id", "fetched_at"])
            db.execute(stmt)
            upserted += 1

        db.commit()
        logger.info("Odds ingest complete: %d fixture snapshots appended", upserted)
        return upserted
    finally:
        db.close()


def odds_coverage_by_gameweek(season: str, horizon: int) -> dict[int, tuple[int, int]]:
    """gameweek -> (fixtures WITH odds, fixtures total) for the next
    ``horizon`` gameweeks (P3.11, 2026-08-16).

    Bookmakers only price near-term fixtures, so the planning horizon is
    routinely longer than the odds window. When a gameweek has no odds,
    ``projection/assemble.py`` silently falls back to a flat
    ``lam_home=1.35, lam_away=1.15`` -- a league-average scoreline for every
    fixture, which erases exactly the fixture-difficulty signal the extra
    horizon was added to exploit.

    Measured at 6 of 10 GW1 fixtures and nothing beyond GW1 on 2026-08-16.
    Nothing here can widen the window; the point is that the horizon's real
    information content stops being an assumption.
    """
    db = get_session()
    try:
        rows = db.execute(text("""
            SELECT f.gameweek,
                   -- DISTINCT on both: fixture_odds is append-only (one row
                   -- per fetch), so a plain COUNT(*) counts fetches, not
                   -- fixtures, and reported 112 "fixtures" for a 10-match
                   -- gameweek.
                   COUNT(DISTINCT f.id) AS fixtures,
                   COUNT(DISTINCT fo.fixture_id) AS with_odds
            FROM fixtures f
            LEFT JOIN fixture_odds fo ON fo.fixture_id = f.id
            WHERE f.season = :season AND f.finished = 0
            GROUP BY f.gameweek
            ORDER BY f.gameweek
        """), {"season": season}).fetchall()
    finally:
        db.close()
    coverage = {int(gw): (int(with_odds), int(fixtures)) for gw, fixtures, with_odds in rows}
    upcoming = sorted(coverage)[:horizon]
    return {gw: coverage[gw] for gw in upcoming}


def log_odds_coverage(season: str, horizon: int) -> dict[int, tuple[int, int]]:
    """Report coverage across the planning horizon, warning on any gameweek
    the model will end up projecting with flat league-average scorelines."""
    coverage = odds_coverage_by_gameweek(season, horizon)
    for gw, (with_odds, total) in coverage.items():
        if with_odds < total:
            logger.warning(
                "Odds coverage GW%d: %d/%d fixtures — the rest project on a flat "
                "league-average scoreline, not real fixture difficulty",
                gw, with_odds, total,
            )
        else:
            logger.info("Odds coverage GW%d: %d/%d fixtures", gw, with_odds, total)
    return coverage


def ingest_odds_sync() -> int:
    return asyncio.run(ingest_odds())
