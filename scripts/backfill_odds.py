#!/usr/bin/env python
"""backfill_odds.py — historical closing odds → historical_fixture_odds (T6).

Source: football-data.co.uk (static per-season CSV, no Cloudflare wall). We take
each match's closing 1X2 (Pinnacle preferred, Bet365 fallback) and Over/Under
2.5, de-vig to probabilities, derive clean-sheet probs, map football-data team
names → that season's FPL team id, and assign the match to a gameweek via the
per-season deadlines already backfilled by T3a. Rows are stamped
``fetched_at = deadline − ε`` so the leakage-free reader keeps them (finding C2).

The parsing/mapping helpers are pure and unit-tested; only ``backfill_odds``
touches the network + DB.
"""

from __future__ import annotations

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
from data.models import Gameweek, HistoricalFixtureOdds
from scripts.backfill_history import (
    TEAMS_CSV_URL,
    VAASTAV_BASE,
    _fetch_csv,
    team_name_to_id,
)

logger = logging.getLogger(__name__)

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# our season -> football-data code
SEASON_CODES = {
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}

# Closing 1X2 column triples, best source first ('C' = closing line).
_1X2_TRIPLES = (
    ("PSCH", "PSCD", "PSCA"),   # Pinnacle closing
    ("PSH", "PSD", "PSA"),      # Pinnacle
    ("B365CH", "B365CD", "B365CA"),
    ("B365H", "B365D", "B365A"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("BbAvH", "BbAvD", "BbAvA"),
)
# Over/Under 2.5 column pairs, best source first.
_OU_PAIRS = (
    ("PC>2.5", "PC<2.5"),
    ("P>2.5", "P<2.5"),
    ("B365C>2.5", "B365C<2.5"),
    ("B365>2.5", "B365<2.5"),
    ("Avg>2.5", "Avg<2.5"),
)

# football-data name -> a token found in the vaastav teams.csv name/short_name.
FD_TEAM_ALIASES = {
    "Man City": "Man City",
    "Man United": "Man Utd",
    "Tottenham": "Spurs",
    "Nott'm Forest": "Nott'm Forest",
    "Newcastle": "Newcastle",
    "Wolves": "Wolves",
    "Sheffield United": "Sheffield Utd",
    "Leeds": "Leeds",
    "Luton": "Luton",
    "West Brom": "West Brom",
    "Leicester": "Leicester",
}

DEADLINE_EPSILON = timedelta(minutes=1)   # fetched_at = deadline − ε (< deadline)


def implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds if decimal_odds and decimal_odds > 0 else 0.0


def normalise_1x2(home: float, draw: float, away: float) -> tuple[float, float, float]:
    total = home + draw + away
    if total <= 0:
        return 0.0, 0.0, 0.0
    return home / total, draw / total, away / total


def cs_probs_from_1x2(
    home_win: float, draw: float, away_win: float, over25: float | None = None
) -> tuple[float, float]:
    """Clean-sheet probabilities for the TRAINING rows, on exactly the scale the
    live path writes.

    Delegates to ``projection.team_goals.clean_sheet_probs_from_odds`` — the
    single canonical derivation — rather than mirroring it. The previous
    implementation *said* it mirrored the live heuristic and then stopped doing
    so the moment that heuristic was corrected, which is precisely the failure a
    shared function removes. Pass ``over25`` whenever the totals market is
    available; it pins the goal total that 1X2 alone cannot.
    """
    from projection.team_goals import clean_sheet_probs_from_odds

    return clean_sheet_probs_from_odds(home_win, draw, away_win, over25)


def over25_prob(over_odds: float, under_odds: float) -> float:
    o, u = implied_prob(over_odds), implied_prob(under_odds)
    return round(o / (o + u), 3) if (o and u) else 0.0


def _pick_triple(row: pd.Series) -> tuple[float, float, float] | None:
    for h, d, a in _1X2_TRIPLES:
        if h in row and pd.notna(row[h]) and pd.notna(row[d]) and pd.notna(row[a]):
            return float(row[h]), float(row[d]), float(row[a])
    return None


def _pick_ou(row: pd.Series) -> tuple[float, float] | None:
    for over, under in _OU_PAIRS:
        if over in row and pd.notna(row[over]) and pd.notna(row[under]):
            return float(row[over]), float(row[under])
    return None


def normalise_fd_team(name: str) -> str:
    return FD_TEAM_ALIASES.get(name.strip(), name.strip())


def resolve_fd_team(name: str, name_to_id: dict[str, int]) -> int | None:
    """football-data team name → that season's FPL team id via the alias map,
    then a case-insensitive substring match against the vaastav names."""
    if not name:
        return None
    alias = normalise_fd_team(name)
    if alias in name_to_id:
        return name_to_id[alias]
    lower = alias.lower()
    for cand, tid in name_to_id.items():
        if lower == cand.lower() or lower in cand.lower() or cand.lower() in lower:
            return tid
    return None


def parse_kickoff(date_str: str, time_str: object) -> datetime | None:
    """football-data Date (dd/mm/yy or dd/mm/yyyy) + optional Time (HH:MM)."""
    if not date_str or pd.isna(date_str):
        return None
    time = str(time_str) if pd.notna(time_str) else "15:00"
    if ":" not in time:
        time = "15:00"
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%y %H:%M"):
        try:
            return datetime.strptime(f"{date_str} {time}", fmt)
        except ValueError:
            continue
    return None


def assign_gameweek(kickoff: datetime, deadlines: dict[int, datetime]) -> int | None:
    """The gameweek a match belongs to = the latest deadline at or before its
    kickoff (matches are played after that GW's deadline)."""
    prior = [gw for gw, dl in deadlines.items() if dl <= kickoff]
    return max(prior) if prior else None


def build_odds_rows(
    df: pd.DataFrame,
    season: str,
    name_to_id: dict[str, int],
    deadlines: dict[int, datetime],
) -> tuple[list[dict], int]:
    """Parse a football-data E0 dataframe → HistoricalFixtureOdds row dicts.
    Pure/testable. Returns ``(rows, skipped)`` (skipped = unmatched team/GW/odds)."""
    rows: list[dict] = []
    skipped = 0
    for _, r in df.iterrows():
        home_id = resolve_fd_team(str(r.get("HomeTeam", "")), name_to_id)
        away_id = resolve_fd_team(str(r.get("AwayTeam", "")), name_to_id)
        triple = _pick_triple(r)
        kickoff = parse_kickoff(r.get("Date"), r.get("Time"))
        if home_id is None or away_id is None or triple is None or kickoff is None:
            skipped += 1
            continue
        gw = assign_gameweek(kickoff, deadlines)
        if gw is None:
            skipped += 1
            continue

        hw, dr, aw = normalise_1x2(*(implied_prob(x) for x in triple))
        ou = _pick_ou(r)
        over25 = over25_prob(*ou) if ou else 0.0
        # over25 first: it is what separates a 1-0 from a 3-2, so the clean
        # sheet derivation needs it rather than inferring the total from 1X2.
        home_cs, away_cs = cs_probs_from_1x2(hw, dr, aw, over25)
        rows.append({
            "season": season,
            "gameweek": gw,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_win_prob": round(hw, 3),
            "draw_prob": round(dr, 3),
            "away_win_prob": round(aw, 3),
            "over25_prob": over25,
            "btts_prob": over25,  # proxy: BTTS tracks over-2.5 (no BTTS col in E0)
            "home_cs_prob": home_cs,
            "away_cs_prob": away_cs,
            "fetched_at": deadlines[gw] - DEADLINE_EPSILON,
        })
    return rows, skipped


def write_odds_rows(rows: list[dict]) -> int:
    """Insert historical odds, idempotent on (season, gameweek, home, away)."""
    db = get_session()
    written = 0
    try:
        for row in rows:
            stmt = (
                insert(HistoricalFixtureOdds)
                .values(**row)
                .on_conflict_do_update(
                    index_elements=["season", "gameweek", "home_team_id", "away_team_id"],
                    set_={k: row[k] for k in (
                        "home_win_prob", "draw_prob", "away_win_prob", "over25_prob",
                        "btts_prob", "home_cs_prob", "away_cs_prob", "fetched_at",
                    )},
                )
            )
            written += db.execute(stmt).rowcount or 0
        db.commit()
    finally:
        db.close()
    return written


def _load_deadlines(season: str) -> dict[int, datetime]:
    """Per-season GW deadlines from the DB (written by T3a's backfill)."""
    db = get_session()
    try:
        rows = db.query(Gameweek.id, Gameweek.deadline_time).filter(
            Gameweek.season == season
        ).all()
        return {gw: dl for gw, dl in rows if dl is not None}
    finally:
        db.close()


async def backfill_odds(seasons: list[str] | None = None) -> None:
    init_db()
    targets = seasons or list(SEASON_CODES)
    async with httpx.AsyncClient() as client:
        for season in targets:
            code = SEASON_CODES.get(season)
            if not code:
                logger.warning("No football-data code for %s", season)
                continue
            deadlines = _load_deadlines(season)
            if not deadlines:
                logger.warning(
                    "Season %s: no GW deadlines in DB — run backfill_history first", season
                )
                continue

            teams_url = TEAMS_CSV_URL.format(base=VAASTAV_BASE, season=season)
            teams_df = await _fetch_csv(client, teams_url)
            if teams_df is None:
                logger.warning("Season %s: no teams.csv — cannot map odds team names", season)
                continue
            name_to_id = team_name_to_id(teams_df)

            resp = await client.get(FOOTBALL_DATA_URL.format(code=code), timeout=30.0)
            if resp.status_code != 200:
                logger.warning("Season %s: football-data HTTP %s", season, resp.status_code)
                continue
            df = pd.read_csv(io.StringIO(resp.text))

            rows, skipped = build_odds_rows(df, season, name_to_id, deadlines)
            written = write_odds_rows(rows)
            logger.info(
                "Season %s: %d matches → %d odds rows written, %d skipped",
                season, len(df), written, skipped,
            )
    logger.info("Historical odds backfill complete")


def main() -> None:
    asyncio.run(backfill_odds())


if __name__ == "__main__":
    main()
