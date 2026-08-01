"""prior_league_translation.py — P11 cross-league translation-factor
calibration: how much a prior-league (non-PL) per-90 attacking output
scales to its PL-equivalent, fit against a real hold-out of players who
actually made that jump in a past season (not asserted from literature).

Deliberately NOT persisted as a table -- build_holdout() is cheap to
recompute (season-aggregate rows, not per-match) and gets more accurate
for free as more PL seasons accumulate in future years. Compute once via
scripts/calibrate_prior_league_factors.py, hand-copy the result into
config/strategy.py's PriorLeagueRules -- same precedent as this session's
scripts/calibrate_risk_constants.py.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from data.db import get_session
from data.ingestors.fbref import SEASON_MAP
from projection.cold_start import MIN_PRIOR_APPEARANCES

# MIN_PRIOR_APPEARANCES full 90-minute appearances' worth, translated from
# cold_start.py's per-appearance bar since prior_league_stats is
# season-aggregate, not per-appearance.
MIN_HOLDOUT_MINUTES = MIN_PRIOR_APPEARANCES * 90

# (prior-league season, immediately-following PL season) -- every transition
# we have BOTH sides of real data for (player_gw_stats covers 2021-22..2025-26).
SEASON_TRANSITIONS: list[tuple[str, str]] = [
    ("2021-22", "2022-23"),
    ("2022-23", "2023-24"),
    ("2023-24", "2024-25"),
    ("2024-25", "2025-26"),
]

MIN_CALIBRATION_SAMPLES = 15

_HOLDOUT_COLUMNS = [
    "code", "prior_goals90", "prior_assists90",
    "realized_goals90", "realized_assists90", "realized_points90",
]


def _prior_side(league: str, prior_season: str) -> pd.DataFrame:
    """Matched (code populated), qualifying prior-league rows for one season."""
    db = get_session()
    try:
        soccerdata_season = SEASON_MAP.get(prior_season, prior_season)
        query = text("""
            SELECT code, goals90, assists90, minutes
            FROM prior_league_stats
            WHERE league = :league AND season = :season
              AND code IS NOT NULL AND minutes >= :min_minutes
        """)
        return pd.read_sql(query, db.bind, params={
            "league": league, "season": soccerdata_season,
            "min_minutes": MIN_HOLDOUT_MINUTES,
        })
    finally:
        db.close()


def _realized_pl_side(codes: list[int], pl_season: str) -> pd.DataFrame:
    """Real PL per-90 output (goals, assists, total points) for a set of
    codes in one PL season, summed across every gameweek/fixture that
    season (a genuine DGW player has two rows for one gameweek -- summing
    is correct here, not a double-count, since we want the season total)."""
    empty = pd.DataFrame(columns=["code", "pl_minutes", "pl_goals", "pl_assists", "pl_points"])
    if not codes:
        return empty
    db = get_session()
    try:
        placeholders = ", ".join(f":code{i}" for i in range(len(codes)))
        params = {f"code{i}": c for i, c in enumerate(codes)}
        params["season"] = pl_season
        query = text(f"""
            SELECT p.code AS code,
                   SUM(g.minutes) AS pl_minutes,
                   SUM(g.goals_scored) AS pl_goals,
                   SUM(g.assists) AS pl_assists,
                   SUM(g.total_points) AS pl_points
            FROM player_gw_stats g
            JOIN players p ON p.id = g.player_id
            WHERE g.season = :season AND p.code IN ({placeholders})
            GROUP BY p.code
        """)
        result = pd.read_sql(query, db.bind, params=params)
        return result if not result.empty else empty
    finally:
        db.close()


def build_holdout(league: str) -> pd.DataFrame:
    """Pooled made-the-jump hold-out for one prior league across every
    available historical season-transition: one row per qualifying player
    with both their prior-league per-90s and their realized PL per-90s."""
    rows = []
    for prior_season, pl_season in SEASON_TRANSITIONS:
        prior = _prior_side(league, prior_season)
        if prior.empty:
            continue
        realized = _realized_pl_side(prior["code"].tolist(), pl_season)
        if realized.empty:
            continue
        realized = realized[realized["pl_minutes"] >= MIN_HOLDOUT_MINUTES]
        if realized.empty:
            continue
        merged = prior.merge(realized, on="code", how="inner")
        if merged.empty:
            continue
        merged["realized_goals90"] = merged["pl_goals"] / merged["pl_minutes"] * 90
        merged["realized_assists90"] = merged["pl_assists"] / merged["pl_minutes"] * 90
        merged["realized_points90"] = merged["pl_points"] / merged["pl_minutes"] * 90
        merged = merged.rename(
            columns={"goals90": "prior_goals90", "assists90": "prior_assists90"}
        )
        rows.append(merged[_HOLDOUT_COLUMNS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=_HOLDOUT_COLUMNS)


def compute_league_stats(holdout: pd.DataFrame) -> tuple[float | None, float | None, int]:
    """(translation_factor, realized_points90_variance, hold-out sample
    size). Both are None when the sample is too sparse to trust -- caller
    falls back to a literature-style default for that league."""
    n = len(holdout)
    if n < MIN_CALIBRATION_SAMPLES:
        return None, None, n
    prior_median = (holdout["prior_goals90"] + holdout["prior_assists90"]).median()
    realized_median = (holdout["realized_goals90"] + holdout["realized_assists90"]).median()
    factor = float(realized_median / prior_median) if prior_median > 0 else None
    variance = float(holdout["realized_points90"].var(ddof=1)) if n > 1 else None
    return factor, variance, n
