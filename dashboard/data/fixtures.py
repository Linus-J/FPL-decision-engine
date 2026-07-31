"""Read-only fixture & double-gameweek queries for the dashboard."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from optimiser.transfers import get_dgw_coverage
from projection.pipeline import (
    _get_current_and_next_gw,
    _get_current_season,
    _get_dgw_gameweeks,
    get_latest_projections,
)


def get_upcoming_fixtures(db: Session, lookahead_gws: int = 8) -> pd.DataFrame:
    _, next_gw = _get_current_and_next_gw()
    season = _get_current_season()
    query = text("""
        SELECT f.gameweek, f.kickoff_time, f.is_dgw,
               th.short_name AS home, ta.short_name AS away
        FROM fixtures f
        JOIN teams th ON th.id = f.team_h_id
        JOIN teams ta ON ta.id = f.team_a_id
        WHERE f.season = :season AND f.gameweek >= :start AND f.gameweek < :end
        ORDER BY f.gameweek, f.kickoff_time
    """)
    return pd.read_sql(
        query, db.bind,
        params={"season": season, "start": next_gw, "end": next_gw + lookahead_gws},
    )


def get_squad_dgw_exposure(
    db: Session, squad_ids: list[int], lookahead_gws: int = 8
) -> dict[int, dict]:
    """Upcoming DGWs affecting ``squad_ids`` and their combined projected xPts.
    Mirrors the exact call shape ``agent/decision_engine.py`` already uses."""
    if not squad_ids:
        return {}
    dgw_gws = _get_dgw_gameweeks(lookahead_gws)
    if not dgw_gws:
        return {}
    players = pd.read_sql(
        text("SELECT id, web_name, position, team_id, now_cost FROM players"), db.bind
    )
    projections = get_latest_projections()
    return get_dgw_coverage(squad_ids, players, dgw_gws, projections)
