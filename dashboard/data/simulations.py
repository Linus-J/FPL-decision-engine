"""Simulation leaderboard queries for the dashboard
(plan/simulation-engine-v1.md)."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


def get_leaderboard(db: Session, season: str) -> pd.DataFrame:
    """One row per persona: config params, cumulative actual_outcome
    (summed across its 'lineup' rows), ranked highest-first."""
    query = text("""
        SELECT sm.id, sm.label, sm.risk_level,
               sm.max_ownership_differential, sm.chip_aggressiveness,
               COALESCE(SUM(sdl.actual_outcome), 0) AS cumulative_actual,
               COUNT(sdl.actual_outcome) AS gws_scored
        FROM sim_managers sm
        LEFT JOIN sim_decision_log sdl
            ON sdl.sim_manager_id = sm.id AND sdl.decision_type = 'lineup'
        WHERE sm.season = :season
        GROUP BY sm.id
        ORDER BY cumulative_actual DESC
    """)
    df = pd.read_sql(query, db.bind, params={"season": season})
    if not df.empty:
        df["rank"] = df["cumulative_actual"].rank(ascending=False, method="min").astype(int)
    return df


def get_real_squad_cumulative_actual(db: Session) -> float:
    row = db.execute(
        text(
            "SELECT COALESCE(SUM(actual_outcome), 0) FROM decision_log "
            "WHERE decision_type = 'lineup'"
        )
    ).fetchone()
    return float(row[0]) if row else 0.0
