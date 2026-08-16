"""Simulation read-outs for the dashboard.

Delegates to ``simulation/analysis.py`` (P2.5) rather than querying
``sim_decision_log`` directly. That module is the season's read-out and
already handles two things this page previously got wrong:

- **Re-decided gameweeks.** A gameweek can be decided more than once (a
  rerun, refining the squad as news lands), which appends another lineup
  row. Summing them double-counted that week.
- **The paired comparison.** Personas share a season, so the difference
  against the baseline control carries far less variance than any absolute
  total. Ranking on ``total_actual`` alone reads a season of shared luck as
  if it were signal.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from simulation.analysis import axis_effect, calibration, persona_season_summary


def get_leaderboard(db: Session, season: str) -> pd.DataFrame:
    """One row per persona that has scored a gameweek, best first.

    Reuses the caller's session so the whole page runs on one connection.
    """
    summary = persona_season_summary(season, db)
    if summary.empty:
        return summary
    return summary.rename(columns={"sim_manager_id": "id"})


def get_axis_effects(season: str, db: Session | None = None) -> pd.DataFrame:
    """Per swept parameter, the value tried against what it scored."""
    return axis_effect(season, db)


def get_calibration(season: str, db: Session | None = None) -> pd.DataFrame:
    """Per-gameweek predicted vs actual across the cohort."""
    return calibration(season, db)


def get_real_squad_cumulative_actual(db: Session) -> float:
    """The real bot's own cumulative scored points.

    Deduplicated per gameweek for the same reason as above -- the real
    squad's log is re-written whenever a gameweek's decision is re-run, and
    pre-season it was re-run several times.
    """
    rows = db.execute(
        text(
            "SELECT gameweek, actual_outcome, created_at FROM decision_log "
            "WHERE decision_type = 'lineup' AND actual_outcome IS NOT NULL "
            "ORDER BY gameweek, created_at"
        )
    ).fetchall()
    if not rows:
        return 0.0
    latest_per_gw = {int(gw): float(outcome) for gw, outcome, _ in rows}
    return float(sum(latest_per_gw.values()))
