"""Decision history queries for the dashboard: past logged decisions with
projected vs (once backfilled) actual outcomes, and the latest chip/transfer
plan for display-only surfacing."""

from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


def get_decision_history(db: Session, limit_gws: int = 20) -> pd.DataFrame:
    """One row per decision_log entry within the last ``limit_gws`` gameweeks,
    most recent first, with ``details`` parsed from JSON."""
    max_gw_row = db.execute(text("SELECT MAX(gameweek) FROM decision_log")).fetchone()
    max_gw = max_gw_row[0] if max_gw_row and max_gw_row[0] is not None else 0
    query = text("""
        SELECT id, gameweek, decision_type, details, projected_gain,
               actual_outcome, dry_run, created_at
        FROM decision_log
        WHERE gameweek >= :min_gw
        ORDER BY gameweek DESC, created_at DESC
    """)
    df = pd.read_sql(query, db.bind, params={"min_gw": max_gw - limit_gws + 1})
    if not df.empty:
        df["details"] = df["details"].apply(json.loads)
    return df


def get_latest_chip_plan(db: Session) -> dict | None:
    row = db.execute(
        text(
            "SELECT details, projected_gain, gameweek FROM decision_log "
            "WHERE decision_type = 'chip' ORDER BY created_at DESC LIMIT 1"
        )
    ).fetchone()
    if not row:
        return None
    details = json.loads(row[0])
    return {
        "gameweek": row[2],
        "chip": details.get("chip"),
        "reason": details.get("reason"),
        "expected_gain": row[1],
    }


def get_latest_transfer_plan(db: Session) -> dict | None:
    row = db.execute(
        text(
            "SELECT details, projected_gain, gameweek FROM decision_log "
            "WHERE decision_type = 'transfers' ORDER BY created_at DESC LIMIT 1"
        )
    ).fetchone()
    if not row:
        return None
    details = json.loads(row[0])
    return {
        "gameweek": row[2],
        "transfers_in": details.get("transfers_in", []),
        "transfers_out": details.get("transfers_out", []),
        "hits_taken": details.get("hits_taken", 0),
        "net_xpts_gain": row[1],
    }
