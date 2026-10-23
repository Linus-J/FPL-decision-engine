"""Point-in-time reads for the append-only snapshot tables.

Every reader here answers one question: what did we know at ``deadline``?
Each returns, per entity, the latest snapshot stamped strictly before the
deadline. A snapshot taken after the deadline is never returned, however
recent it is, which is the property the backtest depends on.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

_STRENGTH_COLS = (
    "strength_overall_home", "strength_overall_away",
    "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
)


def load_team_strength_asof(db: Session, season: str, deadline: datetime) -> pd.DataFrame:
    """One row per team: the latest strengths snapshot before ``deadline``.

    Columns: ``team_id``, ``code``, ``snapshot_ts``, ``is_placeholder``, and the
    six strength columns. Teams with no snapshot before the deadline are absent.
    """
    cols = ", ".join(f"s.{c}" for c in _STRENGTH_COLS)
    query = text(f"""
        SELECT s.team_id, s.code, s.snapshot_ts, s.is_placeholder, {cols}
        FROM team_strength_snapshots s
        WHERE s.season = :season AND s.snapshot_ts < :deadline
          AND s.snapshot_ts = (
              SELECT MAX(s2.snapshot_ts) FROM team_strength_snapshots s2
              WHERE s2.season = s.season AND s2.team_id = s.team_id
                AND s2.snapshot_ts < :deadline
          )
    """)
    return pd.read_sql(query, db.bind, params={"season": season, "deadline": deadline})


def load_setpiece_asof(db: Session, season: str, deadline: datetime) -> pd.DataFrame:
    """One row per player: the latest set-piece snapshot before ``deadline``.

    A player whose latest snapshot before the deadline is a retirement has all
    orders NULL and ``is_penalty_taker`` False, so a retired duty reads as no
    duty rather than as the last duty held.
    """
    query = text("""
        SELECT s.player_id, s.snapshot_ts, s.penalty_order, s.freekick_order,
               s.corner_order, s.is_penalty_taker, s.penalty_xg_per_game,
               s.is_set_piece_taker
        FROM player_setpiece_snapshots s
        WHERE s.season = :season AND s.snapshot_ts < :deadline
          AND s.snapshot_ts = (
              SELECT MAX(s2.snapshot_ts) FROM player_setpiece_snapshots s2
              WHERE s2.season = s.season AND s2.player_id = s.player_id
                AND s2.snapshot_ts < :deadline
          )
    """)
    return pd.read_sql(query, db.bind, params={"season": season, "deadline": deadline})
