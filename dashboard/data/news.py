"""Injury / availability news queries for the dashboard."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


def get_injury_news(db: Session, squad_ids: list[int] | None = None) -> pd.DataFrame:
    """Every player with a non-available status or non-empty news field,
    most recently updated first. ``in_squad`` flags rows for ``squad_ids``
    so the page can surface your own players first."""
    query = text("""
        SELECT p.id AS player_id, p.web_name, p.position, t.short_name AS team_short,
               p.status, p.news, p.news_added, p.chance_of_playing_next_round
        FROM players p
        JOIN teams t ON t.id = p.team_id
        WHERE p.status != 'a' OR p.news != ''
        ORDER BY p.news_added IS NULL, p.news_added DESC
    """)
    df = pd.read_sql(query, db.bind)
    df["in_squad"] = df["player_id"].isin(squad_ids) if squad_ids else False
    return df
