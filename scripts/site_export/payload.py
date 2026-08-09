from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from data.models import Gameweek


def get_projection_distributions(db: Session, gw: int, season: str) -> dict[int, dict[str, float]]:
    """Per-player {p10, median, mean, p90} xPts summary from projection_samples,
    aggregated across every MC scenario for one gameweek."""
    query = text("""
        SELECT player_id, xpts
        FROM projection_samples
        WHERE gameweek = :gw AND season = :season
          AND created_at = (
              SELECT MAX(created_at) FROM projection_samples
              WHERE gameweek = :gw AND season = :season
          )
    """)
    df = pd.read_sql(query, db.bind, params={"gw": gw, "season": season})
    out: dict[int, dict[str, float]] = {}
    for player_id, values in df.groupby("player_id")["xpts"]:
        out[int(player_id)] = {
            "p10": float(values.quantile(0.10)),
            "median": float(values.quantile(0.50)),
            "mean": float(values.mean()),
            "p90": float(values.quantile(0.90)),
        }
    return out


def _team_short_names(db: Session) -> dict[int, str]:
    df = pd.read_sql(text("SELECT id, short_name FROM teams"), db.bind)
    return dict(zip(df["id"], df["short_name"]))


def _label_for_gw(db: Session, season: str, gw: int) -> str:
    row = db.query(Gameweek).filter(Gameweek.season == season, Gameweek.id == gw).first()
    if row and row.deadline_time:
        return f"GW{gw} — {row.deadline_time.day} {row.deadline_time.strftime('%b')}"
    return f"GW{gw}"
