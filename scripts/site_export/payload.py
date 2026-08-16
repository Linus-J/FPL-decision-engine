from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from dashboard.data.decisions import get_decision_history
from dashboard.data.squad import get_current_squad
from data.models import Gameweek
from projection.pipeline import _get_current_season, get_latest_projections

SCHEMA_VERSION = 1


def get_projection_distributions(db: Session, gw: int, season: str) -> dict[int, dict[str, float]]:
    """Per-player {p10, median, mean, p90} xPts summary from projection_samples,
    aggregated across every MC scenario for one gameweek."""
    # created_at is shared across all rows of one persist batch (see
    # projection/assemble.py::_write_projection_samples), so scoping to
    # MAX(created_at) selects exactly the latest run's scenarios. Without
    # this the query averages every historical run together.
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
        return f"GW{gw} — {row.deadline_time.day} {row.deadline_time.strftime('%b %Y')}"
    return f"GW{gw}"


# 10th/90th percentile of a standard normal.
_Z10 = -1.2816


def _xpts_entry(
    player_id: int,
    dist: dict[int, dict[str, float]],
    fallback_mean: float | None,
    fallback_var: float | None = None,
) -> dict[str, float] | None:
    """A player's projected-points summary for the site.

    Real Monte Carlo quantiles when ``projection_samples`` has them. When it
    does not — every pre-season gameweek, since the cold start produces no
    draws — the spread used to collapse to p10 == median == mean == p90, so
    the site showed a bar of zero width for every player.

    ``xpts_var`` is available even then, so the interval is derived from it
    with a normal approximation. That is deliberately approximate: FPL scores
    are discrete and right-skewed (a long tail of hauls above a dense low
    mode), so a symmetric interval has the wrong SHAPE. It has roughly the
    right WIDTH, which is the part that carries information, and a
    zero-width bar is not a more honest alternative — it implies certainty
    that does not exist. Real samples always take precedence.
    """
    if player_id in dist:
        return dist[player_id]
    if fallback_mean is None or pd.isna(fallback_mean):
        return None
    mean = float(fallback_mean)
    if fallback_var is None or pd.isna(fallback_var) or float(fallback_var) <= 0:
        return {"p10": mean, "median": mean, "mean": mean, "p90": mean}
    sd = float(fallback_var) ** 0.5
    # Floored at zero: a negative score needs a card or own goal, which is far
    # rarer than a symmetric normal implies down there.
    return {
        "p10": round(max(0.0, mean + _Z10 * sd), 4),
        "median": round(mean, 4),
        "mean": round(mean, 4),
        "p90": round(mean - _Z10 * sd, 4),
    }


def _build_squad_entries(squad_df: pd.DataFrame, dist: dict[int, dict[str, float]]) -> list[dict]:
    bench = squad_df[~squad_df["is_starting"]]
    gk_bench = bench[bench["position"] == "GKP"]
    other_bench = bench[bench["position"] != "GKP"].sort_values("xpts", ascending=False)
    bench_order_by_player: dict[int, int] = {}
    for order, player_id in enumerate(
        [*gk_bench["player_id"], *other_bench["player_id"]], start=1
    ):
        bench_order_by_player[int(player_id)] = order

    entries = []
    for _, row in squad_df.iterrows():
        player_id = int(row["player_id"])
        entries.append({
            "player_id": player_id,
            "web_name": row["web_name"],
            "position": row["position"],
            "team_short": row["team_short"],
            "now_cost": float(row["now_cost"]),
            "is_starting": bool(row["is_starting"]),
            "is_captain": bool(row["is_captain"]),
            "is_vice_captain": bool(row["is_vice_captain"]),
            "bench_order": bench_order_by_player.get(player_id),
            "xpts": _xpts_entry(
                player_id, dist, row["xpts"], row.get("xpts_var")
            ),
        })
    return entries


def _build_top15_entries(
    projections_df: pd.DataFrame, dist: dict[int, dict[str, float]], team_names: dict[int, str]
) -> list[dict]:
    entries = []
    for _, row in projections_df.head(15).iterrows():
        player_id = int(row["player_id"])
        entries.append({
            "player_id": player_id,
            "web_name": row["web_name"],
            "position": row["position"],
            "team_short": team_names.get(int(row["team_id"]), ""),
            "xpts": _xpts_entry(player_id, dist, row["xpts_mean"]),
        })
    return entries


def _build_history_entries(history_df: pd.DataFrame) -> list[dict]:
    entries = []
    for _, row in history_df.iterrows():
        details = row["details"]
        if row["decision_type"] == "transfers":
            entries.append({
                "gameweek": int(row["gameweek"]),
                "type": "transfers",
                "transfers_in": [t["web_name"] for t in details.get("transfers_in", [])],
                "transfers_out": [t["web_name"] for t in details.get("transfers_out", [])],
                "hits_taken": details.get("hits_taken", 0),
                "net_xpts_gain": float(row["projected_gain"]),
            })
        elif row["decision_type"] == "chip":
            entries.append({
                "gameweek": int(row["gameweek"]),
                "type": "chip",
                "chip": details.get("chip"),
                "reason": details.get("reason", ""),
            })
    return entries


def build_run_payload(db: Session, team_id: int) -> dict:
    squad_df = get_current_squad(db, team_id)
    if squad_df.empty:
        raise RuntimeError("No current squad found -- cannot export site data")

    gw = int(squad_df["gameweek"].iloc[0])
    season = _get_current_season()
    dist = get_projection_distributions(db, gw, season)

    projections_df = get_latest_projections(gw)
    team_names = _team_short_names(db)
    history_df = get_decision_history(db, limit_gws=20)

    return {
        "schema_version": SCHEMA_VERSION,
        "gameweek": gw,
        "label": _label_for_gw(db, season, gw),
        "generated_at": datetime.now(UTC).isoformat(),
        "squad": _build_squad_entries(squad_df, dist),
        "top15": _build_top15_entries(projections_df, dist, team_names),
        "history": _build_history_entries(history_df),
    }
