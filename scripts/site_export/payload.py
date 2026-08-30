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

    ``approx`` marks which of the two a reader is looking at. Under the normal
    approximation ``median`` is not an estimated quantile at all — it equals
    ``mean`` by definition, because a normal is symmetric — so publishing the
    two side by side with nothing to distinguish them invites the reader to
    treat their agreement as evidence about the distribution when it is an
    artefact of the method. Real MC quantiles carry ``approx: false`` and a
    median that genuinely differs.
    """
    if player_id in dist:
        return {**dist[player_id], "approx": False}
    if fallback_mean is None or pd.isna(fallback_mean):
        return None
    mean = float(fallback_mean)
    if fallback_var is None or pd.isna(fallback_var) or float(fallback_var) <= 0:
        return {"p10": mean, "median": mean, "mean": mean, "p90": mean, "approx": True}
    sd = float(fallback_var) ** 0.5
    # Floored at zero: a negative score needs a card or own goal, which is far
    # rarer than a symmetric normal implies down there.
    return {
        "p10": round(max(0.0, mean + _Z10 * sd), 4),
        "median": round(mean, 4),
        "mean": round(mean, 4),
        "p90": round(mean - _Z10 * sd, 4),
        "approx": True,
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
            # `xpts_var` matters here as much as it does for the squad
            # (2026-08-18). Omitting it made `_xpts_entry` take its
            # no-variance branch, which returns p10 == median == mean == p90,
            # so every one of the top fifteen rendered as a zero-width bar.
            # The squad path was fixed for exactly this and this one was left
            # behind; `get_latest_projections` has carried the column all
            # along, non-null and positive for every row.
            "xpts": _xpts_entry(
                player_id, dist, row["xpts_mean"], row.get("xpts_var")
            ),
        })
    return entries


def _transfers_entry(row: pd.Series) -> dict:
    details = row["details"]
    return {
        "gameweek": int(row["gameweek"]),
        "type": "transfers",
        "transfers_in": [t["web_name"] for t in details.get("transfers_in", [])],
        "transfers_out": [t["web_name"] for t in details.get("transfers_out", [])],
        "hits_taken": details.get("hits_taken", 0),
        "net_xpts_gain": float(row["projected_gain"]),
    }


def _chip_entry(row: pd.Series) -> dict:
    details = row["details"]
    return {
        "gameweek": int(row["gameweek"]),
        "type": "chip",
        "chip": details.get("chip"),
        "reason": details.get("reason", ""),
    }


def _is_no_op_transfer(row: pd.Series) -> bool:
    details = row["details"]
    return not details.get("transfers_in") and not details.get("transfers_out")


def _build_history_entries(history_df: pd.DataFrame, up_to_gw: int | None = None) -> list[dict]:
    """One published event per decision per gameweek, newest gameweek first.

    ``decision_log`` gets a fresh row every time the weekly pipeline runs,
    and the pipeline is re-run several times in the days before a deadline
    (2026-08-30). Rendering every row published GW2 as seven entries: three
    superseded transfer plans disagreeing with each other on both the hit
    count and the gain, three no-ops, and two different 3xc lines. Only the
    last run before the deadline was acted on -- and it is the run the squad
    panel already reflects, so publishing the rest put the two halves of the
    page in visible disagreement.

    ``history_df`` arrives ordered gameweek DESC, created_at DESC (see
    ``dashboard.data.decisions.get_decision_history``), so the first row of
    each (gameweek, decision_type) group is that gameweek's final run.

    A no-op transfer row is not an event -- a re-run that recommended nothing
    rendered as "none -> none, +0.0 xPts" -- so it is skipped rather than
    allowed to stand as the gameweek's answer. Skipping rather than stopping
    at the first row matters: a no-op re-run after the deadline would
    otherwise erase the transfers the week actually made.

    GW1 logs no transfers at all, because there is nothing to transfer from;
    its only row is the ``lineup`` decision that drafted the squad, which
    left the week rendering as a blank gap. It gets an explicit
    ``initial_squad`` entry instead. The check is pinned to gameweek 1 rather
    than to "earliest gameweek present" because the query keeps a sliding
    20-gameweek window -- once that window moves past GW1 its earliest
    gameweek is an ordinary week, not the draft.

    ``up_to_gw`` drops gameweeks after the run being exported. The engine
    plans the next gameweek before its deadline, so while GW1 was current
    the log already held GW2 rows -- and the published gw1.json carried
    three GW2 events, transfers into a squad the GW1 panel above them does
    not contain. Selecting GW1 on the site should show GW1 as it stood.
    It defaults to None so callers that want the whole log still get it.
    """
    if up_to_gw is not None:
        history_df = history_df[history_df["gameweek"] <= up_to_gw]

    entries: list[dict] = []
    for gw in sorted({int(g) for g in history_df["gameweek"]}, reverse=True):
        rows = history_df[history_df["gameweek"] == gw]

        transfers = rows[rows["decision_type"] == "transfers"]
        for _, row in transfers.iterrows():
            if _is_no_op_transfer(row):
                continue
            entries.append(_transfers_entry(row))
            break

        chips = rows[rows["decision_type"] == "chip"]
        if not chips.empty:
            entries.append(_chip_entry(chips.iloc[0]))

        drafted = gw == 1 and (rows["decision_type"] == "lineup").any()
        if drafted and not any(e["gameweek"] == gw for e in entries):
            entries.append({"gameweek": gw, "type": "initial_squad"})

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
        "history": _build_history_entries(history_df, up_to_gw=gw),
    }
