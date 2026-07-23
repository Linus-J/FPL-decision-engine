"""cold_start.py — GW1 initial-squad projections with no current-season data.

At GW1 of a new season the within-season rolling features are all empty, so
projections come from the PRIOR season carried across the season boundary
(player_gw_stats is already keyed to the same players.id via the code
crosswalk, T3a). Players with no prior PL data — promoted-team players and
new signings — get a position/price prior instead of a silent 0.0. The §6.5
departure gate (confirmed tier) drops players FPL marks unavailable.

This is the harness + prior-season bridge (plan T7). The richer promoted-team
prior is Phase-2 work; the contract here is: every candidate has a non-default
projection source, and no confirmed leaver enters the pool.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from data.db import get_session

# A player needs at least this many prior-season appearances to use their
# own history rather than the position/price prior.
MIN_PRIOR_APPEARANCES = 5
# Start probability assumed for a player with no usable prior (new signing).
NEW_PLAYER_START_PROB = 0.6
# Position/price prior: expected GW points ≈ base + slope·(price − £4.0m).
_POSITION_BASE = {"GKP": 2.5, "DEF": 2.8, "MID": 3.0, "FWD": 3.2}
_PRICE_SLOPE = 0.15
_MIN_XPTS = 0.5
# FPL status codes that mean "not a Premier League player right now".
_DEPARTED_STATUS = {"u"}


def prior_season_of(season: str) -> str:
    """'2026-27' -> '2025-26'."""
    start = int(season.split("-")[0])
    return f"{start - 1}-{str(start)[-2:]}"


def load_prior_season_features(prior_season: str) -> pd.DataFrame:
    """Per-player prior-season summary (appearances, points-per-appearance,
    start rate) from player_gw_stats. Keyed by players.id."""
    db = get_session()
    try:
        query = text("""
            SELECT
                player_id,
                SUM(CASE WHEN minutes > 0 THEN 1 ELSE 0 END) AS appearances,
                SUM(total_points) AS total_points,
                AVG(CASE WHEN minutes >= 60 THEN 1.0 ELSE 0.0 END) AS starts_rate,
                SUM(CASE WHEN minutes > 0 THEN total_points ELSE 0 END) AS points_when_played
            FROM player_gw_stats
            WHERE season = :season
            GROUP BY player_id
        """)
        df = pd.read_sql(query, db.bind, params={"season": prior_season})
        if df.empty:
            return pd.DataFrame(
                columns=["player_id", "appearances", "ppg_played", "starts_rate"]
            )
        df["appearances"] = df["appearances"].fillna(0).astype(int)
        df["ppg_played"] = (
            df["points_when_played"] / df["appearances"].clip(lower=1)
        ).where(df["appearances"] > 0, 0.0)
        return df[["player_id", "appearances", "ppg_played", "starts_rate"]]
    finally:
        db.close()


def load_current_players() -> pd.DataFrame:
    """Candidate universe for the initial squad: the current bootstrap players."""
    db = get_session()
    try:
        query = text("""
            SELECT id, web_name, position, now_cost, status, team_id
            FROM players
        """)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()


def apply_departure_gate(players: pd.DataFrame) -> pd.DataFrame:
    """§6.5 confirmed tier: drop players FPL marks unavailable (status 'u')."""
    if "status" not in players.columns:
        return players
    return players[~players["status"].isin(_DEPARTED_STATUS)].copy()


def _price_prior(position: str, now_cost: float) -> float:
    base = _POSITION_BASE.get(position, 3.0)
    return max(_MIN_XPTS, base + _PRICE_SLOPE * (now_cost - 4.0))


def project_cold_start(
    players: pd.DataFrame,
    prior_features: pd.DataFrame,
    target_gw: int = 1,
) -> pd.DataFrame:
    """GW1 xPts + start probability per player, tagged with its source.

    proj_source is 'prior_season' for players with enough prior appearances,
    else 'position_price_prior'. Never 0.0 by default — the gate depends on it.
    """
    merged = players.merge(
        prior_features, left_on="id", right_on="player_id", how="left"
    )
    merged["appearances"] = merged["appearances"].fillna(0).astype(int)

    rows: list[dict] = []
    for r in merged.itertuples():
        has_prior = r.appearances >= MIN_PRIOR_APPEARANCES
        if has_prior:
            xpts = max(_MIN_XPTS, float(r.ppg_played))
            start_prob = float(r.starts_rate)
            source = "prior_season"
        else:
            xpts = _price_prior(r.position, float(r.now_cost))
            start_prob = NEW_PLAYER_START_PROB
            source = "position_price_prior"
        rows.append({
            "player_id": int(r.id),
            "gameweek": target_gw,
            "xpts": xpts,
            "start_probability": start_prob,
            "proj_source": source,
        })
    return pd.DataFrame(rows)


def build_initial_squad(season: str, budget: float | None = None):
    """Construct the GW1 initial 15 from prior-season data only.

    Returns (SquadSolution, projections_df). Imports the optimiser lazily so the
    projection layer stays testable without PuLP.
    """
    from config.strategy import SQUAD
    from optimiser.squad import optimise_squad

    budget = SQUAD.budget_total if budget is None else budget
    players = apply_departure_gate(load_current_players())
    prior = load_prior_season_features(prior_season_of(season))
    projections = project_cold_start(players, prior)

    players = players.merge(
        projections[["player_id", "start_probability"]],
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    solution = optimise_squad(
        projections=projections, players=players, budget=budget, horizon=1
    )
    return solution, projections
