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

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.strategy import PRIOR_LEAGUE, SCORING
from data.db import get_session
from projection.assists import expected_assist_points
from projection.goals import expected_goal_points

# A player needs at least this many prior-season appearances to use their
# own history rather than the position/price prior.
MIN_PRIOR_APPEARANCES = 5
# Start probability assumed for a player with no usable prior (new signing).
NEW_PLAYER_START_PROB = 0.6
# Blend weight toward a matched prior-league player's own minutes-share when
# setting their start probability (P11) -- 0.5 is a deliberately moderate
# starting choice, not itself backtested.
_PRIOR_LEAGUE_START_PROB_WEIGHT = 0.5
# Position/price prior: expected GW points ≈ base + slope·(price − £4.0m).
# Last-resort fallback only now (plan/risk-aware-cold-start-v1.md,
# 2026-07-31) -- superseded by _peer_bucket_stats's real peer data
# whenever a (position, price-band) bucket has enough samples.
_POSITION_BASE = {"GKP": 2.5, "DEF": 2.8, "MID": 3.0, "FWD": 3.2}
_PRICE_SLOPE = 0.15
_MIN_XPTS = 0.5
# Variance assumed only when even the position-only peer pool is too
# sparse to estimate one (should be rare -- every position has many
# established players). A moderate, deliberately unremarkable guess.
_FALLBACK_VAR = 4.0
# FPL status codes that mean "not a Premier League player right now".
_DEPARTED_STATUS = {"u"}

# Peer-bucket sizing for new signings / promoted players with no top-
# flight history (plan/risk-aware-cold-start-v1.md). Bucketed by
# (position, price rounded to the nearest amount below); a bucket needs at
# least this many pooled real per-appearance point values before it's
# trusted over a wider fallback.
_PRICE_BUCKET_ROUND = 1.0
_MIN_BUCKET_SAMPLES = 20


def prior_season_of(season: str) -> str:
    """'2026-27' -> '2025-26'."""
    start = int(season.split("-")[0])
    return f"{start - 1}-{str(start)[-2:]}"


def load_prior_season_appearances(prior_season: str) -> pd.DataFrame:
    """Every prior-season (player_id, total_points) row where the player
    actually played (minutes > 0) -- the raw per-appearance values needed
    to compute REAL variance and to pool peer buckets, not just the mean
    ``load_prior_season_features`` already gives. Deliberately one row per
    (player, fixture) -- a genuine DGW gameweek contributes two rows, same
    convention ``load_prior_season_features``'s own ``appearances``/
    ``ppg_played`` already use, so the mean and variance stay consistent
    with each other."""
    db = get_session()
    try:
        query = text("""
            SELECT player_id, total_points
            FROM player_gw_stats
            WHERE season = :season AND minutes > 0
        """)
        return pd.read_sql(query, db.bind, params={"season": prior_season})
    finally:
        db.close()


def _price_bucket(position: str, now_cost: float) -> tuple[str, float]:
    return position, round(now_cost / _PRICE_BUCKET_ROUND) * _PRICE_BUCKET_ROUND


def _build_peer_buckets(
    players: pd.DataFrame, prior_features: pd.DataFrame, raw_appearances: pd.DataFrame
) -> dict[tuple[str, float], np.ndarray]:
    """Pools real per-appearance points from ESTABLISHED players (>=
    MIN_PRIOR_APPEARANCES) into (position, price-bucket) groups -- the
    empirical distribution a new signing/promoted player with no top-
    flight history is assigned from, replacing the old synthetic linear
    formula with actually-observed outcomes."""
    if raw_appearances.empty:
        return {}
    merged = players.merge(prior_features, left_on="id", right_on="player_id", how="left")
    merged["appearances"] = merged["appearances"].fillna(0).astype(int)
    established = merged[merged["appearances"] >= MIN_PRIOR_APPEARANCES]
    pos_price = established.set_index("id")[["position", "now_cost"]]

    grouped = raw_appearances[raw_appearances["player_id"].isin(pos_price.index)]
    buckets: dict[tuple[str, float], list[float]] = {}
    for pid, points in grouped.groupby("player_id")["total_points"]:
        position = pos_price.at[pid, "position"]
        now_cost = float(pos_price.at[pid, "now_cost"])
        key = _price_bucket(position, now_cost)
        buckets.setdefault(key, []).extend(points.tolist())
    return {key: np.array(values) for key, values in buckets.items()}


def _peer_bucket_stats(
    position: str, now_cost: float, buckets: dict[tuple[str, float], np.ndarray]
) -> tuple[float, float] | None:
    """(mean, sample variance) pooled from real peers in the same
    (position, price-bucket), widening to a position-only pool if the
    exact price bucket is too sparse. ``None`` if even the position-only
    pool is too sparse -- caller falls back to the synthetic linear prior
    (should be rare; every position has many established players)."""
    pool = buckets.get(_price_bucket(position, now_cost), np.array([]))
    if len(pool) < _MIN_BUCKET_SAMPLES:
        position_pools = [v for k, v in buckets.items() if k[0] == position]
        pool = np.concatenate(position_pools) if position_pools else np.array([])
    if len(pool) < _MIN_BUCKET_SAMPLES:
        return None
    return float(pool.mean()), float(pool.var(ddof=1))


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
            SELECT id, code, web_name, position, now_cost, status, team_id
            FROM players
        """)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()


def load_prior_league_lookup(season: str) -> dict[int, dict]:
    """code -> matched prior_league_stats row for the season immediately
    before ``season`` (e.g. season="2026-27" reads 2025-26 prior-league
    data) -- the P11 translated prior for players with no PL history.
    Empty dict (never crashes) if nothing has been ingested yet."""
    from data.ingestors.fbref import SEASON_MAP

    prior_season = prior_season_of(season)
    soccerdata_season = SEASON_MAP.get(prior_season, prior_season)
    db = get_session()
    try:
        query = text("""
            SELECT code, league, goals90, assists90, npxg90, xa90, minutes, matches
            FROM prior_league_stats
            WHERE season = :season AND code IS NOT NULL
        """)
        df = pd.read_sql(query, db.bind, params={"season": soccerdata_season})
    finally:
        db.close()
    if df.empty:
        return {}
    # a code should map to exactly one league per season; if a mid-season
    # transfer somehow produced two rows, keep the one with more minutes.
    df = df.sort_values("minutes", ascending=False).drop_duplicates(subset="code", keep="first")
    return df.set_index("code").to_dict("index")


def apply_departure_gate(players: pd.DataFrame) -> pd.DataFrame:
    """§6.5 confirmed tier: drop players FPL marks unavailable (status 'u')."""
    if "status" not in players.columns:
        return players
    return players[~players["status"].isin(_DEPARTED_STATUS)].copy()


def _price_prior(position: str, now_cost: float) -> float:
    base = _POSITION_BASE.get(position, 3.0)
    return max(_MIN_XPTS, base + _PRICE_SLOPE * (now_cost - 4.0))


def _prior_league_projection(position: str, pl_row: dict) -> tuple[float, float, float]:
    """(xpts, xpts_var, start_probability) for a matched prior-league
    player (P11). xpts is built from translated npxG90/xA90 (the smoother,
    luck-adjusted quality metrics -- one prior season's raw goals/assists is
    a small, high-variance sample) plus a flat appearance-points constant.
    The translation factor itself is still fit against realized RAW
    goal+assist output (the actual ground truth being predicted) --
    projection/prior_league_translation.py -- only this application uses
    the smoother inputs. Clean sheets/bonus/cards are NOT estimated:
    prior_league_stats has no defensive data for these players, an honest
    limitation, not an oversight."""
    factor = PRIOR_LEAGUE.translation_factor(pl_row["league"])
    translated_npxg90 = pl_row["npxg90"] * factor
    translated_xa90 = pl_row["xa90"] * factor
    xpts = max(
        _MIN_XPTS,
        expected_goal_points(translated_npxg90, position)
        + expected_assist_points(translated_xa90)
        + SCORING.points_full_appearance,
    )
    xpts_var = PRIOR_LEAGUE.translation_variance(pl_row["league"])
    prior_minutes_share = min(1.0, pl_row["minutes"] / max(1, pl_row["matches"] * 90))
    start_prob = (
        (1 - _PRIOR_LEAGUE_START_PROB_WEIGHT) * NEW_PLAYER_START_PROB
        + _PRIOR_LEAGUE_START_PROB_WEIGHT * prior_minutes_share
    )
    return xpts, xpts_var, start_prob


def project_cold_start(
    players: pd.DataFrame,
    prior_features: pd.DataFrame,
    target_gw: int = 1,
    raw_appearances: pd.DataFrame | None = None,
    prior_league_lookup: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """GW1 xPts + xpts_var + start probability per player, tagged with its
    source.

    proj_source is 'prior_season' (established players, real own-variance),
    'prior_league_prior' (new signings/promoted players matched to a
    translated non-PL prior-season record, P11), 'peer_bucket_prior' (no PL
    or prior-league match, pooled real peer data by position+price), or
    'position_price_prior' (last-resort synthetic fallback). Neither xpts
    nor xpts_var is ever left 0.0/undefined by default -- the gate depends
    on it (plan/risk-aware-cold-start-v1.md, extended to variance).

    ``raw_appearances`` (optional, from ``load_prior_season_appearances``):
    powers the real variance computation. ``None`` (or empty) degrades
    every player straight to the synthetic fallback for BOTH xpts and
    xpts_var -- never crashes.

    ``prior_league_lookup`` (optional, from ``load_prior_league_lookup``):
    code -> translated prior-league row (P11). ``None`` (or a code with no
    entry) falls through to the existing peer_bucket_prior /
    position_price_prior cascade, unchanged.
    """
    if raw_appearances is None:
        raw_appearances = pd.DataFrame(columns=["player_id", "total_points"])
    if prior_league_lookup is None:
        prior_league_lookup = {}

    merged = players.merge(
        prior_features, left_on="id", right_on="player_id", how="left"
    )
    merged["appearances"] = merged["appearances"].fillna(0).astype(int)

    peer_buckets = _build_peer_buckets(players, prior_features, raw_appearances)
    own_appearances = raw_appearances.groupby("player_id")["total_points"]

    rows: list[dict] = []
    for r in merged.itertuples():
        has_prior = r.appearances >= MIN_PRIOR_APPEARANCES
        if has_prior:
            xpts = max(_MIN_XPTS, float(r.ppg_played))
            if r.id in own_appearances.groups:
                own_points = own_appearances.get_group(r.id)
                xpts_var = float(own_points.var(ddof=1)) if len(own_points) > 1 else 0.0
            else:
                xpts_var = 0.0
            start_prob = float(r.starts_rate)
            source = "prior_season"
        else:
            code = getattr(r, "code", None)
            pl_row = (
                prior_league_lookup.get(int(code))
                if code is not None and not pd.isna(code)
                else None
            )
            if pl_row is not None:
                xpts, xpts_var, start_prob = _prior_league_projection(r.position, pl_row)
                source = "prior_league_prior"
            else:
                peer_stats = _peer_bucket_stats(r.position, float(r.now_cost), peer_buckets)
                if peer_stats is not None:
                    xpts, xpts_var = peer_stats
                    xpts = max(_MIN_XPTS, xpts)
                    source = "peer_bucket_prior"
                else:
                    xpts = _price_prior(r.position, float(r.now_cost))
                    xpts_var = _FALLBACK_VAR
                    source = "position_price_prior"
                start_prob = NEW_PLAYER_START_PROB
        rows.append({
            "player_id": int(r.id),
            "gameweek": target_gw,
            "xpts": xpts,
            "xpts_var": xpts_var,
            "start_probability": start_prob,
            "proj_source": source,
        })
    return pd.DataFrame(rows)


def build_initial_squad(
    season: str,
    budget: float | None = None,
    players: pd.DataFrame | None = None,
    config=None,
):
    """Construct the GW1 initial 15 from prior-season data only.

    ``players`` defaults to the live bootstrap (``load_current_players``) —
    the real path this project's own agent uses on the actual 26/27 season's
    opening day. Passing a point-in-time snapshot instead (e.g.
    ``scripts.backtest._load_players_snapshot(season, 1)``) lets this same
    function be validated against a COMPLETED historical season's real GW1
    roster/prices rather than today's live bootstrap — 2026-07-30, the
    user's own request ("we need to have and test a method to start from
    GW1... for the realtime 26/27 season which is approaching").

    ``config`` (optional, ``OptimiserConfig``): passed straight through to
    ``optimise_squad`` — lets the simulation engine cold-start a persona's
    initial squad under its own risk posture. ``None`` is byte-for-byte
    identical to today's behaviour.

    Returns (SquadSolution, projections_df). Imports the optimiser lazily so the
    projection layer stays testable without PuLP.
    """
    from config.strategy import SQUAD
    from optimiser.squad import optimise_squad

    budget = SQUAD.budget_total if budget is None else budget
    if players is None:
        players = load_current_players()
    players = apply_departure_gate(players)
    prior_season = prior_season_of(season)
    prior = load_prior_season_features(prior_season)
    raw_appearances = load_prior_season_appearances(prior_season)
    prior_league_lookup = load_prior_league_lookup(season)
    projections = project_cold_start(
        players, prior, raw_appearances=raw_appearances,
        prior_league_lookup=prior_league_lookup,
    )

    players = players.merge(
        projections[["player_id", "start_probability"]],
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    solution = optimise_squad(
        projections=projections, players=players, budget=budget, horizon=1, season=season,
        config=config,
    )
    return solution, projections
