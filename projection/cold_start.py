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

import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.strategy import PRIOR_LEAGUE, SCORING
from data.db import get_session
from data.overrides import apply_team_overrides
from projection.assists import expected_assist_points
from projection.fixture_adjust import fixture_multiplier
from projection.goals import expected_goal_points

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# AVAILABILITY WEIGHTING (2026-08-16, plan/decision-engine-recovery-plan.md P0.1)
#
# Every prior-based tier below produces a PER-APPEARANCE value: ``ppg_played``
# is ``points_when_played / appearances``, the peer buckets pool the same
# per-appearance numbers, and the prior-league tier builds a per-MATCH figure
# out of per-90 rates. ``projection/assemble.py`` — the in-season engine these
# projections are supposed to be interchangeable with — instead produces a
# scenario mean over minutes bands, which already includes the scenarios where
# the player does not feature at all. The two were therefore on different
# scales, and ``start_probability`` was only ever used as a hard >= 0.4 filter,
# never as a multiplier: a rotation risk scoring 6/appearance was valued
# exactly like a nailed player scoring 6/appearance.
#
# Fix: convert each tier's conditional (mean, variance) to the unconditional
# pair via the standard mixture decomposition, so cold-start xpts means the
# same thing as in-season xpts.
# ---------------------------------------------------------------------------

# Assumed appearance probability when a player has no usable availability
# history anywhere (no PL record, no prior-league match, and their peer bucket
# is too sparse to pool one). Matches NEW_PLAYER_START_PROB's spirit — a new
# signing is assumed a probable but not certain regular — while staying a
# distinct quantity: P(features at all) sits at or above P(starts).
NEW_PLAYER_APPEARANCE_PROB = 0.7


def appearance_probability(
    gws_appeared: int, first_gw: int | None, season_last_gw: int
) -> float:
    """P(features in a given gameweek), measured over the window from the
    player's FIRST appearance to the end of the prior season rather than the
    whole season.

    The window matters: a January arrival who then played every week is a
    nailed player, not a 50%-availability one, and a whole-season denominator
    would halve their projection. A player who instead played the first half
    and was then injured out keeps a full-season window (their first
    appearance is GW1), so genuine availability risk is still priced — which
    is the asymmetry we want.

    Known limitation: a player whose only appearances are a handful of late
    gameweeks gets a short window and so a high probability off a small
    sample. ``MIN_PRIOR_APPEARANCES`` is the existing guard — anyone below it
    never reaches this tier at all.
    """
    if season_last_gw <= 0 or first_gw is None or gws_appeared <= 0:
        return 0.0
    window = season_last_gw - int(first_gw) + 1
    if window <= 0:
        return 0.0
    return min(1.0, gws_appeared / window)


def unconditional_moments(
    p_appear: float, mean_played: float, var_played: float
) -> tuple[float, float]:
    """(E[X], Var(X)) for the unconditional per-gameweek points X, given the
    per-APPEARANCE mean/variance and P(appears).

    X is 0 when the player doesn't feature, so with A = 1{appears}:
        E[X]   = p * mean_played
        Var(X) = E[Var(X|A)] + Var(E[X|A])
               = p * var_played + p*(1-p) * mean_played^2
    The second term is what makes a rotation risk genuinely higher-variance
    than a nailed player with the same per-appearance return — previously
    invisible to the optimiser, which saw only the conditional variance.
    """
    p = min(1.0, max(0.0, float(p_appear)))
    mean = p * mean_played
    var = p * var_played + p * (1.0 - p) * mean_played ** 2
    return mean, var


_POINTS_PER_GOAL = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}


def penalty_bonus(
    position: str, penalty_xg_per_game: float, prior_penalty_xg: float
) -> tuple[float, float]:
    """Per-APPEARANCE (mean, variance) that a newly-appointed penalty taker's
    prior-season record does not already contain.

    The in-season engine (``projection/assemble.py``) reads penalty duty from
    the depth chart; the cold start never did, because it works from prior-
    season points per appearance and those points ALREADY include whatever
    penalties the player took last year. Adding duty on top for an established
    taker would double-count -- Haaland's 25/26 points contain his 25/26
    penalties, and he is still on them.

    The exception is a player who is on penalties NOW and was not before. Their
    prior-season points contain no penalty component at all, so the duty is
    purely additive and they are genuinely under-projected. On 2026-08-17 that
    was six players -- Isak, Solanke, Buendía, Iwobi, Kluivert and Groß -- all
    with 25/26 PL minutes and zero penalty xG in them.

    ``prior_penalty_xg`` (the player's prior-season ``xg - npxg``) is what
    separates the two cases. It is only trustworthy because the shot-level
    npxG ingest landed on 2026-08-16; before that npxg was a verbatim copy of
    xg and this difference was zero for everyone, which would have handed the
    bonus to every taker including the ones already carrying it.

    Penalty goals are modelled as Poisson(rate), so the goal-point
    contribution has mean ``rate * G`` and variance ``rate * G^2``. Assists,
    bonus and the small chance of a miss-driven swing are ignored: this is a
    correction to a known-missing component, not a second scoring model.
    """
    if penalty_xg_per_game <= 0.0 or prior_penalty_xg > _PRIOR_PENALTY_EPS:
        return 0.0, 0.0
    goal_points = _POINTS_PER_GOAL.get(position, 4)
    rate = float(penalty_xg_per_game)
    return rate * goal_points, rate * goal_points ** 2


# Prior-season penalty xG at or below this counts as "took no penalties". A
# single penalty is worth ~0.79 xG, so anything this small is rounding, not a
# spot-kick.
_PRIOR_PENALTY_EPS = 0.05


def _price_bucket(position: str, now_cost: float) -> tuple[str, float]:
    return position, round(now_cost / _PRICE_BUCKET_ROUND) * _PRICE_BUCKET_ROUND


def _build_peer_buckets(
    players: pd.DataFrame, prior_features: pd.DataFrame, raw_appearances: pd.DataFrame
) -> tuple[dict[tuple[str, float], np.ndarray], dict[tuple[str, float], list[float]]]:
    """Pools real per-appearance points from ESTABLISHED players (>=
    MIN_PRIOR_APPEARANCES) into (position, price-bucket) groups -- the
    empirical distribution a new signing/promoted player with no top-
    flight history is assigned from, replacing the old synthetic linear
    formula with actually-observed outcomes.

    Returns ``(points_buckets, appearance_buckets)``: the second pools the
    same peers' own ``p_appear`` (P0.1). A new signing has no availability
    history of their own, so "how often do players like this actually
    feature" is the honest prior -- and it must come from the same peer
    group as the points, or the two halves of ``unconditional_moments``
    would describe different populations."""
    if raw_appearances.empty:
        return {}, {}
    merged = players.merge(prior_features, left_on="id", right_on="player_id", how="left")
    merged["appearances"] = merged["appearances"].fillna(0).astype(int)
    established = merged[merged["appearances"] >= MIN_PRIOR_APPEARANCES]
    cols = ["position", "now_cost"]
    if "p_appear" in established.columns:
        cols = cols + ["p_appear"]
    pos_price = established.set_index("id")[cols]

    grouped = raw_appearances[raw_appearances["player_id"].isin(pos_price.index)]
    buckets: dict[tuple[str, float], list[float]] = {}
    appear_buckets: dict[tuple[str, float], list[float]] = {}
    for pid, points in grouped.groupby("player_id")["total_points"]:
        position = pos_price.at[pid, "position"]
        now_cost = float(pos_price.at[pid, "now_cost"])
        key = _price_bucket(position, now_cost)
        buckets.setdefault(key, []).extend(points.tolist())
        if "p_appear" in pos_price.columns:
            p_appear = pos_price.at[pid, "p_appear"]
            if not pd.isna(p_appear):
                appear_buckets.setdefault(key, []).append(float(p_appear))
    return {key: np.array(values) for key, values in buckets.items()}, appear_buckets


def _peer_bucket_stats(
    position: str,
    now_cost: float,
    buckets: dict[tuple[str, float], np.ndarray],
    appear_buckets: dict[tuple[str, float], list[float]] | None = None,
) -> tuple[float, float, float] | None:
    """(mean, sample variance, mean peer p_appear) pooled from real peers in
    the same (position, price-bucket), widening to a position-only pool if
    the exact price bucket is too sparse. ``None`` if even the position-only
    pool is too sparse -- caller falls back to the synthetic linear prior
    (should be rare; every position has many established players).

    The mean/variance are still PER-APPEARANCE; the third element is the
    availability the caller pairs with them via ``unconditional_moments``.
    Falls back to ``NEW_PLAYER_APPEARANCE_PROB`` when no peer availability
    was poolable (e.g. a ``prior_features`` frame predating P0.1)."""
    key = _price_bucket(position, now_cost)
    pool = buckets.get(key, np.array([]))
    appear_pool = list((appear_buckets or {}).get(key, []))
    if len(pool) < _MIN_BUCKET_SAMPLES:
        position_pools = [v for k, v in buckets.items() if k[0] == position]
        pool = np.concatenate(position_pools) if position_pools else np.array([])
        appear_pool = [
            p for k, v in (appear_buckets or {}).items() if k[0] == position for p in v
        ]
    if len(pool) < _MIN_BUCKET_SAMPLES:
        return None
    p_appear = (
        float(np.mean(appear_pool)) if appear_pool else NEW_PLAYER_APPEARANCE_PROB
    )
    return float(pool.mean()), float(pool.var(ddof=1)), p_appear


def load_prior_season_features(prior_season: str) -> pd.DataFrame:
    """Per-player prior-season summary (appearances, points-per-appearance,
    start rate, appearance probability) from player_gw_stats. Keyed by
    players.id.

    ``p_appear`` (P0.1) is the availability weight ``project_cold_start``
    multiplies the per-appearance figures by -- see
    ``appearance_probability`` for the windowing rule. ``gws_appeared`` is
    DISTINCT gameweeks rather than a row count so a genuine double-gameweek
    (two rows, same gameweek) counts once; ``appearances`` deliberately
    keeps its existing row-count meaning, since ``ppg_played`` divides by
    it and ``load_prior_season_appearances`` pools per-fixture rows."""
    db = get_session()
    try:
        query = text("""
            SELECT
                player_id,
                SUM(CASE WHEN minutes > 0 THEN 1 ELSE 0 END) AS appearances,
                COUNT(DISTINCT CASE WHEN minutes > 0 THEN gameweek END) AS gws_appeared,
                MIN(CASE WHEN minutes > 0 THEN gameweek END) AS first_gw,
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
                columns=[
                    "player_id", "appearances", "ppg_played", "starts_rate", "p_appear",
                ]
            )
        season_last_gw = int(
            db.execute(
                text("SELECT MAX(gameweek) FROM player_gw_stats WHERE season = :season"),
                {"season": prior_season},
            ).scalar()
            or 0
        )
        df["appearances"] = df["appearances"].fillna(0).astype(int)
        df["gws_appeared"] = df["gws_appeared"].fillna(0).astype(int)
        df["ppg_played"] = (
            df["points_when_played"] / df["appearances"].clip(lower=1)
        ).where(df["appearances"] > 0, 0.0)
        df["p_appear"] = [
            appearance_probability(
                int(gws), None if pd.isna(first) else int(first), season_last_gw
            )
            for gws, first in zip(df["gws_appeared"], df["first_gw"], strict=True)
        ]
        return df[
            ["player_id", "appearances", "ppg_played", "starts_rate", "p_appear"]
        ]
    finally:
        db.close()


def load_current_players() -> pd.DataFrame:
    """Candidate universe for the initial squad: the current bootstrap
    players, with any manual team_id correction (Feature B, plan 2026-08-10)
    already applied -- a confirmed transfer FPL hasn't updated team_id for
    yet is visible to the max-3-per-club constraint and fixture lookahead
    from here on."""
    db = get_session()
    try:
        query = text("""
            SELECT id, code, web_name, position, now_cost, status, team_id
            FROM players
        """)
        players = pd.read_sql(query, db.bind)
    finally:
        db.close()
    return apply_team_overrides(players)


def load_prior_league_lookup(season: str) -> dict[int, dict]:
    """code -> matched prior_league_stats row for the season immediately
    before ``season`` (e.g. season="2026-27" reads 2025-26 prior-league
    data) -- the P11 translated prior for players with no PL history.
    Empty dict (never crashes) if nothing has been ingested yet."""
    from data.ingestors.fbref import SEASON_MAP
    from projection.prior_league_translation import MIN_HOLDOUT_MINUTES

    prior_season = prior_season_of(season)
    soccerdata_season = SEASON_MAP.get(prior_season, prior_season)
    db = get_session()
    try:
        query = text("""
            SELECT code, league, goals90, assists90, npxg90, xa90, minutes, matches
            FROM prior_league_stats
            WHERE season = :season AND code IS NOT NULL AND minutes >= :min_minutes
        """)
        df = pd.read_sql(
            query, db.bind,
            params={"season": soccerdata_season, "min_minutes": MIN_HOLDOUT_MINUTES},
        )
    finally:
        db.close()
    if df.empty:
        return {}
    # a code should map to exactly one league per season; if a mid-season
    # transfer somehow produced two rows, keep the one with more minutes.
    df = df.sort_values("minutes", ascending=False).drop_duplicates(subset="code", keep="first")
    return df.set_index("code").to_dict("index")


def load_new_penalty_duty(season: str, prior_season: str) -> dict[int, float]:
    """player_id -> expected penalty GOALS per game, for takers whose prior
    season contains no penalties of their own.

    Established takers are deliberately absent: their prior-season points
    already carry their penalties, so ``project_cold_start`` must not add them
    a second time. See ``penalty_bonus`` for the full argument.

    A player with a depth-chart duty and no prior-season xG row at all (a new
    signing, a promoted player) is included — there is no prior penalty
    component to double-count.
    """
    from sqlalchemy import text

    from data.db import get_session

    db = get_session()
    try:
        rows = db.execute(
            text(
                "SELECT r.player_id, r.penalty_xg_per_game, "
                "       COALESCE(x.pen_xg, 0.0) AS prior_pen_xg "
                "FROM player_setpiece_roles r "
                "LEFT JOIN ("
                "    SELECT player_id, SUM(xg) - SUM(npxg) AS pen_xg "
                "    FROM player_xg_stats WHERE season = :prior GROUP BY player_id"
                ") x ON x.player_id = r.player_id "
                "WHERE r.season = :season AND r.penalty_order IS NOT NULL"
            ),
            {"season": season, "prior": prior_season},
        ).fetchall()
    finally:
        db.close()

    return {
        int(pid): float(rate or 0.0)
        for pid, rate, prior_pen_xg in rows
        if float(rate or 0.0) > 0.0 and float(prior_pen_xg or 0.0) <= _PRIOR_PENALTY_EPS
    }


def load_current_defence_strength(season: str) -> dict[int, float]:
    """team_id -> average(strength_defence_home, strength_defence_away) for
    ``season``, treating an all-zero row (FPL hasn't published it yet -- the
    real 2026-27 pre-season state as of 2026-08-10) as ABSENT rather than a
    genuine 0 -- callers fall through to the prior-season fallback instead
    of being misled by a value on the wrong scale (using
    strength_overall_home/away, which IS populated this early but on an
    incompatible ~2-5 scale vs strength_defence's ~1000-1400, was
    considered and rejected -- see the design spec)."""
    db = get_session()
    try:
        query = text("""
            SELECT team_id, strength_defence_home, strength_defence_away
            FROM team_season_strength WHERE season = :season
        """)
        df = pd.read_sql(query, db.bind, params={"season": season})
    finally:
        db.close()
    result: dict[int, float] = {}
    for r in df.itertuples():
        avg = (r.strength_defence_home + r.strength_defence_away) / 2
        if avg > 0:
            result[int(r.team_id)] = avg
    return result


def load_team_codes(season: str) -> dict[int, int]:
    """team_id -> stable code for ``season`` (only rows where FPL has
    supplied one) -- used to resolve an opponent's PRIOR-season strength via
    the identity that survives promotion/relegation reshuffling team_id."""
    db = get_session()
    try:
        query = text("""
            SELECT team_id, code FROM team_season_strength
            WHERE season = :season AND code IS NOT NULL
        """)
        df = pd.read_sql(query, db.bind, params={"season": season})
    finally:
        db.close()
    return {int(r.team_id): int(r.code) for r in df.itertuples()}


def load_prior_defence_strength_by_code(prior_season: str) -> dict[int, float]:
    """code -> average defence strength from ``prior_season`` -- the
    fallback used when the CURRENT season's strength is still unpublished,
    so Feature A has real fixture-difficulty signal now rather than only
    once FPL catches up close to the GW1 deadline."""
    db = get_session()
    try:
        query = text("""
            SELECT code, strength_defence_home, strength_defence_away
            FROM team_season_strength WHERE season = :season AND code IS NOT NULL
        """)
        df = pd.read_sql(query, db.bind, params={"season": prior_season})
    finally:
        db.close()
    result: dict[int, float] = {}
    for r in df.itertuples():
        avg = (r.strength_defence_home + r.strength_defence_away) / 2
        if avg > 0:
            result[int(r.code)] = avg
    return result


def load_horizon_fixtures(
    players: pd.DataFrame, season: str, target_gw: int, horizon: int,
) -> pd.DataFrame:
    """(player_id, gameweek, opp_defence_strength, was_home) for each of the
    ``horizon`` GWs starting at ``target_gw``, resolved from ``players``'
    OWN team_id column -- post Feature-B override, since it is never
    re-derived by re-querying the players table from the DB. This is what
    lets a manual team_id correction (Feature B) actually change which
    fixtures a player is attributed to.

    opp_defence_strength resolution, per fixture: (1) current season's
    TeamSeasonStrength if non-zero, (2) prior-season TeamSeasonStrength for
    the same club, joined on the stable `code` (not team_id -- a per-season
    alphabetical index that shifts under promotion/relegation), (3) None if
    neither exists -- `fixture_multiplier` already treats None as neutral
    (1.0), so a promoted club with no 2025-26 row degrades safely rather
    than crashing or defaulting to a misleading value.
    """
    empty = pd.DataFrame(columns=["player_id", "gameweek", "opp_defence_strength", "was_home"])
    if players.empty or horizon <= 0:
        return empty

    team_ids = sorted({int(t) for t in players["team_id"].dropna().unique()})
    target_gws = list(range(target_gw, target_gw + horizon))
    if not team_ids or not target_gws:
        return empty

    db = get_session()
    try:
        team_placeholders = ",".join(f":team{i}" for i in range(len(team_ids)))
        gw_placeholders = ",".join(f":gw{i}" for i in range(len(target_gws)))
        params = {
            "season": season,
            **{f"team{i}": tid for i, tid in enumerate(team_ids)},
            **{f"gw{i}": gw for i, gw in enumerate(target_gws)},
        }
        query = text(f"""
            SELECT f.team_h_id, f.team_a_id, f.gameweek
            FROM fixtures f
            WHERE f.season = :season AND f.gameweek IN ({gw_placeholders})
              AND (f.team_h_id IN ({team_placeholders}) OR f.team_a_id IN ({team_placeholders}))
        """)
        raw = pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()
    if raw.empty:
        return empty

    current_strength = load_current_defence_strength(season)
    team_codes = load_team_codes(season)
    prior_strength_by_code = load_prior_defence_strength_by_code(prior_season_of(season))

    def _resolve(opp_team_id: int) -> float | None:
        if opp_team_id in current_strength:
            return current_strength[opp_team_id]
        code = team_codes.get(opp_team_id)
        if code is not None and code in prior_strength_by_code:
            return prior_strength_by_code[code]
        return None

    team_id_set = set(team_ids)
    fixture_rows: list[dict] = []
    for r in raw.itertuples():
        if r.team_h_id in team_id_set:
            fixture_rows.append({
                "team_id": r.team_h_id, "gameweek": r.gameweek,
                "opp_defence_strength": _resolve(r.team_a_id), "was_home": True,
            })
        if r.team_a_id in team_id_set:
            fixture_rows.append({
                "team_id": r.team_a_id, "gameweek": r.gameweek,
                "opp_defence_strength": _resolve(r.team_h_id), "was_home": False,
            })
    fixtures_by_team = pd.DataFrame(
        fixture_rows, columns=["team_id", "gameweek", "opp_defence_strength", "was_home"]
    )
    if fixtures_by_team.empty:
        return empty

    resolved_team_ids = set(fixtures_by_team["team_id"].unique())
    for tid in sorted(team_id_set - resolved_team_ids):
        logger.warning(
            "load_horizon_fixtures: team_id %s has no fixtures in GWs %s-%s (season %s) -- "
            "its players will get zero horizon rows and may drop out of the candidate pool "
            "(check team_id is correct, e.g. a transfer_overrides.yaml typo)",
            tid, target_gw, target_gw + horizon - 1, season,
        )

    merged = players[["id", "team_id"]].merge(fixtures_by_team, on="team_id", how="inner")
    return merged.rename(columns={"id": "player_id"})[
        ["player_id", "gameweek", "opp_defence_strength", "was_home"]
    ]


def apply_departure_gate(players: pd.DataFrame) -> pd.DataFrame:
    """§6.5 confirmed tier: drop players FPL marks unavailable (status 'u')."""
    if "status" not in players.columns:
        return players
    return players[~players["status"].isin(_DEPARTED_STATUS)].copy()


def _price_prior(position: str, now_cost: float) -> float:
    base = _POSITION_BASE.get(position, 3.0)
    return max(_MIN_XPTS, base + _PRICE_SLOPE * (now_cost - 4.0))


# Games in a full season, by league -- Championship plays 46 (24 clubs),
# Bundesliga/Ligue 1 play 34 (18 clubs), La Liga/Serie A/PL play 38 (20
# clubs, the default below). Used as prior_minutes_share's denominator so
# it measures real season-long availability, not "when picked, does he
# start" (which a 3-appearance fringe player could ace).
_SEASON_MATCHES = {"ENG-Championship": 46, "GER-Bundesliga": 34, "FRA-Ligue 1": 34}
_DEFAULT_SEASON_MATCHES = 38


def _season_length(league: str) -> int:
    return _SEASON_MATCHES.get(league, _DEFAULT_SEASON_MATCHES)


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
    # npxG90/xA90 are preferred -- smoother and luck-adjusted, where one
    # season's raw output is a small, high-variance sample. But the FBref
    # prior-league scrape only populated the basic stats: measured 2026-08-16,
    # goals90/assists90 have 7,413/7,097 non-zero rows and npxg90/xa90/sca90
    # have ZERO. Using them regardless projected every one of the 28
    # league-tier players at the 2.0 floor, which makes a genuinely good
    # foreign signing invisible to the optimiser.
    #
    # Falling back to raw goals90/assists90 is also closer to how the
    # translation factor was calibrated in the first place -- it was fit
    # against realized RAW goal+assist output (see
    # projection/prior_league_translation.py); the expected-goals inputs were
    # an application-time refinement.
    npxg90 = float(pl_row.get("npxg90") or 0.0)
    xa90 = float(pl_row.get("xa90") or 0.0)
    if npxg90 <= 0.0 and xa90 <= 0.0:
        npxg90 = float(pl_row.get("goals90") or 0.0)
        xa90 = float(pl_row.get("assists90") or 0.0)
    translated_npxg90 = npxg90 * factor
    translated_xa90 = xa90 * factor
    # Per-MATCH (conditional on featuring): per-90 rates plus the appearance
    # points a player only collects by playing.
    xpts_played = max(
        _MIN_XPTS,
        expected_goal_points(translated_npxg90, position)
        + expected_assist_points(translated_xa90)
        + SCORING.points_full_appearance,
    )
    var_played = PRIOR_LEAGUE.translation_variance(pl_row["league"])
    season_matches = _season_length(pl_row["league"])
    # P0.1: matches played out of their league's own season length -- the
    # prior-league analogue of ``p_appear``. Unlike the PL tier there is no
    # per-gameweek record to window by first appearance, so a mid-season
    # arrival abroad is under-credited here; flagged, not solved.
    p_appear = min(1.0, float(pl_row.get("matches") or 0) / max(1, season_matches))
    xpts, xpts_var = unconditional_moments(p_appear, xpts_played, var_played)
    prior_minutes_share = min(
        1.0, pl_row["minutes"] / max(1, season_matches * 90)
    )
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
    horizon: int = 1,
    season: str | None = None,
    penalty_duty: dict[int, float] | None = None,
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

    ``penalty_duty`` (optional, from ``load_new_penalty_duty``): player_id ->
    expected penalty goals per game, for takers NEW to the duty only. It must
    not contain established takers, whose prior-season points already include
    their penalties — the loader is what enforces that, and passing a raw
    depth chart here would double-count them. ``None`` reproduces the previous
    behaviour exactly.

    The ``prior_league_prior`` tier is deliberately left untouched by it. That
    tier projects from a translated foreign record whose goals already include
    whatever penalties the player took abroad, and nothing in the data says
    whether he was on them — so a bonus there is as likely to double-count as
    to correct. Two players were affected on 2026-08-17, neither near
    selection; the ambiguity was not worth buying.

    ``horizon``/``season`` (default 1/None, preserving today's exact
    single-row-per-player, unscaled-by-fixture behaviour for every existing
    caller, since those never pass ``season``): with ``season`` given, emits
    one row per ``(player, gw)`` for ``gw`` in
    ``[target_gw, target_gw + horizon)`` -- one row when ``horizon == 1`` --
    with xpts/xpts_var scaled by that GW's fixture_multiplier (plan
    2026-08-10, cold-start fixture lookahead). ``season`` is required to
    resolve fixtures/team strengths -- if it is None, degrades to the
    single-GW unscaled base row (regardless of ``horizon``) rather than
    crashing (mirrors ``load_horizon_fixtures`` returning empty when it
    has nothing to resolve).
    """
    if raw_appearances is None:
        raw_appearances = pd.DataFrame(columns=["player_id", "total_points"])
    if prior_league_lookup is None:
        prior_league_lookup = {}
    if penalty_duty is None:
        penalty_duty = {}

    merged = players.merge(
        prior_features, left_on="id", right_on="player_id", how="left"
    )
    merged["appearances"] = merged["appearances"].fillna(0).astype(int)

    peer_buckets, peer_appear_buckets = _build_peer_buckets(
        players, prior_features, raw_appearances
    )
    own_appearances = raw_appearances.groupby("player_id")["total_points"]
    # A prior_features frame predating P0.1 has no p_appear column; treat that
    # as "no availability information" (weight 1.0), which reproduces the old
    # per-appearance behaviour exactly rather than silently deflating.
    has_p_appear = "p_appear" in merged.columns

    rows: list[dict] = []
    for r in merged.itertuples():
        # Per-APPEARANCE penalty component, non-zero only for a player who is
        # on penalties now and was not last season (``load_new_penalty_duty``
        # filters out everyone whose prior points already include them).
        pen_mean, pen_var = penalty_bonus(
            r.position, float(penalty_duty.get(int(r.id), 0.0)), 0.0
        )

        has_prior = r.appearances >= MIN_PRIOR_APPEARANCES
        if has_prior:
            mean_played = max(_MIN_XPTS, float(r.ppg_played)) + pen_mean
            if r.id in own_appearances.groups:
                own_points = own_appearances.get_group(r.id)
                var_played = float(own_points.var(ddof=1)) if len(own_points) > 1 else 0.0
            else:
                var_played = 0.0
            var_played += pen_var
            p_appear = 1.0
            if has_p_appear and not pd.isna(r.p_appear):
                p_appear = float(r.p_appear)
            # P0.1: the floor applies to the per-APPEARANCE mean ("when he
            # plays he is worth at least this"), before availability scales
            # it -- flooring the unconditional value instead would erase the
            # very distinction this change exists to make.
            xpts, xpts_var = unconditional_moments(p_appear, mean_played, var_played)
            start_prob = float(r.starts_rate)
            source = "prior_season"
        else:
            peer_stats = _peer_bucket_stats(
                r.position, float(r.now_cost), peer_buckets, peer_appear_buckets
            )
            if peer_stats is not None:
                fallback_xpts, fallback_xpts_var = unconditional_moments(
                    peer_stats[2],
                    max(_MIN_XPTS, peer_stats[0]) + pen_mean,
                    peer_stats[1] + pen_var,
                )
                fallback_source = "peer_bucket_prior"
            else:
                fallback_xpts, fallback_xpts_var = unconditional_moments(
                    NEW_PLAYER_APPEARANCE_PROB,
                    _price_prior(r.position, float(r.now_cost)) + pen_mean,
                    _FALLBACK_VAR + pen_var,
                )
                fallback_source = "position_price_prior"

            code = getattr(r, "code", None)
            pl_row = (
                prior_league_lookup.get(int(code))
                if code is not None and not pd.isna(code)
                else None
            )
            if pl_row is not None:
                pl_xpts, pl_xpts_var, pl_start_prob = _prior_league_projection(
                    r.position, pl_row
                )
                # the prior-league tier has no defensive/bonus signal, so it
                # must never score a matched player below what the existing
                # peer-bucket/position-price fallback would have given them
                # (plan/p11-prior-league-cold-start.md final review, Fix 2).
                if pl_xpts >= fallback_xpts:
                    xpts, xpts_var = pl_xpts, pl_xpts_var
                    source = "prior_league_prior"
                else:
                    xpts, xpts_var = fallback_xpts, fallback_xpts_var
                    source = fallback_source
                # start_prob always comes from the prior-league minutes-share
                # signal, even when xpts/xpts_var above came from the
                # fallback instead -- a matched player's own playing-time
                # data is real information regardless of whether their
                # attacking output alone cleared the fallback floor. This
                # means `source` no longer indicates which start_prob rule
                # applied (a matched GKP can show fallback_source with a
                # higher start_prob than an unmatched one at the same price).
                start_prob = pl_start_prob
            else:
                xpts, xpts_var = fallback_xpts, fallback_xpts_var
                start_prob = NEW_PLAYER_START_PROB
                source = fallback_source
        rows.append({
            "player_id": int(r.id),
            "gameweek": target_gw,
            "xpts": xpts,
            "xpts_var": xpts_var,
            "start_probability": start_prob,
            "proj_source": source,
        })
    base_df = pd.DataFrame(rows)
    if horizon < 1 or season is None:
        return base_df

    fixtures = load_horizon_fixtures(players, season, target_gw, horizon)
    if fixtures.empty:
        # No resolvable fixture data (e.g. a synthetic/test season with no
        # fixtures rows at all) -- degrade to repeating the base projection
        # at every horizon GW with an implicit neutral multiplier, rather
        # than silently dropping the extra GWs the caller asked for.
        repeated = []
        for gw in range(target_gw, target_gw + horizon):
            gw_df = base_df.copy()
            gw_df["gameweek"] = gw
            repeated.append(gw_df)
        return pd.concat(repeated, ignore_index=True)

    base_by_player = base_df.set_index("player_id")
    horizon_rows: list[dict] = []
    for f in fixtures.itertuples():
        if f.player_id not in base_by_player.index:
            continue
        base = base_by_player.loc[f.player_id]
        mult = fixture_multiplier(f.opp_defence_strength, f.was_home)
        horizon_rows.append({
            "player_id": f.player_id,
            "gameweek": f.gameweek,
            "xpts": base["xpts"] * mult,
            "xpts_var": base["xpts_var"] * mult ** 2,
            "start_probability": base["start_probability"],
            "proj_source": base["proj_source"],
        })
    return pd.DataFrame(horizon_rows)


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
    from config.strategy import OPTIMISER, SQUAD
    from data.overrides import load_p_leave_overrides, log_rumoured_squad_members
    from optimiser.departure_risk import apply_departure_discount
    from optimiser.squad import optimise_squad

    cfg = config or OPTIMISER
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
        horizon=cfg.cold_start_lookahead_gws, season=season,
        penalty_duty=load_new_penalty_duty(season, prior_season),
    )
    # Feature B (plan 2026-08-10): the rumour-discount tier of the
    # already-existing departure-risk gate, fed with real data for the
    # first time -- previously always an empty dict (Phase 4's news layer
    # was never built), so this call was always a no-op before today.
    projections = apply_departure_discount(projections, load_p_leave_overrides())

    players = players.merge(
        projections[["player_id", "start_probability"]].drop_duplicates("player_id"),
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    solution = optimise_squad(
        projections=projections, players=players, budget=budget,
        horizon=cfg.cold_start_lookahead_gws, season=season, config=config,
    )
    log_rumoured_squad_members(solution.squad["id"].tolist(), players)
    return solution, projections
