import pandas as pd
from sqlalchemy import text

from data.db import get_session


def load_fixture_difficulty(season: str | None = None) -> pd.DataFrame:
    """Per player-GW FDR from the point-in-time fixture context stored on the
    stat row (team_id_season / opponent_team_id / was_home) and the per-season
    TeamSeasonStrength table (Phase-1 T3b). This is season-accurate — a player's
    club and opponents in a past season, not their current club. Keyed on
    player_gw_stats rows (historical), same as before; missing strengths default
    to 1200 via add_fdr_features."""
    db = get_session()
    try:
        season_filter = "AND s.season = :season" if season else ""
        params = {"season": season} if season else {}
        query = text(f"""
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                s.team_id_season AS team_id,
                CASE WHEN s.was_home THEN 1 ELSE 0 END AS is_home,
                COALESCE(g.is_bgw, 0) AS is_bgw,
                CASE WHEN s.was_home THEN tss_opp.strength_defence_away
                     ELSE tss_opp.strength_defence_home END AS opp_defence_strength,
                CASE WHEN s.was_home THEN tss_opp.strength_attack_away
                     ELSE tss_opp.strength_attack_home END AS opp_attack_strength,
                CASE WHEN s.was_home THEN tss_own.strength_attack_home
                     ELSE tss_own.strength_attack_away END AS own_attack_strength,
                CASE WHEN s.was_home THEN tss_own.strength_defence_home
                     ELSE tss_own.strength_defence_away END AS own_defence_strength,
                CASE WHEN s.was_home THEN tss_own.strength_overall_home
                     ELSE tss_own.strength_overall_away END AS own_overall_strength,
                COALESCE((
                    SELECT AVG(CASE WHEN s2.was_home THEN t2.strength_defence_away
                                    ELSE t2.strength_defence_home END)
                    FROM player_gw_stats s2
                    JOIN team_season_strength t2
                        ON t2.season = s2.season AND t2.team_id = s2.opponent_team_id
                    WHERE s2.player_id = s.player_id AND s2.season = s.season
                        AND s2.gameweek > s.gameweek AND s2.gameweek <= s.gameweek + 3
                ), 1200) AS next_3gw_avg_opp_defence,
                COALESCE((
                    SELECT AVG(CASE WHEN s2.was_home THEN t2.strength_attack_away
                                    ELSE t2.strength_attack_home END)
                    FROM player_gw_stats s2
                    JOIN team_season_strength t2
                        ON t2.season = s2.season AND t2.team_id = s2.opponent_team_id
                    WHERE s2.player_id = s.player_id AND s2.season = s.season
                        AND s2.gameweek > s.gameweek AND s2.gameweek <= s.gameweek + 3
                ), 1200) AS next_3gw_avg_opp_attack
            FROM player_gw_stats s
            LEFT JOIN team_season_strength tss_own
                ON tss_own.season = s.season AND tss_own.team_id = s.team_id_season
            LEFT JOIN team_season_strength tss_opp
                ON tss_opp.season = s.season AND tss_opp.team_id = s.opponent_team_id
            LEFT JOIN gameweeks g ON g.id = s.gameweek AND g.season = s.season
            WHERE 1 = 1 {season_filter}
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def add_fdr_features(df: pd.DataFrame, fdr: pd.DataFrame) -> pd.DataFrame:
    fdr_cols = [
        "player_id", "gameweek", "season",
        "is_home",
        "is_bgw",
        "opp_defence_strength",
        "opp_attack_strength",
        "own_attack_strength",
        "own_defence_strength",
        "own_overall_strength",
        "next_3gw_avg_opp_defence",
        "next_3gw_avg_opp_attack",
    ]
    fdr_cols = [c for c in fdr_cols if c in fdr.columns or c in ("player_id", "gameweek", "season")]
    keep: list[str] = [c for c in fdr_cols if c in fdr.columns]
    fdr_dedup: pd.DataFrame = fdr.loc[:, keep].drop_duplicates(subset=["player_id", "gameweek", "season"])

    merged = df.merge(fdr_dedup, on=["player_id", "gameweek", "season"], how="left")

    flag_cols = {"is_home", "is_bgw"}
    strength_cols = {
        "opp_defence_strength", "opp_attack_strength", "own_attack_strength",
        "own_defence_strength", "own_overall_strength",
        "next_3gw_avg_opp_defence", "next_3gw_avg_opp_attack",
    }
    for col in flag_cols | strength_cols:
        default = 0.0 if col in flag_cols else 1200.0
        if col not in merged.columns:
            merged[col] = default
        else:
            fill = 0.0 if col in flag_cols else merged[col].median()
            # median() is NaN when every row is NaN (e.g. no historical
            # fixtures backfilled yet) — fall back to the constant default.
            if pd.isna(fill):
                fill = default
            merged[col] = merged[col].fillna(fill)

    merged["attack_vs_defence"] = (
        merged["own_attack_strength"] / merged["opp_defence_strength"].clip(lower=1)
    )
    merged["defence_vs_attack"] = (
        merged["own_defence_strength"] / merged["opp_attack_strength"].clip(lower=1)
    )

    return merged


def load_player_enrichment(season: str | None = None) -> pd.DataFrame:
    """Per (player_id, gameweek, season) enrichment, point-in-time.

    The dynamic fields (transfers/ownership → price_momentum, transfer_velocity)
    are read from the snapshot as-of the gameweek deadline, NOT the mutable
    players.* row broadcast onto every historical row (Phase-1 leak L3). The
    set-piece role fields are per-(player, season) and safe. injury_severity and
    press_sentiment are not point-in-time-recoverable historically → 0.
    """
    db = get_session()
    try:
        season_filter = "AND s.season = :season" if season else ""
        params = {"season": season} if season else {}
        query = text(f"""
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                COALESCE(sp.is_penalty_taker, 0) AS is_penalty_taker,
                COALESCE(sp.penalty_xg_per_game, 0.0) AS penalty_xg_per_game,
                COALESCE(sp.is_set_piece_taker, 0) AS is_set_piece_taker,
                COALESCE(sp.key_passes_per_game, 0.0) AS key_passes_per_game,
                0 AS injury_severity,
                COALESCE(ps.transfers_in_event, 0) AS transfers_in_event,
                COALESCE(ps.transfers_out_event, 0) AS transfers_out_event,
                COALESCE(ps.selected_by_percent, 0.0) AS selected_by_percent_enrich,
                0.0 AS press_sentiment
            FROM player_gw_stats s
            JOIN gameweeks g ON g.id = s.gameweek AND g.season = s.season
            LEFT JOIN player_state_snapshots ps ON ps.id = (
                SELECT ps2.id FROM player_state_snapshots ps2
                WHERE ps2.player_id = s.player_id
                    AND ps2.season = s.season
                    AND ps2.snapshot_ts < g.deadline_time
                ORDER BY ps2.snapshot_ts DESC LIMIT 1
            )
            LEFT JOIN player_setpiece_roles sp
                ON sp.player_id = s.player_id AND sp.season = s.season
            WHERE 1 = 1 {season_filter}
        """)
        df = pd.read_sql(query, db.bind, params=params)
        df["price_momentum"] = (
            (df["transfers_in_event"] - df["transfers_out_event"])
            / (df["selected_by_percent_enrich"].clip(lower=0.1) * 1000)
        ).clip(-1.0, 1.0)
        df["transfer_velocity"] = (
            df["transfers_in_event"] / (df["selected_by_percent_enrich"].clip(lower=0.1) * 1000)
        ).clip(0.0, 1.0)
        return df
    finally:
        db.close()


def add_enrichment_features(df: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    dynamic = [
        "is_penalty_taker", "penalty_xg_per_game", "is_set_piece_taker",
        "key_passes_per_game", "injury_severity", "press_sentiment",
        "price_momentum", "transfer_velocity",
    ]
    keys = ["player_id", "gameweek", "season"]
    if all(k in enrichment.columns for k in keys) and {"gameweek", "season"}.issubset(df.columns):
        available = [c for c in dynamic if c in enrichment.columns]
        enr = enrichment[keys + available].drop_duplicates(subset=keys)
        merged = df.merge(enr, on=keys, how="left")
    else:
        # Fallback (single-row predict without gameweek/season): take each
        # player's latest enrichment row and broadcast on player_id.
        available = [c for c in dynamic if c in enrichment.columns]
        enr = enrichment.sort_values("gameweek").drop_duplicates(
            subset=["player_id"], keep="last"
        ) if "gameweek" in enrichment.columns else enrichment
        merged = df.merge(enr[["player_id", *available]], on="player_id", how="left")
    for col in available:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    return merged


def load_fixture_odds(season: str | None = None) -> pd.DataFrame:
    """Per player-GW odds features (my/opp clean-sheet + over-2.5) for BOTH the
    historical training path and the live in-season path.

    Sourced from ``historical_fixture_odds`` via the point-in-time context on
    the stat row (``team_id_season``/``opponent_team_id``/``was_home``) — NOT
    the old ``fixtures``-table join, which used the player's *current* club and
    latest odds against historical rows (Phase-1 finding L4). The odds row is
    matched on ``(season, gameweek)`` and the season-correct home/away team pair,
    and only counts when ``fetched_at < deadline(season, gw)`` — proving the
    closing odds were stamped at the deadline, not kickoff (finding C2). Missing
    odds default to 0.2 CS / 0.5 over-2.5 via ``add_odds_features``.

    **Live seasons fall back to ``fixture_odds``.** ``historical_fixture_odds``
    is a football-data.co.uk backfill, so it only ever covers *finished*
    seasons — it ran to 2025-26 while the bot was about to play 2026-27. This
    function is the only odds reader ``minutes_model._build_features`` calls, on
    the training AND the prediction pass alike, so without the fallback every
    2026-27 row would take the ``COALESCE`` defaults: all three odds features
    pinned to a constant for the entire live season, in a model fitted on five
    seasons where they varied. The features would look present and be inert.

    The live leg applies the same as-of discipline as
    ``load_live_odds_asof``: the latest snapshot per fixture stamped at or
    before that gameweek's deadline, never after. Historical wins where both
    exist — it is the settled closing line.
    """
    db = get_session()
    try:
        season_filter = "AND s.season = :season" if season else ""
        params = {"season": season} if season else {}
        query = text(f"""
            WITH live AS (
                SELECT fo.fixture_id, fo.home_cs_prob, fo.away_cs_prob,
                       fo.over25_prob, f.season, f.gameweek,
                       f.team_h_id, f.team_a_id
                FROM fixture_odds fo
                JOIN fixtures f ON f.id = fo.fixture_id
                LEFT JOIN gameweeks lg
                    ON lg.id = f.gameweek AND lg.season = f.season
                WHERE (lg.deadline_time IS NULL OR fo.fetched_at <= lg.deadline_time)
                  AND fo.fetched_at = (
                      SELECT MAX(fo2.fetched_at) FROM fixture_odds fo2
                      WHERE fo2.fixture_id = fo.fixture_id
                        AND (lg.deadline_time IS NULL
                             OR fo2.fetched_at <= lg.deadline_time)
                  )
            )
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                CASE WHEN s.was_home
                     THEN COALESCE(o.home_cs_prob, l.home_cs_prob, 0.2)
                     ELSE COALESCE(o.away_cs_prob, l.away_cs_prob, 0.2) END AS my_cs_prob,
                CASE WHEN s.was_home
                     THEN COALESCE(o.away_cs_prob, l.away_cs_prob, 0.2)
                     ELSE COALESCE(o.home_cs_prob, l.home_cs_prob, 0.2) END AS opp_cs_prob,
                COALESCE(o.over25_prob, l.over25_prob, 0.5) AS over25_prob
            FROM player_gw_stats s
            LEFT JOIN gameweeks g ON g.id = s.gameweek AND g.season = s.season
            LEFT JOIN historical_fixture_odds o
                ON o.season = s.season AND o.gameweek = s.gameweek
                AND (
                    (s.was_home = 1 AND o.home_team_id = s.team_id_season
                        AND o.away_team_id = s.opponent_team_id)
                    OR (s.was_home = 0 AND o.away_team_id = s.team_id_season
                        AND o.home_team_id = s.opponent_team_id)
                )
                AND (g.deadline_time IS NULL OR o.fetched_at < g.deadline_time)
            LEFT JOIN live l
                ON l.season = s.season AND l.gameweek = s.gameweek
                AND (
                    (s.was_home = 1 AND l.team_h_id = s.team_id_season
                        AND l.team_a_id = s.opponent_team_id)
                    OR (s.was_home = 0 AND l.team_a_id = s.team_id_season
                        AND l.team_h_id = s.opponent_team_id)
                )
            WHERE 1 = 1 {season_filter}
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def load_live_odds_asof(season: str, gameweek: int) -> pd.DataFrame:
    """As-of read of the append-only ``fixture_odds`` for the LIVE projection
    path: the latest snapshot per fixture with ``fetched_at <= deadline`` of the
    target GW (Phase-1 findings L4/C2). Multiple fetches accumulate per fixture;
    this excludes any stamped at/after the deadline. Keyed on ``(season, gw)``.
    Returns one row per fixture with home/away CS + BTTS probabilities."""
    db = get_session()
    try:
        query = text("""
            SELECT
                f.id AS fixture_id,
                f.team_h_id,
                f.team_a_id,
                fo.home_cs_prob,
                fo.away_cs_prob,
                fo.btts_prob
            FROM fixtures f
            JOIN gameweeks g ON g.id = :gw AND g.season = :season
            JOIN fixture_odds fo ON fo.fixture_id = f.id
                AND fo.fetched_at <= g.deadline_time
                AND fo.fetched_at = (
                    SELECT MAX(fo2.fetched_at) FROM fixture_odds fo2
                    WHERE fo2.fixture_id = f.id AND fo2.fetched_at <= g.deadline_time
                )
            WHERE f.season = :season AND f.gameweek = :gw
        """)
        return pd.read_sql(query, db.bind, params={"season": season, "gw": gameweek})
    finally:
        db.close()


def add_odds_features(df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    odds_dedup = odds.drop_duplicates(subset=["player_id", "gameweek", "season"])
    merged = df.merge(odds_dedup[["player_id", "gameweek", "season", *ODDS_FEATURE_COLS]],
                      on=["player_id", "gameweek", "season"], how="left")
    merged["my_cs_prob"] = merged["my_cs_prob"].fillna(0.2)
    merged["opp_cs_prob"] = merged["opp_cs_prob"].fillna(0.2)
    merged["over25_prob"] = merged["over25_prob"].fillna(0.5)
    return merged


def load_midweek_flags(target_gw: int) -> dict[int, float]:
    from data.ingestors.midweek import get_team_midweek_flags
    flags = get_team_midweek_flags(target_gw)
    return {team_id: float(v) for team_id, v in flags.items()}


FDR_FEATURE_COLS = [
    "is_home",
    "is_bgw",
    "opp_defence_strength",
    "opp_attack_strength",
    "own_attack_strength",
    "own_defence_strength",
    "own_overall_strength",
    "next_3gw_avg_opp_defence",
    "next_3gw_avg_opp_attack",
    "attack_vs_defence",
    "defence_vs_attack",
]

# ``btts_prob`` was here until 2026-08-17 and was never a real feature. The
# training table stored the over-2.5 probability under that name -- a documented
# proxy, and byte-for-byte identical on all 1900 rows -- while the live API
# rejects the BTTS market outright (HTTP 422), leaving the column NULL and the
# served value pinned at the 0.5 default. So the model fitted a coefficient
# against a variable spanning 0.341-0.81 and applied it to a constant. Using
# ``over25_prob`` directly keeps the identical signal, is populated on both
# sides, and stops the column claiming to be something it never was.
ODDS_FEATURE_COLS = [
    "my_cs_prob",
    "opp_cs_prob",
    "over25_prob",
]

ENRICHMENT_FEATURE_COLS = [
    "is_penalty_taker",
    "penalty_xg_per_game",
    "is_set_piece_taker",
    "key_passes_per_game",
    "injury_severity",
    "press_sentiment",
    "price_momentum",
    "transfer_velocity",
]


# ---------------------------------------------------------------------------
# Rate-feature contract (Phase-2 P2, defect D4)
#
# Model features must be RATES, not season-cumulative VOLUME, and must be
# computed identically on the train (backtest) and serve (live pipeline) paths.
# The rolling `avg_*_{n}gw` features already satisfy this: per-GW quantities
# averaged over a shift(1) window (as-of, leakage-free), from data present on
# both paths.
#
# The columns below are BANNED as model features:
#   - ict_index/influence/creativity/threat: stored only as season-CUMULATIVE
#     (snapshot) — volume, not rate, and read from different sources on the two
#     paths (snapshot as-of vs the mutable players row) → train/serve skew.
#   - form: the T3 prior-window proxy, not reconcilable across paths; the
#     rolling avg_pts_*gw features carry recent-form signal as a clean rate.
#
# A true ICT-rate is deferred: it needs either per-GW ICT or cumulative
# exposure (minutes/appearances) stored alongside the cumulative ICT, neither
# of which the spine currently carries. See plan/phase-2-xpts-engine.md P2.
# ---------------------------------------------------------------------------

CUMULATIVE_BANNED_FEATURES = frozenset(
    {"ict_index", "influence", "creativity", "threat", "form"}
)


def assert_rate_only(feature_cols: list[str]) -> None:
    """Guard: fail fast if a banned season-cumulative/proxy column is used as a
    model feature (D4). Called at import by each component model."""
    offenders = CUMULATIVE_BANNED_FEATURES.intersection(feature_cols)
    if offenders:
        raise ValueError(
            f"Cumulative/proxy features are banned as model inputs (D4): "
            f"{sorted(offenders)}. Use the rolling avg_*_{{n}}gw rates instead."
        )
