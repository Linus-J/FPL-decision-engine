import pandas as pd
from sqlalchemy import text

from data.db import get_session


def load_fixture_difficulty(season: str | None = None) -> pd.DataFrame:
    db = get_session()
    try:
        season_filter = "AND s.season = :season" if season else ""
        params = {"season": season} if season else {}
        query = text(f"""
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                p.team_id,
                f.team_h_id,
                f.team_a_id,
                CASE WHEN f.team_h_id = p.team_id THEN 1 ELSE 0 END AS is_home,
                COALESCE(gw.is_bgw, 0) AS is_bgw,
                CASE
                    WHEN f.team_h_id = p.team_id THEN t_opp.strength_defence_away
                    ELSE t_opp.strength_defence_home
                END AS opp_defence_strength,
                CASE
                    WHEN f.team_h_id = p.team_id THEN t_opp.strength_attack_away
                    ELSE t_opp.strength_attack_home
                END AS opp_attack_strength,
                CASE
                    WHEN f.team_h_id = p.team_id THEN t_own.strength_attack_home
                    ELSE t_own.strength_attack_away
                END AS own_attack_strength,
                CASE
                    WHEN f.team_h_id = p.team_id THEN t_own.strength_defence_home
                    ELSE t_own.strength_defence_away
                END AS own_defence_strength,
                CASE
                    WHEN f.team_h_id = p.team_id THEN t_own.strength_overall_home
                    ELSE t_own.strength_overall_away
                END AS own_overall_strength,
                COALESCE((
                    SELECT AVG(CASE WHEN f2.team_h_id = p.team_id THEN t2.strength_defence_away ELSE t2.strength_defence_home END)
                    FROM fixtures f2
                    JOIN teams t2 ON t2.id = CASE WHEN f2.team_h_id = p.team_id THEN f2.team_a_id ELSE f2.team_h_id END
                    WHERE (f2.team_h_id = p.team_id OR f2.team_a_id = p.team_id)
                    AND f2.gameweek > s.gameweek AND f2.gameweek <= s.gameweek + 3
                ), 1200) AS next_3gw_avg_opp_defence,
                COALESCE((
                    SELECT AVG(CASE WHEN f2.team_h_id = p.team_id THEN t2.strength_attack_away ELSE t2.strength_attack_home END)
                    FROM fixtures f2
                    JOIN teams t2 ON t2.id = CASE WHEN f2.team_h_id = p.team_id THEN f2.team_a_id ELSE f2.team_h_id END
                    WHERE (f2.team_h_id = p.team_id OR f2.team_a_id = p.team_id)
                    AND f2.gameweek > s.gameweek AND f2.gameweek <= s.gameweek + 3
                ), 1200) AS next_3gw_avg_opp_attack
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            JOIN fixtures f
                ON f.gameweek = s.gameweek
                AND (f.team_h_id = p.team_id OR f.team_a_id = p.team_id)
            JOIN teams t_own ON t_own.id = p.team_id
            JOIN teams t_opp ON t_opp.id = CASE
                WHEN f.team_h_id = p.team_id THEN f.team_a_id
                ELSE f.team_h_id
            END
            LEFT JOIN gameweeks gw ON gw.name = 'Gameweek ' || s.gameweek
            {season_filter}
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
    db = get_session()
    try:
        season_filter = "AND s.season = :season" if season else ""
        params = {"season": season} if season else {}
        query = text(f"""
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                CASE
                    WHEN f.team_h_id = p.team_id THEN COALESCE(fo.home_cs_prob, 0.2)
                    ELSE COALESCE(fo.away_cs_prob, 0.2)
                END AS my_cs_prob,
                CASE
                    WHEN f.team_h_id = p.team_id THEN COALESCE(fo.away_cs_prob, 0.2)
                    ELSE COALESCE(fo.home_cs_prob, 0.2)
                END AS opp_cs_prob,
                COALESCE(fo.btts_prob, 0.5) AS btts_prob
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            JOIN fixtures f
                ON f.gameweek = s.gameweek
                AND (f.team_h_id = p.team_id OR f.team_a_id = p.team_id)
            LEFT JOIN fixture_odds fo ON fo.fixture_id = f.id
            {season_filter}
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def add_odds_features(df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    odds_dedup = odds.drop_duplicates(subset=["player_id", "gameweek", "season"])
    merged = df.merge(odds_dedup[["player_id", "gameweek", "season", "my_cs_prob", "opp_cs_prob", "btts_prob"]],
                      on=["player_id", "gameweek", "season"], how="left")
    merged["my_cs_prob"] = merged["my_cs_prob"].fillna(0.2)
    merged["opp_cs_prob"] = merged["opp_cs_prob"].fillna(0.2)
    merged["btts_prob"] = merged["btts_prob"].fillna(0.5)
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

ODDS_FEATURE_COLS = [
    "my_cs_prob",
    "opp_cs_prob",
    "btts_prob",
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
