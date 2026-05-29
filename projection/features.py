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
                END AS own_overall_strength
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
            {season_filter}
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def add_fdr_features(df: pd.DataFrame, fdr: pd.DataFrame) -> pd.DataFrame:
    fdr_cols = [
        "player_id", "gameweek", "season",
        "is_home",
        "opp_defence_strength",
        "opp_attack_strength",
        "own_attack_strength",
        "own_defence_strength",
        "own_overall_strength",
    ]
    fdr_dedup = fdr[fdr_cols].drop_duplicates(subset=["player_id", "gameweek", "season"])

    merged = df.merge(fdr_dedup, on=["player_id", "gameweek", "season"], how="left")

    for col in fdr_cols[3:]:
        if col not in merged.columns:
            merged[col] = 1200.0
        else:
            merged[col] = merged[col].fillna(merged[col].median())

    merged["attack_vs_defence"] = (
        merged["own_attack_strength"] / merged["opp_defence_strength"].clip(lower=1)
    )
    merged["defence_vs_attack"] = (
        merged["own_defence_strength"] / merged["opp_attack_strength"].clip(lower=1)
    )

    return merged


def load_player_enrichment() -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT
                p.id AS player_id,
                COALESCE(sp.is_penalty_taker, 0) AS is_penalty_taker,
                COALESCE(sp.penalty_xg_per_game, 0.0) AS penalty_xg_per_game,
                COALESCE(sp.is_set_piece_taker, 0) AS is_set_piece_taker,
                COALESCE(sp.key_passes_per_game, 0.0) AS key_passes_per_game,
                COALESCE(p.injury_severity, 0) AS injury_severity,
                COALESCE(p.transfers_in_event, 0) AS transfers_in_event,
                COALESCE(p.transfers_out_event, 0) AS transfers_out_event,
                COALESCE(p.selected_by_percent, 0.0) AS selected_by_percent_enrich,
                COALESCE(ps.sentiment, 0.0) AS press_sentiment
            FROM players p
            LEFT JOIN player_setpiece_roles sp
                ON sp.player_id = p.id
                AND sp.season = (
                    SELECT season FROM player_gw_stats
                    WHERE player_id = p.id
                    ORDER BY gameweek DESC LIMIT 1
                )
            LEFT JOIN (
                SELECT player_id, AVG(sentiment) AS sentiment
                FROM player_press_signals
                WHERE scraped_date >= date('now', '-7 days')
                GROUP BY player_id
            ) ps ON ps.player_id = p.id
        """)
        df = pd.read_sql(query, db.bind)
        df["price_momentum"] = (
            (df["transfers_in_event"] - df["transfers_out_event"])
            / (df["selected_by_percent_enrich"].clip(lower=0.1) * 1000)
        ).clip(-1.0, 1.0)
        return df
    finally:
        db.close()


def add_enrichment_features(df: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "player_id", "is_penalty_taker", "penalty_xg_per_game",
        "is_set_piece_taker", "key_passes_per_game",
        "injury_severity", "press_sentiment", "price_momentum",
    ]
    merged = df.merge(enrichment[cols], on="player_id", how="left")
    for col in cols[1:]:
        merged[col] = merged[col].fillna(0.0)
    return merged


def load_midweek_flags(target_gw: int) -> dict[int, float]:
    from data.ingestors.midweek import get_team_midweek_flags
    flags = get_team_midweek_flags(target_gw)
    return {team_id: float(v) for team_id, v in flags.items()}


FDR_FEATURE_COLS = [
    "is_home",
    "opp_defence_strength",
    "opp_attack_strength",
    "own_attack_strength",
    "own_defence_strength",
    "own_overall_strength",
    "attack_vs_defence",
    "defence_vs_attack",
]

ENRICHMENT_FEATURE_COLS = [
    "is_penalty_taker",
    "penalty_xg_per_game",
    "is_set_piece_taker",
    "key_passes_per_game",
    "injury_severity",
    "press_sentiment",
    "price_momentum",
]
