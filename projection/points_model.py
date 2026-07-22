import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from config.strategy import SCORING
from data.db import get_session
from projection.features import (
    ENRICHMENT_FEATURE_COLS,
    FDR_FEATURE_COLS,
    ODDS_FEATURE_COLS,
    add_enrichment_features,
    add_fdr_features,
    add_odds_features,
    load_fixture_difficulty,
    load_fixture_odds,
    load_player_enrichment,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/points_model.pkl")


def _load_training_data() -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                s.minutes,
                s.total_points,
                s.goals_scored,
                s.assists,
                s.clean_sheets,
                s.goals_conceded,
                s.saves,
                s.yellow_cards,
                s.red_cards,
                s.bonus,
                s.bps,
                s.value AS now_cost,
                p.position,
                p.team_id,
                COALESCE(ps.ict_index, 0) AS ict_index,
                COALESCE(ps.influence, 0) AS influence,
                COALESCE(ps.creativity, 0) AS creativity,
                COALESCE(ps.threat, 0) AS threat,
                COALESCE(ps.form, 0) AS form,
                COALESCE(ps.selected_by_percent, 0) AS selected_by_percent,
                COALESCE(x.xg, 0) AS xg,
                COALESCE(x.xa, 0) AS xa,
                COALESCE(x.xgi, 0) AS xgi,
                COALESCE(x.npxg, 0) AS npxg,
                COALESCE(x.shots, 0) AS shots,
                COALESCE(x.key_passes, 0) AS key_passes
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            JOIN gameweeks g ON g.id = s.gameweek AND g.season = s.season
            LEFT JOIN player_state_snapshots ps ON ps.id = (
                SELECT ps2.id FROM player_state_snapshots ps2
                WHERE ps2.player_id = s.player_id
                    AND ps2.season = s.season
                    AND ps2.snapshot_ts < g.deadline_time
                ORDER BY ps2.snapshot_ts DESC LIMIT 1
            )
            LEFT JOIN player_xg_stats x
                ON x.player_id = s.player_id
                AND x.gameweek = s.gameweek
                AND x.season = s.season
            WHERE s.minutes > 0
        """)
        df = pd.read_sql(query, db.bind)
        return df
    finally:
        db.close()


def _score_from_stats(row: pd.Series, position: str) -> float:
    pts = 0.0
    pts += SCORING.points_full_appearance if row["minutes"] >= 60 else SCORING.points_sub_appearance

    goal_pts = {
        "GKP": SCORING.points_goal_gk,
        "DEF": SCORING.points_goal_def,
        "MID": SCORING.points_goal_mid,
        "FWD": SCORING.points_goal_fwd,
    }.get(position, SCORING.points_goal_fwd)
    pts += row["goals_scored"] * goal_pts
    pts += row["assists"] * SCORING.points_assist

    cs_pts = {
        "GKP": SCORING.points_cs_gk,
        "DEF": SCORING.points_cs_def,
        "MID": SCORING.points_cs_mid,
        "FWD": SCORING.points_cs_fwd,
    }.get(position, 0)
    if row["minutes"] >= 60:
        pts += row["clean_sheets"] * cs_pts

    if position in ("GKP", "DEF") and row["goals_conceded"] >= SCORING.goals_conceded_per_penalty:
        pts += (row["goals_conceded"] // SCORING.goals_conceded_per_penalty) * SCORING.points_goals_conceded_penalty

    if position == "GKP" and row["saves"] >= SCORING.saves_per_bonus_point:
        pts += (row["saves"] // SCORING.saves_per_bonus_point) * SCORING.points_save_bonus

    pts += row["yellow_cards"] * SCORING.points_yellow_card
    pts += row["red_cards"] * SCORING.points_red_card
    pts += row.get("bonus", 0) * 1

    return float(pts)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()

    grp = df.groupby(["player_id", "season"])

    for window in [3, 5]:
        df[f"avg_xg_{window}gw"] = (
            grp["xg"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_xa_{window}gw"] = (
            grp["xa"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_npxg_{window}gw"] = (
            grp["npxg"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_shots_{window}gw"] = (
            grp["shots"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_pts_{window}gw"] = (
            grp["total_points"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_goals_{window}gw"] = (
            grp["goals_scored"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_assists_{window}gw"] = (
            grp["assists"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_cs_{window}gw"] = (
            grp["clean_sheets"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_saves_{window}gw"] = (
            grp["saves"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_bonus_{window}gw"] = (
            grp["bonus"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_gc_{window}gw"] = (
            grp["goals_conceded"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    global_grp = df.groupby("player_id")
    df["avg_pts_5gw_global"] = global_grp["total_points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    career_avg = global_grp["total_points"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["form_decay_ratio"] = (df["avg_pts_3gw"] / (career_avg + 0.1)).clip(-3.0, 3.0)

    df = pd.get_dummies(df, columns=["position"], prefix="pos", dtype=float)
    for pos in ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]:
        if pos not in df.columns:
            df[pos] = 0.0

    df = df.dropna(subset=["avg_xg_5gw", "avg_pts_5gw"])

    fdr = load_fixture_difficulty()
    df = add_fdr_features(df, fdr)

    enrichment = load_player_enrichment()
    df = add_enrichment_features(df, enrichment)

    odds = load_fixture_odds()
    df = add_odds_features(df, odds)

    return df


FEATURE_COLS = [
    "avg_xg_3gw", "avg_xg_5gw",
    "avg_xa_3gw", "avg_xa_5gw",
    "avg_npxg_3gw", "avg_npxg_5gw",
    "avg_shots_3gw", "avg_shots_5gw",
    "avg_pts_3gw", "avg_pts_5gw",
    "avg_pts_5gw_global", "form_decay_ratio",
    "avg_goals_3gw", "avg_goals_5gw",
    "avg_assists_3gw", "avg_assists_5gw",
    "avg_cs_3gw", "avg_cs_5gw",
    "avg_saves_3gw", "avg_saves_5gw",
    "avg_bonus_3gw", "avg_bonus_5gw",
    "avg_gc_3gw", "avg_gc_5gw",
    "ict_index", "influence", "creativity", "threat",
    "form", "now_cost", "selected_by_percent",
    "pos_GKP", "pos_DEF", "pos_MID", "pos_FWD",
    *FDR_FEATURE_COLS,
    *ODDS_FEATURE_COLS,
    *ENRICHMENT_FEATURE_COLS,
]


def train(save: bool = True, df_override: pd.DataFrame | None = None, fast: bool = False) -> Pipeline:
    df = df_override if df_override is not None else _load_training_data()
    df = _build_features(df)

    y = df["total_points"].astype(float)
    X = df[FEATURE_COLS].astype(float)

    n_estimators = 80 if fast else 300
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            loss="huber",
            random_state=42,
        )),
    ])

    if not fast:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
        logger.info(
            "Points model CV MAE: %.4f ± %.4f",
            -cv_scores.mean(), cv_scores.std(),
        )

    model.fit(X, y)

    if not fast:
        train_preds = model.predict(X)
        logger.info("Train MAE: %.4f", mean_absolute_error(y, train_preds))
        logger.info(
            "Trained on %d samples, position split: %s",
            len(df),
            df[[c for c in df.columns if c.startswith("pos_")]].sum().to_dict(),
        )

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved points model → %s", MODEL_PATH)

    return model


def load() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Points model not found at {MODEL_PATH}. Run scripts/train_models.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_points(
    player_recent_stats: pd.DataFrame,
    model: Pipeline | None = None,
) -> pd.Series:
    if model is None:
        model = load()

    df = _build_features(player_recent_stats)
    X = df[FEATURE_COLS].astype(float)
    preds = model.predict(X)
    return pd.Series(np.clip(preds, 0, None), index=df.index, name="xpts")


def predict_batch(
    all_stats: pd.DataFrame,
    model: Pipeline | None = None,
) -> dict[int, float]:
    if model is None:
        model = load()

    df = _build_features(all_stats)
    X = df[FEATURE_COLS].astype(float)
    preds = np.clip(model.predict(X), 0, None)
    df = df.copy()
    df["_xpts"] = preds
    return df.groupby("player_id")["_xpts"].last().to_dict()
