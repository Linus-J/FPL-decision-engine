import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from config.strategy import OPTIMISER
from data.db import get_session

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/minutes_model.pkl")

# Threshold: 60+ minutes = "started and played full game" for FPL point purposes
FULL_GAME_MINUTES = 60


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
                s.selected,
                s.transfers_in,
                s.transfers_out,
                s.value,
                p.position,
                p.status,
                p.chance_of_playing_next_round,
                p.now_cost,
                p.selected_by_percent,
                p.form,
                p.ict_index,
                p.team_id
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            WHERE s.minutes IS NOT NULL
        """)
        df = pd.read_sql(query, db.bind)
        return df
    finally:
        db.close()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()

    # Rolling form features — last N GWs within the same player+season
    for window in [3, 5]:
        grp = df.groupby(["player_id", "season"])
        df[f"avg_minutes_{window}gw"] = (
            grp["minutes"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"avg_points_{window}gw"] = (
            grp["total_points"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"starts_rate_{window}gw"] = (
            grp["minutes"].transform(
                lambda x: (x.shift(1) >= FULL_GAME_MINUTES).rolling(window, min_periods=1).mean()
            )
        )

    # Season-to-date average minutes (proxy for rotation risk)
    df["season_avg_minutes"] = (
        df.groupby(["player_id", "season"])["minutes"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Position one-hot
    df = pd.get_dummies(df, columns=["position"], prefix="pos", dtype=float)
    for pos in ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]:
        if pos not in df.columns:
            df[pos] = 0.0

    # Availability signal from current status
    df["is_available"] = (df["status"] == "a").astype(float)
    df["cop_next"] = df["chance_of_playing_next_round"].fillna(100) / 100.0

    df = df.dropna(subset=["avg_minutes_5gw", "season_avg_minutes"])
    return df


FEATURE_COLS = [
    "avg_minutes_3gw",
    "avg_minutes_5gw",
    "avg_points_3gw",
    "avg_points_5gw",
    "starts_rate_3gw",
    "starts_rate_5gw",
    "season_avg_minutes",
    "now_cost",
    "selected_by_percent",
    "form",
    "ict_index",
    "is_available",
    "cop_next",
    "pos_GKP",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
]


def train(save: bool = True, df_override: pd.DataFrame | None = None, fast: bool = False) -> Pipeline:
    df = df_override if df_override is not None else _load_training_data()
    df = _build_features(df)

    target_started = (df["minutes"] >= FULL_GAME_MINUTES).astype(int)

    X = df[FEATURE_COLS].astype(float)
    y = target_started

    n_estimators = 50 if fast else 200
    base = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    min_cv_samples = 100
    if len(X) >= min_cv_samples:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic")),
        ])
    else:
        from sklearn.preprocessing import StandardScaler as SS
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        base.fit(X_scaled, y)
        calibrated = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
        calibrated.fit(X_scaled, y)
        pipeline = Pipeline([("scaler", scaler), ("clf", calibrated)])

    if not fast:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_brier_score")
        logger.info(
            "Minutes model CV Brier score: %.4f ± %.4f",
            -cv_scores.mean(), cv_scores.std(),
        )

    pipeline.fit(X, y)

    train_preds = pipeline.predict_proba(X)[:, 1]
    if not fast:
        logger.info("Train log-loss: %.4f", log_loss(y, train_preds))
        logger.info("Train Brier:    %.4f", brier_score_loss(y, train_preds))
        logger.info("Trained on %d samples across %d players", len(df), df["player_id"].nunique())

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved minutes model → %s", MODEL_PATH)

    return pipeline


def load() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Minutes model not found at {MODEL_PATH}. Run scripts/train_models.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_start_probabilities(
    player_recent_stats: pd.DataFrame,
    model: Pipeline | None = None,
) -> pd.Series:
    if model is None:
        model = load()

    df = _build_features(player_recent_stats)
    X = df[FEATURE_COLS].astype(float)
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=df.index, name="start_prob")
