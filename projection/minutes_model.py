import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from data.db import get_session
from projection.features import (
    ENRICHMENT_FEATURE_COLS,
    FDR_FEATURE_COLS,
    ODDS_FEATURE_COLS,
    add_enrichment_features,
    add_fdr_features,
    add_odds_features,
    assert_rate_only,
    load_fixture_difficulty,
    load_fixture_odds,
    load_player_enrichment,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/minutes_model.pkl")

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
                s.value AS now_cost,
                p.position,
                COALESCE(ps.status, 'a') AS status,
                ps.chance_of_playing_next_round AS chance_of_playing_next_round,
                COALESCE(ps.selected_by_percent, 0) AS selected_by_percent,
                COALESCE(ps.form, 0) AS form,
                COALESCE(ps.ict_index, 0) AS ict_index,
                p.team_id
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
            WHERE s.minutes IS NOT NULL
        """)
        df = pd.read_sql(query, db.bind)
        return df
    finally:
        db.close()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()

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

    df["season_avg_minutes"] = (
        df.groupby(["player_id", "season"])["minutes"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    global_grp = df.groupby("player_id")
    df["avg_minutes_5gw_global"] = global_grp["minutes"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    career_avg_min = global_grp["minutes"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["minutes_decay_ratio"] = (df["avg_minutes_3gw"] / (career_avg_min + 1)).clip(0.0, 2.0)

    df = pd.get_dummies(df, columns=["position"], prefix="pos", dtype=float)
    for pos in ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]:
        if pos not in df.columns:
            df[pos] = 0.0

    df["is_available"] = (df["status"] == "a").astype(float)
    df["cop_next"] = df["chance_of_playing_next_round"].fillna(100) / 100.0

    df = df.dropna(subset=["avg_minutes_5gw", "season_avg_minutes"])

    fdr = load_fixture_difficulty()
    df = add_fdr_features(df, fdr)

    enrichment = load_player_enrichment()
    df = add_enrichment_features(df, enrichment)

    odds = load_fixture_odds()
    df = add_odds_features(df, odds)

    return df


FEATURE_COLS = [
    "avg_minutes_3gw",
    "avg_minutes_5gw",
    "avg_minutes_5gw_global",
    "minutes_decay_ratio",
    "avg_points_3gw",
    "avg_points_5gw",
    "starts_rate_3gw",
    "starts_rate_5gw",
    "season_avg_minutes",
    "now_cost",
    "selected_by_percent",
    # D4 (P2): cumulative ict_index + form proxy retired (rolling avg_* rates
    # carry the signal). is_available/cop_next stay for now — availability
    # becomes a deterministic override in P1, not a learned feature.
    "is_available",
    "cop_next",
    "pos_GKP",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
    *FDR_FEATURE_COLS,
    *ODDS_FEATURE_COLS,
    *ENRICHMENT_FEATURE_COLS,
]

assert_rate_only(FEATURE_COLS)


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


def predict_batch(
    all_stats: pd.DataFrame,
    model: Pipeline | None = None,
) -> dict[int, float]:
    if model is None:
        model = load()

    df = _build_features(all_stats)
    X = df[FEATURE_COLS].astype(float)
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["_prob"] = probs
    return df.groupby("player_id")["_prob"].last().to_dict()
