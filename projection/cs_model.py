import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from data.db import get_session
from projection.features import FDR_FEATURE_COLS, add_fdr_features, load_fixture_difficulty

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/cs_model.pkl")


def _load_training_data() -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                s.clean_sheets,
                s.goals_conceded,
                s.minutes,
                p.position,
                p.team_id
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            WHERE s.minutes > 0
              AND p.position IN ('GKP', 'DEF')
        """)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["team_id", "season", "gameweek"]).copy()

    team_grp = df.groupby(["team_id", "season"])

    for window in [3, 5]:
        df[f"team_cs_rate_{window}gw"] = (
            team_grp["clean_sheets"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        )
        df[f"team_gc_rate_{window}gw"] = (
            team_grp["goals_conceded"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        )

    df = pd.get_dummies(df, columns=["position"], prefix="pos", dtype=float)
    for pos in ["pos_GKP", "pos_DEF"]:
        if pos not in df.columns:
            df[pos] = 0.0

    df = df.dropna(subset=["team_cs_rate_5gw"])

    fdr = load_fixture_difficulty()
    df = add_fdr_features(df, fdr)

    return df


FEATURE_COLS = [
    "team_cs_rate_3gw",
    "team_cs_rate_5gw",
    "team_gc_rate_3gw",
    "team_gc_rate_5gw",
    "pos_GKP",
    "pos_DEF",
    "is_home",
    "opp_attack_strength",
    "own_defence_strength",
    "defence_vs_attack",
]


def train(save: bool = True, df_override: pd.DataFrame | None = None) -> Pipeline:
    df = df_override if df_override is not None else _load_training_data()
    df = _build_features(df)

    y = (df["clean_sheets"] > 0).astype(int)
    X = df[FEATURE_COLS].astype(float)

    base = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic")),
    ])

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_brier_score")
    logger.info(
        "CS model CV Brier score: %.4f ± %.4f",
        -cv_scores.mean(), cv_scores.std(),
    )

    pipeline.fit(X, y)

    train_preds = pipeline.predict_proba(X)[:, 1]
    logger.info("CS model train Brier: %.4f", brier_score_loss(y, train_preds))
    logger.info(
        "CS model trained on %d samples (%.1f%% CS rate)",
        len(df), y.mean() * 100,
    )

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved CS model → %s", MODEL_PATH)

    return pipeline


def load() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CS model not found at {MODEL_PATH}. Run scripts/train_models.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_cs_probability(
    team_id: int,
    gameweek: int,
    season: str,
    position: str,
    model: Pipeline | None = None,
) -> float:
    if position not in ("GKP", "DEF"):
        return 0.0

    if model is None:
        model = load()

    db = get_session()
    try:
        history_query = text("""
            SELECT s.gameweek, s.season, s.clean_sheets, s.goals_conceded, s.minutes,
                   p.position, p.team_id
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            WHERE p.team_id = :team_id
              AND s.season = :season
              AND s.gameweek < :gw
              AND p.position IN ('GKP', 'DEF')
              AND s.minutes > 0
            ORDER BY s.gameweek DESC
            LIMIT 50
        """)
        history = pd.read_sql(
            history_query, db.bind,
            params={"team_id": team_id, "season": season, "gw": gameweek},
        )
    finally:
        db.close()

    if history.empty:
        return 0.27

    cs_rate_3 = float((history["clean_sheets"] > 0).head(3).mean())
    cs_rate_5 = float((history["clean_sheets"] > 0).head(5).mean())
    gc_rate_3 = float(history["goals_conceded"].head(3).mean())
    gc_rate_5 = float(history["goals_conceded"].head(5).mean())

    fdr = load_fixture_difficulty(season=season)
    fdr_row = fdr[
        (fdr["player_id"].isin(
            pd.read_sql(
                text("SELECT id FROM players WHERE team_id=:t AND position=:pos LIMIT 1"),
                get_session().bind,
                params={"t": team_id, "pos": position},
            )["id"].tolist()
        )) &
        (fdr["gameweek"] == gameweek)
    ]

    is_home = float(fdr_row["is_home"].iloc[0]) if not fdr_row.empty else 0.5
    opp_attack = float(fdr_row["opp_attack_strength"].iloc[0]) if not fdr_row.empty else 1200.0
    own_defence = float(fdr_row["own_defence_strength"].iloc[0]) if not fdr_row.empty else 1200.0
    def_vs_att = own_defence / max(opp_attack, 1)

    X = pd.DataFrame([{
        "team_cs_rate_3gw": cs_rate_3,
        "team_cs_rate_5gw": cs_rate_5,
        "team_gc_rate_3gw": gc_rate_3,
        "team_gc_rate_5gw": gc_rate_5,
        "pos_GKP": 1.0 if position == "GKP" else 0.0,
        "pos_DEF": 1.0 if position == "DEF" else 0.0,
        "is_home": is_home,
        "opp_attack_strength": opp_attack,
        "own_defence_strength": own_defence,
        "defence_vs_attack": def_vs_att,
    }])

    prob = model.predict_proba(X)[0, 1]
    return float(np.clip(prob, 0.0, 1.0))
