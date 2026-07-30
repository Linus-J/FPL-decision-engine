import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
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


def _trailing_dnp_streak(minutes: pd.Series) -> pd.Series:
    """Consecutive-zero-minutes streak ending at each row (raw — the caller
    shifts it by 1 before use as a leakage-free predictive feature)."""
    is_zero = minutes == 0
    run_id = (is_zero != is_zero.shift(fill_value=False)).cumsum()
    streak = is_zero.groupby(run_id).cumsum()
    return streak.where(is_zero, 0).astype(int)


def _red_card_flag(red_cards: pd.Series) -> pd.Series:
    """1 where a red card was shown that gameweek, else 0 (raw — the caller
    shifts it by 1, same two-step convention as ``_trailing_dnp_streak``, so
    the flag lands on the NEXT gameweek — the one it's actually suspended for)."""
    return red_cards.fillna(0).gt(0).astype(int)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()

    # Real bug found 2026-07-28 (data-completeness audit, user-driven trace
    # review): `status`/`chance_of_playing` are constant ('a') throughout the
    # backfilled history (merged_gw.csv structurally has no such columns —
    # see backfill_history.py::compute_snapshot_rows), so
    # apply_availability_override below NEVER fires during backtesting. That
    # let genuinely-injured players (confirmed live: real minutes=0 for
    # several straight gameweeks) get captained/started with an undiminished
    # or even RISING projection, because the only signal left was diluted
    # rolling-average minutes. `dnp_streak` is a real, leakage-free signal
    # from ALREADY-PLAYED history (no new data source needed) — how many
    # consecutive completed gameweeks a player had zero minutes — and drives
    # a separate deterministic override (apply_recent_absence_override)
    # exactly like the status override does, but from real data.
    streak_grp = df.groupby(["player_id", "season"])
    df["dnp_streak"] = streak_grp["minutes"].transform(_trailing_dnp_streak)
    df["dnp_streak"] = (
        df.groupby(["player_id", "season"])["dnp_streak"]
        .transform(lambda x: x.shift(1))
        .fillna(0)
        .astype(int)
    )

    # Real bug found 2026-07-30 (user's own review: a player sent off with a
    # straight red at GW6 ["-3 points"] got transferred IN for GW7, straight
    # into a suspension — 0 minutes). dnp_streak can't catch this: the player
    # played most of GW6 before being sent off, so that gameweek isn't a
    # blank and the streak stays 0. A red card almost always draws at least
    # a 1-match ban under the FA's disciplinary process — a hard, rule-based
    # signal directly observable in already-played history (red_cards),
    # independent of minutes played that game.
    red_card_grp = df.groupby(["player_id", "season"])
    df["recent_red_card"] = red_card_grp["red_cards"].transform(_red_card_flag)
    df["recent_red_card"] = (
        df.groupby(["player_id", "season"])["recent_red_card"]
        .transform(lambda x: x.shift(1))
        .fillna(0)
        .astype(int)
    )

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

    # Availability (status / chance-of-playing) is NOT a learned feature (M1):
    # it is ~constant in backfilled history so the model can't learn it, and it
    # is applied as a deterministic override on the predicted bands instead
    # (apply_availability_override). The raw `status` / `chance_of_playing_next_round`
    # columns are left on the frame for that override to read.

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
    # carry the signal). M1 (P1): availability (is_available/cop_next) removed
    # as a learned feature — applied as a deterministic override instead.
    "pos_GKP",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
    *FDR_FEATURE_COLS,
    *ODDS_FEATURE_COLS,
    *ENRICHMENT_FEATURE_COLS,
]

assert_rate_only(FEATURE_COLS)

# --- 3-way minutes bands (P1): P({0, 1–59, 60+}) --------------------------
BAND_DNP, BAND_CAMEO, BAND_START = 0, 1, 2   # did-not-play / 1–59 / 60+


def minutes_band(minutes: float | None) -> int:
    if minutes is None or minutes <= 0:
        return BAND_DNP
    if minutes < FULL_GAME_MINUTES:
        return BAND_CAMEO
    return BAND_START


def _bands_from_proba(proba_row, classes) -> tuple[float, float, float]:
    """Map a classifier's proba row onto the fixed (P0, P1, P2) band vector,
    tolerating a class being absent from the training slice (→ 0)."""
    out = [0.0, 0.0, 0.0]
    for cls, p in zip(classes, proba_row, strict=False):
        out[int(cls)] = float(p)
    return out[0], out[1], out[2]


def apply_availability_override(
    p0: float, p1: float, p2: float, status: str | None, cop: float | None
) -> tuple[float, float, float]:
    """Deterministic availability adjustment on the predicted bands (M1).

    Availability is not learned (it is defaulted/constant in backfilled history),
    so it is applied here, not as a feature:
    - status i/u/s (injured/unavailable/suspended) → certain DNP (1, 0, 0);
    - status 'd' (doubtful) → scale the playing mass by chance-of-playing (cop,
      0–1), moving the rest to DNP;
    - otherwise ('a'/None) → unchanged.
    """
    if status in ("i", "u", "s"):
        return (1.0, 0.0, 0.0)
    if status == "d":
        c = 1.0 if cop is None else max(0.0, min(1.0, cop))
        return (1.0 - c * (p1 + p2), c * p1, c * p2)
    return (p0, p1, p2)


# Retention of the ML-predicted playing mass (P1+P2) given a confirmed
# consecutive-zero-minutes streak from real, already-played history (see
# _trailing_dnp_streak). 1 confirmed blank -> real uncertainty (could be
# rotation, a minor knock, or the start of a longer injury) so only half the
# playing mass survives; 2+ -> heavily discounted, since a player out for
# multiple straight matches is statistically unlikely to be a safe near-term
# pick. Untuned starting values pending backtesting, same convention as
# other heuristic constants introduced this session.
_DNP_STREAK_RETENTION = {0: 1.0, 1: 0.5}
_DNP_STREAK_RETENTION_2PLUS = 0.15


def apply_recent_absence_override(
    p0: float, p1: float, p2: float, dnp_streak: int
) -> tuple[float, float, float]:
    """Deterministic band adjustment from a REAL, leakage-free signal (2026-07-28):
    consecutive zero-minutes gameweeks in already-played history. Complements
    apply_availability_override, which is a no-op throughout backtesting
    because status/chance-of-playing are constant in the backfilled data —
    this uses real minutes instead, so it actually has something to act on.
    A player's FIRST blank gameweek can't be predicted this way (no prior
    evidence yet, dnp_streak=0) — this only catches an absence that's
    already confirmed by at least one completed gameweek."""
    retention = _DNP_STREAK_RETENTION.get(dnp_streak, _DNP_STREAK_RETENTION_2PLUS)
    if retention >= 1.0:
        return (p0, p1, p2)
    new_p1 = p1 * retention
    new_p2 = p2 * retention
    return (1.0 - new_p1 - new_p2, new_p1, new_p2)


# Near-certain absence next gameweek — a straight red almost always draws at
# least a 1-match ban; some (violent conduct etc.) draw more, but severity
# isn't inferable from the box score alone, so this only guards the
# guaranteed-minimum case. Untuned starting value pending backtesting, same
# convention as other heuristic constants this session.
_RED_CARD_SUSPENSION_RETENTION = 0.05


def apply_red_card_suspension_override(
    p0: float, p1: float, p2: float, recent_red_card: bool
) -> tuple[float, float, float]:
    """Deterministic band adjustment for a confirmed red card in the player's
    immediately preceding gameweek (see ``recent_red_card`` in
    ``_build_features``). Complements ``apply_recent_absence_override``,
    which can't catch this case — the sent-off player likely played most of
    that match before the red, so it isn't a zero-minutes gameweek and
    ``dnp_streak`` stays 0."""
    if not recent_red_card:
        return (p0, p1, p2)
    new_p1 = p1 * _RED_CARD_SUSPENSION_RETENTION
    new_p2 = p2 * _RED_CARD_SUSPENSION_RETENTION
    return (1.0 - new_p1 - new_p2, new_p1, new_p2)


def expected_appearance_points(p1: float, p2: float, sub: int = 1, full: int = 2) -> float:
    """Expected appearance points from the band probabilities (1 for a cameo,
    2 for 60+) — the minutes model's contribution to xPts (P10)."""
    return p1 * sub + p2 * full


def _fit_calibrated(base, X, y):
    """Multiclass fit + probability calibration, robust to small/sparse slices
    (the backtest trains per-GW). Falls back from CV-isotonic to prefit-sigmoid
    to raw when there aren't enough per-class samples to calibrate."""
    from collections import Counter

    counts = Counter(y)
    if len(counts) < 2:  # degenerate slice — nothing to calibrate
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", base)])
        pipe.fit(X, y)
        return pipe
    min_class = min(counts.values())
    if len(X) >= 100 and min_class >= 9:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base, cv=3, method="isotonic")),
        ])
        pipe.fit(X, y)
        return pipe
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    base.fit(x_scaled, y)
    cal = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
    cal.fit(x_scaled, y)
    return Pipeline([("scaler", scaler), ("clf", cal)])


def train(save: bool = True, df_override: pd.DataFrame | None = None, fast: bool = False) -> Pipeline:
    df = df_override if df_override is not None else _load_training_data()
    df = _build_features(df)

    y = df["minutes"].apply(minutes_band)
    X = df[FEATURE_COLS].astype(float)

    n_estimators = 50 if fast else 200
    base = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    pipeline = _fit_calibrated(base, X, y)

    if not fast:
        proba = pipeline.predict_proba(X)
        band_dist = y.value_counts(normalize=True).round(3).to_dict()
        logger.info(
            "Minutes model (3-way) train log-loss: %.4f",
            log_loss(y, proba, labels=[BAND_DNP, BAND_CAMEO, BAND_START]),
        )
        logger.info(
            "Trained on %d samples (band dist %s) across %d players",
            len(df), band_dist, df["player_id"].nunique(),
        )

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


def _bands_frame(all_stats: pd.DataFrame, model: Pipeline | None) -> pd.DataFrame:
    """Feature-build + 3-way predict + availability override, per input row."""
    if model is None:
        model = load()
    df = _build_features(all_stats).copy()
    proba = model.predict_proba(df[FEATURE_COLS].astype(float))
    classes = list(model.classes_)

    statuses = (df["status"] if "status" in df.columns
                else pd.Series([None] * len(df), index=df.index))
    cops = (df["chance_of_playing_next_round"] if "chance_of_playing_next_round" in df.columns
            else pd.Series([None] * len(df), index=df.index))
    streaks = (df["dnp_streak"] if "dnp_streak" in df.columns
               else pd.Series([0] * len(df), index=df.index))
    red_cards = (df["recent_red_card"] if "recent_red_card" in df.columns
                 else pd.Series([0] * len(df), index=df.index))

    p0s, p1s, p2s = [], [], []
    for row, status, cp, streak, red_card in zip(
        proba, statuses, cops, streaks, red_cards, strict=False
    ):
        b0, b1, b2 = _bands_from_proba(row, classes)
        # Real-history overrides first (2026-07-28/07-30) -- always have
        # something to act on, unlike the status override below, which is a
        # no-op throughout backtesting (see apply_recent_absence_override).
        b0, b1, b2 = apply_recent_absence_override(b0, b1, b2, int(streak))
        b0, b1, b2 = apply_red_card_suspension_override(b0, b1, b2, bool(red_card))
        cop = None if pd.isna(cp) else float(cp) / 100.0
        b0, b1, b2 = apply_availability_override(b0, b1, b2, status, cop)
        p0s.append(b0)
        p1s.append(b1)
        p2s.append(b2)
    df["_p0"], df["_p1"], df["_p2"] = p0s, p1s, p2s
    return df


def predict_minutes_bands(
    all_stats: pd.DataFrame,
    model: Pipeline | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Per player → (P(0 min), P(1–59), P(60+)) after the availability override,
    taking each player's latest row. The full distribution the components need
    (P5 conditions clean sheets on P(60+); appearance points on P1/P2)."""
    df = _bands_frame(all_stats, model)
    last = df.groupby("player_id")[["_p0", "_p1", "_p2"]].last()
    return {int(pid): (r["_p0"], r["_p1"], r["_p2"]) for pid, r in last.iterrows()}


def predict_batch(
    all_stats: pd.DataFrame,
    model: Pipeline | None = None,
) -> dict[int, float]:
    """Back-compatible scalar 'start probability' = P(60+), from the bands."""
    return {pid: bands[2] for pid, bands in predict_minutes_bands(all_stats, model).items()}


def predict_start_probabilities(
    player_recent_stats: pd.DataFrame,
    model: Pipeline | None = None,
) -> pd.Series:
    df = _bands_frame(player_recent_stats, model)
    return pd.Series(df["_p2"].values, index=df.index, name="start_prob")
