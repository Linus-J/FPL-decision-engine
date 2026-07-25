"""assemble.py — P10 Monte-Carlo assembly (D1/D5).

Sums the 8 components' samples via the P-COV joint scheme into a per-player
xPts distribution, replacing the monolithic ``points_model.py`` regressor.

Per fixture, per scenario: draw the shared team-level latents ONCE (goals-for
for each side, via ``covariance.sample_team_goals`` anchored on the odds-implied
λ from ``team_goals.py``), draw each player's minutes band, then condition
every other component on those two shared draws — goals/assists split the
drawn team total via ``covariance.split_multinomial`` (weighted by attacking
weight × *that scenario's* realised minutes, not the static average, so a
player drawn as DNP this scenario can't also be drawn a goal this scenario);
clean-sheet/concede/saves all read the same drawn "goals conceded" integer
(the source of the P-COV teammate covariance). DefCon and cards are sampled
per-player (no fixture-level shared latent in the current design — the gate
only requires the CS covariance, not defensive-action covariance). Bonus is
sampled per fixture-scenario, once, across BOTH teams' reduced-BPS event rows
(bonus is a fixture-relative top-3-BPS rank, not a marginal quantity).

Impure at the edges only: the `_load_*` functions hit the DB; `sample_fixture`
and `compute_defcon_field_shares`'s inner math are pure given their inputs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

from config.strategy import DEFCON, SCORING
from data.db import get_session
from projection import bonus as bonus_mod
from projection import clean_sheets, defcon, goals, saves
from projection.assists import ASSIST_FRACTION, expected_assist_points
from projection.covariance import sample_team_goals, split_multinomial
from projection.minutes_model import predict_minutes_bands
from projection.team_goals import team_goals_from_odds

logger = logging.getLogger(__name__)

DEFAULT_N_SCENARIOS = 150

# Average fraction of 90 minutes a "1-59" cameo appearance represents — used
# only to scale a scenario's attacking-output share for a sub, not to award
# FPL points (those are exact from the band, not this approximation).
CAMEO_MINUTES_FRAC = 0.33

_GK_POSITIONS = {"GK", "GKP"}
_DEF_CBIT_FIELDS = ("clearances", "blocks", "interceptions", "tackles")
_MID_FWD_CBIRT_FIELDS = ("clearances", "blocks", "interceptions", "tackles", "recoveries")


# ---------------------------------------------------------------------------
# Pure core: one fixture, N scenarios, both teams jointly.
# ---------------------------------------------------------------------------

def _minutes_scale(band: int) -> float:
    """This-scenario attacking-output scale from a drawn minutes band — 0 for
    DNP, a partial share for a cameo, full for 60+. Used only to gate/scale
    THIS scenario's goal/assist/defcon/card draws on THIS scenario's minutes,
    so a player drawn DNP can't also be drawn a goal in the same scenario."""
    return {0: 0.0, 1: CAMEO_MINUTES_FRAC, 2: 1.0}[band]


def _draw_band(rng: np.random.Generator, p0: float, p1: float, p2: float) -> int:
    probs = np.array([max(0.0, p0), max(0.0, p1), max(0.0, p2)])
    total = probs.sum()
    probs = probs / total if total > 0 else np.array([1.0, 0.0, 0.0])
    return int(rng.choice(3, p=probs))


def _defcon_split(
    rng: np.random.Generator, actions: int, position: str, shares: Mapping[str, Mapping[str, float]]
) -> dict[str, int]:
    if actions <= 0:
        fields = _DEF_CBIT_FIELDS if position == "DEF" else _MID_FWD_CBIRT_FIELDS
        return dict.fromkeys(fields, 0)
    if position == "DEF":
        fields, share = _DEF_CBIT_FIELDS, shares["DEF"]
    else:
        fields, share = _MID_FWD_CBIRT_FIELDS, shares["MID_FWD"]
    probs = np.array([share[f] for f in fields])
    counts = rng.multinomial(actions, probs / probs.sum())
    return dict(zip(fields, counts.tolist(), strict=True))


def _sample_side(
    rng: np.random.Generator,
    players: Sequence[Mapping],
    own_lambda: float,
    conceded: int,
    defcon_field_shares: Mapping[str, Mapping[str, float]],
) -> tuple[dict[int, float], dict[int, dict]]:
    """One scenario for one team's players, conditioned on the shared
    ``conceded`` draw (this side's clean-sheet/concede/saves latent). Returns
    (points_before_bonus_by_pid, reduced_bps_event_by_pid)."""
    ids = [int(p["player_id"]) for p in players]
    bands = {
        pid: _draw_band(rng, p["p0"], p["p1"], p["p2"])
        for pid, p in zip(ids, players, strict=True)
    }
    scales = {pid: _minutes_scale(b) for pid, b in bands.items()}

    own_goals = sample_team_goals(rng, own_lambda)
    own_assists = sample_team_goals(rng, own_lambda * ASSIST_FRACTION)
    goal_split = split_multinomial(
        rng, own_goals,
        [{"player_id": pid, "weight": p.get("goal_weight", 0.0), "minutes_frac": scales[pid]}
         for pid, p in zip(ids, players, strict=True)],
    )
    assist_split = split_multinomial(
        rng, own_assists,
        [{"player_id": pid, "weight": p.get("assist_weight", 0.0), "minutes_frac": scales[pid]}
         for pid, p in zip(ids, players, strict=True)],
    )

    # Saves/shots-faced anchored on the SAME drawn `conceded` (P-COV spirit —
    # a scenario where the team conceded more is one where its keeper faced
    # more shots), not redrawn independently from the odds-implied mean.
    shots_faced = conceded / saves.LEAGUE_CONVERSION
    save_rate = shots_faced * (1.0 - saves.LEAGUE_CONVERSION)

    points_by_pid: dict[int, float] = {}
    events_by_pid: dict[int, dict] = {}
    for p in players:
        pid = int(p["player_id"])
        position = p["position"]
        band = bands[pid]
        played_any = band > 0
        played_60 = band == 2
        scale = scales[pid]

        pts = SCORING.points_full_appearance if played_60 else (
            SCORING.points_sub_appearance if played_any else 0.0
        )

        goals_ct = goal_split.get(pid, 0)
        assists_ct = assist_split.get(pid, 0)
        pts += goals.expected_goal_points(goals_ct, position)
        pts += expected_assist_points(assists_ct)

        pts += clean_sheets.sample_clean_sheet_points(
            rng, 0.0, played_60, position, conceded=conceded, played_any=played_any
        )

        saves_ct = 0
        if position in _GK_POSITIONS and played_any:
            saves_ct = int(rng.poisson(max(0.0, save_rate)))
            pts += (saves_ct // SCORING.saves_per_bonus_point) * SCORING.points_save_bonus

        defcon_actions = 0
        defcon_fields: dict[str, int] = {}
        thr = defcon.defcon_threshold(position)
        if thr is not None and played_any:
            defcon_rate = p.get("defcon_rate", 0.0) * scale
            defcon_actions = int(rng.poisson(max(0.0, defcon_rate)))
            if defcon_actions >= thr:
                pts += DEFCON.points
            defcon_fields = _defcon_split(rng, defcon_actions, position, defcon_field_shares)

        key_passes_ct = 0
        yellow_ct = 0
        red_ct = 0
        if played_any:
            key_passes_ct = int(rng.poisson(max(0.0, p.get("key_pass_rate", 0.0) * scale)))
            yellow_ct = int(rng.random() < min(1.0, max(0.0, p.get("yellow_rate", 0.0))))
            red_ct = int(rng.random() < min(1.0, max(0.0, p.get("red_rate", 0.0))))
            pts += yellow_ct * SCORING.points_yellow_card + red_ct * SCORING.points_red_card

        points_by_pid[pid] = pts
        events_by_pid[pid] = {
            "position": position,
            "minutes": 60 if played_60 else (30 if played_any else 0),
            "goals": goals_ct, "assists": assists_ct,
            "clean_sheet": 1 if (played_60 and conceded == 0) else 0,
            "saves": saves_ct, "key_passes": key_passes_ct,
            "yellow_cards": yellow_ct, "red_cards": red_ct,
            **defcon_fields,
        }
    return points_by_pid, events_by_pid


def sample_fixture(
    rng: np.random.Generator,
    home_players: Sequence[Mapping],
    away_players: Sequence[Mapping],
    lam_home: float,
    lam_away: float,
    n_scenarios: int,
    defcon_field_shares: Mapping[str, Mapping[str, float]],
) -> dict[int, np.ndarray]:
    """N MC scenarios for one fixture, both teams jointly (bonus is a
    fixture-relative rank across everyone on the pitch). Returns
    ``{player_id: points_samples}`` (shape ``(n_scenarios,)``) for every
    player in ``home_players``/``away_players``."""
    all_ids = (
        [int(p["player_id"]) for p in home_players] + [int(p["player_id"]) for p in away_players]
    )
    out = {pid: np.empty(n_scenarios, dtype=float) for pid in all_ids}

    for s in range(n_scenarios):
        conceded_home = sample_team_goals(rng, lam_away)  # home concedes away's goals
        conceded_away = sample_team_goals(rng, lam_home)

        home_pts, home_events = _sample_side(
            rng, home_players, lam_home, conceded_home, defcon_field_shares
        )
        away_pts, away_events = _sample_side(
            rng, away_players, lam_away, conceded_away, defcon_field_shares
        )

        # Only players who actually featured are bonus-eligible — an unused
        # sub (minutes==0, bps==0 by construction) must never win a tied
        # bps=0 slot in a low-action match; award_bonus's tie rule would
        # otherwise hand them a share of the bonus for not playing.
        events = {
            pid: ev for pid, ev in {**home_events, **away_events}.items() if ev["minutes"] > 0
        }
        bonus_by_pid = bonus_mod.sample_fixture_bonus(events)

        for pid, pts in home_pts.items():
            out[pid][s] = pts + bonus_by_pid.get(pid, 0)
        for pid, pts in away_pts.items():
            out[pid][s] = pts + bonus_by_pid.get(pid, 0)

    return out


# ---------------------------------------------------------------------------
# DB-backed calibration + loaders (impure).
# ---------------------------------------------------------------------------

def compute_defcon_field_shares(season: str) -> dict[str, dict[str, float]]:
    """Global average share of each raw BPS defensive field within DEF's CBIT
    / MID+FWD's CBIRT pooled count, from real ``player_match_events``. Used to
    split the one sampled pooled defcon-actions count (``defcon.py`` only
    models the pooled CBIT/CBIRT rate, per its own docstring — the split is a
    P10 concern) across the individual fields the reduced-BPS event dict
    wants. A fixed global split, not per-player (P8's own reduction is already
    an approximation — this is a second, smaller one on top of it)."""
    db = get_session()
    try:
        df = pd.read_sql(
            text("""
                SELECT position, clearances, blocks, interceptions, tackles, recoveries
                FROM player_match_events WHERE season = :season AND minutes > 0
            """),
            db.bind, params={"season": season},
        )
    finally:
        db.close()

    def _shares(rows: pd.DataFrame, fields: tuple[str, ...]) -> dict[str, float]:
        totals = {f: float(rows[f].sum()) if f in rows.columns else 0.0 for f in fields}
        grand = sum(totals.values())
        if grand <= 0:
            return dict.fromkeys(fields, 1.0 / len(fields))
        return {f: v / grand for f, v in totals.items()}

    return {
        "DEF": _shares(df[df["position"] == "DEF"], _DEF_CBIT_FIELDS),
        "MID_FWD": _shares(df[df["position"].isin(["MID", "FWD"])], _MID_FWD_CBIRT_FIELDS),
    }


def load_match_odds(season: str) -> pd.DataFrame:
    """Raw de-vigged 1X2 + O/U2.5 per fixture (``historical_fixture_odds``,
    T6) — the ``team_goals_from_odds`` input. Loaded once per season (a few
    hundred rows), not per gameweek."""
    db = get_session()
    try:
        return pd.read_sql(
            text("""
                SELECT season, gameweek, home_team_id, away_team_id,
                       home_win_prob, draw_prob, away_win_prob, over25_prob
                FROM historical_fixture_odds WHERE season = :season
            """),
            db.bind, params={"season": season},
        )
    finally:
        db.close()


def load_defcon_events(season: str) -> pd.DataFrame:
    """Per (player, gw) raw defensive-action fields from ``player_match_events``
    — merged into the rolling-rate build alongside ``player_xg_stats``/
    ``player_gw_stats`` (P7's rate, computed here per its own docstring)."""
    db = get_session()
    try:
        return pd.read_sql(
            text("""
                SELECT player_id, gameweek, season,
                       clearances, blocks, interceptions, tackles, recoveries
                FROM player_match_events WHERE season = :season
            """),
            db.bind, params={"season": season},
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Orchestrator: history → rolling rates + bands → per-fixture MC → per-(player,
# gw) xPts. Drop-in replacement for points_model.predict_batch in
# backtest.py::_build_gw_projections.
# ---------------------------------------------------------------------------

_ROLLING_WINDOW = 5


def _build_rolling_features(history: pd.DataFrame, defcon_events: pd.DataFrame) -> pd.DataFrame:
    """Per player, as-of ``history`` (rows with ``gameweek < target_gw``):
    leakage-free (``shift(1)`` rolling) rates feeding the components — same
    pattern as ``points_model._build_features``, over the raw columns the MC
    components need instead of an ML model's feature set. Indexed by
    ``player_id``, one row (latest as-of) each."""
    df = history.sort_values(["player_id", "gameweek"]).copy()
    if not defcon_events.empty:
        de = defcon_events.copy()
        de["cbit"] = de["clearances"] + de["blocks"] + de["interceptions"] + de["tackles"]
        de["cbirt"] = de["cbit"] + de["recoveries"]
        df = df.merge(
            de[["player_id", "gameweek", "season", "cbit", "cbirt"]],
            on=["player_id", "gameweek", "season"], how="left",
        )
    for col in ("cbit", "cbirt"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    grp = df.groupby("player_id")
    rate_cols = {
        "xg": "goal_weight", "xa": "assist_weight", "key_passes": "key_pass_rate",
        "yellow_cards": "yellow_rate", "red_cards": "red_rate",
        "cbit": "cbit_rate", "cbirt": "cbirt_rate",
    }
    for src, out in rate_cols.items():
        if src not in df.columns:
            df[src] = 0.0
        df[out] = grp[src].transform(
            lambda x: x.shift(1).rolling(_ROLLING_WINDOW, min_periods=1).mean()
        )

    last = df.groupby("player_id").last()
    last["defcon_rate"] = np.where(
        last["position"] == "DEF", last["cbit_rate"], last["cbirt_rate"]
    )
    cols = ["position", "team_id_season", "goal_weight", "assist_weight",
            "key_pass_rate", "yellow_rate", "red_rate", "defcon_rate"]
    return last[cols].fillna(0.0)


def _player_dicts(
    pids, feat: pd.DataFrame, bands: dict[int, tuple[float, float, float]]
) -> list[dict]:
    out = []
    for pid in pids:
        pid = int(pid)
        if pid not in feat.index:
            continue
        row = feat.loc[pid]
        p0, p1, p2 = bands.get(pid, (0.5, 0.0, 0.5))
        out.append({
            "player_id": pid, "position": row["position"],
            "goal_weight": row["goal_weight"], "assist_weight": row["assist_weight"],
            "key_pass_rate": row["key_pass_rate"], "yellow_rate": row["yellow_rate"],
            "red_rate": row["red_rate"], "defcon_rate": row["defcon_rate"],
            "p0": p0, "p1": p1, "p2": p2,
        })
    return out


def assemble_gw_projections(
    history: pd.DataFrame,
    all_stats: pd.DataFrame,
    minutes_model,
    target_gw: int,
    horizon: int,
    match_odds: pd.DataFrame,
    defcon_events: pd.DataFrame,
    defcon_field_shares: Mapping[str, Mapping[str, float]],
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    seed: int = 42,
) -> pd.DataFrame:
    """P10: replaces ``points_model.predict_batch`` as ``_build_gw_projections``'s
    xPts source. ``all_stats`` (unfiltered, ALL gameweeks) supplies fixture
    context (opponent/home) for the horizon GWs — safe because that's schedule
    information known in advance, never an outcome, the same reasoning
    ``_opponent_context`` already relies on; ``history`` (strictly
    ``gameweek < target_gw``) is the only source for rolling rates. Returns one
    row per (player_id, gameweek) for gameweek in
    ``[target_gw, target_gw + horizon)`` — columns ``player_id``, ``gameweek``,
    ``xpts``, ``xpts_mean``, ``xpts_var``, ``start_probability`` — the same
    shape ``_build_gw_projections`` already emits, so it drops in without
    changing the optimiser's read side."""
    if history.empty:
        return pd.DataFrame()

    feat = _build_rolling_features(history, defcon_events)
    bands = predict_minutes_bands(history, minutes_model)

    rng = np.random.default_rng(seed)
    target_gws = list(range(target_gw, target_gw + horizon))
    fixture_rows = all_stats[all_stats["gameweek"].isin(target_gws)][
        ["player_id", "gameweek", "team_id_season", "opponent_team_id", "was_home"]
    ].dropna(subset=["opponent_team_id"]).drop_duplicates(subset=["player_id", "gameweek"])
    fixture_rows = fixture_rows.assign(was_home=fixture_rows["was_home"].fillna(False).astype(bool))

    rows = []
    for gw in target_gws:
        gw_fixtures = fixture_rows[fixture_rows["gameweek"] == gw]
        if gw_fixtures.empty:
            continue
        odds_gw = match_odds[match_odds["gameweek"] == gw]
        home_side = gw_fixtures[gw_fixtures["was_home"]]
        pairs = home_side[["team_id_season", "opponent_team_id"]].drop_duplicates()

        for home_team, away_team in pairs.itertuples(index=False):
            home_team, away_team = int(home_team), int(away_team)
            home_ids = gw_fixtures[
                (gw_fixtures["team_id_season"] == home_team)
                & (gw_fixtures["opponent_team_id"] == away_team)
                & gw_fixtures["was_home"]
            ]["player_id"].unique()
            away_ids = gw_fixtures[
                (gw_fixtures["team_id_season"] == away_team)
                & (gw_fixtures["opponent_team_id"] == home_team)
                & ~gw_fixtures["was_home"]
            ]["player_id"].unique()

            odds_row = odds_gw[
                (odds_gw["home_team_id"] == home_team) & (odds_gw["away_team_id"] == away_team)
            ]
            if not odds_row.empty:
                r = odds_row.iloc[0]
                lam_home, lam_away = team_goals_from_odds(
                    float(r["home_win_prob"]), float(r["draw_prob"]),
                    float(r["away_win_prob"]), float(r["over25_prob"]),
                )
            else:
                lam_home, lam_away = 1.35, 1.15

            home_players = _player_dicts(home_ids, feat, bands)
            away_players = _player_dicts(away_ids, feat, bands)
            if not home_players and not away_players:
                continue

            samples = sample_fixture(
                rng, home_players, away_players, lam_home, lam_away,
                n_scenarios, defcon_field_shares,
            )
            for pid, arr in samples.items():
                mean = float(arr.mean())
                var = float(arr.var(ddof=1)) if n_scenarios > 1 else 0.0
                p2 = bands.get(pid, (0.5, 0.0, 0.5))[2]
                rows.append({
                    "player_id": pid, "gameweek": gw,
                    "xpts": mean, "xpts_mean": mean, "xpts_var": var,
                    "start_probability": float(p2),
                })

    return pd.DataFrame(rows)
