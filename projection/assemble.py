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
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert

from config.strategy import DEFCON, SCORING
from data.db import get_session
from data.models import ProjectionSample
from projection import bonus as bonus_mod
from projection import clean_sheets, defcon, goals, saves
from projection.assists import ASSIST_FRACTION, expected_assist_points
from projection.covariance import sample_team_goals, split_multinomial
from projection.minutes_model import predict_minutes_bands
from projection.team_goals import (
    NEUTRAL_LAMBDA_AWAY,
    NEUTRAL_LAMBDA_HOME,
    team_goals_from_odds,
    team_goals_from_strength,
)

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
        dribbles_ct = 0
        if played_any:
            key_passes_ct = int(rng.poisson(max(0.0, p.get("key_pass_rate", 0.0) * scale)))
            yellow_ct = int(rng.random() < min(1.0, max(0.0, p.get("yellow_rate", 0.0))))
            red_ct = int(rng.random() < min(1.0, max(0.0, p.get("red_rate", 0.0))))
            dribbles_ct = int(rng.poisson(max(0.0, p.get("dribble_rate", 0.0) * scale)))
            pts += yellow_ct * SCORING.points_yellow_card + red_ct * SCORING.points_red_card

        points_by_pid[pid] = pts
        events_by_pid[pid] = {
            "position": position,
            "minutes": 60 if played_60 else (30 if played_any else 0),
            "goals": goals_ct, "assists": assists_ct,
            "clean_sheet": 1 if (played_60 and conceded == 0) else 0,
            "saves": saves_ct, "key_passes": key_passes_ct,
            "yellow_cards": yellow_ct, "red_cards": red_ct,
            "dribbles": dribbles_ct,
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


# Below this a published strength is FPL's pre-season placeholder, not a real
# rating. Same convention and same floor as projection/features.py.
_PLAUSIBLE_STRENGTH_FLOOR = 100


def load_team_strength_rel(season: str) -> dict[int, dict[str, float]]:
    """team_id -> attack/defence strengths RELATIVE to the league average, for
    ``team_goals_from_strength`` (2026-08-18, engine review §2).

    Prefers the current season's published ratings and falls back to the prior
    season's, matched on the cross-season-stable team ``code``. That fallback
    is not an edge case: FPL publishes attack/defence as 0 until a season is
    underway, so at GW1 — exactly when the planning horizon reaches furthest
    past the last priced fixture — the current season has nothing usable and
    every team resolves through it.

    League averages are taken over whatever set is actually resolved, so the
    ratios are relative within a consistent population rather than to a
    hard-coded scale.

    Returns an empty dict when neither season has real ratings; the caller then
    uses the neutral league-average fixture.
    """
    # Imported lazily: cold_start already reaches back into this module, and a
    # top-level import here would close the loop.
    from projection.cold_start import prior_season_of

    db = get_session()
    try:
        rows = pd.read_sql(
            text("""
                SELECT season, team_id, code,
                       strength_attack_home, strength_attack_away,
                       strength_defence_home, strength_defence_away
                FROM team_season_strength
                WHERE season IN (:season, :prior)
            """),
            db.bind, params={"season": season, "prior": prior_season_of(season)},
        )
    finally:
        db.close()
    if rows.empty:
        return {}

    cols = [
        "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]
    # Per COLUMN, not per row (2026-08-18). Requiring all four to be published
    # dropped a team outright when FPL had released, say, defence but not
    # attack -- even though `team_goals_from_strength` is explicitly built to
    # degrade one term at a time. Below-floor values become NaN and that
    # promise is honoured; a row with nothing usable falls out naturally.
    usable = rows.copy()
    for col in cols:
        usable.loc[usable[col] < _PLAUSIBLE_STRENGTH_FLOOR, col] = np.nan
    usable = usable[usable[cols].notna().any(axis=1)]
    if usable.empty:
        return {}

    current = usable[usable["season"] == season].set_index("team_id")
    prior_by_code = usable[usable["season"] != season].dropna(subset=["code"])
    prior_by_code = prior_by_code.set_index("code")
    # team_id -> code, taken from whichever season knows it.
    code_by_team = (
        rows.dropna(subset=["code"]).set_index("team_id")["code"].astype(int).to_dict()
    )

    resolved: dict[int, dict[str, float]] = {}
    for team_id in rows["team_id"].unique():
        team_id = int(team_id)
        if team_id in current.index:
            src = current.loc[team_id]
        else:
            code = code_by_team.get(team_id)
            if code is None or code not in prior_by_code.index:
                continue
            src = prior_by_code.loc[code]
        resolved[team_id] = {
            c: float(src[c]) for c in cols if pd.notna(src[c])
        }

    if not resolved:
        return {}

    means = {}
    for c in cols:
        vals = [v[c] for v in resolved.values() if c in v]
        means[c] = float(np.mean(vals)) if vals else 0.0
    return {
        team_id: {
            c: v[c] / means[c] for c in cols if c in v and means[c] > 0
        }
        for team_id, v in resolved.items()
    }


def load_all_stats(season: str) -> pd.DataFrame:
    """All ``player_gw_stats`` rows for a season, joined to the point-in-time
    player state snapshot and per-GW xG stats. Shared by the backtest path
    (``scripts/backtest.py``, historically this lived there as
    ``_load_all_stats``) and the live-serving path (``projection/pipeline.py``)
    so both feed ``assemble_gw_projections`` from the identical query shape —
    moved here (P3-0) rather than duplicated, so a live/backtest divergence in
    this join can't silently creep in.

    For a live, in-progress season this naturally contains only PLAYED
    gameweeks (``player_gw_stats`` has no rows for a game that hasn't
    happened yet) — exactly the ``history`` role ``assemble_gw_projections``
    needs. It does NOT carry future-fixture context, which is why the live
    caller passes a separately-built fixture-context frame as ``all_stats``
    instead of this one for that role (see ``pipeline.py``).
    """
    db = get_session()
    try:
        # Dynamic player attributes (ict/influence/creativity/threat/form/
        # selected_by_percent/status/chance) come from the point-in-time
        # snapshot as-of the gameweek deadline — NOT the mutable players.*
        # columns, which would leak the latest value into historical training
        # rows (Phase-1 leaks L1/L2). player_gw_stats/xg_stats stay as-is
        # (already point-in-time). Static columns (position, team_id) come from
        # players.
        query = text("""
            SELECT
                s.player_id, s.gameweek, s.season,
                s.minutes, s.total_points, s.goals_scored, s.assists,
                s.clean_sheets, s.goals_conceded, s.saves,
                s.yellow_cards, s.red_cards, s.bonus, s.bps,
                s.value AS now_cost,
                s.team_id_season, s.opponent_team_id, s.was_home,
                p.position, p.team_id,
                COALESCE(ps.ict_index, 0) AS ict_index,
                COALESCE(ps.influence, 0) AS influence,
                COALESCE(ps.creativity, 0) AS creativity,
                COALESCE(ps.threat, 0) AS threat,
                COALESCE(ps.form, 0) AS form,
                COALESCE(ps.selected_by_percent, 0) AS selected_by_percent,
                COALESCE(ps.status, 'a') AS status,
                ps.chance_of_playing_next_round AS chance_of_playing_next_round,
                COALESCE(x.xg, 0) AS xg, COALESCE(x.xa, 0) AS xa,
                COALESCE(x.xgi, 0) AS xgi, COALESCE(x.npxg, 0) AS npxg,
                COALESCE(x.shots, 0) AS shots, COALESCE(x.key_passes, 0) AS key_passes
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
            WHERE s.season = :season
            ORDER BY s.player_id, s.gameweek
        """)
        return pd.read_sql(query, db.bind, params={"season": season})
    finally:
        db.close()


def load_defcon_events(season: str) -> pd.DataFrame:
    """Per (player, gw) raw defensive-action + dribbles fields from
    ``player_match_events`` — merged into the rolling-rate build alongside
    ``player_xg_stats``/``player_gw_stats`` (P7's rate, computed here per its
    own docstring). ``dribbles`` feeds the bonus/BPS ``successful_dribble``
    channel (P10 finding: forwards were badly under-credited for bonus
    without it — real FWD P(bonus>0) 15.1% vs 8.6% modelled)."""
    db = get_session()
    try:
        return pd.read_sql(
            text("""
                SELECT player_id, gameweek, season,
                       clearances, blocks, interceptions, tackles, recoveries,
                       dribbles
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


def load_penalty_duty(season: str) -> dict[int, float]:
    """player_id -> expected penalty GOAL value per game, for a season whose
    taker duty is known (2026-08-16).

    Empty dict when no depth chart has been loaded, which is the signal
    ``_build_rolling_features`` uses to keep its previous behaviour.
    """
    db = get_session()
    try:
        rows = db.execute(
            text(
                "SELECT player_id, penalty_xg_per_game FROM player_setpiece_roles "
                "WHERE season = :season AND penalty_order IS NOT NULL"
            ),
            {"season": season},
        ).fetchall()
    finally:
        db.close()
    return {int(pid): float(xg or 0.0) for pid, xg in rows}


# How much weight the PRIOR season's per-match rate carries, expressed in
# current-season gameweeks (2026-08-18, engine review §20 follow-up). At n
# played gameweeks the current season gets n/(n + this) of the blend, so:
#
#   after 1 GW   -> 25% current, 75% prior
#   after 3 GWs  -> 50/50
#   after 10 GWs -> 77% current
#   after 20 GWs -> 87% current
#
# Deliberately modest. A rate measured over one match is nearly worthless on
# its own, and the alternative the engine used was not "wait for data" but
# "assume zero" -- which is a far stronger and far wronger claim than "he is
# probably similar to last year". Untuned starting value, same convention as
# the other heuristic constants; 0.0 disables blending exactly.
_PRIOR_SEASON_BLEND_GWS = 3.0

_PRIOR_RATE_SOURCES = (
    "xg", "npxg", "xa", "key_passes", "yellow_cards", "red_cards",
    "cbit", "cbirt", "dribbles",
)


def load_prior_season_rates(prior_season: str) -> pd.DataFrame:
    """Per-player per-match means from the PRIOR season, indexed by player_id.

    ``players.id`` is the stable cross-season identity (the table is upserted
    on the FPL ``code``), so this joins straight onto the current season's
    players without a name match — verified: 534 player_ids appear in both
    2024-25 and 2025-26.

    Returns an empty frame when the prior season has no data, which makes the
    blend below a no-op rather than an error.
    """
    hist = load_all_stats(prior_season)
    if hist.empty:
        return pd.DataFrame()
    df = hist.copy()
    defcon_events = load_defcon_events(prior_season)
    if not defcon_events.empty:
        de = defcon_events.copy()
        de["cbit"] = de["clearances"] + de["blocks"] + de["interceptions"] + de["tackles"]
        de["cbirt"] = de["cbit"] + de["recoveries"]
        df = df.merge(
            de[["player_id", "gameweek", "season", "cbit", "cbirt", "dribbles"]],
            on=["player_id", "gameweek", "season"], how="left",
        )
    if "minutes" in df.columns:
        df = df[df["minutes"] > 0]
    if df.empty:
        return pd.DataFrame()
    for col in _PRIOR_RATE_SOURCES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    return df.groupby("player_id")[list(_PRIOR_RATE_SOURCES)].mean()


def _blend_toward_prior(
    last: pd.DataFrame,
    played: pd.Series,
    prior_rates: pd.DataFrame,
    rate_cols: dict[str, str],
    blend_gws: float = _PRIOR_SEASON_BLEND_GWS,
) -> pd.DataFrame:
    """Shrink each current-season rate toward the same player's prior-season
    rate, weighted by how many gameweeks the current one is actually built on.

    This is the general form of the §20 fix. Removing the stray ``shift(1)``
    stopped the engine throwing away its newest gameweek, but it could not help
    the deeper problem: at GW2 a rate rests on ONE match, and by GW5 on four.
    The engine's implicit answer to "what do I know about this player" was
    whatever those few matches happened to say — and, before §20, literally
    zero. A whole prior season of real per-match rates sat unused two feet away.

    Weighting by sample size is the standard answer and needs no tuning
    beyond the prior's strength: a player with one match is mostly last
    season's player, a player with fifteen is mostly this season's. New
    signings and promoted-club players simply have no prior row and are left
    on their current-season rate alone.
    """
    if prior_rates.empty or blend_gws <= 0:
        return last
    out = last.copy()
    n = played.reindex(out.index).fillna(0.0).astype(float)
    weight_current = n / (n + blend_gws)
    for src, col in rate_cols.items():
        if src not in prior_rates.columns or col not in out.columns:
            continue
        prior = prior_rates[src].reindex(out.index)
        blended = weight_current * out[col] + (1.0 - weight_current) * prior
        # A player absent from the prior season keeps their current rate.
        out[col] = blended.where(prior.notna(), out[col])
    return out


def _build_rolling_features(
    history: pd.DataFrame,
    defcon_events: pd.DataFrame,
    penalty_duty: dict[int, float] | None = None,
    target_gw: int | None = None,
    prior_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per player, as-of ``history``: leakage-free rolling rates feeding the
    components — same
    pattern as ``points_model._build_features``, over the raw columns the MC
    components need instead of an ML model's feature set. Indexed by
    ``player_id``, one row (latest as-of) each.

    ``target_gw`` (2026-08-18, engine review §20): the gameweek being projected.
    Rows at or after it are dropped here, so this function owns the leakage
    boundary rather than trusting callers to have pre-truncated.

    **The ``shift(1)`` this used to carry has been removed, and that is the
    point of the parameter.** It was inherited from
    ``points_model._build_features``, where the frame legitimately contains the
    row being predicted and shifting is the only thing standing between you and
    a leak. Here the frame is already strictly prior to ``target_gw``, so the
    shift was protecting against a leak that truncation had already prevented —
    and it cost the most recent, most informative gameweek, permanently:

    - With ONE played gameweek, ``shift(1)`` on a single row is NaN, and the
      ``fillna(0.0)`` below turned that into a confident zero. So at GW2, the
      first in-season decision of the season, EVERY rate was 0: goal_weight,
      assist_weight, defcon_rate, key_pass_rate, dribble_rate, cards. Attacking
      returns went unattributed, DefCon could not reach its threshold, and
      projections collapsed to appearance points plus clean sheets and saves.
    - From then on the five-gameweek "form" window was really gameweeks n-5..n-1
      rather than n-4..n — always a week stale, and worst precisely after an
      injury return or a transfer, when the ignored match is the informative one.

    Verified by giving one player a different CBIT each week and checking which
    gameweeks the resulting rate could have come from.
    """
    df = history.sort_values(["player_id", "gameweek"]).copy()
    if target_gw is not None and "gameweek" in df.columns:
        df = df[df["gameweek"] < target_gw]
    if not defcon_events.empty:
        de = defcon_events.copy()
        de["cbit"] = de["clearances"] + de["blocks"] + de["interceptions"] + de["tackles"]
        de["cbirt"] = de["cbit"] + de["recoveries"]
        df = df.merge(
            de[["player_id", "gameweek", "season", "cbit", "cbirt", "dribbles"]],
            on=["player_id", "gameweek", "season"], how="left",
        )
    for col in ("cbit", "cbirt", "dribbles"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    grp = df.groupby("player_id")
    rate_cols = {
        "xg": "goal_weight", "npxg": "npxg_rate",
        "xa": "assist_weight", "key_passes": "key_pass_rate",
        "yellow_cards": "yellow_rate", "red_cards": "red_rate",
        "cbit": "cbit_rate", "cbirt": "cbirt_rate", "dribbles": "dribble_rate",
    }
    for src, out in rate_cols.items():
        if src not in df.columns:
            df[src] = 0.0
        df[out] = grp[src].transform(
            lambda x: x.rolling(_ROLLING_WINDOW, min_periods=1).mean()
        )

    last = df.groupby("player_id").last()
    # Gameweeks each player's current-season rate is actually built on, which
    # is what decides how far to lean on last season below.
    played_gws = df.groupby("player_id")["gameweek"].nunique()
    if prior_rates is not None:
        last = _blend_toward_prior(last, played_gws, prior_rates, rate_cols)

    # Penalty duty (2026-08-16). `goal_weight` is the share by which a team's
    # drawn goals get attributed to each player, in expected-goals-per-game
    # units. Deriving it from rolling `xg` bakes in whatever penalties a
    # player happened to take in the PAST, which is wrong in both directions
    # across a transfer window: Isak's Newcastle penalties followed him to
    # Liverpool in the data, and Woltemade inherited Newcastle's duty with no
    # history to show for it.
    #
    # When the season's duty is known, decompose instead:
    #     goal_weight = rolling non-penalty xG  +  this season's penalty duty
    # which is exact rather than a correction -- npxg and xg differ by
    # precisely the penalty component. With no depth chart loaded the frame
    # keeps its previous `xg` basis, so historical backtests are unaffected.
    # The decomposition is only VALID if npxg is genuinely non-penalty xG.
    # It is not always: data/ingestors/understat_xg.py's per-gameweek feed has
    # no penalty split and stores npxg = xg verbatim (its own docstring says
    # so), and it wins the upsert against the season-level ingest that does
    # carry real npxG. Applying the decomposition against a copy would ADD a
    # taker's penalty expectation on top of an xG that already contains it --
    # double-counting precisely the premium players the optimiser's curse
    # already over-selects, which is worse than not decomposing at all.
    has_real_npxg = bool((df["npxg"] < df["xg"] - 1e-9).any())
    if penalty_duty and has_real_npxg:
        last["goal_weight"] = last["npxg_rate"].fillna(0.0) + [
            penalty_duty.get(int(pid), 0.0) for pid in last.index
        ]
    elif penalty_duty:
        logger.warning(
            "Penalty duty is known for %d players but npxg is not distinct from "
            "xg in this history, so goal_weight keeps its total-xG basis — the "
            "penalty component cannot be separated out without double-counting. "
            "Fix by giving player_xg_stats a real non-penalty xG.",
            len(penalty_duty),
        )

    last["defcon_rate"] = np.where(
        last["position"] == "DEF", last["cbit_rate"], last["cbirt_rate"]
    )
    cols = ["position", "team_id_season", "goal_weight", "assist_weight",
            "key_pass_rate", "yellow_rate", "red_rate", "defcon_rate", "dribble_rate"]
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
            "dribble_rate": row["dribble_rate"],
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
    persist_samples: bool = False,
    season: str | None = None,
    strength_rel: Mapping[int, Mapping[str, float]] | None = None,
    prior_rates: pd.DataFrame | None = None,
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
    changing the optimiser's read side.

    ``persist_samples`` (P3-1, requires ``season``): writes each fixture's raw
    per-scenario draws to ``ProjectionSample`` instead of discarding them
    after the mean/var reduction — the vehicle for real teammate covariance
    (P-COV), since ``ProjectionSample.scenario_id`` is only meaningful shared
    randomness for players drawn in the SAME fixture. Off by default (the
    33-GW backtest walk-forward calls this hundreds of times and does not
    want tens of thousands of DB rows per call) — the live-serving path
    (``pipeline.py``) turns it on. Each fixture gets its own disjoint
    scenario_id range within a gameweek (offset by ``n_scenarios`` per
    fixture already assembled that GW) specifically so a consumer joining on
    (season, gameweek, scenario_id) alone can't accidentally correlate two
    players from DIFFERENT matches who happen to share a raw scenario index —
    only real teammates (same fixture) ever share a scenario_id range.
    Storage/retention policy is not addressed here (flagged, not solved —
    same open item P0 originally noted).

    ``strength_rel`` (§2, optional): league-relative team strengths from
    ``load_team_strength_rel``, used to derive λ for fixtures the bookmakers
    have not priced. Odds always win where they exist. Omitting it (or passing
    an empty mapping) falls back to the league-average fixture, which is the
    old behaviour except that the constants are now the calibrated league
    means rather than values 0.25 goals a game below them."""
    if history.empty:
        return pd.DataFrame()
    if persist_samples and not season:
        raise ValueError("persist_samples=True needs season (for the ProjectionSample rows)")
    strength_rel = strength_rel or {}

    feat = _build_rolling_features(
        history, defcon_events,
        penalty_duty=load_penalty_duty(season) if season else None,
        target_gw=target_gw,
        prior_rates=prior_rates,
    )
    bands = predict_minutes_bands(history, minutes_model)

    rng = np.random.default_rng(seed)
    target_gws = list(range(target_gw, target_gw + horizon))
    # P12: dedupe on the fixture's own identity, NOT just (player_id, gameweek)
    # -- a genuine double-gameweek player has TWO real rows here (same
    # gameweek, different opponent_team_id/was_home), and both must survive
    # so both fixtures get sampled; the old (player_id, gameweek) dedupe
    # silently dropped a DGW player's second fixture entirely.
    fixture_rows = all_stats[all_stats["gameweek"].isin(target_gws)][
        ["player_id", "gameweek", "team_id_season", "opponent_team_id", "was_home"]
    ].dropna(subset=["opponent_team_id"]).drop_duplicates(
        subset=["player_id", "gameweek", "opponent_team_id", "was_home"]
    )
    fixture_rows = fixture_rows.assign(was_home=fixture_rows["was_home"].fillna(False).astype(bool))

    rows = []
    sample_rows: list[dict] = []
    for gw in target_gws:
        gw_fixtures = fixture_rows[fixture_rows["gameweek"] == gw]
        if gw_fixtures.empty:
            continue
        odds_gw = match_odds[match_odds["gameweek"] == gw]
        home_side = gw_fixtures[gw_fixtures["was_home"]]
        pairs = home_side[["team_id_season", "opponent_team_id"]].drop_duplicates()
        scenario_offset = 0  # disjoint per-fixture range within this GW (P3-1)
        # P12: a DGW player appears in >1 pair iteration this GW -- merge
        # their per-fixture contributions into ONE row per (player_id, gw)
        # (summing xpts/var, since independent fixtures' variances add;
        # combining start_probability as P(starts >= one) rather than
        # overwriting) so downstream 1-row-per-player consumers (e.g.
        # optimise_starting_xi's merge) see the correct DOUBLED total
        # instead of silent duplication or truncation to one fixture.
        gw_row_by_pid: dict[int, dict] = {}

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
                # §2 (2026-08-18): odds cover the next week or two at most,
                # while the transfer planner looks three gameweeks ahead and
                # the wildcard evaluation five. This used to hand every
                # unpriced fixture the same flat pair, so beyond the priced
                # window no fixture was distinguishable from any other — a
                # defender's clean-sheet probability did not depend on who
                # they played, across the entire planning horizon.
                #
                # Team strengths are a much weaker signal than the market, and
                # are only ever consulted where the market is silent.
                h = strength_rel.get(home_team)
                a = strength_rel.get(away_team)
                if h is None and a is None:
                    lam_home, lam_away = NEUTRAL_LAMBDA_HOME, NEUTRAL_LAMBDA_AWAY
                else:
                    h, a = h or {}, a or {}
                    lam_home, lam_away = team_goals_from_strength(
                        home_attack_rel=h.get("strength_attack_home"),
                        home_defence_rel=h.get("strength_defence_home"),
                        away_attack_rel=a.get("strength_attack_away"),
                        away_defence_rel=a.get("strength_defence_away"),
                    )

            home_players = _player_dicts(home_ids, feat, bands)
            away_players = _player_dicts(away_ids, feat, bands)
            if not home_players and not away_players:
                continue

            samples = sample_fixture(
                rng, home_players, away_players, lam_home, lam_away,
                n_scenarios, defcon_field_shares,
            )
            # P(clean sheet) per side, in CLOSED FORM (2026-08-16).
            # sample_team_goals draws Poisson(λ), so P(concede 0) is exactly
            # exp(-λ_opponent) — no need to reduce the scenarios, and no
            # change to sample_fixture's return contract. Multiplied by
            # P(60+ minutes), since FPL only awards a clean sheet to a player
            # who reaches 60.
            #
            # Fills a column that had been 0.0 on every row ever written:
            # assemble.py has each player's per-scenario clean sheet
            # internally (it feeds the BPS simulator) but never surfaced it,
            # so scripts/plot_analysis.py's clean-sheet-by-team chart was
            # permanently blank -- not a pre-season artefact.
            cs_home = float(np.exp(-max(0.0, lam_away)))
            cs_away = float(np.exp(-max(0.0, lam_home)))
            home_id_set = {int(p["player_id"]) for p in home_players}

            for pid, arr in samples.items():
                mean = float(arr.mean())
                var = float(arr.var(ddof=1)) if n_scenarios > 1 else 0.0
                p2 = float(bands.get(pid, (0.5, 0.0, 0.5))[2])
                cs_p = (cs_home if pid in home_id_set else cs_away) * p2
                if pid in gw_row_by_pid:
                    prev = gw_row_by_pid[pid]
                    prev["xpts"] += mean
                    prev["xpts_mean"] += mean
                    prev["xpts_var"] += var
                    prev["start_probability"] = 1.0 - (1.0 - prev["start_probability"]) * (1.0 - p2)
                    # A double gameweek gives two chances at a clean sheet;
                    # combined as P(at least one), matching how
                    # start_probability is merged just above.
                    prev["cs_probability"] = 1.0 - (1.0 - prev["cs_probability"]) * (1.0 - cs_p)
                else:
                    gw_row_by_pid[pid] = {
                        "player_id": pid, "gameweek": gw,
                        "xpts": mean, "xpts_mean": mean, "xpts_var": var,
                        "start_probability": p2, "cs_probability": cs_p,
                    }
                if persist_samples:
                    sample_rows.extend(
                        {
                            "player_id": pid, "gameweek": gw, "season": season,
                            "scenario_id": scenario_offset + i, "xpts": float(x),
                        }
                        for i, x in enumerate(arr)
                    )
            if persist_samples:
                scenario_offset += n_scenarios

        rows.extend(gw_row_by_pid.values())

    if persist_samples and sample_rows:
        _write_projection_samples(sample_rows)

    return pd.DataFrame(rows)


MIN_SHRINKAGE_GROUP_SIZE = 3

# Width of the price band that per-player shrinkage regresses within (§19).
# £1.0m matches cold_start's own peer-bucket rounding, so the two agree on what
# "a similar player" means.
_SHRINKAGE_PRICE_BAND = 1.0

# Uniform shrinkage strength toward the (gameweek, position) group mean (see
# apply_curse_shrinkage). Untuned starting value pending backtesting, same
# convention as the P3-6 constant this supersedes; 0.0 disables shrinkage
# exactly. Deliberately NOT derived from xpts_var (see the function
# docstring for why a variance-RATIO shrink factor was tried and reverted).
CURSE_SHRINKAGE_STRENGTH = 0.15


def apply_curse_shrinkage(projections: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Shrinks ``xpts`` toward its (gameweek, position) group mean by a
    fixed fraction (``CURSE_SHRINKAGE_STRENGTH``) — corrects the "optimiser's
    curse" (2026-07-28 data-completeness audit finding): repeatedly picking
    whoever's CURRENT projection looks highest systematically overselects
    players whose estimate is inflated by noise rather than true ability.
    Confirmed empirically against three sample gameweeks: while the raw
    model is essentially unbiased across the whole player pool (bias ≈ 0),
    the top-50 players BY PROJECTED xpts each week — exactly the pool an
    optimiser draws from — showed a consistent +1.2 to +1.3 point/player
    bias. P3-6 (2026-07-28, earlier the same day) discounted this only
    inside the weekly transfer ILP via a flat, untuned
    ``transfer_variance_penalty``; this supersedes it with a single
    correction applied once at the projection-assembly boundary, so every
    consumer (squad-building, starting-XI/captaincy, transfers, live
    serving) sees the corrected value automatically instead of needing its
    own copy of the same fix.

    **Two modes.** When the frame carries an ``estimation_se`` column — how
    well each player's MEAN is known — shrinkage is per-player empirical Bayes:
    each estimate keeps ``tau^2 / (tau^2 + se^2)`` of its distance from the
    group mean, so a well-measured player barely moves and a pooled guess about
    an unknown collapses toward the mean. This REORDERS players, which is the
    only way to actually correct a selection bias (engine review §19). Without
    that column it falls back to the flat ``CURSE_SHRINKAGE_STRENGTH`` below,
    which corrects the level but provably cannot change who gets picked: a
    uniform affine map inside a group preserves the ranking exactly, and with
    squad quotas fixed the offset term is constant across every legal squad.

    The distinction that makes this work is WHICH variance is used. See below
    for the one that does not.

    A first version weighted the shrinkage per-player by ``xpts_var``
    relative to the group's between-player variance (textbook James-Stein
    empirical-Bayes shrinkage) — reverted after a live walk-forward gate run
    showed it collapsing `predicted` to a near-constant ~22-24 pts/GW
    regardless of squad. Root cause: ``xpts_var`` is the MC simulator's
    OUTCOME variance (how spiky a player's week-to-week returns are, e.g. a
    explosive-returns forward legitimately has high xpts_var with a
    precisely-known mean), not the model's ESTIMATION uncertainty about that
    mean — conflating the two is wrong, and in practice mean per-player
    xpts_var (checked live: ~3.3-4.5 across positions) is comparable to or
    LARGER than the real between-player variance (~1.7-2.9), so the ratio
    shrunk nearly every player toward the mean regardless of confidence,
    destroying the ranking signal instead of correcting a bias. A flat,
    uniform strength avoids this conflation entirely at the cost of not
    adapting per-player — a deliberate, safer trade-off pending a real
    estimation-uncertainty signal (e.g. multi-seed reassembly variance)
    being available to shrink by instead.

    The ORIGINAL (unshrunk) value is preserved as ``xpts_raw`` — ``xpts``
    itself becomes the shrunk, decision-facing value; ``xpts_mean``/
    ``xpts_var`` (the simulator's own honest per-scenario summary) are left
    untouched.

    A (gameweek, position) group with fewer than
    ``MIN_SHRINKAGE_GROUP_SIZE`` players is left unshrunk for that group.
    Returns ``projections`` unchanged (no-op, not even ``xpts_raw`` added)
    if it's empty or ``players`` lacks ``position`` — a minimal test/caller
    fixture without position data has nothing this can safely group by."""
    has_position = "position" in projections.columns or "position" in players.columns
    if projections.empty or not has_position:
        return projections

    # Take only what `projections` does not already carry -- the cold-start
    # frame now brings its own `position`, and merging a second one produces
    # position_x/position_y and a KeyError below.
    merge_cols = ["id"]
    if "position" not in projections.columns:
        merge_cols.append("position")
    if "now_cost" in players.columns and "now_cost" not in projections.columns:
        merge_cols.append("now_cost")
    out = projections.merge(
        players[merge_cols], left_on="player_id", right_on="id", how="left"
    ).drop(columns=["id"])
    out["xpts_raw"] = out["xpts"]
    shrunk = out["xpts"].copy()

    # Only players who are actually expected to feature take part (2026-08-18,
    # engine review §3). A zero here is not a low estimate — it is a statement
    # that the player will not play at all: the unavailable are zeroed and
    # departures discounted to 0.0 BEFORE this runs.
    #
    # Shrinking those toward a positive group mean resurrects them. At the
    # default strength a zeroed player in a group averaging 2.0 comes back out
    # at 0.30 xpts, so a confirmed leaver the departure gate had just
    # eliminated becomes selectable again. They also drag the mean down, which
    # made every real player's correction depend on how many non-participants
    # happened to be in that week's frame — the correction for Salah should
    # not move because a fringe player got injured.
    #
    # Excluding them fixes both. The curse being corrected is a SELECTION
    # effect, and these rows were never selection candidates.
    plays = out["xpts"] > 0.0
    # §19 (2026-08-18): shrink each player by HIS OWN estimation uncertainty
    # when it is known, rather than everybody by the same fraction.
    has_se = (
        "estimation_se" in out.columns
        and out.loc[plays, "estimation_se"].notna().any()
    )

    # What each estimate is shrunk TOWARD (§19). Position alone is right for the
    # flat mode, but wrong once shrinkage is per-player: an unknown's estimate
    # IS the pooled mean of players like him, so pulling him further toward the
    # whole position's average — which includes every premium — inflates him.
    # Measured: cheap peer-bucket players were pulled UP by 0.22 xPts on
    # average, and the GW1 squad swapped a premium defender for budget names on
    # the strength of it. Price is the market's own expectation, so banding by
    # it means a £4.5m unknown regresses toward other £4.5m players rather than
    # toward Bruno Fernandes.
    #
    # The flat path keeps the original position-only grouping, byte for byte.
    group_keys = ["gameweek", "position"]
    if has_se and "now_cost" in out.columns:
        out["_price_band"] = (out["now_cost"] // _SHRINKAGE_PRICE_BAND) * _SHRINKAGE_PRICE_BAND
        group_keys.append("_price_band")

    for _key, group in out[plays].groupby(group_keys):
        if len(group) < MIN_SHRINKAGE_GROUP_SIZE:
            continue
        group_mean = group["xpts"].mean()
        deviation = group["xpts"] - group_mean
        if has_se:
            # Empirical Bayes. An estimate keeps the share of its deviation
            # that its own precision justifies:
            #
            #     keep = tau^2 / (tau^2 + se^2)
            #
            # tau^2 is the spread of true ability across the group; se^2 is how
            # badly this particular estimate is measured. A 38-appearance
            # prior-season record keeps almost all of its distance from the
            # mean; a peer-bucket guess about an unknown keeps little. THAT is
            # what corrects the optimiser's curse -- it reorders players, which
            # a uniform shrink provably cannot do.
            #
            # Where an SE is missing the player falls back to the flat rate, so
            # a partially-annotated frame degrades one player at a time.
            se = pd.to_numeric(group.get("estimation_se"), errors="coerce")
            tau_sq = max(float(deviation.var(ddof=1)) if len(group) > 1 else 0.0, 1e-9)
            keep = tau_sq / (tau_sq + se.pow(2))
            keep = keep.fillna(1.0 - CURSE_SHRINKAGE_STRENGTH).clip(0.0, 1.0)
        else:
            keep = 1.0 - CURSE_SHRINKAGE_STRENGTH
        shrunk.loc[group.index] = group_mean + keep * deviation

    out["xpts"] = shrunk
    return out.drop(columns=[c for c in ("position", "now_cost", "_price_band")
                             if c in out.columns])


def _write_projection_samples(sample_rows: list[dict]) -> int:
    """Bulk-writes P-COV scenario draws (P3-1). One INSERT for the whole
    batch rather than per-row execute — a single gameweek's live projection
    run can be tens of thousands of rows (n_scenarios × players × fixtures).
    Returns the attempted row count (not a DB-confirmed count — the
    executemany-style bulk insert's result doesn't expose a reliable
    ``rowcount`` across dialects; ``on_conflict_do_nothing`` means the true
    written count could be lower on a re-run, which is fine here — this
    return value is informational/logging only, not relied on for control
    flow)."""
    db = get_session()
    try:
        created_at = datetime.utcnow()
        for row in sample_rows:
            row["created_at"] = created_at
        stmt = insert(ProjectionSample).on_conflict_do_nothing()
        db.execute(stmt, sample_rows)
        db.commit()
        return len(sample_rows)
    finally:
        db.close()
