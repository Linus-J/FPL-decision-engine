import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert

from config.strategy import OPTIMISER
from data.db import get_session
from data.models import Gameweek, PlayerProjection
from data.overrides import load_start_probability_caps
from optimiser.rotation_risk import apply_rotation_risk
from projection import assemble
from projection.cold_start import cold_start_projections, prior_season_of
from projection.minutes_model import _build_features as _minutes_features
from projection.minutes_model import train as train_minutes

logger = logging.getLogger(__name__)


def _get_current_and_next_gw() -> tuple[int, int]:
    db = get_session()
    try:
        current = db.query(Gameweek).filter(Gameweek.is_current.is_(True)).first()
        next_gw = db.query(Gameweek).filter(Gameweek.is_next.is_(True)).first()
        current_id = current.id if current else 1
        next_id = next_gw.id if next_gw else current_id + 1
        return current_id, next_id
    finally:
        db.close()


def _get_current_season(default: str = "2026-27") -> str:
    """Season of the live gameweek. Needed to scope (season, gw)-keyed reads
    now that gameweeks/fixtures are keyed per season (Phase-1 finding M1)."""
    db = get_session()
    try:
        current = db.query(Gameweek).filter(Gameweek.is_current.is_(True)).first()
        if current is None:
            current = db.query(Gameweek).filter(Gameweek.is_next.is_(True)).first()
        return current.season if current else default
    finally:
        db.close()


def _get_dgw_gameweeks(lookahead: int) -> set[int]:
    db = get_session()
    try:
        _, next_gw = _get_current_and_next_gw()
        season = _get_current_season()
        dgw_gws = (
            db.query(Gameweek.id)
            .filter(
                Gameweek.season == season,
                Gameweek.id >= next_gw,
                Gameweek.id < next_gw + lookahead,
                Gameweek.is_dgw.is_(True),
            )
            .all()
        )
        return {row[0] for row in dgw_gws}
    finally:
        db.close()


def _get_bgw_gameweeks(lookahead: int) -> set[int]:
    db = get_session()
    try:
        _, next_gw = _get_current_and_next_gw()
        season = _get_current_season()
        bgw_gws = (
            db.query(Gameweek.id)
            .filter(
                Gameweek.season == season,
                Gameweek.id >= next_gw,
                Gameweek.id < next_gw + lookahead,
                Gameweek.is_bgw.is_(True),
            )
            .all()
        )
        return {row[0] for row in bgw_gws}
    finally:
        db.close()


def _get_all_players() -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT id, fpl_id, web_name, position, team_id, now_cost,
                   status, chance_of_playing_next_round, selected_by_percent,
                   form, ict_index, influence, creativity, threat, injury_severity
            FROM players
        """)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()


# Real bug found 2026-07-30 (user's own review, after adding a real Guardian
# API key): data.ingestors.injury_parser has parsed FPL's free-text news
# field into players.injury_severity (0-3) since it was written, but NOTHING
# downstream ever read the column -- a fully-wired, fully-dead signal.
# projection/features.py's OWN enrichment query hardcodes it to 0 (correctly
# -- today's news can't be recovered for a past training row, so wiring it
# into minutes_model.py's shared training/backtest pipeline the way status/
# dnp_streak are would LEAK today's information onto every historical row).
# This applies it as a direct post-hoc discount on LIVE projections only,
# for players NOT already hard-excluded by status (i/u/s, zeroed below) --
# some informative news text ahead of an official status change is real,
# actionable signal a human reviewer would want reflected before they even
# open the report.
_INJURY_SEVERITY_RETENTION = {0: 1.0, 1: 0.7, 2: 0.35, 3: 0.05}


def _apply_injury_severity_discount(
    projections_df: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    if "injury_severity" not in players.columns or projections_df.empty:
        return projections_df
    severity_by_id = players.set_index("id")["injury_severity"].fillna(0).astype(int)
    retention = (
        projections_df["player_id"]
        .map(severity_by_id)
        .fillna(0)
        .astype(int)
        .map(lambda s: _INJURY_SEVERITY_RETENTION.get(s, 0.05))
    )
    for col in ("xpts", "xpts_mean", "xpts_var", "start_probability"):
        if col in projections_df.columns:
            projections_df[col] = projections_df[col] * retention
    return projections_df


def _build_live_fixture_context(season: str, target_gws: list[int]) -> pd.DataFrame:
    """(player_id, gameweek, team_id_season, opponent_team_id, was_home) for
    the LIVE horizon, from the season-aware ``fixtures`` table joined to each
    player's CURRENT team (P-FIX/P3-0). The backtest path gets this from a
    player's own LATER ``player_gw_stats`` rows (safe — fixture info known in
    advance, not an outcome), but those rows don't exist yet for an unplayed
    fixture, so live serving needs its own source. Same shape as
    ``assemble.load_all_stats``'s fixture columns — drops into
    ``assemble_gw_projections``'s ``all_stats`` role directly.

    DGW note: a player with two fixtures in one gameweek gets two rows here;
    ``assemble_gw_projections`` currently dedupes to one (P12 defers proper
    per-team DGW handling) — same known simplification the backtest path
    already has, not solved here.
    """
    if not target_gws:
        return pd.DataFrame(
            columns=["player_id", "gameweek", "team_id_season", "opponent_team_id", "was_home"]
        )
    db = get_session()
    try:
        placeholders = ",".join(f":gw{i}" for i in range(len(target_gws)))
        params = {"season": season, **{f"gw{i}": gw for i, gw in enumerate(target_gws)}}
        query = text(f"""
            SELECT p.id AS player_id, f.gameweek,
                   p.team_id AS team_id_season,
                   CASE WHEN f.team_h_id = p.team_id THEN f.team_a_id ELSE f.team_h_id END
                       AS opponent_team_id,
                   CASE WHEN f.team_h_id = p.team_id THEN 1 ELSE 0 END AS was_home
            FROM players p
            JOIN fixtures f ON (f.team_h_id = p.team_id OR f.team_a_id = p.team_id)
            WHERE f.season = :season AND f.gameweek IN ({placeholders})
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def _load_live_match_odds(season: str, target_gws: list[int]) -> pd.DataFrame:
    """Raw de-vigged 1X2 + O/U2.5 for the live horizon's fixtures, as-of each
    target GW's own deadline (the latest ``fixture_odds`` fetch at/before that
    deadline — same leakage-free posture as ``features.load_live_odds_asof``,
    generalised across a whole horizon and returning the raw
    ``team_goals_from_odds`` inputs instead of the derived CS/BTTS fields)."""
    if not target_gws:
        return pd.DataFrame(columns=[
            "gameweek", "home_team_id", "away_team_id",
            "home_win_prob", "draw_prob", "away_win_prob", "over25_prob",
        ])
    db = get_session()
    try:
        placeholders = ",".join(f":gw{i}" for i in range(len(target_gws)))
        params = {"season": season, **{f"gw{i}": gw for i, gw in enumerate(target_gws)}}
        query = text(f"""
            SELECT f.gameweek, f.team_h_id AS home_team_id, f.team_a_id AS away_team_id,
                   fo.home_win_prob, fo.draw_prob, fo.away_win_prob, fo.over25_prob
            FROM fixtures f
            JOIN gameweeks g ON g.id = f.gameweek AND g.season = f.season
            JOIN fixture_odds fo ON fo.fixture_id = f.id
                AND fo.fetched_at <= g.deadline_time
                AND fo.fetched_at = (
                    SELECT MAX(fo2.fetched_at) FROM fixture_odds fo2
                    WHERE fo2.fixture_id = f.id AND fo2.fetched_at <= g.deadline_time
                )
            WHERE f.season = :season AND f.gameweek IN ({placeholders})
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def season_has_played_history(season: str) -> bool:
    """True once there's real rolling history to condition on for this
    season -- i.e. we're past the true pre-season cold-start gap. Callers
    that need to distinguish "genuinely no projections available" from
    "still pre-season, use the cold-start path" should check this rather
    than any state that persists across cold-start reruns (e.g. whether a
    squad was already recorded)."""
    return not assemble.load_all_stats(season).empty


# Below this many usable rows, the current season alone cannot train the
# minutes model and all available history is used instead. One PL gameweek
# yields roughly 550 rows before feature-building and none after, so this
# clears the degenerate early-season case without displacing a real season.
MIN_CURRENT_SEASON_TRAINING_ROWS = 1000


def run_projections(
    season: str = "2026-27",
    horizon: int | None = None,
    persist: bool = True,
    n_scenarios: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """P3-0: live-serving projections via the P10 MC assembly
    (``projection.assemble``) — replaces the old monolithic
    ``points_model``/``minutes_model``/``cs_model`` combo, which wrote
    ``xpts_mean``/``xpts_var`` as inert 0.0 on every row. Real per-fixture
    odds-implied λ + the P-COV shared-latent joint sampling now apply live,
    the same engine already validated in the backtest harness.

    Known limitation NOT solved here: GW1 cold start. If ``season`` has no
    played gameweeks yet, ``assemble_gw_projections`` has no rolling history
    to condition on and returns nothing — this is the same gap T7's
    cold-start harness / P11's prior-league priors exist to fill, tracked
    separately, not addressed by this live-serving rewiring.
    """
    # P3.5: build the LONGEST horizon any consumer slices, not just the
    # transfer planner's -- see config.strategy.assert_horizons_consistent.
    horizon = horizon or OPTIMISER.projection_horizon_gws
    n_scenarios = n_scenarios or assemble.DEFAULT_N_SCENARIOS
    _, next_gw = _get_current_and_next_gw()
    target_gws = list(range(next_gw, next_gw + horizon))

    logger.info("Running projections (P10 MC assembly) for GWs %s", target_gws)

    history = assemble.load_all_stats(season)
    if history.empty:
        logger.warning(
            "No played gameweeks yet for %s — cold start, no rolling history "
            "for assemble.py to condition on. Returning no projections (see "
            "T7/P11 for the separate cold-start path).",
            season,
        )
        empty_cols = ["player_id", "gameweek", "xpts", "xpts_mean", "xpts_var", "start_probability"]
        if persist:
            persist_projections(pd.DataFrame(columns=empty_cols))
        return pd.DataFrame(columns=empty_cols)

    # The minutes model trains on the CURRENT season's history when there is
    # enough of it, and on all available history when there is not.
    #
    # Early in a season there is not. _build_features derives avg_minutes_5gw
    # and season_avg_minutes with .shift(1) grouped by (player_id, season) and
    # then drops rows where they are null, so a player's FIRST appearance of a
    # season never survives. One gameweek in, every row is a first appearance:
    # 571 rows go in and 0 come out, and train() then dies on an empty frame
    # (IndexError: single positional indexer is out-of-bounds). Hit live at
    # GW2 of 2026-27 on 2026-08-25 -- the pipeline guarded "no gameweeks
    # played", which routes to the cold start, but not "one gameweek played",
    # which falls between the two paths.
    #
    # Widening the training set is sound here rather than merely expedient:
    # this model predicts MINUTES, and minutes are unaffected by the scoring
    # changes that make older seasons a poor guide to points. It is the same
    # reasoning cold_start.py already uses to carry prior-season evidence
    # across the boundary.
    # _build_features derives avg_minutes_5gw and season_avg_minutes with
    # .shift(1) grouped by (player_id, season), then drops rows where they are
    # null -- so a player's FIRST appearance of a season never survives. One
    # gameweek in, EVERY row is a first appearance and the frame comes out
    # empty, for serving as much as for training.
    #
    # That is not a gap the in-season path can paper over: with no usable rows
    # there is nothing to fit and nothing to predict from. Hit live at GW2 of
    # 2026-27 on 2026-08-25, where run_agent died on `IndexError: single
    # positional indexer is out-of-bounds` inside train(), and then on
    # `Found array with 0 sample(s)` at serve time once training was widened.
    #
    # The season is past its cold start by the decision engine's definition --
    # a gameweek has been played and a squad exists to make transfers from --
    # so rebuilding a squad from scratch would be wrong. Only the NUMBERS are
    # unavailable. So take those from the cold start's prior-season evidence
    # and leave the decision path alone: existing squad, transfer optimiser,
    # chip logic, all unchanged. Resolves itself at GW3, when the second
    # played gameweek gives every row a predecessor.
    usable = len(_minutes_features(history))
    if usable == 0:
        logger.warning(
            "%s has played gameweeks but none survive feature-building (%d rows "
            "in, 0 out) -- too early for rolling features, which need a prior "
            "gameweek within the season. Projecting GWs %s from prior-season "
            "evidence instead; the decision path is unchanged.",
            season, len(history), target_gws,
        )
        projections_df, _, _ = cold_start_projections(
            season, target_gw=target_gws[0], horizon=horizon,
        )
        if persist:
            persist_projections(projections_df)
        return projections_df

    if usable < MIN_CURRENT_SEASON_TRAINING_ROWS:
        logger.warning(
            "Only %d of %d current-season rows survive feature-building (need "
            "%d) -- too early in %s to train the minutes model on it alone. "
            "Training on all available history instead.",
            usable, len(history), MIN_CURRENT_SEASON_TRAINING_ROWS, season,
        )
        min_model = train_minutes(save=False, fast=True)
    else:
        min_model = train_minutes(df_override=history, save=False, fast=True)
    fixture_context = _build_live_fixture_context(season, target_gws)
    match_odds = _load_live_match_odds(season, target_gws)
    defcon_events = assemble.load_defcon_events(season)
    defcon_field_shares = assemble.compute_defcon_field_shares(season)

    projections_df = assemble.assemble_gw_projections(
        history=history,
        all_stats=fixture_context,
        minutes_model=min_model,
        target_gw=next_gw,
        horizon=horizon,
        match_odds=match_odds,
        defcon_events=defcon_events,
        defcon_field_shares=defcon_field_shares,
        n_scenarios=n_scenarios,
        seed=seed,
        persist_samples=persist,  # P3-1: real teammate covariance for Phase 3
        season=season,
        # §2: fixture differentiation past the priced window. Odds cover the
        # next week or two; this horizon runs to five.
        strength_rel=assemble.load_team_strength_rel(season),
        # §20 follow-up: early in a season a rolling rate rests on one or two
        # matches. Lean on last season's per-match rates until this one has
        # enough of its own to speak for itself.
        prior_rates=assemble.load_prior_season_rates(prior_season_of(season)),
    )

    if not projections_df.empty:
        all_players = _get_all_players()
        unavailable = set(
            all_players.loc[all_players["status"].isin(["i", "u", "s"]), "id"].astype(int)
        )
        if unavailable:
            mask = projections_df["player_id"].isin(unavailable)
            projections_df.loc[mask, ["xpts", "xpts_mean", "xpts_var", "start_probability"]] = 0.0
        # Injury-severity discount (2026-07-30) — after the hard-unavailable
        # zeroing above (so an i/u/s player stays at exactly 0, not
        # re-scaled by their own severity on top of it) and before curse
        # shrinkage, same ordering rationale as the zeroing itself.
        projections_df = _apply_injury_severity_discount(projections_df, all_players)
        # Hand-entered rotation-risk ceilings (2026-08-18). Applied here as
        # well as in the cold start, and in the same position, because the two
        # paths must not disagree about who is nailed on: an override that
        # shaped the initial squad and then silently stopped applying in
        # gameweek 2 would have the transfer planner undo it.
        projections_df = apply_rotation_risk(projections_df, load_start_probability_caps())
        # Optimiser's-curse correction (2026-07-28) — after the unavailable-
        # player zeroing above, so a genuinely unavailable player (xpts=0)
        # doesn't get shrunk back up toward its group mean.
        if OPTIMISER.curse_shrinkage_enabled:
            projections_df = assemble.apply_curse_shrinkage(projections_df, all_players)
        projections_df["created_at"] = datetime.utcnow()

    if persist:
        persist_projections(projections_df)

    logger.info(
        "Projections complete: %d player-GW rows for GWs %s",
        len(projections_df), target_gws,
    )
    return projections_df


def _estimate_cs_probability(team_id: int, position: str, gw: int) -> float:
    if position not in ("GKP", "DEF"):
        return 0.0

    db = get_session()
    try:
        query = text("""
            SELECT fo.home_cs_prob, fo.away_cs_prob, f.team_h_id, f.team_a_id
            FROM fixture_odds fo
            JOIN fixtures f ON f.id = fo.fixture_id
            WHERE f.gameweek = :gw
              AND (f.team_h_id = :team_id OR f.team_a_id = :team_id)
            LIMIT 1
        """)
        row = db.execute(query, {"gw": gw, "team_id": team_id}).fetchone()
        if not row:
            return 0.25

        home_cs, away_cs, team_h, team_a = row
        if team_id == team_h:
            return float(home_cs or 0.25)
        else:
            return float(away_cs or 0.25)
    finally:
        db.close()


def persist_projections(df: pd.DataFrame) -> None:
    """Persists the P10 MC assembly's output.

    ``cs_probability`` used to be left at its column default (0.0) on every
    row ever written, because assemble.py had each player's clean-sheet
    outcome internally (for the BPS simulator) but never surfaced it. Fixed
    2026-08-16: it is exp(-λ_opponent) x P(60+ minutes), closed-form from the
    Poisson the sampler already draws from. The cold-start frame has no λ, so
    it legitimately omits the column and falls back to the default."""
    # Stamped here rather than required of the caller: the cold-start frame
    # has no created_at column, and the read side keys "latest run" on it.
    created_at = datetime.utcnow()
    db = get_session()
    try:
        for _, row in df.iterrows():
            stmt = (
                insert(PlayerProjection)
                .values(
                    player_id=int(row["player_id"]),
                    gameweek=int(row["gameweek"]),
                    xpts=float(row["xpts"]),
                    xpts_mean=float(row.get("xpts_mean", row["xpts"])),
                    xpts_var=float(row.get("xpts_var", 0.0)),
                    start_probability=float(row["start_probability"]),
                    cs_probability=float(row.get("cs_probability", 0.0) or 0.0),
                    created_at=row.get("created_at") or created_at,
                )
                .on_conflict_do_nothing()
            )
            db.execute(stmt)
        db.commit()
        logger.info("Persisted %d projection rows", len(df))
    finally:
        db.close()


def get_latest_projections(gw: int | None = None, horizon: int = 1) -> pd.DataFrame:
    """Latest persisted projections for ``horizon`` gameweeks starting at
    ``gw`` (default: the next gameweek).

    ``horizon`` (P1.1, 2026-08-16, plan/decision-engine-recovery-plan.md):
    ``run_projections`` builds and persists
    ``OPTIMISER.transfer_planning_horizon_gws`` gameweeks, but this function
    used to return exactly ONE, unconditionally. Every live consumer
    therefore saw a single-gameweek frame: ``evaluate_transfers`` computed
    ``H = 1`` and its whole multi-period structure (free-transfer carry,
    ``ft_terminal_value``, planning a move for next week) was unreachable;
    ``recommend_chip`` compared a one-gameweek wildcard gain against a
    five-gameweek threshold; and ``_run_decision_cycle``'s blank-gameweek
    count treated every missing row as a blank. The backtest never hit any
    of this because it passes its own full multi-gameweek frame straight to
    the optimisers.

    Defaults to 1 so the callers that genuinely want a single gameweek
    (site export, the dashboard squad page) are unchanged; the decision
    engine and DGW coverage pass a real horizon.
    """
    db = get_session()
    try:
        _, next_gw = _get_current_and_next_gw()
        target_gw = gw or next_gw
        last_gw = target_gw + max(1, horizon) - 1

        query = text("""
            SELECT
                pp.player_id,
                pp.gameweek,
                pp.xpts,
                pp.xpts_mean,
                pp.xpts_var,
                pp.start_probability,
                pp.cs_probability,
                pp.created_at,
                p.web_name,
                p.position,
                p.team_id,
                p.now_cost,
                p.status,
                p.selected_by_percent
            FROM player_projections pp
            JOIN players p ON p.id = pp.player_id
            WHERE pp.gameweek >= :gw AND pp.gameweek <= :last_gw
              AND pp.created_at = (
                  SELECT MAX(created_at) FROM player_projections
                  WHERE player_id = pp.player_id AND gameweek = pp.gameweek
              )
            ORDER BY pp.gameweek, pp.xpts DESC
        """)
        df = pd.read_sql(query, db.bind, params={"gw": target_gw, "last_gw": last_gw})
        return df
    finally:
        db.close()
