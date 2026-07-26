import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert

from config.strategy import OPTIMISER
from data.db import get_session
from data.models import Gameweek, PlayerProjection
from projection import assemble
from projection.minutes_model import train as train_minutes

logger = logging.getLogger(__name__)


def _get_current_and_next_gw() -> tuple[int, int]:
    db = get_session()
    try:
        current = db.query(Gameweek).filter(Gameweek.is_current == True).first()
        next_gw = db.query(Gameweek).filter(Gameweek.is_next == True).first()
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
                Gameweek.is_dgw == True,
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
                Gameweek.is_bgw == True,
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
                   form, ict_index, influence, creativity, threat
            FROM players
        """)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()


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
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
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
            _persist_projections(pd.DataFrame(columns=empty_cols))
        return pd.DataFrame(columns=empty_cols)

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
    )

    if not projections_df.empty:
        all_players = _get_all_players()
        unavailable = set(
            all_players.loc[all_players["status"].isin(["i", "u", "s"]), "id"].astype(int)
        )
        if unavailable:
            mask = projections_df["player_id"].isin(unavailable)
            projections_df.loc[mask, ["xpts", "xpts_mean", "xpts_var", "start_probability"]] = 0.0
        projections_df["created_at"] = datetime.utcnow()

    if persist:
        _persist_projections(projections_df)

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


def _persist_projections(df: pd.DataFrame) -> None:
    """Persists the P10 MC assembly's output. ``cs_probability`` is left at
    its column default (0.0) — assemble.py computes each player's
    clean-sheet component internally (P5) but doesn't currently surface it
    as a standalone output column; the only consumer is a reporting script
    (scripts/plot_analysis.py), not core decision logic, so this is a
    documented gap rather than a silent one, not a P3-0 blocker."""
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
                    created_at=row["created_at"],
                )
                .on_conflict_do_nothing()
            )
            db.execute(stmt)
        db.commit()
        logger.info("Persisted %d projection rows", len(df))
    finally:
        db.close()


def get_latest_projections(gw: int | None = None) -> pd.DataFrame:
    db = get_session()
    try:
        _, next_gw = _get_current_and_next_gw()
        target_gw = gw or next_gw

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
            WHERE pp.gameweek = :gw
              AND pp.created_at = (
                  SELECT MAX(created_at) FROM player_projections
                  WHERE player_id = pp.player_id AND gameweek = pp.gameweek
              )
            ORDER BY pp.xpts DESC
        """)
        df = pd.read_sql(query, db.bind, params={"gw": target_gw})
        return df
    finally:
        db.close()
