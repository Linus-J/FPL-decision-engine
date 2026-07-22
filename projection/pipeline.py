import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert

from config.strategy import DGW, OPTIMISER
from data.db import get_session
from data.models import Gameweek, Player, PlayerProjection
from projection import cs_model, minutes_model, points_model

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


def _load_player_recent_stats(season: str = "2026-27") -> pd.DataFrame:
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
                s.value,
                p.position,
                p.team_id,
                p.now_cost,
                p.ict_index,
                p.influence,
                p.creativity,
                p.threat,
                p.form,
                p.selected_by_percent,
                p.status,
                p.chance_of_playing_next_round,
                COALESCE(x.xg, 0)          AS xg,
                COALESCE(x.xa, 0)          AS xa,
                COALESCE(x.xgi, 0)         AS xgi,
                COALESCE(x.npxg, 0)        AS npxg,
                COALESCE(x.shots, 0)       AS shots,
                COALESCE(x.key_passes, 0)  AS key_passes
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            LEFT JOIN player_xg_stats x
                ON x.player_id = s.player_id
                AND x.gameweek  = s.gameweek
                AND x.season    = s.season
            WHERE s.season = :season
            ORDER BY s.player_id, s.gameweek
        """)
        return pd.read_sql(query, db.bind, params={"season": season})
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


def _get_team_fixture_count(gw: int) -> dict[int, int]:
    db = get_session()
    try:
        query = text("""
            SELECT team_h_id AS team_id, COUNT(*) AS fixtures FROM fixtures WHERE gameweek = :gw GROUP BY team_h_id
            UNION ALL
            SELECT team_a_id AS team_id, COUNT(*) AS fixtures FROM fixtures WHERE gameweek = :gw GROUP BY team_a_id
        """)
        rows = db.execute(query, {"gw": gw}).fetchall()
        counts: dict[int, int] = {}
        for team_id, cnt in rows:
            counts[team_id] = counts.get(team_id, 0) + cnt
        return counts
    finally:
        db.close()


def _apply_dgw_bgw_multipliers(
    xpts: float,
    gw: int,
    team_id: int,
    dgw_gws: set[int],
    bgw_gws: set[int],
    fixture_counts: dict[int, int],
) -> float:
    if gw in bgw_gws and fixture_counts.get(team_id, 1) == 0:
        return xpts * DGW.bgw_xpts_multiplier
    if gw in dgw_gws and fixture_counts.get(team_id, 1) > 1:
        return xpts * DGW.dgw_xpts_multiplier
    return xpts


def _precompute_cs_probabilities(
    all_players: pd.DataFrame,
    target_gws: list[int],
    season: str,
    model,
    recent_stats: pd.DataFrame,
) -> dict[tuple[int, str, int], float]:
    from projection.features import FDR_FEATURE_COLS, add_fdr_features, load_fixture_difficulty

    defender_positions = {"GKP", "DEF"}
    def_players = all_players[all_players["position"].isin(defender_positions)].copy()

    if def_players.empty or recent_stats.empty:
        return {}

    team_ids = def_players["team_id"].unique().tolist()

    team_stats = recent_stats[
        recent_stats["player_id"].isin(def_players["id"])
    ].copy()

    fdr_df = load_fixture_difficulty(season=season)

    rows = []
    for team_id in team_ids:
        team_history = team_stats[team_stats["team_id"] == team_id].sort_values("gameweek")

        cs_history = (team_history["clean_sheets"] > 0).tolist()
        gc_history = team_history["goals_conceded"].tolist()

        cs_rate_3 = float(sum(cs_history[-3:]) / max(len(cs_history[-3:]), 1))
        cs_rate_5 = float(sum(cs_history[-5:]) / max(len(cs_history[-5:]), 1))
        gc_rate_3 = float(sum(gc_history[-3:]) / max(len(gc_history[-3:]), 1))
        gc_rate_5 = float(sum(gc_history[-5:]) / max(len(gc_history[-5:]), 1))

        team_fdr = fdr_df[fdr_df["player_id"].isin(
            def_players[def_players["team_id"] == team_id]["id"].tolist()
        )]

        for gw in target_gws:
            fdr_row = team_fdr[team_fdr["gameweek"] == gw]
            is_home = float(fdr_row["is_home"].iloc[0]) if not fdr_row.empty else 0.5
            opp_att = float(fdr_row["opp_attack_strength"].iloc[0]) if not fdr_row.empty else 1200.0
            own_def = float(fdr_row["own_defence_strength"].iloc[0]) if not fdr_row.empty else 1200.0
            def_vs_att = own_def / max(opp_att, 1)

            for pos in ("GKP", "DEF"):
                rows.append({
                    "team_id": team_id,
                    "position": pos,
                    "gameweek": gw,
                    "team_cs_rate_3gw": cs_rate_3,
                    "team_cs_rate_5gw": cs_rate_5,
                    "team_gc_rate_3gw": gc_rate_3,
                    "team_gc_rate_5gw": gc_rate_5,
                    "pos_GKP": 1.0 if pos == "GKP" else 0.0,
                    "pos_DEF": 1.0 if pos == "DEF" else 0.0,
                    "is_home": is_home,
                    "opp_attack_strength": opp_att,
                    "own_defence_strength": own_def,
                    "defence_vs_attack": def_vs_att,
                })

    if not rows:
        return {}

    from projection.cs_model import FEATURE_COLS as CS_FEATURE_COLS
    batch_df = pd.DataFrame(rows)
    X = batch_df[CS_FEATURE_COLS].astype(float)
    probs = model.predict_proba(X)[:, 1]

    return {
        (int(r["team_id"]), r["position"], int(r["gameweek"])): float(p)
        for r, p in zip(rows, probs)
    }


def run_projections(
    season: str = "2026-27",
    horizon: int | None = None,
    persist: bool = True,
) -> pd.DataFrame:
    horizon = horizon or OPTIMISER.transfer_planning_horizon_gws
    _, next_gw = _get_current_and_next_gw()
    target_gws = list(range(next_gw, next_gw + horizon))

    dgw_gws = _get_dgw_gameweeks(horizon)
    bgw_gws = _get_bgw_gameweeks(horizon)

    logger.info(
        "Running projections for GWs %s (DGWs: %s, BGWs: %s)",
        target_gws, sorted(dgw_gws), sorted(bgw_gws),
    )

    min_model = minutes_model.load()
    pts_model = points_model.load()
    cs_ml_model = cs_model.load()

    recent_stats = _load_player_recent_stats(season)
    all_players = _get_all_players()

    if recent_stats.empty:
        logger.warning("No current-season stats found — projections will rely on player metadata only")

    cs_lookup = _precompute_cs_probabilities(
        all_players=all_players,
        target_gws=target_gws,
        season=season,
        model=cs_ml_model,
        recent_stats=recent_stats,
    )

    start_prob_map = minutes_model.predict_batch(recent_stats, min_model)
    xpts_map = points_model.predict_batch(recent_stats, pts_model)

    fixture_counts_by_gw = {gw: _get_team_fixture_count(gw) for gw in target_gws}

    rows: list[dict] = []

    for _, player in all_players.iterrows():
        player_id = int(player["id"])
        team_id = int(player["team_id"])
        position = player["position"]

        start_prob = start_prob_map.get(player_id, 0.5)
        base_xpts = xpts_map.get(player_id, 0.0)

        if player["status"] in ("i", "u", "s"):
            start_prob = 0.0
            base_xpts = 0.0
        elif player["status"] == "d":
            cop = (player["chance_of_playing_next_round"] or 50) / 100.0
            start_prob *= cop
            base_xpts *= cop

        for gw in target_gws:
            adjusted_xpts = _apply_dgw_bgw_multipliers(
                base_xpts, gw, team_id, dgw_gws, bgw_gws, fixture_counts_by_gw[gw]
            )

            cs_prob = cs_lookup.get((team_id, position, gw), 0.0)

            rows.append({
                "player_id": player_id,
                "gameweek": gw,
                "xpts": round(adjusted_xpts, 4),
                "start_probability": round(start_prob, 4),
                "cs_probability": round(cs_prob, 4),
                "created_at": datetime.utcnow(),
            })

    projections_df = pd.DataFrame(rows)

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
    db = get_session()
    try:
        for _, row in df.iterrows():
            stmt = (
                insert(PlayerProjection)
                .values(
                    player_id=int(row["player_id"]),
                    gameweek=int(row["gameweek"]),
                    xpts=float(row["xpts"]),
                    start_probability=float(row["start_probability"]),
                    cs_probability=float(row["cs_probability"]),
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
