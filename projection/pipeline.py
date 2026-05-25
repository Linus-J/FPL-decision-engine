import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert

from config.strategy import DGW, OPTIMISER
from data.db import get_session
from data.models import Gameweek, Player, PlayerProjection
from projection import minutes_model, points_model

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


def _get_dgw_gameweeks(lookahead: int) -> set[int]:
    db = get_session()
    try:
        _, next_gw = _get_current_and_next_gw()
        dgw_gws = (
            db.query(Gameweek.id)
            .filter(
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
        bgw_gws = (
            db.query(Gameweek.id)
            .filter(
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

    recent_stats = _load_player_recent_stats(season)
    all_players = _get_all_players()

    if recent_stats.empty:
        logger.warning("No current-season stats found — projections will rely on player metadata only")

    rows: list[dict] = []

    for _, player in all_players.iterrows():
        player_id = int(player["id"])
        team_id = int(player["team_id"])
        position = player["position"]

        player_stats = recent_stats[recent_stats["player_id"] == player_id].copy()

        if player_stats.empty:
            player_stats = pd.DataFrame([{
                "player_id": player_id,
                "gameweek": 0,
                "season": season,
                "minutes": 0,
                "total_points": 0,
                "goals_scored": 0,
                "assists": 0,
                "clean_sheets": 0,
                "goals_conceded": 0,
                "saves": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "bonus": 0,
                "bps": 0,
                "value": player["now_cost"],
                "position": position,
                "team_id": team_id,
                "now_cost": player["now_cost"],
                "ict_index": player["ict_index"],
                "influence": player["influence"],
                "creativity": player["creativity"],
                "threat": player["threat"],
                "form": player["form"],
                "selected_by_percent": player["selected_by_percent"],
                "status": player["status"],
                "chance_of_playing_next_round": player["chance_of_playing_next_round"],
                "xg": 0, "xa": 0, "xgi": 0, "npxg": 0, "shots": 0, "key_passes": 0,
            }])

        try:
            start_prob_series = minutes_model.predict_start_probabilities(player_stats, min_model)
            start_prob = float(start_prob_series.iloc[-1]) if not start_prob_series.empty else 0.5

            xpts_series = points_model.predict_points(player_stats, pts_model)
            base_xpts = float(xpts_series.iloc[-1]) if not xpts_series.empty else 0.0
        except Exception as exc:
            logger.debug("Projection failed for player %d: %s", player_id, exc)
            start_prob = 0.5
            base_xpts = 0.0

        if player["status"] in ("i", "u", "s"):
            start_prob = 0.0
            base_xpts = 0.0
        elif player["status"] == "d":
            cop = (player["chance_of_playing_next_round"] or 50) / 100.0
            start_prob *= cop
            base_xpts *= cop

        for gw in target_gws:
            fixture_counts = _get_team_fixture_count(gw)
            adjusted_xpts = _apply_dgw_bgw_multipliers(
                base_xpts, gw, team_id, dgw_gws, bgw_gws, fixture_counts
            )

            cs_prob = _estimate_cs_probability(team_id, position, gw)

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
            ORDER BY pp.xpts DESC
        """)
        df = pd.read_sql(query, db.bind, params={"gw": target_gw})
        return df
    finally:
        db.close()
