import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import pandas as pd
from sqlalchemy import text

from config.strategy import CHIP_TIMING, CHIPS, SQUAD
from data.db import get_session
from optimiser.squad import optimise_squad

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_wc_half_boundary() -> int:
    db = get_session()
    try:
        total = db.execute(text("SELECT COUNT(*) FROM gameweeks")).scalar() or 38
        return total // 2
    except Exception:
        return CHIPS.wildcard_first_half_deadline_gw
    finally:
        db.close()


class Chip(str, Enum):
    WILDCARD = "wildcard"
    FREE_HIT = "freehit"
    BENCH_BOOST = "bboost"
    TRIPLE_CAPTAIN = "3xc"


@dataclass
class ChipRecommendation:
    chip: Chip | None
    reason: str
    expected_gain: float


def chips_used_this_season(decision_log: pd.DataFrame) -> set[Chip]:
    if decision_log.empty or "chip_played" not in decision_log.columns:
        return set()
    used = decision_log["chip_played"].dropna().unique()
    return {Chip(c) for c in used if c}


def _wildcards_remaining(used: set[Chip], current_gw: int) -> int:
    wc_uses = sum(1 for c in used if c == Chip.WILDCARD)
    half_boundary = _get_wc_half_boundary()
    if current_gw <= half_boundary:
        available = 1
    else:
        first_half_used = min(wc_uses, 1)
        available = 2 - first_half_used - max(0, wc_uses - 1)
    return max(0, available - wc_uses)


def recommend_chip(
    current_gw: int,
    current_squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    available_budget: float,
    free_transfers: int,
    chips_used: set[Chip],
    bench_xpts: float | None = None,
    dgw_gws: set[int] | None = None,
    bgw_affected_count: int = 0,
    squad_age_gws: int = 99,
) -> ChipRecommendation:
    dgw_gws = dgw_gws or set()
    horizon = CHIP_TIMING.wildcard_eval_horizon_gws

    gws = sorted(projections["gameweek"].unique())[:horizon]
    current_xpts = float(
        projections[
            projections["gameweek"].isin(gws) & projections["player_id"].isin(current_squad_ids)
        ]["xpts"].sum()
    )

    if Chip.TRIPLE_CAPTAIN not in chips_used:
        tc_gain = _evaluate_triple_captain(current_squad_ids, projections, current_gw)
        if tc_gain >= CHIP_TIMING.triple_captain_min_gain:
            logger.info("TC recommended: gain=%.2f", tc_gain)
            return ChipRecommendation(Chip.TRIPLE_CAPTAIN, f"TC gain {tc_gain:.1f} xPts", tc_gain)

    if Chip.BENCH_BOOST not in chips_used and bench_xpts is not None:
        dgw_active = bool(dgw_gws and current_gw in dgw_gws)
        if dgw_active and bench_xpts >= CHIP_TIMING.bench_boost_min_bench_xpts:
            logger.info("BB recommended: bench_xpts=%.2f in DGW%d", bench_xpts, current_gw)
            return ChipRecommendation(
                Chip.BENCH_BOOST,
                f"DGW bench xPts {bench_xpts:.1f} exceeds threshold",
                bench_xpts,
            )

    if Chip.FREE_HIT not in chips_used and bgw_affected_count >= 5:
        fh_solution = optimise_squad(
            projections=projections,
            players=players,
            budget=available_budget,
            horizon=1,
        )
        fh_gws = sorted(projections["gameweek"].unique())[:1]
        fh_xpts = float(
            projections[
                projections["gameweek"].isin(fh_gws)
                & projections["player_id"].isin(fh_solution.squad["id"].tolist())
            ]["xpts"].sum()
        )
        current_gw_xpts = float(
            projections[
                (projections["gameweek"] == current_gw)
                & projections["player_id"].isin(current_squad_ids)
            ]["xpts"].sum()
        )
        gain = fh_xpts - current_gw_xpts
        if gain >= CHIP_TIMING.free_hit_single_gw_gain_threshold:
            logger.info("FH recommended: gain=%.2f (BGW blanks=%d)", gain, bgw_affected_count)
            return ChipRecommendation(Chip.FREE_HIT, f"BGW free hit gain {gain:.1f} xPts", gain)

    wc_remaining = _wildcards_remaining(chips_used, current_gw)
    if wc_remaining > 0 and squad_age_gws >= CHIP_TIMING.wildcard_min_managed_gws:
        wc_solution = optimise_squad(
            projections=projections,
            players=players,
            budget=available_budget,
            horizon=horizon,
            current_squad_ids=current_squad_ids,
            free_transfers=15,
        )
        wc_gws_xpts = float(
            projections[
                projections["gameweek"].isin(gws)
                & projections["player_id"].isin(wc_solution.squad["id"].tolist())
            ]["xpts"].sum()
        )
        wc_gain = wc_gws_xpts - current_xpts
        if wc_gain >= CHIP_TIMING.wildcard_pts_gain_threshold:
            logger.info("WC recommended: gain=%.2f over %d GWs", wc_gain, horizon)
            return ChipRecommendation(
                Chip.WILDCARD,
                f"WC gain {wc_gain:.1f} xPts over {horizon} GWs",
                wc_gain,
            )

    return ChipRecommendation(None, "No chip threshold met", 0.0)


def _evaluate_triple_captain(
    squad_ids: list[int],
    projections: pd.DataFrame,
    gw: int,
) -> float:
    gw_proj = projections[
        (projections["gameweek"] == gw) & projections["player_id"].isin(squad_ids)
    ].sort_values("xpts", ascending=False)

    if len(gw_proj) < 2:
        return 0.0

    best_xpts = float(gw_proj.iloc[0]["xpts"])
    second_xpts = float(gw_proj.iloc[1]["xpts"])

    tc_gain = best_xpts - second_xpts
    return tc_gain
