import json
import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import pandas as pd
from sqlalchemy import text

from config.strategy import CHIP_TIMING, CHIPS, SQUAD
from data.db import get_session
from optimiser.chip_scenarios import gain_distribution, load_scenario_totals
from optimiser.squad import optimise_squad

logger = logging.getLogger(__name__)


def _clears_threshold(
    point_value: float,
    threshold: float,
    scenario_values: pd.Series,
    min_probability: float,
) -> bool:
    """The old rule was ``point_value >= threshold`` (point-estimate xpts vs
    a fixed constant). P3-5: when real MC scenarios (P3-1) exist for the
    decision, replace it with the actual ``P(scenario_value >= threshold) >=
    min_probability`` — a genuine probability-of-payoff, not just whether the
    mean clears the bar, so a reliable small gain can outrank a
    coin-flip gain with the same mean. Falls back to the point-estimate rule
    when no scenario data is available (cold start, or the backtest
    walk-forward, which never persists samples per P3-1)."""
    if scenario_values.empty:
        return point_value >= threshold
    return float((scenario_values >= threshold).mean()) >= min_probability


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


@lru_cache(maxsize=1)
def _get_total_gws() -> int:
    db = get_session()
    try:
        return db.execute(text("SELECT COUNT(*) FROM gameweeks")).scalar() or 38
    except Exception:
        return 38
    finally:
        db.close()


def _current_half_expiry_gw(current_gw: int) -> int:
    """The last gameweek ``current_gw``'s half's chips are still usable —
    the half boundary itself for the first half, the season's last GW for
    the second (see ``_chip_uses_remaining``'s no-carryover rule)."""
    half_boundary = _get_wc_half_boundary()
    return half_boundary if current_gw <= half_boundary else _get_total_gws()


def _panic_shrink(current_gw: int) -> float:
    """1.0 outside the panic window; linearly decays to
    ``CHIP_TIMING.panic_threshold_shrink`` as ``current_gw`` approaches its
    half's expiry, so a real-but-marginal chip opportunity is far more
    likely to clear its threshold before evaporating unused. See
    ``ChipTimingThresholds.panic_window_gws`` for the rationale."""
    gws_remaining = _current_half_expiry_gw(current_gw) - current_gw
    window = CHIP_TIMING.panic_window_gws
    if gws_remaining < 0 or gws_remaining >= window:
        return 1.0
    frac = gws_remaining / window
    return CHIP_TIMING.panic_threshold_shrink + frac * (1.0 - CHIP_TIMING.panic_threshold_shrink)


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


def chips_used_this_season(decision_log: pd.DataFrame) -> list[tuple[Chip, int]]:
    """(chip, gameweek) for every chip actually played this season.

    Real bug found 2026-07-28: this used to read a ``chip_played`` column
    decision_log never had (chip decisions are logged as
    ``decision_type="chip"`` with the chip name inside the JSON ``details``
    column — see ``agent/decision_engine.py::_log_decision`` call sites) —
    ``decision_log["chip_played"]`` would ``KeyError`` the first time this
    ran against any real accumulated log, silently masked pre-launch only
    because an EMPTY log short-circuits before touching the column. Also
    returns a plain list, not a set: a chip can legitimately be played
    twice a season (see ``_chip_uses_remaining``), and a set can't
    represent that.
    """
    if decision_log.empty or "decision_type" not in decision_log.columns:
        return []
    chip_rows = decision_log[decision_log["decision_type"] == "chip"]
    used: list[tuple[Chip, int]] = []
    for _, row in chip_rows.iterrows():
        try:
            chip = Chip(json.loads(row["details"])["chip"])
        except (TypeError, ValueError, KeyError):
            continue
        used.append((chip, int(row["gameweek"])))
    return used


def _chip_uses_remaining(chip: Chip, used: list[tuple[Chip, int]], current_gw: int) -> int:
    """FPL 2025/26+ rule (real bug found 2026-07-28 — the user's own review
    flagged only one wildcard ever getting played across a full season):
    each of the 4 chips gets exactly 1 use per half of the season (2
    total), with NO carryover — an unused first-half chip is lost once the
    GW19-deadline boundary passes, not banked for the second half.
    Previously only the wildcard had this half-aware accounting
    (``_wildcards_remaining``); the other three chips used a naive
    ``chip not in chips_used`` check against a de-duplicating ``set``,
    which structurally cannot represent "used once already, one more
    available in the other half" — worse, feeding a `set` into a
    multiplicity count silently caps every chip at one use for the whole
    season regardless of what the config says."""
    half_boundary = _get_wc_half_boundary()
    if current_gw <= half_boundary:
        uses_this_half = sum(1 for c, gw in used if c == chip and gw <= half_boundary)
    else:
        uses_this_half = sum(1 for c, gw in used if c == chip and gw > half_boundary)
    return max(0, 1 - uses_this_half)


def recommend_chip(
    current_gw: int,
    current_squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    available_budget: float,
    free_transfers: int,
    chips_used: list[tuple[Chip, int]],
    bench_xpts: float | None = None,
    dgw_gws: set[int] | None = None,
    bgw_affected_count: int = 0,
    squad_age_gws: int = 99,
    season: str | None = None,
) -> ChipRecommendation:
    dgw_gws = dgw_gws or set()
    horizon = CHIP_TIMING.wildcard_eval_horizon_gws

    gws = sorted(projections["gameweek"].unique())[:horizon]
    current_xpts = float(
        projections[
            projections["gameweek"].isin(gws) & projections["player_id"].isin(current_squad_ids)
        ]["xpts"].sum()
    )

    if _chip_uses_remaining(Chip.TRIPLE_CAPTAIN, chips_used, current_gw) > 0:
        tc_gain, best_id, second_id = _evaluate_triple_captain(
            current_squad_ids, projections, current_gw
        )
        tc_scenarios = pd.Series(dtype=float)
        if season is not None and best_id is not None and second_id is not None:
            tc_scenarios = gain_distribution(season, current_gw, [best_id], [second_id])
        if _clears_threshold(
            tc_gain, CHIP_TIMING.triple_captain_min_gain * _panic_shrink(current_gw), tc_scenarios,
            CHIP_TIMING.triple_captain_min_payoff_probability,
        ):
            logger.info("TC recommended: gain=%.2f", tc_gain)
            return ChipRecommendation(Chip.TRIPLE_CAPTAIN, f"TC gain {tc_gain:.1f} xPts", tc_gain)

    bb_remaining = _chip_uses_remaining(Chip.BENCH_BOOST, chips_used, current_gw) > 0
    if bb_remaining and bench_xpts is not None:
        dgw_active = bool(dgw_gws and current_gw in dgw_gws)
        if dgw_active:
            bb_scenarios = pd.Series(dtype=float)
            if season is not None:
                bench_ids = _bench_player_ids(current_squad_ids, projections, current_gw)
                if bench_ids:
                    bb_scenarios = load_scenario_totals(season, current_gw, bench_ids)
            if _clears_threshold(
                bench_xpts,
                CHIP_TIMING.bench_boost_min_bench_xpts * _panic_shrink(current_gw),
                bb_scenarios,
                CHIP_TIMING.bench_boost_min_payoff_probability,
            ):
                logger.info("BB recommended: bench_xpts=%.2f in DGW%d", bench_xpts, current_gw)
                return ChipRecommendation(
                    Chip.BENCH_BOOST,
                    f"DGW bench xPts {bench_xpts:.1f} exceeds threshold",
                    bench_xpts,
                )

    fh_remaining = _chip_uses_remaining(Chip.FREE_HIT, chips_used, current_gw) > 0
    dgw_active_for_fh = bool(dgw_gws and current_gw in dgw_gws)
    # 2026-07-30 (user's own review: "the free hit is usually handy during
    # double game weeks where it is not worth triple captaining") -- Free
    # Hit used to only ever trigger on a BGW blank-filling rationale. A DGW
    # is the other classic real-world Free Hit case: load the WHOLE XI with
    # double-fixture players for one week without permanently restructuring
    # the squad, when no single player's TC gain (evaluated above) is big
    # enough on its own to justify Triple Captain instead.
    if fh_remaining and (bgw_affected_count >= 5 or dgw_active_for_fh):
        fh_solution = optimise_squad(
            projections=projections,
            players=players,
            budget=available_budget,
            horizon=1,
        )
        fh_gws = sorted(projections["gameweek"].unique())[:1]
        fh_squad_ids = fh_solution.squad["id"].tolist()
        fh_xpts = float(
            projections[
                projections["gameweek"].isin(fh_gws)
                & projections["player_id"].isin(fh_squad_ids)
            ]["xpts"].sum()
        )
        current_gw_xpts = float(
            projections[
                (projections["gameweek"] == current_gw)
                & projections["player_id"].isin(current_squad_ids)
            ]["xpts"].sum()
        )
        gain = fh_xpts - current_gw_xpts
        fh_scenarios = pd.Series(dtype=float)
        if season is not None and fh_gws == [current_gw]:
            fh_scenarios = gain_distribution(
                season, current_gw, fh_squad_ids, current_squad_ids
            )
        if _clears_threshold(
            gain,
            CHIP_TIMING.free_hit_single_gw_gain_threshold * _panic_shrink(current_gw),
            fh_scenarios,
            CHIP_TIMING.free_hit_min_payoff_probability,
        ):
            trigger = "DGW" if dgw_active_for_fh and bgw_affected_count < 5 else "BGW"
            logger.info(
                "FH recommended: gain=%.2f (%s, blanks=%d)", gain, trigger, bgw_affected_count
            )
            return ChipRecommendation(
                Chip.FREE_HIT, f"{trigger} free hit gain {gain:.1f} xPts", gain
            )

    wc_remaining = _chip_uses_remaining(Chip.WILDCARD, chips_used, current_gw)
    if wc_remaining > 0 and squad_age_gws >= CHIP_TIMING.wildcard_min_managed_gws:
        wc_solution = optimise_squad(
            projections=projections,
            players=players,
            budget=available_budget,
            horizon=horizon,
            current_squad_ids=current_squad_ids,
            free_transfers=15,
        )
        wc_squad_ids = wc_solution.squad["id"].tolist()
        wc_gws_xpts = float(
            projections[
                projections["gameweek"].isin(gws)
                & projections["player_id"].isin(wc_squad_ids)
            ]["xpts"].sum()
        )
        wc_gain = wc_gws_xpts - current_xpts
        wc_scenarios = pd.Series(dtype=float)
        if season is not None:
            wc_scenarios = gain_distribution(season, gws, wc_squad_ids, current_squad_ids)
        if _clears_threshold(
            wc_gain,
            CHIP_TIMING.wildcard_pts_gain_threshold * _panic_shrink(current_gw),
            wc_scenarios,
            CHIP_TIMING.wildcard_min_payoff_probability,
        ):
            logger.info("WC recommended: gain=%.2f over %d GWs", wc_gain, horizon)
            return ChipRecommendation(
                Chip.WILDCARD,
                f"WC gain {wc_gain:.1f} xPts over {horizon} GWs",
                wc_gain,
            )

    # Last resort (user's own words: "at worst the default behaviour is to
    # panic and use the triple captain on the last day before the chips
    # reset or the season ends"). Reached only once every threshold above
    # -- already shrunk by _panic_shrink for the whole panic window -- has
    # failed to clear even on this half's final gameweek. Triple Captain is
    # the one chip that's close to always worth SOMETHING (it just doubles
    # whatever captain pick this week's decision already makes), unlike
    # Free Hit/Bench Boost/Wildcard, which need a real structural
    # opportunity to be worth anything at all -- forcing one of those
    # instead could easily be a genuine net negative, forcing TC essentially
    # never is.
    if (
        current_gw == _current_half_expiry_gw(current_gw)
        and _chip_uses_remaining(Chip.TRIPLE_CAPTAIN, chips_used, current_gw) > 0
    ):
        tc_gain, best_id, _second_id = _evaluate_triple_captain(
            current_squad_ids, projections, current_gw
        )
        if best_id is not None:
            logger.info(
                "TC PANIC-forced at half/season expiry GW%d: gain=%.2f", current_gw, tc_gain
            )
            return ChipRecommendation(
                Chip.TRIPLE_CAPTAIN, f"Panic TC at expiry (gain {tc_gain:.1f} xPts)", tc_gain
            )

    return ChipRecommendation(None, "No chip threshold met", 0.0)


def _bench_player_ids(
    squad_ids: list[int], projections: pd.DataFrame, gw: int
) -> list[int]:
    """Approximates bench membership as "beyond the top 11 by xpts" among
    ``squad_ids`` for this gameweek — the same approximation
    ``decision_engine._bench_xpts`` already uses for the point-estimate
    ``bench_xpts`` this function's caller receives, so the scenario gate
    stays consistent with the metric it's gating."""
    gw_proj = projections[
        (projections["gameweek"] == gw) & projections["player_id"].isin(squad_ids)
    ].sort_values("xpts", ascending=False)
    if len(gw_proj) <= 11:
        return []
    return gw_proj.iloc[11:]["player_id"].tolist()


def _evaluate_triple_captain(
    squad_ids: list[int],
    projections: pd.DataFrame,
    gw: int,
) -> tuple[float, int | None, int | None]:
    gw_proj = projections[
        (projections["gameweek"] == gw) & projections["player_id"].isin(squad_ids)
    ].sort_values("xpts", ascending=False)

    if len(gw_proj) < 2:
        return 0.0, None, None

    best_xpts = float(gw_proj.iloc[0]["xpts"])
    second_xpts = float(gw_proj.iloc[1]["xpts"])
    best_id = int(gw_proj.iloc[0]["player_id"])
    second_id = int(gw_proj.iloc[1]["player_id"])

    tc_gain = best_xpts - second_xpts
    return tc_gain, best_id, second_id
