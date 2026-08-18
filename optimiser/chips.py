import json
import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import pandas as pd
from sqlalchemy import text

from config.strategy import CHIP_TIMING, CHIPS, ChipTimingThresholds, OptimiserConfig
from data.db import get_session
from optimiser.chip_scenarios import gain_distribution, load_scenario_totals
from optimiser.squad import optimise_squad
from optimiser.transfers import evaluate_transfers, squad_xpts

logger = logging.getLogger(__name__)

# Gameweeks in a normal Premier League season -- the fallback when the
# gameweeks table can't answer for a specific season (P3.10).
_DEFAULT_SEASON_GWS = 38


def _clears_threshold(
    point_value: float,
    threshold: float,
    scenario_values: pd.Series,
    min_probability: float,
) -> bool:
    """Both bars, as ``config.strategy`` has always specified: the
    point-estimate threshold, AND — where MC scenarios exist — a minimum
    ``P(gain >= 0)``.

    Fixed 2026-08-18 (engine review §14). This used to test
    ``P(scenario_value >= threshold) >= min_probability`` and to apply it
    *instead of* the point estimate, discarding ``point_value`` entirely
    whenever samples existed. Both halves diverged from the documented design,
    which reads: *"minimum P(gain >= 0) ... required, IN ADDITION to the
    point-estimate thresholds above"*.

    The effect was a far stricter gate than any of the constants intend.
    "Mean gain >= 25" and "60% chance of a gain >= 25" are wildly different
    tests, and for a right-skewed FPL distribution the second is much harder —
    a wildcard needing a 60% probability of a >=25-point five-gameweek gain
    would essentially never fire. It mattered doubly because samples exist in
    live serving and NEVER in the backtest (P3-1 does not persist them there),
    so the gate that ran in production was not the gate that ran in any run
    used to tune it.

    ``scenario_values`` is each chip's own gain distribution: an explicit
    ``gain_distribution`` for Wildcard and Free Hit, whose rebuilds can
    genuinely come out worse than the squad they replace; and the raw totals
    for Triple Captain and Bench Boost, for which the gain simply IS the total
    (one extra copy of the captain, or the bench's own points). The bar
    therefore rarely binds for those two, which is correct — their downside is
    bounded near zero by construction, not by luck.

    Falls back to the point estimate alone when no scenario data is available
    (cold start, or the backtest walk-forward).
    """
    if point_value < threshold:
        return False
    if scenario_values.empty:
        return True
    return float((scenario_values >= 0.0).mean()) >= min_probability


@lru_cache(maxsize=16)
def _get_wc_half_boundary(season: str | None = None) -> int:
    """Real bug found 2026-07-30 (user's own review: TC "should NEVER be
    left unplayed in both halves of the season" -- it never fired even at
    full panic strength). This query used to count ``gameweeks`` rows with
    NO season filter at all -- harmless when the table only ever held one
    season, but this project's own backfill covers 6 seasons at once
    (227 total rows), so the "half boundary" silently came out as 113
    (227 // 2) instead of 19. Every 2025-26 backtest gameweek (6-38) is
    ``<= 113``, so the code believed it was ALWAYS still in the first
    half, and both the no-carryover per-half chip cap and the panic/expiry
    logic could never see a real boundary crossing within the season at
    all. Scoping the count to the season actually being decided for fixes
    both at once."""
    # P3.10 (2026-08-16): with no season there is no safe query. Counting
    # `gameweeks` unscoped is what produced the original 227 // 2 = 113
    # boundary this docstring describes; falling back to the configured
    # first-half deadline is the same failure class, closed properly.
    if season is None:
        return CHIPS.wildcard_first_half_deadline_gw
    db = get_session()
    try:
        total = (
            db.execute(
                text("SELECT COUNT(*) FROM gameweeks WHERE season = :season"),
                {"season": season},
            ).scalar()
            or _DEFAULT_SEASON_GWS
        )
        return total // 2
    except Exception:
        return CHIPS.wildcard_first_half_deadline_gw
    finally:
        db.close()


@lru_cache(maxsize=16)
def _get_total_gws(season: str | None = None) -> int:
    # P3.10: same season-unscoped hazard as _get_wc_half_boundary -- an
    # unscoped COUNT(*) over a multi-season DB returns the total across every
    # season (227 here), not this season's 38.
    if season is None:
        return _DEFAULT_SEASON_GWS
    db = get_session()
    try:
        return (
            db.execute(
                text("SELECT COUNT(*) FROM gameweeks WHERE season = :season"),
                {"season": season},
            ).scalar()
            or _DEFAULT_SEASON_GWS
        )
    except Exception:
        return _DEFAULT_SEASON_GWS
    finally:
        db.close()


def _current_half_expiry_gw(current_gw: int, season: str | None = None) -> int:
    """The last gameweek ``current_gw``'s half's chips are still usable —
    the half boundary itself for the first half, the season's last GW for
    the second (see ``_chip_uses_remaining``'s no-carryover rule)."""
    half_boundary = _get_wc_half_boundary(season)
    return half_boundary if current_gw <= half_boundary else _get_total_gws(season)


def _panic_shrink(
    current_gw: int, season: str | None = None, chip_timing: ChipTimingThresholds | None = None
) -> float:
    """1.0 outside the panic window; linearly decays to
    ``chip_timing.panic_threshold_shrink`` as ``current_gw`` approaches its
    half's expiry, so a real-but-marginal chip opportunity is far more
    likely to clear its threshold before evaporating unused. See
    ``ChipTimingThresholds.panic_window_gws`` for the rationale.

    ``chip_timing`` (optional): overrides the global ``CHIP_TIMING`` for
    this call only; ``None`` is byte-for-byte identical to today's
    behaviour."""
    timing = chip_timing or CHIP_TIMING
    gws_remaining = _current_half_expiry_gw(current_gw, season) - current_gw
    window = timing.panic_window_gws
    if gws_remaining < 0 or gws_remaining >= window:
        return 1.0
    frac = gws_remaining / window
    return timing.panic_threshold_shrink + frac * (1.0 - timing.panic_threshold_shrink)


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
    # P1.8 (2026-08-16): de-duplicate on (chip, gameweek). A chip can only be
    # played ONCE in a given gameweek, but `_record_decision` appends a row
    # every run -- so re-running a gameweek's decision (a dry-run rehearsal, a
    # crash and retry, refining the squad as news lands) wrote N rows for the
    # same chip in the same gameweek, and `_chip_uses_remaining` counts rows.
    # Three reruns therefore consumed a chip that was never actually played
    # twice, permanently, for the rest of the half. Preserves order and still
    # represents the legitimate "used once per half" multiplicity, since those
    # are genuinely different gameweeks.
    return list(dict.fromkeys(used))


def _chip_uses_remaining(
    chip: Chip, used: list[tuple[Chip, int]], current_gw: int, season: str | None = None
) -> int:
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
    half_boundary = _get_wc_half_boundary(season)
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
    chip_timing: ChipTimingThresholds | None = None,
    config: OptimiserConfig | None = None,
    bank: float | None = None,
    purchase_prices: dict[int, float] | None = None,
) -> ChipRecommendation:
    """``bank``/``purchase_prices`` (optional, 2026-08-18): the real
    affordability ledger, forwarded to the wildcard's own
    ``evaluate_transfers`` call so the squad the chip is judged on is the squad
    that can actually be bought. Omitting them reproduces the old
    budget-cap-only behaviour exactly (see ``evaluate_transfers``).

    ``chip_timing``/``config`` (optional): override the global
    ``CHIP_TIMING``/``OPTIMISER`` for this call only — used by the
    simulation engine to vary chip-timing aggressiveness and risk posture
    per persona (the latter feeds the internal Free Hit/Wildcard scenario
    rebuilds via ``optimise_squad``). ``None`` (every real-bot call site) is
    byte-for-byte identical to reading the globals directly, as before
    these parameters existed."""
    timing = chip_timing or CHIP_TIMING
    dgw_gws = dgw_gws or set()
    horizon = timing.wildcard_eval_horizon_gws
    dgw_active_now = bool(dgw_gws and current_gw in dgw_gws)
    # Only within whatever short lookahead the caller's dgw_gws already
    # covers (typically the transfer-planning horizon) -- a real but
    # narrow-sighted approximation of "is a DGW coming up later this half",
    # not a full scan to the half boundary.
    dgw_visible_ahead = bool(
        dgw_gws and not dgw_active_now and any(g > current_gw for g in dgw_gws)
    )

    gws = sorted(projections["gameweek"].unique())[:horizon]

    def _try_tc() -> ChipRecommendation | None:
        if _chip_uses_remaining(Chip.TRIPLE_CAPTAIN, chips_used, current_gw, season) <= 0:
            return None
        tc_gain, best_id, second_id = _evaluate_triple_captain(
            current_squad_ids, projections, current_gw
        )
        tc_scenarios = pd.Series(dtype=float)
        if season is not None and best_id is not None:
            tc_scenarios = load_scenario_totals(season, current_gw, [best_id])
        threshold = timing.triple_captain_min_gain * _panic_shrink(current_gw, season, timing)
        if dgw_visible_ahead:
            # 2026-07-30 (user's own review): TC's own marginal value is
            # basically always positive (see _evaluate_triple_captain), so
            # the real question isn't "is it worth it" but "is THIS week
            # worth spending one of only 2 season uses on, versus a coming
            # DGW captain who'll likely return far more from two fixtures."
            # Raises the bar rather than blocking outright -- an
            # exceptional normal week can still justify using it now.
            threshold *= timing.triple_captain_dgw_wait_multiplier
        if not _clears_threshold(
            tc_gain, threshold, tc_scenarios, timing.triple_captain_min_payoff_probability
        ):
            return None
        logger.info("TC recommended: captain xPts=%.2f", tc_gain)
        return ChipRecommendation(Chip.TRIPLE_CAPTAIN, f"TC captain xPts {tc_gain:.1f}", tc_gain)

    def _try_bb() -> ChipRecommendation | None:
        if _chip_uses_remaining(Chip.BENCH_BOOST, chips_used, current_gw, season) <= 0:
            return None
        if bench_xpts is None or not dgw_active_now:
            return None
        bb_scenarios = pd.Series(dtype=float)
        if season is not None:
            bench_ids = _bench_player_ids(current_squad_ids, projections, current_gw)
            if bench_ids:
                bb_scenarios = load_scenario_totals(season, current_gw, bench_ids)
        threshold = timing.bench_boost_min_bench_xpts * _panic_shrink(current_gw, season, timing)
        if not _clears_threshold(
            bench_xpts, threshold, bb_scenarios, timing.bench_boost_min_payoff_probability
        ):
            return None
        logger.info("BB recommended: bench_xpts=%.2f in DGW%d", bench_xpts, current_gw)
        return ChipRecommendation(
            Chip.BENCH_BOOST, f"DGW bench xPts {bench_xpts:.1f} exceeds threshold", bench_xpts
        )

    def _try_fh() -> ChipRecommendation | None:
        if _chip_uses_remaining(Chip.FREE_HIT, chips_used, current_gw, season) <= 0:
            return None
        # 2026-07-30 (user's own review: "the free hit is usually handy
        # during double game weeks where it is not worth triple
        # captaining") -- Free Hit used to only ever trigger on a BGW
        # blank-filling rationale. A DGW is the other classic real-world
        # Free Hit case: load the WHOLE XI with double-fixture players for
        # one week without permanently restructuring the squad.
        if not (bgw_affected_count >= 5 or dgw_active_now):
            return None
        fh_solution = optimise_squad(
            projections=projections, players=players, budget=available_budget, horizon=1,
            config=config,
        )
        fh_gws = sorted(projections["gameweek"].unique())[:1]
        fh_squad_ids = fh_solution.squad["id"].tolist()
        # §13: a Free Hit plays ELEVEN, and doubles a captain. This used to sum
        # all fifteen and credit no captain, so the gain included four players
        # who would never take the field. `squad_xpts` is the project's single
        # definition of what a squad is worth; both sides must use it.
        fh_xpts = squad_xpts(fh_squad_ids, projections, horizon=1)
        current_gw_xpts = squad_xpts(current_squad_ids, projections, horizon=1)
        gain = fh_xpts - current_gw_xpts
        fh_scenarios = pd.Series(dtype=float)
        if season is not None and fh_gws == [current_gw]:
            fh_scenarios = gain_distribution(season, current_gw, fh_squad_ids, current_squad_ids)
        threshold = timing.free_hit_single_gw_gain_threshold * _panic_shrink(
            current_gw, season, timing
        )
        if not _clears_threshold(
            gain, threshold, fh_scenarios, timing.free_hit_min_payoff_probability
        ):
            return None
        trigger = "DGW" if dgw_active_now and bgw_affected_count < 5 else "BGW"
        logger.info("FH recommended: gain=%.2f (%s, blanks=%d)", gain, trigger, bgw_affected_count)
        return ChipRecommendation(Chip.FREE_HIT, f"{trigger} free hit gain {gain:.1f} xPts", gain)

    def _try_wc() -> ChipRecommendation | None:
        if _chip_uses_remaining(Chip.WILDCARD, chips_used, current_gw, season) <= 0:
            return None
        if squad_age_gws < timing.wildcard_min_managed_gws:
            return None
        # §12 (2026-08-18): decide with the SAME optimiser that will execute
        # it. A played wildcard is run through
        # `evaluate_transfers(wildcard_active=True)` by the decision engine, so
        # evaluating it here with `optimise_squad` measured the gain of a
        # rebuild that would never actually be built — a different objective, no
        # bank or purchase-price constraint, and no multi-period view. The chip
        # fired on one number and delivered another.
        #
        # `horizon` is passed explicitly because evaluate_transfers otherwise
        # plans over transfer_planning_horizon_gws (3) while the wildcard
        # threshold is written for wildcard_eval_horizon_gws (5).
        wc_plan = evaluate_transfers(
            current_squad_ids=current_squad_ids,
            projections=projections,
            players=players,
            free_transfers=free_transfers,
            available_budget=available_budget,
            wildcard_active=True,
            config=config,
            bank=bank,
            purchase_prices=purchase_prices,
            horizon=horizon,
        )
        # xpts_gain is already squad_xpts-based (best XI + captain) on both
        # sides, so §13 is satisfied by construction here.
        wc_gain = wc_plan.xpts_gain
        out_ids = {t["player_id"] for t in wc_plan.transfers_out}
        wc_squad_ids = [p for p in current_squad_ids if p not in out_ids] + [
            t["player_id"] for t in wc_plan.transfers_in
        ]
        wc_scenarios = pd.Series(dtype=float)
        if season is not None:
            wc_scenarios = gain_distribution(season, gws, wc_squad_ids, current_squad_ids)
        threshold = timing.wildcard_pts_gain_threshold * _panic_shrink(current_gw, season, timing)
        if not _clears_threshold(
            wc_gain, threshold, wc_scenarios, timing.wildcard_min_payoff_probability
        ):
            return None
        logger.info("WC recommended: gain=%.2f over %d GWs", wc_gain, horizon)
        return ChipRecommendation(
            Chip.WILDCARD, f"WC gain {wc_gain:.1f} xPts over {horizon} GWs", wc_gain
        )

    # On an ACTIVE DGW week, give Bench Boost/Free Hit first refusal --
    # they can extract the WHOLE squad's double-fixture value, which TC's
    # now much-easier-to-clear absolute threshold could otherwise routinely
    # preempt (TC only ever amplifies the ALREADY-CHOSEN captain, so it
    # loses nothing meaningful by going last that week — if BB/FH don't
    # fire, TC still gets a fair shot at the same DGW captain right after).
    order = (
        [_try_bb, _try_fh, _try_tc, _try_wc] if dgw_active_now
        else [_try_tc, _try_bb, _try_fh, _try_wc]
    )
    for candidate in order:
        rec = candidate()
        if rec is not None:
            return rec

    # Last resort (user's own words: TC "should NEVER be left unplayed in
    # both halves of the season" -- "at worst the default behaviour is to
    # panic and use the triple captain on the last day before the chips
    # reset or the season ends"). Reached only once every threshold above
    # -- already shrunk by _panic_shrink for the whole panic window -- has
    # failed to clear. Triggers on the FINAL TWO gameweeks of the half, not
    # just the literal last one, so a single skipped/missing decision point
    # right at the boundary (a postponement, a data gap) can't silently
    # cost the whole half's chip -- once forced, ``chips_used`` marks it
    # used immediately, so this can't double-fire across both gameweeks.
    if (
        current_gw >= _current_half_expiry_gw(current_gw, season) - 1
        and _chip_uses_remaining(Chip.TRIPLE_CAPTAIN, chips_used, current_gw, season) > 0
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
    """Real fix found 2026-07-30 (user's own review: "how can it ever be
    worth not playing [TC]? It is only negative if the player gets < 0
    points"). ``tc_gain`` used to be ``best_xpts - second_xpts`` — the GAP
    between the top two captain candidates. That's the wrong question: TC
    doesn't change WHO you captain (you'd pick the same best player either
    way), it only changes the multiplier (2x -> 3x) on whoever you'd
    already captain — so its real marginal value is one extra copy of that
    player's own points, i.e. ``best_xpts`` on its own, not the gap to
    whoever's in second place. Correct, strictly-additive framing: TC is
    virtually always worth playing (downside needs the captain to score
    negative points — a red card / own goal / missed penalty pile-up, rare
    and small next to typical upside); the only genuine cost is that only
    one chip can be played per gameweek, and only 2 TC uses exist all
    season (1 per half, no carryover) — so the real tradeoff is spending
    it now vs. saving it for a probably-better week later this half (see
    the DGW-hold-back bias in ``recommend_chip``), not a risk of loss.
    ``second_id`` is kept for compatibility with callers that still want
    the runner-up captain candidate; it no longer affects ``tc_gain``."""
    gw_proj = projections[
        (projections["gameweek"] == gw) & projections["player_id"].isin(squad_ids)
    ].sort_values("xpts", ascending=False)

    if len(gw_proj) < 2:
        return 0.0, None, None

    best_xpts = float(gw_proj.iloc[0]["xpts"])
    best_id = int(gw_proj.iloc[0]["player_id"])
    second_id = int(gw_proj.iloc[1]["player_id"])

    return best_xpts, best_id, second_id
