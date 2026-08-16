import dataclasses
import json
import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from config.settings import settings
from config.strategy import (
    CHIP_TIMING,
    OPTIMISER,
    SQUAD,
    ChipTimingThresholds,
    OptimiserConfig,
)
from data.db import get_session
from data.models import DecisionLog, SimDecisionLog, SimManager
from data.overrides import apply_team_overrides, load_p_leave_overrides, log_rumoured_squad_members
from optimiser.chips import Chip, ChipRecommendation, chips_used_this_season, recommend_chip
from optimiser.departure_risk import apply_departure_discount
from optimiser.squad import optimise_squad, optimise_starting_xi
from optimiser.transfers import (
    TransferPlan,
    evaluate_transfers,
    get_dgw_coverage,
    roll_forward_free_transfers,
    selling_price,
)
from projection import cold_start
from projection.pipeline import (
    _get_bgw_gameweeks,
    _get_current_and_next_gw,
    _get_dgw_gameweeks,
    get_latest_projections,
    run_projections,
    season_has_played_history,
)

logger = logging.getLogger(__name__)


def _load_players() -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT id, code, fpl_id, web_name, position, team_id, now_cost,
                   status, chance_of_playing_next_round, selected_by_percent,
                   form, ict_index, influence, creativity, threat
            FROM players
        """)
        players = pd.read_sql(query, db.bind)
    finally:
        db.close()
    return apply_team_overrides(players)


@dataclasses.dataclass
class SquadState:
    """What the bot believes it owns going into a gameweek.

    ``bank``/``purchase_prices`` are P1.6. Before them the engine carried a
    single ``budget`` float that was seeded at 100.0 by the cold start and
    written back unchanged every week thereafter, so it never moved for price
    changes or for money spent, and the optimiser's affordability constraint
    was wrong in both directions -- under-spending as the squad appreciated,
    and proposing transfers that could not actually be paid for if it
    depreciated. ``purchase_prices`` is what makes selling prices knowable
    (FPL pays back only half of a price rise)."""

    squad_ids: list[int]
    budget: float
    free_transfers: int
    bank: float
    purchase_prices: dict[int, float]


def _load_squad_state(
    sim_manager_id: int | None, team_id: int, config: OptimiserConfig
) -> SquadState:
    """``sim_manager_id is None`` reads the real bot's own ``decision_log``
    (``team_id`` is informational only, not used in the query -- the real
    squad is one continuous decision history, not partitioned per team_id).
    A ``sim_manager_id`` reads that persona's own ``sim_decision_log`` rows
    instead, completely isolated from the real squad's history.

    ``budget``'s fallback-when-missing value used to be
    ``config.transfer_planning_horizon_gws`` (3.0) -- a units mismatch
    inherited from the pre-refactor ``_load_my_squad``, flagged at the time
    rather than changed. Fixed here (P1.6): a missing budget means "the
    season's starting budget", not "three million pounds"."""
    db = get_session()
    try:
        if sim_manager_id is not None:
            query = text("""
                SELECT details FROM sim_decision_log
                WHERE sim_manager_id = :sim_manager_id AND decision_type = 'lineup'
                ORDER BY created_at DESC LIMIT 1
            """)
            row = db.execute(query, {"sim_manager_id": sim_manager_id}).fetchone()
        else:
            query = text("""
                SELECT dl.details
                FROM decision_log dl
                WHERE dl.decision_type = 'lineup'
                ORDER BY dl.created_at DESC
                LIMIT 1
            """)
            row = db.execute(query).fetchone()
        if row:
            details = json.loads(row[0])
            squad_ids = details.get("squad_ids", [])
            budget = details.get("budget", SQUAD.budget_total)
            free_transfers = details.get("free_transfers", 1)
            # JSON object keys are strings; the rest of the engine keys
            # players by int.
            purchase_prices = {
                int(pid): float(price)
                for pid, price in (details.get("purchase_prices") or {}).items()
            }
            bank = details.get("bank")
            return SquadState(
                squad_ids=squad_ids,
                budget=float(budget),
                free_transfers=int(free_transfers),
                bank=float(bank) if bank is not None else 0.0,
                purchase_prices=purchase_prices,
            )
        return SquadState([], SQUAD.budget_total, 1, 0.0, {})
    finally:
        db.close()


def _load_own_decision_log(sim_manager_id: int | None) -> pd.DataFrame:
    db = get_session()
    try:
        if sim_manager_id is not None:
            query = text(
                "SELECT * FROM sim_decision_log WHERE sim_manager_id = :id "
                "ORDER BY created_at DESC"
            )
            return pd.read_sql(query, db.bind, params={"id": sim_manager_id})
        query = text("SELECT * FROM decision_log ORDER BY created_at DESC")
        return pd.read_sql(query, db.bind)
    finally:
        db.close()


def _record_decision(
    sim_manager_id: int | None,
    gameweek: int,
    decision_type: str,
    details: dict,
    projected_gain: float = 0.0,
    dry_run: bool = True,
) -> None:
    db = get_session()
    try:
        entry: SimDecisionLog | DecisionLog
        if sim_manager_id is not None:
            entry = SimDecisionLog(
                sim_manager_id=sim_manager_id,
                gameweek=gameweek,
                decision_type=decision_type,
                details=json.dumps(details),
                projected_gain=projected_gain,
                created_at=datetime.utcnow(),
            )
        else:
            entry = DecisionLog(
                gameweek=gameweek,
                decision_type=decision_type,
                details=json.dumps(details),
                projected_gain=projected_gain,
                dry_run=dry_run,
                created_at=datetime.utcnow(),
            )
        db.add(entry)
        db.commit()
    finally:
        db.close()


def _squad_age_gws(
    decision_log: pd.DataFrame, chips_used: list[tuple[Chip, int]], next_gw: int
) -> int:
    """How many gameweeks this squad has been managed since it was last
    rebuilt from scratch (P1.9, 2026-08-16).

    ``recommend_chip`` gates the wildcard behind
    ``wildcard_min_managed_gws`` using this, and defaults it to 99 when the
    caller omits it. ``scripts/backtest.py`` tracks and passes it; this
    module never did, so the gate was inert live and a wildcard could fire
    against a one-week-old squad. Counts from the last wildcard (which is
    exactly "rebuilt from scratch"), else from the first gameweek this bot
    ever recorded a lineup for."""
    wildcard_gws = [gw for chip, gw in chips_used if chip == Chip.WILDCARD]
    if wildcard_gws:
        return max(0, next_gw - max(wildcard_gws))
    if decision_log.empty or "decision_type" not in decision_log.columns:
        return 0
    lineups = decision_log[decision_log["decision_type"] == "lineup"]
    if lineups.empty or "gameweek" not in lineups.columns:
        return 0
    return max(0, next_gw - int(lineups["gameweek"].min()))


def _settle_transfers(
    state: SquadState, plan: TransferPlan, players: pd.DataFrame
) -> tuple[float, dict[int, float]]:
    """(new bank, new purchase prices) after ``plan`` is executed (P1.6).

    Selling credits the SELLING price -- half of any rise since purchase, all
    of any fall -- and buying debits the current price. A player whose
    purchase price we never recorded (a squad carried over from before P1.6)
    is treated as bought at their current price, i.e. no rise to share, which
    is the same assumption the engine made implicitly before."""
    now_cost = dict(zip(players["id"], players["now_cost"], strict=True))
    prices = dict(state.purchase_prices)
    bank = state.bank

    for out in plan.transfers_out:
        pid = out["player_id"]
        current = float(now_cost.get(pid, out.get("cost", 0.0)))
        bank += selling_price(prices.get(pid, current), current)
        prices.pop(pid, None)
    for incoming in plan.transfers_in:
        pid = incoming["player_id"]
        current = float(now_cost.get(pid, incoming.get("cost", 0.0)))
        bank -= current
        prices[pid] = current

    return round(bank, 1), prices


def _bench_xpts(squad_ids: list[int], projections: pd.DataFrame, gw: int) -> float:
    gw_proj = projections[
        (projections["gameweek"] == gw) & projections["player_id"].isin(squad_ids)
    ].sort_values("xpts", ascending=False)
    if len(gw_proj) <= 11:
        return 0.0
    return float(gw_proj.iloc[11:]["xpts"].sum())


def _run_decision_cycle(
    season: str,
    dry_run: bool,
    force_chip: Chip | None,
    config: OptimiserConfig,
    chip_timing: ChipTimingThresholds,
    team_id: int | None,
    sim_manager_id: int | None,
    refresh_projections: bool = True,
) -> dict:
    """The actual decision loop, shared by the real bot (``run()``) and every
    simulated persona (``run_for_persona()``). Behaviour is governed
    entirely by ``config``/``chip_timing`` and by which storage
    (``sim_manager_id`` or not) it reads/writes -- never by mutating global
    state. See plan/simulation-engine-v1.md.

    ``refresh_projections`` (default True, ``run()``'s only behaviour):
    re-runs and persists the projection pipeline before reading it back.
    Simulated personas pass False -- ``scripts/run_simulations.py`` runs
    right after ``scripts/run_agent.py`` in the same scheduled job, which
    has already refreshed this gameweek's projections; recomputing (and
    re-persisting near-duplicate rows) per persona would be 100x wasted
    work and DB writes for identical numbers."""
    current_gw, next_gw = _get_current_and_next_gw()

    logger.info(
        "Decision cycle starting: current_gw=%d next_gw=%d dry_run=%s sim_manager_id=%s",
        current_gw, next_gw, dry_run, sim_manager_id,
    )

    if refresh_projections:
        logger.info("Running projection pipeline...")
        run_projections(season=season, persist=True)
    # P1.1: the multi-period transfer ILP, the chip evaluator and the
    # blank-gameweek count all read this frame -- it must span the whole
    # planning horizon, not just the gameweek being decided.
    projections = get_latest_projections(horizon=config.transfer_planning_horizon_gws)
    # Feature B (plan 2026-08-10): rumour-discount tier, real data for the
    # first time (previously always an empty dict).
    projections = apply_departure_discount(projections, load_p_leave_overrides())

    state = _load_squad_state(sim_manager_id, team_id, config)
    squad_ids = state.squad_ids
    available_budget = state.budget
    free_transfers = state.free_transfers

    if projections.empty:
        if squad_ids and season_has_played_history(season):
            logger.error("No projections available — aborting")
            return {"error": "no_projections"}
        # Real gap found 2026-07-30 (the user's own live-smoke-test request):
        # a true pre-season GW1 has no current-season history for
        # run_projections' trained-model pipeline to condition on (it
        # correctly returns empty rather than inventing signal), but `run`
        # had no fallback at all — it just aborted with "no_projections",
        # meaning the live agent could never actually build the real
        # season-opening squad. Falls back to the same prior-season-
        # carryover projection cold_start.py already provides (T7),
        # matching scripts/backtest.py's GW1 handling (2026-07-30).
        #
        # Second real gap found 2026-08-09: gating this purely on
        # `squad_ids` meant that once the FIRST cold-start run recorded a
        # squad, every later rerun during the same still-pre-season window
        # took the abort branch instead — the user could never re-run
        # --dry-run to refine the initial squad as new signal data (odds,
        # injuries, press) came in. `season_has_played_history` distinguishes
        # "genuinely mid-season and something broke" (real abort) from
        # "still pre-season, rebuild from prior-season data" (cold start),
        # independent of whether a squad was already recorded.
        logger.warning(
            "No projections available and still pre-season — cold start, "
            "(re)building initial squad from prior-season data"
        )
        cs_players = _load_players()
        solution, cs_projections = cold_start.build_initial_squad(
            season, players=cs_players, config=config
        )
        xi_solution = optimise_starting_xi(
            solution.squad, cs_projections, next_gw, season=season, config=config
        )
        result = {
            "gameweek": next_gw,
            "dry_run": dry_run,
            "chip": None,
            "chip_reason": "cold start — no chip decision on the initial build",
            "transfers_in": [],
            "transfers_out": [],
            "hits_taken": 0,
            "net_xpts_gain": 0.0,
            "squad": xi_solution.squad[[
                "id", "web_name", "position", "now_cost",
                "is_starting", "is_captain", "is_vice_captain", "bench_order",
            ]].to_dict("records"),
            "captain_id": xi_solution.captain_id,
            "vice_captain_id": xi_solution.vice_captain_id,
            "total_xpts": round(xi_solution.total_xpts, 2),
            "total_cost": round(solution.total_cost, 1),
            "dgw_coverage": {},
            "cold_start": True,
        }
        _record_decision(
            sim_manager_id,
            gameweek=next_gw,
            decision_type="lineup",
            details={
                "squad_ids": solution.squad["id"].tolist(),
                "starting_ids": xi_solution.starting_xi["id"].tolist(),
                "captain_id": xi_solution.captain_id,
                "vice_captain_id": xi_solution.vice_captain_id,
                "budget": SQUAD.budget_total,
                "free_transfers": 1,
                # P1.6: the season opens with every player bought at their
                # current price and whatever the build didn't spend left in
                # the bank. This is the ONLY point at which purchase prices
                # are knowable for free -- from here they have to be carried.
                "bank": round(SQUAD.budget_total - solution.total_cost, 1),
                "purchase_prices": {
                    int(r.id): float(r.now_cost) for r in solution.squad.itertuples()
                },
            },
            projected_gain=xi_solution.total_xpts,
            dry_run=dry_run,
        )
        logger.info(
            "Cold-start decision complete: xPts=%.2f captain=%s",
            xi_solution.total_xpts,
            solution.squad.loc[
                solution.squad["id"] == xi_solution.captain_id, "web_name"
            ].values[0] if xi_solution.captain_id else "?",
        )
        return result

    players = _load_players()
    players = players.merge(
        projections[projections["gameweek"] == next_gw][["player_id", "start_probability"]],
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    if not squad_ids:
        logger.warning("No saved squad found — running full squad optimisation (season start / first run)")
        available_budget = 100.0
        free_transfers = 15

    dgw_gws = _get_dgw_gameweeks(config.transfer_planning_horizon_gws)
    bgw_gws = _get_bgw_gameweeks(config.transfer_planning_horizon_gws)
    # P1.5 (2026-08-16): count blanks in the gameweek being DECIDED, not
    # anywhere in the lookahead window. Free Hit's eligibility gate consumes
    # this to justify playing the chip THIS week, so a blank two gameweeks
    # away used to trigger it two gameweeks early. (The lookahead window is
    # still the right input for transferring blanking players out ahead of
    # time -- that planning does not exist yet; see plan P3.4.)
    #
    # This was also trivially 15/15 before P1.1: with only the current
    # gameweek in `projections`, every future gameweek had no rows at all, so
    # `.sum() == 0` was true for every player.
    bgw_now = {gw for gw in bgw_gws if gw == next_gw}
    bgw_affected = sum(
        1 for pid in squad_ids
        if any(
            projections[
                (projections["gameweek"] == gw) & (projections["player_id"] == pid)
            ]["xpts"].sum() == 0
            for gw in bgw_now
        )
    ) if bgw_now and squad_ids else 0

    decision_log = _load_own_decision_log(sim_manager_id)
    chips_used = chips_used_this_season(decision_log)

    bench_pts = _bench_xpts(squad_ids, projections, next_gw) if squad_ids else 0.0

    chip_rec: ChipRecommendation
    if force_chip:
        chip_rec = ChipRecommendation(force_chip, "forced by operator", 0.0)
    else:
        chip_rec = recommend_chip(
            current_gw=next_gw,
            current_squad_ids=squad_ids,
            projections=projections,
            players=players,
            available_budget=available_budget,
            free_transfers=free_transfers,
            chips_used=chips_used,
            bench_xpts=bench_pts,
            dgw_gws=dgw_gws,
            bgw_affected_count=bgw_affected,
            squad_age_gws=_squad_age_gws(decision_log, chips_used, next_gw),
            season=season,
            chip_timing=chip_timing,
            config=config,
        )

    wildcard_active = chip_rec.chip == Chip.WILDCARD
    free_hit_active = chip_rec.chip == Chip.FREE_HIT

    if free_hit_active:
        transfer_plan = TransferPlan(
            transfers_in=[], transfers_out=[], hits_taken=0, xpts_gain=0.0, net_xpts_gain=0.0
        )
        squad_solution = optimise_squad(
            projections=projections,
            players=players,
            budget=available_budget,
            horizon=1,
            season=season,
            config=config,
        )
    else:
        transfer_plan = evaluate_transfers(
            current_squad_ids=squad_ids,
            projections=projections,
            players=players,
            free_transfers=free_transfers,
            available_budget=available_budget,
            wildcard_active=wildcard_active,
            dgw_gws=dgw_gws,
            config=config,
            # P1.6: real affordability. Without these the optimiser priced
            # every owned player at their current cost and spent from a
            # budget frozen at 100.0 since the cold start.
            bank=state.bank,
            purchase_prices=state.purchase_prices,
        )

        new_squad_ids = (
            [t["player_id"] for t in transfer_plan.transfers_in]
            + [pid for pid in squad_ids if pid not in {t["player_id"] for t in transfer_plan.transfers_out}]
        ) if transfer_plan.transfers_in else squad_ids

        squad_df = players[players["id"].isin(new_squad_ids)].copy()
        squad_solution = optimise_starting_xi(
            squad_df, projections, next_gw, season=season, config=config
        )

    xi_solution = squad_solution

    dgw_coverage = get_dgw_coverage(
        squad_solution.squad["id"].tolist(), players, dgw_gws, projections
    )

    log_rumoured_squad_members(squad_solution.squad["id"].tolist(), players)

    # P1.6: a Free Hit squad is handed back at the end of the gameweek, so
    # neither the bank nor the purchase-price ledger moves.
    settled_bank, settled_prices = (
        (state.bank, state.purchase_prices)
        if free_hit_active
        else _settle_transfers(state, transfer_plan, players)
    )

    result = {
        "gameweek": next_gw,
        "dry_run": dry_run,
        "chip": chip_rec.chip.value if chip_rec.chip else None,
        "chip_reason": chip_rec.reason,
        "transfers_in": transfer_plan.transfers_in,
        "transfers_out": transfer_plan.transfers_out,
        "hits_taken": transfer_plan.hits_taken,
        "net_xpts_gain": round(transfer_plan.net_xpts_gain, 2),
        "squad": squad_solution.squad[["id", "web_name", "position", "now_cost", "is_starting", "is_captain", "is_vice_captain", "bench_order"]].to_dict("records"),
        "captain_id": xi_solution.captain_id,
        "vice_captain_id": xi_solution.vice_captain_id,
        "total_xpts": round(xi_solution.total_xpts, 2),
        "total_cost": round(squad_solution.total_cost, 1),
        "dgw_coverage": dgw_coverage,
    }

    _record_decision(
        sim_manager_id,
        gameweek=next_gw,
        decision_type="transfers",
        details={
            "transfers_in": transfer_plan.transfers_in,
            "transfers_out": transfer_plan.transfers_out,
            "hits_taken": transfer_plan.hits_taken,
        },
        projected_gain=transfer_plan.net_xpts_gain,
        dry_run=dry_run,
    )

    _record_decision(
        sim_manager_id,
        gameweek=next_gw,
        decision_type="lineup",
        details={
            "squad_ids": squad_solution.squad["id"].tolist(),
            "starting_ids": xi_solution.starting_xi["id"].tolist(),
            "captain_id": xi_solution.captain_id,
            "vice_captain_id": xi_solution.vice_captain_id,
            "budget": available_budget,
            # P1.2: was `max(0, free_transfers - len(transfers_in))` — no weekly
            # +1, no cap, and it hit 0 after the first transfer, which made the
            # transfer ILP infeasible from then on. Shared with the backtest so
            # the two cannot drift again.
            "free_transfers": roll_forward_free_transfers(
                free_transfers,
                len(transfer_plan.transfers_in),
                wildcard_played=wildcard_active,
                free_hit_played=free_hit_active,
            ),
            # P1.6: a Free Hit's squad is reverted after the gameweek, so its
            # transfers must not move the bank or the purchase-price ledger.
            "bank": settled_bank,
            "purchase_prices": {str(pid): price for pid, price in settled_prices.items()},
        },
        projected_gain=xi_solution.total_xpts,
        dry_run=dry_run,
    )

    if chip_rec.chip:
        _record_decision(
            sim_manager_id,
            gameweek=next_gw,
            decision_type="chip",
            details={"chip": chip_rec.chip.value, "reason": chip_rec.reason},
            projected_gain=chip_rec.expected_gain,
            dry_run=dry_run,
        )

    logger.info(
        "Decision complete: chip=%s transfers=%d→%d hits=%d xPts=%.2f captain=%s",
        chip_rec.chip,
        len(transfer_plan.transfers_out),
        len(transfer_plan.transfers_in),
        transfer_plan.hits_taken,
        xi_solution.total_xpts,
        squad_solution.squad.loc[squad_solution.squad["id"] == xi_solution.captain_id, "web_name"].values[0]
        if xi_solution.captain_id else "?",
    )

    return result


def run(
    season: str = "2026-27",
    force_chip: Chip | None = None,
    dry_run: bool | None = None,
) -> dict:
    dry_run = settings.dry_run if dry_run is None else dry_run
    return _run_decision_cycle(
        season=season,
        dry_run=dry_run,
        force_chip=force_chip,
        config=OPTIMISER,
        chip_timing=CHIP_TIMING,
        team_id=settings.fpl_team_id,
        sim_manager_id=None,
    )


def run_for_persona(persona: SimManager, season: str = "2026-27") -> dict:
    """Runs one simulated persona through the exact same decision logic as
    the real bot (plan/simulation-engine-v1.md) -- never touches
    ``agent/fpl_client.py``; no submission path exists in this code path at
    all, not a disabled flag. ``persona`` supplies risk_level/
    max_ownership_differential/chip_aggressiveness; every other config
    field (including mu_baseline/mu_range) stays at today's real default."""
    config = dataclasses.replace(
        OPTIMISER,
        risk_level=persona.risk_level,
        max_ownership_differential=persona.max_ownership_differential,
    )
    chip_timing = dataclasses.replace(
        CHIP_TIMING,
        wildcard_pts_gain_threshold=(
            CHIP_TIMING.wildcard_pts_gain_threshold * persona.chip_aggressiveness
        ),
        free_hit_single_gw_gain_threshold=(
            CHIP_TIMING.free_hit_single_gw_gain_threshold * persona.chip_aggressiveness
        ),
        bench_boost_min_bench_xpts=(
            CHIP_TIMING.bench_boost_min_bench_xpts * persona.chip_aggressiveness
        ),
        triple_captain_min_gain=(
            CHIP_TIMING.triple_captain_min_gain * persona.chip_aggressiveness
        ),
    )
    return _run_decision_cycle(
        season=season,
        dry_run=True,
        force_chip=None,
        config=config,
        chip_timing=chip_timing,
        team_id=None,
        sim_manager_id=persona.id,
        refresh_projections=False,
    )
