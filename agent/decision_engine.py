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
    TRANSFERS,
    ChipTimingThresholds,
    OptimiserConfig,
    TransferRules,
)
from data.db import get_session
from data.ingestors.ownership import load_latest_ownership
from data.models import ChipComparisonLog, DecisionLog, SimDecisionLog, SimManager
from data.overrides import apply_team_overrides, load_p_leave_overrides, log_rumoured_squad_members
from optimiser.chip_comparison import compare_chip_options
from optimiser.chips import (
    Chip,
    ChipRecommendation,
    chips_available_this_half,
    chips_used_this_season,
    recommend_chip,
)
from optimiser.departure_risk import apply_departure_discount
from optimiser.squad import optimise_squad_joint, optimise_starting_xi
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
    persist_projections,
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
    sim_manager_id: int | None,
    team_id: int,
    config: OptimiserConfig,
    *,
    decided_gw: int | None = None,
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
    season's starting budget", not "three million pounds".

    ``decided_gw`` scopes the lookup to a gameweek STRICTLY EARLIER than the
    one being decided (2026-08-28). Without it this took the latest lineup row
    outright, and the ``free_transfers`` field on that row is NEXT gameweek's
    allowance -- ``roll_forward_free_transfers`` has already added the weekly
    +1 (the current week's count is stored separately as
    ``free_transfers_available``). So re-running a gameweek read the previous
    run's roll-forward as its own allowance and banked a free transfer every
    time. Live on the GW2 deadline: a second run saw 2 free transfers where
    there was 1, made two transfers, and booked zero hits instead of -4.
    ``None`` keeps the old unscoped behaviour for callers that have no
    gameweek in hand (tests only -- the one production caller passes
    ``next_gw``)."""
    db = get_session()
    try:
        if sim_manager_id is not None:
            query = text("""
                SELECT details FROM sim_decision_log
                WHERE sim_manager_id = :sim_manager_id AND decision_type = 'lineup'
                  AND (:decided_gw IS NULL OR gameweek < :decided_gw)
                ORDER BY created_at DESC LIMIT 1
            """)
            row = db.execute(
                query, {"sim_manager_id": sim_manager_id, "decided_gw": decided_gw}
            ).fetchone()
        else:
            query = text("""
                SELECT dl.details
                FROM decision_log dl
                WHERE dl.decision_type = 'lineup'
                  AND (:decided_gw IS NULL OR dl.gameweek < :decided_gw)
                ORDER BY dl.created_at DESC
                LIMIT 1
            """)
            row = db.execute(query, {"decided_gw": decided_gw}).fetchone()
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
    ever recorded a lineup for.

    A wildcard recorded against ``next_gw`` ITSELF is excluded (2026-09-02, the
    same rule ``_chip_uses_remaining`` applies to uses): it is this gameweek's
    own earlier run, and on a re-run the rebuild has not happened -- that is
    precisely the decision being remade. Counting it reported age 0, so
    ``wildcard_min_managed_gws`` refused the wildcard on the re-run and the run
    after that offered it again, reproducing the chip-alternation defect through
    the age gate instead of the uses count."""
    wildcard_gws = [
        gw for chip, gw in chips_used if chip == Chip.WILDCARD and gw != next_gw
    ]
    if wildcard_gws:
        return max(0, next_gw - max(wildcard_gws))
    if decision_log.empty or "decision_type" not in decision_log.columns:
        return 0
    lineups = decision_log[decision_log["decision_type"] == "lineup"]
    if lineups.empty or "gameweek" not in lineups.columns:
        return 0
    return max(0, next_gw - int(lineups["gameweek"].min()))


def _comparison_eligible_chips(
    chips_used: list[tuple[Chip, int]],
    next_gw: int,
    season: str,
    squad_age_gws: int,
    chip_timing: ChipTimingThresholds,
) -> set[Chip]:
    """Which of Free Hit / Wildcard the comparison may nominate.

    ``chips_available_this_half`` only knows about USES REMAINING. It has no
    idea that ``recommend_chip`` refuses a wildcard on a squad younger than
    ``wildcard_min_managed_gws``, so on the GW3 frame the comparison was
    offered a chip the engine would never play, nominated it, and had that
    nomination discarded downstream -- stranding a Free Hit that had cleared
    its own margin. A chip the engine will refuse does not belong in the
    comparison at all.

    ``squad_age_gws`` is the caller's single value, the same one handed to
    ``recommend_chip``: computing it a second way here is how the two gates
    drift apart.
    """
    eligible = {Chip.FREE_HIT, Chip.WILDCARD} & set(
        chips_available_this_half(chips_used, next_gw, season)
    )
    if squad_age_gws < chip_timing.wildcard_min_managed_gws:
        eligible.discard(Chip.WILDCARD)
    return eligible


def _lineup_shape(squad: pd.DataFrame) -> dict:
    """The fields an outcome scorer needs to replay FPL's auto-substitutions
    (P2.1, 2026-08-16).

    ``scripts/backfill_decision_outcomes.py`` computes each decision's
    ``actual_outcome`` via ``_score_squad``, which supports autosubs and the
    vice-captain fallback -- but only when handed ``minutes``, ``positions``
    AND ``bench_order`` together. It could pass none of them, because the
    lineup ``details`` recorded here never carried positions or bench order.
    So every recorded outcome, for the real bot and for all 100 simulated
    personas, scored a blanking starter as 0 with no substitute.

    That is not a cosmetic understatement: it biases the record hardest
    exactly where the bench matters, which is the question the whole
    simulation cohort exists to answer."""
    if squad.empty or "id" not in squad.columns:
        return {"positions": {}, "bench_order": {}}
    positions = (
        {int(pid): str(pos) for pid, pos in zip(squad["id"], squad["position"], strict=True)}
        if "position" in squad.columns
        else {}
    )
    if "bench_order" not in squad.columns:
        return {"positions": positions, "bench_order": {}}
    # bench_order is -1 for starters (see optimise_starting_xi); only bench
    # slots carry a real priority, and only those are what autosubs consume.
    bench_order = {
        int(pid): int(order)
        for pid, order in zip(squad["id"], squad["bench_order"], strict=True)
        if pd.notna(order) and int(order) >= 0
    }
    return {"positions": positions, "bench_order": bench_order}


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


def _chip_comparison_rows(
    *,
    season: str,
    gameweek: int,
    sim_manager_id: int | None,
    comparison,
    live_chip,
) -> list[dict]:
    """One row per option, marking what the live path did and what the
    comparison would have done. Pure, so it is testable without a DB."""
    names = {None: "none", Chip.FREE_HIT: "free_hit", Chip.WILDCARD: "wildcard"}
    shadow_chip = comparison.best.chip if comparison.best is not None else None
    rows = []
    for option in comparison.options:
        rows.append({
            "season": season,
            "gameweek": gameweek,
            "sim_manager_id": sim_manager_id,
            "option": names.get(option.chip, str(option.chip)),
            "horizon_xpts": round(option.horizon_xpts, 4),
            "detail": option.detail,
            "chosen_live": option.chip == live_chip,
            "chosen_shadow": option.chip == shadow_chip,
        })
    return rows


def _persist_chip_comparison(db, rows: list[dict]) -> None:
    """Best-effort: a logging failure must never break a decision run."""
    try:
        for row in rows:
            db.add(ChipComparisonLog(**row))
        db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("chip comparison log not written: %s", exc)
        db.rollback()


def _run_decision_cycle(
    season: str,
    dry_run: bool,
    force_chip: Chip | None,
    config: OptimiserConfig,
    chip_timing: ChipTimingThresholds,
    team_id: int | None,
    sim_manager_id: int | None,
    transfer_rules: TransferRules | None = None,
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
    # Read the FULL persisted frame; each consumer slices it to its own
    # horizon (evaluate_transfers, recommend_chip), so handing them a
    # short frame silently shortens their planning instead.
    projections = get_latest_projections(horizon=config.projection_horizon_gws)
    # Feature B (plan 2026-08-10): rumour-discount tier, real data for the
    # first time (previously always an empty dict).
    projections = apply_departure_discount(projections, load_p_leave_overrides())

    # P3.2: effective ownership, read for the first time by anything. Empty
    # until the Overall league has ranked entries (post-GW1 deadline), and
    # a no-op for the real bot regardless while risk_level is 0 -- but the
    # cohort's risk_level axis exercises it, and it needs no second change
    # once real rows land.
    ownership = load_latest_ownership()

    state = _load_squad_state(sim_manager_id, team_id, config, decided_gw=next_gw)
    squad_ids = state.squad_ids
    available_budget = state.budget
    free_transfers = state.free_transfers

    # THIRD gap, found 2026-08-16 by auditing the decision log: this branched
    # on `projections.empty`, which is a fact about what happens to be
    # PERSISTED rather than about the season. The moment the cold start began
    # persisting its own projections (so the site and dashboard had numbers to
    # show), the frame stopped being empty pre-season and this took the
    # IN-SEASON path instead — which ran recommend_chip and recorded a Triple
    # Captain as played in GW1, before a ball had been kicked.
    #
    # Whether this is a cold start is a property of the SEASON: either it has
    # played gameweeks to condition on or it does not. Nothing about the
    # projections table can change that answer.
    if not season_has_played_history(season):
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
        # Persist the cold-start projections (2026-08-16). run_projections
        # correctly writes nothing pre-season (it has no rolling history to
        # condition on), and this branch never wrote its own -- so
        # `player_projections` stayed empty for the whole pre-season and every
        # reader of it showed a squad with no numbers: the site export's xpts
        # came back None for all 15, and the dashboard squad page likewise.
        # Same shape and same table as the in-season path, so nothing
        # downstream needs to know which built them.
        # Only the REAL bot writes to player_projections. It is a shared
        # table with no persona scoping, and get_latest_projections takes
        # MAX(created_at) per (player, gameweek) — so a persona persisting its
        # own cold-start frame decides what the real bot, the site export and
        # the dashboard all read. Measured before this guard: 90 personas had
        # written 235,720 rows across 83 batches, and whichever ran last owned
        # the read.
        if sim_manager_id is None:
            persist_projections(cs_projections)
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
                # P2.1: without these the outcome scorer cannot apply
                # auto-substitutions and every blanking starter is recorded
                # as a plain 0.
                **_lineup_shape(xi_solution.squad),
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

    if projections.empty:
        # Mid-season with nothing projected is a genuine failure, not a cold
        # start -- the pipeline should have produced something.
        logger.error("No projections available mid-season — aborting")
        return {"error": "no_projections"}

    players = _load_players()
    players = players.merge(
        projections[projections["gameweek"] == next_gw][["player_id", "start_probability"]],
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    if not squad_ids:
        logger.warning(
            "No saved squad found — running full squad optimisation "
            "(season start / first run)"
        )
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

    # 2026-09-02 (Task 7): the horizon-total comparison, computed BEFORE
    # recommend_chip so it can be consulted rather than only logged after the
    # fact. Best-effort — a squad-optimisation failure inside the comparison
    # must never take down a decision run that the legacy threshold path
    # would otherwise have completed fine.
    # One value, two consumers: the eligibility filter below and
    # `recommend_chip`'s own wildcard gate must agree, or the comparison can
    # again nominate a chip the engine refuses.
    squad_age_gws = _squad_age_gws(decision_log, chips_used, next_gw)

    comparison = None
    if squad_ids:
        try:
            comparison = compare_chip_options(
                current_squad_ids=squad_ids,
                projections=projections,
                players=players,
                free_transfers=free_transfers,
                current_gw=next_gw,
                horizon=chip_timing.chip_comparison_horizon_gws,
                free_hit_chip=Chip.FREE_HIT,
                wildcard_chip=Chip.WILDCARD,
                free_hit_margin=chip_timing.free_hit_comparison_margin,
                wildcard_margin=chip_timing.wildcard_comparison_margin,
                eligible_chips=_comparison_eligible_chips(
                    chips_used, next_gw, season, squad_age_gws, chip_timing,
                ),
                available_budget=available_budget,
                bank=state.bank,
                purchase_prices=state.purchase_prices,
                ownership=ownership,
                season=season,
                config=config,
                transfer_rules=transfer_rules,
            )
        except Exception as exc:  # noqa: BLE001 -- shadow work never breaks a run
            logger.warning("chip comparison skipped: %s", exc)

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
            squad_age_gws=squad_age_gws,
            season=season,
            chip_timing=chip_timing,
            config=config,
            # 2026-08-18 (§12): the wildcard is evaluated with the same
            # evaluate_transfers call that will execute it, so it needs the
            # same affordability ledger. Without these it would judge a
            # rebuild it could not afford to buy.
            bank=state.bank,
            purchase_prices=state.purchase_prices,
            comparison=comparison,
        )

    if comparison is not None:
        # The session construction belongs inside the guard too. Only
        # `_persist_chip_comparison` was best-effort, so a `get_session()`
        # failure -- a locked or missing database, say -- still propagated and
        # killed a decision run over shadow logging that nothing in the
        # decision path reads.
        db = None
        try:
            db = get_session()
            _persist_chip_comparison(
                db,
                _chip_comparison_rows(
                    season=season, gameweek=next_gw, sim_manager_id=sim_manager_id,
                    comparison=comparison, live_chip=chip_rec.chip,
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- logging never breaks a run
            logger.warning("chip comparison session not opened: %s", exc)
        finally:
            if db is not None:
                db.close()

    wildcard_active = chip_rec.chip == Chip.WILDCARD
    free_hit_active = chip_rec.chip == Chip.FREE_HIT

    if free_hit_active:
        transfer_plan = TransferPlan(
            transfers_in=[], transfers_out=[], hits_taken=0, xpts_gain=0.0, net_xpts_gain=0.0
        )
        # optimise_squad_joint, not optimise_squad: at the shipped mu of 0 it
        # short-circuits to exactly the same answer without touching the DB,
        # and when mu is non-zero it re-ranks the pool on the real joint
        # scenarios (optimiser/joint_risk.py). Wiring it here is what makes a
        # calibrated mu reach the live decision rather than only the backtest.
        squad_solution = optimise_squad_joint(
            projections,
            players,
            budget=available_budget,
            horizon=1,
            season=season,
            gameweek=next_gw,
            ownership=ownership,
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
            ownership=ownership,
            config=config,
            transfer_rules=transfer_rules,
            # P1.6: real affordability. Without these the optimiser priced
            # every owned player at their current cost and spent from a
            # budget frozen at 100.0 since the cold start.
            bank=state.bank,
            purchase_prices=state.purchase_prices,
        )

        new_squad_ids = (
            [t["player_id"] for t in transfer_plan.transfers_in]
            + [
                pid for pid in squad_ids
                if pid not in {t["player_id"] for t in transfer_plan.transfers_out}
            ]
        ) if transfer_plan.transfers_in else squad_ids

        squad_df = players[players["id"].isin(new_squad_ids)].copy()
        squad_solution = optimise_starting_xi(
            squad_df, projections, next_gw, season=season,
            ownership=ownership, config=config,
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
        "squad": squad_solution.squad[[
            "id", "web_name", "position", "now_cost", "is_starting",
            "is_captain", "is_vice_captain", "bench_order",
        ]].to_dict("records"),
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
            # P2.1: see _lineup_shape -- required for autosub-aware scoring.
            **_lineup_shape(squad_solution.squad),
            # P2.2: hits are booked on the separate `transfers` decision, so
            # the outcome scorer had no way to net them off. Recorded here
            # too, on the row whose actual_outcome they reduce.
            "hits_taken": transfer_plan.hits_taken,
            # P2.6: the counterfactual. At season end "what went wrong" is
            # unanswerable beyond the score unless what was DECLINED was
            # recorded too. A chip that never fired left no trace at all --
            # only played chips got a row -- so an entire season of
            # near-misses was invisible. `free_transfers_used` separates
            # "banked deliberately" from "found nothing worth doing".
            "chip_considered": chip_rec.chip.value if chip_rec.chip else None,
            "chip_reason": chip_rec.reason,
            "chip_expected_gain": round(chip_rec.expected_gain, 2),
            "free_transfers_available": free_transfers,
            "free_transfers_used": len(transfer_plan.transfers_in),
            "transfer_xpts_gain": round(transfer_plan.xpts_gain, 2),
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
                transfer_rules=transfer_rules,
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
        squad_solution.squad.loc[
            squad_solution.squad["id"] == xi_solution.captain_id, "web_name"
        ].values[0]
        if xi_solution.captain_id else "?",
    )

    return result


def run(
    season: str = "2026-27",
    force_chip: Chip | None = None,
) -> dict:
    """Build this gameweek's decision. Nothing is submitted anywhere.

    The ``dry_run`` parameter was removed on 2026-08-18 with the submission
    path. ``DecisionLog.dry_run`` survives as a column because it carries
    real history — every decision this project ever recorded was a dry run —
    and dropping a column in SQLite costs more than the row of Trues is
    worth. It is written True and never read as a switch.
    """
    return _run_decision_cycle(
        season=season,
        dry_run=True,
        force_chip=force_chip,
        config=OPTIMISER,
        chip_timing=CHIP_TIMING,
        team_id=settings.fpl_team_id,
        sim_manager_id=None,
    )


def run_for_persona(persona: SimManager, season: str = "2026-27") -> dict:
    """Runs one simulated persona through the exact same decision logic as
    the real bot (plan/simulation-engine-v1.md) -- never touches
    any submission path; none exists anywhere in this project as of 2026-08-18, at
    all, not a disabled flag. ``persona`` supplies risk_level/
    max_ownership_differential/chip_aggressiveness; every other config
    field (including mu_baseline/mu_range) stays at today's real default."""
    config = dataclasses.replace(
        OPTIMISER,
        risk_level=persona.risk_level,
        max_ownership_differential=persona.max_ownership_differential,
        # P2.4: the knobs the cohort exists to test. Previously pinned to the
        # real bot's values across every persona, so the sweep could say
        # nothing about any of them.
        transfer_planning_horizon_gws=persona.transfer_planning_horizon_gws,
        bench_value_weight=persona.bench_value_weight,
        mu_baseline=persona.mu_baseline,
    )
    transfer_rules = dataclasses.replace(
        TRANSFERS,
        transfer_switching_cost=persona.transfer_switching_cost,
        ft_terminal_value=persona.ft_terminal_value,
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
        transfer_rules=transfer_rules,
        refresh_projections=False,
    )
