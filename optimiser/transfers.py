import logging
from dataclasses import dataclass

import pandas as pd
import pulp

from config.strategy import OPTIMISER, SQUAD, TRANSFERS, OptimiserConfig, TransferRules
from data.overrides import load_excluded_player_ids
from optimiser.departure_risk import confirmed_p_leave, is_hard_excluded
from optimiser.scoring import lambda_mu_for_risk_level, risk_adjusted_score

logger = logging.getLogger(__name__)

SQUAD_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
STARTING_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}


@dataclass
class TransferPlan:
    transfers_in: list[dict]
    transfers_out: list[dict]
    hits_taken: int
    xpts_gain: float
    net_xpts_gain: float


def roll_forward_free_transfers(
    free_transfers: int,
    transfers_made: int,
    wildcard_played: bool = False,
    free_hit_played: bool = False,
    transfer_rules: TransferRules | None = None,
) -> int:
    """Next gameweek's free-transfer allowance (P1.2, 2026-08-16,
    plan/decision-engine-recovery-plan.md).

    Real bug this exists to kill: ``agent/decision_engine.py`` stored
    ``max(0, free_transfers - transfers_made)`` -- missing both the weekly
    ``+1`` allowance and the cap -- while ``scripts/backtest.py`` carried the
    correct formula inline. So the backtest banked transfers properly and the
    live agent never did; worse, the moment the live count reached 0 the
    transfer ILP became infeasible (``ft`` has ``lowBound=1``) and silently
    returned an empty plan, permanently, from the first week a transfer was
    made. One shared function so the two paths cannot drift again.

    ``transfers_made`` may exceed ``free_transfers`` (hits were taken); the
    surplus is paid in points, not in next week's allowance, so the result
    still floors at the weekly allowance.

    **Wildcard and Free Hit behave identically here** (fixed 2026-08-18,
    engine review §10). Both chips' transfers are outside the allowance
    entirely, and FPL RETAINS your saved free transfers across either one —
    confirmed against the Premier League's own worked example: two saved
    before a Gameweek 6 wildcard leaves THREE for Gameweek 7 (the two saved,
    plus GW7's allotment). You simply do not earn an extra one in the week you
    play the chip, which is what ``transfers_made = 0`` expresses.

    This previously returned the bare weekly allowance after a wildcard,
    destroying up to four banked transfers — roughly 16 points of avoidable
    hits, twice a season — while the Free Hit branch directly below already
    had it right. ``ft[0]`` in the multi-period ILP is seeded from this value,
    so every subsequent week planned against a wrong allowance too."""
    trules = transfer_rules or TRANSFERS
    if wildcard_played or free_hit_played:
        transfers_made = 0
    carried = free_transfers - transfers_made + trules.free_transfers_per_gw
    return min(trules.max_banked_free_transfers, max(trules.free_transfers_per_gw, carried))


def selling_price(purchase_price: float, now_cost: float) -> float:
    """What FPL actually pays you for a player (P1.6, 2026-08-16).

    You keep only HALF of any price RISE since you bought them, rounded DOWN
    to the nearest £0.1m; a price FALL is taken in full. Nothing in this
    project modelled that -- ``agent/fpl_client.py`` read ``selling_price``
    off the API purely to build a submission payload, and the optimiser's
    affordability constraint used ``now_cost`` for everything, so it believed
    an appreciated squad was worth more than it could actually be sold for.

    Works in integer tenths because prices are quoted in £0.1m and float
    arithmetic on 0.1s does not round the way FPL's does."""
    purchase_tenths = round(purchase_price * 10)
    now_tenths = round(now_cost * 10)
    if now_tenths <= purchase_tenths:
        return now_tenths / 10
    return (purchase_tenths + (now_tenths - purchase_tenths) // 2) / 10


def squad_xpts(squad_ids: list[int], projections: pd.DataFrame, horizon: int) -> float:
    """What a squad is actually worth over ``horizon`` gameweeks: the best
    ELEVEN each week, plus a second copy of the best of those for the captain.

    Public because it is the project's one definition of squad value and
    ``optimiser/chips.py`` must use the same one (2026-08-18, engine review
    §13). The chip evaluations previously summed all FIFTEEN players' xpts and
    credited no captain — so a Free Hit, which plays eleven, was judged partly
    on four players who would not take the field, against thresholds
    calibrated for neither definition. Bench Boost is the one chip for which a
    fifteen-man sum is correct, and it has its own path.
    """
    gws = sorted(projections["gameweek"].unique())[:horizon]
    subset = projections[
        projections["gameweek"].isin(gws) & projections["player_id"].isin(squad_ids)
    ]
    total = 0.0
    for gw in gws:
        gw_xpts = subset[subset["gameweek"] == gw]["xpts"].nlargest(11)
        captain_bonus = float(gw_xpts.max()) if len(gw_xpts) > 0 else 0.0
        total += float(gw_xpts.sum()) + captain_bonus
    return total


def evaluate_transfers(
    current_squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    free_transfers: int = 1,
    available_budget: float | None = None,
    wildcard_active: bool = False,
    solver_time_limit: int | None = None,
    ownership: pd.DataFrame | None = None,
    config: OptimiserConfig | None = None,
    transfer_rules: TransferRules | None = None,
    bank: float | None = None,
    purchase_prices: dict[int, float] | None = None,
    horizon: int | None = None,
) -> TransferPlan:
    """``horizon`` (optional): how many gameweeks to plan over. Defaults to
    ``cfg.transfer_planning_horizon_gws``, which is what every ordinary weekly
    call wants. ``optimiser/chips.py`` overrides it with
    ``CHIP_TIMING.wildcard_eval_horizon_gws`` so a wildcard is judged over the
    window its threshold was written for — the horizons must match or the gain
    is compared against a bar meant for a different number of gameweeks, which
    is the failure ``config.strategy.assert_horizons_consistent`` exists to
    prevent.

    ``ownership`` (P3-3, optional, default None): see ``optimise_squad``'s
    docstring — ``None`` (the current live reality — EO sampling can't
    produce real data pre-GW1) makes EO a uniform 0% for every candidate, a
    constant rescale that doesn't change which transfers get made.

    ``config``/``transfer_rules`` (optional): see ``optimise_squad``'s
    docstring — overrides the global ``OPTIMISER``/``TRANSFERS`` for this
    call only; ``None`` is byte-for-byte identical to today's behaviour.

    ``bank``/``purchase_prices`` (P1.6, optional): real FPL affordability.
    Money is tracked as a per-gameweek bank balance — selling credits the
    player's SELLING price (``selling_price``: half the rise, all of the
    fall), buying debits their current price, and the balance may never go
    negative. Omitting both is exactly equivalent to the previous
    ``Σ now_cost ≤ budget`` constraint: with no purchase prices every
    player's selling price is their current price, and the bank balance is
    then ``budget − Σ now_cost`` by construction, so ``bank ≥ 0`` is the same
    inequality. It is a generalisation, not a second code path."""
    cfg = config or OPTIMISER
    trules = transfer_rules or TRANSFERS
    horizon = horizon if horizon is not None else cfg.transfer_planning_horizon_gws
    # P1.2: `ft` has lowBound=1, so ft[0] == 0 makes the whole model
    # Infeasible -- which this function catches and reports as "no transfers",
    # indistinguishable from a genuine decision not to transfer. FPL always
    # grants at least one free transfer per gameweek, so a count below that is
    # a caller bug (it was the live decision engine's, for months); clamp and
    # say so rather than silently doing nothing.
    if free_transfers < trules.free_transfers_per_gw:
        logger.warning(
            "free_transfers=%d is below the weekly allowance (%d) — clamping. "
            "FPL never grants fewer; check the caller's roll-forward.",
            free_transfers, trules.free_transfers_per_gw,
        )
        free_transfers = trules.free_transfers_per_gw
    current_squad = players[players["id"].isin(current_squad_ids)].copy()
    squad_now_cost = float(current_squad["now_cost"].sum())
    budget = available_budget or squad_now_cost
    # P1.6: cash not tied up in players. Derived from `budget` when the caller
    # has no explicit bank, which reproduces the old constraint exactly.
    initial_bank = float(bank) if bank is not None else max(0.0, budget - squad_now_cost)
    purchase_prices = purchase_prices or {}

    gws = sorted(projections["gameweek"].unique())[:horizon]
    H = len(gws)
    if H == 0:
        return TransferPlan([], [], 0, 0.0, 0.0)

    candidate_pids = set(current_squad_ids)
    if "start_probability" in projections.columns:
        candidate_pids |= set(
            projections[projections["start_probability"] >= cfg.min_start_probability]["player_id"]
        )
    else:
        candidate_pids |= set(projections["player_id"].unique())

    df = players[players["id"].isin(candidate_pids)].copy()
    # Status filter applies to NOT-currently-owned candidates only (never buy
    # an unavailable/departed player). An OWNED player is kept regardless of
    # status so a confirmed departure (status='u') can be explicitly modelled
    # as a forced sale below, rather than silently dropped from the ILP's
    # variable set entirely — the latter left the squad-size constraint to
    # coincidentally force a replacement transfer in, but with no `tout`
    # variable for the departed player at all, so they never appeared in the
    # reported `transfers_out` (v2-build-plan §6.5 departure-risk gate fix).
    owned_mask = df["id"].isin(current_squad_ids)
    df = df[owned_mask | df["status"].isin(["a", "d"])]
    # Hand-entered hard vetoes get exactly the treatment a confirmed departure
    # gets, and for the same reason (2026-08-18). An unowned vetoed player is
    # dropped from the pool so he can never be bought; an OWNED one is kept as
    # a variable and force-sold below, because removing his variable outright
    # would leave him missing from `transfers_out` while the squad-size
    # constraint quietly conjured a replacement.
    vetoed = load_excluded_player_ids()
    if vetoed:
        df = df[df["id"].isin(current_squad_ids) | ~df["id"].isin(vetoed)]
    df = df.reset_index(drop=True)

    pid_list = df["id"].tolist()

    costs = df["now_cost"].tolist()
    positions = df["position"].tolist()
    teams = df["team_id"].tolist()
    statuses = df["status"].tolist()
    in_current = [1 if pid in set(current_squad_ids) else 0 for pid in pid_list]
    confirmed_departure_ids = {
        pid for pid, status, owned in zip(pid_list, statuses, in_current, strict=True)
        if owned and is_hard_excluded(confirmed_p_leave(status))
    }
    # An owned player under a hard veto is sold on the same terms as a departure.
    owned_ids = set(current_squad_ids)
    confirmed_departure_ids |= {
        pid for pid in pid_list if pid in vetoed and pid in owned_ids
    }

    lam, mu = lambda_mu_for_risk_level(
        cfg.risk_level, cfg.max_ownership_differential, cfg.mu_baseline, cfg.mu_range
    )
    # The optimiser's-curse correction now lives upstream, in
    # projection.assemble.apply_curse_shrinkage (2026-07-28) — `projections`
    # arrives here with `xpts` already shrunk, superseding the old P3-6
    # transfer_variance_penalty (which only covered this one call site).
    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        eo_by_pid = {pid: float(eo_map.get(pid, 0.0)) for pid in pid_list}
    else:
        eo_by_pid = dict.fromkeys(pid_list, 0.0)
    has_var = "xpts_var" in projections.columns

    # xpts_gain/net_xpts_gain reporting reads straight from `projections` via
    # squad_xpts (true expected points) -- only the ILP's own objective
    # coefficients need the risk-adjusted score.
    scores_pw: dict[tuple[int, int], float] = {}
    for w, gw in enumerate(gws):
        gw_df = projections[projections["gameweek"] == gw].set_index("player_id")
        gw_proj = gw_df["xpts"]
        gw_var = gw_df["xpts_var"] if has_var else None
        for pid in pid_list:
            x = float(gw_proj.get(pid, 0.0))
            v = float(gw_var.get(pid, 0.0)) if gw_var is not None else 0.0
            # Discounted by how far ahead the gameweek is (2026-08-18, matching
            # optimiser.squad._decay_weights). Without it a transfer justified
            # only by week three of the plan counted exactly as much as one
            # paying off this week, on a projection with no bookmaker odds
            # behind it. `hit_cost_points` and the switching cost are NOT
            # decayed: those are paid in real points, now, whatever the plan
            # later turns out to be worth.
            scores_pw[(pid, w)] = (
                cfg.gameweek_decay ** w
            ) * risk_adjusted_score(x, v, eo_by_pid[pid], lam, mu)

    prob = pulp.LpProblem("mp_transfers", pulp.LpMaximize)

    squad = {
        (pid, w): pulp.LpVariable(f"sq_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    starting = {
        (pid, w): pulp.LpVariable(f"st_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    tin = {
        (pid, w): pulp.LpVariable(f"ti_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    tout = {
        (pid, w): pulp.LpVariable(f"to_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    captain = {
        (pid, w): pulp.LpVariable(f"cp_{pid}_{w}", cat="Binary")
        for pid in pid_list for w in range(H)
    }
    # upBound is the WILDCARD allowance (15), not the banking cap (P1.3): the
    # wildcard branch below pins ft[0] == 15, which against the old upBound=5
    # made the whole model Infeasible -- caught and returned as an empty plan
    # behind a warning, so a played wildcard produced zero transfers while
    # still being recorded as used. Ordinary weeks are still capped at
    # max_banked_free_transfers by the ft[w+1] constraint in the loop.
    ft = {
        w: pulp.LpVariable(f"ft_{w}", lowBound=1, upBound=SQUAD.squad_size, cat="Integer")
        for w in range(H + 1)
    }
    hit = {w: pulp.LpVariable(f"hit_{w}", lowBound=0, upBound=15, cat="Integer") for w in range(H)}
    n_trans = {
        w: pulp.LpVariable(f"nt_{w}", lowBound=0, upBound=15, cat="Integer") for w in range(H)
    }
    # P1.6: cash in the bank at the START of each week. lowBound=0 IS the
    # affordability constraint -- you cannot spend money you do not have.
    bank_var = {w: pulp.LpVariable(f"bank_{w}", lowBound=0) for w in range(H + 1)}
    prob += bank_var[0] == initial_bank

    cost_by_pid = dict(zip(pid_list, costs, strict=True))
    # A player already owned sells for their selling price; anyone else is
    # bought at (and, absent price-change modelling, resells at) their current
    # price. With no purchase prices supplied every player sells at now_cost,
    # which collapses the cash flow back to the old squad-cost constraint.
    sell_by_pid = {
        pid: selling_price(purchase_prices[pid], cost) if pid in purchase_prices else cost
        for pid, cost in cost_by_pid.items()
    }

    if wildcard_active:
        prob += ft[0] == SQUAD.squad_size
    else:
        prob += ft[0] == free_transfers

    # Departure-risk gate (§6.5): a confirmed departure (status='u') already
    # owned must be sold immediately, not merely made ineligible to buy back
    # (the generic "can't tin an owned player at w=0" constraint below
    # already prevents re-buying) — forcing tout==1 here (rather than
    # omitting them from the model, the prior behaviour) means they
    # correctly show up in the reported transfers_out and go through the
    # normal hit/FT accounting.
    for pid in confirmed_departure_ids:
        prob += tout[(pid, 0)] == 1

    for w in range(H):
        for i, pid in enumerate(pid_list):
            if w == 0:
                prob += squad[(pid, w)] == in_current[i] + tin[(pid, w)] - tout[(pid, w)]
            else:
                prob += squad[(pid, w)] == squad[(pid, w - 1)] + tin[(pid, w)] - tout[(pid, w)]

            prob += tin[(pid, w)] + tout[(pid, w)] <= 1

            if in_current[i] == 0 and w == 0:
                prob += tout[(pid, w)] == 0
            if in_current[i] == 1 and w == 0:
                prob += tin[(pid, w)] == 0

            prob += captain[(pid, w)] <= starting[(pid, w)]
            prob += starting[(pid, w)] <= squad[(pid, w)]

        prob += pulp.lpSum(squad[(pid, w)] for pid in pid_list) == SQUAD.squad_size
        prob += pulp.lpSum(starting[(pid, w)] for pid in pid_list) == 11
        # P1.6: affordability as a real cash flow -- selling credits the
        # SELLING price, buying debits the current price, and the balance can
        # never go negative. Replaces `Σ now_cost * squad <= budget`, which
        # valued a kept player at their (possibly risen) current price and so
        # believed an appreciated squad had spending power it does not have.
        prob += bank_var[w + 1] == (
            bank_var[w]
            + pulp.lpSum(sell_by_pid[pid] * tout[(pid, w)] for pid in pid_list)
            - pulp.lpSum(cost_by_pid[pid] * tin[(pid, w)] for pid in pid_list)
        )

        for pos, count in SQUAD_COUNTS.items():
            pos_pids = [pid for pid, p in zip(pid_list, positions) if p == pos]
            prob += pulp.lpSum(squad[(pid, w)] for pid in pos_pids) == count
            prob += pulp.lpSum(starting[(pid, w)] for pid in pos_pids) >= STARTING_MIN[pos]
            prob += pulp.lpSum(starting[(pid, w)] for pid in pos_pids) <= STARTING_MAX[pos]

        team_ids = list(set(teams))
        for tid in team_ids:
            tid_pids = [pid for pid, t in zip(pid_list, teams) if t == tid]
            prob += pulp.lpSum(squad[(pid, w)] for pid in tid_pids) <= SQUAD.max_players_per_club

        prob += pulp.lpSum(tin[(pid, w)] for pid in pid_list) == pulp.lpSum(
            tout[(pid, w)] for pid in pid_list
        )
        prob += n_trans[w] == pulp.lpSum(tin[(pid, w)] for pid in pid_list)
        prob += pulp.lpSum(captain[(pid, w)] for pid in pid_list) == 1

        prob += hit[w] >= n_trans[w] - ft[w]

        if wildcard_active and w == 0:
            prob += hit[0] == 0
        else:
            prob += hit[w] <= trules.max_hits_per_gw

        # P1.4: `+ hit[w]` is the slack that makes HITS LEGAL. Without it,
        # `ft[w+1] <= ft[w] - n_trans[w] + 1` combined with `ft[w+1] >= 1`
        # forced n_trans[w] <= ft[w] in every week, so the model could never
        # take a hit at all -- verified against three candidates worth +29
        # xPts/GW each with one free transfer: it took one transfer and zero
        # hits. Transfers beyond the allowance are paid for in points (the
        # objective's hit term), not out of next week's allowance.
        prob += ft[w + 1] <= ft[w] - n_trans[w] + hit[w] + trules.free_transfers_per_gw
        prob += ft[w + 1] <= trules.max_banked_free_transfers
        prob += ft[w + 1] >= trules.free_transfers_per_gw

    def _switching_cost(w: int) -> float:
        """The flat per-transfer cost, zeroed on a wildcard week (2026-08-18,
        engine review §11).

        ``transfer_switching_cost`` exists for one specific purpose, set out at
        length in ``strategy.py``: stop a noise-sized edge triggering churn
        WITHIN the free allowance. A wildcard has no allowance to churn against
        — unlimited transfers are the entire point of the chip, and it is
        scarce (one per half). Charging the tax anyway docked a ten-player
        rebuild 15 points against its own objective, so the solver
        systematically under-used a chip it had just decided to spend.

        The hit term is already disabled for exactly this reason a few lines
        above (``hit[0] == 0``); this is the other half of the same idea, which
        was missed.
        """
        return 0.0 if (wildcard_active and w == 0) else trules.transfer_switching_cost

    # P1.7: bench players contributed exactly nothing here, so every transfer
    # treated bench quality as worthless and eroded it to fodder over a season
    # -- optimise_squad already weights the bench (bench_value_weight) and this
    # is the same term, so an in-season transfer can no longer undo what the
    # squad build deliberately paid for.
    # Slot-weighted, matching optimiser.squad._bench_objective exactly
    # (2026-08-18). The two must agree: the squad build now pays real money
    # for a first substitute who plays, and if this ILP still valued every
    # bench player at a flat 0.15 it would sell him again the following week
    # -- undoing the purchase and charging a transfer for the privilege.
    # Keeping them in sync is the whole point of this term existing.
    pos_of = dict(zip(pid_list, positions, strict=True))
    outfield_pids = [pid for pid in pid_list if pos_of[pid] != "GKP"]
    keeper_pids = [pid for pid in pid_list if pos_of[pid] == "GKP"]
    bench_slot_w = [w * cfg.bench_value_weight for w in cfg.bench_slot_weights]
    bench_gk_w = cfg.bench_gk_weight * cfg.bench_value_weight

    bslot = {
        (pid, w, k): pulp.LpVariable(f"bslot_{pid}_{w}_{k}", cat="Binary")
        for pid in outfield_pids
        for w in range(H)
        for k in range(len(bench_slot_w))
    }
    for w in range(H):
        for pid in outfield_pids:
            prob += (
                pulp.lpSum(bslot[(pid, w, k)] for k in range(len(bench_slot_w)))
                == squad[(pid, w)] - starting[(pid, w)]
            )
        for k in range(len(bench_slot_w)):
            prob += pulp.lpSum(bslot[(pid, w, k)] for pid in outfield_pids) == 1

    prob += pulp.lpSum(
        scores_pw[(pid, w)] * starting[(pid, w)]
        + scores_pw[(pid, w)] * captain[(pid, w)]
        for pid in pid_list
        for w in range(H)
    ) + pulp.lpSum(
        bench_slot_w[k] * scores_pw[(pid, w)] * bslot[(pid, w, k)]
        for pid in outfield_pids
        for w in range(H)
        for k in range(len(bench_slot_w))
    ) + pulp.lpSum(
        bench_gk_w * scores_pw[(pid, w)] * (squad[(pid, w)] - starting[(pid, w)])
        for pid in keeper_pids
        for w in range(H)
    ) + trules.ft_terminal_value * ft[H] - pulp.lpSum(
        # P1.4: both of these used to sit INSIDE the per-player loop above, so
        # a single hit cost `4 * len(pid_list)` (~900+ points at real pool
        # sizes) rather than 4. Masked while hits were structurally impossible;
        # it would have made them impossible again the moment they were legal.
        hit[w] * abs(trules.hit_cost_points) + _switching_cost(w) * n_trans[w]
        for w in range(H)
    )

    solver_args = {"msg": False}
    if solver_time_limit:
        solver_args["timeLimit"] = solver_time_limit
    prob.solve(pulp.PULP_CBC_CMD(**solver_args))

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Not Solved"):
        logger.warning("Multi-period ILP status: %s — returning no transfers", status)
        return TransferPlan([], [], 0, 0.0, 0.0)

    gw0_in = [pid for pid in pid_list if (pulp.value(tin[(pid, 0)]) or 0) > 0.5]
    gw0_out = [pid for pid in pid_list if (pulp.value(tout[(pid, 0)]) or 0) > 0.5]

    actual_hits = max(
        0, len(gw0_in) - (SQUAD.squad_size if wildcard_active else free_transfers)
    )

    new_squad_ids = [pid for pid in current_squad_ids if pid not in set(gw0_out)] + gw0_in
    xpts_after = squad_xpts(new_squad_ids, projections, horizon)
    xpts_before = squad_xpts(current_squad_ids, projections, horizon)
    xpts_gain = xpts_after - xpts_before
    net_xpts_gain = xpts_gain + actual_hits * trules.hit_cost_points

    def _player_info(pid: int) -> dict:
        row = players[players["id"] == pid]
        return {
            "player_id": pid,
            "web_name": row["web_name"].values[0] if len(row) else str(pid),
            "cost": float(row["now_cost"].values[0]) if len(row) else 0.0,
        }

    plan = TransferPlan(
        transfers_in=[_player_info(pid) for pid in gw0_in],
        transfers_out=[_player_info(pid) for pid in gw0_out],
        hits_taken=actual_hits,
        xpts_gain=xpts_gain,
        net_xpts_gain=net_xpts_gain,
    )

    logger.info(
        "Transfer plan: %d in / %d out, %d hits, net gain %.2f xPts [ILP status: %s]",
        len(plan.transfers_in), len(plan.transfers_out),
        plan.hits_taken, plan.net_xpts_gain, status,
    )
    return plan


def get_dgw_coverage(
    squad_ids: list[int],
    players: pd.DataFrame,
    dgw_gws: set[int],
    projections: pd.DataFrame,
) -> dict[int, dict]:
    """Real bug found 2026-07-30 (the user's own live-smoke-test request):
    ``agent/decision_engine.py`` has imported and called this function
    since its very first commit, but it never actually existed anywhere in
    the codebase — an ``ImportError`` at module load, meaning the live
    decision engine could never run at all, this whole time, undetected
    because nothing in the backtest-focused test suite ever imports
    ``agent.decision_engine``.

    For each known upcoming double-gameweek, how many of the CURRENT
    squad's players are on a team with two real fixtures that gameweek
    (from the ``fixtures`` table — the actual schedule, not an
    approximation), and their combined projected xPts — lets a human
    reviewer see DGW exposure at a glance without cross-referencing the
    fixture list themselves. Real, not free-text: a team with 2 fixtures
    that gameweek is a genuine DGW for every squad player on it."""
    if not dgw_gws or not squad_ids or players.empty:
        return {}

    from sqlalchemy import text

    from data.db import get_session
    from projection.pipeline import _get_current_season

    team_by_player = players.set_index("id")["team_id"].to_dict()
    season = _get_current_season()

    db = get_session()
    try:
        coverage: dict[int, dict] = {}
        for gw in sorted(dgw_gws):
            rows = db.execute(
                text(
                    "SELECT team_h_id, team_a_id FROM fixtures "
                    "WHERE season = :season AND gameweek = :gw"
                ),
                {"season": season, "gw": gw},
            ).fetchall()
            fixture_count: dict[int, int] = {}
            for team_h_id, team_a_id in rows:
                fixture_count[team_h_id] = fixture_count.get(team_h_id, 0) + 1
                fixture_count[team_a_id] = fixture_count.get(team_a_id, 0) + 1
            dgw_teams = {tid for tid, n in fixture_count.items() if n >= 2}
            dgw_player_ids = [
                pid for pid in squad_ids if team_by_player.get(pid) in dgw_teams
            ]
            combined_xpts = float(
                projections[
                    (projections["gameweek"] == gw)
                    & projections["player_id"].isin(dgw_player_ids)
                ]["xpts"].sum()
            )
            coverage[gw] = {
                "squad_players_involved": len(dgw_player_ids),
                "combined_xpts": round(combined_xpts, 2),
            }
        return coverage
    finally:
        db.close()
