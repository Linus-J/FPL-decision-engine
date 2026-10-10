import dataclasses
import logging
from dataclasses import dataclass

import pandas as pd
import pulp

from config.strategy import OPTIMISER, SQUAD, OptimiserConfig
from optimiser.bench_weights import (
    bench_config_for_xi,
    bench_gk_weight_for_week,
    bench_slot_weight_for_week,
    blank_probability_map,
)
from optimiser.captaincy import scenario_based_captain
from optimiser.scoring import lambda_mu_for_risk_level, risk_adjusted_score

logger = logging.getLogger(__name__)

POSITIONS = ("GKP", "DEF", "MID", "FWD")

SQUAD_COUNTS = {
    "GKP": SQUAD.gk_count,
    "DEF": SQUAD.def_count,
    "MID": SQUAD.mid_count,
    "FWD": SQUAD.fwd_count,
}

STARTING_MIN = {
    "GKP": SQUAD.starting_gk,
    "DEF": SQUAD.starting_def_min,
    "MID": SQUAD.starting_mid_min,
    "FWD": SQUAD.starting_fwd_min,
}

STARTING_MAX = {
    "GKP": SQUAD.starting_gk,
    "DEF": SQUAD.starting_def_max,
    "MID": SQUAD.starting_mid_max,
    "FWD": SQUAD.starting_fwd_max,
}


class SolverTimeout(RuntimeError):
    """A solve stopped on its time limit rather than proving infeasibility.

    Subclasses RuntimeError so every existing `except RuntimeError` keeps
    working. The distinction matters because generate_squad_pool treats a
    RuntimeError as "the cuts have exhausted the legal squads" and stops
    quietly — which, for a timeout, silently returns a short pool.
    """


def _cbc(cfg: OptimiserConfig) -> pulp.PULP_CBC_CMD:
    """CBC configured with the configured wall-clock limit.

    A falsy ``solver_time_limit_seconds`` passes no ``timeLimit`` to
    ``PULP_CBC_CMD`` at all -- CBC treats an explicit ``0`` as an immediate
    stop rather than "no limit", so disabling the limit means omitting the
    argument, not passing a zero value.
    """
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=cfg.solver_time_limit_seconds or None)


def _raise_if_not_optimal(
    status: str, cfg: OptimiserConfig, label: str, sol_status: int | None = None
) -> None:
    """Raise on a non-optimal CBC result, distinguishing a timeout from a
    genuinely infeasible problem.

    The status STRING is not sufficient. pulp rewrites a timed-out-with-
    incumbent CBC result to "Optimal" and records the truth in `sol_status`
    (coin_api.py:363-366), so checking the string alone silently accepts a
    suboptimal squad as the answer -- which is the case this guard exists for.
    CBC reports "Not Solved" only when it stops on the time limit WITHOUT any
    incumbent at all; that is also not the same as Infeasible.
    """
    if status == "Optimal":
        if sol_status == pulp.constants.LpSolutionIntegerFeasible:
            logger.error(
                "%s stopped on its %.0fs time limit with a feasible but "
                "non-optimal solution", label, cfg.solver_time_limit_seconds,
            )
            raise SolverTimeout(f"{label} stopped on its time limit (integer feasible)")
        return
    if status == "Not Solved":
        logger.error(
            "%s stopped on its %.0fs time limit; this is a timeout, not an "
            "infeasible problem", label, cfg.solver_time_limit_seconds,
        )
        raise SolverTimeout(f"{label} stopped on its time limit ({status})")
    raise RuntimeError(f"{label} did not find optimal solution: {status}")


def _solve(prob: pulp.LpProblem, cfg: OptimiserConfig, label: str) -> None:
    """Solve ``prob`` with CBC under the configured wall-clock limit, then
    raise on any non-Optimal status. See ``_raise_if_not_optimal``."""
    prob.solve(_cbc(cfg))
    _raise_if_not_optimal(pulp.LpStatus[prob.status], cfg, label, sol_status=prob.sol_status)


@dataclass
class SquadSolution:
    squad: pd.DataFrame
    starting_xi: pd.DataFrame
    captain_id: int
    vice_captain_id: int
    total_xpts: float
    total_cost: float
    hits_taken: int


def _decay_weights(gws: list, decay: float) -> dict:
    """``decay ** i`` for the i-th gameweek in the horizon, nearest first.

    A five-gameweek horizon summed with EQUAL weight treats a projection five
    weeks out as being worth exactly as much as the one for the match about to
    kick off. It is not: bookmakers price roughly one round ahead, so on the
    live GW1 frame 22% of a squad's projected points came from gameweeks with
    real odds and 78% from the strength model with most teams still on
    prior-season fallback. Equal weighting puts full confidence in the least
    reliable numbers in the system.

    Every serious FPL optimiser discounts. Sertalp Çay's
    ``solve_multi_period_fpl`` defaults to ``decay_base = 0.84`` and
    FPLReview's solvers recommend 0.80-0.95; 0.85 sits in the middle of both.

    This is a weighting on the OBJECTIVE only. Reported ``total_xpts`` stays
    the true undiscounted sum — the same separation the risk-adjusted score
    already keeps between ``effective_score`` and ``xpts_total``, and for the
    same reason: a decision aid must not quietly restate the thing it is
    predicting.
    """
    return {gw: decay ** i for i, gw in enumerate(gws)}


def _multi_gw_xpts(projections: pd.DataFrame, horizon: int, decay: float = 1.0) -> pd.Series:
    gws = sorted(projections["gameweek"].unique())[:horizon]
    subset = projections[projections["gameweek"].isin(gws)]
    if decay == 1.0:
        return subset.groupby("player_id")["xpts"].sum()
    weights = subset["gameweek"].map(_decay_weights(gws, decay))
    return (subset["xpts"] * weights).groupby(subset["player_id"]).sum()


def _multi_gw_var(projections: pd.DataFrame, horizon: int, decay: float = 1.0) -> pd.Series:
    """P3-3: per-player summed xpts_var over the horizon (own-variance only —
    see optimiser/scoring.py for why teammate covariance isn't here). Empty
    Series (not an error) if the caller's projections predate P10's
    distributional columns."""
    if "xpts_var" not in projections.columns:
        return pd.Series(dtype=float)
    gws = sorted(projections["gameweek"].unique())[:horizon]
    subset = projections[projections["gameweek"].isin(gws)]
    if decay == 1.0:
        return subset.groupby("player_id")["xpts_var"].sum()
    weights = subset["gameweek"].map(_decay_weights(gws, decay))
    return (subset["xpts_var"] * weights).groupby(subset["player_id"]).sum()


def _multi_gw_semidev(
    projections: pd.DataFrame, horizon: int, col: str, decay: float = 1.0
) -> pd.Series:
    """Per-player summed upper/lower semi-deviation over the horizon.

    Summed, like ``xpts``, rather than combined in quadrature: these feed a
    LINEAR objective alongside summed points, so they have to be on the same
    footing. Treating them as independent and adding variances would make a
    five-gameweek risk term incomparable to a five-gameweek points term.
    """
    if col not in projections.columns:
        return pd.Series(dtype=float)
    gws = sorted(projections["gameweek"].unique())[:horizon]
    subset = projections[projections["gameweek"].isin(gws)]
    if decay == 1.0:
        return subset.groupby("player_id")[col].sum()
    # Decayed LINEARLY, like the points they sit beside in the objective, not
    # quadratically as a scaled variance would be. These are semi-deviations
    # in points; the whole reason they replaced a variance term is that the
    # objective adds them to a points total directly.
    weights = subset["gameweek"].map(_decay_weights(gws, decay))
    return (subset[col] * weights).groupby(subset["player_id"]).sum()


def _semidev_by_id(df, mu: float) -> dict | None:
    """The tail that matches the appetite: upper semi-deviation when chasing
    upside, lower when avoiding blanks. ``None`` when the frame carries
    neither, so captaincy keeps its covariance-aware behaviour."""
    col = "upside" if mu >= 0 else "downside"
    if col not in df.columns or df[col].isna().all():
        return None
    return dict(zip(df["id"], df[col].fillna(0.0), strict=True))


def _per_week_scores(
    projections: pd.DataFrame,
    player_ids: list[int],
    horizon: int,
    decay: float,
    lam: float,
    mu: float,
    eo_by_pid: dict[int, float],
) -> dict[tuple[int, int], float]:
    """Risk-adjusted score per (player_id, week index), decayed by ``decay ** w``.

    Deliberately the same shape as ``transfers.py``'s ``scores_pw``. Until
    2026-09-06 this function's job was done by ``_multi_gw_xpts``, which summed
    the horizon into ONE number per player before the ILP ever saw it — so the
    solver could not tell a player who is excellent in week 1 and useless in
    week 3 from one who is mediocre in both. That is fine for deciding who to
    own and wrong for deciding who to start.

    Each week's inputs are grouped and summed exactly the way ``_multi_gw_xpts``
    and friends aggregate the whole horizon — an absent player or an absent
    column reading 0.0, a NaN reading as the group sum does. That is what makes
    ``horizon == 1`` reproduce the collapsed coefficient to the last bit.
    ``risk_adjusted_score`` is linear in both points and the risk term, so the
    per-week decayed scores also still sum to the collapsed one at any horizon
    whenever the XI does not move — the reformulation adds freedom, it does not
    re-weight anything.

    ``lam``/``mu`` are taken as parameters, not recomputed here (2026-09-07):
    ``optimise_squad`` also uses them for ``effective_score``, which drives the
    scenario-based captain pick, and the two must agree or the ILP's captain
    variable and ``scenario_based_captain`` would be answering different
    questions.
    """
    gws = sorted(projections["gameweek"].unique())[:horizon]
    # An empty projection frame still has to produce a solvable model, which is
    # what it produced before: one week, every coefficient zero.
    weeks = max(1, len(gws))
    scores = {(pid, w): 0.0 for pid in player_ids for w in range(weeks)}

    def _summed(gw_df: pd.DataFrame, col: str) -> pd.Series | None:
        return gw_df.groupby("player_id")[col].sum() if col in gw_df.columns else None

    def _at(series: pd.Series | None, pid: int) -> float:
        return float(series.get(pid, 0.0)) if series is not None else 0.0

    for w, gw in enumerate(gws):
        gw_df = projections[projections["gameweek"] == gw]
        xpts = _summed(gw_df, "xpts")
        var = _summed(gw_df, "xpts_var")
        # Absent semi-deviation columns read as 0.0 rather than None, so the
        # risk term does NOT silently fall back to `xpts_var` — the same choice
        # `optimise_squad` has documented since 2026-08-18, and the reason `mu`
        # is inert rather than mis-scaled on frames that carry no semi-deviations.
        up = _summed(gw_df, "upside")
        down = _summed(gw_df, "downside")
        for pid in player_ids:
            scores[(pid, w)] = (decay ** w) * risk_adjusted_score(
                _at(xpts, pid), _at(var, pid), eo_by_pid.get(pid, 0.0), lam, mu,
                _at(up, pid), _at(down, pid),
            )
    return scores


def _xi_objective(
    selected: list,
    starting: dict,
    captain: dict,
    vice: dict,
    scores_pw: dict,
    player_ids: list[int],
    weeks: int,
    cfg: OptimiserConfig,
):
    """The starting-XI half of the objective, summed over every week.

    Factored out 2026-09-06. This expression was written twice — once for the
    primary solve and once for the `max_transfers` fallback — and per-week
    structure landed on it the same day. `_bench_objective` was already shared
    for exactly this reason; the XI half was missed.

    `starting`/`captain`/`vice` are keyed (index, week). At `weeks == 1` this is
    arithmetically identical to the previous single-week form, which is what the
    H=1 equivalence test pins.

    `selected` is unused today and is taken anyway so the signature does not
    have to change when a term needs it.
    """
    return pulp.lpSum(
        scores_pw[(player_ids[i], w)] * (starting[(i, w)] + captain[(i, w)])
        # The armband only passes to the vice when the captain does not
        # feature, so it is worth P(captain blanks) x his score. Without this
        # term `vice` is constrained but unvalued, every legal choice ties, and
        # the solver returns whichever it branched on -- a goalkeeper, on the
        # live frame, while a 7.43-xPts defender sat in the same XI.
        + cfg.vice_captain_weight * scores_pw[(player_ids[i], w)] * vice[(i, w)]
        for i in range(len(player_ids))
        for w in range(weeks)
    )


def _bench_objective(
    prob: pulp.LpProblem,
    selected: list,
    starting: dict,
    scores_pw: dict,
    player_ids: list[int],
    positions: list[str],
    weeks: int,
    cfg: OptimiserConfig,
    tag: str,
    bb_target_week: int | None = None,
):
    """Objective terms for the bench, weighted by SLOT rather than uniformly.

    A bench is an ordered queue. FPL's automatic substitutions promote the
    first eligible bench player when a starter does not appear, so the value
    of slot 1 is P(at least one starter blanks) -- 0.53 on the live GW1 XI --
    while slot 3 needs three simultaneous absences and is worth 0.03. A flat
    weight underpays the slot that gets used and overpays the two that do not,
    which is what buys four £4.0m non-players instead of one real substitute.

    Per-week since 2026-09-06, alongside the XI: the squad is fixed across the
    horizon but who sits on the bench is not, so `selected[i] - starting[(i, w)]`
    is the week-`w` bench indicator and the slot assignment is made afresh each
    week.

    ``bb_target_week`` (2026-09-06): the week, if any, a Bench Boost is being
    priced for. That week values every bench slot and the reserve keeper at
    1.0 -- under the chip they all genuinely play -- via
    ``optimiser.bench_weights.bench_slot_weight_for_week`` /
    ``bench_gk_weight_for_week``, shared with ``optimiser/transfers.py`` so the
    two ILPs never price the bench differently. ``None`` reproduces the
    ordinary auto-substitution weights in every week, exactly as before this
    parameter existed.

    Adds the slot-assignment constraints to ``prob`` and returns the terms to
    add to the objective. Slot weights are strictly decreasing outside the
    target week, so the solver puts its best bench player in slot 1 without
    needing an ordering constraint -- it is a maximisation and any other
    assignment is dominated. In the target week every slot ties at 1.0, so any
    assignment is equally optimal.
    """
    n = len(player_ids)
    n_slots = len(cfg.bench_slot_weights)
    outfield = [i for i in range(n) if positions[i] != "GKP"]
    keepers = [i for i in range(n) if positions[i] == "GKP"]

    slot = {
        (i, k, w): pulp.LpVariable(f"bench_{tag}_{i}_{k}_{w}", cat="Binary")
        for i in outfield
        for k in range(n_slots)
        for w in range(weeks)
    }
    for w in range(weeks):
        for i in outfield:
            prob += (
                pulp.lpSum(slot[i, k, w] for k in range(n_slots))
                == selected[i] - starting[(i, w)]
            )
        for k in range(n_slots):
            prob += pulp.lpSum(slot[i, k, w] for i in outfield) == 1

    terms = pulp.lpSum(
        bench_slot_weight_for_week(k, w, bb_target_week, cfg)
        * scores_pw[(player_ids[i], w)]
        * slot[i, k, w]
        for i in outfield
        for k in range(n_slots)
        for w in range(weeks)
    )
    # The reserve keeper has no queue to inherit from: he plays only if the
    # first-choice keeper does not.
    terms += pulp.lpSum(
        bench_gk_weight_for_week(w, bb_target_week, cfg)
        * scores_pw[(player_ids[i], w)]
        * (selected[i] - starting[(i, w)])
        for i in keepers
        for w in range(weeks)
    )
    return terms


def _add_no_good_cuts(
    prob: pulp.LpProblem,
    selected: list,
    idx: dict,
    forbidden_squads: list[list[int]] | None,
) -> None:
    """Forbid each given squad exactly, without forbidding any of its members.

    ``sum(selected[i] for i in S) <= len(S) - 1`` rules out picking all fifteen
    of S again while leaving every fourteen-man subset legal. Re-solving with
    one of these added yields the next-best squad, which is how a solution pool
    is built on a solver with no native pool support (CBC); Gurobi and CPLEX
    expose the same idea directly as PoolSearchMode / populate.

    A squad whose members are not all still in the candidate pool is skipped
    rather than cut down: it is already unreachable, and a cut written over the
    survivors would forbid legal squads that merely resemble it.
    """
    for squad_ids in forbidden_squads or []:
        positions_in_pool = [idx[pid] for pid in squad_ids if pid in idx]
        if len(positions_in_pool) != len(squad_ids):
            continue
        prob += pulp.lpSum(selected[i] for i in positions_in_pool) <= len(squad_ids) - 1


def generate_squad_pool(
    projections: pd.DataFrame,
    players: pd.DataFrame,
    n: int = 10,
    **kwargs,
) -> list[SquadSolution]:
    """The ``n`` best distinct squads, best first.

    Exists because a single answer hides how much of itself is real. A squad
    reported alone cannot say whether its 12th pick beat the alternative by
    four points or by two hundredths, and the difference is the difference
    between a conviction and a coin toss. Reading which players survive across
    the whole pool is a far better confidence signal than any single solve:
    a player in all ten squads is one the model genuinely wants, and one in
    three of ten is the model shrugging.

    Stops early and returns what it has if the problem becomes infeasible,
    which it will once the cuts exhaust the legal squads in a small pool.
    """
    pool: list[SquadSolution] = []
    forbidden: list[list[int]] = list(kwargs.pop("forbidden_squads", None) or [])
    for _ in range(max(0, n)):
        try:
            solution = optimise_squad(
                projections, players, forbidden_squads=forbidden, **kwargs
            )
        except SolverTimeout:
            # A timeout is not an exhausted pool. Propagating it means a caller
            # sees a real failure instead of a silently short list of squads.
            raise
        except RuntimeError:
            break
        pool.append(solution)
        forbidden.append(sorted(int(pid) for pid in solution.squad["id"]))
    return pool


def optimise_squad_joint(
    projections: pd.DataFrame,
    players: pd.DataFrame,
    *,
    season: str | None = None,
    gameweek: int | None = None,
    sample_rows: list | None = None,
    matrices=None,
    config: OptimiserConfig | None = None,
    pool_size: int | None = None,
    pool: list[SquadSolution] | None = None,
    **kwargs,
) -> SquadSolution:
    """``optimise_squad``, then a covariance-aware re-rank of the pool.

    The pool is always generated at ``mu = 0`` -- the pure-mean objective --
    and the risk appetite is applied afterwards by ``joint_risk``. That keeps
    one pool valid for every ``mu`` in a calibration sweep, and makes ``mu = 0``
    an exact no-op against the linear objective's own pick.

    ``sample_rows`` is the backtest's route in: the ``sample_sink`` list from
    ``assemble.assemble_gw_projections``. Live callers pass ``season`` and
    ``gameweek`` instead, and the samples are read from ``projection_samples``.

    ``pool`` lets a caller supply an already-generated pool. A calibration
    sweep needs this: the pool depends only on the pure-mean objective, so
    regenerating it for every candidate ``mu`` would repeat identical MILP
    solves once per candidate -- 6x the work for the same answer.
    """
    from optimiser import joint_risk

    cfg = config or OPTIMISER
    _, mu = lambda_mu_for_risk_level(
        cfg.risk_level, cfg.max_ownership_differential, cfg.mu_baseline, cfg.mu_range
    )
    mean_cfg = dataclasses.replace(cfg, risk_level=0.0, mu_baseline=0.0, mu_range=0.0)
    n = pool_size if pool_size is not None else cfg.joint_rerank_pool_size
    horizon = kwargs.get("horizon") or cfg.transfer_planning_horizon_gws

    if pool is None:
        pool = generate_squad_pool(projections, players, n=n, config=mean_cfg, **kwargs)
    if not pool:
        raise RuntimeError("optimise_squad_joint: no feasible squad")

    if matrices is None and sample_rows is not None and gameweek is not None:
        matrices = joint_risk.matrices_from_rows(
            sample_rows, range(gameweek, gameweek + horizon)
        )

    return joint_risk.covariance_aware_squad(
        pool, mu=mu, cfg=cfg, season=season, gameweek=gameweek,
        matrices=matrices, horizon=horizon, decay=cfg.gameweek_decay,
    )


def optimise_squad(
    projections: pd.DataFrame,
    players: pd.DataFrame,
    budget: float = SQUAD.budget_total,
    horizon: int | None = None,
    current_squad_ids: list[int] | None = None,
    free_transfers: int = 1,
    max_transfers: int | None = None,
    force_include_ids: list[int] | None = None,
    force_exclude_ids: list[int] | None = None,
    ownership: pd.DataFrame | None = None,
    season: str | None = None,
    config: OptimiserConfig | None = None,
    forbidden_squads: list[list[int]] | None = None,
    bb_target_week: int | None = None,
) -> SquadSolution:
    """``bb_target_week`` (optional): the week index (0 = the earliest week in
    the horizon), if any, a Bench Boost is being built for. That week values
    the bench at full weight in ``_bench_objective``; every other week keeps
    the ordinary auto-substitution weights. ``None`` (every call site today)
    is byte-for-byte identical to before this parameter existed.

    ``ownership`` (P3-3, optional): a ``(player_id, top10k_selected_pct)``
    frame (P3-2) feeding the risk-adjusted objective's differential term.
    ``None`` (the current live reality — EO sampling can't produce real data
    pre-GW1) makes every player's EO 0%, which is a uniform rescale of the
    objective, not a ranking change — behaviour is identical to before
    this parameter existed.

    ``season`` (P3-4, optional): enables scenario-based captaincy (see
    ``optimiser/captaincy.py``) for the EARLIEST gameweek in the horizon —
    the one whose captain choice is actually about to be locked in.
    ``None`` (most callers — this function builds/rebuilds a SQUAD; final
    captaincy is usually decided later, per-GW, by ``optimise_starting_xi``)
    keeps the plain linear-argmax pick.

    ``config`` (optional): overrides the global ``OPTIMISER`` singleton for
    this call only — used by the simulation engine to run the same
    optimiser under a different risk posture per persona. ``None`` (every
    real-bot call site) is byte-for-byte identical to reading the global
    directly, as before this parameter existed."""
    # The XI is chosen ONCE PER GAMEWEEK in the horizon (2026-09-06); only
    # week 0's is reported, because week 0 is the one being locked in. The
    # squad itself is a single horizon-wide choice, so the later weeks exist
    # to price what the fifteen will be able to DO, not to plan transfers.
    # Before this, the horizon was collapsed to one number per player and one
    # XI covering all of it, so two players whose good fixtures alternate
    # earned nothing for alternating — the exact case a wildcard or a cold
    # start is deciding. ``optimiser/transfers.py`` never had this defect.
    cfg = config or OPTIMISER
    horizon = horizon or cfg.transfer_planning_horizon_gws
    force_include_ids = set(force_include_ids or [])
    force_exclude_ids = set(force_exclude_ids or [])
    horizon_gws = sorted(projections["gameweek"].unique())[:horizon]
    target_gw = horizon_gws[0] if horizon_gws else None

    # The bench-boost override is `if w == bb_target_week` (bench_weights.py),
    # so an index past the last week in this slice matches nothing and silently
    # changes no coefficient -- the same shape of failure as the bb_ready_config
    # no-op this replaced. `bb_horizon_and_index` derives the index by
    # ARITHMETIC (target_gw - next_gw) while the slice above is by POSITION;
    # they agree only while the frame's gameweeks are consecutive. Raise rather
    # than quietly build an unboosted squad and report it as boost-ready.
    if bb_target_week is not None and not 0 <= bb_target_week < len(horizon_gws):
        raise ValueError(
            f"bb_target_week={bb_target_week} is outside the {len(horizon_gws)}-week "
            f"horizon {horizon_gws}; it would price no week at full bench weight"
        )

    # Two aggregates over the same horizon (2026-08-18): the TRUE sum, which
    # is what gets reported, and the decayed sum. See _decay_weights — a
    # projection five gameweeks out is not worth as much as one for the match
    # about to kick off, and every serious FPL optimiser discounts it.
    # Reporting the decayed figure as "expected points" would be a lie about
    # the quantity being predicted.
    #
    # The ILP itself reads neither of these any more (2026-09-06): its
    # coefficients come from `_per_week_scores`, which applies the same decay
    # week by week instead of summing it away first. The decayed collapse
    # survives here because the scenario-based captain pick below ranks on it,
    # and it must keep answering the same question the solver's own captain
    # variable does.
    decay = cfg.gameweek_decay
    xpts_by_player = _multi_gw_xpts(projections, horizon)
    xpts_for_objective = _multi_gw_xpts(projections, horizon, decay)
    var_by_player = _multi_gw_var(projections, horizon, decay)

    df = players.copy()
    df = df[df["status"].isin(["a", "d"])]
    if "start_probability" in df.columns:
        df = df[df["start_probability"] >= cfg.min_start_probability]
    df = df[~df["id"].isin(force_exclude_ids)]

    df["xpts_total"] = df["id"].map(xpts_by_player).fillna(0.0)
    df["xpts_objective"] = df["id"].map(xpts_for_objective).fillna(0.0)
    df["var_total"] = df["id"].map(var_by_player).fillna(0.0)
    # One-sided risk over the same horizon (2026-08-18). Without these the
    # objective falls back to variance, which is symmetric and so cannot tell a
    # player with big good weeks from one with big bad weeks -- the distinction
    # the whole risk axis exists to express.
    for _col in ("upside", "downside"):
        # A player absent from the projections has no points either, so a
        # zero risk term is both neutral and never selected on. NaN here would
        # propagate straight into the objective and make the ILP nonsense.
        df[_col] = df["id"].map(
            _multi_gw_semidev(projections, horizon, _col, decay)
        ).astype(float).fillna(0.0)

    lam, mu = lambda_mu_for_risk_level(
        cfg.risk_level, cfg.max_ownership_differential, cfg.mu_baseline, cfg.mu_range
    )
    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        df["eo_pct"] = df["id"].map(eo_map).fillna(0.0)
    else:
        df["eo_pct"] = 0.0
    # Reporting only since 2026-09-06: the horizon-collapsed coefficient the
    # ILP used to optimise on. The solve now reads `_per_week_scores` instead,
    # and this column survives because it is what the returned squad frame has
    # always carried and what the scenario-based captain pick below ranks on.
    df["effective_score"] = [
        risk_adjusted_score(x, v, e, lam, mu, up, down)
        for x, v, e, up, down in zip(
            df["xpts_objective"], df["var_total"], df["eo_pct"],
            df["upside"], df["downside"], strict=True,
        )
    ]

    if current_squad_ids:
        in_squad = set(current_squad_ids)
    else:
        in_squad = set()

    player_ids = df["id"].tolist()
    idx = {pid: i for i, pid in enumerate(player_ids)}
    n = len(player_ids)
    # One decision week per gameweek in the horizon. The squad is fixed across
    # all of them; only the XI moves. `max(1, ...)` keeps an empty projection
    # frame solvable, as the collapsed form was.
    weeks = max(1, len(horizon_gws))

    costs = df["now_cost"].tolist()
    positions = df["position"].tolist()
    teams = df["team_id"].tolist()
    eo_by_pid = dict(zip(player_ids, df["eo_pct"], strict=True))
    # P3-3 risk-adjusted, now per (player, week); true xpts_total for reporting
    # is read straight off `df`/`starting_xi`.
    scores_pw = _per_week_scores(projections, player_ids, horizon, decay, lam, mu, eo_by_pid)

    def _solve_once(cfg: OptimiserConfig) -> SquadSolution:
        """One full solve (plus its own ``max_transfers`` fallback) under ``cfg``.

        Everything computed above this closure -- ``df``, ``scores_pw``,
        ``lam``/``mu``, ``player_ids`` and so on -- does not depend on the
        fields the bench-weight fixed-point loop below mutates
        (``bench_slot_weights``, ``bench_gk_weight``), so it is computed once
        outside rather than recomputed on every iteration.
        """
        prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

        selected = [pulp.LpVariable(f"sel_{i}", cat="Binary") for i in range(n)]
        starting = {
            (i, w): pulp.LpVariable(f"sta_{i}_{w}", cat="Binary")
            for i in range(n) for w in range(weeks)
        }
        captain = {
            (i, w): pulp.LpVariable(f"cap_{i}_{w}", cat="Binary")
            for i in range(n) for w in range(weeks)
        }
        vice = {
            (i, w): pulp.LpVariable(f"vic_{i}_{w}", cat="Binary")
            for i in range(n) for w in range(weeks)
        }

        # 2026-07-30: a bench player used to contribute nothing to the objective
        # (only starting[i]/captain[i] did), so the solver had no reason to pick
        # anything but the cheapest feasible fodder once the starting XI was
        # set. `selected[i] - starting[i]` is 1 exactly when a player is on the
        # bench, so this adds a fractional (bench_value_weight) share of their
        # own score — real insurance value against an unpredicted blank in the
        # XI — without letting bench quality compete with the starting XI for
        # budget on equal terms.
        prob += _xi_objective(
            selected, starting, captain, vice, scores_pw, player_ids, weeks, cfg
        ) + _bench_objective(
            prob, selected, starting, scores_pw, player_ids, positions, weeks, cfg, "a",
            bb_target_week=bb_target_week,
        )

        # Squad-level constraints. These stay on the single horizon-wide
        # `selected` — you own one fifteen for the whole plan, and duplicating
        # these per week would let the solver imagine a squad that changes shape
        # between gameweeks without a transfer.
        prob += pulp.lpSum(selected) == SQUAD.squad_size
        prob += pulp.lpSum(costs[i] * selected[i] for i in range(n)) <= budget

        for pos in POSITIONS:
            pos_idx = [i for i, p in enumerate(positions) if p == pos]
            prob += pulp.lpSum(selected[i] for i in pos_idx) == SQUAD_COUNTS[pos]

        team_ids = list(set(teams))
        for tid in team_ids:
            team_idx = [i for i, t in enumerate(teams) if t == tid]
            prob += pulp.lpSum(selected[i] for i in team_idx) <= SQUAD.max_players_per_club

        # XI-shape constraints, one copy per week.
        for w in range(weeks):
            prob += pulp.lpSum(starting[(i, w)] for i in range(n)) == 11
            prob += pulp.lpSum(captain[(i, w)] for i in range(n)) == 1
            prob += pulp.lpSum(vice[(i, w)] for i in range(n)) == 1
            for pos in POSITIONS:
                pos_idx = [i for i, p in enumerate(positions) if p == pos]
                prob += pulp.lpSum(starting[(i, w)] for i in pos_idx) >= STARTING_MIN[pos]
                prob += pulp.lpSum(starting[(i, w)] for i in pos_idx) <= STARTING_MAX[pos]
            for i in range(n):
                prob += starting[(i, w)] <= selected[i]
                prob += captain[(i, w)] <= starting[(i, w)]
                prob += vice[(i, w)] <= starting[(i, w)]
                prob += captain[(i, w)] + vice[(i, w)] <= 1

        for pid in force_include_ids:
            if pid in idx:
                prob += selected[idx[pid]] == 1

        _add_no_good_cuts(prob, selected, idx, forbidden_squads)

        if current_squad_ids and max_transfers is not None:
            new_player = [pulp.LpVariable(f"new_{i}", cat="Binary") for i in range(n)]
            for i, pid in enumerate(player_ids):
                if pid in in_squad:
                    prob += new_player[i] == 0
                else:
                    prob += new_player[i] == selected[i]
            prob += pulp.lpSum(new_player) <= max_transfers

        prob.solve(_cbc(cfg))

        needs_fallback = (
            pulp.LpStatus[prob.status] != "Optimal"
            and current_squad_ids
            and max_transfers is not None
        )
        if needs_fallback:
            prob2 = pulp.LpProblem("fpl_squad_fallback", pulp.LpMaximize)
            selected2 = [pulp.LpVariable(f"sel2_{i}", cat="Binary") for i in range(n)]
            starting2 = {
                (i, w): pulp.LpVariable(f"sta2_{i}_{w}", cat="Binary")
                for i in range(n) for w in range(weeks)
            }
            captain2 = {
                (i, w): pulp.LpVariable(f"cap2_{i}_{w}", cat="Binary")
                for i in range(n) for w in range(weeks)
            }
            vice2 = {
                (i, w): pulp.LpVariable(f"vic2_{i}_{w}", cat="Binary")
                for i in range(n) for w in range(weeks)
            }
            prob2 += _xi_objective(
                selected2, starting2, captain2, vice2, scores_pw, player_ids, weeks, cfg
            ) + _bench_objective(
                prob2, selected2, starting2, scores_pw, player_ids, positions, weeks, cfg, "b",
                bb_target_week=bb_target_week,
            )
            prob2 += pulp.lpSum(selected2) == SQUAD.squad_size
            prob2 += pulp.lpSum(costs[i] * selected2[i] for i in range(n)) <= budget
            for pos in POSITIONS:
                pos_idx = [i for i, p in enumerate(positions) if p == pos]
                prob2 += pulp.lpSum(selected2[i] for i in pos_idx) == SQUAD_COUNTS[pos]
            for tid in list(set(teams)):
                team_idx = [i for i, t in enumerate(teams) if t == tid]
                prob2 += pulp.lpSum(selected2[i] for i in team_idx) <= SQUAD.max_players_per_club
            for w in range(weeks):
                prob2 += pulp.lpSum(starting2[(i, w)] for i in range(n)) == 11
                prob2 += pulp.lpSum(captain2[(i, w)] for i in range(n)) == 1
                prob2 += pulp.lpSum(vice2[(i, w)] for i in range(n)) == 1
                for pos in POSITIONS:
                    pos_idx = [i for i, p in enumerate(positions) if p == pos]
                    prob2 += pulp.lpSum(starting2[(i, w)] for i in pos_idx) >= STARTING_MIN[pos]
                    prob2 += pulp.lpSum(starting2[(i, w)] for i in pos_idx) <= STARTING_MAX[pos]
                for i in range(n):
                    prob2 += starting2[(i, w)] <= selected2[i]
                    prob2 += captain2[(i, w)] <= starting2[(i, w)]
                    prob2 += vice2[(i, w)] <= starting2[(i, w)]
                    prob2 += captain2[(i, w)] + vice2[(i, w)] <= 1
            for pid in force_include_ids:
                if pid in idx:
                    prob2 += selected2[idx[pid]] == 1
            _add_no_good_cuts(prob2, selected2, idx, forbidden_squads)
            prob2.solve(_cbc(cfg))
            if pulp.LpStatus[prob2.status] == "Optimal":
                logger.warning(
                    "max_transfers=%d infeasible; falling back to unconstrained squad",
                    max_transfers,
                )
                selected = selected2
                starting = starting2
                captain = captain2
                vice = vice2
                prob = prob2

        _raise_if_not_optimal(
            pulp.LpStatus[prob.status], cfg, "ILP solver", sol_status=prob.sol_status
        )

        # Week 0 is what gets reported. `starting_xi`, `captain_id` and
        # `vice_captain_id` describe the gameweek about to be locked in — the only
        # one any caller can act on — and every later week exists so that the
        # squad is chosen knowing what it will be able to do with them.
        selected_ids = {player_ids[i] for i in range(n) if pulp.value(selected[i]) > 0.5}
        starting_ids = {player_ids[i] for i in range(n) if pulp.value(starting[(i, 0)]) > 0.5}
        captain_id = next(player_ids[i] for i in range(n) if pulp.value(captain[(i, 0)]) > 0.5)
        vice_id = next(player_ids[i] for i in range(n) if pulp.value(vice[(i, 0)]) > 0.5)

        if season is not None and target_gw is not None:
            # The decayed figure, matching the ILP's own captain variable, which
            # ranks on `effective_score`. If these two disagreed the linear argmax
            # inside the solve and the scenario-based pick after it would be
            # answering different questions.
            xpts_by_id = dict(zip(df["id"], df["xpts_objective"], strict=True))
            var_by_id = dict(zip(df["id"], df["var_total"], strict=True))
            captain_id = scenario_based_captain(
                season, target_gw, list(starting_ids), xpts_by_id, var_by_id, mu,
                semidev_by_id=_semidev_by_id(df, mu),
            )
            if captain_id == vice_id:
                remaining = [pid for pid in starting_ids if pid != captain_id]
                vice_id = max(remaining, key=lambda pid: xpts_by_id.get(pid, 0.0))

        squad_df = df[df["id"].isin(selected_ids)].copy()
        squad_df["is_starting"] = squad_df["id"].isin(starting_ids)
        squad_df["is_captain"] = squad_df["id"] == captain_id
        squad_df["is_vice_captain"] = squad_df["id"] == vice_id

        bench = squad_df[~squad_df["is_starting"]].copy()
        bench = bench.sort_values(
            ["position", "xpts_total"],
            key=lambda s: (
                s.map({"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}) if s.name == "position" else -s
            ),
            ascending=[True, True],
        )
        bench_order = {pid: i for i, pid in enumerate(bench["id"])}
        squad_df["bench_order"] = squad_df["id"].map(bench_order).fillna(-1).astype(int)

        starting_xi = squad_df[squad_df["is_starting"]].copy()

        if current_squad_ids:
            incoming = selected_ids - in_squad
            transfers_made = len(incoming)
            hits = max(0, transfers_made - free_transfers)
        else:
            hits = 0

        # TRUE expected points (P3-3: the ILP's own objective value is now the
        # risk-adjusted `scores`, not real xpts — report the real figure,
        # computed straight from the starting XI + captain bonus).
        total_xpts = float(
            starting_xi["xpts_total"].sum()
            + starting_xi.loc[starting_xi["id"] == captain_id, "xpts_total"].sum()
        )
        total_cost = float(sum(
            df.loc[df["id"] == pid, "now_cost"].values[0]
            for pid in selected_ids
        ))

        logger.info(
            "Squad optimised: xPts=%.2f cost=£%.1fm hits=%d captain=%s",
            total_xpts, total_cost, hits,
            df.loc[df["id"] == captain_id, "web_name"].values[0],
        )

        return SquadSolution(
            squad=squad_df,
            starting_xi=starting_xi,
            captain_id=captain_id,
            vice_captain_id=vice_id,
            total_xpts=total_xpts,
            total_cost=total_cost,
            hits_taken=hits,
        )

    solution = _solve_once(cfg)

    # One derive-and-re-solve pass on the bench weights (2026-09-08). The
    # weights depend on the XI and the XI depends on the weights, so solve,
    # derive from what came back, solve once more. Deliberately NOT a fixed
    # point: the loop this replaced kept only the immediately preceding XI, so
    # an A->B->A->B cycle never tripped its equality check, burned the
    # iteration cap, and returned a solution whose XI was B priced under A's
    # weights. A single pass has no cycle to fall into and costs exactly two
    # solves (0.5s -> 1.1s measured on the live GW4 frame).
    #
    # Every path that reaches here REBUILDS the squad -- free hit, wildcard,
    # cold start -- so deriving from the solved XI is right here and hoisting
    # from the incumbent would be describing a squad about to be sold. The
    # ordinary weekly path never arrives at all; it calls `evaluate_transfers`,
    # which takes its weights from `bench_weights.derive_bench_config` instead.
    #
    # Named `_solve_once` here rather than `_solve`, which the brief sketched:
    # a module-level `_solve(prob, cfg, label)` already exists in this file
    # (used by `optimise_starting_xi`) for a different purpose, and a nested
    # function of the same name would silently shadow it for the rest of this
    # function -- confusing to anyone grepping the file, even though nothing
    # here actually calls the module-level one.
    if cfg.derive_bench_weights_per_solve:
        if "start_probability" not in players.columns:
            logger.warning(
                "derive_bench_weights_per_solve is set but `players` has no "
                "start_probability column; keeping the static bench weights"
            )
        else:
            derived = bench_config_for_xi(
                cfg,
                [int(pid) for pid in solution.starting_xi["id"]],
                blank_probability_map(players),
                dict(zip(df["id"], df["position"], strict=True)),
            )
            logger.info(
                "Bench weights derived from the solved XI: slots=%s gk=%.3f "
                "(was slots=%s gk=%.3f)",
                tuple(round(w, 3) for w in derived.bench_slot_weights),
                derived.bench_gk_weight,
                tuple(round(w, 3) for w in cfg.bench_slot_weights),
                cfg.bench_gk_weight,
            )
            solution = _solve_once(derived)

    return solution


def optimise_starting_xi(
    squad: pd.DataFrame,
    projections: pd.DataFrame,
    gw: int,
    ownership: pd.DataFrame | None = None,
    season: str | None = None,
    config: OptimiserConfig | None = None,
) -> SquadSolution:
    """``ownership`` (P3-3, optional, default None): see ``optimise_squad``'s
    docstring — ``None`` (every call site today, including the P-XI backtest
    harness) makes EO a uniform 0% for every candidate, which is a constant
    rescale of the objective and changes NEITHER the captain pick NOR the
    starting XI versus the pre-P3-3 pure-argmax behaviour — the P-XI exit
    gate's already-reported numbers stay reproducible byte-for-byte.

    ``season`` (P3-4, optional, default None): enables scenario-based
    captaincy for this ``gw`` (see ``optimiser/captaincy.py``) — real joint
    MC samples over the additive own-variance approximation, where P3-1 has
    persisted them. A no-op whenever the effective ``mu`` is exactly 0.0
    (``captaincy.scenario_based_captain`` short-circuits before touching
    the DB) — no longer ``OPTIMISER``'s default since
    plan/risk-aware-cold-start-v1.md gave ``risk_level=0`` a real, non-zero
    ``mu_baseline``. ``scripts/backtest.py`` pins its own calls to
    ``mu_baseline=0`` explicitly (``_BACKTEST_CONFIG``) so the P-XI gate's
    already-reported numbers stay a stable, comparable yardstick.

    ``config`` (optional): see ``optimise_squad``'s docstring — overrides
    the global ``OPTIMISER`` for this call only; ``None`` is byte-for-byte
    identical to today's behaviour."""
    cfg = config or OPTIMISER
    # Carry the one-sided risk columns through, not just xpts/xpts_var
    # (2026-08-25). Without them `risk_adjusted_score` below received
    # upside=downside=None and fell back to `xpts_var` -- and `mu` is
    # calibrated against quantities in POINTS while a variance is in
    # points-SQUARED. Since variance scales with the mean, a negative `mu`
    # then penalised the best players hardest and inverted the XI outright.
    #
    # Measured on the live GW2 frame at mu=-0.25: B.Fernandes went from 7.71
    # xPts to an effective -1.48 (13th of 15) and Gabriel from 7.07 to -5.13
    # (last), while Gibbs-White's 4.35 came top on 1.42. The armband followed,
    # landing on the goalkeeper. `optimise_squad` has passed these since
    # 2026-08-18; this function was missed, and `captaincy.py` carries a
    # comment describing the very same units bug being fixed there.
    #
    # Absent columns stay 0.0 rather than falling back to variance -- the same
    # choice `optimise_squad` documents. That makes `mu` inert on projections
    # which carry no semi-deviations (the in-season assemble path) instead of
    # silently switching it to a quantity on a different scale.
    risk_cols = [c for c in ("xpts_var", "upside", "downside") if c in projections.columns]
    gw_proj = projections[projections["gameweek"] == gw][
        ["player_id", "xpts", *risk_cols]
    ].copy()
    df = squad.merge(gw_proj, left_on="id", right_on="player_id", how="left")
    df["xpts"] = df["xpts"].fillna(0.0)
    for _col in ("xpts_var", "upside", "downside"):
        df[_col] = df[_col].astype(float).fillna(0.0) if _col in df.columns else 0.0

    lam, mu = lambda_mu_for_risk_level(
        cfg.risk_level, cfg.max_ownership_differential, cfg.mu_baseline, cfg.mu_range
    )
    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        df["eo_pct"] = df["id"].map(eo_map).fillna(0.0)
    else:
        df["eo_pct"] = 0.0
    df["effective_score"] = [
        risk_adjusted_score(x, v, e, lam, mu, up, down)
        for x, v, e, up, down in zip(
            df["xpts"], df["xpts_var"], df["eo_pct"],
            df["upside"], df["downside"], strict=True,
        )
    ]

    player_ids = df["id"].tolist()
    n = len(player_ids)
    positions = df["position"].tolist()
    scores = df["effective_score"].tolist()  # P3-3 risk-adjusted (objective);
    # true xpts for reporting is read straight off `df`/`starting_xi_df`

    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)

    starting = [pulp.LpVariable(f"sta_{i}", cat="Binary") for i in range(n)]
    captain = [pulp.LpVariable(f"cap_{i}", cat="Binary") for i in range(n)]
    vice = [pulp.LpVariable(f"vic_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(
        scores[i] * (starting[i] + captain[i])
        + cfg.vice_captain_weight * scores[i] * vice[i]
        for i in range(n)
    )

    prob += pulp.lpSum(starting) == 11
    prob += pulp.lpSum(captain) == 1
    prob += pulp.lpSum(vice) == 1

    for pos in POSITIONS:
        pos_idx = [i for i, p in enumerate(positions) if p == pos]
        prob += pulp.lpSum(starting[i] for i in pos_idx) >= STARTING_MIN[pos]
        prob += pulp.lpSum(starting[i] for i in pos_idx) <= STARTING_MAX[pos]

    for i in range(n):
        prob += captain[i] <= starting[i]
        prob += vice[i] <= starting[i]
        prob += captain[i] + vice[i] <= 1

    _solve(prob, cfg, "Starting XI solver")

    starting_ids = {player_ids[i] for i in range(n) if pulp.value(starting[i]) > 0.5}
    captain_id = next(player_ids[i] for i in range(n) if pulp.value(captain[i]) > 0.5)
    vice_id = next(player_ids[i] for i in range(n) if pulp.value(vice[i]) > 0.5)

    if season is not None:
        xpts_by_id = dict(zip(df["id"], df["xpts"], strict=True))
        var_by_id = dict(zip(df["id"], df["xpts_var"], strict=True))
        captain_id = scenario_based_captain(
            season, gw, list(starting_ids), xpts_by_id, var_by_id, mu,
            semidev_by_id=_semidev_by_id(df, mu),
        )
        if captain_id == vice_id:
            remaining = [pid for pid in starting_ids if pid != captain_id]
            vice_id = max(remaining, key=lambda pid: xpts_by_id.get(pid, 0.0))

    squad_out = df.copy()
    squad_out["is_starting"] = squad_out["id"].isin(starting_ids)
    squad_out["is_captain"] = squad_out["id"] == captain_id
    squad_out["is_vice_captain"] = squad_out["id"] == vice_id

    bench = squad_out[~squad_out["is_starting"]].copy()
    bench = bench.sort_values(
        ["position", "xpts"],
        key=lambda s: (
            s.map({"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}) if s.name == "position" else -s
        ),
        ascending=[True, True],
    )
    bench_order = {pid: i for i, pid in enumerate(bench["id"])}
    squad_out["bench_order"] = squad_out["id"].map(bench_order).fillna(-1).astype(int)

    starting_xi_df = squad_out[squad_out["is_starting"]]
    total_xpts = float(
        starting_xi_df["xpts"].sum()
        + starting_xi_df.loc[starting_xi_df["id"] == captain_id, "xpts"].sum()
    )

    return SquadSolution(
        squad=squad_out,
        starting_xi=starting_xi_df,
        captain_id=captain_id,
        vice_captain_id=vice_id,
        total_xpts=total_xpts,
        total_cost=float(squad["now_cost"].sum()),
        hits_taken=0,
    )
