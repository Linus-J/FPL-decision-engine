import logging
from dataclasses import dataclass

import pandas as pd
import pulp

from config.strategy import OPTIMISER, SQUAD, OptimiserConfig
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


def _bench_objective(
    prob: pulp.LpProblem,
    selected: list,
    starting: list,
    scores: list[float],
    positions: list[str],
    cfg: OptimiserConfig,
    tag: str,
):
    """Objective terms for the bench, weighted by SLOT rather than uniformly.

    A bench is an ordered queue. FPL's automatic substitutions promote the
    first eligible bench player when a starter does not appear, so the value
    of slot 1 is P(at least one starter blanks) -- 0.53 on the live GW1 XI --
    while slot 3 needs three simultaneous absences and is worth 0.03. A flat
    weight underpays the slot that gets used and overpays the two that do not,
    which is what buys four £4.0m non-players instead of one real substitute.

    Adds the slot-assignment constraints to ``prob`` and returns the terms to
    add to the objective. Slot weights are strictly decreasing, so the solver
    puts its best bench player in slot 1 without needing an ordering
    constraint -- it is a maximisation and any other assignment is dominated.
    """
    weights = [w * cfg.bench_value_weight for w in cfg.bench_slot_weights]
    gk_weight = cfg.bench_gk_weight * cfg.bench_value_weight
    n = len(scores)
    outfield = [i for i in range(n) if positions[i] != "GKP"]
    keepers = [i for i in range(n) if positions[i] == "GKP"]

    slot = {
        (i, k): pulp.LpVariable(f"bench_{tag}_{i}_{k}", cat="Binary")
        for i in outfield
        for k in range(len(weights))
    }
    for i in outfield:
        prob += pulp.lpSum(slot[i, k] for k in range(len(weights))) == selected[i] - starting[i]
    for k in range(len(weights)):
        prob += pulp.lpSum(slot[i, k] for i in outfield) == 1

    terms = pulp.lpSum(
        weights[k] * scores[i] * slot[i, k] for i in outfield for k in range(len(weights))
    )
    # The reserve keeper has no queue to inherit from: he plays only if the
    # first-choice keeper does not.
    terms += pulp.lpSum(
        gk_weight * scores[i] * (selected[i] - starting[i]) for i in keepers
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
        except RuntimeError:
            break
        pool.append(solution)
        forbidden.append(sorted(int(pid) for pid in solution.squad["id"]))
    return pool


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
) -> SquadSolution:
    """``ownership`` (P3-3, optional): a ``(player_id, top10k_selected_pct)``
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
    cfg = config or OPTIMISER
    horizon = horizon or cfg.transfer_planning_horizon_gws
    force_include_ids = set(force_include_ids or [])
    force_exclude_ids = set(force_exclude_ids or [])
    horizon_gws = sorted(projections["gameweek"].unique())[:horizon]
    target_gw = horizon_gws[0] if horizon_gws else None

    # Two aggregates over the same horizon (2026-08-18): the TRUE sum, which
    # is what gets reported, and the decayed sum, which is what gets optimised.
    # See _decay_weights — a projection five gameweeks out is not worth as much
    # as one for the match about to kick off, and every serious FPL optimiser
    # discounts it. Reporting the decayed figure as "expected points" would be
    # a lie about the quantity being predicted.
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

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

    selected = [pulp.LpVariable(f"sel_{i}", cat="Binary") for i in range(n)]
    starting = [pulp.LpVariable(f"sta_{i}", cat="Binary") for i in range(n)]
    captain = [pulp.LpVariable(f"cap_{i}", cat="Binary") for i in range(n)]
    vice = [pulp.LpVariable(f"vic_{i}", cat="Binary") for i in range(n)]

    scores = df["effective_score"].tolist()   # P3-3 risk-adjusted (objective);
    # true xpts_total for reporting is read straight off `df`/`starting_xi`
    costs = df["now_cost"].tolist()
    positions = df["position"].tolist()
    teams = df["team_id"].tolist()

    # 2026-07-30: a bench player used to contribute nothing to the objective
    # (only starting[i]/captain[i] did), so the solver had no reason to pick
    # anything but the cheapest feasible fodder once the starting XI was
    # set. `selected[i] - starting[i]` is 1 exactly when a player is on the
    # bench, so this adds a fractional (bench_value_weight) share of their
    # own score — real insurance value against an unpredicted blank in the
    # XI — without letting bench quality compete with the starting XI for
    # budget on equal terms.
    prob += pulp.lpSum(
        scores[i] * (starting[i] + captain[i])
        # The armband only passes to the vice when the captain does not
        # feature, so it is worth P(captain blanks) x his score. Without this
        # term `vice` is constrained but unvalued, every legal choice ties, and
        # the solver returns whichever it branched on -- a goalkeeper, on the
        # live frame, while a 7.43-xPts defender sat in the same XI.
        + cfg.vice_captain_weight * scores[i] * vice[i]
        for i in range(n)
    ) + _bench_objective(prob, selected, starting, scores, positions, cfg, "a")

    prob += pulp.lpSum(selected) == SQUAD.squad_size
    prob += pulp.lpSum(costs[i] * selected[i] for i in range(n)) <= budget
    prob += pulp.lpSum(starting) == 11
    prob += pulp.lpSum(captain) == 1
    prob += pulp.lpSum(vice) == 1

    for pos in POSITIONS:
        pos_idx = [i for i, p in enumerate(positions) if p == pos]
        prob += pulp.lpSum(selected[i] for i in pos_idx) == SQUAD_COUNTS[pos]
        prob += pulp.lpSum(starting[i] for i in pos_idx) >= STARTING_MIN[pos]
        prob += pulp.lpSum(starting[i] for i in pos_idx) <= STARTING_MAX[pos]

    team_ids = list(set(teams))
    for tid in team_ids:
        team_idx = [i for i, t in enumerate(teams) if t == tid]
        prob += pulp.lpSum(selected[i] for i in team_idx) <= SQUAD.max_players_per_club

    for i in range(n):
        prob += starting[i] <= selected[i]
        prob += captain[i] <= starting[i]
        prob += vice[i] <= starting[i]
        prob += captain[i] + vice[i] <= 1

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

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal" and current_squad_ids and max_transfers is not None:
        prob2 = pulp.LpProblem("fpl_squad_fallback", pulp.LpMaximize)
        selected2 = [pulp.LpVariable(f"sel2_{i}", cat="Binary") for i in range(n)]
        starting2 = [pulp.LpVariable(f"sta2_{i}", cat="Binary") for i in range(n)]
        captain2 = [pulp.LpVariable(f"cap2_{i}", cat="Binary") for i in range(n)]
        vice2 = [pulp.LpVariable(f"vic2_{i}", cat="Binary") for i in range(n)]
        prob2 += pulp.lpSum(
            scores[i] * (starting2[i] + captain2[i])
            + cfg.vice_captain_weight * scores[i] * vice2[i]
            for i in range(n)
        ) + _bench_objective(prob2, selected2, starting2, scores, positions, cfg, "b")
        prob2 += pulp.lpSum(selected2) == SQUAD.squad_size
        prob2 += pulp.lpSum(costs[i] * selected2[i] for i in range(n)) <= budget
        prob2 += pulp.lpSum(starting2) == 11
        prob2 += pulp.lpSum(captain2) == 1
        prob2 += pulp.lpSum(vice2) == 1
        for pos in POSITIONS:
            pos_idx = [i for i, p in enumerate(positions) if p == pos]
            prob2 += pulp.lpSum(selected2[i] for i in pos_idx) == SQUAD_COUNTS[pos]
            prob2 += pulp.lpSum(starting2[i] for i in pos_idx) >= STARTING_MIN[pos]
            prob2 += pulp.lpSum(starting2[i] for i in pos_idx) <= STARTING_MAX[pos]
        for tid in list(set(teams)):
            team_idx = [i for i, t in enumerate(teams) if t == tid]
            prob2 += pulp.lpSum(selected2[i] for i in team_idx) <= SQUAD.max_players_per_club
        for i in range(n):
            prob2 += starting2[i] <= selected2[i]
            prob2 += captain2[i] <= starting2[i]
            prob2 += vice2[i] <= starting2[i]
            prob2 += captain2[i] + vice2[i] <= 1
        for pid in force_include_ids:
            if pid in idx:
                prob2 += selected2[idx[pid]] == 1
        _add_no_good_cuts(prob2, selected2, idx, forbidden_squads)
        prob2.solve(pulp.PULP_CBC_CMD(msg=False))
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

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(
            f"ILP solver did not find optimal solution: {pulp.LpStatus[prob.status]}"
        )

    selected_ids = {player_ids[i] for i in range(n) if pulp.value(selected[i]) > 0.5}
    starting_ids = {player_ids[i] for i in range(n) if pulp.value(starting[i]) > 0.5}
    captain_id = next(player_ids[i] for i in range(n) if pulp.value(captain[i]) > 0.5)
    vice_id = next(player_ids[i] for i in range(n) if pulp.value(vice[i]) > 0.5)

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
    gw_proj = projections[projections["gameweek"] == gw][["player_id", "xpts"]].copy()
    if "xpts_var" in projections.columns:
        gw_var = projections[projections["gameweek"] == gw][["player_id", "xpts_var"]]
        gw_proj = gw_proj.merge(gw_var, on="player_id", how="left")
    df = squad.merge(gw_proj, left_on="id", right_on="player_id", how="left")
    df["xpts"] = df["xpts"].fillna(0.0)
    df["xpts_var"] = df["xpts_var"].fillna(0.0) if "xpts_var" in df.columns else 0.0

    lam, mu = lambda_mu_for_risk_level(
        cfg.risk_level, cfg.max_ownership_differential, cfg.mu_baseline, cfg.mu_range
    )
    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        df["eo_pct"] = df["id"].map(eo_map).fillna(0.0)
    else:
        df["eo_pct"] = 0.0
    df["effective_score"] = [
        risk_adjusted_score(x, v, e, lam, mu)
        for x, v, e in zip(df["xpts"], df["xpts_var"], df["eo_pct"], strict=True)
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

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Starting XI solver failed: {pulp.LpStatus[prob.status]}")

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
