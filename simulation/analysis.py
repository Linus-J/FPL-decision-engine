"""Season-level read-out for the shadow cohort (P2.5,
plan/decision-engine-recovery-plan.md).

The cohort is a designed experiment (see ``simulation/personas.py``), and
this is how it gets read. Three questions, in order of what a season can
actually answer:

1. **Did the swept parameter matter?** ``axis_effect`` — each axis's values
   against the points they produced, with the baseline for reference.
2. **How did each persona do?** ``persona_season_summary`` — one row each,
   including the paired difference against the baseline control.
3. **Is the engine calibrated?** ``calibration`` — predicted versus actual,
   per gameweek. This is the live instrument for the +8 pts/GW over-
   prediction found in the backtest on 2026-08-16: the naive-XI probe showed
   the projection layer is unbiased and the DECISION layer is where the bias
   enters, so it needs measuring on the decision path, which is exactly what
   these rows are.

**Read the paired differences, not the totals.** Every persona faces the
same fixtures, the same projections and the same luck, so
``delta_vs_baseline`` carries far less variance than any persona's absolute
score. A 40-point gap between two personas over a season is a much weaker
signal than it looks; the same gap expressed as a consistent weekly
difference is much stronger. ``gws_better_than_baseline`` is included for
exactly that reason.

Everything here is read-only and derives from ``sim_decision_log`` rows the
weekly run already writes. Nothing feeds back into decisions.
"""

from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import text

from data.db import get_session

_LINEUP_QUERY = """
    SELECT
        m.id                AS sim_manager_id,
        m.label             AS label,
        m.swept_axis        AS swept_axis,
        m.risk_level, m.chip_aggressiveness, m.transfer_switching_cost,
        m.ft_terminal_value, m.bench_value_weight,
        m.transfer_planning_horizon_gws, m.mu_baseline,
        l.gameweek          AS gameweek,
        l.projected_gain    AS predicted,
        l.actual_outcome    AS actual,
        l.details           AS details
    FROM sim_decision_log l
    JOIN sim_managers m ON m.id = l.sim_manager_id
    WHERE m.season = :season AND l.decision_type = 'lineup'
    ORDER BY m.id, l.gameweek
"""


def load_lineup_history(season: str) -> pd.DataFrame:
    """One row per (persona, gameweek) lineup decision, with the persona's
    parameters joined on. ``actual`` is NULL until the gameweek has finished
    and ``scripts/backfill_decision_outcomes.py`` has run."""
    db = get_session()
    try:
        df = pd.read_sql(text(_LINEUP_QUERY), db.bind, params={"season": season})
    finally:
        db.close()
    if df.empty:
        return df
    # A gameweek can be re-decided (a rerun, refining the squad as news
    # lands); only the last decision for that gameweek was the real one.
    df = df.drop_duplicates(subset=["sim_manager_id", "gameweek"], keep="last")
    df["hits_taken"] = [
        int(json.loads(d).get("hits_taken") or 0) for d in df["details"]
    ]
    return df.drop(columns=["details"])


def persona_season_summary(season: str) -> pd.DataFrame:
    """One row per persona, ranked. ``delta_vs_baseline`` and
    ``gws_better_than_baseline`` are the paired comparisons -- read those
    ahead of ``total_actual``, which carries a whole season of shared luck.
    """
    history = load_lineup_history(season)
    if history.empty:
        return pd.DataFrame()

    scored = history.dropna(subset=["actual"])
    if scored.empty:
        return pd.DataFrame()

    summary = (
        scored.groupby(["sim_manager_id", "label", "swept_axis"], as_index=False)
        .agg(
            gws_scored=("actual", "size"),
            total_actual=("actual", "sum"),
            mean_actual=("actual", "mean"),
            total_predicted=("predicted", "sum"),
            hits_taken=("hits_taken", "sum"),
        )
    )
    summary["mean_bias"] = (
        summary["total_predicted"] - summary["total_actual"]
    ) / summary["gws_scored"]

    baseline_rows = scored[scored["swept_axis"] == "baseline"]
    if not baseline_rows.empty:
        baseline_by_gw = baseline_rows.set_index("gameweek")["actual"]
        paired = scored.assign(
            baseline_actual=scored["gameweek"].map(baseline_by_gw)
        ).dropna(subset=["baseline_actual"])
        paired["diff"] = paired["actual"] - paired["baseline_actual"]
        deltas = paired.groupby("sim_manager_id", as_index=False).agg(
            delta_vs_baseline=("diff", "sum"),
            gws_better_than_baseline=("diff", lambda s: int((s > 0).sum())),
        )
        summary = summary.merge(deltas, on="sim_manager_id", how="left")

    summary = summary.sort_values("total_actual", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)
    return summary


def axis_effect(season: str) -> pd.DataFrame:
    """Per swept axis, the value tried against what it scored -- the shape
    of each parameter's effect. The baseline appears in every axis's group
    (at its default value) so each curve has its reference point."""
    summary = persona_season_summary(season)
    if summary.empty:
        return summary

    history = load_lineup_history(season)
    params = history.drop_duplicates(subset=["sim_manager_id"]).set_index("sim_manager_id")

    rows = []
    for r in summary.itertuples():
        axis = r.swept_axis
        if axis == "baseline":
            continue
        rows.append({
            "swept_axis": axis,
            "value": params.at[r.sim_manager_id, axis],
            "gws_scored": r.gws_scored,
            "total_actual": r.total_actual,
            "mean_actual": round(r.mean_actual, 2),
            "delta_vs_baseline": getattr(r, "delta_vs_baseline", None),
            "gws_better_than_baseline": getattr(r, "gws_better_than_baseline", None),
            "hits_taken": r.hits_taken,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["swept_axis", "value"]).reset_index(drop=True)


def calibration(season: str) -> pd.DataFrame:
    """Per gameweek across the whole cohort: mean predicted, mean actual and
    the signed bias.

    Mean points are noisy; a mean SIGNED error is a different statistic, and
    with ~90 personas per gameweek it converges quickly. Four or five
    gameweeks of this settles whether the backtest's +7.98 pts/GW
    over-prediction is real on the live path -- with no harness in the loop,
    which is the objection the backtest number could not answer.
    """
    history = load_lineup_history(season).dropna(subset=["actual"])
    if history.empty:
        return pd.DataFrame()

    out = history.groupby("gameweek", as_index=False).agg(
        personas=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        mean_actual=("actual", "mean"),
    )
    out["bias"] = out["mean_predicted"] - out["mean_actual"]
    for column in ("mean_predicted", "mean_actual", "bias"):
        out[column] = out[column].round(2)
    return out
