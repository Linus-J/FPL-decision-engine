"""Persona generation for the live simulation engine
(plan/decision-engine-recovery-plan.md P2.4, superseding
plan/simulation-engine-v1.md's random sweep).

**Why the design changed (2026-08-16).** The original cohort sampled
``risk_level``, ``max_ownership_differential`` and ``chip_aggressiveness``
independently at random. Two problems:

- ``max_ownership_differential`` is completely inert. No call site passes
  ``ownership=`` to any optimiser, and ``ownership_snapshots`` is empty, so
  a third of the sampled variation did nothing at all.
- Every genuinely untuned, load-bearing parameter -- ``transfer_switching_cost``,
  ``ft_terminal_value``, ``bench_value_weight``, the planning horizon,
  ``mu_baseline`` -- was pinned to the real bot's value across all 100
  personas, so the cohort could not speak to any of them.

**Why one-factor-at-a-time.** A season is a single experiment and weekly FPL
scores are dominated by noise. 100 points sampled independently across six
axes would leave every main effect confounded with the others and with luck.
Sweeping one axis at a time around the defaults spends the whole cohort on
main effects, which is the most a single noisy season can support.

The comparison is also **paired**, which is the property that makes this
work at all: every persona faces the same fixtures, the same projections and
the same season, so differences between them carry far less variance than
comparing any of them to an external benchmark. Interactions are deliberately
not addressed; they need more seasons, not a cleverer split of one.

Personas are generated once per season and never change mid-season.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from config.strategy import OPTIMISER, TRANSFERS
from data.models import SimManager

# Fixed, arbitrary constant -- reproducibility (every season regenerates the
# exact same personas if sim_managers is ever rebuilt from empty), not
# something to vary per run.
SIM_SEED = 20260731

# Personas per continuously-swept axis (the integer horizon axis uses its
# own distinct values instead). See SIM_COUNT below for the cohort size.
_PER_AXIS = 14


def _spread(low: float, high: float, n: int) -> list[float]:
    """`n` evenly spaced values across [low, high], inclusive of both ends."""
    if n <= 1:
        return [low]
    step = (high - low) / (n - 1)
    return [round(low + step * i, 4) for i in range(n)]


# Each axis: the SimManager field it sets, and the values to sweep. Ranges
# deliberately straddle the current default so the cohort can say whether the
# default is too low as well as too high.
#
# `transfer_planning_horizon_gws` is capped by OPTIMISER.projection_horizon_gws
# -- personas read the persisted projection frame rather than building their
# own, so a longer horizon than the pipeline persists silently does nothing.
SWEPT_AXES: dict[str, list[float]] = {
    # The top question after 2026-08-16's finding that the projection layer is
    # unbiased but the DECISION layer over-predicts by ~8pts/GW: the transfer
    # step is where selection bias enters, so how high should its bar be?
    "transfer_switching_cost": _spread(0.0, 4.0, _PER_AXIS),
    # How much is an unspent transfer worth? Never measurably tested: the one
    # grid search that varied it spanned 1.4 points across a 10x range.
    "ft_terminal_value": _spread(0.0, 6.0, _PER_AXIS),
    # How much to spend on a bench that only pays off when a starter blanks.
    "bench_value_weight": _spread(0.0, 0.5, _PER_AXIS),
    # Does planning further ahead help, or just add churn? Only as many
    # personas as there are distinct horizons -- padding to _PER_AXIS would
    # spend a tenth of the cohort on exact duplicates.
    "transfer_planning_horizon_gws": [
        float(gw) for gw in range(1, OPTIMISER.projection_horizon_gws + 1)
    ],
    # Variance-awareness is switched off at the default (mu_baseline=0.0 won a
    # single reduced-window calibration). Negative actively prefers
    # low-variance picks at equal mean.
    "mu_baseline": _spread(-0.1, 0.2, _PER_AXIS),
    # How eagerly to spend chips.
    "chip_aggressiveness": _spread(0.5, 1.5, _PER_AXIS),
    # Kept from the original cohort. Drives lambda (inert until ownership is
    # wired in -- see plan P3.2) and shifts mu around mu_baseline.
    "risk_level": _spread(-1.0, 1.0, _PER_AXIS),
}

# The baseline control plus every swept value. Not a round 100 by
# design -- the cohort size follows from the experiment, not the reverse.
SIM_COUNT = 1 + sum(len(values) for values in SWEPT_AXES.values())

# Every swept axis must be a real SimManager field, because that is how the
# season read-out finds a persona's setting: simulation/analysis.py indexes
# the joined frame BY THE AXIS NAME. An axis that is not a column would raise
# there — after a season of runs, when the data is finally being read. Assert
# it at import instead, where it is free.
_SIM_MANAGER_FIELDS = {c.name for c in SimManager.__table__.columns}
_unknown = set(SWEPT_AXES) - _SIM_MANAGER_FIELDS
if _unknown:
    raise ValueError(
        f"SWEPT_AXES names that are not SimManager columns: {sorted(_unknown)} — "
        f"simulation/analysis.py could not read them back"
    )


def _defaults() -> dict:
    return {
        "risk_level": OPTIMISER.risk_level,
        "max_ownership_differential": OPTIMISER.max_ownership_differential,
        "chip_aggressiveness": 1.0,
        "transfer_switching_cost": TRANSFERS.transfer_switching_cost,
        "ft_terminal_value": TRANSFERS.ft_terminal_value,
        "bench_value_weight": OPTIMISER.bench_value_weight,
        "transfer_planning_horizon_gws": OPTIMISER.transfer_planning_horizon_gws,
        "mu_baseline": OPTIMISER.mu_baseline,
    }


def generate_personas(season: str, seed: int = SIM_SEED) -> list[dict]:
    """Deterministic: the same ``season`` always produces byte-identical
    personas. Returns plain dicts (not ORM rows) so this stays testable
    without a DB.

    ``seed`` is accepted for call-compatibility and reproducibility bookkeeping
    only -- the design is a fixed grid, with no randomness left to seed."""
    del seed

    # The all-defaults control. Its decisions should match the real bot's
    # exactly, which makes it the reference every swept persona is read
    # against -- and a live check that the simulation path and the real path
    # have not drifted.
    personas: list[dict] = [
        {"season": season, "label": "baseline", "swept_axis": "baseline", **_defaults()}
    ]

    for axis, values in SWEPT_AXES.items():
        for value in values:
            params = _defaults()
            params[axis] = (
                int(value) if axis == "transfer_planning_horizon_gws" else float(value)
            )
            personas.append({
                "season": season,
                "label": f"{axis}={params[axis]}",
                "swept_axis": axis,
                **params,
            })
    return personas


def _assert_persisted_axes_still_readable(
    existing: list[SimManager], season: str
) -> None:
    """``swept_axis`` is an unversioned free-text string written once, at the
    season's first run, and read months later by ``simulation.analysis``.

    Renaming or removing an axis mid-season leaves the persisted rows pointing
    at a name the code no longer knows. ``axis_effect`` groups on that string
    and looks the value up as a column of the joined lineup frame, so a stale
    name either raises in November against a cohort that cannot be regenerated,
    or -- worse -- splits one axis into two groups that each look like a
    smaller, noisier experiment. Fail at load, where the message can still say
    what happened.
    """
    persisted = {p.swept_axis for p in existing}
    expected = set(SWEPT_AXES) | {"baseline"}
    stale = persisted - expected
    if stale:
        raise ValueError(
            f"season {season} has personas on axes {sorted(stale)}, which are no "
            f"longer in SWEPT_AXES {sorted(SWEPT_AXES)}. An axis was renamed or "
            f"removed after the cohort was created; simulation/analysis.py reads "
            f"the axis name as a column, so this season's results would be "
            f"unreadable or silently split. Restore the old name, or start a "
            f"fresh cohort under a new season key."
        )


def load_or_create_personas(
    db: Session, season: str, seed: int = SIM_SEED
) -> list[SimManager]:
    """This season's persisted persona rows, generating them only on the
    season's first-ever call. Never regenerates once any rows exist -- a
    persona's identity and configuration stay stable for the whole season,
    which is what makes its decision history interpretable."""
    existing = (
        db.query(SimManager)
        .filter(SimManager.season == season)
        .order_by(SimManager.id)
        .all()
    )
    if existing:
        _assert_persisted_axes_still_readable(existing, season)
        return existing

    rows = [SimManager(**p) for p in generate_personas(season, seed)]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows
