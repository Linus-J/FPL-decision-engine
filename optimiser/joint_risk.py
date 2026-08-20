"""joint_risk.py — Objective v2: covariance-aware squad selection.

The projection stage already models teammate correlation properly:
``projection/covariance.py`` draws each team's goals ONCE per fixture-scenario
and conditions every player on that shared draw, so a goalkeeper and his own
centre-back share a clean sheet. Squad selection then threw that structure
away, because the MILP objective consumes per-player summary statistics.

Note what the old objective actually assumed. Since 2026-08-18
``scoring.risk_adjusted_score`` sums per-player SEMI-DEVIATIONS, and summing
standard deviations is the PERFECT-CORRELATION assumption — not the
independence one the README described. Both are wrong. This module replaces
them with the empirical joint distribution: every candidate squad is scored on
the raw scenarios, where two Arsenal defenders rise and fall together because
they did so in the draw.

The candidate pool is generated from the pure-mean objective and ``mu`` enters
only here, during re-ranking. That is what lets a calibration sweep reuse one
pool per gameweek across every candidate ``mu``, and what makes ``mu = 0``
provably reproduce the linear objective's pick.

Known limitation: cross-gameweek correlation is zero by construction, because
each gameweek's fixtures are drawn independently upstream. Real team form
persists week to week; this does not model that.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioMatrix:
    """Dense ``(scenario, player)`` Monte-Carlo draws.

    ``column_index`` maps ``player_id`` to a column in ``values``. A player
    absent from it has no draws — callers must treat that as missing data, not
    as a zero score.
    """

    values: np.ndarray
    column_index: dict[int, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.values.size == 0 or not self.column_index


EMPTY_MATRIX = ScenarioMatrix(values=np.empty((0, 0)), column_index={})


def _pivot(df: pd.DataFrame) -> ScenarioMatrix:
    """``df`` needs columns ``player_id``, ``scenario_id``, ``xpts``."""
    if df.empty:
        return EMPTY_MATRIX
    wide = df.pivot_table(
        index="scenario_id", columns="player_id", values="xpts", aggfunc="mean"
    ).sort_index()
    # A player drawn in a DIFFERENT fixture has his draws under a disjoint
    # scenario_id range (assemble.py assigns one range per fixture), so the
    # pivot leaves NaN wherever two fixtures' ranges do not overlap. Those NaNs
    # are not missing data — they are another fixture's rows — so each column
    # is compacted independently and the per-fixture blocks stack into one
    # matrix. Scenario k of fixture A is paired with scenario k of fixture B,
    # which is sound precisely because the two fixtures are independent.
    columns = list(wide.columns)
    packed = [wide[c].dropna().to_numpy() for c in columns]
    depth = min((len(a) for a in packed), default=0)
    if depth == 0:
        return EMPTY_MATRIX
    values = np.column_stack([a[:depth] for a in packed])
    return ScenarioMatrix(
        values=values, column_index={int(c): i for i, c in enumerate(columns)}
    )


def matrix_from_rows(rows: Sequence[Mapping], gameweek: int) -> ScenarioMatrix:
    """In-memory adapter: the ``sample_sink`` rows from
    ``assemble.assemble_gw_projections``, restricted to one gameweek."""
    subset = [r for r in rows if int(r["gameweek"]) == int(gameweek)]
    if not subset:
        return EMPTY_MATRIX
    return _pivot(pd.DataFrame(subset))


def load_scenario_matrix(
    season: str, gameweek: int, player_ids: Sequence[int]
) -> ScenarioMatrix:
    """Live adapter: the latest persisted MC run from ``projection_samples``.

    Delegates to ``captaincy.load_latest_samples`` rather than repeating its
    SQL, so the single-run ``created_at`` filter stays defined in one place.
    """
    from optimiser.captaincy import load_latest_samples

    df = load_latest_samples(season, gameweek, player_ids)
    if df.empty:
        return EMPTY_MATRIX
    return _pivot(df)


def matrices_from_rows(
    rows: Sequence[Mapping], gameweeks: Sequence[int]
) -> dict[int, ScenarioMatrix]:
    """One matrix per horizon gameweek, keyed by gameweek.

    Gameweeks with no draws are omitted rather than mapped to an empty matrix,
    so a caller can tell "not sampled" from "sampled and empty".
    """
    out: dict[int, ScenarioMatrix] = {}
    for gw in gameweeks:
        m = matrix_from_rows(rows, int(gw))
        if not m.is_empty:
            out[int(gw)] = m
    return out


def load_scenario_matrices(
    season: str, gameweeks: Sequence[int], player_ids: Sequence[int]
) -> dict[int, ScenarioMatrix]:
    """Live equivalent of ``matrices_from_rows``."""
    out: dict[int, ScenarioMatrix] = {}
    for gw in gameweeks:
        m = load_scenario_matrix(season, int(gw), player_ids)
        if not m.is_empty:
            out[int(gw)] = m
    return out
