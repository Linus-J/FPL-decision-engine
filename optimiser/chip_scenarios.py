"""chip_scenarios.py — P3-5 scenario-EV chip evaluation (v2-build-plan §5).

``chips.py``'s chip recommendations compared a POINT-ESTIMATE gain (built
from mean xpts) against a fixed constant threshold — literally "does the
mean outcome clear this bar." That can't distinguish a reliable, low-
variance gain from a coin-flip with the same mean, which is exactly the
distinction that matters for a chip: chips are single-use, so playing one on
a decision that clears the bar on average but has a real chance of a
below-threshold outcome is a worse decision than the raw mean gain
suggests.

This module reuses P3-1's real persisted joint MC draws (``captaincy.py``'s
``load_latest_samples``) to build an actual per-scenario TOTAL for an
arbitrary set of players in a gameweek, so callers can subtract two such
totals to get a real per-scenario GAIN DISTRIBUTION and report
``P(gain >= 0)`` alongside the mean, instead of only the mean.

Fixture-group composition: cross-fixture covariance is exactly 0 (each
fixture gets its own disjoint ``scenario_id`` range — see captaincy.py's
docstring), so any FIXED pairing of one fixture group's draws with
another's is a valid joint sample of their sum — the fixtures are
independent, so which specific draw from fixture B gets added to which draw
from fixture A doesn't matter for the SUM's distribution, only that the
same pairing rule is applied consistently. This module pairs by each
fixture group's own row order (rank within its group, i.e. position after
sorting by ``scenario_id``) — never by comparing raw ``scenario_id`` values
across groups, since those ranges are disjoint and NOT jointly meaningful
across fixtures.

**Degrades to an empty ``pd.Series`` when no real samples exist** (cold
start, or the backtest walk-forward, which never persists samples per
P3-1) — callers must treat an empty result as "no scenario data available"
and fall back to a point-estimate threshold; this module never raises for
missing data, only for genuinely invalid input (no player_ids).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from optimiser.captaincy import load_latest_samples


def _one_gw_total(season: str, gameweek: int, player_ids: Sequence[int]) -> pd.Series:
    """Per-scenario total for ``player_ids`` within a SINGLE gameweek,
    composed across that gameweek's fixture groups by position (see module
    docstring). Empty if no samples exist for this gameweek."""
    df = load_latest_samples(season, gameweek, player_ids)
    if df.empty:
        return pd.Series(dtype=float)

    groups: dict[tuple[int, int], list[pd.Series]] = {}
    for pid, sub in df.groupby("player_id"):
        key = (int(sub["scenario_id"].min()), int(sub["scenario_id"].max()))
        ranked = sub.sort_values("scenario_id")["xpts"].reset_index(drop=True).rename(pid)
        groups.setdefault(key, []).append(ranked)

    group_totals = [pd.concat(series_list, axis=1).sum(axis=1) for series_list in groups.values()]

    min_len = min(len(g) for g in group_totals)
    total = group_totals[0].iloc[:min_len].reset_index(drop=True)
    for g in group_totals[1:]:
        total = total + g.iloc[:min_len].reset_index(drop=True)
    return total


def load_scenario_totals(
    season: str, gameweeks: int | Sequence[int], player_ids: Sequence[int]
) -> pd.Series:
    """Per-scenario team total for ``player_ids`` across one or more
    gameweeks. ``assemble.py`` resets ``scenario_id`` to 0 at the start of
    EVERY gameweek (each GW is its own independent sampling run — no shared
    latent across gameweeks), so a multi-GW total is composed the same way
    fixture groups are composed WITHIN one gameweek: sum each gameweek's own
    total positionally (any fixed pairing of independent draws is a valid
    joint sample of the sum). Empty if any requested gameweek has no
    persisted samples — a partial sum over only SOME of the requested
    gameweeks would silently understate a multi-GW decision like Wildcard,
    so this is all-or-nothing rather than a partial result."""
    player_ids = list(dict.fromkeys(int(pid) for pid in player_ids))
    if not player_ids:
        return pd.Series(dtype=float)

    gws = [int(gameweeks)] if isinstance(gameweeks, (int, np.integer)) else list(gameweeks)
    per_gw_totals = [_one_gw_total(season, gw, player_ids) for gw in gws]
    if any(t.empty for t in per_gw_totals):
        return pd.Series(dtype=float)

    min_len = min(len(t) for t in per_gw_totals)
    total = per_gw_totals[0].iloc[:min_len].reset_index(drop=True)
    for t in per_gw_totals[1:]:
        total = total + t.iloc[:min_len].reset_index(drop=True)
    return total


def gain_distribution(
    season: str,
    gameweeks: int | Sequence[int],
    plus_ids: Sequence[int],
    minus_ids: Sequence[int],
) -> pd.Series:
    """Per-scenario ``total(plus_ids) - total(minus_ids)`` over the same
    gameweek(s). Shared fixtures between the two sides stay correlated (same
    underlying run, same ascending-``scenario_id`` position convention in
    both calls), so this is not just two independent totals subtracted — a
    player common to a shared fixture in both sides cancels consistently,
    not by coincidence. Empty if either side has no scenario data."""
    plus_total = load_scenario_totals(season, gameweeks, plus_ids)
    minus_total = load_scenario_totals(season, gameweeks, minus_ids)
    if plus_total.empty or minus_total.empty:
        return pd.Series(dtype=float)

    min_len = min(len(plus_total), len(minus_total))
    return plus_total.iloc[:min_len].reset_index(drop=True) - minus_total.iloc[
        :min_len
    ].reset_index(drop=True)
