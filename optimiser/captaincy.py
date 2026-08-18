"""captaincy.py — P3-4 scenario-based captaincy (v2-build-plan §5).

``optimise_squad``/``optimise_starting_xi``'s captain variable is chosen by
linear argmax over ``effective_score`` (P3-3): among the starting XI, whoever
scores highest gets doubled. That's provably optimal for a MEAN objective
(doubling the top scorer is always weakly optimal under a linear objective),
and P3-3's own-variance term (``mu * xpts_var``, additive per player) doesn't
change that — it just reranks who counts as "top" before the same argmax
runs.

That argument breaks once captaincy is evaluated against the TRUE team-total
variance instead of a per-player proxy: doubling a player's score doesn't
just add their own variance once more, it adds ``Var(2X) = 4*Var(X)`` MINUS
the one copy already counted, i.e. ``+3*Var(X)`` if X is uncorrelated with
the rest of the team — and if X shares a fixture with other starting-XI
players (teammates or opponents), doubling also doubles X's COVARIANCE
contribution to the team total, which the additive per-player term can't see
at all (it's a sum of independent variances, not the joint distribution).

This module uses the real joint MC draws P3-1 persists (``ProjectionSample``)
to compute the TRUE team-total variance under each captaincy choice. Players
only share meaningful joint randomness with players in the SAME FIXTURE
(``assemble.py`` gives each fixture its own disjoint ``scenario_id`` range
within a gameweek — see its docstring); cross-fixture covariance is exactly
0 by construction, so team-total variance decomposes additively across
fixture groups, and only the candidate captain's own fixture group needs
recomputing per candidate.

**Degrades exactly to the pre-P3-4 behaviour when no real samples exist**
(cold start, backtest — which never persists samples per P3-1 — or a
candidate whose fixture has no persisted rows): falls back to the same
additive own-variance approximation P3-3 already used, so a candidate with
no sample data scores ``xpts + mu*xpts_var`` exactly as before. At
``mu == 0`` this short-circuits to plain mean argmax without touching the
DB at all — verified by test not assumed.

**DORMANT AT THE DEFAULT CONFIGURATION** (corrected 2026-08-18, engine
review §16). This docstring used to claim ``mu`` was "no longer 0 by
default", which was true when written and was falsified by the
``mu_baseline`` calibration of 2026-07-31: that sweep chose **0.0**, and
``mu = mu_baseline + risk_level * mu_range`` with the default
``risk_level = 0`` therefore gives ``mu = 0``. The short-circuit above is
consequently taken on EVERY real-bot and persona call, and none of the
covariance machinery below runs.

Nothing is wrong with that — plain mean argmax is provably optimal for a
linear mean objective, which is what the engine currently maximises. But
it is dormant, not live, and the same is true of the two other risk
layers (``optimiser/scoring.py``'s variance term and its ownership
weighting, both zero at ``risk_level = 0``). Reviving any of them means
re-running the ``mu_baseline`` calibration over GW6-38, as that
calibration's own note recommends — and separately for captaincy, since a
``mu`` that is wrong for squad selection may be right here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd
from sqlalchemy import bindparam, text

from data.db import get_session


def pick_captain(
    candidate_ids: Sequence[int],
    xpts_by_id: Mapping[int, float],
    var_by_id: Mapping[int, float],
    mu: float,
    fixture_groups: Sequence[pd.DataFrame],
) -> int:
    """Pure core. ``fixture_groups``: one DataFrame per real fixture (index=
    scenario id, columns=player_id restricted to ``candidate_ids``, values=
    MC xpts draws) — see ``load_fixture_groups``. A candidate absent from
    every group falls back to ``var_by_id`` additively (no covariance
    information available for them)."""
    candidate_ids = list(candidate_ids)
    if not candidate_ids:
        raise ValueError("pick_captain: candidate_ids must be non-empty")
    if mu == 0.0:
        return max(candidate_ids, key=lambda pid: xpts_by_id.get(pid, 0.0))

    candidate_set = set(candidate_ids)
    grouped_pid_data: dict[int, tuple[pd.Series, float, pd.Series]] = {}
    total_var_baseline = 0.0

    for g in fixture_groups:
        cols = [c for c in g.columns if c in candidate_set]
        if not cols:
            continue
        gsum = g[cols].sum(axis=1)
        var_g = float(gsum.var(ddof=1)) if len(gsum) > 1 else 0.0
        total_var_baseline += var_g
        for pid in cols:
            grouped_pid_data[pid] = (gsum, var_g, g[pid])

    for pid in candidate_ids:
        if pid not in grouped_pid_data:
            total_var_baseline += var_by_id.get(pid, 0.0)

    best_pid, best_score = None, float("-inf")
    for pid in candidate_ids:
        mean_c = xpts_by_id.get(pid, 0.0)
        if pid in grouped_pid_data:
            gsum, var_g, own_vec = grouped_pid_data[pid]
            new_sum = gsum + own_vec
            new_var = float(new_sum.var(ddof=1)) if len(new_sum) > 1 else 0.0
            var_with_c = total_var_baseline - var_g + new_var
        else:
            var_with_c = total_var_baseline + var_by_id.get(pid, 0.0)
        # Team-total STANDARD DEVIATION, not variance (2026-08-18). `mu` is
        # calibrated against quantities in POINTS -- see
        # optimiser/scoring.risk_adjusted_score, which multiplies it by an
        # upper semi-deviation. Multiplying it by a variance here instead put
        # the two on different scales by a factor of the spread itself, and at
        # a negative `mu` the variance term swamped the mean outright: the
        # risk-averse personas captained whoever had the least variance, which
        # is whoever was worst, and produced captains like a 1.2-xPts fourth
        # forward.
        score = mean_c + mu * math.sqrt(max(0.0, var_with_c))
        if score > best_score:
            best_score, best_pid = score, pid
    return best_pid


def load_latest_samples(season: str, gameweek: int, player_ids: Sequence[int]) -> pd.DataFrame:
    """Loads the latest persisted MC run's raw samples for ``player_ids`` in
    this (season, gameweek). ``created_at`` filters to a single run: the live
    pipeline can re-run for the same gameweek before its deadline, and a
    re-run's fresh RNG draws must never be paired scenario-by-scenario with a
    DIFFERENT run's draws for another player (that would correlate two
    unrelated random numbers as if they were the same joint scenario). Shared
    by ``load_fixture_groups`` (below) and ``chip_scenarios.load_scenario_totals``.
    Returns an empty DataFrame if nothing is persisted."""
    player_ids = list(dict.fromkeys(int(pid) for pid in player_ids))
    if not player_ids:
        return pd.DataFrame(columns=["player_id", "scenario_id", "xpts", "created_at"])

    db = get_session()
    try:
        stmt = text("""
            SELECT player_id, scenario_id, xpts, created_at
            FROM projection_samples
            WHERE season = :season AND gameweek = :gameweek
              AND player_id IN :player_ids
        """).bindparams(bindparam("player_ids", expanding=True))
        df = pd.read_sql(
            stmt, db.bind,
            params={"season": season, "gameweek": gameweek, "player_ids": player_ids},
        )
    finally:
        db.close()

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"])
    latest = df["created_at"].max()
    return df[df["created_at"] == latest]


def load_fixture_groups(
    season: str, gameweek: int, player_ids: Sequence[int]
) -> list[pd.DataFrame]:
    """Loads the latest persisted MC run's samples for ``player_ids`` in this
    (season, gameweek) and groups them by fixture. ``ProjectionSample`` has no
    fixture column, so fixture membership is recovered from the disjoint
    per-fixture ``scenario_id`` range ``assemble.py`` assigns (players sharing
    an identical (min, max) scenario_id span were drawn in the same fixture —
    a documented invariant, not a guess)."""
    df = load_latest_samples(season, gameweek, player_ids)
    if df.empty:
        return []

    groups: dict[tuple[int, int], list[pd.Series]] = {}
    for pid, sub in df.groupby("player_id"):
        key = (int(sub["scenario_id"].min()), int(sub["scenario_id"].max()))
        groups.setdefault(key, []).append(sub.set_index("scenario_id")["xpts"].rename(pid))

    return [pd.concat(series_list, axis=1) for series_list in groups.values()]


def scenario_based_captain(
    season: str,
    gameweek: int,
    candidate_ids: Sequence[int],
    xpts_by_id: Mapping[int, float],
    var_by_id: Mapping[int, float],
    mu: float,
) -> int:
    """Orchestrator: skips the DB entirely at ``mu == 0`` (the common case —
    balanced risk mode, or any caller not opting into risk-aware captaincy),
    otherwise loads real fixture groups and defers to ``pick_captain``."""
    candidate_ids = list(candidate_ids)
    if not candidate_ids:
        raise ValueError("scenario_based_captain: candidate_ids must be non-empty")
    if mu == 0.0:
        return max(candidate_ids, key=lambda pid: xpts_by_id.get(pid, 0.0))
    fixture_groups = load_fixture_groups(season, gameweek, candidate_ids)
    return pick_captain(candidate_ids, xpts_by_id, var_by_id, mu, fixture_groups)
