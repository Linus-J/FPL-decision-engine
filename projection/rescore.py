"""rescore.py — P-RS: re-score 25/26 actuals under 26/27 rules (finding C1).

The Phase-2 exit gate needs the PREDICTED and ACTUAL sides on one scoring basis.
`player_gw_stats.total_points` is the as-played (old-rules-bonus) total. Standard
scoring and DefCon are UNCHANGED 25/26→26/27 (config.strategy — only the BPS
weights that decide bonus changed: being_tackled removed, cbi_per_point 2→3,
penalty_saved 8→7, big_chance_saved 0→+1). So the 26/27-equivalent total is
exactly the as-played total with its bonus swapped for the 26/27-recomputed one:

    points_2627 = total_points − bonus_as_played + bonus_2627

`bonus_2627` comes from `recomputed_bonus` (T5b/P8), summed across a DGW
player's matches in that gameweek (FPL's total_points is a per-GW aggregate).
Player-GWs with no event coverage (unmatched names, or seasons before the FBref
scrape) keep their as-played total unchanged — no assumption is made about
their 26/27 bonus.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from data.models import RecomputedBonus


def rescore_points(total_points: int, bonus_as_played: int, bonus_2627: int) -> int:
    """One player-GW: swap the as-played bonus for the 26/27-recomputed one."""
    return int(total_points) - int(bonus_as_played) + int(bonus_2627)


def load_bonus_2627_map(session: Session, season: str) -> dict[tuple[int, int], int]:
    """(player_id, gameweek) → Σ bonus_2627 across that GW's matches (DGW-safe)."""
    rows = (
        session.query(
            RecomputedBonus.player_id,
            RecomputedBonus.gameweek,
            func.sum(RecomputedBonus.bonus_2627),
        )
        .filter(RecomputedBonus.season == season, RecomputedBonus.gameweek.isnot(None))
        .group_by(RecomputedBonus.player_id, RecomputedBonus.gameweek)
        .all()
    )
    return {(int(pid), int(gw)): int(b or 0) for pid, gw, b in rows}


def rescore_actuals(
    all_stats: pd.DataFrame, bonus_2627_map: dict[tuple[int, int], int]
) -> pd.DataFrame:
    """Add a ``total_points_2627`` column to a stats frame (player_id, gameweek,
    total_points, bonus required). Falls back to the as-played total where the
    map has no entry (no event coverage for that player-GW) — never invents a
    26/27 bonus it can't compute."""
    df = all_stats.copy()

    def _rescore(row: pd.Series) -> int:
        key = (int(row["player_id"]), int(row["gameweek"]))
        b2627 = bonus_2627_map.get(key)
        if b2627 is None:
            return int(row["total_points"])
        return rescore_points(row["total_points"], row["bonus"], b2627)

    df["total_points_2627"] = df.apply(_rescore, axis=1)
    return df


def rescore_coverage(all_stats: pd.DataFrame, bonus_2627_map: dict[tuple[int, int], int]) -> float:
    """Fraction of ALL player-GW rows that got a real 26/27 re-score (vs falling
    back to the as-played total). Most `player_gw_stats` rows are 0-minute
    squad players who never earn bonus either way, so this is dominated by
    rows where a rescore is moot — use `rescore_coverage_relevant` for the
    meaningful gate number."""
    if all_stats.empty:
        return 0.0
    keys = set(zip(all_stats["player_id"].astype(int), all_stats["gameweek"].astype(int)))
    covered = keys & set(bonus_2627_map.keys())
    return len(covered) / len(keys)


def rescore_coverage_relevant(
    all_stats: pd.DataFrame, bonus_2627_map: dict[tuple[int, int], int]
) -> float:
    """Coverage among player-GW rows that actually earned as-played bonus
    (`bonus > 0`) — the rows where getting the 26/27 recompute right actually
    matters. This is the honest exit-gate metric (~96% on 25/26 events)."""
    relevant = all_stats[all_stats["bonus"] > 0] if not all_stats.empty else all_stats
    return rescore_coverage(relevant, bonus_2627_map)
