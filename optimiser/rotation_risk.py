"""rotation_risk.py — hand-entered ceilings on start probability (2026-08-18).

The minutes model projects a player's minutes from his own history, which means
a summer signing carries his previous club's status wholesale. Elliot Anderson
arrived at Manchester City on the back of 37 starts for Nottingham Forest and
was handed ``start_probability = 0.97`` on that basis, with no Manchester City
minutes in evidence anywhere. Nothing in the pipeline can notice that, because
competition for a place is a fact about a squad, not about a player's record.

This is NOT a blanket new-signing discount. That was measured before this
module was written and the data does not support one: across 1,149
player-seasons, prior-season regulars who changed club retained 95.6-97.2% of
the minutes share that stayers retained — a difference indistinguishable from
noise — and both groups decline similarly (0.82 -> 0.65 of available minutes)
through ordinary regression to the mean. Whether a move costs minutes depends
entirely on who else plays there, so it is entered by hand, per player, with a
reason and a date, in ``config/transfer_overrides.yaml``.

A cap only ever lowers. If the model already rates a player below the ceiling
it is left alone, so an override cannot accidentally promote anyone.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

logger = logging.getLogger(__name__)


def apply_rotation_risk(
    projections: pd.DataFrame,
    caps_by_player: Mapping[int, float],
) -> pd.DataFrame:
    """Caps ``start_probability`` and scales the points that follow from it.

    Expected points are very nearly proportional to the probability of playing
    — a player who does not feature scores nothing — so lowering the start
    probability without scaling ``xpts`` would leave the frame internally
    inconsistent, and the optimiser reads ``xpts``. Both move together, by the
    ratio between the cap and the model's own estimate.

    ``xpts_var``, ``upside`` and ``downside`` are scaled by the same ratio, on
    the same reasoning as the cold start's own per-appearance scaling: a week
    the player does not feature contributes neither points nor spread.

    Returns a copy. Players absent from ``caps_by_player`` are untouched, and
    an empty mapping is a no-op.
    """
    if not caps_by_player or projections.empty:
        return projections
    if "start_probability" not in projections.columns:
        logger.warning(
            "apply_rotation_risk: no start_probability column; overrides not applied"
        )
        return projections

    out = projections.copy()
    scaled_cols = [
        c for c in ("xpts", "xpts_mean", "xpts_var", "upside", "downside")
        if c in out.columns
    ]
    for pid, cap in caps_by_player.items():
        mask = out["player_id"] == pid
        if not mask.any():
            continue
        current = out.loc[mask, "start_probability"].astype(float)
        # Only ever downwards, and only where the model is above the ceiling.
        ratio = (cap / current.where(current > 0.0, other=1.0)).clip(upper=1.0)
        for col in scaled_cols:
            out.loc[mask, col] = out.loc[mask, col] * ratio
        out.loc[mask, "start_probability"] = current.clip(upper=cap)
    return out


def log_capped_squad_members(
    squad_ids: list[int], players: pd.DataFrame, details: Mapping[int, dict]
) -> None:
    """Names any selected player carrying a rotation-risk override, with the
    reason. A capped player being picked anyway is a legitimate outcome — he
    may still be the best option at his price — but it should never happen
    silently, because the cap is a human judgement and it deserves to be
    re-examined when it is load-bearing."""
    if not details:
        return
    names = (
        players.set_index("id")["web_name"].to_dict() if "web_name" in players.columns else {}
    )
    for pid in squad_ids:
        entry = details.get(int(pid))
        if entry is None:
            continue
        logger.warning(
            "Squad contains %s, capped at start_probability=%.2f (%s, as of %s)",
            names.get(int(pid), pid), entry["start_probability"],
            entry.get("reason", "no reason given"), entry.get("as_of", "unknown"),
        )
