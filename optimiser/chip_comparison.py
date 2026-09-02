"""Horizon-total comparison of Free Hit / Wildcard against the transfer plan
they displace (2026-09-02).

``optimiser/chips.py`` gates each chip on a gain measured against DOING
NOTHING with the current squad, and ``agent/decision_engine.py`` decides the
chip BEFORE ``evaluate_transfers`` ever runs. So the two options that are
mutually exclusive with the transfer plan -- Free Hit and Wildcard -- are
never compared against it, and the free transfers a Free Hit preserves are
never priced at all.

Those are opposing errors: measuring against "do nothing" OVERSTATES a chip,
ignoring banked transfers UNDERSTATES it. They partly cancel, which is why
this has not surfaced as an obviously bad decision.

The fix is not to subtract one gain from another -- Free Hit gain is one
gameweek, the transfer planner runs to three and the wildcard to five, and
``chips.py`` already excludes the wildcard from its own forced-chip
comparison for exactly that reason. Instead every option is scored as TOTAL
squad xPts over one common horizon, which makes them commensurable and prices
the banked transfers via the Free Hit's continuation rather than an invented
coefficient.

Triple Captain and Bench Boost are deliberately absent: they are orthogonal
to transfers (you play them AND make your normal moves), so subtracting a
transfer-plan baseline from them would be a new bug.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from optimiser.squad import optimise_squad_joint
from optimiser.transfers import (
    TransferPlan,
    evaluate_transfers,
    roll_forward_free_transfers,
    squad_xpts,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChipOption:
    """One mutually-exclusive way to play the current gameweek.

    ``horizon_xpts`` is a TOTAL, not a gain against some private baseline --
    that is the whole point. Two options are comparable only because they are
    measured the same way over the same gameweeks.
    """

    chip: object | None  # Chip | None; typed loosely to avoid a circular import
    horizon_xpts: float
    plan: TransferPlan
    detail: str


@dataclass(frozen=True)
class ChipComparison:
    """Every option that solved, plus the choice the margins imply."""

    options: list[ChipOption]
    no_chip: ChipOption | None
    best: ChipOption | None


def _base_xpts(
    current_squad_ids: list[int], projections: pd.DataFrame, horizon: int
) -> float:
    """The current squad's value over ``horizon``, doing nothing.

    ``squad_xpts`` is the project's single definition of what a squad is worth
    (best eleven plus a captain), and every option here must use it or the
    totals are not on one scale.
    """
    return squad_xpts(current_squad_ids, projections, horizon=horizon)


def build_no_chip_option(
    current_squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    *,
    free_transfers: int,
    horizon: int,
    available_budget: float | None = None,
    bank: float | None = None,
    purchase_prices: dict[int, float] | None = None,
    ownership: pd.DataFrame | None = None,
    config=None,
    transfer_rules=None,
) -> ChipOption | None:
    """The alternative a chip actually displaces: keep the squad, make the
    best transfers. ``None`` when the solve fails -- see module docstring on
    why that is not a zero.
    """
    try:
        plan = evaluate_transfers(
            current_squad_ids=current_squad_ids,
            projections=projections,
            players=players,
            free_transfers=free_transfers,
            available_budget=available_budget,
            bank=bank,
            purchase_prices=purchase_prices,
            ownership=ownership,
            config=config,
            transfer_rules=transfer_rules,
            horizon=horizon,
        )
    except Exception as exc:  # noqa: BLE001 -- an unsolved option is unknown, not worthless
        logger.warning("chip comparison: no-chip option did not solve (%s)", exc)
        return None
    total = _base_xpts(current_squad_ids, projections, horizon) + plan.net_xpts_gain
    return ChipOption(
        chip=None,
        horizon_xpts=total,
        plan=plan,
        detail=(
            f"no chip: {len(plan.transfers_in)} transfer(s), "
            f"{plan.hits_taken} hit(s), net {plan.net_xpts_gain:+.2f}"
        ),
    )


def build_wildcard_option(
    current_squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    *,
    horizon: int,
    chip,
    available_budget: float | None = None,
    bank: float | None = None,
    purchase_prices: dict[int, float] | None = None,
    ownership: pd.DataFrame | None = None,
    config=None,
    transfer_rules=None,
) -> ChipOption | None:
    """A wildcard is a transfer plan with the allowance lifted, so it is
    evaluated by the SAME function that will execute it (chips.py §12, 2026-08-18
    -- evaluating it with a different optimiser meant the chip fired on one
    number and delivered another)."""
    try:
        plan = evaluate_transfers(
            current_squad_ids=current_squad_ids,
            projections=projections,
            players=players,
            free_transfers=1,
            available_budget=available_budget,
            wildcard_active=True,
            bank=bank,
            purchase_prices=purchase_prices,
            ownership=ownership,
            config=config,
            transfer_rules=transfer_rules,
            horizon=horizon,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("chip comparison: wildcard option did not solve (%s)", exc)
        return None
    total = _base_xpts(current_squad_ids, projections, horizon) + plan.net_xpts_gain
    return ChipOption(
        chip=chip,
        horizon_xpts=total,
        plan=plan,
        detail=f"wildcard: rebuild worth {plan.net_xpts_gain:+.2f} over {horizon} GWs",
    )


def build_free_hit_option(
    current_squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    *,
    free_transfers: int,
    horizon: int,
    current_gw: int,
    chip,
    available_budget: float | None = None,
    bank: float | None = None,
    purchase_prices: dict[int, float] | None = None,
    ownership: pd.DataFrame | None = None,
    season: str | None = None,
    config=None,
    transfer_rules=None,
) -> ChipOption | None:
    """One week of the best legal eleven, then the ORIGINAL squad back --
    still holding the free transfers the chip did not spend.

    That continuation is where banked transfers get priced. No coefficient is
    invented for them: the continuation genuinely still has them, so its plan
    is simply worth more. This is the whole reason the comparison is done on
    horizon totals rather than on one-week gains.
    """
    gws = sorted(projections["gameweek"].unique())[:horizon]
    if not gws:
        return None
    fh_gw, rest_gws = gws[0], gws[1:]

    try:
        solution = optimise_squad_joint(
            projections[projections["gameweek"] == fh_gw],
            players,
            budget=available_budget,
            horizon=1,
            season=season,
            gameweek=current_gw,
            ownership=ownership,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("chip comparison: free hit option did not solve (%s)", exc)
        return None

    fh_squad_ids = solution.squad["id"].tolist()
    fh_week = squad_xpts(
        fh_squad_ids, projections[projections["gameweek"] == fh_gw], horizon=1
    )

    if not rest_gws:
        # At the horizon edge there is nothing after this week to plan.
        return ChipOption(
            chip=chip,
            horizon_xpts=fh_week,
            plan=TransferPlan(
                transfers_in=[], transfers_out=[], hits_taken=0,
                xpts_gain=0.0, net_xpts_gain=0.0,
            ),
            detail=f"free hit: {fh_week:.2f} over 1 GW (no continuation)",
        )

    rest = projections[projections["gameweek"].isin(rest_gws)]
    # transfers_made=0: a Free Hit's transfers are outside the allowance
    # entirely, so nothing is spent and the saved transfers survive.
    continuation_fts = roll_forward_free_transfers(
        free_transfers, transfers_made=0, free_hit_played=True,
        transfer_rules=transfer_rules,
    )
    try:
        plan = evaluate_transfers(
            current_squad_ids=current_squad_ids,
            projections=rest,
            players=players,
            free_transfers=continuation_fts,
            available_budget=available_budget,
            bank=bank,
            purchase_prices=purchase_prices,
            ownership=ownership,
            config=config,
            transfer_rules=transfer_rules,
            horizon=len(rest_gws),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("chip comparison: free hit continuation did not solve (%s)", exc)
        return None

    total = fh_week + _base_xpts(current_squad_ids, rest, len(rest_gws)) + plan.net_xpts_gain
    return ChipOption(
        chip=chip,
        horizon_xpts=total,
        plan=plan,
        detail=(
            f"free hit: {fh_week:.2f} in GW{fh_gw}, then {continuation_fts} FT(s) "
            f"into a continuation worth {plan.net_xpts_gain:+.2f}"
        ),
    )


def pick_best(
    options: list[ChipOption],
    *,
    free_hit_margin: float,
    wildcard_margin: float,
    free_hit_chip,
    wildcard_chip,
) -> ChipOption | None:
    """The chip that beats the no-chip option by at least its margin, or None.

    Returns None when the no-chip option is absent: there is no honest margin
    to measure against a baseline that did not solve, so the caller must fall
    back to the legacy path rather than guess.
    """
    no_chip = next((o for o in options if o.chip is None), None)
    if no_chip is None:
        return None
    margins = {free_hit_chip: free_hit_margin, wildcard_chip: wildcard_margin}
    qualifying = [
        o
        for o in options
        if o.chip is not None
        and o.horizon_xpts - no_chip.horizon_xpts >= margins.get(o.chip, float("inf"))
    ]
    if not qualifying:
        return None
    return max(qualifying, key=lambda o: o.horizon_xpts)
