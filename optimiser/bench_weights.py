"""Bench slot weights derived from the chosen XI (2026-09-06).

A bench is an ordered queue: FPL promotes the first eligible bench player when
a starter does not appear, so slot 1 is worth P(at least one starter blanks)
and slot 3 needs three simultaneous absences.

Until now those probabilities were three constants — (0.53, 0.15, 0.03) —
computed once from the live GW1 XI, whose mean start probability was 0.93, and
never recomputed. They are therefore wrong for any other XI, and wrong in the
direction that matters: a squad carrying several 0.78s needs a much fatter slot
1 than a nailed-on one, and gets the same.

`docs/bench-and-correlation-research-2026-08-18.md` §1 names deriving these per
solve as the better answer and notes the circularity — the weights depend on
the XI, which depends on the weights.

That circularity is answered in two different shapes, because the two solvers
that consume the weights are asked two different questions (2026-09-08):

* **Ordinary weekly path.** `agent/decision_engine.py` never calls
  `optimise_squad` there — it calls `evaluate_transfers`, and a transfer ILP
  has no squad solve to derive from. `derive_bench_config` therefore hoists ONE
  derivation from the INCUMBENT XI ahead of every consumer, which is what makes
  the squad build and the weekly planner price the same bench slot identically
  (`optimiser/transfers.py:417-431` on why they must).
* **Rebuild paths** (free hit, wildcard, cold start). `optimise_squad` solves,
  derives from that solution's XI, and re-solves ONCE. Not a fixed-point loop:
  the loop this replaced kept only the immediately preceding XI, so an
  A->B->A->B cycle never tripped its equality check and it returned a solution
  whose XI was B priced under A's weights.

The two answers are not interchangeable and the gap is not an error bar. On the
live GW4 frame they differ by 0.058 / 0.049 / 0.017 on the slots and 0.102 on
the keeper, systematically and always in the same direction: `optimise_squad`
selects nailed-on players, so a solved XI's mean P(start) is the higher of the
two. Hoisting asks "how often will the bench I OWN get used?", which is the
right question for pricing a transfer; the rebuild shape asks it of the squad
being bought, which is the right question when the incumbent is about to be
sold.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping

import pandas as pd

from config.strategy import OptimiserConfig

logger = logging.getLogger(__name__)

DEFAULT_SLOTS = 3

# Start probability assumed for a squad member the projection frame cannot
# price. 0.9 is roughly the mean over players who ARE priced (0.911 on the live
# GW4 incumbent XI) and is what B3's fixed point already used as its bare
# literal in three places; it is a deliberate mid-nailed assumption, not a
# neutral one. It fires far more often than a reader would guess: the hoist
# takes its squad from the DECISION LOG, so it bypasses `optimise_squad`'s
# `start_probability >= min_start_probability` filter, and 386 of 1021 players
# carried a NaN on the live GW4 frame.
FALLBACK_START_PROBABILITY = 0.9


def at_least_k_probabilities(p_blank: list[float], k_max: int) -> list[float]:
    """[P(>=1 blank), P(>=2), ...] from independent per-player blank chances.

    Exact Poisson-binomial by convolution — the count is at most eleven, so
    there is no reason to approximate. Independence is an assumption: a
    postponed fixture blanks a whole club's players together. That correlation
    would raise the tail and so raise slots 2 and 3, making these weights
    mildly conservative, which is the safe direction.
    """
    dist = [1.0]
    for p in p_blank:
        nxt = [0.0] * (len(dist) + 1)
        for count, mass in enumerate(dist):
            nxt[count] += mass * (1.0 - p)
            nxt[count + 1] += mass * p
        dist = nxt

    out: list[float] = []
    for k in range(1, k_max + 1):
        # A tail sum is a probability and belongs in [0, 1] by construction,
        # but floating-point summation can drift a few ULPs past 1.0 when a
        # per-player blank chance is itself exactly (or clamped to) 1.0 --
        # observed as 1.0000000000000004 with an 11-player XI. Clamped here
        # rather than left to whatever reads the weight next.
        value = float(sum(dist[k:])) if k < len(dist) else 0.0
        out.append(min(1.0, max(0.0, value)))
    return out


def derive_slot_weights(
    blank_probabilities: list[float], n_slots: int = DEFAULT_SLOTS
) -> tuple[float, ...]:
    """Outfield bench slot weights from the chosen XI's blank probabilities.

    ``blank_probabilities`` means P(0 minutes) per starter — the actual FPL
    autosub trigger — not P(<60 minutes). Feeding this ``1 - P(60+)`` instead
    (as this function did until 2026-09-22) folds in a 1-59 minute cameo as if
    it were a blank, which it is not: measured 2x too high for nailed-on
    players on the live 2026-27 frame. Callers without a real P(0 minutes)
    figure (e.g. the cold-start frame, which has no minutes-model bands) still
    pass ``1 - start_probability`` as their best available approximation —
    that conflation is now the CALLER's decision, not this function's.

    Returns strictly decreasing weights, which is what lets
    ``_bench_objective`` skip an explicit ordering constraint: with decreasing
    weights any assignment other than best-player-in-slot-1 is dominated.

    Raises ``ValueError`` on a non-finite input. Defence in depth behind
    ``bench_config_for_xi``, which sanitises: a NaN used to come back out as
    ``(nan, nan, nan)``, and PuLP then raised ``PulpError`` — not a
    ``RuntimeError``, so ``generate_squad_pool``'s handler did not catch it.
    """
    if not blank_probabilities:
        return tuple([0.0] * n_slots)
    blanks = [_finite(p, "a starting XI") for p in blank_probabilities]
    return tuple(at_least_k_probabilities(blanks, n_slots))


def derive_gk_weight(keeper_blank_probability: float) -> float:
    """The reserve keeper plays only if the first choice blanks (0 minutes) —
    one keeper, one chance, no queue to inherit from.

    Takes P(0 minutes) directly, for the same reason ``derive_slot_weights``
    now does (see its docstring, 2026-09-22).

    Raises ``ValueError`` on a non-finite input. Clamped to ``[0, 1]`` rather
    than trusting the input outright: an out-of-range upstream value used to
    reach here as ``1 - p`` for ``p`` already clamped by the caller, but a
    direct blank probability has no such built-in ceiling.
    """
    return min(1.0, max(0.0, _finite(keeper_blank_probability, "the first-choice keeper")))


def _finite(value: float, what: str) -> float:
    """``float(value)``, or ``ValueError`` if it is not a real number."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite start probability {value!r} for {what}")
    return out


FALLBACK_BLANK_PROBABILITY = 1.0 - FALLBACK_START_PROBABILITY


def _blank_probability(
    blank_probabilities: Mapping[int, float], player_id: int
) -> float:
    """One squad member's blank probability, sanitised to a usable number.

    Two distinct holes, and only one of them is obvious. A player MISSING from
    the mapping (transferred out, departed) hits the ``get`` default; a player
    PRESENT with a NaN does not — ``{1: nan}.get(1, 0.1)`` is ``nan``, and a
    ``players.merge(..., how="left")`` leaves every unmatched id present with
    exactly that. Both end up on ``FALLBACK_BLANK_PROBABILITY`` here, which is
    the single place either can.
    """
    raw = blank_probabilities.get(player_id, FALLBACK_BLANK_PROBABILITY)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        logger.warning(
            "Bench weights: player %s has no usable blank probability (%r); "
            "assuming %.2f",
            player_id, raw, FALLBACK_BLANK_PROBABILITY,
        )
        return FALLBACK_BLANK_PROBABILITY
    if not 0.0 <= value <= 1.0:
        # A negative or >1 blank chance makes the Poisson-binomial convolution
        # produce weights outside [0, 1] and the ILP's bench term nonsense.
        # Clamping is a repair, so it is logged.
        logger.warning(
            "Bench weights: player %s has an out-of-range blank probability "
            "(%.4f); clamping",
            player_id, value,
        )
        return min(1.0, max(0.0, value))
    return value


def blank_probability_map(players: pd.DataFrame) -> dict[int, float]:
    """Per-player P(0 minutes), preferring the real thing over the old proxy.

    Reads the ``blank_probability`` column (``assemble.py``'s p0, straight
    from the minutes model's own bands) when present. Falls back to
    ``1 - start_probability`` (P(60+ minutes)) otherwise — every bench-weight
    consumer's behaviour before 2026-09-22, and still the best available
    figure for a frame with no minutes-model bands to draw p0 from, such as
    the cold start's ``start_probability`` (which already means bare P(appear)
    there, not P(60+)). The fallback OVERSTATES the true blank chance by
    folding in a 1-59 minute cameo — measured 2x too high for nailed-on
    players on the live 2026-27 frame — so it is a deliberately inferior
    stand-in, not an equivalent choice.
    """
    if "blank_probability" in players.columns:
        blank = players["blank_probability"].astype(float)
    else:
        blank = 1.0 - players["start_probability"].astype(float)
    return dict(zip(players["id"], blank, strict=True))


def bench_config_for_xi(
    cfg: OptimiserConfig,
    xi_ids: list[int],
    blank_probabilities: Mapping[int, float],
    positions_by_id: Mapping[int, str],
) -> OptimiserConfig:
    """``cfg`` with the bench weights this XI implies.

    The single derivation point for both shapes described in the module
    docstring, and so the single place a missing or NaN blank probability is
    handled — see ``_blank_probability``. ``blank_probabilities`` should be
    real P(0 minutes) figures (``blank_probability_map`` builds them); pass
    ``1 - start_probability`` yourself only where no better number exists.

    An XI with no keeper in it falls back rather than raising: that means the
    caller handed us something that is not a legal XI, which is a caller bug,
    but pricing the reserve keeper on an assumption beats failing a whole
    decision run over the bench's fourth-most-important coefficient.
    """
    outfield = [
        _blank_probability(blank_probabilities, pid)
        for pid in xi_ids
        if positions_by_id.get(pid) != "GKP"
    ]
    keepers = [
        _blank_probability(blank_probabilities, pid)
        for pid in xi_ids
        if positions_by_id.get(pid) == "GKP"
    ]
    if not keepers:
        logger.warning(
            "Bench weights: the XI has no goalkeeper in it; pricing the "
            "reserve keeper off %.2f",
            FALLBACK_BLANK_PROBABILITY,
        )
    return dataclasses.replace(
        cfg,
        bench_slot_weights=derive_slot_weights(outfield),
        bench_gk_weight=derive_gk_weight(
            keepers[0] if keepers else FALLBACK_BLANK_PROBABILITY
        ),
    )


def derive_bench_config(
    cfg: OptimiserConfig,
    squad_ids: list[int],
    projections: pd.DataFrame,
    players: pd.DataFrame,
    gw: int,
) -> OptimiserConfig:
    """One bench-weight derivation per decision, shared by every consumer.

    Derives from the INCUMBENT squad's projected XI, which is the only shape
    available on the ordinary weekly path: ``evaluate_transfers`` never solves
    a squad, so nothing downstream of it can derive weights of its own, and a
    derivation that lived inside ``optimise_squad`` could not reach it at all.

    Returns ``cfg`` unchanged — each with a log line saying which — when the
    flag is off, when ``squad_ids`` is empty, when start probabilities are
    unavailable, when the saved squad is too short to field an eleven, or when
    the XI solve fails. An empty ``squad_ids`` is the COLD START, and returning
    ``cfg`` there is correct rather than a silent disabling: a cold start is a
    rebuild path and gets its weights from ``optimise_squad``'s own
    derive-and-re-solve pass instead.

    ``gw`` is the gameweek the XI is chosen for — the one being decided.
    """
    if not cfg.derive_bench_weights_per_solve:
        return cfg
    if not squad_ids:
        logger.info(
            "Bench weights: no incumbent squad to derive from; keeping the "
            "static weights (a cold start derives from its own solve instead)"
        )
        return cfg
    if "start_probability" not in players.columns:
        logger.warning(
            "Bench weights: `players` has no start_probability column; "
            "keeping the static weights"
        )
        return cfg

    squad = players[players["id"].isin(squad_ids)]
    if len(squad) < 11:
        logger.warning(
            "Bench weights: only %d of %d squad members are in `players`, "
            "which cannot field an eleven; keeping the static weights",
            len(squad), len(squad_ids),
        )
        return cfg

    # Lazy: `optimiser.squad` imports THIS module, so a top-level import here
    # would be circular. Reusing `optimise_starting_xi` rather than a private
    # greedy pick is the point -- it is the project's one definition of which
    # eleven start, and its objective does not read the bench weights, so
    # asking it here is not circular in the other sense either.
    from optimiser.squad import optimise_starting_xi

    try:
        # season=None deliberately: scenario-based captaincy would open the
        # database, and which player wears the armband does not move a bench
        # weight by a single decimal.
        xi = optimise_starting_xi(squad, projections, gw, config=cfg)
    except Exception as exc:  # noqa: BLE001 -- a failed XI solve is not a failed decision
        logger.warning(
            "Bench weights: the incumbent XI did not solve (%s); keeping the "
            "static weights", exc,
        )
        return cfg

    out = bench_config_for_xi(
        cfg,
        [int(pid) for pid in xi.starting_xi["id"]],
        blank_probability_map(squad),
        dict(zip(squad["id"], squad["position"], strict=True)),
    )
    logger.info(
        "Bench weights derived from the incumbent XI: slots=%s gk=%.3f "
        "(was slots=%s gk=%.3f)",
        tuple(round(w, 3) for w in out.bench_slot_weights), out.bench_gk_weight,
        tuple(round(w, 3) for w in cfg.bench_slot_weights), cfg.bench_gk_weight,
    )
    return out


def bench_slot_weight_for_week(
    slot: int, w: int, bb_target_week: int | None, cfg: OptimiserConfig
) -> float:
    """Bench slot weight for one slot in one week.

    Under a bench boost all four bench players play, so every slot is worth
    1.0 in the target week and the ordinary auto-substitution weights apply
    everywhere else. A flat 1.0 across the whole horizon would build a squad
    that is BB-optimal and mediocre in the other 37 weeks.

    Lives here, once, and is imported by BOTH optimiser/squad.py and
    optimiser/transfers.py. They must not drift: the squad build pays real
    money for a substitute who plays, and if the weekly transfer ILP valued him
    lower it would sell him the following week and charge a transfer for the
    privilege (see transfers.py:417-431). A single definition makes that
    structural rather than something a test has to notice afterwards.
    """
    if bb_target_week is not None and w == bb_target_week:
        return 1.0
    return cfg.bench_slot_weights[slot] * cfg.bench_value_weight


def bench_gk_weight_for_week(
    w: int, bb_target_week: int | None, cfg: OptimiserConfig
) -> float:
    """As above, for the reserve keeper."""
    if bb_target_week is not None and w == bb_target_week:
        return 1.0
    return cfg.bench_gk_weight * cfg.bench_value_weight
