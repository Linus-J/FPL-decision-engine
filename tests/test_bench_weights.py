"""Bench slot weights derived from the XI actually chosen (2026-09-06).

The shipped constants (0.53, 0.15, 0.03) were computed once from a GW1 XI
whose mean start probability was 0.93 and never recomputed. An XI carrying
several 0.78s needs a materially fatter slot 1; the live squad bought three
bench players at P(start) <= 0.03 partly because slot 3 is worth 0.03."""

import pytest

from optimiser.bench_weights import (
    at_least_k_probabilities,
    derive_gk_weight,
    derive_slot_weights,
)


def test_a_certain_blank_gives_probability_one():
    assert at_least_k_probabilities([1.0], 1) == pytest.approx([1.0])


def test_nobody_can_blank_gives_probability_zero():
    assert at_least_k_probabilities([0.0, 0.0], 2) == pytest.approx([0.0, 0.0])


def test_two_independent_coin_flips():
    """P(>=1) = 1 - 0.25 = 0.75; P(>=2) = 0.25."""
    assert at_least_k_probabilities([0.5, 0.5], 2) == pytest.approx([0.75, 0.25])


def test_probabilities_are_non_increasing_in_k():
    out = at_least_k_probabilities([0.2, 0.3, 0.4, 0.1, 0.25], 3)
    assert out[0] >= out[1] >= out[2]


def test_asking_for_more_slots_than_players_is_not_an_error():
    out = at_least_k_probabilities([0.5], 3)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)


def test_reproduces_the_shipped_constants_on_the_xi_they_came_from():
    """Ten outfield starters at the documented mean P(start) of 0.93 (blank
    0.07) should land near (0.53, 0.15, 0.03) — the numbers this replaces."""
    weights = derive_slot_weights([0.07] * 10)
    assert weights[0] == pytest.approx(0.53, abs=0.03)
    assert weights[1] == pytest.approx(0.15, abs=0.03)
    assert weights[2] == pytest.approx(0.03, abs=0.02)


def test_a_fragile_xi_buys_more_insurance_than_a_nailed_on_one():
    fragile = derive_slot_weights([0.22] * 10)
    nailed = derive_slot_weights([0.03] * 10)
    assert fragile[0] > nailed[0]
    assert fragile[1] > nailed[1]


def test_weights_are_strictly_decreasing_so_the_solver_orders_the_bench():
    """_bench_objective relies on this: any other slot assignment is dominated,
    so no explicit ordering constraint is needed."""
    weights = derive_slot_weights([0.85] * 10)
    assert weights[0] > weights[1] > weights[2]


def test_the_reserve_keeper_is_worth_the_first_choice_keepers_blank_chance():
    assert derive_gk_weight(0.03) == pytest.approx(0.03)
    assert derive_gk_weight(0.20) == pytest.approx(0.20)


def test_an_empty_xi_falls_back_to_zero_rather_than_raising():
    assert derive_slot_weights([]) == pytest.approx((0.0, 0.0, 0.0))


def test_flag_off_reproduces_the_static_weights_exactly(monkeypatch):
    """The whole point of the flag: the change must be measurable, not assumed."""
    import dataclasses

    from helpers_squad import flat_projections, sixteen_players

    from config.strategy import OPTIMISER
    from optimiser.squad import optimise_squad

    players = sixteen_players()
    proj = flat_projections(players, gws=(1,))
    off = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=False)
    a = optimise_squad(proj, players, budget=100.0, horizon=1, config=off)
    b = optimise_squad(proj, players, budget=100.0, horizon=1, config=off)
    assert set(a.squad["id"]) == set(b.squad["id"])


def test_flag_on_lifts_the_bench_of_a_fragile_squad():
    """A squad of rotation risks should buy a better first substitute than the
    static 0.53 would justify."""
    import dataclasses

    from helpers_squad import flat_projections, sixteen_players

    from config.strategy import OPTIMISER
    from optimiser.squad import optimise_squad

    players = sixteen_players()
    players["start_probability"] = 0.70
    proj = flat_projections(players, gws=(1,))

    on = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)
    out = optimise_squad(proj, players, budget=100.0, horizon=1, config=on)
    assert len(out.squad) == 15


# --- B3c: the hoist, and the NaN hardening it makes a precondition ---------
#
# `derive_bench_config` reads start probabilities off the INCUMBENT squad,
# taken from the decision log, and so bypasses `optimise_squad`'s
# `start_probability >= min_start_probability` filter. NaN fails every `>=`,
# so that filter was what kept a NaN out of the weights; without it a squad
# member absent from this gameweek's projections reaches the derivation
# directly. On the live GW4 frame 386 of 1021 players carry a NaN.


def test_a_nan_start_probability_does_not_poison_the_weights():
    """`players.merge(how="left")` leaves an absent player PRESENT with a NaN,
    so `probs.get(pid, default)` returns the NaN — the default fires only on a
    MISSING key. One NaN made `derive_slot_weights` return (nan, nan, nan),
    and PuLP then raised `PulpError`, which is not a `RuntimeError` and so is
    not caught by `generate_squad_pool`'s handler."""
    import math

    from config.strategy import OPTIMISER
    from optimiser.bench_weights import bench_config_for_xi

    xi = list(range(1, 12))
    probs = {pid: 0.9 for pid in xi}
    probs[5] = float("nan")
    positions = {pid: ("GKP" if pid == 1 else "MID") for pid in xi}

    out = bench_config_for_xi(OPTIMISER, xi, probs, positions)
    assert all(math.isfinite(w) for w in out.bench_slot_weights)
    assert out.bench_slot_weights[0] > out.bench_slot_weights[1] > out.bench_slot_weights[2]


def test_a_nan_keeper_probability_is_not_silently_priced_at_zero():
    """`derive_gk_weight(nan)` returned 0.0 — it did not propagate the poison,
    it quietly valued the bench keeper at nothing. That is the harder failure
    of the two to notice."""
    from config.strategy import OPTIMISER
    from optimiser.bench_weights import FALLBACK_START_PROBABILITY, bench_config_for_xi

    xi = list(range(1, 12))
    probs = {pid: 0.9 for pid in xi}
    probs[1] = float("nan")
    positions = {pid: ("GKP" if pid == 1 else "MID") for pid in xi}

    out = bench_config_for_xi(OPTIMISER, xi, probs, positions)
    assert out.bench_gk_weight == pytest.approx(1.0 - FALLBACK_START_PROBABILITY)
    assert out.bench_gk_weight > 0.0


def test_a_squad_member_missing_from_the_frame_falls_back_rather_than_vanishing():
    """A departed player can be in the saved squad and absent from the merge
    entirely. Ten outfielders must still be priced as ten, not as nine."""
    from config.strategy import OPTIMISER
    from optimiser.bench_weights import FALLBACK_BLANK_PROBABILITY, bench_config_for_xi

    xi = list(range(1, 12))
    positions = {pid: ("GKP" if pid == 1 else "MID") for pid in xi}
    probs = {pid: FALLBACK_BLANK_PROBABILITY for pid in xi}

    complete = bench_config_for_xi(OPTIMISER, xi, probs, positions)
    del probs[5]
    with_a_hole = bench_config_for_xi(OPTIMISER, xi, probs, positions)
    assert with_a_hole.bench_slot_weights == pytest.approx(complete.bench_slot_weights)


def test_the_primitives_refuse_a_non_finite_input():
    """Defence in depth. After the boundary sanitises, these can only be
    reached with a NaN by a caller that skipped it — and then the failure must
    be loud rather than `(nan, nan, nan)` or a keeper worth 0.0."""
    with pytest.raises(ValueError):
        derive_slot_weights([0.9, float("nan"), 0.8])
    with pytest.raises(ValueError):
        derive_gk_weight(float("nan"))
    with pytest.raises(ValueError):
        derive_gk_weight(float("inf"))


def test_a_probability_outside_zero_to_one_is_clamped_not_negated():
    """An out-of-range blank chance makes the Poisson-binomial convolution
    produce weights outside [0, 1] and the ILP nonsense. Clamp rather than
    trust the upstream column."""
    from config.strategy import OPTIMISER
    from optimiser.bench_weights import bench_config_for_xi

    xi = list(range(1, 12))
    positions = {pid: ("GKP" if pid == 1 else "MID") for pid in xi}
    probs = {pid: 0.1 for pid in xi}
    probs[5] = 1.4
    out = bench_config_for_xi(OPTIMISER, xi, probs, positions)
    assert all(0.0 <= w <= 1.0 for w in out.bench_slot_weights)
