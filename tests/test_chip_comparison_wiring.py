"""With the comparison disabled, every existing path must be byte-identical.

This is the constraint the whole change is subordinate to: the comparison
ships switched off and stays off until the persona cohort has a season on it.
"""
from __future__ import annotations

import dataclasses
import inspect

from config.strategy import CHIP_TIMING
from optimiser import chips


def test_recommend_chip_accepts_an_optional_comparison_defaulting_to_none():
    sig = inspect.signature(chips.recommend_chip)
    assert "comparison" in sig.parameters
    assert sig.parameters["comparison"].default is None


def test_comparison_is_ignored_when_the_flag_is_off():
    """A comparison that would scream 'play the free hit' must be inert."""
    from optimiser.chip_comparison import ChipComparison, ChipOption
    from optimiser.transfers import TransferPlan

    loud = ChipOption(
        chip=chips.Chip.FREE_HIT, horizon_xpts=10_000.0,
        plan=TransferPlan([], [], 0, 0.0, 0.0), detail="",
    )
    quiet = ChipOption(
        chip=None, horizon_xpts=0.0,
        plan=TransferPlan([], [], 0, 0.0, 0.0), detail="",
    )
    comparison = ChipComparison(options=[quiet, loud], no_chip=quiet, best=loud)
    timing = dataclasses.replace(CHIP_TIMING, chip_comparison_enabled=False)
    assert chips._comparison_choice(comparison, chips.Chip.FREE_HIT, timing) is None


def test_comparison_is_consulted_when_the_flag_is_on():
    from optimiser.chip_comparison import ChipComparison, ChipOption
    from optimiser.transfers import TransferPlan

    loud = ChipOption(
        chip=chips.Chip.FREE_HIT, horizon_xpts=10_000.0,
        plan=TransferPlan([], [], 0, 0.0, 0.0), detail="beat it by miles",
    )
    quiet = ChipOption(
        chip=None, horizon_xpts=0.0,
        plan=TransferPlan([], [], 0, 0.0, 0.0), detail="",
    )
    comparison = ChipComparison(options=[quiet, loud], no_chip=quiet, best=loud)
    timing = dataclasses.replace(CHIP_TIMING, chip_comparison_enabled=True)
    rec = chips._comparison_choice(comparison, chips.Chip.FREE_HIT, timing)
    assert rec is not None
    assert rec.chip is chips.Chip.FREE_HIT


def test_tc_and_bb_are_never_governed_by_the_comparison():
    """Triple Captain and Bench Boost are ORTHOGONAL to transfers -- you play
    them AND make your normal moves. A comparison verdict must not touch them
    even with the flag on, or we have replaced one bug with another."""
    from optimiser.chip_comparison import ChipComparison, ChipOption
    from optimiser.transfers import TransferPlan

    none_opt = ChipOption(None, 0.0, TransferPlan([], [], 0, 0.0, 0.0), "")
    loud = ChipOption(chips.Chip.FREE_HIT, 10_000.0, TransferPlan([], [], 0, 0.0, 0.0), "")
    comparison = ChipComparison([none_opt, loud], none_opt, loud)
    timing = dataclasses.replace(CHIP_TIMING, chip_comparison_enabled=True)

    for chip in (chips.Chip.TRIPLE_CAPTAIN, chips.Chip.BENCH_BOOST):
        assert chips._comparison_choice(comparison, chip, timing) is None


def test_the_forced_before_expiry_path_ignores_the_comparison():
    """must_play_a_chip_now salvages a chip that would otherwise be DESTROYED
    at the half boundary. There the question is 'which chip do I rescue', not
    'is this better than transfers' -- applying a margin could bin a chip to
    protect a comparison, which is strictly worse than playing it on a
    mediocre week.

    Asserted structurally: the forced block must not consult the comparison.
    """
    source = inspect.getsource(chips.recommend_chip)
    forced_block = source.split("must_play_a_chip_now")[1]
    assert "_comparison_choice" not in forced_block
    assert "comparison" not in forced_block
