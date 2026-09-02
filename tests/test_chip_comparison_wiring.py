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
    comparison = ChipComparison(
        options=[quiet, loud], no_chip=quiet, best=loud, ranked=[loud],
    )
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
    comparison = ChipComparison(
        options=[quiet, loud], no_chip=quiet, best=loud, ranked=[loud],
    )
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
    comparison = ChipComparison([none_opt, loud], none_opt, loud, [loud])
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

    This only checks the forced block's OWN source text, so it cannot see
    through a `candidate(force=True)` call into `_try_fh`/`_try_wc` -- it does
    not, on its own, prove the force path is unaffected by the comparison.
    See `test_forced_free_hit_still_fires_when_the_comparison_ran_but_rejected_it`
    below for a behavioural test that exercises the force path directly.
    """
    source = inspect.getsource(chips.recommend_chip)
    forced_block = source.split("must_play_a_chip_now")[1]
    assert "_comparison_choice" not in forced_block
    assert "comparison" not in forced_block


def _solvable_squad_scenario() -> tuple[object, list[int], object]:
    """A legal 15 of poor (owned) players inside a much better pool, over one
    gameweek -- mirrors `tests/test_chips.py::_wildcard_scenario`, trimmed to
    a single gameweek since this only needs to exercise `_try_fh`'s own
    `optimise_squad(horizon=1)` solve, not a multi-GW wildcard rebuild.
    """
    import pandas as pd

    rows = []
    pid = 0
    for team in range(1, 16):
        for pos, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            for _ in range(count):
                pid += 1
                rows.append({
                    "id": pid, "web_name": f"p{pid}", "position": pos,
                    "team_id": team, "now_cost": 4.0, "status": "a",
                })
    players = pd.DataFrame(rows)
    squad, per_club = [], {}
    for pos, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        taken = 0
        for r in players[players["position"] == pos].itertuples():
            if per_club.get(r.team_id, 0) < 3:
                squad.append(r.id)
                per_club[r.team_id] = per_club.get(r.team_id, 0) + 1
                taken += 1
            if taken == count:
                break
    owned = set(squad)
    proj = pd.DataFrame([
        {"player_id": r.id, "gameweek": 5,
         "xpts": 1.0 if r.id in owned else 6.0,
         "xpts_var": 1.0, "start_probability": 0.9}
        for r in players.itertuples()
    ])
    return players, squad, proj


def test_forced_free_hit_still_fires_when_the_comparison_ran_but_rejected_it(monkeypatch):
    """Real regression, not just the structural check above.

    `recommend_chip`'s forced-before-expiry salvage block calls
    `_try_fh(force=True)` to rescue a chip that would otherwise be DESTROYED
    at the half boundary. Before this fix, `_try_fh`'s comparison statements
    sat ABOVE every `force` check, so a comparison that ran, solved its
    baseline (`no_chip` present), and did NOT pick the Free Hit (`best` is
    None here -- nothing beat the baseline) made `_try_fh` return `None`
    unconditionally, even under `force=True`. That excludes Free Hit from the
    forced candidates and bins a chip that should have been salvaged.

    Exercised through the real `chips.recommend_chip` entry point (never by
    inspecting source): at the literal expiry gameweek, with Triple Captain,
    Bench Boost and Wildcard all already spent this half, Free Hit is the
    ONLY chip `must_play_a_chip_now` can force. If the bug is present, the
    whole recommendation collapses to "No chip threshold met" even though a
    chip is about to be destroyed for nothing -- that is the failure this
    test catches, and inspecting the forced block's source text alone cannot.
    """
    import dataclasses

    from optimiser.chip_comparison import ChipComparison, ChipOption
    from optimiser.transfers import TransferPlan

    chips._get_wc_half_boundary.cache_clear()
    monkeypatch.setattr(chips, "_get_wc_half_boundary", lambda season=None: 5)

    players, squad, proj = _solvable_squad_scenario()

    no_chip = ChipOption(
        chip=None, horizon_xpts=500.0,
        plan=TransferPlan([], [], 0, 0.0, 0.0), detail="baseline solved",
    )
    comparison = ChipComparison(options=[no_chip], no_chip=no_chip, best=None)
    timing = dataclasses.replace(CHIP_TIMING, chip_comparison_enabled=True)

    rec = chips.recommend_chip(
        current_gw=5,
        current_squad_ids=squad,
        projections=proj,
        players=players,
        available_budget=200.0,
        free_transfers=1,
        # Only Free Hit is left available this half -- TC/BB/WC already spent.
        chips_used=[
            (chips.Chip.TRIPLE_CAPTAIN, 5),
            (chips.Chip.BENCH_BOOST, 5),
            (chips.Chip.WILDCARD, 5),
        ],
        squad_age_gws=99,
        chip_timing=timing,
        comparison=comparison,
    )

    assert rec.chip == chips.Chip.FREE_HIT
    assert "Forced before expiry" in rec.reason


def test_chip_comparison_log_columns():
    from data.models import ChipComparisonLog

    cols = {c.name for c in ChipComparisonLog.__table__.columns}
    assert {
        "season", "gameweek", "sim_manager_id", "option",
        "horizon_xpts", "chosen_live", "chosen_shadow", "created_at",
    } <= cols


def test_persist_marks_live_and_shadow_choices_separately():
    """Divergence is the measurement, so both must be recorded per option."""
    from agent.decision_engine import _chip_comparison_rows
    from optimiser.chip_comparison import ChipComparison, ChipOption
    from optimiser.chips import Chip
    from optimiser.transfers import TransferPlan

    none_opt = ChipOption(None, 100.0, TransferPlan([], [], 0, 0.0, 0.0), "no chip")
    fh_opt = ChipOption(Chip.FREE_HIT, 120.0, TransferPlan([], [], 0, 0.0, 0.0), "fh")
    comparison = ChipComparison([none_opt, fh_opt], none_opt, fh_opt)

    rows = _chip_comparison_rows(
        season="2026-27", gameweek=3, sim_manager_id=None,
        comparison=comparison, live_chip=None,
    )
    by_option = {r["option"]: r for r in rows}
    assert by_option["none"]["chosen_live"] is True      # live played no chip
    assert by_option["none"]["chosen_shadow"] is False
    assert by_option["free_hit"]["chosen_shadow"] is True  # comparison wanted FH
    assert by_option["free_hit"]["chosen_live"] is False
