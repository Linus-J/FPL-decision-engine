"""The comparison may nominate a chip the engine will never play.

Task 7b (2026-09-02). The GW3 live frame produced this comparison:

    none      274.30
    free_hit  288.49   (+14.19 over no-chip, margin 12.0 -- qualifies)
    wildcard  323.69   (+49.39 over no-chip, margin 25.0 -- qualifies, and wins)

`pick_best` nominated the wildcard, but the squad was three gameweeks old and
`wildcard_min_managed_gws` is six, so `_try_wc` refused it before ever reading
the comparison. `_try_fh` had already run, seen that the single nominated
winner was not the Free Hit, and returned None. The engine played no chip and
took a -4 hit, with a qualifying Free Hit sitting unplayed.

The lesson is not "special-case the wildcard age gate": ANY chip-specific
downstream guard reproduces this shape. A single winner is the wrong contract
for a decision that can be refused after the fact, so the comparison now also
exposes every option that cleared its own margin, in preference order.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from config.strategy import CHIP_TIMING
from optimiser import chips
from optimiser.chip_comparison import ChipComparison, ChipOption, rank_qualifying
from optimiser.transfers import TransferPlan


def _plan() -> TransferPlan:
    return TransferPlan([], [], 0, 0.0, 0.0)


def _squad_scenario(gameweek: int = 5) -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """A legal 15 of poor (owned) players inside a much better pool.

    Mirrors `tests/test_chip_comparison_wiring.py::_solvable_squad_scenario`.
    The gap between owned (1.0) and pool (6.0) is deliberately huge so the
    LEGACY Free Hit threshold clears easily -- that is what makes the
    "genuinely failed its margin is still blocked" test meaningful: only the
    guard can be suppressing the chip there, never a failed legacy solve.
    """
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
        {"player_id": r.id, "gameweek": gameweek,
         "xpts": 1.0 if r.id in owned else 6.0,
         "xpts_var": 1.0, "start_probability": 0.9}
        for r in players.itertuples()
    ])
    return players, squad, proj


@pytest.fixture
def half_boundary_at_19(monkeypatch):
    """No forced-before-expiry salvage anywhere in these tests.

    Every scenario here sits at GW5 with a boundary at GW19, so
    `must_play_a_chip_now` is false and the recommendation under test is the
    ordinary threshold path -- not the legacy salvage block, which is
    deliberately out of the comparison's reach.

    The cache is cleared before patching, not after: `monkeypatch` restores the
    real `lru_cache`-wrapped function on teardown, and nothing repopulated it
    while the stand-in was in place.
    """
    chips._get_wc_half_boundary.cache_clear()
    monkeypatch.setattr(chips, "_get_wc_half_boundary", lambda season=None: 19)


def test_a_qualifying_free_hit_survives_a_wildcard_nomination_the_engine_refuses(
    half_boundary_at_19,
):
    """The live GW3 bug, reproduced.

    Both chips cleared their margins; the wildcard out-ranked the free hit and
    was then refused on squad age. The free hit must still be played.
    """
    none_opt = ChipOption(None, 274.30, _plan(), "no chip: 2 transfer(s), 1 hit(s)")
    fh_opt = ChipOption(chips.Chip.FREE_HIT, 288.49, _plan(), "free hit: 72.61 in GW3")
    wc_opt = ChipOption(chips.Chip.WILDCARD, 323.69, _plan(), "wildcard: rebuild +65.21")
    comparison = ChipComparison(
        options=[none_opt, fh_opt, wc_opt],
        no_chip=none_opt,
        best=wc_opt,
        ranked=[wc_opt, fh_opt],
    )
    timing = dataclasses.replace(CHIP_TIMING, chip_comparison_enabled=True)
    players, squad, proj = _squad_scenario()

    rec = chips.recommend_chip(
        current_gw=5,
        current_squad_ids=squad,
        projections=proj,
        players=players,
        available_budget=200.0,
        free_transfers=1,
        # TC and BB already spent, so neither can preempt the FH/WC pair --
        # and with two chips left against fourteen gameweeks nothing is forced.
        chips_used=[(chips.Chip.TRIPLE_CAPTAIN, 5), (chips.Chip.BENCH_BOOST, 5)],
        # Three gameweeks old against wildcard_min_managed_gws of six: the
        # wildcard the comparison nominated is unplayable.
        squad_age_gws=3,
        chip_timing=timing,
        comparison=comparison,
    )

    assert rec.chip is chips.Chip.FREE_HIT, (
        "the nominated wildcard was refused on squad age, so the free hit -- "
        "which cleared its own margin by 14.19 -- must be the fallback"
    )
    assert "14.2" in rec.reason or "14.19" in rec.reason


def test_a_chip_that_failed_its_margin_is_still_blocked(half_boundary_at_19):
    """The fallthrough guard's original purpose must survive.

    The free hit is absent from `ranked` (it did not clear its margin), so it
    must not be played even though the LEGACY threshold would have fired on
    this scenario's enormous one-week gain. Approving on qualification rather
    than on winning must not degrade into approving everything.
    """
    none_opt = ChipOption(None, 274.30, _plan(), "no chip")
    wc_opt = ChipOption(chips.Chip.WILDCARD, 323.69, _plan(), "wildcard")
    comparison = ChipComparison(
        options=[none_opt, wc_opt], no_chip=none_opt, best=wc_opt, ranked=[wc_opt],
    )
    timing = dataclasses.replace(CHIP_TIMING, chip_comparison_enabled=True)
    players, squad, proj = _squad_scenario()

    rec = chips.recommend_chip(
        current_gw=5,
        current_squad_ids=squad,
        projections=proj,
        players=players,
        available_budget=200.0,
        free_transfers=1,
        # The wildcard is spent too, so the ranked winner is unavailable and
        # only the free hit is left -- and it must stay blocked.
        chips_used=[
            (chips.Chip.TRIPLE_CAPTAIN, 5),
            (chips.Chip.BENCH_BOOST, 5),
            (chips.Chip.WILDCARD, 5),
        ],
        squad_age_gws=99,
        chip_timing=timing,
        comparison=comparison,
    )

    assert rec.chip is None


def test_the_better_ranked_chip_is_attempted_first(half_boundary_at_19):
    """B3: rank decides which of the FH/WC pair gets first refusal.

    Same comparison as the regression test above, but the squad is now old
    enough for the wildcard -- so the higher-ranked wildcard must win, rather
    than the free hit taking it merely by sitting earlier in the static order.
    """
    none_opt = ChipOption(None, 274.30, _plan(), "no chip")
    fh_opt = ChipOption(chips.Chip.FREE_HIT, 288.49, _plan(), "free hit")
    wc_opt = ChipOption(chips.Chip.WILDCARD, 323.69, _plan(), "wildcard")
    comparison = ChipComparison(
        options=[none_opt, fh_opt, wc_opt], no_chip=none_opt,
        best=wc_opt, ranked=[wc_opt, fh_opt],
    )
    timing = dataclasses.replace(CHIP_TIMING, chip_comparison_enabled=True)
    players, squad, proj = _squad_scenario()

    rec = chips.recommend_chip(
        current_gw=5,
        current_squad_ids=squad,
        projections=proj,
        players=players,
        available_budget=200.0,
        free_transfers=1,
        chips_used=[(chips.Chip.TRIPLE_CAPTAIN, 5), (chips.Chip.BENCH_BOOST, 5)],
        squad_age_gws=99,
        chip_timing=timing,
        comparison=comparison,
    )

    assert rec.chip is chips.Chip.WILDCARD


def test_triple_captain_and_bench_boost_stay_outside_the_ranking(half_boundary_at_19):
    """TC and BB are orthogonal to transfers: you play them AND transfer.

    Listing them in `ranked` (which the comparison never does) must not make
    `_comparison_choice` approve them, or the ranked fallback has smuggled in
    the very coupling the comparison was written to avoid.
    """
    none_opt = ChipOption(None, 0.0, _plan(), "")
    tc_opt = ChipOption(chips.Chip.TRIPLE_CAPTAIN, 10_000.0, _plan(), "")
    bb_opt = ChipOption(chips.Chip.BENCH_BOOST, 9_000.0, _plan(), "")
    comparison = ChipComparison(
        options=[none_opt, tc_opt, bb_opt], no_chip=none_opt,
        best=tc_opt, ranked=[tc_opt, bb_opt],
    )
    timing = dataclasses.replace(CHIP_TIMING, chip_comparison_enabled=True)

    for chip in (chips.Chip.TRIPLE_CAPTAIN, chips.Chip.BENCH_BOOST):
        assert chips._comparison_choice(comparison, chip, timing) is None


def test_rank_qualifying_lists_every_qualifier_best_first():
    none_opt = ChipOption(None, 100.0, _plan(), "")
    fh_opt = ChipOption(chips.Chip.FREE_HIT, 115.0, _plan(), "")
    wc_opt = ChipOption(chips.Chip.WILDCARD, 130.0, _plan(), "")
    ranked = rank_qualifying(
        [none_opt, fh_opt, wc_opt],
        free_hit_margin=12.0, wildcard_margin=25.0,
        free_hit_chip=chips.Chip.FREE_HIT, wildcard_chip=chips.Chip.WILDCARD,
    )
    assert ranked == [wc_opt, fh_opt]


def test_rank_qualifying_drops_a_chip_that_missed_its_margin():
    none_opt = ChipOption(None, 100.0, _plan(), "")
    fh_opt = ChipOption(chips.Chip.FREE_HIT, 105.0, _plan(), "")   # +5 < 12
    wc_opt = ChipOption(chips.Chip.WILDCARD, 130.0, _plan(), "")   # +30 >= 25
    ranked = rank_qualifying(
        [none_opt, fh_opt, wc_opt],
        free_hit_margin=12.0, wildcard_margin=25.0,
        free_hit_chip=chips.Chip.FREE_HIT, wildcard_chip=chips.Chip.WILDCARD,
    )
    assert ranked == [wc_opt]


def test_rank_qualifying_is_empty_without_a_no_chip_baseline():
    """No baseline, no honest margin -- the same rule `pick_best` follows."""
    fh_opt = ChipOption(chips.Chip.FREE_HIT, 115.0, _plan(), "")
    assert rank_qualifying(
        [fh_opt],
        free_hit_margin=12.0, wildcard_margin=25.0,
        free_hit_chip=chips.Chip.FREE_HIT, wildcard_chip=chips.Chip.WILDCARD,
    ) == []


def test_best_is_the_head_of_ranked():
    """`best` and `ranked[0]` answer different questions but must agree."""
    from optimiser.chip_comparison import pick_best

    none_opt = ChipOption(None, 100.0, _plan(), "")
    fh_opt = ChipOption(chips.Chip.FREE_HIT, 115.0, _plan(), "")
    wc_opt = ChipOption(chips.Chip.WILDCARD, 130.0, _plan(), "")
    kwargs = dict(
        free_hit_margin=12.0, wildcard_margin=25.0,
        free_hit_chip=chips.Chip.FREE_HIT, wildcard_chip=chips.Chip.WILDCARD,
    )
    options = [none_opt, fh_opt, wc_opt]
    ranked = rank_qualifying(options, **kwargs)
    assert ranked
    assert pick_best(options, **kwargs) is ranked[0]


def test_compare_chip_options_populates_ranked_alongside_best():
    """The invariant on the real assembled object, not just on its parts.

    The three option builders are stubbed because what is under test is the
    assembly -- that `compare_chip_options` publishes the ranking it computed
    and that its head is the same object `best` points at. Actually solving
    three ILPs here would test the optimiser instead.
    """
    from optimiser import chip_comparison as cc

    none_opt = ChipOption(None, 100.0, _plan(), "")
    fh_opt = ChipOption(chips.Chip.FREE_HIT, 115.0, _plan(), "")
    wc_opt = ChipOption(chips.Chip.WILDCARD, 130.0, _plan(), "")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cc, "build_no_chip_option", lambda *a, **k: none_opt)
        mp.setattr(cc, "build_free_hit_option", lambda *a, **k: fh_opt)
        mp.setattr(cc, "build_wildcard_option", lambda *a, **k: wc_opt)
        comparison = cc.compare_chip_options(
            [1], pd.DataFrame({"gameweek": [5]}), pd.DataFrame(),
            free_transfers=1, current_gw=5, horizon=5,
            free_hit_chip=chips.Chip.FREE_HIT, wildcard_chip=chips.Chip.WILDCARD,
            free_hit_margin=12.0, wildcard_margin=25.0,
        )

    assert comparison.ranked == [wc_opt, fh_opt]
    assert comparison.best is comparison.ranked[0]


def test_the_wildcard_is_withheld_from_a_squad_too_young_to_play_it():
    """Part A: the eligibility set must respect `wildcard_min_managed_gws`.

    `chips_available_this_half` only knows about uses remaining. Offering the
    comparison a chip the engine will refuse is what let a discarded wildcard
    nomination strand a qualifying free hit in the first place.
    """
    from agent.decision_engine import _comparison_eligible_chips

    timing = dataclasses.replace(CHIP_TIMING, wildcard_min_managed_gws=6)
    eligible = _comparison_eligible_chips(
        chips_used=[], next_gw=3, season="2026-27", squad_age_gws=3,
        chip_timing=timing,
    )
    assert chips.Chip.WILDCARD not in eligible
    assert chips.Chip.FREE_HIT in eligible


def test_the_wildcard_is_offered_once_the_squad_is_old_enough():
    from agent.decision_engine import _comparison_eligible_chips

    timing = dataclasses.replace(CHIP_TIMING, wildcard_min_managed_gws=6)
    eligible = _comparison_eligible_chips(
        chips_used=[], next_gw=9, season="2026-27", squad_age_gws=6,
        chip_timing=timing,
    )
    assert eligible == {chips.Chip.FREE_HIT, chips.Chip.WILDCARD}


def test_a_spent_chip_is_still_excluded_regardless_of_squad_age():
    """The uses-remaining filter the age gate is layered on top of."""
    from agent.decision_engine import _comparison_eligible_chips

    timing = dataclasses.replace(CHIP_TIMING, wildcard_min_managed_gws=6)
    eligible = _comparison_eligible_chips(
        chips_used=[(chips.Chip.FREE_HIT, 4)], next_gw=9, season="2026-27",
        squad_age_gws=99, chip_timing=timing,
    )
    assert eligible == {chips.Chip.WILDCARD}


def test_chip_comparison_log_is_keyed_on_the_run_not_the_gameweek():
    """Part C: one row per option per RUN, re-runs kept side by side.

    `created_at` is in the key on purpose. Re-running a gameweek is normal and
    a changed verdict is the measurement, so an upsert on
    (season, gameweek, sim_manager_id, option) would delete exactly the data
    the table exists to hold. What the key does buy is that a single run cannot
    write the same option twice.
    """
    from data.models import ChipComparisonLog

    constraints = {
        c.name: {col.name for col in c.columns}
        for c in ChipComparisonLog.__table__.constraints
        if c.name is not None
    }
    assert constraints.get("uq_chip_comparison") == {
        "season", "gameweek", "sim_manager_id", "option", "created_at",
    }
