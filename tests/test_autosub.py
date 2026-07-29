"""Real bug found 2026-07-30 (user's own report review): scripts/backtest.py
never modelled FPL's auto-substitution rule, so a starting pick that ended up
not playing simply scored 0 for every benchmark (v2, v1, frozen template) --
understating what a real manager holding that squad would actually score.
Covers _apply_autosubs (GK swap, outfield priority, formation-legality
gating, a blank bench sub never being eligible) and _score_squad's captain
armband transfer + backward-compatible no-autosub call signature.
"""

from __future__ import annotations

from scripts.backtest import _apply_autosubs, _score_squad

# A minimal 15-man squad shape reused across tests. Starting XI (11):
# 1 GKP, 5 DEF (at STARTING_MAX), 3 MID, 2 FWD. Bench (4): GKP, DEF, MID, FWD.
POSITIONS = {
    1: "GKP", 2: "GKP",
    3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 17: "DEF", 7: "DEF",
    8: "MID", 9: "MID", 10: "MID", 12: "MID",
    13: "FWD", 14: "FWD", 15: "FWD",
}
SQUAD_IDS = [1, 2, 3, 4, 5, 6, 17, 7, 8, 9, 10, 12, 13, 14, 15]
STARTING_IDS = [1, 3, 4, 5, 6, 17, 8, 9, 10, 13, 14]
BENCH_ORDER = {2: 0, 7: 1, 12: 2, 15: 3}


def _played(minutes: dict[int, int] | None = None) -> dict[int, int]:
    """Every squad member defaults to 90 minutes; override via ``minutes``."""
    m = dict.fromkeys(SQUAD_IDS, 90)
    m.update(minutes or {})
    return m


def test_no_blanks_leaves_starting_xi_unchanged():
    minutes = _played()
    effective = _apply_autosubs(SQUAD_IDS, STARTING_IDS, POSITIONS, BENCH_ORDER, minutes)
    assert effective == STARTING_IDS


def test_gk_swap_is_unconditional_on_formation():
    minutes = _played(minutes={1: 0, 2: 90})
    effective = _apply_autosubs(SQUAD_IDS, STARTING_IDS, POSITIONS, BENCH_ORDER, minutes)
    assert 2 in effective and 1 not in effective


def test_gk_blank_with_bench_gk_also_blank_stays_unsubbed():
    minutes = _played(minutes={1: 0, 2: 0})
    effective = _apply_autosubs(SQUAD_IDS, STARTING_IDS, POSITIONS, BENCH_ORDER, minutes)
    assert effective == STARTING_IDS


def test_outfield_blank_swapped_for_same_position_bench_sub():
    # DEF 3 blanks; bench DEF (7, priority 1) played -- same-position swap.
    minutes = _played(minutes={3: 0})
    effective = _apply_autosubs(SQUAD_IDS, STARTING_IDS, POSITIONS, BENCH_ORDER, minutes)
    assert 7 in effective and 3 not in effective


def test_bench_player_with_zero_minutes_is_never_subbed_on():
    # DEF 3 blanks; every outfield bench player ALSO has 0 minutes -- nobody
    # is eligible to cover the blank, regardless of formation legality.
    minutes = _played(minutes={3: 0, 7: 0, 12: 0, 15: 0})
    effective = _apply_autosubs(SQUAD_IDS, STARTING_IDS, POSITIONS, BENCH_ORDER, minutes)
    assert effective == STARTING_IDS


def test_formation_legality_skips_to_next_priority_sub():
    # MID 8 blanks. Bench DEF (7, priority 1) played, but DEF is already at
    # STARTING_MAX (5) -- swapping DEF-for-MID would push DEF to 6, illegal,
    # so sub 7 must be skipped in favour of bench MID (12, priority 2).
    minutes = _played(minutes={8: 0, 7: 90, 12: 90})
    effective = _apply_autosubs(SQUAD_IDS, STARTING_IDS, POSITIONS, BENCH_ORDER, minutes)
    assert 12 in effective and 8 not in effective
    assert 7 not in effective  # correctly rejected, not just skipped by luck


def test_priority_order_uses_lowest_bench_order_first():
    # Two blanks (DEF 3, MID 8); both eligible bench subs played. Each sub
    # should fill the position it actually matches, regardless of processing
    # order, since same-position swaps are tried first for every sub.
    minutes = _played(minutes={3: 0, 8: 0})
    effective = _apply_autosubs(SQUAD_IDS, STARTING_IDS, POSITIONS, BENCH_ORDER, minutes)
    assert 7 in effective and 3 not in effective
    assert 12 in effective and 8 not in effective


def test_score_squad_transfers_armband_to_vice_when_captain_blanks():
    actual = {pid: 5 for pid in STARTING_IDS}
    actual[13] = 0  # captain blanked -- 0 minutes means 0 points too
    actual[9] = 10  # vice-captain's real points
    minutes = _played(minutes={13: 0})  # captain blanks
    total = _score_squad(
        SQUAD_IDS, STARTING_IDS, captain_id=13, actual_points=actual,
        vice_captain_id=9, minutes=minutes, positions=POSITIONS, bench_order=BENCH_ORDER,
    )
    # captain (13) blanked -> 0 pts regardless of multiplier; vice (9) doubled instead.
    expected = sum(actual[pid] for pid in STARTING_IDS if pid not in (13, 9)) + 0 + 10 * 2
    assert total == expected


def test_score_squad_no_multiplier_when_captain_and_vice_both_blank():
    actual = {pid: 5 for pid in STARTING_IDS}
    actual[13] = 0  # captain blanked
    actual[9] = 0  # vice-captain also blanked
    minutes = _played(minutes={13: 0, 9: 0})
    total = _score_squad(
        SQUAD_IDS, STARTING_IDS, captain_id=13, actual_points=actual,
        vice_captain_id=9, minutes=minutes, positions=POSITIONS, bench_order=BENCH_ORDER,
    )
    expected = sum(actual[pid] for pid in STARTING_IDS if pid not in (13, 9)) + 0 + 0
    assert total == expected


def test_score_squad_without_minutes_args_keeps_old_no_autosub_behaviour():
    # Backward compatibility: omitting minutes/positions/bench_order must
    # reproduce the pre-fix behaviour exactly, for call sites not yet updated.
    actual = {pid: 5 for pid in STARTING_IDS}
    actual[3] = 0  # "blanked" but with no minutes info, nobody can react to it
    total = _score_squad(SQUAD_IDS, STARTING_IDS, captain_id=13, actual_points=actual)
    expected = sum(actual[pid] for pid in STARTING_IDS if pid != 13) + actual[13] * 2
    assert total == expected


def test_score_squad_bench_boost_total_is_invariant_to_the_autosub_swap():
    # Under Bench Boost every one of the 15 counts once either way, so an
    # autosub swap must not change the final total -- it only relabels which
    # of the two loops (starting vs "bench") a given player's points fall into.
    actual = dict.fromkeys(SQUAD_IDS, 4)
    actual[7] = 9  # the sub who comes on
    minutes = _played(minutes={3: 0})  # triggers a DEF 3 -> DEF 7 swap
    total = _score_squad(
        SQUAD_IDS, STARTING_IDS, captain_id=13, actual_points=actual,
        bench_boost=True, vice_captain_id=9,
        minutes=minutes, positions=POSITIONS, bench_order=BENCH_ORDER,
    )
    expected = sum(actual.values()) + actual[13]  # every player once + captain's extra
    assert total == expected
