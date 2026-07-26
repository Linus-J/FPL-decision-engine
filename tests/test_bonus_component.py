"""P8 bonus component — reduced-BPS mapping + agreement measure."""

from __future__ import annotations

import pytest

from projection import bonus as B


def test_reduce_keeps_only_modelled_fields():
    ev = {"position": "MID", "minutes": 90, "goals": 1, "key_passes": 3,
          "big_chances_created": 2, "open_play_crosses": 4, "pass_completion_pct": 91}
    red = B.reduce_to_modelled(ev)
    assert red == {"position": "MID", "minutes": 90, "goals": 1, "key_passes": 3}
    assert "big_chances_created" not in red and "open_play_crosses" not in red


def test_sample_fixture_bonus_awards_321():
    events = {
        1: {"position": "FWD", "minutes": 90, "goals": 2},   # highest BPS
        2: {"position": "MID", "minutes": 90, "goals": 1, "assists": 1},
        3: {"position": "DEF", "minutes": 90, "clean_sheet": 1},
        4: {"position": "DEF", "minutes": 20},
    }
    bonus = B.sample_fixture_bonus(events)
    assert bonus[1] == 3
    assert sorted(bonus.values(), reverse=True)[:3] == [3, 2, 1]
    assert bonus[4] == 0


def test_reduced_full_agreement_perfect_when_only_modelled_events():
    # events using only modelled fields → reduced == full → perfect agreement
    events = {
        1: {"position": "FWD", "minutes": 90, "goals": 2},
        2: {"position": "MID", "minutes": 90, "assists": 1, "key_passes": 2},
        3: {"position": "DEF", "minutes": 90, "clean_sheet": 1},
    }
    m = B.reduced_full_agreement(events)
    assert m["slot_exact_rate"] == pytest.approx(1.0)
    assert m["recipient_jaccard"] == pytest.approx(1.0)


def test_reduced_drops_unmodelled_bps():
    # a player whose BPS is mostly from unmodelled events loses it under reduce
    full = {"position": "MID", "minutes": 90, "big_chances_created": 5,
            "open_play_crosses": 6, "key_passes": 1}
    assert B.player_bps(full, reduced=False) > B.player_bps(full, reduced=True)


def test_sample_fixture_bonus_dampens_gk_saves_for_ranking_only():
    # GK's saves count is dampened for the bonus RANKING (the calibration fix
    # for GK winning ~3x too much real bonus, per plan/phase-2-xpts-engine.md
    # P10): an undampened GK with 6 saves (raw bps 6 appearance + 6*2 saves =
    # 18) would trivially beat a bare-appearance DEF, but the dampened
    # (6 * 0.15 = 0.9 effective saves) version must not.
    events = {
        1: {"position": "GKP", "minutes": 90, "saves": 6},
        2: {"position": "DEF", "minutes": 90, "clean_sheet": 1},  # 6 + 12 = 18 real bps
    }
    raw_gk_bps = B.player_bps(events[1])
    assert raw_gk_bps > 15  # confirms the undampened GK really would dominate
    bonus = B.sample_fixture_bonus(events)
    assert bonus[2] == 3         # the DEF (real clean-sheet bps) wins, not the GK
    assert bonus[2] > bonus[1]   # dampened GK no longer outranks the DEF


def test_gk_bonus_save_scale_only_applies_to_gk_positions():
    # the same "saves" value produces a smaller GK bps-equivalent than an
    # (nonsensical but position-gate-testing) DEF with identical saves —
    # confirms the dampening is gated strictly on position, not blanket.
    gk_event = {"position": "GKP", "minutes": 90, "saves": 6}
    def_event = {"position": "DEF", "minutes": 90, "saves": 6}
    gk_dampened_bps = B.compute_player_bps({**gk_event, "saves": 6 * B.GK_BONUS_SAVE_SCALE})
    def_undampened_bps = B.compute_player_bps(def_event)
    assert gk_dampened_bps < def_undampened_bps
    # sample_fixture_bonus applied to each solo confirms the same asymmetry
    # indirectly: a GK alongside a competitor just above the dampened bps
    # loses, while the identical competitor loses to an undampened DEF
    # bps = 6 (appearance) + 9 (assist) = 15 -- between the dampened-GK bps
    # (7.8) and the undampened-DEF bps (18), so it separates the two cases
    competitor = {"position": "MID", "minutes": 90, "assists": 1}
    gk_result = B.sample_fixture_bonus({1: gk_event, 2: competitor})
    def_result = B.sample_fixture_bonus({1: def_event, 2: competitor})
    assert gk_result[1] < def_result[1]


def test_reduce_keeps_dribbles():
    # P10 finding: real FWD P(bonus>0) was 15.1% vs 8.6% modelled -- dribbles
    # (now sourced for free from WhoScored) were missing from MODELLED_BPS_FIELDS
    ev = {"position": "FWD", "minutes": 90, "dribbles": 4, "big_chances_created": 2}
    red = B.reduce_to_modelled(ev)
    assert red["dribbles"] == 4
    assert "big_chances_created" not in red  # still genuinely unavailable


def test_dribbles_contribute_to_bps():
    base = {"position": "FWD", "minutes": 90}
    with_dribbles = {**base, "dribbles": 5}
    assert B.player_bps(with_dribbles) > B.player_bps(base)
