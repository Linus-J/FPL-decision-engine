"""T5a gate — deterministic 26/27 BPS simulator.

Locks the BPS arithmetic, the 26/27-vs-old-rules deltas (the sanity harness
the plan calls for, run synthetically until FBref events land), FPL tie-aware
bonus awarding, and DefCon/BPS independence.
"""

from __future__ import annotations

from dataclasses import replace

from config.strategy import BPS_WEIGHTS, DEFCON
from projection.bps_sim import (
    award_bonus,
    compute_defcon_points,
    compute_fixture_bonus,
    compute_player_bps,
)

# 25/26 (old) weights: the four numeric BPS changes for 26/27.
OLD_WEIGHTS = replace(
    BPS_WEIGHTS,
    being_tackled=-1,   # 26/27: removed (0)
    cbi_per_point=2,    # 26/27: 1 per 3 (was per 2)
    penalty_saved=8,    # 26/27: 7
    big_chance_saved=0,  # 26/27: +1 (new)
)


def test_locked_bps_arithmetic():
    ev = {
        "position": "MID", "minutes": 90, "goals": 1, "assists": 1,
        "key_passes": 2, "clean_sheet": 1,               # MID CS = no BPS
        "clearances": 2, "blocks": 1, "interceptions": 2,  # cbi=5 → 1
        "recoveries": 6,                                   # → 2
        "tackles": 3,                                      # → 6
        "passes": 40, "pass_completion_pct": 85,           # 80–89 → +4
        "yellow_cards": 1,                                 # −3
    }
    # 6 + 18 + 9 + 2 + 1 + 2 + 6 + 4 − 3 = 45
    assert compute_player_bps(ev) == 45


def test_2627_deltas_vs_old_rules():
    ev = {
        "position": "GK", "minutes": 90,
        "being_tackled": 2, "clearances": 4,  # cbi=4
        "penalties_saved": 1, "big_chances_saved": 1,
    }
    # new: play60(6) + tackled 0 + cbi 4//3=1 + pen 7 + bcs 1 = 15
    assert compute_player_bps(ev, BPS_WEIGHTS) == 15
    # old: play60(6) + tackled 2*-1=-2 + cbi 4//2=2 + pen 8 + bcs 0 = 14
    assert compute_player_bps(ev, OLD_WEIGHTS) == 14


def test_appearance_bands():
    assert compute_player_bps({"position": "FWD", "minutes": 0}) == 0
    assert compute_player_bps({"position": "FWD", "minutes": 45}) == BPS_WEIGHTS.play_1_to_60
    assert compute_player_bps({"position": "FWD", "minutes": 60}) == BPS_WEIGHTS.play_over_60


def test_clean_sheet_only_gk_def_60plus():
    base = {"minutes": 90, "clean_sheet": 1}
    assert compute_player_bps({**base, "position": "DEF"}) == 6 + 12
    assert compute_player_bps({**base, "position": "MID"}) == 6          # no CS BPS
    assert compute_player_bps({**base, "position": "GK", "minutes": 45}) == 3  # <60, no CS


def test_award_bonus_no_ties():
    assert award_bonus({1: 30, 2: 25, 3: 20, 4: 10}) == {1: 3, 2: 2, 3: 1, 4: 0}


def test_award_bonus_tie_for_first():
    # 2 tie for 1st → both 3, next gets 1 (2 skipped)
    assert award_bonus({1: 30, 2: 30, 3: 20}) == {1: 3, 2: 3, 3: 1}


def test_award_bonus_tie_for_second():
    # 2 tie for 2nd → top 3, both 2 (no 1 awarded)
    assert award_bonus({1: 30, 2: 25, 3: 25}) == {1: 3, 2: 2, 3: 2}


def test_award_bonus_triple_tie_for_first():
    assert award_bonus({1: 30, 2: 30, 3: 30, 4: 5}) == {1: 3, 2: 3, 3: 3, 4: 0}


def test_compute_fixture_bonus_end_to_end():
    events = {
        1: {"position": "FWD", "minutes": 90, "goals": 2},   # 6 + 48 = 54
        2: {"position": "MID", "minutes": 90, "goals": 1, "assists": 1},  # 6+18+9=33
        3: {"position": "DEF", "minutes": 90, "clean_sheet": 1},  # 6+12=18
    }
    assert compute_fixture_bonus(events) == {1: 3, 2: 2, 3: 1}


def test_defcon_thresholds():
    # DEF: CBIT (clear+block+intercept+tackle) ≥ 10
    assert compute_defcon_points(
        {"position": "DEF", "clearances": 5, "blocks": 2, "interceptions": 2, "tackles": 1}
    ) == DEFCON.points  # 10
    assert compute_defcon_points(
        {"position": "DEF", "clearances": 4, "blocks": 2, "interceptions": 2, "tackles": 1}
    ) == 0  # 9
    # MID/FWD: CBIRT (adds recoveries) ≥ 12
    assert compute_defcon_points(
        {"position": "MID", "clearances": 3, "interceptions": 3, "tackles": 2, "recoveries": 4}
    ) == DEFCON.points  # 12
    # GK never earns DefCon
    assert compute_defcon_points({"position": "GK", "clearances": 20}) == 0


def test_defcon_and_bps_are_independent():
    """A DefCon-earning player's BPS is the BPS formula alone — the +2 DefCon
    is never folded into BPS (no shared term)."""
    ev = {"position": "DEF", "minutes": 90, "clearances": 6, "blocks": 3,
          "interceptions": 2, "tackles": 1}  # CBIT=12 → DefCon +2
    # BPS: play60(6) + cbi(6+3+2=11 //3 = 3) + tackles(1*2=2) = 11
    assert compute_player_bps(ev) == 6 + 3 + 2
    assert compute_defcon_points(ev) == DEFCON.points
    # the two are computed by separate functions; neither reads the other
