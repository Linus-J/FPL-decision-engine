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
