"""A chip played IN gameweek N must not block gameweek N's own re-decision
(2026-09-02).

``chips_used_this_season`` is consulted at the TOP of a decision cycle, before
that cycle writes any of its own rows. ``agent/decision_engine.py`` writes the
``lineup`` row before the ``chip`` row within a run, so after a chip-playing run
the chip row is the newest row for that gameweek -- by design, and correctly, so
that the supersede rule keeps a chip the newest run chose.

The consequence is that re-running one gameweek reads the PREVIOUS run of that
same gameweek as settled history. Live on GW3: ``chips_used_this_season``
reported ``[(TRIPLE_CAPTAIN, 2), (FREE_HIT, 3)]`` while GW3 was the gameweek
about to be decided, so a re-run would refuse the Free Hit it had just chosen,
write a lineup row that supersedes the chip row, and the run after that would
offer the Free Hit again. The chip decision alternated with the parity of the
re-run count.

A chip recorded as used in the gameweek currently being decided is this same
decision's own earlier attempt, not spent history. A chip used in an EARLIER
gameweek still counts normally -- that is the second test here, and it is what
stops the fix making chips infinitely reusable.
"""

from __future__ import annotations

import json

import pandas as pd

from optimiser.chips import (
    Chip,
    _chip_uses_remaining,
    chips_available_this_half,
    chips_used_this_season,
    must_play_a_chip_now,
)


def _rows(*specs: tuple[int, str, str, str]) -> pd.DataFrame:
    """(gameweek, decision_type, details_json, created_at) -> decision_log frame."""
    return pd.DataFrame(
        [
            {"gameweek": gw, "decision_type": dt, "details": d, "created_at": ts}
            for gw, dt, d, ts in specs
        ]
    )


_FH = json.dumps({"chip": "freehit", "reason": "BGW free hit gain 14.2 xPts"})
_TC = json.dumps({"chip": "3xc", "reason": "TC captain xPts 7.7"})
_LINEUP = json.dumps({"squad_ids": [1, 2, 3]})


# The live GW3 decision_log, abridged: the run that played the Free Hit wrote
# its lineup row first and its chip row second, so the chip row is newest.
_LIVE_GW3 = _rows(
    (2, "lineup", _LINEUP, "2026-08-25 14:46:30.150594"),
    (2, "chip", _TC, "2026-08-25 14:46:30.152479"),
    (3, "lineup", _LINEUP, "2026-09-01 15:22:34.914622"),
    (3, "chip", _FH, "2026-09-01 15:22:34.930524"),
)


def test_a_chip_played_in_the_gameweek_being_decided_is_still_available():
    """Deciding GW3 again must not read GW3's own Free Hit as spent."""
    used = chips_used_this_season(_LIVE_GW3)
    # The full history is unchanged -- the supersede rule is correct and this
    # fix does not touch it.
    assert (Chip.FREE_HIT, 3) in used
    assert _chip_uses_remaining(Chip.FREE_HIT, used, current_gw=3, season=None) == 1
    assert Chip.FREE_HIT in chips_available_this_half(used, current_gw=3, season=None)


def test_a_chip_played_in_an_earlier_gameweek_is_still_spent():
    """The converse: GW2's Triple Captain stays spent when deciding GW3."""
    used = chips_used_this_season(_LIVE_GW3)
    assert _chip_uses_remaining(Chip.TRIPLE_CAPTAIN, used, current_gw=3, season=None) == 0
    assert Chip.TRIPLE_CAPTAIN not in chips_available_this_half(used, current_gw=3, season=None)


def test_the_same_chip_played_earlier_in_the_half_is_still_spent():
    """A Free Hit played in GW2 blocks GW3 even though GW3 is being decided --
    only the CURRENT gameweek's own row is discounted, not the chip's history."""
    used = chips_used_this_season(
        _rows(
            (2, "lineup", _LINEUP, "2026-08-25 14:46:30.150594"),
            (2, "chip", _FH, "2026-08-25 14:46:30.152479"),
        )
    )
    assert _chip_uses_remaining(Chip.FREE_HIT, used, current_gw=3, season=None) == 0


def test_excluding_the_current_gameweek_returns_the_chip_to_the_expiry_arithmetic():
    """``must_play_a_chip_now`` counts chips against remaining slots. A chip
    this gameweek's own earlier run 'played' is back in hand for the re-run, so
    it must be counted as available rather than quietly written off."""
    used = chips_used_this_season(
        _rows(
            (17, "lineup", _LINEUP, "2026-12-01 10:00:00.000000"),
            (17, "chip", _FH, "2026-12-01 10:00:00.000100"),
        )
    )
    # GW17 with the boundary at 19: three slots left (17, 18, 19) and, once
    # GW17's own row is discounted, four chips. Skipping today bins one.
    assert len(chips_available_this_half(used, current_gw=17, season=None)) == 4
    assert must_play_a_chip_now(used, current_gw=17, season=None) is True


def test_squad_age_ignores_a_wildcard_played_in_the_gameweek_being_decided():
    """The same defect one layer down, found while auditing the consumers.

    ``agent.decision_engine._squad_age_gws`` counts from the last wildcard,
    because a wildcard is exactly "rebuilt from scratch". Read while re-deciding
    the gameweek that wildcard was played in, it reports age 0 -- so
    ``wildcard_min_managed_gws`` refuses the wildcard on the re-run, and the run
    after that offers it again. That is item 1's alternation reproduced through
    the age gate rather than the uses count, and fixing only
    ``_chip_uses_remaining`` would leave the wildcard case still flipping.

    On the re-run the rebuild has NOT happened -- that is the decision being
    remade -- so the age is still measured from the previous rebuild.
    """
    from agent.decision_engine import _squad_age_gws

    log = _rows(
        (1, "lineup", _LINEUP, "2026-08-01 10:00:00.000000"),
        (3, "lineup", _LINEUP, "2026-09-01 15:22:34.914622"),
    )
    used = [(Chip.WILDCARD, 3)]
    # Squad first fielded in GW1, so deciding GW3 sees a two-gameweek-old squad.
    assert _squad_age_gws(log, used, next_gw=3) == 2
    # A wildcard played in an EARLIER gameweek still resets the clock.
    assert _squad_age_gws(log, [(Chip.WILDCARD, 2)], next_gw=3) == 1
