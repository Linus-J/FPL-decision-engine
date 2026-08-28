"""A superseded chip recommendation must not consume the chip (2026-08-28).

``chips_used_this_season`` counts every ``decision_type='chip'`` row as a chip
PLAYED. But this engine has no submission path -- a chip row is the engine's
record of "I decided to play this", written by the run that decided it. Re-run
the same gameweek and decide differently, and the earlier run's row still
stands, so the chip stays consumed for the rest of the half.

Live on the GW2 deadline: the 2026-08-25 run recommended Triple Captain
("TC captain xPts 7.7"). Later runs of the SAME gameweek did not. The engine
still read the chip as used, so ``_chip_uses_remaining`` returned 0 and
``_try_tc`` bailed before evaluating anything -- while the gate itself passed
on the numbers (7.808 against a 7.500 threshold). The chip was reported as
"No chip threshold met" when the truth was "already spent, on a decision that
was superseded and never played".

Same class as the free-transfer re-run bug fixed the same day: an earlier run
of a gameweek leaves state that a later run of that gameweek reads as settled
history. The P1.8 de-duplication handles N rows for one (chip, gameweek); it
cannot see a LATER run that chose no chip, because choosing no chip writes no
row.

The rule, matching every other consumer ('the latest run of a gameweek wins'):
a chip row counts only if no lineup row for that SAME gameweek is newer. A
later lineup means a later run superseded that chip decision. Within one run
the chip row is written after its own lineup row, so a chip from the newest
run is correctly kept.
"""

from __future__ import annotations

import json

import pandas as pd

from optimiser.chips import Chip, chips_used_this_season


def _rows(*specs: tuple[int, str, str, str]) -> pd.DataFrame:
    """(gameweek, decision_type, details_json, created_at) -> decision_log frame."""
    return pd.DataFrame(
        [
            {"gameweek": gw, "decision_type": dt, "details": d, "created_at": ts}
            for gw, dt, d, ts in specs
        ]
    )


_CHIP = json.dumps({"chip": "3xc", "reason": "TC captain xPts 7.7"})
_LINEUP = json.dumps({"squad_ids": [1, 2, 3]})


def test_a_chip_superseded_by_a_later_run_of_the_same_gameweek_is_not_used():
    """The live GW2 case."""
    df = _rows(
        (2, "lineup", _LINEUP, "2026-08-25 14:46:30.150594"),
        (2, "chip", _CHIP, "2026-08-25 14:46:30.152479"),
        (2, "lineup", _LINEUP, "2026-08-28 13:07:06.134844"),
    )
    assert chips_used_this_season(df) == []


def test_a_chip_from_the_latest_run_is_still_counted():
    """The newest run DID choose the chip -- its own lineup is older than it."""
    df = _rows(
        (2, "lineup", _LINEUP, "2026-08-25 14:46:30.150594"),
        (2, "lineup", _LINEUP, "2026-08-28 13:07:06.134844"),
        (2, "chip", _CHIP, "2026-08-28 13:07:06.140000"),
    )
    assert chips_used_this_season(df) == [(Chip.TRIPLE_CAPTAIN, 2)]


def test_a_later_gameweeks_lineup_does_not_supersede_an_earlier_chip():
    """A chip played in GW2 stays played once GW3 is decided."""
    df = _rows(
        (2, "lineup", _LINEUP, "2026-08-25 14:46:30.150594"),
        (2, "chip", _CHIP, "2026-08-25 14:46:30.152479"),
        (3, "lineup", _LINEUP, "2026-09-01 09:00:00.000000"),
    )
    assert chips_used_this_season(df) == [(Chip.TRIPLE_CAPTAIN, 2)]


def test_chips_still_count_when_no_lineup_rows_exist():
    """The backtest path writes chip rows without this module's lineup rows;
    nothing later can exist, so nothing is superseded."""
    df = _rows((2, "chip", _CHIP, "2026-08-25 14:46:30.152479"))
    assert chips_used_this_season(df) == [(Chip.TRIPLE_CAPTAIN, 2)]


def test_the_p1_8_duplicate_dedup_still_holds():
    """Re-running a gameweek that KEPT the chip must still consume it once."""
    df = _rows(
        (2, "lineup", _LINEUP, "2026-08-28 13:00:00.000000"),
        (2, "chip", _CHIP, "2026-08-28 13:00:00.100000"),
        (2, "lineup", _LINEUP, "2026-08-28 14:00:00.000000"),
        (2, "chip", _CHIP, "2026-08-28 14:00:00.100000"),
    )
    assert chips_used_this_season(df) == [(Chip.TRIPLE_CAPTAIN, 2)]


def test_an_empty_log_is_unchanged():
    assert chips_used_this_season(pd.DataFrame()) == []
