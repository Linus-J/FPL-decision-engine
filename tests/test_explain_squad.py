"""The decision explainer must surface the two things the objective cannot see:
clubs at the selection cap, and a goalkeeper sharing a clean sheet with his own
centre-back. Both are invisible to an objective that sums per-player variances
as if they were independent, which is why they are worth printing.
"""

import pandas as pd

from config.strategy import SQUAD
from scripts.explain_squad import _exposure


def _squad(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"web_name": n, "team": t, "position": p} for n, t, p in rows]
    )


def test_flags_a_club_at_the_selection_cap():
    rows = [(f"p{i}", "Arsenal", "DEF") for i in range(SQUAD.max_players_per_club)]
    rows.append(("keeper", "Everton", "GKP"))
    text = "\n".join(_exposure(_squad(rows)))

    assert "at the cap" in text
    assert "corner solution" in text


def test_does_not_cry_wolf_below_the_cap():
    rows = [(f"p{i}", "Arsenal", "MID") for i in range(SQUAD.max_players_per_club - 1)]
    rows.append(("keeper", "Everton", "GKP"))
    text = "\n".join(_exposure(_squad(rows)))

    assert "at the cap" not in text
    assert "corner solution" not in text


def test_reports_a_keeper_and_defender_sharing_one_clean_sheet():
    """The tightest correlation in the game: a single clean-sheet event paid to
    two players, which the summed-variance objective treats as two independent
    bets."""
    text = "\n".join(_exposure(_squad([
        ("Raya", "Arsenal", "GKP"),
        ("Gabriel", "Arsenal", "DEF"),
        ("Virgil", "Liverpool", "DEF"),
    ])))

    assert "Clean-sheet exposure" in text
    assert "Raya" in text and "Gabriel" in text
    # Liverpool has a defender but no keeper here, so it is not doubled up.
    assert "Virgil" not in text


def test_no_clean_sheet_section_when_nothing_is_doubled_up():
    text = "\n".join(_exposure(_squad([
        ("Raya", "Arsenal", "GKP"),
        ("Virgil", "Liverpool", "DEF"),
    ])))

    assert "Clean-sheet exposure" not in text
