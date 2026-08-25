"""GW2 gap: a season with exactly one played gameweek can neither train nor
serve the in-season minutes model, because every row is a first appearance.

Found live 2026-08-25 running the real GW2 cycle: run_agent died with
`IndexError: single positional indexer is out-of-bounds` in train(), and then
with `Found array with 0 sample(s)` at serve time once training was widened.
The pipeline guarded "no gameweeks played" (routes to the cold start) but not
"one gameweek played", which fell between the two paths.
"""

from __future__ import annotations

import pandas as pd

from projection.minutes_model import _build_features


def _stats(gameweeks: list[int], players: int = 3) -> pd.DataFrame:
    rows = []
    for pid in range(1, players + 1):
        for gw in gameweeks:
            rows.append({
                "player_id": pid, "season": "2026-27", "gameweek": gw,
                "minutes": 90, "total_points": 5, "goals_scored": 0, "assists": 0,
                "clean_sheets": 0, "goals_conceded": 0, "saves": 0,
                "yellow_cards": 0, "red_cards": 0, "bonus": 0, "bps": 10,
                "position": "MID", "was_home": 1, "opponent_team_id": 2,
                "status": "a", "chance_of_playing_next_round": 100,
            })
    return pd.DataFrame(rows)


def test_one_played_gameweek_yields_no_usable_rows():
    """The precise failure. Every row is a player's first appearance, so
    avg_minutes_5gw and season_avg_minutes are null and all rows are dropped."""
    assert len(_build_features(_stats([1]))) == 0


def test_two_played_gameweeks_yield_usable_rows():
    """From the second played gameweek every later row has a predecessor, so
    the in-season path becomes viable again and the fallback stops firing."""
    built = _build_features(_stats([1, 2]))

    assert len(built) > 0
    assert set(built["gameweek"]) == {2}, "only rows with a predecessor survive"
