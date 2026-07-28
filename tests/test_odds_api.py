"""Regression tests for two real bugs found in data/ingestors/odds_api.py
during the 2026-07-28 data-completeness audit (found by a parallel
ingestor-audit pass, not the earlier name-matcher work)."""

from __future__ import annotations

from data.ingestors.odds_api import _extract_h2h, _match_fixture


def test_extract_h2h_does_not_swap_when_away_team_sorts_first_alphabetically():
    # Real bug: the old version sorted the two non-Draw outcome names
    # ALPHABETICALLY and assigned the first to home -- "Arsenal" (away)
    # sorts before "Wolves" (home), which used to silently swap the probs.
    bookmakers = [
        {
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Wolves", "price": 1.5},  # home, should win more often
                        {"name": "Arsenal", "price": 6.0},  # away
                        {"name": "Draw", "price": 4.0},
                    ],
                }
            ]
        }
    ]
    home_win, draw, away_win = _extract_h2h(bookmakers, "Wolves", "Arsenal")
    assert home_win > away_win


def test_extract_h2h_missing_team_name_falls_back_to_default():
    bookmakers = [{"markets": [{"key": "h2h", "outcomes": [{"name": "Draw", "price": 4.0}]}]}]
    assert _extract_h2h(bookmakers, "Wolves", "Arsenal") == (0.33, 0.33, 0.33)


def test_match_fixture_requires_both_home_and_away_to_match():
    # Real bug: the old version matched on the home team alone, so a team
    # with two unfinished home fixtures in the window got both odds
    # snapshots attached to whichever fixture query order returned first.
    db_fixtures = [
        {"id": 1, "team_h_name": "Wolves", "team_a_name": "Arsenal", "kickoff_time": None},
        {"id": 2, "team_h_name": "Wolves", "team_a_name": "Chelsea", "kickoff_time": None},
    ]
    assert _match_fixture("Wolves", "Chelsea", "", db_fixtures) == 2
    assert _match_fixture("Wolves", "Arsenal", "", db_fixtures) == 1


def test_match_fixture_no_away_match_returns_none():
    db_fixtures = [
        {"id": 1, "team_h_name": "Wolves", "team_a_name": "Arsenal", "kickoff_time": None},
    ]
    assert _match_fixture("Wolves", "Everton", "", db_fixtures) is None


def test_match_fixture_breaks_genuine_tie_by_nearest_kickoff():
    db_fixtures = [
        {
            "id": 1,
            "team_h_name": "Wolves",
            "team_a_name": "Arsenal",
            "kickoff_time": "2026-08-01T12:00:00",
        },
        {
            "id": 2,
            "team_h_name": "Wolves",
            "team_a_name": "Arsenal",
            "kickoff_time": "2026-08-01T20:00:00",
        },
    ]
    result = _match_fixture("Wolves", "Arsenal", "2026-08-01T19:45:00Z", db_fixtures)
    assert result == 2
