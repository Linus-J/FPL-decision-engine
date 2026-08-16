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


# --- P3.11 (2026-08-16): make the odds window visible --------------------


def _coverage_db(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import data.ingestors.odds_api as odds_module
    from data.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'cov.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(odds_module, "get_session", lambda: Local())
    return odds_module, Local


def test_odds_coverage_counts_fixtures_not_fetches(tmp_path, monkeypatch):
    """fixture_odds is append-only -- one row per fetch. A plain COUNT(*)
    over the join counts fetches, which reported 112 "fixtures" for a
    ten-match gameweek against the live database."""
    from datetime import datetime

    from data.models import Fixture, FixtureOdds

    odds_module, Local = _coverage_db(tmp_path, monkeypatch)
    s = Local()
    s.add_all([
        Fixture(id=1, fpl_id=1, season="2026-27",
                gameweek=1, team_h_id=1, team_a_id=2, finished=False),
        Fixture(id=2, fpl_id=2, season="2026-27",
                gameweek=1, team_h_id=3, team_a_id=4, finished=False),
    ])
    # fixture 1 fetched three times, fixture 2 never
    for hour in (10, 11, 12):
        s.add(FixtureOdds(
            fixture_id=1, home_win_prob=0.5, draw_prob=0.3, away_win_prob=0.2,
            fetched_at=datetime(2026, 8, 20, hour),
        ))
    s.commit()
    s.close()

    coverage = odds_module.odds_coverage_by_gameweek("2026-27", horizon=3)
    assert coverage[1] == (1, 2)


def test_odds_coverage_is_limited_to_the_horizon(tmp_path, monkeypatch):
    from data.models import Fixture

    odds_module, Local = _coverage_db(tmp_path, monkeypatch)
    s = Local()
    for gw in range(1, 6):
        s.add(Fixture(id=gw, fpl_id=gw, season="2026-27", gameweek=gw,
                      team_h_id=1, team_a_id=2, finished=False))
    s.commit()
    s.close()

    assert sorted(odds_module.odds_coverage_by_gameweek("2026-27", horizon=2)) == [1, 2]


def test_odds_coverage_warns_on_an_uncovered_gameweek(tmp_path, monkeypatch, caplog):
    """An uncovered gameweek projects on a flat league-average scoreline,
    erasing the fixture-difficulty signal the horizon exists to exploit --
    that has to be visible, not silent."""
    from data.models import Fixture

    odds_module, Local = _coverage_db(tmp_path, monkeypatch)
    s = Local()
    s.add(Fixture(id=1, fpl_id=1, season="2026-27", gameweek=1,
                  team_h_id=1, team_a_id=2, finished=False))
    s.commit()
    s.close()

    with caplog.at_level("WARNING"):
        odds_module.log_odds_coverage("2026-27", horizon=1)
    assert "Odds coverage GW1: 0/1" in caplog.text


def test_odds_coverage_excludes_finished_fixtures(tmp_path, monkeypatch):
    from data.models import Fixture

    odds_module, Local = _coverage_db(tmp_path, monkeypatch)
    s = Local()
    s.add_all([
        Fixture(id=1, fpl_id=1, season="2026-27",
                gameweek=1, team_h_id=1, team_a_id=2, finished=True),
        Fixture(id=2, fpl_id=2, season="2026-27",
                gameweek=2, team_h_id=3, team_a_id=4, finished=False),
    ])
    s.commit()
    s.close()

    assert list(odds_module.odds_coverage_by_gameweek("2026-27", horizon=5)) == [2]
