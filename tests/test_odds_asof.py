"""T6 gate — historical odds backfill helpers + leakage-free odds reads.

Pure parsing/mapping helpers are tested directly; the as-of reads are tested on
a temp DB. Covers finding L4 (no future/current odds leaking onto history) and
C2 (closing odds stamped at the deadline, not kickoff, so the `< deadline`
filter keeps them).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import (
    Base,
    Fixture,
    FixtureOdds,
    Gameweek,
    HistoricalFixtureOdds,
    PlayerGameweekStats,
)
from projection import features
from scripts import backfill_odds as bo


# --- pure helpers -----------------------------------------------------------
def test_implied_and_normalise():
    assert bo.implied_prob(2.0) == 0.5
    assert bo.implied_prob(0) == 0.0
    assert bo.normalise_1x2(0.5, 0.25, 0.25) == (0.5, 0.25, 0.25)
    assert bo.normalise_1x2(0, 0, 0) == (0.0, 0.0, 0.0)


def test_cs_probs_from_1x2():
    # mirrors the live odds_api heuristic exactly, for scale-consistency:
    # home_cs = draw + away_win*0.3; away_cs = draw + home_win*0.3
    home_cs, away_cs = bo.cs_probs_from_1x2(0.5, 0.25, 0.25)
    assert home_cs == 0.325
    assert away_cs == 0.40


def test_over25_prob():
    assert bo.over25_prob(1.9, 1.9) == 0.5
    assert bo.over25_prob(0, 2.0) == 0.0


def test_resolve_fd_team_alias_and_fuzzy():
    name_to_id = {"Man Utd": 1, "Arsenal": 2, "Spurs": 3}
    assert bo.resolve_fd_team("Man United", name_to_id) == 1  # alias
    assert bo.resolve_fd_team("Tottenham", name_to_id) == 3   # alias
    assert bo.resolve_fd_team("Arsenal", name_to_id) == 2     # exact
    assert bo.resolve_fd_team("Nonexistent FC", name_to_id) is None


def test_parse_kickoff_formats():
    assert bo.parse_kickoff("10/08/2024", "15:00") == datetime(2024, 8, 10, 15, 0)
    assert bo.parse_kickoff("10/08/24", None) == datetime(2024, 8, 10, 15, 0)
    assert bo.parse_kickoff("", "15:00") is None


def test_assign_gameweek():
    deadlines = {1: datetime(2024, 8, 9, 11, 30), 2: datetime(2024, 8, 16, 11, 30)}
    assert bo.assign_gameweek(datetime(2024, 8, 10, 15, 0), deadlines) == 1
    assert bo.assign_gameweek(datetime(2024, 8, 17, 15, 0), deadlines) == 2
    assert bo.assign_gameweek(datetime(2024, 8, 1, 15, 0), deadlines) is None


def test_build_odds_rows_end_to_end():
    df = pd.DataFrame([{
        "HomeTeam": "Arsenal", "AwayTeam": "Man United",
        "Date": "10/08/2024", "Time": "15:00",
        "PSH": 2.0, "PSD": 3.5, "PSA": 4.0,
        "P>2.5": 1.9, "P<2.5": 1.9,
    }])
    name_to_id = {"Arsenal": 2, "Man Utd": 1}
    deadlines = {1: datetime(2024, 8, 9, 11, 30)}
    rows, skipped = bo.build_odds_rows(df, "2024-25", name_to_id, deadlines)
    assert skipped == 0
    assert len(rows) == 1
    row = rows[0]
    assert (row["home_team_id"], row["away_team_id"]) == (2, 1)
    assert row["gameweek"] == 1
    assert row["over25_prob"] == 0.5
    # stamped strictly before the deadline (C2)
    assert row["fetched_at"] < deadlines[1]


def test_build_odds_rows_skips_unmatched():
    df = pd.DataFrame([{
        "HomeTeam": "Unknown", "AwayTeam": "Arsenal",
        "Date": "10/08/2024", "Time": "15:00",
        "PSH": 2.0, "PSD": 3.5, "PSA": 4.0,
    }])
    rows, skipped = bo.build_odds_rows(df, "2024-25", {"Arsenal": 2}, {1: datetime(2024, 8, 9)})
    assert rows == []
    assert skipped == 1


# --- DB-backed as-of reads --------------------------------------------------
@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odds.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(features, "get_session", lambda: Local())
    return Local


def _seed_historical(Local, odds_fetched_at):
    s = Local()
    try:
        s.add(Gameweek(id=1, season="2024-25", name="GW1",
                       deadline_time=datetime(2024, 8, 10, 11, 30)))
        # home player (team 2), away player (team 1)
        s.add(PlayerGameweekStats(player_id=100, gameweek=1, season="2024-25",
                                  team_id_season=2, opponent_team_id=1, was_home=True))
        s.add(PlayerGameweekStats(player_id=200, gameweek=1, season="2024-25",
                                  team_id_season=1, opponent_team_id=2, was_home=False))
        s.add(HistoricalFixtureOdds(season="2024-25", gameweek=1,
                                    home_team_id=2, away_team_id=1,
                                    home_cs_prob=0.35, away_cs_prob=0.20,
                                    btts_prob=0.55, fetched_at=odds_fetched_at))
        s.commit()
    finally:
        s.close()


def test_historical_odds_read_home_and_away(session):
    _seed_historical(session, datetime(2024, 8, 10, 11, 29))  # before deadline
    df = features.load_fixture_odds("2024-25").set_index("player_id")
    # home player: my_cs = home_cs, opp_cs = away_cs
    assert df.loc[100, "my_cs_prob"] == 0.35
    assert df.loc[100, "opp_cs_prob"] == 0.20
    # away player: my_cs = away_cs, opp_cs = home_cs
    assert df.loc[200, "my_cs_prob"] == 0.20
    assert df.loc[200, "opp_cs_prob"] == 0.35
    assert df.loc[100, "btts_prob"] == 0.55


def test_historical_odds_after_deadline_excluded_C2(session):
    # odds stamped AFTER the deadline (the kickoff−ε mistake) must not be read
    _seed_historical(session, datetime(2024, 8, 10, 12, 0))  # after 11:30 deadline
    df = features.load_fixture_odds("2024-25").set_index("player_id")
    assert df.loc[100, "my_cs_prob"] == 0.2   # defaulted, not 0.35
    assert df.loc[100, "btts_prob"] == 0.5


def test_live_odds_asof_picks_latest_before_deadline(session):
    Local = session
    s = Local()
    try:
        s.add(Gameweek(id=1, season="2025-26", name="GW1",
                       deadline_time=datetime(2025, 8, 15, 11, 30)))
        s.add(Fixture(id=10, fpl_id=1, season="2025-26", gameweek=1,
                      team_h_id=2, team_a_id=1))
        # append-only: three snapshots for the SAME fixture
        s.add(FixtureOdds(fixture_id=10, home_cs_prob=0.30,
                          fetched_at=datetime(2025, 8, 14, 10, 0)))   # earlier, before
        s.add(FixtureOdds(fixture_id=10, home_cs_prob=0.40,
                          fetched_at=datetime(2025, 8, 15, 10, 0)))   # latest before deadline
        s.add(FixtureOdds(fixture_id=10, home_cs_prob=0.99,
                          fetched_at=datetime(2025, 8, 15, 12, 0)))   # after deadline → excluded
        s.commit()
    finally:
        s.close()
    df = features.load_live_odds_asof("2025-26", 1)
    assert len(df) == 1
    assert df.iloc[0]["home_cs_prob"] == 0.40
