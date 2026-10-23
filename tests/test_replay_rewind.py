"""The rewind is the foundation of every replay. Each assertion here is a way a
past gameweek could quietly see the future: a price, a strength, a duty, a
decision or a line stamped after the deadline surviving the rewind."""

import sqlite3
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import (
    Base,
    DecisionLog,
    Fixture,
    FixtureOdds,
    Gameweek,
    Player,
    PlayerGameweekStats,
    PlayerSetPieceRole,
    PlayerSetPieceSnapshot,
    PlayerStateSnapshot,
    Team,
    TeamSeasonStrength,
    TeamStrengthSnapshot,
)
from data.replay import REPLAY_ENV, replay_active, rewind_database, verify_rewind

SEASON = "2026-27"
N = 6
DEADLINE = datetime(2026, 10, 10, 10, 0)
BEFORE = datetime(2026, 10, 9, 12, 0)
AFTER = datetime(2026, 10, 12, 12, 0)


@pytest.fixture
def live_db(tmp_path):
    path = tmp_path / "live.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        # Two seasons of calendar. The old one still has is_current set, which
        # the engine would pick up if the rewind did not clear it.
        s.add(Gameweek(id=5, season=SEASON, name="GW5", deadline_time=datetime(2026, 10, 3, 10),
                       finished=True, data_checked=True, is_current=True))
        s.add(Gameweek(id=N, season=SEASON, name="GW6", deadline_time=DEADLINE,
                       is_next=True))
        s.add(Gameweek(id=7, season=SEASON, name="GW7", deadline_time=datetime(2026, 10, 17, 10)))
        s.add(Gameweek(id=38, season="2025-26", name="old", deadline_time=datetime(2026, 5, 1),
                       is_current=True))
        s.add(Team(id=1, name="Chelsea", short_name="CHE", strength_defence_away=9999,
                   strength_defence_home=9999, strength_attack_home=9999,
                   strength_attack_away=9999, strength_overall_home=9999,
                   strength_overall_away=9999))
        s.add(Player(id=100, fpl_id=100, code=100, web_name="Pedro", first_name="J",
                     second_name="P", team_id=1, position="FWD", now_cost=6.0,
                     status="a", form=9.0))
        s.add(Fixture(id=10, fpl_id=10, season=SEASON, gameweek=N, team_h_id=1, team_a_id=2,
                      finished=True, team_h_score=2, team_a_score=1,
                      kickoff_time=datetime(2026, 10, 11, 15)))
        s.add(PlayerGameweekStats(player_id=100, gameweek=5, season=SEASON,
                                  team_id_season=1, opponent_team_id=2, was_home=True))
        s.add(PlayerGameweekStats(player_id=100, gameweek=N, season=SEASON,
                                  team_id_season=1, opponent_team_id=2, was_home=True))
        # Price was 4.5 at the deadline and 6.0 now.
        s.add(PlayerStateSnapshot(player_id=100, snapshot_ts=BEFORE, season=SEASON,
                                  now_cost=4.5, status="d", form=3.0, selected_by_percent=2.0))
        s.add(PlayerStateSnapshot(player_id=100, snapshot_ts=AFTER, season=SEASON,
                                  now_cost=6.0, status="a", form=9.0, selected_by_percent=9.0))
        s.add(TeamStrengthSnapshot(season=SEASON, team_id=1, snapshot_ts=BEFORE, code=1,
                                   strength_defence_away=1150, strength_defence_home=1150,
                                   strength_attack_home=1150, strength_attack_away=1150,
                                   strength_overall_home=1150, strength_overall_away=1150))
        s.add(TeamStrengthSnapshot(season=SEASON, team_id=1, snapshot_ts=AFTER, code=1,
                                   strength_defence_away=9999, strength_defence_home=9999,
                                   strength_attack_home=9999, strength_attack_away=9999,
                                   strength_overall_home=9999, strength_overall_away=9999))
        s.add(TeamSeasonStrength(season=SEASON, team_id=1, code=1,
                                 strength_defence_away=9999, strength_defence_home=9999,
                                 strength_attack_home=9999, strength_attack_away=9999,
                                 strength_overall_home=9999, strength_overall_away=9999))
        # Duty: a penalty taker appointed after the deadline.
        s.add(PlayerSetPieceRole(player_id=100, season=SEASON, is_penalty_taker=True,
                                 penalty_order=1, penalty_xg_per_game=0.79,
                                 is_set_piece_taker=True, key_passes_per_game=1.7))
        # Coverage: the season has duty snapshots before the deadline (for another
        # player), so the rewind knows "no snapshot" means "no duty" for player 100.
        s.add(PlayerSetPieceSnapshot(player_id=101, season=SEASON, snapshot_ts=BEFORE,
                                     penalty_order=None, is_penalty_taker=False,
                                     penalty_xg_per_game=0.0, is_set_piece_taker=False,
                                     source="fpl"))
        s.add(PlayerSetPieceSnapshot(player_id=100, season=SEASON, snapshot_ts=AFTER,
                                     penalty_order=1, is_penalty_taker=True,
                                     penalty_xg_per_game=0.79, is_set_piece_taker=True,
                                     source="fpl"))
        # Decisions: the one made before the deadline survives, the later one goes.
        s.add(DecisionLog(gameweek=N, decision_type="lineup", details="{}",
                          projected_gain=1.0, created_at=BEFORE))
        s.add(DecisionLog(gameweek=N, decision_type="lineup", details="{}",
                          projected_gain=2.0, created_at=AFTER))
        s.add(FixtureOdds(fixture_id=10, home_cs_prob=0.3, fetched_at=BEFORE))
        s.add(FixtureOdds(fixture_id=10, home_cs_prob=0.9, fetched_at=AFTER))
        s.commit()
    finally:
        s.close()
    return path


@pytest.fixture
def replay_db(live_db, tmp_path):
    dst = tmp_path / "replay" / "gw6.db"
    rewind_database(live_db, dst, SEASON, N)
    return dst


def _one(db, sql, params=()):
    con = sqlite3.connect(db)
    try:
        return con.execute(sql, params).fetchone()
    finally:
        con.close()


def test_rewind_leaves_no_post_deadline_data(replay_db):
    assert verify_rewind(replay_db, SEASON, N) == []


def test_player_price_and_form_are_the_deadline_values(replay_db):
    assert _one(replay_db, "SELECT now_cost, status, form FROM players WHERE id = 100") == (
        4.5, "d", 3.0,
    )


def test_team_strength_is_the_deadline_value_in_both_tables(replay_db):
    assert _one(replay_db, "SELECT strength_defence_away FROM teams WHERE id = 1") == (1150,)
    assert _one(
        replay_db,
        "SELECT strength_defence_away FROM team_season_strength WHERE team_id = 1 AND season = ?",
        (SEASON,),
    ) == (1150,)


def test_duty_appointed_after_the_deadline_is_not_visible(replay_db):
    row = _one(
        replay_db,
        "SELECT is_penalty_taker, penalty_order, penalty_xg_per_game, key_passes_per_game "
        "FROM player_setpiece_roles WHERE player_id = 100 AND season = ?",
        (SEASON,),
    )
    assert row == (0, None, 0.0, 1.7)  # key passes are not snapshotted: kept


def test_decision_made_after_the_deadline_is_removed(replay_db):
    rows = _one(replay_db, "SELECT COUNT(*) FROM decision_log")
    assert rows == (1,)
    assert _one(replay_db, "SELECT projected_gain FROM decision_log") == (1.0,)


def test_odds_stamped_after_the_deadline_are_removed(replay_db):
    assert _one(replay_db, "SELECT home_cs_prob FROM fixture_odds") == (0.3,)


def test_gameweek_flags_stand_in_the_gap_before_the_deadline(replay_db):
    assert _one(
        replay_db,
        "SELECT is_current, is_next, finished FROM gameweeks WHERE season = ? AND id = 5",
        (SEASON,),
    ) == (1, 0, 1)
    assert _one(
        replay_db,
        "SELECT is_next, finished FROM gameweeks WHERE season = ? AND id = ?",
        (SEASON, N),
    ) == (1, 0)


def test_other_seasons_do_not_compete_for_current(replay_db):
    assert _one(replay_db, "SELECT is_current FROM gameweeks WHERE season = '2025-26'") == (0,)


def test_unplayed_fixture_loses_its_result(replay_db):
    assert _one(
        replay_db, "SELECT finished, team_h_score FROM fixtures WHERE id = 10"
    ) == (0, None)


def test_gameweek_stats_from_the_decided_week_are_removed(replay_db):
    assert _one(
        replay_db,
        "SELECT COUNT(*) FROM player_gw_stats WHERE season = ? AND gameweek >= ?",
        (SEASON, N),
    ) == (0,)


def test_rewind_never_writes_over_the_live_database(live_db):
    with pytest.raises(ValueError):
        rewind_database(live_db, live_db, SEASON, N)


def test_rewind_refuses_to_overwrite_an_existing_copy(live_db, tmp_path):
    dst = tmp_path / "exists.db"
    dst.write_bytes(b"")
    with pytest.raises(FileExistsError):
        rewind_database(live_db, dst, SEASON, N)


def test_verifier_catches_a_leak(replay_db):
    """A rewind that is incomplete must be caught, not trusted."""
    con = sqlite3.connect(replay_db)
    with con:
        con.execute(
            "INSERT INTO decision_log (gameweek, decision_type, details, projected_gain, "
            "dry_run, created_at) VALUES (6, 'lineup', '{}', 0, 1, '2026-10-12 12:00:00')"
        )
    con.close()
    violations = verify_rewind(replay_db, SEASON, N)
    assert any("decision_log" in v for v in violations)


def test_replay_flag_is_read_from_the_environment(monkeypatch):
    monkeypatch.delenv(REPLAY_ENV, raising=False)
    assert replay_active() is False
    monkeypatch.setenv(REPLAY_ENV, "1")
    assert replay_active() is True


def test_without_setpiece_coverage_the_current_duty_is_kept(live_db, tmp_path):
    """No snapshot before the deadline means the rewind cannot say who held duty
    then. It must keep the current answer and report the gap, not invent "none"."""
    con = sqlite3.connect(live_db)
    with con:
        con.execute("DELETE FROM player_setpiece_snapshots WHERE snapshot_ts < ?",
                    ("2026-10-10 10:00:00",))
    con.close()
    dst = tmp_path / "nocov.db"
    counts = rewind_database(live_db, dst, SEASON, N)
    assert counts["coverage_setpieces"] == 0
    assert _one(
        dst,
        "SELECT is_penalty_taker FROM player_setpiece_roles WHERE player_id = 100",
    ) == (1,)


def test_schema_is_upgraded_in_the_copy(tmp_path):
    """A live database that predates a table or column must still replay."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE gameweeks (id INTEGER, season VARCHAR(7), name VARCHAR, "
        "deadline_time DATETIME, finished BOOLEAN, is_current BOOLEAN, is_next BOOLEAN, "
        "average_entry_score FLOAT, highest_score FLOAT, is_dgw BOOLEAN, is_bgw BOOLEAN, "
        "data_checked BOOLEAN, PRIMARY KEY (id, season));"
        "INSERT INTO gameweeks (id, season, deadline_time, is_next) "
        "VALUES (6, '2026-27', '2026-10-10 10:00:00.000000', 1);"
    )
    con.commit()
    con.close()
    dst = tmp_path / "upgraded.db"
    from data.replay import _upgrade_schema
    _upgrade_schema(path)
    _upgrade_schema(path)  # idempotent
    con = sqlite3.connect(path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    cols = {r[1] for r in con.execute("PRAGMA table_info(decision_log)")}
    con.close()
    assert "team_strength_snapshots" in tables
    assert "manifest_id" in cols
    assert not dst.exists()
