"""Training features read team strength and set-piece duty as-of each
gameweek's deadline. A snapshot stamped after the deadline must never reach a
row that was decided before it existed.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import (
    Base,
    Gameweek,
    Player,
    PlayerGameweekStats,
    PlayerSetPieceRole,
    PlayerSetPieceSnapshot,
    TeamSeasonStrength,
    TeamStrengthSnapshot,
)
from projection import features

SEASON = "2026-27"
DEADLINE = datetime(2026, 9, 4, 17, 30)


@pytest.fixture
def Local(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'asof.db'}")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(features, "get_session", lambda: Session())
    return Session


def _strength_row(team_id, ts, defence_away):
    return TeamStrengthSnapshot(
        season=SEASON, team_id=team_id, snapshot_ts=ts, code=team_id,
        strength_overall_home=1200, strength_overall_away=1200,
        strength_attack_home=1200, strength_attack_away=1200,
        strength_defence_home=1200, strength_defence_away=defence_away,
    )


def _seed_fixture_context(s):
    s.add(Gameweek(id=1, season=SEASON, name="GW1", deadline_time=DEADLINE))
    s.add(PlayerGameweekStats(
        player_id=100, gameweek=1, season=SEASON,
        team_id_season=2, opponent_team_id=1, was_home=True,
    ))


def test_fixture_difficulty_uses_the_strength_at_the_deadline(Local):
    s = Local()
    try:
        _seed_fixture_context(s)
        # Opponent (team 1) is strong before the deadline and far stronger after.
        s.add(_strength_row(1, datetime(2026, 8, 1), defence_away=1150))
        s.add(_strength_row(1, datetime(2026, 9, 10), defence_away=9999))
        s.commit()
    finally:
        s.close()

    df = features.load_fixture_difficulty(SEASON).set_index("player_id")
    # Home player: the opponent's AWAY defence is the relevant figure.
    assert df.loc[100, "opp_defence_strength"] == 1150


def test_fixture_difficulty_falls_back_to_season_table_without_snapshots(Local):
    """Backfilled seasons have no snapshots. They read the season table, which
    is not point-in-time -- the fallback is named in the docstring and this test
    pins that it still works rather than returning nulls."""
    s = Local()
    try:
        _seed_fixture_context(s)
        s.add(TeamSeasonStrength(
            season=SEASON, team_id=1, code=1,
            strength_overall_home=1200, strength_overall_away=1200,
            strength_attack_home=1200, strength_attack_away=1200,
            strength_defence_home=1200, strength_defence_away=1111,
        ))
        s.commit()
    finally:
        s.close()

    df = features.load_fixture_difficulty(SEASON).set_index("player_id")
    assert df.loc[100, "opp_defence_strength"] == 1111


def _player(s, code):
    player = Player(
        fpl_id=code, code=code, web_name=f"P{code}", first_name="F", second_name="S",
        team_id=2, position="MID", now_cost=5.0,
    )
    s.add(player)
    s.commit()
    return player.id


def test_set_piece_duty_is_read_as_of_the_deadline(Local):
    s = Local()
    try:
        _seed_fixture_context(s)
        pid = _player(s, 300)
        s.add(PlayerGameweekStats(
            player_id=pid, gameweek=1, season=SEASON,
            team_id_season=2, opponent_team_id=1, was_home=True,
        ))
        # Before the deadline: no penalty duty. After: appointed penalty taker.
        s.add(PlayerSetPieceSnapshot(
            player_id=pid, season=SEASON, snapshot_ts=datetime(2026, 8, 20),
            penalty_order=None, is_penalty_taker=False, penalty_xg_per_game=0.0,
            is_set_piece_taker=False, source="fpl",
        ))
        s.add(PlayerSetPieceSnapshot(
            player_id=pid, season=SEASON, snapshot_ts=datetime(2026, 9, 10),
            penalty_order=1, is_penalty_taker=True, penalty_xg_per_game=0.79,
            is_set_piece_taker=True, source="fpl",
        ))
        s.add(PlayerSetPieceRole(
            player_id=pid, season=SEASON, is_penalty_taker=True,
            penalty_xg_per_game=0.79, is_set_piece_taker=True, key_passes_per_game=1.5,
        ))
        s.commit()
    finally:
        s.close()

    df = features.load_player_enrichment(SEASON).set_index("player_id")
    assert not bool(df.loc[pid, "is_penalty_taker"])
    assert df.loc[pid, "penalty_xg_per_game"] == 0.0
    # Key passes are not snapshotted and still come from the roles table.
    assert df.loc[pid, "key_passes_per_game"] == 1.5
