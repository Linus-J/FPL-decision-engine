"""As-of reads over the append-only snapshot tables.

The canary: a snapshot stamped after the deadline must never be returned, even
when it is the most recent row. That is the property a backtest relies on.
"""

from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.asof import load_setpiece_asof, load_team_strength_asof
from data.models import Base, Player, PlayerSetPieceSnapshot, TeamStrengthSnapshot

SEASON = "2026-27"
DEADLINE = datetime(2026, 9, 4, 17, 30)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _strength(team_id, ts, overall_home, **kw):
    return TeamStrengthSnapshot(
        season=SEASON, team_id=team_id, snapshot_ts=ts, code=team_id,
        strength_overall_home=overall_home, strength_overall_away=overall_home,
        strength_attack_home=overall_home, strength_attack_away=overall_home,
        strength_defence_home=overall_home, strength_defence_away=overall_home,
        is_placeholder=kw.get("is_placeholder", False),
    )


def test_team_strength_returns_latest_before_deadline(db):
    db.add_all([
        _strength(1, datetime(2026, 8, 1), 1100),
        _strength(1, datetime(2026, 9, 1), 1150),
    ])
    db.commit()
    df = load_team_strength_asof(db, SEASON, DEADLINE)
    assert len(df) == 1
    assert df.iloc[0]["strength_overall_home"] == 1150


def test_team_strength_ignores_snapshots_after_deadline(db):
    # Canary: the post-deadline row is the most recent and must not be returned.
    db.add_all([
        _strength(1, datetime(2026, 9, 1), 1150),
        _strength(1, datetime(2026, 9, 10), 9999),
    ])
    db.commit()
    df = load_team_strength_asof(db, SEASON, DEADLINE)
    assert df.iloc[0]["strength_overall_home"] == 1150


def test_team_strength_excludes_teams_with_no_earlier_snapshot(db):
    db.add_all([
        _strength(1, datetime(2026, 8, 1), 1100),
        _strength(2, datetime(2026, 9, 10), 1300),
    ])
    db.commit()
    df = load_team_strength_asof(db, SEASON, DEADLINE)
    assert list(df["team_id"]) == [1]


def test_team_strength_keeps_placeholder_flag(db):
    db.add(_strength(1, datetime(2026, 9, 1), 0, is_placeholder=True))
    db.commit()
    df = load_team_strength_asof(db, SEASON, DEADLINE)
    assert bool(df.iloc[0]["is_placeholder"]) is True


def _player(db, code):
    player = Player(
        fpl_id=code, code=code, web_name=f"P{code}", first_name="F", second_name="S",
        team_id=1, position="MID", now_cost=5.0,
    )
    db.add(player)
    db.commit()
    return player.id


def _setpiece(player_id, ts, **kw):
    return PlayerSetPieceSnapshot(
        player_id=player_id, season=SEASON, snapshot_ts=ts,
        penalty_order=kw.get("penalty_order"), freekick_order=None, corner_order=None,
        is_penalty_taker=kw.get("is_penalty_taker", False),
        penalty_xg_per_game=kw.get("penalty_xg", 0.0), is_set_piece_taker=False,
        source="fpl",
    )


def test_setpiece_returns_duty_held_at_deadline(db):
    pid = _player(db, 100)
    db.add_all([
        _setpiece(pid, datetime(2026, 8, 1), penalty_order=1, is_penalty_taker=True),
        _setpiece(pid, datetime(2026, 9, 1), penalty_order=2),
    ])
    db.commit()
    df = load_setpiece_asof(db, SEASON, DEADLINE)
    assert df.iloc[0]["penalty_order"] == 2
    assert not bool(df.iloc[0]["is_penalty_taker"])


def test_setpiece_ignores_duty_granted_after_deadline(db):
    # Canary: a penalty-taker appointed after the deadline must not be visible.
    pid = _player(db, 101)
    db.add(_setpiece(pid, datetime(2026, 9, 20), penalty_order=1, is_penalty_taker=True))
    db.commit()
    df = load_setpiece_asof(db, SEASON, DEADLINE)
    assert df.empty


def test_setpiece_retirement_reads_as_no_duty(db):
    pid = _player(db, 102)
    db.add_all([
        _setpiece(pid, datetime(2026, 8, 1), penalty_order=1, is_penalty_taker=True),
        _setpiece(pid, datetime(2026, 9, 1)),  # retirement: all orders NULL
    ])
    db.commit()
    df = load_setpiece_asof(db, SEASON, DEADLINE)
    assert pd.isna(df.iloc[0]["penalty_order"])
    assert not bool(df.iloc[0]["is_penalty_taker"])
