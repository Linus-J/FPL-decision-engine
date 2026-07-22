"""T2 acceptance gate — append-only snapshot-WRITE path.

Self-contained: a synthetic bootstrap + a throwaway temp DB with
``fpl_api.get_session`` monkeypatched. Never touches the live FPL API or
fpl_bot.db. Proves:
  - two captures (distinct snapshot_ts) => 2 snapshot rows per player,
  - ``players`` stays 1 row per player (snapshots never UPDATE the table),
  - re-running with the same snapshot_ts is idempotent (nothing inserted),
  - DISTINCT snapshot_ts count == number of captures.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from data.ingestors import fpl_api
from data.models import Base, Player, PlayerStateSnapshot


def _bootstrap() -> dict:
    """Minimal bootstrap-static payload for three players over one 'next' GW."""
    elements = [
        {
            "id": 101, "now_cost": 55, "status": "a", "selected_by_percent": "12.3",
            "form": "4.5", "ict_index": "8.1", "influence": "20.0",
            "creativity": "15.0", "threat": "30.0", "news": "",
            "transfers_in_event": 1000, "transfers_out_event": 200,
            "chance_of_playing_this_round": 100, "chance_of_playing_next_round": 100,
        },
        {
            "id": 102, "now_cost": 120, "status": "d", "selected_by_percent": "45.0",
            "form": "6.0", "ict_index": "12.0", "influence": "40.0",
            "creativity": "25.0", "threat": "60.0", "news": "Knock - 75%",
            "news_added": "2026-07-20T09:00:00Z",
            "transfers_in_event": 5000, "transfers_out_event": 8000,
            "chance_of_playing_this_round": 75, "chance_of_playing_next_round": 75,
        },
        {
            "id": 103, "now_cost": 45, "status": "a", "selected_by_percent": "1.2",
            "form": "1.0", "ict_index": "2.0", "influence": "5.0",
            "creativity": "3.0", "threat": "4.0", "news": "",
            "transfers_in_event": 10, "transfers_out_event": 5,
            "chance_of_playing_this_round": None, "chance_of_playing_next_round": None,
        },
    ]
    events = [
        {"id": 1, "is_current": True, "is_next": False},
        {"id": 2, "is_current": False, "is_next": True},
    ]
    return {"elements": elements, "events": events}


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    db_path = tmp_path / "t2.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fpl_api, "get_session", lambda: Local())

    # Seed the players that upsert_players would have created.
    seed = Local()
    try:
        for fpl_id in (101, 102, 103):
            seed.add(
                Player(
                    fpl_id=fpl_id,
                    first_name="F", second_name="L", web_name=f"P{fpl_id}",
                    team_id=1, position="MID", now_cost=5.0,
                )
            )
        seed.commit()
    finally:
        seed.close()
    return Local


def test_two_captures_two_rows_per_player(temp_session):
    bs = _bootstrap()
    ts1 = datetime(2026, 8, 1, 10, 0, 0)
    ts2 = ts1 + timedelta(days=1)

    n1 = fpl_api.write_player_snapshots(bs, ts1, season="2026-27")
    n2 = fpl_api.write_player_snapshots(bs, ts2, season="2026-27")
    assert n1 == 3
    assert n2 == 3

    s = temp_session()
    try:
        # 2 snapshot rows per player, players table untouched (still 3 rows).
        assert s.query(func.count(PlayerStateSnapshot.id)).scalar() == 6
        assert s.query(func.count(Player.id)).scalar() == 3
        distinct_ts = s.query(func.count(func.distinct(PlayerStateSnapshot.snapshot_ts))).scalar()
        assert distinct_ts == 2  # == number of captures
    finally:
        s.close()


def test_same_ts_is_idempotent(temp_session):
    bs = _bootstrap()
    ts = datetime(2026, 8, 1, 10, 0, 0)

    first = fpl_api.write_player_snapshots(bs, ts, season="2026-27")
    second = fpl_api.write_player_snapshots(bs, ts, season="2026-27")
    assert first == 3
    assert second == 0  # unique (player_id, snapshot_ts) => nothing inserted

    s = temp_session()
    try:
        assert s.query(func.count(PlayerStateSnapshot.id)).scalar() == 3
    finally:
        s.close()


def test_snapshot_values_and_gw_context(temp_session):
    bs = _bootstrap()
    ts = datetime(2026, 8, 1, 10, 0, 0)
    fpl_api.write_player_snapshots(bs, ts, season="2026-27")

    s = temp_session()
    try:
        p101 = s.query(Player).filter_by(fpl_id=101).one()
        snap = (
            s.query(PlayerStateSnapshot)
            .filter_by(player_id=p101.id, snapshot_ts=ts)
            .one()
        )
        assert snap.now_cost == pytest.approx(5.5)  # 55 / 10
        assert snap.status == "a"
        assert snap.selected_by_percent == pytest.approx(12.3)
        assert snap.transfers_in_event == 1000
        assert snap.gameweek_context == 2  # is_next event id
        assert snap.season == "2026-27"

        # doubtful player carried news + chance-of-playing through
        p102 = s.query(Player).filter_by(fpl_id=102).one()
        snap2 = s.query(PlayerStateSnapshot).filter_by(player_id=p102.id).one()
        assert snap2.status == "d"
        assert snap2.chance_of_playing_this_round == 75
        assert snap2.news == "Knock - 75%"
        assert snap2.news_added is not None
    finally:
        s.close()
