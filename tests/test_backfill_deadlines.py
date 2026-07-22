"""T3a acceptance gate — per-season deadlines + cross-season code crosswalk.

Self-contained: pure transforms tested on synthetic frames; DB writers tested
against a throwaway temp DB with get_session monkeypatched. No network, no
live fpl_bot.db.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Player
from scripts import backfill_history as bh

# --- pure transforms ---

def test_compute_gw_deadlines_min_kickoff_minus_90():
    df = pd.DataFrame(
        {
            "event": [1, 1, 2, None],
            "kickoff_time": [
                "2025-08-15T19:00:00Z",  # GW1 first match
                "2025-08-16T14:00:00Z",
                "2025-08-22T19:30:00Z",
                None,  # postponed / TBC → ignored
            ],
        }
    )
    dl = bh.compute_gw_deadlines(df)
    assert dl[1] == datetime(2025, 8, 15, 17, 30)  # 19:00 UTC − 90 min, naive
    assert dl[2] == datetime(2025, 8, 22, 18, 0)
    assert None not in dl and 3 not in dl


def test_compute_gw_deadlines_missing_columns():
    assert bh.compute_gw_deadlines(pd.DataFrame({"foo": [1]})) == {}


def test_element_code_map():
    df = pd.DataFrame({"id": [1, 2, 3], "code": [1001, 1002, None]})
    assert bh.element_code_map(df) == {1: 1001, 2: 1002}


def test_resolve_player_id_via_code():
    elem_to_code = {1: 1001, 2: 1002, 3: 1003}
    code_to_dbid = {1001: 50, 1003: 70}  # 1002's player left the league
    assert bh.resolve_player_id(1, elem_to_code, code_to_dbid) == 50
    assert bh.resolve_player_id(2, elem_to_code, code_to_dbid) is None  # left
    assert bh.resolve_player_id(3, elem_to_code, code_to_dbid) == 70
    assert bh.resolve_player_id(999, elem_to_code, code_to_dbid) is None  # unknown


# --- DB writers against a temp DB ---

@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 't3a.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(bh, "get_session", lambda: Local())
    return Local


def test_build_code_to_dbid_map(temp_session):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=1001, first_name="F", second_name="L",
                     web_name="A", team_id=1, position="MID", now_cost=5.0))
        s.add(Player(fpl_id=2, code=None, first_name="G", second_name="M",
                     web_name="B", team_id=1, position="DEF", now_cost=4.5))
        s.commit()
    finally:
        s.close()
    m = bh.build_code_to_dbid_map()
    assert 1001 in m  # coded player present
    assert len(m) == 1  # the code=None player is excluded


def test_upsert_gameweek_deadlines_and_idempotency(temp_session):
    from data.models import Gameweek

    d1 = {1: datetime(2024, 8, 16, 17, 30), 2: datetime(2024, 8, 24, 11, 30)}
    n = bh.upsert_gameweek_deadlines("2024-25", d1)
    assert n == 2

    s = temp_session()
    try:
        assert s.query(Gameweek).count() == 2
        gw1 = s.query(Gameweek).filter_by(season="2024-25", id=1).one()
        assert gw1.deadline_time == datetime(2024, 8, 16, 17, 30)
    finally:
        s.close()

    # re-run with an updated deadline → same row updated, not duplicated
    bh.upsert_gameweek_deadlines("2024-25", {1: datetime(2024, 8, 16, 18, 0)})
    s = temp_session()
    try:
        assert s.query(Gameweek).count() == 2  # no dupe
        gw1 = s.query(Gameweek).filter_by(season="2024-25", id=1).one()
        assert gw1.deadline_time == datetime(2024, 8, 16, 18, 0)  # updated
    finally:
        s.close()


def test_deadlines_are_season_scoped(temp_session):
    """Same GW-number in two seasons coexists (composite key from T2.5)."""
    from data.models import Gameweek

    bh.upsert_gameweek_deadlines("2023-24", {1: datetime(2023, 8, 11, 19, 0)})
    bh.upsert_gameweek_deadlines("2024-25", {1: datetime(2024, 8, 16, 18, 0)})
    s = temp_session()
    try:
        assert s.query(Gameweek).filter_by(id=1).count() == 2
    finally:
        s.close()
