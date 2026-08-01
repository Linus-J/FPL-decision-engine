"""data/ingestors/fpl_api.py -- upsert_gameweeks' deadline/name refresh.

Real bug found 2026-08-01, live-testing the GW1 cold-start path: our local
gameweeks table for 2026-27 held deadline_time values a full year stale
(GW1 = 2025-08-15, real FPL API = 2026-08-21). upsert_gameweeks's
on_conflict_do_update excluded deadline_time/name from the update set, so
a re-sync could never correct a stale row once it existed -- and PL
deadlines routinely shift for TV rescheduling, so this wasn't hypothetical.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import data.ingestors.fpl_api as fpl_api
from data.models import Base, Gameweek


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'fpl_api.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fpl_api, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _bootstrap_gw(gw_id: int, name: str, deadline: str) -> dict:
    return {
        "id": gw_id, "name": name, "deadline_time": deadline,
        "finished": False, "is_current": False, "is_next": False,
    }


def test_upsert_gameweeks_refreshes_a_rescheduled_deadline(session):
    fpl_api.upsert_gameweeks(
        {"events": [_bootstrap_gw(1, "Gameweek 1", "2025-08-15T17:30:00Z")]},
        season="2026-27",
    )
    row = session.execute(select(Gameweek).filter_by(id=1, season="2026-27")).scalar_one()
    assert row.deadline_time.year == 2025

    # real FPL API now reports the correct, rescheduled deadline
    fpl_api.upsert_gameweeks(
        {"events": [_bootstrap_gw(1, "Gameweek 1", "2026-08-21T17:30:00Z")]},
        season="2026-27",
    )
    session.expire_all()
    row = session.execute(select(Gameweek).filter_by(id=1, season="2026-27")).scalar_one()
    assert row.deadline_time.year == 2026
    assert row.deadline_time.month == 8 and row.deadline_time.day == 21


def test_upsert_gameweeks_is_idempotent_not_duplicating(session):
    fpl_api.upsert_gameweeks(
        {"events": [_bootstrap_gw(1, "Gameweek 1", "2026-08-21T17:30:00Z")]},
        season="2026-27",
    )
    fpl_api.upsert_gameweeks(
        {"events": [_bootstrap_gw(1, "Gameweek 1", "2026-08-21T17:30:00Z")]},
        season="2026-27",
    )
    rows = session.execute(select(Gameweek).filter_by(id=1, season="2026-27")).scalars().all()
    assert len(rows) == 1
