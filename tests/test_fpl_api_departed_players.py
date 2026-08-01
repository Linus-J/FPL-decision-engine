"""data/ingestors/fpl_api.py::run_full_ingest -- departed players must not
be sent to ingest_player_history.

Real bug found 2026-08-01, live-testing on the user's machine: 169/733
players failed with a 404 on one real run, every single one status='u'
(confirmed departed -- their OLD fpl_id's element-summary endpoint is gone
from FPL's API once they leave). This wasn't a transient failure; it was
a guaranteed 404 repeated on every sync, spamming warnings for data that
was never coming back.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import data.ingestors.fpl_api as fpl_api
from data.models import Base


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'fpl_api.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fpl_api, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _bootstrap_team(team_id: int) -> dict:
    return {
        "id": team_id, "name": f"Team{team_id}", "short_name": f"T{team_id}",
        "strength_overall_home": 1200, "strength_overall_away": 1200,
        "strength_attack_home": 1200, "strength_attack_away": 1200,
        "strength_defence_home": 1200, "strength_defence_away": 1200,
    }


def _bootstrap_element(fpl_id: int, code: int, status: str) -> dict:
    return {
        "id": fpl_id, "code": code, "first_name": "P", "second_name": str(fpl_id),
        "web_name": f"p{fpl_id}", "team": 1, "element_type": 3, "now_cost": 50,
        "status": status,
    }


@pytest.mark.asyncio
async def test_run_full_ingest_skips_history_for_departed_players(session, monkeypatch):
    bootstrap = {
        "teams": [_bootstrap_team(1)],
        "events": [],
        "elements": [
            _bootstrap_element(1, 101, "a"),   # active -- should be ingested
            _bootstrap_element(2, 102, "u"),   # departed -- must be skipped
        ],
    }

    async def _fake_bootstrap():
        return bootstrap

    async def _fake_fixtures():
        return []

    called_with: list[int] = []

    async def _fake_ingest_history(fpl_id: int, db_id: int, season: str = "2026-27") -> None:
        called_with.append(fpl_id)

    monkeypatch.setattr(fpl_api, "fetch_bootstrap", _fake_bootstrap)
    monkeypatch.setattr(fpl_api, "fetch_fixtures", _fake_fixtures)
    monkeypatch.setattr(fpl_api, "ingest_player_history", _fake_ingest_history)
    monkeypatch.setattr(
        fpl_api, "write_player_snapshots",
        lambda bootstrap, ts, season="2026-27": 0,
    )

    await fpl_api.run_full_ingest("2026-27")

    assert called_with == [1]  # only the active player, never the departed one


@pytest.mark.asyncio
async def test_run_full_ingest_still_ingests_history_for_all_active_players(
    session, monkeypatch
):
    bootstrap = {
        "teams": [_bootstrap_team(1)],
        "events": [],
        "elements": [
            _bootstrap_element(1, 101, "a"),
            _bootstrap_element(2, 102, "d"),  # doubtful, NOT departed -- must still run
            _bootstrap_element(3, 103, "i"),  # injured, NOT departed -- must still run
        ],
    }

    async def _fake_bootstrap():
        return bootstrap

    async def _fake_fixtures():
        return []

    called_with: list[int] = []

    async def _fake_ingest_history(fpl_id: int, db_id: int, season: str = "2026-27") -> None:
        called_with.append(fpl_id)

    monkeypatch.setattr(fpl_api, "fetch_bootstrap", _fake_bootstrap)
    monkeypatch.setattr(fpl_api, "fetch_fixtures", _fake_fixtures)
    monkeypatch.setattr(fpl_api, "ingest_player_history", _fake_ingest_history)
    monkeypatch.setattr(
        fpl_api, "write_player_snapshots",
        lambda bootstrap, ts, season="2026-27": 0,
    )

    await fpl_api.run_full_ingest("2026-27")

    assert sorted(called_with) == [1, 2, 3]
