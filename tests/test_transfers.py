"""optimiser/transfers.py -- get_dgw_coverage.

Real bug found 2026-07-30 (the user's own live-smoke-test request):
agent/decision_engine.py has imported and called `get_dgw_coverage` from
this module since its very first commit, but the function never actually
existed anywhere in the codebase -- an ImportError at module load, meaning
the live decision engine could never run at all. Undetected all this time
because nothing in the (backtest-focused) test suite imports
agent.decision_engine.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import data.db as data_db
import projection.pipeline as pipeline_module
from data.models import Base, Fixture, Team
from optimiser.transfers import get_dgw_coverage


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'dgw_coverage.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(data_db, "get_session", lambda: Local())
    # get_dgw_coverage also calls projection.pipeline._get_current_season,
    # which bound its OWN `get_session` at pipeline.py's import time (a
    # top-level `from data.db import get_session`) -- patching data.db's
    # attribute alone has no effect on that already-bound name once
    # pipeline.py has been imported anywhere else first (order-dependent
    # under the full suite, not in this file alone).
    monkeypatch.setattr(pipeline_module, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _seed_teams(session, ids: list[int]) -> None:
    session.add_all([
        Team(id=tid, name=f"Team{tid}", short_name=f"T{tid}") for tid in ids
    ])
    session.commit()


def test_get_dgw_coverage_counts_players_on_a_two_fixture_team(session):
    _seed_teams(session, [1, 2, 3, 4])
    # Team 1 plays TWICE in GW26 (a real DGW) -- vs team 2, then vs team 3.
    # Team 4 plays once (a normal week).
    session.add_all([
        Fixture(fpl_id=1, season="2026-27", gameweek=26, team_h_id=1, team_a_id=2),
        Fixture(fpl_id=2, season="2026-27", gameweek=26, team_h_id=3, team_a_id=1),
        Fixture(fpl_id=3, season="2026-27", gameweek=26, team_h_id=4, team_a_id=2),
    ])
    session.commit()

    players = pd.DataFrame({"id": [10, 11, 12], "team_id": [1, 1, 4]})
    projections = pd.DataFrame({
        "player_id": [10, 11, 12],
        "gameweek": [26, 26, 26],
        "xpts": [8.0, 5.0, 4.0],
    })
    coverage = get_dgw_coverage(
        squad_ids=[10, 11, 12], players=players, dgw_gws={26}, projections=projections
    )
    assert coverage[26]["squad_players_involved"] == 2  # players 10, 11 (team 1)
    assert coverage[26]["combined_xpts"] == pytest.approx(13.0)  # excludes player 12


def test_get_dgw_coverage_empty_when_no_dgw_gws(session):
    players = pd.DataFrame({"id": [10], "team_id": [1]})
    projections = pd.DataFrame({"player_id": [10], "gameweek": [26], "xpts": [8.0]})
    assert get_dgw_coverage([10], players, set(), projections) == {}


def test_get_dgw_coverage_empty_when_no_squad(session):
    players = pd.DataFrame({"id": [10], "team_id": [1]})
    projections = pd.DataFrame({"player_id": [10], "gameweek": [26], "xpts": [8.0]})
    assert get_dgw_coverage([], players, {26}, projections) == {}


def test_get_dgw_coverage_zero_involved_when_squad_has_no_dgw_teams(session):
    _seed_teams(session, [1, 2])
    session.add(Fixture(fpl_id=1, season="2026-27", gameweek=26, team_h_id=1, team_a_id=2))
    session.commit()

    players = pd.DataFrame({"id": [10], "team_id": [2]})  # team 2 plays once, not a DGW
    projections = pd.DataFrame({"player_id": [10], "gameweek": [26], "xpts": [8.0]})
    coverage = get_dgw_coverage([10], players, {26}, projections)
    assert coverage[26] == {"squad_players_involved": 0, "combined_xpts": 0.0}
