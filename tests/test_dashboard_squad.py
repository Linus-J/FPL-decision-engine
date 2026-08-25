"""dashboard/data/squad.py::get_current_squad

Mocks the cross-module helpers (live FPL fetch, projections) at the point of
use in squad.py's own namespace -- the module's own DB queries run against a
real temp SQLite DB, matching this repo's existing test convention."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import dashboard.data.squad as squad_module
from data.models import Base, DecisionLog, Player, Team


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'squad.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def _seed_players(session) -> None:
    session.add(Team(id=1, name="Team A", short_name="TMA"))
    session.add_all([
        Player(id=1, fpl_id=101, code=101, first_name="A", second_name="One", web_name="One",
               team_id=1, position="GKP", now_cost=5.0),
        Player(id=2, fpl_id=102, code=102, first_name="B", second_name="Two", web_name="Two",
               team_id=1, position="DEF", now_cost=6.0),
        Player(id=3, fpl_id=103, code=103, first_name="C", second_name="Three", web_name="Three",
               team_id=1, position="FWD", now_cost=9.0),
    ])
    session.commit()


def test_live_picks_used_when_available(session, monkeypatch):
    _seed_players(session)
    monkeypatch.setattr(squad_module, "_get_current_and_next_gw", lambda: (5, 6))
    picks_payload = {"picks": [
        {"element": 101, "position": 1, "multiplier": 2,
         "is_captain": True, "is_vice_captain": False},
        {"element": 102, "position": 2, "multiplier": 1,
         "is_captain": False, "is_vice_captain": True},
        {"element": 103, "position": 12, "multiplier": 0,
         "is_captain": False, "is_vice_captain": False},
    ]}
    monkeypatch.setattr(
        squad_module, "get_picks",
        lambda team_id, gw: picks_payload if gw == 6 else {},
    )
    monkeypatch.setattr(
        squad_module, "get_latest_projections",
        lambda gw: pd.DataFrame({"player_id": [1, 2, 3], "xpts": [4.0, 3.0, 1.0]}),
    )

    squad = squad_module.get_current_squad(session, team_id=12345)

    assert len(squad) == 3
    assert squad["gameweek"].iloc[0] == 6
    starters = squad[squad["is_starting"]]
    assert set(starters["player_id"]) == {1, 2}
    bench = squad[~squad["is_starting"]]
    assert list(bench["player_id"]) == [3]
    captain = squad[squad["is_captain"]]
    assert captain["player_id"].iloc[0] == 1


def test_falls_back_to_decision_log_when_no_live_picks(session, monkeypatch):
    _seed_players(session)
    session.add(DecisionLog(
        gameweek=7, decision_type="lineup",
        details=json.dumps({
            "squad_ids": [1, 2, 3], "starting_ids": [1, 2],
            "captain_id": 2, "vice_captain_id": 1,
        }),
        projected_gain=7.0, dry_run=True,
    ))
    session.commit()

    monkeypatch.setattr(squad_module, "_get_current_and_next_gw", lambda: (7, 8))
    monkeypatch.setattr(squad_module, "get_picks", lambda team_id, gw: {})
    monkeypatch.setattr(
        squad_module, "get_latest_projections",
        lambda gw: pd.DataFrame({"player_id": [1, 2], "xpts": [4.0, 3.0]}),
    )

    squad = squad_module.get_current_squad(session, team_id=12345)

    assert len(squad) == 3
    assert squad["gameweek"].iloc[0] == 7
    starters = squad[squad["is_starting"]]
    bench = squad[~squad["is_starting"]]
    assert set(starters["player_id"]) == {1, 2}
    assert list(bench["player_id"]) == [3]
    captain = squad[squad["is_captain"]]
    assert captain["player_id"].iloc[0] == 2


def test_fallback_with_no_projections_yet_reports_nan_xpts_and_decision_total(session, monkeypatch):
    """True cold-start case: player_projections has no rows at all, so
    per-player xPts should be NaN (not misleadingly 0.0), and the squad's
    .attrs should carry the decision's own recorded total for display."""
    _seed_players(session)
    session.add(DecisionLog(
        gameweek=1, decision_type="lineup",
        details=json.dumps({
            "squad_ids": [1, 2, 3], "starting_ids": [1, 2],
            "captain_id": 1, "vice_captain_id": 2,
        }),
        projected_gain=42.5, dry_run=True,
    ))
    session.commit()

    monkeypatch.setattr(squad_module, "_get_current_and_next_gw", lambda: (1, 1))
    monkeypatch.setattr(squad_module, "get_picks", lambda team_id, gw: {})
    monkeypatch.setattr(
        squad_module, "get_latest_projections",
        lambda gw: pd.DataFrame(columns=["player_id", "xpts"]),
    )

    squad = squad_module.get_current_squad(session, team_id=12345)

    assert squad["xpts"].isna().all()
    assert squad.attrs["fallback_projected_total"] == 42.5


def test_returns_empty_when_nothing_available(session, monkeypatch):
    _seed_players(session)
    monkeypatch.setattr(squad_module, "_get_current_and_next_gw", lambda: (7, 8))
    monkeypatch.setattr(squad_module, "get_picks", lambda team_id, gw: {})

    squad = squad_module.get_current_squad(session, team_id=12345)

    assert squad.empty


def test_reused_fpl_id_does_not_duplicate_the_squad(session, monkeypatch):
    """A stale prior-season row sharing an fpl_id must not join into the squad.

    FPL renumbers its elements every summer, so an fpl_id identifies a player
    only WITHIN a season -- on the live DB 975 player rows carry 756 distinct
    fpl_ids. Merging picks on fpl_id alone therefore matched the departed
    player as well as the current one, and the published site showed 19 players
    in a 15-man squad (2026-08-25). The current row is the one the latest
    ingest touched.
    """
    session.add(Team(id=1, name="Team A", short_name="TMA"))
    session.add_all([
        # Same fpl_id, different players, different seasons. The departed one
        # is listed FIRST and has the lower id, so a naive query returns it.
        Player(id=1, fpl_id=101, code=900, first_name="Gone", second_name="Away",
               web_name="Departed", team_id=1, position="GKP", now_cost=4.5,
               status="u", updated_at=datetime(2026, 7, 23, 7, 19)),
        Player(id=2, fpl_id=101, code=901, first_name="Here", second_name="Now",
               web_name="Current", team_id=1, position="GKP", now_cost=5.0,
               status="a", updated_at=datetime(2026, 8, 20, 8, 28)),
    ])
    session.commit()

    monkeypatch.setattr(squad_module, "_get_current_and_next_gw", lambda: (5, 6))
    monkeypatch.setattr(squad_module, "get_picks", lambda team_id, gw: {"picks": [
        {"element": 101, "position": 1, "multiplier": 1,
         "is_captain": False, "is_vice_captain": False},
    ]} if gw == 6 else {})
    monkeypatch.setattr(squad_module, "get_latest_projections", lambda gw: pd.DataFrame())

    squad = squad_module.get_current_squad(session, team_id=1)

    assert len(squad) == 1, f"one pick must yield one row, got {len(squad)}"
    assert squad.iloc[0]["web_name"] == "Current"
    assert squad.iloc[0]["player_id"] == 2
