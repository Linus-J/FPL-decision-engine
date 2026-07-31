"""dashboard/data/decisions.py"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dashboard.data.decisions import (
    get_decision_history,
    get_latest_chip_plan,
    get_latest_transfer_plan,
)
from data.models import Base, DecisionLog


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'decisions.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def _log(session, gw: int, decision_type: str, details: dict, projected_gain: float = 0.0,
          actual_outcome: float | None = None) -> None:
    session.add(DecisionLog(
        gameweek=gw, decision_type=decision_type, details=json.dumps(details),
        projected_gain=projected_gain, actual_outcome=actual_outcome, dry_run=True,
    ))
    session.commit()


def test_empty_log_returns_empty_frame(session):
    df = get_decision_history(session)
    assert df.empty


def test_history_parses_details_and_respects_gw_window(session):
    _log(session, 1, "lineup", {"squad_ids": [1, 2]}, projected_gain=50.0, actual_outcome=48)
    _log(session, 25, "lineup", {"squad_ids": [3, 4]}, projected_gain=55.0)
    df = get_decision_history(session, limit_gws=5)
    assert set(df["gameweek"]) == {25}
    assert df.iloc[0]["details"] == {"squad_ids": [3, 4]}


def test_latest_chip_plan_returns_none_when_absent(session):
    assert get_latest_chip_plan(session) is None


def test_latest_chip_plan_returns_most_recent(session):
    _log(session, 10, "chip", {"chip": "wildcard", "reason": "old"}, projected_gain=10.0)
    _log(session, 11, "chip", {"chip": "3xc", "reason": "DGW incoming"}, projected_gain=15.0)
    plan = get_latest_chip_plan(session)
    assert plan == {"gameweek": 11, "chip": "3xc", "reason": "DGW incoming", "expected_gain": 15.0}


def test_latest_transfer_plan_returns_most_recent(session):
    _log(session, 10, "transfers", {
        "transfers_in": [{"player_id": 1, "web_name": "A", "cost": 5.0}],
        "transfers_out": [{"player_id": 2, "web_name": "B", "cost": 5.5}],
        "hits_taken": 0,
    }, projected_gain=3.2)
    plan = get_latest_transfer_plan(session)
    assert plan["gameweek"] == 10
    assert plan["hits_taken"] == 0
    assert plan["net_xpts_gain"] == 3.2
    assert plan["transfers_in"][0]["web_name"] == "A"
