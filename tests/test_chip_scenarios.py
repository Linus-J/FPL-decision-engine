"""P3-5 — scenario-EV chip data plumbing (optimiser/chip_scenarios.py).

Key property under test: composing per-scenario totals across independent
fixture groups (and independent gameweeks) by POSITION, not by raw
``scenario_id`` value, since those ranges are only jointly meaningful WITHIN
one fixture. A shared fixture between two player sets must stay correlated
when the two totals are later subtracted (``gain_distribution``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, ProjectionSample
from optimiser import captaincy, chip_scenarios


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'chip_scenarios.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(captaincy, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _insert(session, rows: list[dict]) -> None:
    session.add_all([ProjectionSample(**row) for row in rows])
    session.commit()


def _rows_for(
    pid: int, gw: int, season: str, offset: int, values: list[float], created
) -> list[dict]:
    return [
        {"player_id": pid, "gameweek": gw, "season": season,
         "scenario_id": offset + i, "xpts": v, "created_at": created}
        for i, v in enumerate(values)
    ]


# --- load_scenario_totals ---------------------------------------------------

def test_load_scenario_totals_sums_one_fixture_group(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    rows = (
        _rows_for(1, 5, "2099-00", 0, [1.0, 2.0, 3.0], created)
        + _rows_for(2, 5, "2099-00", 0, [10.0, 20.0, 30.0], created)
    )
    _insert(session, rows)
    total = chip_scenarios.load_scenario_totals("2099-00", 5, [1, 2])
    assert total.tolist() == [11.0, 22.0, 33.0]


def test_load_scenario_totals_composes_across_fixture_groups_by_position(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    # fixture A: player 1, scenario_id [0, 2); fixture B: player 2, scenario_id [2, 4)
    rows = (
        _rows_for(1, 5, "2099-00", 0, [1.0, 2.0], created)
        + _rows_for(2, 5, "2099-00", 2, [100.0, 200.0], created)
    )
    _insert(session, rows)
    total = chip_scenarios.load_scenario_totals("2099-00", 5, [1, 2])
    # composed by LOCAL rank within each group (position 0, 1), not raw scenario_id
    assert total.tolist() == [101.0, 202.0]


def test_load_scenario_totals_empty_player_ids_returns_empty():
    assert chip_scenarios.load_scenario_totals("2099-00", 5, []).empty


def test_load_scenario_totals_no_rows_returns_empty(session):
    assert chip_scenarios.load_scenario_totals("2099-00", 5, [1]).empty


def test_load_scenario_totals_accepts_numpy_int_gameweek(session):
    # recommend_chip's real caller passes current_gw as a numpy.int64 (from a
    # pandas groupby/unique), not a plain Python int -- must not crash.
    created = pd.Timestamp.now("UTC").to_pydatetime()
    _insert(session, _rows_for(1, 5, "2099-00", 0, [1.0, 2.0], created))
    total = chip_scenarios.load_scenario_totals("2099-00", np.int64(5), [1])
    assert total.tolist() == [1.0, 2.0]


def test_load_scenario_totals_multi_gw_sums_across_gameweeks(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    rows = (
        _rows_for(1, 5, "2099-00", 0, [1.0, 2.0], created)
        + _rows_for(1, 6, "2099-00", 0, [10.0, 20.0], created)
    )
    _insert(session, rows)
    total = chip_scenarios.load_scenario_totals("2099-00", [5, 6], [1])
    assert total.tolist() == [11.0, 22.0]


def test_load_scenario_totals_multi_gw_missing_one_gw_is_all_or_nothing(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    _insert(session, _rows_for(1, 5, "2099-00", 0, [1.0, 2.0], created))
    # no samples persisted for gw 6 at all
    total = chip_scenarios.load_scenario_totals("2099-00", [5, 6], [1])
    assert total.empty


# --- gain_distribution -------------------------------------------------------

def test_gain_distribution_subtracts_two_totals(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    rows = (
        _rows_for(1, 5, "2099-00", 0, [10.0, 20.0], created)
        + _rows_for(2, 5, "2099-00", 0, [1.0, 2.0], created)
    )
    _insert(session, rows)
    gains = chip_scenarios.gain_distribution("2099-00", 5, [1], [2])
    assert gains.tolist() == [9.0, 18.0]


def test_gain_distribution_shared_fixture_cancels_correlated_component(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    # players 1 and 2 share fixture A (perfectly correlated draws); player 3
    # is the "minus" side and lives in an unrelated fixture B.
    rows = (
        _rows_for(1, 5, "2099-00", 0, [5.0, 9.0], created)
        + _rows_for(2, 5, "2099-00", 0, [5.0, 9.0], created)
        + _rows_for(3, 5, "2099-00", 2, [1.0, 1.0], created)
    )
    _insert(session, rows)
    # plus = {1, 2} (same fixture, so their SUM has amplified variance);
    # minus = {3}. plus_total = [10, 18], minus_total = [1, 1].
    gains = chip_scenarios.gain_distribution("2099-00", 5, [1, 2], [3])
    assert gains.tolist() == [9.0, 17.0]


def test_gain_distribution_empty_when_either_side_missing(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    _insert(session, _rows_for(1, 5, "2099-00", 0, [1.0, 2.0], created))
    assert chip_scenarios.gain_distribution("2099-00", 5, [1], [2]).empty
