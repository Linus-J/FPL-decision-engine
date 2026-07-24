"""Phase-2 P0 — per-GW fixture-specific projections + distributional contract.

Covers D3 (the flat-broadcast bug): horizon GWs must differ by their opponent.
Plus the output-contract schema (xpts_mean/xpts_var + projection_samples with a
shared scenario_id for P-COV covariance).
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import scripts.backtest as bt
from data.models import Base, Player, PlayerProjection, ProjectionSample
from projection.fixture_adjust import fixture_multiplier


# --- pure fixture multiplier ------------------------------------------------
def test_fixture_multiplier_direction():
    weak_home = fixture_multiplier(800.0, was_home=True)     # weak opp, home → easy
    strong_away = fixture_multiplier(1600.0, was_home=False)  # strong opp, away → hard
    assert weak_home > 1.0 > strong_away
    # bounded
    assert fixture_multiplier(1.0, True) <= 1.40
    assert fixture_multiplier(99999.0, False) >= 0.70


def test_fixture_multiplier_missing_opponent_is_neutralish():
    # no opponent strength → home/away-only adjustment, never 0
    assert fixture_multiplier(None, None) == 1.0
    assert fixture_multiplier(None, True) > 1.0
    assert fixture_multiplier(None, False) < 1.0


# --- D3 regression: horizon GWs differ by fixture ---------------------------
def test_build_gw_projections_is_fixture_specific(monkeypatch):
    history = pd.DataFrame({"player_id": [1, 1], "gameweek": [1, 2], "minutes": [90, 90]})
    players = pd.DataFrame([{"id": 1, "status": "a"}])
    monkeypatch.setattr(bt, "minutes_batch", lambda h, m: {1: 0.9})
    monkeypatch.setattr(bt, "points_batch", lambda h, m: {1: 5.0})

    # GW5: weak opponent at home (easy); GW6: strong opponent away (hard)
    opp_ctx = {(1, 5): (800.0, True), (1, 6): (1600.0, False)}
    proj = bt._build_gw_projections(
        history, players, None, None, target_gw=5, horizon=2, opp_ctx=opp_ctx
    )
    xp5 = proj[proj["gameweek"] == 5]["xpts"].iloc[0]
    xp6 = proj[proj["gameweek"] == 6]["xpts"].iloc[0]
    assert xp5 > xp6                       # NOT the old flat broadcast
    assert xp5 != pytest.approx(xp6)
    assert (proj["xpts"] == proj["xpts_mean"]).all()   # mean alias populated


def test_build_gw_projections_unavailable_player_zeroed(monkeypatch):
    history = pd.DataFrame({"player_id": [1], "gameweek": [1], "minutes": [90]})
    players = pd.DataFrame([{"id": 1, "status": "i"}])  # injured
    monkeypatch.setattr(bt, "minutes_batch", lambda h, m: {1: 0.9})
    monkeypatch.setattr(bt, "points_batch", lambda h, m: {1: 5.0})
    proj = bt._build_gw_projections(history, players, None, None, 5, 2, {(1, 5): (800.0, True)})
    assert (proj["xpts"] == 0.0).all()


# --- output-contract schema -------------------------------------------------
@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'p0.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    s.add(Player(id=1, fpl_id=1, code=1, first_name="A", second_name="A",
                 web_name="a", team_id=1, position="MID", now_cost=5.0))
    s.commit()
    yield s
    s.close()


def test_projection_mean_var_columns(session):
    session.add(PlayerProjection(player_id=1, gameweek=5, xpts=6.4,
                                 xpts_mean=6.4, xpts_var=2.1))
    session.commit()
    row = session.query(PlayerProjection).one()
    assert (row.xpts_mean, row.xpts_var) == (6.4, 2.1)


def test_projection_samples_scenario_roundtrip(session):
    # three players' draws under two scenarios share the scenario_id (P-COV)
    for scen in (0, 1):
        session.add(ProjectionSample(player_id=1, gameweek=5, season="2025-26",
                                     scenario_id=scen, xpts=4.0 + scen))
    session.commit()
    draws = session.query(ProjectionSample).filter_by(player_id=1, gameweek=5).all()
    assert {d.scenario_id for d in draws} == {0, 1}
    assert sorted(d.xpts for d in draws) == [4.0, 5.0]
