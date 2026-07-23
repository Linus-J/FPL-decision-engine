"""T7 gate — GW1 cold-start projections + departure gate.

Self-contained (temp DB). Proves the contract: every candidate gets a
non-default projection source (prior-season or position/price prior — never a
silent 0.0), and confirmed leavers (status 'u') are dropped.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Player, PlayerGameweekStats
from projection import cold_start as cs


def test_prior_season_of():
    assert cs.prior_season_of("2026-27") == "2025-26"
    assert cs.prior_season_of("2024-25") == "2023-24"


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'cs.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(cs, "get_session", lambda: Local())
    return Local


def _seed(Local):
    s = Local()
    try:
        # p1: established (10 prior appearances), p2: promoted/new (no prior),
        # p3: confirmed leaver (status 'u').
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Estab",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.add(Player(fpl_id=2, code=2, first_name="B", second_name="B", web_name="NewSign",
                     team_id=1, position="FWD", now_cost=6.5, status="a"))
        s.add(Player(fpl_id=3, code=3, first_name="C", second_name="C", web_name="Leaver",
                     team_id=1, position="DEF", now_cost=5.0, status="u"))
        s.commit()
        p1 = s.query(Player.id).filter_by(fpl_id=1).scalar()
        # 10 prior-season appearances for p1, avg 6 pts when played, all starts
        for gw in range(1, 11):
            s.add(PlayerGameweekStats(player_id=p1, gameweek=gw, season="2025-26",
                                      minutes=90, total_points=6))
        s.commit()
        return {"p1": p1}
    finally:
        s.close()


def test_load_prior_season_features(temp_session):
    ids = _seed(temp_session)
    feats = cs.load_prior_season_features("2025-26")
    row = feats[feats["player_id"] == ids["p1"]].iloc[0]
    assert row["appearances"] == 10
    assert row["ppg_played"] == pytest.approx(6.0)
    assert row["starts_rate"] == pytest.approx(1.0)


def test_departure_gate_drops_confirmed_leaver(temp_session):
    _seed(temp_session)
    players = cs.apply_departure_gate(cs.load_current_players())
    names = set(players["web_name"])
    assert "Leaver" not in names          # status 'u' dropped
    assert {"Estab", "NewSign"} <= names


def test_projection_sources_and_no_silent_zero(temp_session):
    _seed(temp_session)
    players = cs.apply_departure_gate(cs.load_current_players())
    prior = cs.load_prior_season_features("2025-26")
    proj = cs.project_cold_start(players, prior)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    src = proj.set_index("name")["proj_source"].to_dict()

    assert src["Estab"] == "prior_season"
    assert src["NewSign"] == "position_price_prior"   # no prior PL data
    # the deliverable's core contract: no silent 0.0, every slot has a source
    assert (proj["xpts"] > 0).all()
    assert proj["proj_source"].notna().all()
    # established player's xpts tracks prior ppg
    estab = proj[proj["name"] == "Estab"].iloc[0]
    assert estab["xpts"] == pytest.approx(6.0)
    assert estab["start_probability"] == pytest.approx(1.0)


def test_price_prior_monotonic_in_price():
    assert cs._price_prior("MID", 10.0) > cs._price_prior("MID", 5.0)
    assert cs._price_prior("FWD", 4.0) >= cs._MIN_XPTS
