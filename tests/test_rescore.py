"""P-RS — 26/27 re-score of 25/26 actuals (finding C1).

Standard scoring + DefCon are unchanged 25/26->26/27, so the re-score is exactly
total_points - bonus_as_played + bonus_2627. Covers the DGW-summing map load and
the honest fallback (no event coverage -> keep the as-played total).
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, RecomputedBonus
from projection.rescore import (
    load_bonus_2627_map,
    rescore_actuals,
    rescore_coverage,
    rescore_coverage_relevant,
    rescore_points,
)


def test_rescore_points_swaps_bonus_only():
    # total 12 pts included 3 as-played bonus; 26/27 recompute awards only 1
    assert rescore_points(total_points=12, bonus_as_played=3, bonus_2627=1) == 10
    # more bonus under 26/27 rules -> total goes up
    assert rescore_points(total_points=8, bonus_as_played=0, bonus_2627=3) == 11
    # no bonus either way -> unchanged
    assert rescore_points(total_points=6, bonus_as_played=0, bonus_2627=0) == 6


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rs.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def test_load_bonus_2627_map_sums_dgw_matches(session):
    # player 1 has a DGW in GW5 (two matches) -> bonus_2627 sums to 2+1=3
    session.add(RecomputedBonus(player_id=1, season="2025-26", gameweek=5,
                                game_id="g1", bps_2627=40, bonus_2627=2))
    session.add(RecomputedBonus(player_id=1, season="2025-26", gameweek=5,
                                game_id="g2", bps_2627=20, bonus_2627=1))
    session.add(RecomputedBonus(player_id=2, season="2025-26", gameweek=5,
                                game_id="g1", bps_2627=30, bonus_2627=3))
    session.commit()
    m = load_bonus_2627_map(session, "2025-26")
    assert m == {(1, 5): 3, (2, 5): 3}


def test_rescore_actuals_falls_back_when_uncovered():
    df = pd.DataFrame([
        {"player_id": 1, "gameweek": 5, "total_points": 12, "bonus": 3},  # covered
        {"player_id": 2, "gameweek": 5, "total_points": 6, "bonus": 0},   # NOT covered
    ])
    bonus_map = {(1, 5): 1}
    out = rescore_actuals(df, bonus_map)
    assert out.set_index("player_id")["total_points_2627"].to_dict() == {1: 10, 2: 6}


def test_rescore_actuals_applies_dgw_delta_once_not_per_row():
    # player 1 has a genuine DGW in GW5 (two real fixture rows, matching the
    # fixed PlayerGameweekStats schema). bonus_2627_map's value is already the
    # WHOLE gameweek's recomputed bonus (5, per load_bonus_2627_map's DGW sum).
    # As-played bonus summed across both rows is 2+1=3, so the correct total
    # 26/27 points across the gameweek is (10+5) - 3 + 5 = 17, achieved by
    # adding the +2 delta to exactly one row and leaving the other untouched.
    df = pd.DataFrame([
        {"player_id": 1, "gameweek": 5, "total_points": 10, "bonus": 2},
        {"player_id": 1, "gameweek": 5, "total_points": 5, "bonus": 1},
        {"player_id": 2, "gameweek": 5, "total_points": 6, "bonus": 0},  # single fixture, uncovered
    ])
    bonus_map = {(1, 5): 5}
    out = rescore_actuals(df, bonus_map)
    p1_rows = out[out["player_id"] == 1]["total_points_2627"].tolist()
    assert p1_rows == [12, 5]
    assert sum(p1_rows) == 17
    assert out[out["player_id"] == 2]["total_points_2627"].tolist() == [6]


def test_rescore_coverage():
    df = pd.DataFrame([
        {"player_id": 1, "gameweek": 5, "total_points": 12, "bonus": 3},
        {"player_id": 2, "gameweek": 5, "total_points": 6, "bonus": 0},
    ])
    assert rescore_coverage(df, {(1, 5): 1}) == pytest.approx(0.5)
    assert rescore_coverage(df, {(1, 5): 1, (2, 5): 0}) == pytest.approx(1.0)
    assert rescore_coverage(df, {}) == pytest.approx(0.0)
    assert rescore_coverage(pd.DataFrame(), {}) == 0.0


def test_rescore_coverage_relevant_ignores_zero_bonus_rows():
    # most player_gw_stats rows are 0-minute squad players who never earn
    # bonus either way -- the "relevant" metric should ignore them entirely
    df = pd.DataFrame([
        {"player_id": 1, "gameweek": 5, "total_points": 12, "bonus": 3},  # covered, relevant
        {"player_id": 2, "gameweek": 5, "total_points": 8, "bonus": 2},   # not covered, relevant
        {"player_id": 3, "gameweek": 5, "total_points": 0, "bonus": 0},   # irrelevant, no bonus
    ])
    # raw coverage counts all 3 rows; relevant coverage only counts the 2 bonus>0 rows
    assert rescore_coverage(df, {(1, 5): 1}) == pytest.approx(1 / 3)
    assert rescore_coverage_relevant(df, {(1, 5): 1}) == pytest.approx(0.5)
