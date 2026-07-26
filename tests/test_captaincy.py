"""P3-4 — scenario-based captaincy (optimiser/captaincy.py).

The key property under test: the additive own-variance approximation
(P3-3's per-player ``mu*xpts_var`` term) cannot distinguish a candidate whose
fixture-mates are strongly positively correlated with them from one who
isn't — both look identical if their OWN variance is the same. Real joint
MC samples (P3-1) can. ``pick_captain`` degrades exactly to the additive
approximation when no real samples exist for a candidate (cold start,
backtest — which never persists samples).
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, ProjectionSample
from optimiser import captaincy


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'captaincy.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(captaincy, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _insert_samples(session, rows: list[dict]) -> None:
    session.add_all([ProjectionSample(**row) for row in rows])
    session.commit()


# --- pick_captain (pure core) -----------------------------------------------

def test_pick_captain_balanced_mode_is_plain_mean_argmax():
    xpts = {1: 5.0, 2: 8.0, 3: 3.0}
    var = {1: 100.0, 2: 0.1, 3: 0.1}  # variance would flip the pick if used
    assert captaincy.pick_captain([1, 2, 3], xpts, var, mu=0.0, fixture_groups=[]) == 2


def test_pick_captain_no_sample_data_matches_additive_variance_approx():
    xpts = {1: 5.0, 2: 4.9}
    var = {1: 1.0, 2: 10.0}
    # aggressive: mu>0 rewards variance -> additive approx (xpts + mu*var)
    # should flip the pick towards player 2 despite lower mean
    assert captaincy.pick_captain([1, 2], xpts, var, mu=0.5, fixture_groups=[]) == 2
    # safe: mu<0 penalises variance -> stick with player 1
    assert captaincy.pick_captain([1, 2], xpts, var, mu=-0.5, fixture_groups=[]) == 1


def test_pick_captain_solo_group_captures_doubling_nonlinearity():
    # One starting-XI player in this match (no teammate/opponent in our XI) --
    # Var(2X) = 4*Var(X), not the naive "+1x" the additive approx would give.
    group = pd.DataFrame({1: [0.0, 10.0]})  # mean 5, var(ddof=1) = 50
    xpts = {1: 5.0, 2: 5.0}
    var = {1: 50.0, 2: 50.0}  # candidate 2 has NO group data -> additive fallback
    aggressive = captaincy.pick_captain([1, 2], xpts, var, mu=1.0, fixture_groups=[group])
    # true delta for 1 is +3*50=150 (4x - 1x already counted); for 2 it's +50 (additive)
    assert aggressive == 1
    safe = captaincy.pick_captain([1, 2], xpts, var, mu=-1.0, fixture_groups=[group])
    assert safe == 2


def test_pick_captain_captures_true_covariance_beyond_additive_approx():
    # Players 1 and 2 share a fixture and are PERFECTLY correlated (identical
    # draws) -- doubling either amplifies team variance far more than the
    # additive per-player term (which sees only their own variance) could
    # know. Player 3 has the SAME declared own-variance but no real sample
    # data (fallback), so the additive approximation treats all three
    # identically -- only the real joint data breaks the tie correctly.
    group = pd.DataFrame({1: [0.0, 10.0], 2: [0.0, 10.0]})  # each var(ddof=1)=50
    xpts = {1: 5.0, 2: 5.0, 3: 5.0}
    var = {1: 50.0, 2: 50.0, 3: 50.0}

    aggressive = captaincy.pick_captain([1, 2, 3], xpts, var, mu=1.0, fixture_groups=[group])
    assert aggressive in (1, 2)  # correlated pair beats the independent candidate

    safe = captaincy.pick_captain([1, 2, 3], xpts, var, mu=-1.0, fixture_groups=[group])
    assert safe == 3  # independent (lower true variance) candidate wins under risk-aversion


def test_pick_captain_empty_candidates_raises():
    with pytest.raises(ValueError, match="candidate_ids"):
        captaincy.pick_captain([], {}, {}, mu=1.0, fixture_groups=[])


def test_pick_captain_group_columns_not_in_candidates_are_ignored():
    # A fixture group may contain a player who isn't one of OUR candidates
    # (e.g. an opponent not in our starting XI) -- must not leak into the sum.
    group = pd.DataFrame({1: [0.0, 10.0], 99: [100.0, 100.0]})
    xpts = {1: 5.0}
    var = {1: 50.0}
    assert captaincy.pick_captain([1], xpts, var, mu=1.0, fixture_groups=[group]) == 1


# --- load_fixture_groups (DB-backed) ----------------------------------------

def test_load_fixture_groups_splits_by_disjoint_scenario_range(session):
    created = pd.Timestamp.now('UTC').to_pydatetime()
    rows = []
    for pid in (1, 2):  # fixture A: scenario_id [0, 3)
        for s in range(3):
            rows.append({"player_id": pid, "gameweek": 5, "season": "2099-00",
                         "scenario_id": s, "xpts": float(pid * 10 + s), "created_at": created})
    for pid in (3,):  # fixture B: scenario_id [3, 6)
        for s in range(3, 6):
            rows.append({"player_id": pid, "gameweek": 5, "season": "2099-00",
                         "scenario_id": s, "xpts": float(pid * 10 + s), "created_at": created})
    _insert_samples(session, rows)

    groups = captaincy.load_fixture_groups("2099-00", 5, [1, 2, 3])
    assert len(groups) == 2
    sizes = sorted(len(g.columns) for g in groups)
    assert sizes == [1, 2]
    two_col = next(g for g in groups if len(g.columns) == 2)
    assert set(two_col.columns) == {1, 2}


def test_load_fixture_groups_uses_latest_run_only(session):
    old = pd.Timestamp.now('UTC').to_pydatetime() - pd.Timedelta(hours=1)
    new = pd.Timestamp.now('UTC').to_pydatetime()
    rows = [
        {"player_id": 1, "gameweek": 5, "season": "2099-00", "scenario_id": 0,
         "xpts": 999.0, "created_at": old},
        {"player_id": 1, "gameweek": 5, "season": "2099-00", "scenario_id": 0,
         "xpts": 7.0, "created_at": new},
    ]
    _insert_samples(session, rows)
    groups = captaincy.load_fixture_groups("2099-00", 5, [1])
    assert len(groups) == 1
    assert groups[0][1].tolist() == [7.0]


def test_load_fixture_groups_no_rows_returns_empty(session):
    assert captaincy.load_fixture_groups("2099-00", 5, [1, 2]) == []


def test_load_fixture_groups_empty_player_ids_returns_empty():
    assert captaincy.load_fixture_groups("2099-00", 5, []) == []


# --- scenario_based_captain (orchestrator) ----------------------------------

def test_scenario_based_captain_balanced_mode_skips_db_entirely(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("load_fixture_groups must not be called at mu=0")

    monkeypatch.setattr(captaincy, "load_fixture_groups", _boom)
    xpts = {1: 5.0, 2: 8.0}
    var = {1: 0.0, 2: 0.0}
    result = captaincy.scenario_based_captain("2099-00", 5, [1, 2], xpts, var, mu=0.0)
    assert result == 2


def test_scenario_based_captain_end_to_end(session):
    created = pd.Timestamp.now('UTC').to_pydatetime()
    rows = []
    for pid in (1, 2):
        for s, val in enumerate([0.0, 10.0]):
            rows.append({"player_id": pid, "gameweek": 5, "season": "2099-00",
                         "scenario_id": s, "xpts": val, "created_at": created})
    _insert_samples(session, rows)
    xpts = {1: 5.0, 2: 5.0}
    var = {1: 50.0, 2: 50.0}
    captain = captaincy.scenario_based_captain("2099-00", 5, [1, 2], xpts, var, mu=1.0)
    assert captain in (1, 2)


def test_scenario_based_captain_empty_candidates_raises():
    with pytest.raises(ValueError, match="candidate_ids"):
        captaincy.scenario_based_captain("2099-00", 5, [], {}, {}, mu=1.0)
