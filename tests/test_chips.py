"""P3-5 — scenario-EV chip gating (optimiser/chips.py).

Key property under test: ``_clears_threshold`` replaces "does the mean clear
the bar" with "P(scenario value clears the bar) >= min_probability" whenever
real MC scenarios (P3-1) exist, and falls back to the old point-estimate
rule when they don't (cold start, or the backtest walk-forward, which never
persists samples) -- so a chip that looks good on the mean alone can be
correctly blocked once its real payoff probability is known to be low, while
staying byte-identical to pre-P3-5 behaviour whenever no samples exist.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, ProjectionSample
from optimiser import captaincy, chips


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'chips.db'}")
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


# --- _clears_threshold (pure core) ------------------------------------------

def test_clears_threshold_no_scenario_data_uses_point_estimate():
    assert chips._clears_threshold(7.0, 6.0, pd.Series(dtype=float), 0.6) is True
    assert chips._clears_threshold(5.0, 6.0, pd.Series(dtype=float), 0.6) is False


def test_clears_threshold_scenario_probability_can_block_a_passing_mean():
    # mean of these scenarios is well above the threshold, but only 2/5 draws
    # actually clear it -- a real chip that's a coin-flip, not a sure thing.
    scenarios = pd.Series([7.0, 7.0, -100.0, -100.0, -100.0])
    assert chips._clears_threshold(7.0, 6.0, scenarios, 0.6) is False


def test_clears_threshold_scenario_probability_can_pass():
    scenarios = pd.Series([7.0, 7.0, 7.0, 7.0, 3.0])
    assert chips._clears_threshold(7.0, 6.0, scenarios, 0.6) is True


# --- _bench_player_ids -------------------------------------------------------

def test_bench_player_ids_returns_beyond_top_11():
    projections = pd.DataFrame({
        "player_id": list(range(1, 13)),
        "gameweek": [5] * 12,
        "xpts": list(range(12, 0, -1)),  # player 1 highest, player 12 lowest
    })
    bench = chips._bench_player_ids(list(range(1, 13)), projections, 5)
    assert bench == [12]


def test_bench_player_ids_empty_when_squad_too_small():
    projections = pd.DataFrame({"player_id": [1, 2], "gameweek": [5, 5], "xpts": [5.0, 4.0]})
    assert chips._bench_player_ids([1, 2], projections, 5) == []


# --- _evaluate_triple_captain -------------------------------------------------

def test_evaluate_triple_captain_returns_gain_and_candidate_ids():
    projections = pd.DataFrame({
        "player_id": [1, 2, 3],
        "gameweek": [5, 5, 5],
        "xpts": [10.0, 6.0, 1.0],
    })
    gain, best_id, second_id = chips._evaluate_triple_captain([1, 2, 3], projections, 5)
    assert gain == pytest.approx(4.0)
    assert (best_id, second_id) == (1, 2)


def test_evaluate_triple_captain_fewer_than_two_players_returns_zero():
    projections = pd.DataFrame({"player_id": [1], "gameweek": [5], "xpts": [10.0]})
    assert chips._evaluate_triple_captain([1], projections, 5) == (0.0, None, None)


# --- recommend_chip: TC scenario gate end-to-end ----------------------------

def _minimal_projections(gw: int, best_xpts: float, second_xpts: float) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [1, 2],
        "gameweek": [gw, gw],
        "xpts": [best_xpts, second_xpts],
    })


def _skip_bb_fh_wc_kwargs() -> dict:
    from optimiser.chips import Chip
    return {
        "chips_used": {Chip.BENCH_BOOST, Chip.FREE_HIT, Chip.WILDCARD},
        "squad_age_gws": 0,
    }


def test_recommend_chip_tc_fallback_triggers_without_season():
    projections = _minimal_projections(5, best_xpts=10.0, second_xpts=3.0)  # gain=7 >= 6.0
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip == chips.Chip.TRIPLE_CAPTAIN


def test_recommend_chip_tc_blocked_by_low_payoff_probability(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    # Real per-scenario gain is mostly negative despite a mean that (if it
    # matched these draws) would clear the point threshold -- P(gain>=6) < 0.6.
    rows = (
        _rows_for(1, 5, "2099-00", 0, [20.0, 20.0, 1.0, 1.0, 1.0], created)
        + _rows_for(2, 5, "2099-00", 0, [13.0, 13.0, 1.0, 1.0, 1.0], created)
    )
    _insert(session, rows)
    projections = _minimal_projections(5, best_xpts=10.0, second_xpts=3.0)  # gain=7 >= 6.0
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season="2099-00",
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip is None


def test_recommend_chip_tc_passes_with_high_payoff_probability(session):
    created = pd.Timestamp.now("UTC").to_pydatetime()
    # Real per-scenario gain clears the threshold in 4/5 scenarios -> P=0.8 >= 0.6.
    rows = (
        _rows_for(1, 5, "2099-00", 0, [20.0, 20.0, 20.0, 20.0, 1.0], created)
        + _rows_for(2, 5, "2099-00", 0, [13.0, 13.0, 13.0, 13.0, 1.0], created)
    )
    _insert(session, rows)
    projections = _minimal_projections(5, best_xpts=10.0, second_xpts=3.0)  # gain=7 >= 6.0
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season="2099-00",
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip == chips.Chip.TRIPLE_CAPTAIN


def test_recommend_chip_no_chip_when_nothing_qualifies():
    projections = _minimal_projections(5, best_xpts=5.0, second_xpts=4.9)  # gain=0.1 < 6.0
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip is None
