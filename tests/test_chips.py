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


def _skip_bb_fh_wc_kwargs(current_gw: int = 5) -> dict:
    from optimiser.chips import Chip
    # Mark BB/FH/WC as already used THIS half (same half as current_gw) so
    # _chip_uses_remaining reports 0 for them, leaving only TC available.
    return {
        "chips_used": [
            (Chip.BENCH_BOOST, current_gw),
            (Chip.FREE_HIT, current_gw),
            (Chip.WILDCARD, current_gw),
        ],
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


# --- _chip_uses_remaining / chips_used_this_season --------------------------
# Real bugs found 2026-07-28 (user's own squad-trace review: "only one
# wildcard chip was played when we should have 2 of each"). FPL 2025/26+
# gives 1 use of EACH of the 4 chips per half of the season (2 total, no
# carryover) -- confirmed against the Premier League's own 2025/26 changes
# announcement. The project previously modelled only the wildcard this way;
# the other three chips used a naive `not in a set` check that structurally
# capped every chip at one use for the WHOLE season.

@pytest.fixture(autouse=True)
def _fixed_half_boundary(monkeypatch):
    chips._get_wc_half_boundary.cache_clear()
    monkeypatch.setattr(chips, "_get_wc_half_boundary", lambda: 19)
    yield


def test_chip_uses_remaining_unused_chip_has_one_available():
    assert chips._chip_uses_remaining(chips.Chip.WILDCARD, [], current_gw=5) == 1


def test_chip_uses_remaining_used_once_this_half_is_zero():
    used = [(chips.Chip.WILDCARD, 5)]
    assert chips._chip_uses_remaining(chips.Chip.WILDCARD, used, current_gw=10) == 0


def test_chip_uses_remaining_used_in_first_half_available_again_second_half():
    # The exact bug: a chip used once in the first half must be usable AGAIN
    # in the second half -- previously every non-wildcard chip stayed
    # permanently exhausted after one use, all season.
    used = [(chips.Chip.TRIPLE_CAPTAIN, 10)]
    assert chips._chip_uses_remaining(chips.Chip.TRIPLE_CAPTAIN, used, current_gw=25) == 1


def test_chip_uses_remaining_unused_first_half_chip_is_lost_not_banked():
    # FPL rule: no carryover -- NOT using your first-half chip does not give
    # you 2 available in the second half.
    used: list[tuple[chips.Chip, int]] = []
    assert chips._chip_uses_remaining(chips.Chip.BENCH_BOOST, used, current_gw=25) == 1
    # even after never using it in H1, using it once in H2 exhausts H2 too.
    used = [(chips.Chip.BENCH_BOOST, 25)]
    assert chips._chip_uses_remaining(chips.Chip.BENCH_BOOST, used, current_gw=30) == 0


def test_chip_uses_remaining_applies_per_chip_independently():
    used = [(chips.Chip.WILDCARD, 5)]
    assert chips._chip_uses_remaining(chips.Chip.WILDCARD, used, current_gw=10) == 0
    assert chips._chip_uses_remaining(chips.Chip.FREE_HIT, used, current_gw=10) == 1


def test_chips_used_this_season_parses_chip_from_json_details():
    # Real bug: the old version read a `chip_played` column decision_log
    # never had (chip decisions are logged as decision_type="chip" with the
    # chip name inside the JSON `details` string) -- this would KeyError the
    # first time it ran against any real accumulated log.
    log = pd.DataFrame([
        {"gameweek": 5, "decision_type": "chip", "details": '{"chip": "wildcard", "reason": "x"}'},
        {"gameweek": 12, "decision_type": "lineup", "details": '{"captain_id": 1}'},
        {"gameweek": 25, "decision_type": "chip", "details": '{"chip": "3xc", "reason": "y"}'},
    ])
    used = chips.chips_used_this_season(log)
    assert used == [(chips.Chip.WILDCARD, 5), (chips.Chip.TRIPLE_CAPTAIN, 25)]


def test_chips_used_this_season_empty_log_returns_empty_list():
    assert chips.chips_used_this_season(pd.DataFrame()) == []
