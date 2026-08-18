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

from config.strategy import CHIPS
from data.models import Base, Gameweek, ProjectionSample
from optimiser import captaincy, chips

# Captured at import time, before the autouse `_fixed_half_boundary` fixture
# below ever monkeypatches `chips._get_wc_half_boundary` -- lets one test
# exercise the REAL (season-scoped) implementation against a real DB.
_real_get_wc_half_boundary = chips._get_wc_half_boundary


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
    # The mean clears the bar, but the gain is actually NEGATIVE in 3/5 draws
    # -- a coin-flip chip, not a sure thing. P(gain >= 0) = 0.4 < 0.6.
    scenarios = pd.Series([7.0, 7.0, -100.0, -100.0, -100.0])
    assert chips._clears_threshold(7.0, 6.0, scenarios, 0.6) is False


def test_clears_threshold_scenario_probability_can_pass():
    scenarios = pd.Series([7.0, 7.0, 7.0, 7.0, 3.0])
    assert chips._clears_threshold(7.0, 6.0, scenarios, 0.6) is True


def test_clears_threshold_keeps_the_point_estimate_bar_when_samples_exist():
    """Regression, 2026-08-18 (§14). The scenario test used to be applied
    INSTEAD of the point estimate, so a sub-threshold mean passed as long as
    the draws were reliably positive. config.strategy has always specified
    "IN ADDITION to the point-estimate thresholds"."""
    reliable = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    assert chips._clears_threshold(2.0, 6.0, reliable, 0.6) is False
    assert chips._clears_threshold(7.0, 6.0, reliable, 0.6) is True


def test_clears_threshold_does_not_reuse_the_threshold_as_a_probability_bar():
    """The specific mis-implementation: ``P(value >= threshold)``. These draws
    never reach the 6.0 threshold individually, but the gain is always
    positive and the mean clears — so the chip must be allowed."""
    always_positive_never_six = pd.Series([1.0, 2.0, 1.5, 2.5, 1.0])
    assert chips._clears_threshold(7.0, 6.0, always_positive_never_six, 0.6) is True


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
    # 2026-07-30: gain is the top captain's own absolute xPts, not the gap
    # to the second-best candidate (see _evaluate_triple_captain docstring).
    projections = pd.DataFrame({
        "player_id": [1, 2, 3],
        "gameweek": [5, 5, 5],
        "xpts": [10.0, 6.0, 1.0],
    })
    gain, best_id, second_id = chips._evaluate_triple_captain([1, 2, 3], projections, 5)
    assert gain == pytest.approx(10.0)
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
    projections = _minimal_projections(5, best_xpts=10.0, second_xpts=3.0)  # gain=10.0 >= 4.0
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip == chips.Chip.TRIPLE_CAPTAIN


def test_recommend_chip_tc_blocked_when_the_gain_is_often_negative(session):
    """2026-08-18 (§14): the scenario bar is ``P(gain >= 0)``, per
    config.strategy. For Triple Captain the gain IS the captain's own points
    (one extra copy of them), so this blocks only a pick whose downside is
    genuinely realised — red cards, own goals — not merely one that blanks.

    Previously this tested ``P(points >= 4.0)``, i.e. the threshold was reused
    as a probability bar. Player 1 scores negative in 3/5 scenarios here, so
    P(gain >= 0) = 0.4 < 0.6.
    """
    created = pd.Timestamp.now("UTC").to_pydatetime()
    rows = _rows_for(1, 5, "2099-00", 0, [20.0, 20.0, -2.0, -2.0, -2.0], created)
    _insert(session, rows)
    projections = _minimal_projections(5, best_xpts=10.0, second_xpts=3.0)  # gain=10.0 >= 4.0
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season="2099-00",
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip is None


def test_recommend_chip_tc_passes_with_high_payoff_probability(session):
    """Player 1 goes negative in only 1/5 scenarios -> P(gain >= 0) = 0.8."""
    created = pd.Timestamp.now("UTC").to_pydatetime()
    rows = _rows_for(1, 5, "2099-00", 0, [20.0, 20.0, 20.0, 20.0, -2.0], created)
    _insert(session, rows)
    projections = _minimal_projections(5, best_xpts=10.0, second_xpts=3.0)  # gain=10.0 >= 4.0
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season="2099-00",
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip == chips.Chip.TRIPLE_CAPTAIN


def test_recommend_chip_point_estimate_bar_still_applies_when_samples_exist(session):
    """Regression, 2026-08-18 (§14), the half that actually loosened things.

    The scenario bar was applied INSTEAD of the point estimate, not in
    addition, so ``point_value`` was discarded entirely whenever samples
    existed. A captain projecting well below ``triple_captain_min_gain`` would
    still fire the chip as long as its draws were reliably non-negative.

    Here the point estimate (1.0) is far under the 4.0 bar while every
    scenario is comfortably positive — the chip must still be declined.
    """
    created = pd.Timestamp.now("UTC").to_pydatetime()
    rows = _rows_for(1, 5, "2099-00", 0, [3.0, 3.0, 3.0, 3.0, 3.0], created)
    _insert(session, rows)
    projections = _minimal_projections(5, best_xpts=1.0, second_xpts=0.5)
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season="2099-00",
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip is None


def _wildcard_scenario() -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """A legal 15 of poor players inside a much better pool, over the full
    wildcard evaluation horizon."""
    rows = []
    pid = 0
    for team in range(1, 16):
        for pos, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            for _ in range(count):
                pid += 1
                rows.append({
                    "id": pid, "web_name": f"p{pid}", "position": pos,
                    "team_id": team, "now_cost": 4.0, "status": "a",
                })
    players = pd.DataFrame(rows)
    squad, per_club = [], {}
    for pos, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        taken = 0
        for r in players[players["position"] == pos].itertuples():
            if per_club.get(r.team_id, 0) < 3:
                squad.append(r.id)
                per_club[r.team_id] = per_club.get(r.team_id, 0) + 1
                taken += 1
            if taken == count:
                break
    owned = set(squad)
    proj = pd.DataFrame([
        {"player_id": r.id, "gameweek": gw,
         "xpts": 1.0 if r.id in owned else 6.0,
         "xpts_var": 1.0, "start_probability": 0.9}
        for gw in range(5, 10) for r in players.itertuples()
    ])
    return players, squad, proj


def test_recommend_chip_wildcard_fires_and_is_evaluated_by_the_executing_optimiser():
    """2026-08-18 (§12). Nothing previously asserted the wildcard was ever
    RECOMMENDED, so `_try_wc`'s body was unexercised by the suite.

    It now evaluates the rebuild with the same
    ``evaluate_transfers(wildcard_active=True)`` call the decision engine uses
    to execute a played wildcard — previously it used ``optimise_squad``, a
    different objective with no bank or purchase-price constraint and no
    multi-period view, so the chip fired on a gain that would never be
    realised.
    """
    from optimiser.chips import Chip

    players, squad, proj = _wildcard_scenario()
    rec = chips.recommend_chip(
        current_gw=5,
        current_squad_ids=squad,
        projections=proj,
        players=players,
        available_budget=200.0,
        free_transfers=1,
        # Only the wildcard is left available this half.
        chips_used=[(Chip.BENCH_BOOST, 5), (Chip.FREE_HIT, 5), (Chip.TRIPLE_CAPTAIN, 5)],
        squad_age_gws=99,
    )
    assert rec.chip == Chip.WILDCARD
    # The gain must be a squad_xpts figure (best XI + captain), not a 15-man
    # sum: 11 starters improving by 5.0 plus a doubled captain, over 5 GWs.
    assert rec.expected_gain == pytest.approx(5 * (11 * 5.0 + 5.0), rel=0.05)


def test_recommend_chip_chip_timing_param_overrides_without_monkeypatch():
    """Simulation-engine entry point: an explicit ``chip_timing=`` override
    must actually change the recommendation, not just be harmless when
    omitted (already covered by the untouched suite). gain=2.0 is below the
    default 4.0 triple_captain_min_gain floor, but clears a lowered one."""
    import dataclasses

    from config.strategy import CHIP_TIMING

    projections = _minimal_projections(5, best_xpts=2.0, second_xpts=1.9)
    lenient = dataclasses.replace(CHIP_TIMING, triple_captain_min_gain=1.0)

    blocked = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(),
    )
    assert blocked.chip is None

    allowed = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        chip_timing=lenient,
        **_skip_bb_fh_wc_kwargs(),
    )
    assert allowed.chip == chips.Chip.TRIPLE_CAPTAIN


def test_recommend_chip_no_chip_when_nothing_qualifies():
    # 2026-07-30: gain is now the captain's own absolute xPts, so this needs
    # a genuinely weak captain projection (not just a close second place)
    # to stay below the (much lower) 4.0 floor.
    projections = _minimal_projections(5, best_xpts=2.0, second_xpts=1.9)  # gain=2.0 < 4.0
    rec = chips.recommend_chip(
        current_gw=5, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(),
    )
    assert rec.chip is None


# --- panic decay / last-resort force (2026-07-30, user's own review: chips ---
# going completely unused all season "is just not acceptable"; "at worst the
# default behaviour is to panic and use the triple captain on the last day
# before the chips reset"). Half boundary fixed at 19 by the autouse
# `_fixed_half_boundary` fixture above.

def test_panic_shrink_is_full_strength_outside_the_window():
    assert chips._panic_shrink(10) == pytest.approx(1.0)  # 9 GWs from expiry
    assert chips._panic_shrink(16) == pytest.approx(1.0)  # exactly at the window edge (3 left)


def test_panic_shrink_decays_linearly_inside_the_window():
    # 2 GWs left: frac=2/3 -> 0.3 + (2/3)*0.7
    assert chips._panic_shrink(17) == pytest.approx(0.3 + (2 / 3) * 0.7)
    # 0 GWs left (the expiry GW itself) -> the floor value exactly
    assert chips._panic_shrink(19) == pytest.approx(0.3)


def test_current_half_expiry_gw_first_vs_second_half():
    assert chips._current_half_expiry_gw(10) == 19
    assert chips._current_half_expiry_gw(19) == 19


def test_recommend_chip_tc_blocked_far_from_expiry_but_passes_near_it_via_decay():
    # A captain projected at 3.5 xPts is below the normal 4.0 floor and
    # stays blocked far from expiry, but the SAME projection clears the
    # panic-shrunk threshold (4.0*0.5333=2.13 at 1 GW out) once the half is
    # nearly over -- proving the decay itself, not just the final hard
    # force, lets a real marginal captain through.
    far_projections = _minimal_projections(10, best_xpts=3.5, second_xpts=3.4)
    far = chips.recommend_chip(
        current_gw=10, current_squad_ids=[1, 2], projections=far_projections,
        players=pd.DataFrame(), available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(current_gw=10),
    )
    assert far.chip is None

    near_projections = _minimal_projections(18, best_xpts=3.5, second_xpts=3.4)
    near = chips.recommend_chip(
        current_gw=18, current_squad_ids=[1, 2], projections=near_projections,
        players=pd.DataFrame(), available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(current_gw=18),
    )
    assert near.chip == chips.Chip.TRIPLE_CAPTAIN
    # decay let the normal TC gate fire here, not the last-resort force
    assert "Panic" not in near.reason


def test_recommend_chip_panic_forces_tc_on_expiry_gw_when_nothing_else_clears():
    # A 1.0 xPts captain is below even the panic-shrunk threshold
    # (4.0*0.3=1.2) at the literal expiry GW -- only the final "use it or
    # lose it" force should fire.
    projections = _minimal_projections(19, best_xpts=1.0, second_xpts=0.5)
    rec = chips.recommend_chip(
        current_gw=19, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(current_gw=19),
    )
    assert rec.chip == chips.Chip.TRIPLE_CAPTAIN
    assert "Forced before expiry" in rec.reason


def test_recommend_chip_panic_forces_tc_one_gw_before_expiry_too():
    # Robustness margin: the force triggers on the final TWO gameweeks of
    # the half (not just the literal last one), so a single skipped/missing
    # decision point right at the boundary can't cost the whole half's chip.
    # 1.0 xPts is below the panic-shrunk threshold at 1 GW out (4.0*0.5333=2.13).
    projections = _minimal_projections(18, best_xpts=1.0, second_xpts=0.9)
    rec = chips.recommend_chip(
        current_gw=18, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(current_gw=18),
    )
    assert rec.chip == chips.Chip.TRIPLE_CAPTAIN
    assert "Forced before expiry" in rec.reason


def test_recommend_chip_no_panic_force_away_from_expiry():
    projections = _minimal_projections(10, best_xpts=1.0, second_xpts=0.9)  # weak, below 4.0 floor
    rec = chips.recommend_chip(
        current_gw=10, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        **_skip_bb_fh_wc_kwargs(current_gw=10),
    )
    assert rec.chip is None


# --- DGW-triggered Free Hit (2026-07-30, user's own review: "the free hit --
# is usually handy during double game weeks where it is not worth triple
# captaining"). Previously Free Hit only ever triggered on a BGW rationale.

def _dgw_free_hit_pool() -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    positions = ["GKP"] * 4 + ["DEF"] * 8 + ["MID"] * 8 + ["FWD"] * 5
    rows = []
    for i, position in enumerate(positions):
        pid = i + 1
        rows.append({
            "id": pid, "position": position, "now_cost": 4.5,
            "team_id": 1 + (i % 8), "status": "a", "start_probability": 0.9,
            "web_name": f"p{pid}",
        })
    players = pd.DataFrame(rows)

    # A well-spread, formation-valid 15 (2 GKP, 5 DEF, 5 MID, 3 FWD) with a
    # big DGW-week haul; everyone else (including the "current" squad) blanks.
    great_ids = [1, 2, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 21, 22, 23]
    gw = 10
    projections = pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "xpts": 10.0 if pid in great_ids else 2.0}
        for pid in players["id"]
    ])
    current_squad_ids = [pid for pid in players["id"] if pid not in great_ids][:10]
    return players, projections, current_squad_ids


def test_recommend_chip_free_hit_triggers_on_dgw_without_bgw_blanks():
    players, projections, current_squad_ids = _dgw_free_hit_pool()
    rec = chips.recommend_chip(
        current_gw=10, current_squad_ids=current_squad_ids, projections=projections,
        players=players, available_budget=100.0, free_transfers=1, season=None,
        chips_used=[(chips.Chip.WILDCARD, 10)], squad_age_gws=0,
        dgw_gws={10}, bgw_affected_count=0,
    )
    assert rec.chip == chips.Chip.FREE_HIT
    assert "DGW" in rec.reason


def test_recommend_chip_free_hit_can_trigger_on_merit_without_a_dgw_or_bgw():
    """Changed 2026-08-18. This used to assert that Free Hit CANNOT fire
    without a double or blank gameweek. That gate was wrong: doubles and blanks
    only arise from postponements, so a half can contain none at all — and a
    half's unused chips are destroyed at the boundary rather than carried over.
    Requiring a structural event to unlock the chip therefore guaranteed it was
    wasted in exactly those halves.

    A Free Hit's value is the one-week gain of the best legal XI over the
    current one, which is measurable every gameweek. A blank or a double simply
    makes that number large, which is what the threshold is for. Here the pool
    offers a big enough one-week upgrade to clear it on merit alone.
    """
    players, projections, current_squad_ids = _dgw_free_hit_pool()
    rec = chips.recommend_chip(
        current_gw=10, current_squad_ids=current_squad_ids, projections=projections,
        players=players, available_budget=100.0, free_transfers=1, season=None,
        chips_used=[(chips.Chip.WILDCARD, 10)], squad_age_gws=0,
        dgw_gws=set(), bgw_affected_count=0,
    )
    assert rec.chip == chips.Chip.FREE_HIT


def test_recommend_chip_free_hit_still_declines_a_weak_week():
    """The threshold, not a structural gate, is what holds the chip back now —
    so it must still hold it back when the upgrade is not worth it."""
    players, projections, current_squad_ids = _dgw_free_hit_pool()
    # Flatten the pool: nothing to gain from a one-week rebuild.
    projections = projections.copy()
    projections["xpts"] = 2.0
    rec = chips.recommend_chip(
        current_gw=10, current_squad_ids=current_squad_ids, projections=projections,
        players=players, available_budget=100.0, free_transfers=1, season=None,
        chips_used=[(chips.Chip.WILDCARD, 10)], squad_age_gws=0,
        dgw_gws=set(), bgw_affected_count=0,
    )
    assert rec.chip != chips.Chip.FREE_HIT


# --- must-play arithmetic (2026-08-18) ---------------------------------------

def test_must_play_when_chips_outnumber_remaining_gameweeks():
    """Only one chip may be played per gameweek and a half's leftovers are
    destroyed at the boundary, so once the chips outnumber the slots, declining
    today mathematically guarantees one is binned."""
    # GW17, first half expires at GW19 -> slots 17, 18, 19 = 3.
    used_all_but_three = [(chips.Chip.WILDCARD, 5)]
    assert chips.must_play_a_chip_now(used_all_but_three, current_gw=17) is True   # 3 left, 3 slots
    # With only one chip left and three slots there is still real slack.
    only_tc_left = [
        (chips.Chip.WILDCARD, 5), (chips.Chip.FREE_HIT, 6), (chips.Chip.BENCH_BOOST, 7),
    ]
    assert chips.must_play_a_chip_now(only_tc_left, current_gw=16) is False
    # ...but never let the final two gameweeks pass holding one.
    assert chips.must_play_a_chip_now(only_tc_left, current_gw=18) is True
    # Nothing left to play is not a must-play.
    all_used = [
        (chips.Chip.WILDCARD, 5), (chips.Chip.FREE_HIT, 6),
        (chips.Chip.BENCH_BOOST, 7), (chips.Chip.TRIPLE_CAPTAIN, 8),
    ]
    assert chips.must_play_a_chip_now(all_used, current_gw=18) is False


def test_chips_available_this_half_counts_only_unused():
    used = [(chips.Chip.WILDCARD, 5), (chips.Chip.BENCH_BOOST, 7)]
    available = chips.chips_available_this_half(used, current_gw=10)
    assert set(available) == {chips.Chip.FREE_HIT, chips.Chip.TRIPLE_CAPTAIN}


# --- TC vs. a coming DGW / an active DGW (2026-07-30) ------------------------
# User's own review: TC's own EV is basically always positive, so the real
# remaining question is scarcity (1 use per half) -- is this week worth
# spending it, versus waiting for a probably-bigger DGW captain, and on an
# ACTUAL DGW week, does TC crowd out Bench Boost/Free Hit's shot at the
# whole squad's double-fixture value.

def test_recommend_chip_tc_holds_back_when_a_dgw_is_visible_ahead():
    # 9.0 clears the normal 4.0 floor easily, but not the raised bar
    # (4.0*2.5=10.0) used while a DGW is visible ahead but hasn't arrived.
    projections = _minimal_projections(10, best_xpts=9.0, second_xpts=8.0)
    rec = chips.recommend_chip(
        current_gw=10, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        dgw_gws={13}, bgw_affected_count=0,
        **_skip_bb_fh_wc_kwargs(current_gw=10),
    )
    assert rec.chip is None


def test_recommend_chip_tc_fires_normally_with_no_dgw_visible():
    # Same 9.0 captain, no DGW anywhere visible -- normal (unraised) floor applies.
    projections = _minimal_projections(10, best_xpts=9.0, second_xpts=8.0)
    rec = chips.recommend_chip(
        current_gw=10, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None,
        dgw_gws=set(), bgw_affected_count=0,
        **_skip_bb_fh_wc_kwargs(current_gw=10),
    )
    assert rec.chip == chips.Chip.TRIPLE_CAPTAIN


def test_recommend_chip_bench_boost_preempts_tc_on_an_active_dgw_week():
    # Both TC (captain xPts=9.0 >= 4.0) and BB (bench_xpts=25.0 >= 20.0)
    # would independently clear their own thresholds this week -- on an
    # ACTIVE DGW week, BB/FH get first refusal so TC can't routinely crowd
    # out the whole-squad DGW play just because its own bar is now so easy
    # to clear.
    projections = _minimal_projections(10, best_xpts=9.0, second_xpts=8.0)
    rec = chips.recommend_chip(
        current_gw=10, current_squad_ids=[1, 2], projections=projections, players=pd.DataFrame(),
        available_budget=100.0, free_transfers=1, season=None, bench_xpts=25.0,
        dgw_gws={10}, bgw_affected_count=0,
        chips_used=[(chips.Chip.FREE_HIT, 10), (chips.Chip.WILDCARD, 10)], squad_age_gws=0,
    )
    assert rec.chip == chips.Chip.BENCH_BOOST


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
    monkeypatch.setattr(chips, "_get_wc_half_boundary", lambda season=None: 19)
    yield


def test_get_wc_half_boundary_is_scoped_by_season_not_global(tmp_path, monkeypatch):
    # Real bug found 2026-07-30 (user's own review: TC "should NEVER be left
    # unplayed in both halves of the season" -- it turned out to never fire
    # even at full panic strength). This query used to have NO season
    # filter, so with multiple seasons' gameweeks all living in the same
    # table (this project's own reality -- 6 backfilled seasons, 227 total
    # rows), the boundary came out as 113 (227 // 2) instead of 19. Every
    # 2025-26 backtest gameweek (6-38) is <= 113, so the code believed it
    # was ALWAYS still the first half and could never see a real boundary
    # crossing -- or an expiry -- within the season at all.
    engine = create_engine(f"sqlite:///{tmp_path / 'half_scoped.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(chips, "get_session", lambda: Local())
    s = Local()
    now = pd.Timestamp.now("UTC").to_pydatetime()
    for season, n_gws in [("2024-25", 38), ("2025-26", 38), ("2026-27", 38)]:
        s.add_all([
            Gameweek(id=gw, season=season, name=f"GW{gw}", deadline_time=now)
            for gw in range(1, n_gws + 1)
        ])
    s.commit()
    s.close()

    _real_get_wc_half_boundary.cache_clear()
    try:
        assert _real_get_wc_half_boundary(season="2025-26") == 19
        # P3.10 (2026-08-16): season=None used to run the SAME unscoped
        # COUNT(*) that caused the bug above, returning (38 * 3) // 2 = 57
        # here and 113 against the real 6-season DB. There is no safe query
        # without a season, so it now falls back to the configured first-half
        # deadline instead of a number derived from unrelated seasons.
        assert _real_get_wc_half_boundary(season=None) == CHIPS.wildcard_first_half_deadline_gw
    finally:
        _real_get_wc_half_boundary.cache_clear()


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
