"""P1 regression gate — free-transfer banking, wildcard, hits, bench value.

Every defect covered here was live for months behind a green suite, because
the existing transfer tests only ever exercised ``free_transfers=1``,
single-transfer, no-chip scenarios, and ``scripts/backtest.py`` bypassed the
broken paths entirely by calling ``optimise_squad`` directly for wildcards.

See docs/superpowers/plans/decision-engine-recovery-plan.md P1.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from config.strategy import OPTIMISER, TRANSFERS
from optimiser.transfers import evaluate_transfers, roll_forward_free_transfers

# Isolate the mechanic under test from the other frictions in the objective
# (a 1.5pt switching cost and a terminal value on banked transfers both blur
# small gains on their own -- they have their own tests in test_p3_objective).
_NO_FRICTION = dataclasses.replace(
    TRANSFERS, transfer_switching_cost=0.0, ft_terminal_value=0.0
)


def _pool() -> tuple[pd.DataFrame, list[int]]:
    """A legal 15 (<= 3 per club) drawn from a 15-club pool, so there is
    always real room to transfer without tripping the club constraint."""
    rows = []
    pid = 0
    for team in range(1, 16):
        for pos, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            for k in range(count):
                pid += 1
                rows.append({
                    "id": pid, "web_name": f"p{pid}", "position": pos,
                    "team_id": team, "now_cost": 4.0 + k * 0.5, "status": "a",
                })
    players = pd.DataFrame(rows)
    squad: list[int] = []
    per_club: dict[int, int] = {}
    for pos, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        taken = 0
        for r in players[players["position"] == pos].itertuples():
            if per_club.get(r.team_id, 0) < 3:
                squad.append(r.id)
                per_club[r.team_id] = per_club.get(r.team_id, 0) + 1
                taken += 1
            if taken == count:
                break
    assert len(squad) == 15
    return players, squad


def _flat_projections(players: pd.DataFrame, squad: list[int], gws=(1, 2, 3)) -> pd.DataFrame:
    """Owned players flat at 4.0; everyone else deliberately worse, so any
    transfer the model makes has to be justified by a test's own override."""
    return pd.DataFrame([
        {
            "player_id": r.id, "gameweek": gw,
            "xpts": 4.0 if r.id in set(squad) else 1.0,
            "xpts_var": 1.0, "start_probability": 0.9,
        }
        for gw in gws for r in players.itertuples()
    ])


# --- P1.2 free-transfer roll-forward --------------------------------------


def test_roll_forward_banks_an_unused_transfer():
    """The headline bug: the live engine stored `max(0, ft - made)`, so an
    unused transfer never became two and banking to 5 was unreachable."""
    assert roll_forward_free_transfers(1, 0) == 2
    assert roll_forward_free_transfers(2, 0) == 3


def test_roll_forward_never_returns_zero():
    """`ft` has lowBound=1 in the ILP, so a stored 0 made the model
    Infeasible -- which evaluate_transfers reports as "no transfers". That is
    how the live bot locked itself out of transferring permanently after its
    first transfer."""
    assert roll_forward_free_transfers(1, 1) == TRANSFERS.free_transfers_per_gw
    # hits: more transfers than allowance, paid in points not in allowance
    assert roll_forward_free_transfers(1, 3) == TRANSFERS.free_transfers_per_gw


def test_roll_forward_caps_at_the_banking_limit():
    assert roll_forward_free_transfers(5, 0) == TRANSFERS.max_banked_free_transfers
    assert roll_forward_free_transfers(
        TRANSFERS.max_banked_free_transfers, 0
    ) == TRANSFERS.max_banked_free_transfers


def test_roll_forward_resets_after_a_wildcard():
    assert roll_forward_free_transfers(4, 11, wildcard_played=True) == 1


def test_roll_forward_free_hit_transfers_do_not_spend_the_allowance():
    """A Free Hit squad is reverted, so its transfers never counted -- but
    the weekly allowance still accrues (the backtest used to `pass` here,
    keeping the count flat)."""
    assert roll_forward_free_transfers(2, 15, free_hit_played=True) == 3


def test_roll_forward_banks_to_the_cap_over_consecutive_quiet_weeks():
    """End to end: five quiet gameweeks must actually reach the cap."""
    ft = 1
    seen = [ft]
    for _ in range(6):
        ft = roll_forward_free_transfers(ft, 0)
        seen.append(ft)
    assert seen == [1, 2, 3, 4, 5, 5, 5]


# --- P1.2 the ILP no longer dies silently on a bad count -------------------


def test_zero_free_transfers_clamps_instead_of_returning_an_empty_plan(caplog):
    players, squad = _pool()
    proj = _flat_projections(players, squad)
    proj.loc[proj["player_id"] == 200, "xpts"] = 40.0  # an obvious upgrade

    with caplog.at_level("WARNING"):
        plan = evaluate_transfers(
            squad, proj, players, free_transfers=0, available_budget=200.0,
            transfer_rules=_NO_FRICTION,
        )
    assert plan.transfers_in, "a 0 free-transfer count must not silently mean 'no transfers'"
    assert "below the weekly allowance" in caplog.text


# --- P1.3 wildcard --------------------------------------------------------


def test_wildcard_produces_a_full_rebuild_not_an_empty_plan():
    """`ft[0] == 15` against the old `upBound=5` made the model Infeasible,
    so a played wildcard made ZERO transfers while still being recorded as
    used -- one of two season wildcards burned for nothing. The backtest
    never caught it because it rebuilds wildcards via optimise_squad."""
    players, squad = _pool()
    proj = _flat_projections(players, squad)
    # make every non-owned player clearly better, so a rebuild is obviously right
    proj.loc[~proj["player_id"].isin(squad), "xpts"] = 9.0

    plan = evaluate_transfers(
        squad, proj, players, free_transfers=1, available_budget=200.0,
        wildcard_active=True, transfer_rules=_NO_FRICTION,
    )
    assert len(plan.transfers_in) > 1
    assert plan.hits_taken == 0, "a wildcard is free -- it must never book hits"


# --- P1.4 hits ------------------------------------------------------------


def _one_week_spike(spike: float) -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """15 owned (flat 4.0) plus two extra FWD candidates whose advantage
    exists ONLY in the first horizon gameweek. That matters: with a
    persistent advantage the model can (correctly) take one transfer now and
    another next week off the banked allowance, so no hit is ever needed --
    a hit is only rational when the gain is time-critical."""
    rows = []
    for i, pos in enumerate(["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3):
        rows.append({
            "id": i + 1, "position": pos, "now_cost": 5.0,
            "team_id": 1 + (i % 5), "status": "a", "web_name": f"p{i + 1}",
        })
    rows.append({"id": 16, "position": "FWD", "now_cost": 5.0, "team_id": 6,
                 "status": "a", "web_name": "c1"})
    rows.append({"id": 17, "position": "FWD", "now_cost": 5.0, "team_id": 7,
                 "status": "a", "web_name": "c2"})
    players = pd.DataFrame(rows)
    squad = list(range(1, 16))
    gws = [10, 11, 12]
    rows_proj = [
        {"player_id": p, "gameweek": g, "xpts": 4.0, "start_probability": 0.9}
        for p in squad for g in gws
    ]
    rows_proj += [
        {"player_id": c, "gameweek": g,
         "xpts": 4.0 + (spike if g == 10 else 0.0), "start_probability": 0.9}
        for c in (16, 17) for g in gws
    ]
    return players, squad, pd.DataFrame(rows_proj)


def test_hit_is_taken_when_a_time_critical_gain_clears_its_cost():
    """`ft[w+1] <= ft[w] - n_trans[w] + 1` with `ft[w+1] >= 1` forced
    n_trans <= ft every week, so hits were structurally impossible -- proven
    against three candidates worth +29 xPts/GW each, which produced one
    transfer and zero hits."""
    players, squad, proj = _one_week_spike(10.0)
    plan = evaluate_transfers(
        squad, proj, players, free_transfers=1, available_budget=100.0,
        transfer_rules=_NO_FRICTION,
    )
    assert len(plan.transfers_in) == 2
    assert plan.hits_taken == 1


def test_hit_is_declined_when_the_gain_does_not_clear_its_cost():
    """The other half: the hit must be charged ONCE. It used to sit inside
    the per-player loop, costing `4 * len(pid_list)` (~900+ points), which
    would have made hits impossible again the moment they became legal."""
    players, squad, proj = _one_week_spike(3.0)  # under the 4pt hit cost
    plan = evaluate_transfers(
        squad, proj, players, free_transfers=1, available_budget=100.0,
        transfer_rules=_NO_FRICTION,
    )
    assert plan.hits_taken == 0
    assert len(plan.transfers_in) == 1


def test_hit_threshold_sits_at_the_real_hit_cost():
    """Pins the boundary: just under `hit_cost_points` declines, just over
    accepts. A mis-scaled hit term moves this, which is exactly the failure
    that hid for months."""
    cost = abs(TRANSFERS.hit_cost_points)
    players, squad, under = _one_week_spike(cost - 0.1)
    _, _, over = _one_week_spike(cost + 0.1)
    kwargs = {"free_transfers": 1, "available_budget": 100.0, "transfer_rules": _NO_FRICTION}
    assert evaluate_transfers(squad, under, players, **kwargs).hits_taken == 0
    assert evaluate_transfers(squad, over, players, **kwargs).hits_taken == 1


def _persistent_gain(gain: float, gws=(10, 11, 12)) -> pd.DataFrame:
    _, squad, _ = _one_week_spike(0.0)
    rows = [
        {"player_id": p, "gameweek": g, "xpts": 4.0, "start_probability": 0.9}
        for p in squad for g in gws
    ]
    rows += [
        {"player_id": c, "gameweek": g, "xpts": 4.0 + gain, "start_probability": 0.9}
        for c in (16, 17) for g in gws
    ]
    return pd.DataFrame(rows)


def test_persistent_gain_is_banked_rather_than_hit_for():
    """Behavioural counterpart: when the advantage PERSISTS, deferring the
    second move to next week's banked transfer costs only one gameweek of
    that advantage, so a hit is only rational once the per-gameweek gain
    exceeds the hit cost itself. Below that line the model must bank.

    This is the multi-period reasoning the live path could not do at all
    while `get_latest_projections` handed it a single gameweek."""
    players, squad, _ = _one_week_spike(0.0)
    kwargs = {"free_transfers": 1, "available_budget": 100.0, "transfer_rules": _NO_FRICTION}

    below = evaluate_transfers(squad, _persistent_gain(3.5), players, **kwargs)
    assert below.hits_taken == 0
    assert len(below.transfers_in) == 1

    # above the line the hit becomes correct -- the same mechanism, not a
    # blanket aversion to hits
    above = evaluate_transfers(squad, _persistent_gain(4.5), players, **kwargs)
    assert above.hits_taken == 1


# --- P1.7 bench value in the transfer objective ---------------------------


def _bench_gk_choice() -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """Owned squad with a worthless backup GK, plus a much better backup GK
    available at the same price. Only ONE goalkeeper can start, so the
    upgrade is purely a bench decision."""
    rows = []
    for i, pos in enumerate(["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3):
        rows.append({
            "id": i + 1, "position": pos, "now_cost": 5.0,
            "team_id": 1 + (i % 5), "status": "a", "web_name": f"p{i + 1}",
        })
    rows.append({"id": 16, "position": "GKP", "now_cost": 5.0, "team_id": 9,
                 "status": "a", "web_name": "goodbackup"})
    players = pd.DataFrame(rows)
    squad = list(range(1, 16))
    gws = [10, 11, 12]
    proj = []
    for g in gws:
        for p in squad:
            # p1 is the (clearly better) starting keeper, p2 the dead-weight backup
            xpts = 8.0 if p == 1 else (0.0 if p == 2 else 4.0)
            proj.append({"player_id": p, "gameweek": g, "xpts": xpts, "start_probability": 0.9})
        proj.append({"player_id": 16, "gameweek": g, "xpts": 6.0, "start_probability": 0.9})
    return players, squad, pd.DataFrame(proj)


def test_bench_value_weight_zero_leaves_the_dead_weight_backup_gk():
    """Pre-P1.7 behaviour: the transfer objective scored only `starting` and
    `captain`, so a bench player was worth exactly nothing and every
    in-season transfer eroded the bench optimise_squad had paid for."""
    players, squad, proj = _bench_gk_choice()
    plan = evaluate_transfers(
        squad, proj, players, free_transfers=1, available_budget=100.0,
        config=dataclasses.replace(OPTIMISER, bench_value_weight=0.0),
        transfer_rules=dataclasses.replace(TRANSFERS, ft_terminal_value=0.0),
    )
    assert not any(t["player_id"] == 16 for t in plan.transfers_in)


def test_bench_value_weight_upgrades_the_backup_gk():
    players, squad, proj = _bench_gk_choice()
    plan = evaluate_transfers(
        squad, proj, players, free_transfers=1, available_budget=100.0,
        config=dataclasses.replace(OPTIMISER, bench_value_weight=0.15),
        transfer_rules=dataclasses.replace(TRANSFERS, ft_terminal_value=0.0),
    )
    assert any(t["player_id"] == 16 for t in plan.transfers_in)


# --- P1.1 the multi-gameweek frame the above all depends on ---------------


def test_multi_gw_frame_changes_the_decision_a_single_gw_frame_would_make():
    """`get_latest_projections` returned exactly one gameweek regardless of
    the configured horizon, so the live ILP ran at H=1.

    Uses the REAL frictions deliberately: a +0.6/GW upgrade is worth 0.6 over
    one gameweek (under the 1.5pt switching cost -- correctly declined) but
    1.8 over the planning horizon (worth doing). Same player, same squad,
    opposite decisions -- which is what the live path was silently getting
    wrong every week."""
    players, squad, _ = _one_week_spike(0.0)
    full = _persistent_gain(0.6)
    single = full[full["gameweek"] == 10]

    single_plan = evaluate_transfers(
        squad, single, players, free_transfers=1, available_budget=100.0
    )
    full_plan = evaluate_transfers(
        squad, full, players, free_transfers=1, available_budget=100.0
    )

    assert not single_plan.transfers_in, "one gameweek of gain cannot clear the switching cost"
    assert full_plan.transfers_in, "over the real horizon the same upgrade is worth making"


def test_get_latest_projections_horizon_spans_the_planning_window(monkeypatch):
    """Guards the actual defect: the query was `WHERE pp.gameweek = :gw`."""
    from projection import pipeline

    captured: dict[str, object] = {}

    class _FakeDB:
        bind = object()

        def close(self) -> None:
            pass

    def _fake_read_sql(query, _bind, params):
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(pipeline, "get_session", lambda: _FakeDB())
    monkeypatch.setattr(pipeline, "_get_current_and_next_gw", lambda: (4, 5))
    monkeypatch.setattr(pipeline.pd, "read_sql", _fake_read_sql)

    pipeline.get_latest_projections(horizon=3)
    assert captured["params"] == {"gw": 5, "last_gw": 7}

    pipeline.get_latest_projections()  # default stays single-gameweek
    assert captured["params"] == {"gw": 5, "last_gw": 5}


@pytest.mark.parametrize("horizon", [0, -1])
def test_get_latest_projections_degrades_on_a_nonsense_horizon(monkeypatch, horizon):
    from projection import pipeline

    captured: dict[str, object] = {}

    class _FakeDB:
        bind = object()

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "get_session", lambda: _FakeDB())
    monkeypatch.setattr(pipeline, "_get_current_and_next_gw", lambda: (4, 5))
    monkeypatch.setattr(
        pipeline.pd, "read_sql",
        lambda q, b, params: (captured.__setitem__("params", params), pd.DataFrame())[1],
    )

    pipeline.get_latest_projections(horizon=horizon)
    assert captured["params"] == {"gw": 5, "last_gw": 5}
