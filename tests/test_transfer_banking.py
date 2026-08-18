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


def test_roll_forward_keeps_banked_transfers_through_a_wildcard():
    """Regression, 2026-08-18 (engine review §10).

    This used to assert ``== 1`` — that a wildcard RESET the allowance. That
    is not FPL's rule: saved free transfers are retained across both a
    Wildcard and a Free Hit. The Premier League's own worked example: two
    saved before a GW6 wildcard leaves three for GW7 (two saved + GW7's
    allotment). You just do not earn an extra one in the week you play it.

    The old behaviour destroyed up to four banked transfers per wildcard,
    twice a season, and seeded ``ft[0]`` in the multi-period ILP with the
    wrong number for every week after.
    """
    # The PL's worked example, exactly.
    assert roll_forward_free_transfers(2, 15, wildcard_played=True) == 3
    # A full bank survives, subject to the cap.
    assert roll_forward_free_transfers(4, 11, wildcard_played=True) == 5
    assert roll_forward_free_transfers(5, 15, wildcard_played=True) == 5
    # And a manager on the bare allowance is unchanged.
    assert roll_forward_free_transfers(1, 15, wildcard_played=True) == 2


def test_roll_forward_treats_both_free_squad_chips_alike():
    """Wildcard and Free Hit both put their transfers outside the allowance,
    so for any state they must roll forward identically. The two used to
    diverge, which is what made §10 visible."""
    for ft in range(1, 6):
        assert (
            roll_forward_free_transfers(ft, 15, wildcard_played=True)
            == roll_forward_free_transfers(ft, 15, free_hit_played=True)
        )


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


def _marginal_upgrades() -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """A squad where three non-owned players are worth slightly more than the
    men they would replace — a per-week edge of 0.3 over a 3-gameweek horizon,
    so ~0.9 total, comfortably UNDER the 1.5 switching cost.

    One owned player is pinned far above everyone so captaincy never moves;
    otherwise upgrading the top scorer would add a second copy of the gain and
    muddy what is being measured.
    """
    players, squad = _pool()
    proj = _flat_projections(players, squad)
    owned = set(squad)
    proj.loc[proj["player_id"] == squad[0], "xpts"] = 20.0          # permanent captain
    upgrades = [r.id for r in players.itertuples() if r.id not in owned][:3]
    proj.loc[proj["player_id"].isin(upgrades), "xpts"] = 4.3
    return players, squad, proj


def test_switching_cost_still_suppresses_marginal_churn_on_a_normal_week():
    """The control for the test below: with a real switching cost and plenty
    of free transfers, a sub-threshold edge must NOT trigger churn. This is
    the behaviour `transfer_switching_cost` exists to produce."""
    players, squad, proj = _marginal_upgrades()
    plan = evaluate_transfers(
        squad, proj, players, free_transfers=5, available_budget=200.0,
        wildcard_active=False,
        transfer_rules=dataclasses.replace(TRANSFERS, ft_terminal_value=0.0),
    )
    assert plan.transfers_in == []


def test_a_wildcard_is_not_taxed_by_the_switching_cost():
    """Regression, 2026-08-18 (engine review §11).

    `transfer_switching_cost` exists to stop a noise-sized edge triggering
    churn WITHIN the free allowance. A wildcard has no allowance to churn
    against — unlimited transfers are the whole point of the chip — but the
    objective charged the tax on wildcard weeks anyway, docking a ten-player
    rebuild 15 points and making the solver under-use a chip it had just
    decided to spend. The hit term was already zeroed for the same reason
    (`hit[0] == 0`); this was the missing half.

    Same squad and same marginal edges as the control above: taxed, they are
    correctly declined; on a wildcard they must be taken.
    """
    players, squad, proj = _marginal_upgrades()
    plan = evaluate_transfers(
        squad, proj, players, free_transfers=1, available_budget=200.0,
        wildcard_active=True,
        transfer_rules=dataclasses.replace(TRANSFERS, ft_terminal_value=0.0),
    )
    assert plan.transfers_in != [], "a wildcard must not be taxed into inaction"
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


def test_bench_gk_weight_zero_leaves_the_dead_weight_backup_gk():
    """Pre-P1.7 behaviour: the transfer objective scored only `starting` and
    `captain`, so a bench player was worth exactly nothing and every
    in-season transfer eroded the bench optimise_squad had paid for.

    The reserve keeper is weighted by `bench_gk_weight` (2026-08-18) rather
    than the old flat `bench_value_weight`: he has no substitution queue to
    inherit from and plays only if the first-choice keeper does not.
    """
    players, squad, proj = _bench_gk_choice()
    plan = evaluate_transfers(
        squad, proj, players, free_transfers=1, available_budget=100.0,
        config=dataclasses.replace(OPTIMISER, bench_gk_weight=0.0),
        transfer_rules=dataclasses.replace(TRANSFERS, ft_terminal_value=0.0),
    )
    assert not any(t["player_id"] == 16 for t in plan.transfers_in)


def test_bench_gk_weight_upgrades_the_backup_gk():
    players, squad, proj = _bench_gk_choice()
    plan = evaluate_transfers(
        squad, proj, players, free_transfers=1, available_budget=100.0,
        config=dataclasses.replace(OPTIMISER, bench_gk_weight=0.15),
        transfer_rules=dataclasses.replace(TRANSFERS, ft_terminal_value=0.0),
    )
    assert any(t["player_id"] == 16 for t in plan.transfers_in)


def test_transfers_and_squad_build_agree_on_what_a_first_substitute_is_worth():
    """The two objectives must use the same bench weights.

    The squad build pays real money for a substitute who actually plays. If
    the weekly transfer ILP still valued every bench player at a flat rate it
    would sell him the following week, undoing the purchase and charging a
    transfer for it — the failure this term was added to prevent, reintroduced
    from the other side.
    """
    import inspect

    from optimiser import squad as squad_module
    from optimiser import transfers as transfers_module

    for module in (squad_module, transfers_module):
        source = inspect.getsource(module)
        assert "bench_slot_weights" in source, f"{module.__name__} must use the slot weights"
        assert "bench_gk_weight" in source, f"{module.__name__} must use the keeper weight"


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


# --- P1.8 chip usage must not be consumed by re-running a gameweek --------


def test_chips_used_deduplicates_reruns_of_the_same_gameweek():
    """`_record_decision` appends a row every run, and
    `_chip_uses_remaining` counts rows -- so re-running a gameweek's
    decision (a dry-run rehearsal, a crash and retry, refining the squad as
    news lands) consumed a chip that was only ever played once."""
    import json as _json

    from optimiser.chips import Chip, _chip_uses_remaining, chips_used_this_season

    log = pd.DataFrame([
        {"decision_type": "chip", "gameweek": 7, "details": _json.dumps({"chip": "3xc"})},
        {"decision_type": "chip", "gameweek": 7, "details": _json.dumps({"chip": "3xc"})},
        {"decision_type": "chip", "gameweek": 7, "details": _json.dumps({"chip": "3xc"})},
    ])
    used = chips_used_this_season(log)
    assert used == [(Chip.TRIPLE_CAPTAIN, 7)]
    # still available in the SECOND half -- one use per half, not per season
    assert _chip_uses_remaining(Chip.TRIPLE_CAPTAIN, used, current_gw=30, season=None) == 1
    # and correctly spent for the first half
    assert _chip_uses_remaining(Chip.TRIPLE_CAPTAIN, used, current_gw=7, season=None) == 0


def test_chips_used_keeps_genuinely_separate_gameweeks():
    """The dedupe must not collapse the legitimate once-per-half multiplicity."""
    import json as _json

    from optimiser.chips import Chip, chips_used_this_season

    log = pd.DataFrame([
        {"decision_type": "chip", "gameweek": 7, "details": _json.dumps({"chip": "3xc"})},
        {"decision_type": "chip", "gameweek": 30, "details": _json.dumps({"chip": "3xc"})},
    ])
    assert chips_used_this_season(log) == [
        (Chip.TRIPLE_CAPTAIN, 7), (Chip.TRIPLE_CAPTAIN, 30),
    ]


# --- P1.9 squad age, so the wildcard's min-managed-gameweeks gate binds ----


def test_squad_age_counts_from_the_last_wildcard():
    from agent.decision_engine import _squad_age_gws
    from optimiser.chips import Chip

    log = pd.DataFrame([{"decision_type": "lineup", "gameweek": 1}])
    chips = [(Chip.WILDCARD, 12)]
    assert _squad_age_gws(log, chips, next_gw=14) == 2


def test_squad_age_counts_from_the_first_lineup_when_no_wildcard_played():
    from agent.decision_engine import _squad_age_gws

    log = pd.DataFrame([
        {"decision_type": "lineup", "gameweek": 1},
        {"decision_type": "lineup", "gameweek": 2},
    ])
    assert _squad_age_gws(log, [], next_gw=8) == 7


def test_squad_age_is_zero_with_no_history():
    """A fresh bot must read as age 0, not the 99 the parameter defaults to
    -- which is what made the wildcard gate inert live."""
    from agent.decision_engine import _squad_age_gws

    assert _squad_age_gws(pd.DataFrame(), [], next_gw=1) == 0


# --- P1.6 real budget: selling prices and a bank that actually moves ------


def test_selling_price_returns_half_of_a_rise():
    """FPL pays back only half of any price RISE, rounded DOWN to £0.1m."""
    from optimiser.transfers import selling_price

    assert selling_price(5.0, 5.2) == pytest.approx(5.1)   # +0.2 -> keep 0.1
    assert selling_price(5.0, 5.4) == pytest.approx(5.2)   # +0.4 -> keep 0.2
    # odd number of tenths rounds DOWN, not to the nearest
    assert selling_price(5.0, 5.1) == pytest.approx(5.0)
    assert selling_price(5.0, 5.3) == pytest.approx(5.1)


def test_selling_price_takes_a_fall_in_full():
    from optimiser.transfers import selling_price

    assert selling_price(5.0, 4.7) == pytest.approx(4.7)
    assert selling_price(5.0, 5.0) == pytest.approx(5.0)


def test_selling_price_is_exact_at_awkward_tenths():
    """Prices are quoted in £0.1m and float arithmetic on tenths does not
    round the way FPL's does -- hence the integer-tenths implementation."""
    from optimiser.transfers import selling_price

    assert selling_price(4.3, 4.7) == pytest.approx(4.5)
    assert selling_price(10.7, 11.2) == pytest.approx(10.9)


def test_bank_flow_reproduces_the_old_budget_constraint_exactly():
    """The cash-flow formulation is a generalisation, not a second code
    path: with no purchase prices every player sells at their current cost,
    which collapses it back to `sum(now_cost) <= budget`."""
    players, squad = _pool()
    proj = _flat_projections(players, squad)
    proj.loc[~proj["player_id"].isin(squad), "xpts"] = 9.0

    squad_cost = float(players[players["id"].isin(squad)]["now_cost"].sum())
    explicit = evaluate_transfers(
        squad, proj, players, free_transfers=5, bank=3.0,
        transfer_rules=_NO_FRICTION,
    )
    derived = evaluate_transfers(
        squad, proj, players, free_transfers=5, available_budget=squad_cost + 3.0,
        transfer_rules=_NO_FRICTION,
    )
    assert {t["player_id"] for t in explicit.transfers_in} == {
        t["player_id"] for t in derived.transfers_in
    }


def test_an_empty_bank_blocks_an_upgrade_it_cannot_afford():
    players, squad = _pool()
    proj = _flat_projections(players, squad)
    # a brilliant but expensive target, and no money to fund the difference
    target = int(players[~players["id"].isin(squad)]["id"].iloc[0])
    players.loc[players["id"] == target, "now_cost"] = 14.0
    proj.loc[proj["player_id"] == target, "xpts"] = 40.0

    broke = evaluate_transfers(
        squad, proj, players, free_transfers=1, bank=0.0, transfer_rules=_NO_FRICTION
    )
    funded = evaluate_transfers(
        squad, proj, players, free_transfers=1, bank=20.0, transfer_rules=_NO_FRICTION
    )
    assert not any(t["player_id"] == target for t in broke.transfers_in)
    assert any(t["player_id"] == target for t in funded.transfers_in)


def test_purchase_prices_reduce_spending_power_after_a_price_rise():
    """The core P1.6 defect: an appreciated squad was valued at now_cost, so
    the optimiser believed it could spend money that selling would not
    actually realise."""
    players, squad = _pool()
    proj = _flat_projections(players, squad)
    target = int(players[~players["id"].isin(squad)]["id"].iloc[0])
    players.loc[players["id"] == target, "now_cost"] = 9.0
    proj.loc[proj["player_id"] == target, "xpts"] = 40.0
    # every owned player has risen £1.0m since purchase -> only £0.5m each is
    # realisable, so the squad is worth notably less than its sticker price
    risen = {pid: float(players.loc[players["id"] == pid, "now_cost"].iloc[0]) - 1.0
             for pid in squad}

    kwargs = {"free_transfers": 1, "bank": 0.0, "transfer_rules": _NO_FRICTION}
    naive = evaluate_transfers(squad, proj, players, **kwargs)
    real = evaluate_transfers(squad, proj, players, purchase_prices=risen, **kwargs)

    naive_spend = sum(t["cost"] for t in naive.transfers_in)
    real_spend = sum(t["cost"] for t in real.transfers_in)
    assert real_spend <= naive_spend, (
        "knowing the real selling prices must never increase spending power"
    )


def test_settle_transfers_moves_the_bank_and_the_ledger():
    from agent.decision_engine import SquadState, _settle_transfers
    from optimiser.transfers import TransferPlan

    players = pd.DataFrame([
        {"id": 1, "now_cost": 6.0},   # bought at 5.0, so sells for 5.5
        {"id": 2, "now_cost": 7.5},
    ])
    state = SquadState(
        squad_ids=[1], budget=100.0, free_transfers=1,
        bank=2.0, purchase_prices={1: 5.0},
    )
    plan = TransferPlan(
        transfers_in=[{"player_id": 2, "web_name": "in", "cost": 7.5}],
        transfers_out=[{"player_id": 1, "web_name": "out", "cost": 6.0}],
        hits_taken=0, xpts_gain=0.0, net_xpts_gain=0.0,
    )
    bank, prices = _settle_transfers(state, plan, players)

    assert bank == pytest.approx(2.0 + 5.5 - 7.5)
    assert prices == {2: 7.5}, "the sold player leaves the ledger, the bought one joins at cost"


def test_settle_transfers_treats_an_unknown_purchase_price_as_no_rise():
    """A squad carried over from before P1.6 has no ledger; assuming the
    purchase price equals the current price is exactly what the engine
    implicitly assumed before, so nothing regresses."""
    from agent.decision_engine import SquadState, _settle_transfers
    from optimiser.transfers import TransferPlan

    players = pd.DataFrame([{"id": 1, "now_cost": 6.0}, {"id": 2, "now_cost": 6.0}])
    state = SquadState([1], 100.0, 1, bank=0.0, purchase_prices={})
    plan = TransferPlan(
        transfers_in=[{"player_id": 2, "web_name": "in", "cost": 6.0}],
        transfers_out=[{"player_id": 1, "web_name": "out", "cost": 6.0}],
        hits_taken=0, xpts_gain=0.0, net_xpts_gain=0.0,
    )
    bank, prices = _settle_transfers(state, plan, players)
    assert bank == pytest.approx(0.0)
    assert prices == {2: 6.0}
