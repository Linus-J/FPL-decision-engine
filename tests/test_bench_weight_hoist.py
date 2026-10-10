"""One bench-weight derivation per decision, shared by every consumer (B3c).

B3 derived the weights INSIDE `optimise_squad`, which cannot work on an
ordinary week: `agent/decision_engine.py` never calls `optimise_squad` there,
it calls `evaluate_transfers`, and a transfer ILP has no squad solve to derive
from. So the squad build priced slot 1 at the derived value while the weekly
planner priced it at the static 0.53 — the buys-a-substitute-the-planner-
sells-back divergence `optimiser/transfers.py:417-431` exists to prevent.

Two shapes replace it:

* ordinary weekly path — hoist once, from the INCUMBENT XI, ahead of every
  consumer;
* rebuild paths (free hit, wildcard, cold start) — solve, derive from that
  solution's XI, re-solve ONCE. No loop, so it cannot oscillate.

Measured on the live GW4 frame, the two answers differ by 0.058 / 0.049 /
0.017 on the slots and 0.102 on the keeper, and the difference is systematic
rather than noise: `optimise_squad` selects nailed-on players, so a solved XI's
mean P(start) is always the higher of the two. They are different questions,
not one question with an error bar.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from config.strategy import OPTIMISER
from optimiser.bench_weights import derive_bench_config, derive_slot_weights


def _legal_fifteen(start_probability: float = 0.75) -> pd.DataFrame:
    """A legal 15 (2 GKP / 5 DEF / 5 MID / 3 FWD) inside the 3-per-club cap."""
    rows = []
    pid = 0
    for pos, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        for _ in range(count):
            pid += 1
            rows.append({
                "id": pid, "web_name": f"p{pid}", "position": pos,
                "now_cost": 4.0, "team_id": pid, "status": "a",
                "start_probability": start_probability,
            })
    return pd.DataFrame(rows)


def _projections(players: pd.DataFrame, gws=(1,)) -> pd.DataFrame:
    """Distinct xPts so the XI pick is unique rather than a tie broken by the
    solver's internal ordering."""
    rows = []
    for gw in gws:
        for n, (_, p) in enumerate(players.iterrows()):
            rows.append({
                "player_id": int(p["id"]), "gameweek": gw,
                "xpts": 5.0 - 0.1 * n, "xpts_var": 1.0,
                "upside": 0.5, "downside": 0.5,
                "start_probability": p.get("start_probability", 0.75),
            })
    return pd.DataFrame(rows)


def test_the_hoist_derives_from_the_incumbent_xi():
    """The whole squad sits at start_probability 0.75 (blank 0.25, no
    blank_probability column so the fallback applies), so whichever legal XI
    is chosen the ten outfielders are ten 0.25 blanks and the answer is
    pinned."""
    players = _legal_fifteen(0.75)
    cfg = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)
    out = derive_bench_config(
        cfg, players["id"].tolist(), _projections(players), players, gw=1
    )
    assert out.bench_slot_weights == pytest.approx(derive_slot_weights([0.25] * 10))
    assert out.bench_gk_weight == pytest.approx(0.25)


def test_a_fragile_squad_gets_a_fatter_slot_one_than_the_static_constant():
    players = _legal_fifteen(0.75)
    cfg = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)
    out = derive_bench_config(
        cfg, players["id"].tolist(), _projections(players), players, gw=1
    )
    assert out.bench_slot_weights[0] > OPTIMISER.bench_slot_weights[0]


def test_the_flag_off_leaves_the_config_untouched():
    players = _legal_fifteen(0.75)
    out = derive_bench_config(
        OPTIMISER, players["id"].tolist(), _projections(players), players, gw=1
    )
    assert out is OPTIMISER


def test_no_incumbent_squad_leaves_the_config_untouched():
    """A cold start has no incumbent squad by definition. It is a REBUILD
    path and gets its weights from `optimise_squad`'s own derive-and-re-solve
    pass instead, so returning `cfg` unchanged here is correct rather than a
    silent disabling."""
    players = _legal_fifteen(0.75)
    cfg = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)
    assert derive_bench_config(cfg, [], _projections(players), players, gw=1) is cfg


def test_a_frame_with_no_start_probability_column_leaves_the_config_untouched():
    players = _legal_fifteen(0.75).drop(columns=["start_probability"])
    cfg = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)
    out = derive_bench_config(
        cfg, players["id"].tolist(), _projections(players), players, gw=1
    )
    assert out is cfg


def test_a_squad_too_short_to_field_an_eleven_leaves_the_config_untouched():
    """Departed players can leave the saved squad with fewer than 11 rows in
    `players`. That must degrade to the static weights, not raise."""
    players = _legal_fifteen(0.75)
    cfg = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)
    out = derive_bench_config(
        cfg, [1, 2, 3], _projections(players), players, gw=1
    )
    assert out is cfg


# --- the consistency test: one weight set reaches BOTH ILPs ----------------


def test_the_transfer_ilp_and_the_squad_ilp_are_handed_the_same_bench_weights(
    monkeypatch,
):
    """The failure this exists for: `optimise_squad` rebound `cfg` locally, so
    the derived weights never reached `evaluate_transfers`.

    `tests/test_transfer_banking.py:418-421` greps module source for the field
    NAMES, which is why it could not see this — both modules mention
    `bench_slot_weights` either way. This asserts on the values that actually
    reach the two solvers.
    """
    from agent import decision_engine as de
    from optimiser.transfers import TransferPlan

    players = _legal_fifteen(0.75)
    projections = _projections(players, gws=(4, 5))
    squad_ids = players["id"].tolist()
    seen: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def _fake_evaluate_transfers(*_a, config=None, **_k):
        seen["transfers"] = config
        return TransferPlan([], [], 0, 0.0, 0.0)

    def _fake_optimise_starting_xi(*_a, config=None, **_k):
        seen["squad"] = config
        raise _Stop()

    monkeypatch.setattr(de, "_get_current_and_next_gw", lambda: (3, 4))
    monkeypatch.setattr(de, "season_has_played_history", lambda *a, **k: True)
    monkeypatch.setattr(de, "get_latest_projections", lambda **_: projections)
    monkeypatch.setattr(de, "_load_players", lambda: players.drop(
        columns=["start_probability"]
    ))
    monkeypatch.setattr(
        de, "_load_squad_state",
        lambda *a, **k: de.SquadState(squad_ids, 100.0, 1, 0.0, {}),
    )
    monkeypatch.setattr(
        de, "_load_own_decision_log", lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(de, "_get_dgw_gameweeks", lambda *a, **k: set())
    monkeypatch.setattr(de, "_get_bgw_gameweeks", lambda *a, **k: set())
    monkeypatch.setattr(de, "load_latest_ownership", lambda *a, **k: None)
    monkeypatch.setattr(de, "evaluate_transfers", _fake_evaluate_transfers)
    monkeypatch.setattr(de, "optimise_starting_xi", _fake_optimise_starting_xi)
    monkeypatch.setattr(
        de, "recommend_chip",
        lambda **_k: de.ChipRecommendation(None, "test", 0.0),
    )

    cfg = dataclasses.replace(OPTIMISER, derive_bench_weights_per_solve=True)
    timing = dataclasses.replace(de.CHIP_TIMING, chip_comparison_enabled=False)
    with pytest.raises(_Stop):
        de._run_decision_cycle(
            season="2026-27", dry_run=True, force_chip=None,
            config=cfg, chip_timing=timing,
            team_id=None, sim_manager_id=None, refresh_projections=False,
        )

    assert "transfers" in seen and "squad" in seen
    assert seen["transfers"].bench_slot_weights == seen["squad"].bench_slot_weights
    assert seen["transfers"].bench_gk_weight == seen["squad"].bench_gk_weight
    # And they are the DERIVED weights, not the static tuple that both would
    # trivially share if the hoist had not run at all.
    assert seen["transfers"].bench_slot_weights != OPTIMISER.bench_slot_weights
    assert seen["transfers"].bench_slot_weights == pytest.approx(
        derive_slot_weights([0.25] * 10)
    )
