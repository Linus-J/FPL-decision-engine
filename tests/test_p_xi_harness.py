"""P-XI naive-best-XI harness — the testable seams (merge logic + scoring reuse).

Full end-to-end run_naive_xi_backtest is verified against the live DB (like
run_backtest/optimise_squad, which also have no synthetic-DB unit tests in this
repo — the walk-forward + ILP integration is a live-gate concern). This file
regression-tests _merge_squad_dynamic, whose column-collision bug (position/
team_id/web_name duplicated as _x/_y when merging the full player snapshot)
was caught during that live smoke run.
"""

from __future__ import annotations

import pandas as pd

from scripts.backtest import _merge_squad_dynamic


def _squad_static():
    return pd.DataFrame([
        {"id": 1, "position": "FWD", "team_id": 10, "web_name": "Striker"},
        {"id": 2, "position": "DEF", "team_id": 11, "web_name": "Defender"},
    ])


def test_merge_keeps_static_columns_unduplicated():
    # players snapshot ALSO carries position/team_id/web_name (as it does live)
    players = pd.DataFrame([
        {"id": 1, "position": "FWD", "team_id": 10, "web_name": "Striker",
         "now_cost": 14.5, "start_probability": 0.9},
        {"id": 2, "position": "DEF", "team_id": 11, "web_name": "Defender",
         "now_cost": 5.0, "start_probability": 0.7},
    ])
    out = _merge_squad_dynamic(_squad_static(), players, squad_ids=[1, 2])
    # bare column names must survive — no _x/_y suffixing (the regression)
    assert set(out.columns) == {"id", "position", "team_id", "web_name",
                                 "now_cost", "start_probability"}
    assert out.loc[out["id"] == 1, "web_name"].iloc[0] == "Striker"
    assert out.loc[out["id"] == 1, "now_cost"].iloc[0] == 14.5


def test_merge_fills_missing_dynamic_data():
    # player 2 has no snapshot row this GW (e.g. new/unmatched) -> safe defaults
    players = pd.DataFrame([{"id": 1, "now_cost": 14.5, "start_probability": 0.9}])
    out = _merge_squad_dynamic(_squad_static(), players, squad_ids=[1, 2])
    row2 = out.loc[out["id"] == 2].iloc[0]
    assert row2["now_cost"] == 0.0
    assert row2["start_probability"] == 0.5
    assert row2["web_name"] == "Defender"   # static identity still present
