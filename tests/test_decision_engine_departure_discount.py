"""_run_decision_cycle's live path must apply the rumour-discount tier
(Feature B, plan 2026-08-10) to whatever projections it reads, and must log
a warning for any rumoured player that still makes the final squad."""

from __future__ import annotations

import pandas as pd
import pytest

from agent import decision_engine as de


def test_run_decision_cycle_applies_p_leave_discount_to_projections(monkeypatch):
    captured = {}

    def _fake_apply_departure_discount(projections, p_leave_by_player, rules=None):
        captured["p_leave_by_player"] = p_leave_by_player
        captured["called"] = True
        return projections

    monkeypatch.setattr(de, "load_p_leave_overrides", lambda: {7: 0.5})
    monkeypatch.setattr(de, "apply_departure_discount", _fake_apply_departure_discount)
    monkeypatch.setattr(de, "get_latest_projections", lambda **_: pd.DataFrame([
        {"player_id": 7, "gameweek": 1, "xpts": 5.0, "start_probability": 0.9},
    ]))
    monkeypatch.setattr(de, "_get_current_and_next_gw", lambda: (1, 1))
    monkeypatch.setattr(de, "_load_squad_state", lambda *a, **k: ([], 100.0, 1))

    # Real squad-building is out of scope for this unit test -- short-circuit
    # once the discount call itself has been observed.
    class _Stop(Exception):
        pass

    def _boom(*a, **k):
        raise _Stop()

    monkeypatch.setattr(de, "_load_players", _boom)

    with pytest.raises(_Stop):
        de._run_decision_cycle(
            season="2026-27", dry_run=True, force_chip=None,
            config=de.OPTIMISER, chip_timing=de.CHIP_TIMING,
            team_id=None, sim_manager_id=None, refresh_projections=False,
        )

    assert captured.get("called") is True
    assert captured["p_leave_by_player"] == {7: 0.5}
