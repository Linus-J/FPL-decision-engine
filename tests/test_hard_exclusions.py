"""Hand-entered hard vetoes: the manager's judgement is not up for negotiation.

The softer `rotation_risk` tier only discounts, and on the live GW1 frame two
of five capped players were selected anyway — a cap leaves the optimiser free
to decide the player is worth it at his price. `exclude` is for when that is
not the question being asked.

An unowned vetoed player must never be bought. An OWNED one must be force-SOLD
rather than silently dropped from the ILP's variable set: that was a real bug
in the departure gate, where removing the variable left the player missing from
`transfers_out` while the squad-size constraint quietly conjured a replacement.
"""

from __future__ import annotations

import pandas as pd

from projection.cold_start import apply_departure_gate


def _players() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": 1, "web_name": "keep", "status": "a"},
        {"id": 2, "web_name": "vetoed", "status": "a"},
        {"id": 3, "web_name": "departed", "status": "u"},
    ])


def test_a_vetoed_player_is_removed_from_the_candidate_pool(monkeypatch):
    monkeypatch.setattr("data.overrides.load_excluded_player_ids", lambda: {2})
    out = apply_departure_gate(_players())

    assert set(out["id"]) == {1}, "the veto and the departure must both be gone"


def test_the_gate_still_drops_confirmed_departures_without_any_veto(monkeypatch):
    monkeypatch.setattr("data.overrides.load_excluded_player_ids", lambda: set())
    out = apply_departure_gate(_players())

    assert set(out["id"]) == {1, 2}


def test_an_empty_veto_list_changes_nothing(monkeypatch):
    monkeypatch.setattr("data.overrides.load_excluded_player_ids", lambda: set())
    baseline = apply_departure_gate(_players())
    monkeypatch.setattr("data.overrides.load_excluded_player_ids", lambda: {9999})
    with_stale = apply_departure_gate(_players())

    assert set(baseline["id"]) == set(with_stale["id"]), (
        "a veto naming a player who is not in the pool must be inert, not "
        "an error — the override file outlives any given squad"
    )


def test_the_gate_survives_a_frame_with_no_status_column(monkeypatch):
    """Some callers pass a reduced frame; the veto must still apply and the
    missing status must not raise."""
    monkeypatch.setattr("data.overrides.load_excluded_player_ids", lambda: {2})
    frame = pd.DataFrame([{"id": 1, "web_name": "keep"}, {"id": 2, "web_name": "vetoed"}])
    out = apply_departure_gate(frame)

    assert set(out["id"]) == {1}


def test_the_two_tiers_are_independent(monkeypatch):
    """A player under a soft cap must NOT be removed from the pool — that is
    the whole distinction between the tiers, and collapsing them would make
    every rotation doubt a veto."""
    from optimiser.rotation_risk import apply_rotation_risk

    monkeypatch.setattr("data.overrides.load_excluded_player_ids", lambda: set())
    pool = apply_departure_gate(_players())
    assert 2 in set(pool["id"])

    projections = pd.DataFrame([
        {"player_id": 2, "gameweek": 1, "start_probability": 0.95, "xpts": 5.0}
    ])
    capped = apply_rotation_risk(projections, {2: 0.5})
    assert capped["xpts"].iloc[0] < 5.0, "a cap discounts rather than removing"
