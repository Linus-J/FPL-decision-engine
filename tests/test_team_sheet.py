"""agent/team_sheet.py::build_picks

Real bug found 2026-08-01 (user's own repo-cleanup request): the old
per-player position helper always returned a FIXED slot per position
(e.g. every starting DEF got position=2), so any squad with more than one
starter in a position submitted duplicate `position` values to FPL's API
-- which requires 15 unique integers 1-15. Untested and never exercised
live (dry-run only so far, so this was a live landmine, not yet triggered).
"""

from __future__ import annotations

from agent.team_sheet import build_picks


def _squad_with_realistic_formation() -> list[dict]:
    """1 GKP, 4 DEF, 4 MID, 2 FWD starting (11), + GK/DEF/MID/FWD bench (4)
    -- a real formation with MULTIPLE starters per outfield position, the
    exact case the old bug silently collided on."""
    squad = []
    pid = 1

    def add(position: str, is_starting: bool, bench_order: int = 99) -> int:
        nonlocal pid
        squad.append({
            "id": pid, "position": position, "is_starting": is_starting,
            "bench_order": bench_order,
        })
        pid += 1
        return pid - 1

    gk = add("GKP", True)
    defs = [add("DEF", True) for _ in range(4)]
    mids = [add("MID", True) for _ in range(4)]
    fwds = [add("FWD", True) for _ in range(2)]
    bench_gk = add("GKP", False, bench_order=0)
    bench_def = add("DEF", False, bench_order=1)
    bench_mid = add("MID", False, bench_order=3)
    bench_fwd = add("FWD", False, bench_order=2)

    return squad, {
        "gk": gk, "defs": defs, "mids": mids, "fwds": fwds,
        "bench_gk": bench_gk, "bench_def": bench_def,
        "bench_mid": bench_mid, "bench_fwd": bench_fwd,
    }


def test_every_position_value_is_unique():
    squad, _ = _squad_with_realistic_formation()
    picks = build_picks(squad, captain_id=squad[1]["id"], vice_captain_id=squad[2]["id"])
    positions = [p["position"] for p in picks]
    assert len(positions) == 15
    assert len(set(positions)) == 15, "duplicate position values would be rejected by FPL's API"
    assert sorted(positions) == list(range(1, 16))


def test_starting_goalkeeper_is_always_slot_one():
    squad, ids = _squad_with_realistic_formation()
    picks = build_picks(squad, captain_id=ids["gk"], vice_captain_id=ids["defs"][0])
    gk_pick = next(p for p in picks if p["element"] == ids["gk"])
    assert gk_pick["position"] == 1


def test_starters_are_grouped_before_bench():
    squad, ids = _squad_with_realistic_formation()
    picks = build_picks(squad, captain_id=ids["gk"], vice_captain_id=ids["defs"][0])
    by_id = {p["element"]: p["position"] for p in picks}
    starter_ids = [ids["gk"], *ids["defs"], *ids["mids"], *ids["fwds"]]
    bench_ids = [ids["bench_gk"], ids["bench_def"], ids["bench_mid"], ids["bench_fwd"]]
    assert max(by_id[pid] for pid in starter_ids) <= 11
    assert min(by_id[pid] for pid in bench_ids) >= 12


def test_bench_order_field_is_respected_not_squad_list_order():
    """Bench players appear in the squad list in a DIFFERENT order than
    their bench_order field (GK=0, DEF=1, FWD=2, MID=3 -- FWD before MID)
    -- the payload must follow bench_order, not raw list position."""
    squad, ids = _squad_with_realistic_formation()
    picks = build_picks(squad, captain_id=ids["gk"], vice_captain_id=ids["defs"][0])
    by_id = {p["element"]: p["position"] for p in picks}
    assert by_id[ids["bench_gk"]] == 12
    assert by_id[ids["bench_def"]] == 13
    assert by_id[ids["bench_fwd"]] == 14  # bench_order=2, appears before mid (bench_order=3)
    assert by_id[ids["bench_mid"]] == 15


def test_captain_and_vice_captain_flags_set_correctly():
    squad, ids = _squad_with_realistic_formation()
    picks = build_picks(squad, captain_id=ids["gk"], vice_captain_id=ids["defs"][0])
    by_id = {p["element"]: p for p in picks}
    assert by_id[ids["gk"]]["is_captain"] is True
    assert by_id[ids["defs"][0]]["is_vice_captain"] is True
    assert by_id[ids["defs"][1]]["is_captain"] is False
    assert by_id[ids["defs"][1]]["is_vice_captain"] is False
