"""T3 acceptance gate — reconciled snapshot backfill (the C1 guard).

Self-contained: pure transforms on synthetic merged_gw frames; DB writer against
a throwaway temp DB. The key test is the parity check: a snapshot informing GW g
carries the season-cumulative stats bootstrap would hold after GWs 1..g-1 — same
quantity the live path writes, so no train/serve skew.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from data.models import Base, Player, PlayerStateSnapshot
from scripts import backfill_history as bh

DEADLINES = {g: datetime(2024, 8, 1) + timedelta(days=7 * g) for g in range(1, 6)}


def _row(element, gw, ict, infl=0.0, crea=0.0, thr=0.0, value=50, selected=1000, tp=0,
         tin=0, tout=0):
    return {
        "element": element, "GW": gw, "ict_index": ict, "influence": infl,
        "creativity": crea, "threat": thr, "value": value, "selected": selected,
        "total_points": tp, "transfers_in": tin, "transfers_out": tout,
    }


def test_cumulative_is_through_previous_gw():
    """C1 PARITY: snapshot for GW g == sum of per-GW ICT over GWs < g (what the
    live bootstrap holds after g-1 GWs are played)."""
    df = pd.DataFrame([
        _row(10, 1, ict=10, infl=100, crea=50, thr=30),
        _row(10, 2, ict=20, infl=200, crea=60, thr=40),
        _row(10, 3, ict=30, infl=300, crea=70, thr=50),
        _row(10, 4, ict=5, infl=10, crea=5, thr=5),
    ])
    rows = {r["gameweek_context"]: r for r in bh.compute_snapshot_rows(df, DEADLINES)}
    assert rows[1]["ict_index"] == 0.0                 # no prior GWs
    assert rows[2]["ict_index"] == 10.0                # GW1
    assert rows[3]["ict_index"] == 30.0                # GW1+2
    assert rows[4]["ict_index"] == 60.0                # GW1+2+3
    # other cumulative components track identically
    assert rows[4]["influence"] == 600.0
    assert rows[3]["creativity"] == 110.0
    assert rows[4]["threat"] == 120.0


def test_dgw_two_fixtures_sum_into_one_gw():
    """A double-gameweek (two fixtures, two rows) sums into that GW's total."""
    df = pd.DataFrame([
        _row(20, 1, ict=5), _row(20, 1, ict=7),   # DGW: 5 + 7 = 12
        _row(20, 2, ict=3),
    ])
    rows = {r["gameweek_context"]: r for r in bh.compute_snapshot_rows(df, DEADLINES)}
    assert rows[2]["ict_index"] == 12.0  # cumulative through GW1 == the DGW total


def test_form_is_prior_window_mean():
    df = pd.DataFrame([
        _row(30, 1, ict=1, tp=2),
        _row(30, 2, ict=1, tp=6),
        _row(30, 3, ict=1, tp=2),
        _row(30, 4, ict=1, tp=8),
    ])
    rows = {r["gameweek_context"]: r for r in bh.compute_snapshot_rows(df, DEADLINES)}
    assert rows[1]["form"] == 0.0                       # no prior
    assert rows[2]["form"] == 2.0                       # mean(2)
    assert rows[3]["form"] == 4.0                       # mean(2,6)
    assert rows[4]["form"] == pytest.approx(10 / 3)     # mean(2,6,2)


def test_selected_by_percent_scale():
    """Σ(selected)/15 ≈ managers → owner of 400k in a 1M-manager league ≈ 40%."""
    rows_in = [_row(1, 1, ict=0, selected=400_000)]          # target: 40%
    rows_in += [_row(100 + i, 1, ict=0, selected=100_000) for i in range(146)]  # Σ→15M
    df = pd.DataFrame(rows_in)
    out = {r["element"]: r for r in bh.compute_snapshot_rows(df, DEADLINES)}
    assert out[1]["selected_by_percent"] == pytest.approx(40.0, abs=0.5)
    assert 0.0 <= out[101]["selected_by_percent"] <= 100.0


def test_snapshot_ts_just_before_deadline():
    df = pd.DataFrame([_row(40, 1, ict=1)])
    row = bh.compute_snapshot_rows(df, DEADLINES)[0]
    assert row["snapshot_ts"] == DEADLINES[1] - timedelta(minutes=1)
    assert row["snapshot_ts"] < DEADLINES[1]
    assert row["now_cost"] == pytest.approx(5.0)  # value 50 / 10


def test_missing_required_columns_returns_empty():
    assert bh.compute_snapshot_rows(pd.DataFrame({"element": [1]}), DEADLINES) == []


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 't3.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(bh, "get_session", lambda: Local())
    return Local


def test_write_snapshot_rows_maps_and_skips_departed(temp_session):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=5001, first_name="F", second_name="L",
                     web_name="A", team_id=1, position="MID", now_cost=5.0))
        s.commit()
        player_id = s.query(Player.id).filter_by(code=5001).scalar()
    finally:
        s.close()

    df = pd.DataFrame([
        _row(10, 1, ict=10, value=55, selected=1000),
        _row(10, 2, ict=20, value=56, selected=1200),
        _row(99, 1, ict=5),   # element 99 -> code 9999 -> no current player (departed)
    ])
    rows = bh.compute_snapshot_rows(df, DEADLINES)
    elem_to_code = {10: 5001, 99: 9999}
    code_to_dbid = {5001: player_id}   # 9999 absent → departed player skipped

    written, skipped = bh.write_snapshot_rows("2024-25", rows, elem_to_code, code_to_dbid)
    assert written == 2       # only element 10's two snapshots
    assert skipped == 1       # element 99 skipped, not misjoined

    s = temp_session()
    try:
        assert s.query(func.count(PlayerStateSnapshot.id)).scalar() == 2
        gw2 = (
            s.query(PlayerStateSnapshot)
            .filter_by(player_id=player_id, gameweek_context=2)
            .one()
        )
        assert gw2.ict_index == 10.0            # cumulative through GW1
        assert gw2.now_cost == pytest.approx(5.6)  # GW2 value 56 / 10
        assert gw2.season == "2024-25"
    finally:
        s.close()


def test_write_snapshot_rows_idempotent(temp_session):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=5001, first_name="F", second_name="L",
                     web_name="A", team_id=1, position="MID", now_cost=5.0))
        s.commit()
        player_id = s.query(Player.id).filter_by(code=5001).scalar()
    finally:
        s.close()
    df = pd.DataFrame([_row(10, 1, ict=10), _row(10, 2, ict=20)])
    rows = bh.compute_snapshot_rows(df, DEADLINES)
    m = {10: 5001}
    d = {5001: player_id}
    first, _ = bh.write_snapshot_rows("2024-25", rows, m, d)
    second, _ = bh.write_snapshot_rows("2024-25", rows, m, d)
    assert first == 2
    assert second == 0  # (player_id, snapshot_ts) unique → nothing re-inserted
