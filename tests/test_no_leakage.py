"""T4 leakage guards — the CI gate that keeps the spine honest.

1. Grep guard: no mutable players.* dynamic column is read into any
   training/backtest SQL path (Phase-1 leaks L1/L2/L3). Extended per M4 to
   the transfer/injury columns.
2. As-of canary: a future-dated snapshot is never returned by an as-of read.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Gameweek, Player, PlayerStateSnapshot

ROOT = Path(__file__).resolve().parents[1]

# The mutable players.* columns that must never be read as "latest" into a
# historical/training row. SQL alias form `p.<col>` — distinct from `ps.<col>`
# (the point-in-time snapshot) and DataFrame access like player["status"].
FORBIDDEN = [
    "p.form", "p.ict_index", "p.influence", "p.creativity", "p.threat",
    "p.selected_by_percent", "p.status", "p.chance_of_playing",
    "p.transfers_in_event", "p.transfers_out_event", "p.injury_severity",
]

# Files whose SQL feeds model training or the backtest. points_model.py/
# cs_model.py removed 2026-08-01 (confirmed dead in the live and backtest
# paths -- superseded by projection/assemble.py's P10 MC assembly).
GUARDED_FILES = [
    "scripts/backtest.py",
    "projection/features.py",
    "projection/minutes_model.py",
]


@pytest.mark.parametrize("relpath", GUARDED_FILES)
def test_no_leaked_columns_in_read_paths(relpath):
    text = (ROOT / relpath).read_text()
    hits = [tok for tok in FORBIDDEN if tok in text]
    assert not hits, f"{relpath} reads mutable players.* columns (leak): {hits}"


@pytest.mark.parametrize("relpath", GUARDED_FILES)
def test_read_paths_use_snapshots(relpath):
    """Positive check: each guarded read path sources point-in-time snapshots."""
    text = (ROOT / relpath).read_text()
    assert "player_state_snapshots" in text, f"{relpath} does not read the snapshot table"


@pytest.fixture
def temp_backtest(monkeypatch):
    import scripts.backtest as bt

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(bt, "get_session", lambda: Local())
    return bt, Local


def test_asof_read_excludes_future_snapshot(temp_backtest):
    bt, Local = temp_backtest
    deadline = datetime(2024, 10, 5, 11, 0)

    s = Local()
    try:
        s.add(Player(fpl_id=1, code=1, first_name="F", second_name="L",
                     web_name="A", team_id=1, position="MID", now_cost=5.0))
        s.add(Gameweek(id=8, season="2024-25", name="Gameweek 8", deadline_time=deadline))
        # as-of value the reader MUST return
        s.add(PlayerStateSnapshot(player_id=1, season="2024-25",
                                  snapshot_ts=deadline - timedelta(minutes=1),
                                  gameweek_context=8, ict_index=100.0, now_cost=5.0))
        # future-dated value the reader MUST NOT leak
        s.add(PlayerStateSnapshot(player_id=1, season="2024-25",
                                  snapshot_ts=deadline + timedelta(days=1),
                                  gameweek_context=9, ict_index=999.0, now_cost=9.9))
        s.commit()
    finally:
        s.close()

    df = bt._load_players_snapshot("2024-25", 8)
    assert len(df) == 1
    assert df.iloc[0]["ict_index"] == 100.0   # as-of, not the future 999
    assert df.iloc[0]["now_cost"] == 5.0       # not the future 9.9

    # regression guard: available_gws are numpy int64 in the backtest loop;
    # SQLite silently matches none unless the param is cast to int.
    import numpy as np
    assert len(bt._load_players_snapshot("2024-25", np.int64(8))) == 1


def test_asof_read_returns_empty_before_any_snapshot(temp_backtest):
    """GW1 with no prior snapshot → empty (no fabricated current-state row)."""
    bt, Local = temp_backtest
    s = Local()
    try:
        s.add(Player(fpl_id=1, code=1, first_name="F", second_name="L",
                     web_name="A", team_id=1, position="MID", now_cost=5.0))
        s.add(Gameweek(id=1, season="2024-25", name="Gameweek 1",
                       deadline_time=datetime(2024, 8, 16, 11, 0)))
        s.commit()
    finally:
        s.close()
    df = bt._load_players_snapshot("2024-25", 1)
    assert df.empty
