"""P3-2: effective ownership ingestion — pure aggregation only.

The live network path (sample_top_entries/fetch_entry_picks/
ingest_ownership_snapshot) is marked no-cover, same posture as the other
live-network ingestors (fbref.py, understat_xg.py) — this covers the two
functions that don't need a live/mocked HTTP session: parsing one standings
page response, and aggregating a sample of picks into ownership/captaincy %.

UNVERIFIED AGAINST LIVE DATA (see data/ingestors/ownership.py's module
docstring) — these tests validate the aggregation math against the
well-documented FPL API response shape, not a real populated response
(impossible to obtain pre-GW1, when this was authored).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.ingestors.ownership import _extract_entries, aggregate_ownership
from data.models import Base, OwnershipSnapshot, Player


def test_extract_entries_from_standings_page():
    resp = {
        "standings": {
            "has_next": True,
            "results": [
                {"entry": 111, "rank": 1, "player_name": "A"},
                {"entry": 222, "rank": 2, "player_name": "B"},
            ],
        }
    }
    entries, has_next = _extract_entries(resp)
    assert entries == [111, 222]
    assert has_next is True


def test_extract_entries_empty_page_means_no_next():
    resp = {"standings": {"has_next": False, "results": []}}
    entries, has_next = _extract_entries(resp)
    assert entries == []
    assert has_next is False


def test_extract_entries_missing_standings_key_is_safe():
    assert _extract_entries({}) == ([], False)


def _picks(*element_captain_pairs):
    return [
        {"element": el, "is_captain": is_cap, "position": i + 1, "multiplier": 2 if is_cap else 1}
        for i, (el, is_cap) in enumerate(element_captain_pairs)
    ]


def test_aggregate_ownership_basic_percentages():
    # 4 managers: player 1 owned by all 4, captained by 2; player 2 owned by 1
    picks_by_entry = [
        _picks((1, True), (2, False)),
        _picks((1, True), (3, False)),
        _picks((1, False), (3, False)),
        _picks((1, False), (4, False)),
    ]
    agg = aggregate_ownership(picks_by_entry)
    assert agg[1]["selected_pct"] == 100.0
    assert agg[1]["captaincy_pct"] == 50.0
    assert agg[2]["selected_pct"] == 25.0
    assert agg[2]["captaincy_pct"] == 0.0


def test_aggregate_ownership_empty_sample_returns_empty():
    assert aggregate_ownership([]) == {}


def test_aggregate_ownership_never_captained_player_has_zero_captaincy():
    picks_by_entry = [_picks((5, False)), _picks((5, False))]
    agg = aggregate_ownership(picks_by_entry)
    assert agg[5]["selected_pct"] == 100.0
    assert agg[5]["captaincy_pct"] == 0.0


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ownership.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def test_ownership_snapshot_schema_roundtrip(session):
    session.add(Player(id=1, fpl_id=1, code=1, first_name="A", second_name="A",
                       web_name="a", team_id=1, position="MID", now_cost=5.0))
    session.commit()
    session.add(OwnershipSnapshot(
        player_id=1, snapshot_ts=datetime(2026, 8, 20),
        overall_selected_pct=12.5, top10k_selected_pct=34.0,
        captaincy_pct_top10k=5.5, sample_size=1000,
    ))
    session.commit()
    row = session.query(OwnershipSnapshot).one()
    assert row.overall_selected_pct == 12.5
    assert row.top10k_selected_pct == 34.0
    assert row.captaincy_pct_top10k == 5.5
    assert row.captaincy_pct_overall is None  # documented gap, not a silent 0
    assert row.sample_size == 1000


# --- P3.2 (2026-08-16): the read side, wired for the first time ----------


def test_load_latest_ownership_is_empty_before_anything_is_ingested(tmp_path, monkeypatch):
    """Pre-GW1 reality: sampling the Overall league returns no ranked
    entries, so the table is empty. Consumers treat an empty frame as "0% EO
    for everyone", which is a uniform rescale and changes no ranking -- so
    this must be an empty frame with the right columns, not an error."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import data.ingestors.ownership as ownership_module
    from data.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'eo.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(ownership_module, "get_session", lambda: Local())

    df = ownership_module.load_latest_ownership()
    assert df.empty
    assert list(df.columns) == ["player_id", "top10k_selected_pct"]


def test_load_latest_ownership_keeps_only_the_most_recent_sample(tmp_path, monkeypatch):
    """EO is re-sampled every gameweek, so the table accumulates rows per
    player. Feeding stale duplicates into the objective would weight a
    player by an ownership they no longer have."""
    from datetime import datetime

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import data.ingestors.ownership as ownership_module
    from data.models import Base, OwnershipSnapshot

    engine = create_engine(f"sqlite:///{tmp_path / 'eo2.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(ownership_module, "get_session", lambda: Local())

    s = Local()
    s.add_all([
        OwnershipSnapshot(player_id=1, top10k_selected_pct=20.0,
                          snapshot_ts=datetime(2026, 8, 20)),
        OwnershipSnapshot(player_id=1, top10k_selected_pct=55.0,
                          snapshot_ts=datetime(2026, 8, 27)),
        OwnershipSnapshot(player_id=2, top10k_selected_pct=10.0,
                          snapshot_ts=datetime(2026, 8, 27)),
    ])
    s.commit()
    s.close()

    df = ownership_module.load_latest_ownership().set_index("player_id")
    assert len(df) == 2
    assert df.loc[1, "top10k_selected_pct"] == 55.0
    assert df.loc[2, "top10k_selected_pct"] == 10.0
