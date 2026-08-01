"""P11 prior-league ingest — pure per-90 + row-mapping helpers (network-free)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.ingestors import fbref_prior as fp
from data.models import Base, Player, PriorLeagueStats


def test_compute_per90():
    r = fp.compute_per90(minutes=900, goals=10, assists=5, npxg=9.0, xa=4.5)
    assert r == {"goals90": 1.0, "assists90": 0.5, "npxg90": 0.9, "xa90": 0.45}


def test_compute_per90_zero_minutes_is_zero_not_error():
    assert fp.compute_per90(0, 3, 3, 3, 3) == {
        "goals90": 0.0, "assists90": 0.0, "npxg90": 0.0, "xa90": 0.0
    }


def test_row_to_prior_stats_maps_and_normalises():
    row = {
        "player": "Prolific Striker", "team": "Leeds", "pos": "FW",
        "Playing Time Min": 1800, "Playing Time MP": 20,
        "Performance Gls": 20, "Performance Ast": 10,
        "Expected npxG": 18.0, "Expected xAG": 9.0,
    }
    out = fp.row_to_prior_stats(row, "ENG-Championship", "2025-2026")
    assert out["player_name"] == "Prolific Striker"
    assert out["league"] == "ENG-Championship"
    assert out["minutes"] == 1800 and out["matches"] == 20
    assert out["goals90"] == 1.0    # 20 / (1800/90)
    assert out["npxg90"] == 0.9
    assert out["xa90"] == 0.45


def test_row_to_prior_stats_skips_zero_minutes():
    assert fp.row_to_prior_stats({"player": "Benchwarmer", "Playing Time Min": 0},
                                 "ESP-La Liga", "2025-2026") is None


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'prior.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fp, "get_session", lambda: Local())
    return Local


def test_backfill_prior_league_codes_matches_established_and_leaves_unmatched_null(
    temp_session,
):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=42, first_name="Prolific", second_name="Striker",
                     web_name="Striker", team_id=1, position="FWD", now_cost=6.5))
        s.add(PriorLeagueStats(
            player_name="Prolific Striker", team="Leeds", league="ENG-Championship",
            season="2025-2026", position="FW", minutes=3000, matches=34,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.add(PriorLeagueStats(
            player_name="Nobody Matches This Name", team="Leeds",
            league="ENG-Championship", season="2025-2026", position="FW",
            minutes=1000, matches=15, goals90=0.1, assists90=0.0, npxg90=0.1, xa90=0.0,
        ))
        s.commit()
    finally:
        s.close()

    matched = fp.backfill_prior_league_codes()
    assert matched == 1

    s = temp_session()
    try:
        rows = {r.player_name: r.code for r in s.query(PriorLeagueStats).all()}
    finally:
        s.close()
    assert rows["Prolific Striker"] == 42
    assert rows["Nobody Matches This Name"] is None


def test_backfill_prior_league_codes_is_idempotent(temp_session):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=42, first_name="Prolific", second_name="Striker",
                     web_name="Striker", team_id=1, position="FWD", now_cost=6.5))
        s.add(PriorLeagueStats(
            player_name="Prolific Striker", team="Leeds", league="ENG-Championship",
            season="2025-2026", code=42, position="FW", minutes=3000, matches=34,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.commit()
    finally:
        s.close()

    matched = fp.backfill_prior_league_codes()
    assert matched == 0  # already has a code -- nothing left to backfill
