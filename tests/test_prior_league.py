"""P11 prior-league ingest — pure per-90 + row-mapping helpers (network-free)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.ingestors import fbref_prior as fp
from data.models import Base, Player, PriorLeagueStats


def test_compute_per90():
    r = fp.compute_per90(minutes=900, goals=10, assists=5, npxg=9.0, xa=4.5)
    assert r == {"goals90": 1.0, "assists90": 0.5, "npg90": 0.0,
                 "npxg90": 0.9, "xa90": 0.45}


def test_compute_per90_zero_minutes_is_zero_not_error():
    assert fp.compute_per90(0, 3, 3, 3, 3) == {
        "goals90": 0.0, "assists90": 0.0, "npg90": 0.0,
        "npxg90": 0.0, "xa90": 0.0,
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


# --- non-penalty goals + honest reporting (2026-08-17) ----------------------
def test_compute_per90_carries_non_penalty_goals():
    """FBref's "G-PK" was in the scraped table all along and simply never
    read, so the cold start fell back to goals INCLUDING penalties."""
    from data.ingestors.fbref_prior import compute_per90

    per90 = compute_per90(minutes=900, goals=10, assists=5, npxg=0.0, xa=0.0, npg=7)
    assert per90["goals90"] == 1.0
    assert per90["npg90"] == 0.7
    assert per90["npxg90"] == 0.0


def test_row_to_prior_stats_reads_the_g_pk_column():
    """The real flattened FBref column name, as it appears in the cache."""
    from data.ingestors.fbref_prior import row_to_prior_stats

    row = {
        "player": "A Player", "team": "A Team", "pos": "FW",
        "Playing Time Min": 900, "Playing Time MP": 10,
        "Performance Gls": 10, "Performance Ast": 5, "Performance G-PK": 7,
    }
    vals = row_to_prior_stats(row, "ESP-La Liga", "2025-2026")
    assert vals["npg90"] == 0.7
    assert vals["goals90"] == 1.0


def test_missing_metrics_are_named_rather_than_implied_by_a_row_count():
    """The ingest used to report "N rows written" for a scrape whose source
    carried no Expected columns at all, so a user re-running it on 2026-08-17
    was told it had repopulated when nothing had changed. Row count cannot
    distinguish real data from a column of defaults -- or from soccerdata
    caching a blocked response."""
    from data.ingestors.fbref_prior import report_missing_metrics

    rows = [
        {"goals90": 1.0, "assists90": 0.0, "npg90": 0.7, "npxg90": 0.0, "xa90": 0.0},
        {"goals90": 0.0, "assists90": 0.5, "npg90": 0.0, "npxg90": 0.0, "xa90": 0.0},
    ]
    missing = report_missing_metrics("ESP-La Liga", "2025-2026", rows)
    assert set(missing) == {"npxg90", "xa90"}
    assert "goals90" not in missing and "npg90" not in missing


def test_a_scrape_that_returned_nothing_is_reported_as_such():
    from data.ingestors.fbref_prior import report_missing_metrics

    assert report_missing_metrics("ESP-La Liga", "2025-2026", []) == [
        "goals90", "assists90", "npg90", "npxg90", "xa90"
    ]


def test_prior_league_projection_prefers_non_penalty_goals_over_raw_goals():
    """A prior-league penalty taker was flattered by raw goals. npg90 is the
    same quantity the in-season engine works in."""
    from projection.cold_start import _prior_league_projection

    base = {"league": "ESP-La Liga", "npxg90": 0.0, "xa90": 0.0,
            "assists90": 0.2, "minutes": 2700, "matches": 30}
    raw_only = _prior_league_projection("FWD", {**base, "goals90": 0.8, "npg90": 0.0})
    with_npg = _prior_league_projection("FWD", {**base, "goals90": 0.8, "npg90": 0.5})

    assert with_npg[0] < raw_only[0], "penalties must not inflate the projection"


def test_penalty_duty_reaches_a_prior_league_player_on_penalty_free_input():
    """The tier was excluded from the cold-start penalty bonus because raw
    goals90 baked in penalties taken abroad, unattributably. npxg90 and npg90
    are non-penalty by construction, so for those rows the double-count risk
    is gone and withholding the bonus just under-rates a designated taker."""
    from projection.cold_start import _prior_league_projection

    row = {"league": "ESP-La Liga", "npxg90": 0.5, "xa90": 0.2,
           "goals90": 0.6, "npg90": 0.5, "assists90": 0.2,
           "minutes": 2700, "matches": 30}
    without = _prior_league_projection("FWD", row)
    with_duty = _prior_league_projection("FWD", row, penalty_duty_rate=0.0806)
    assert with_duty[0] > without[0]
    assert with_duty[1] > without[1], "the extra goals carry extra variance too"


def test_penalty_duty_is_withheld_when_only_penalty_inclusive_goals_exist():
    """A Championship player has no Understat coverage and, if G-PK is also
    missing, projects from raw goals that already contain his penalties.
    Adding duty there would pay him twice."""
    from projection.cold_start import _prior_league_projection

    row = {"league": "ENG-Championship", "npxg90": 0.0, "xa90": 0.0,
           "goals90": 0.6, "npg90": 0.0, "assists90": 0.2,
           "minutes": 2700, "matches": 30}
    without = _prior_league_projection("FWD", row)
    with_duty = _prior_league_projection("FWD", row, penalty_duty_rate=0.0806)
    assert with_duty[0] == without[0]
