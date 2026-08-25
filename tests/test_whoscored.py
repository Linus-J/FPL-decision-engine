"""WhoScored raw-event adapter — pure aggregation (P10 follow-up).

FBref's summary table structurally lacks clearances/blocks/recoveries (P10);
WhoScored's raw Opta event stream carries them as distinct event types. These
tests cover the pure event->counts aggregation, not the live network/browser
path (ingest_whoscored_season, marked no-cover like the FBref equivalent).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from data.ingestors.whoscored import (
    _to_naive,
    aggregate_match_events,
    played_match_ids,
    sum_by_gameweek,
)


def _events(rows):
    cols = ["game_id", "player_id", "player", "type", "outcome_type"]
    return pd.DataFrame(rows, columns=cols)


def test_aggregate_counts_defensive_event_types():
    """Clearances, interceptions, blocks and recoveries are achieved outcomes
    by definition, so every event counts. Tackles are not — see the test
    below."""
    events = _events([
        (1, 10, "Alice", "Tackle", "Successful"),
        (1, 10, "Alice", "Tackle", "Unsuccessful"),   # a LOST tackle: does not count
        (1, 10, "Alice", "Interception", "Successful"),
        (1, 10, "Alice", "Clearance", "Successful"),
        (1, 10, "Alice", "BlockedPass", "Successful"),
        (1, 10, "Alice", "BallRecovery", "Successful"),
        (1, 11, "Bob", "Tackle", "Successful"),
    ])
    out = aggregate_match_events(events).set_index("player_id")
    assert out.loc[10, "tackles"] == 1
    assert out.loc[10, "interceptions"] == 1
    assert out.loc[10, "clearances"] == 1
    assert out.loc[10, "blocks"] == 1
    assert out.loc[10, "recoveries"] == 1
    assert out.loc[11, "tackles"] == 1
    assert out.loc[11, "clearances"] == 0


def test_aggregate_tackles_only_counts_won_tackles():
    """Regression, 2026-08-18 (engine review §5).

    This file previously asserted that a lost tackle counted, on the stated
    belief that "BPS/DefCon count attempts". They do not. Opta's definition,
    which FPL uses: *"A tackle is defined when a player connects with the ball
    in a legal, ground-level challenge and SUCCESSFULLY takes the ball away
    from the opposition player."* A lost tackle — contact made, possession to
    the opponent — is not a tackle for CBIT/CBIRT or for BPS's
    ``successful_tackle``.

    It mattered because two sources write this one column: FBref maps
    ``tackles`` to "Performance TklW" (won), and WhoScored OVERWRITES it when
    it patches a row. Counting attempts there put ~80% more tackles per 90 on
    the patched rows than the FBref-only ones (DEF 1.69 vs 0.92 on 2025-26),
    inflating both the highest-weight defensive BPS action and a hard DefCon
    threshold that a scale error does not wash out of.
    """
    events = _events([
        (1, 10, "Alice", "Tackle", "Successful"),
        (1, 10, "Alice", "Tackle", "Unsuccessful"),
        (1, 10, "Alice", "Tackle", "Unsuccessful"),
    ])
    out = aggregate_match_events(events).set_index("player_id")
    assert out.loc[10, "tackles"] == 1


def test_aggregate_dribbles_only_counts_successful_takeons():
    events = _events([
        (1, 10, "Alice", "TakeOn", "Successful"),
        (1, 10, "Alice", "TakeOn", "Unsuccessful"),
        (1, 10, "Alice", "TakeOn", "Successful"),
    ])
    out = aggregate_match_events(events).set_index("player_id")
    assert out.loc[10, "dribbles"] == 2


def test_aggregate_ignores_events_without_a_player():
    events = _events([
        (1, None, None, "FormationChange", None),
        (1, 10, "Alice", "Tackle", "Successful"),
    ])
    out = aggregate_match_events(events)
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == 10


def test_aggregate_ignores_untracked_event_types():
    events = _events([(1, 10, "Alice", "Pass", "Successful")])
    out = aggregate_match_events(events)
    assert out.empty


def test_aggregate_empty_input():
    out = aggregate_match_events(_events([]))
    assert out.empty


def test_to_naive_strips_tz_and_leaves_naive_alone():
    aware = pd.Timestamp("2025-08-15 15:00:00", tz="UTC")
    assert _to_naive(aware) == pd.Timestamp("2025-08-15 15:00:00")
    assert _to_naive(aware).tzinfo is None
    naive = datetime(2025, 8, 15, 15, 0, 0)
    assert _to_naive(naive) == naive


def test_sum_by_gameweek_handles_tz_aware_kickoffs():
    # the live bug: WhoScored's schedule dates come back tz-aware while
    # deadlines round-trip through SQLite as naive -- assign_gameweek must
    # not blow up when kickoff_of already has tz-aware values (belt-and-
    # suspenders in case a caller forgets to normalise before this call)
    agg = pd.DataFrame([
        {"game_id": 1, "player": "Alice", "tackles": 1, "interceptions": 0,
         "clearances": 0, "blocks": 0, "recoveries": 0, "dribbles": 0},
    ])
    kickoff_of = {1: _to_naive(pd.Timestamp("2025-08-16 15:00:00", tz="UTC"))}
    deadlines = {5: datetime(2025, 8, 15)}
    name_map = {"alice": 10}
    totals, unmatched = sum_by_gameweek(agg, kickoff_of, deadlines, name_map)
    assert unmatched == 0
    assert totals[(10, 5)]["tackles"] == 1


def test_sum_by_gameweek_matches_and_sums_dgw():
    agg = pd.DataFrame([
        {"game_id": 1, "player": "Alice", "tackles": 2, "interceptions": 0,
         "clearances": 1, "blocks": 0, "recoveries": 0, "dribbles": 0},
        {"game_id": 2, "player": "Alice", "tackles": 1, "interceptions": 1,
         "clearances": 0, "blocks": 0, "recoveries": 0, "dribbles": 0},
    ])
    kickoff_of = {1: datetime(2025, 8, 16), 2: datetime(2025, 8, 18)}  # same DGW week
    deadlines = {5: datetime(2025, 8, 15)}
    name_map = {"alice": 10}
    totals, unmatched = sum_by_gameweek(agg, kickoff_of, deadlines, name_map)
    assert unmatched == 0
    assert totals[(10, 5)]["tackles"] == 3
    assert totals[(10, 5)]["interceptions"] == 1
    assert totals[(10, 5)]["clearances"] == 1


def test_sum_by_gameweek_unmatched_player_or_missing_kickoff():
    agg = pd.DataFrame([
        {"game_id": 1, "player": "Unknown Player", "tackles": 1, "interceptions": 0,
         "clearances": 0, "blocks": 0, "recoveries": 0, "dribbles": 0},
        {"game_id": 99, "player": "Alice", "tackles": 1, "interceptions": 0,
         "clearances": 0, "blocks": 0, "recoveries": 0, "dribbles": 0},
    ])
    kickoff_of = {1: datetime(2025, 8, 16)}  # game_id 99 has no kickoff
    deadlines = {5: datetime(2025, 8, 15)}
    name_map = {"alice": 10}
    totals, unmatched = sum_by_gameweek(agg, kickoff_of, deadlines, name_map)
    assert unmatched == 2
    assert totals == {}


def test_played_match_ids_excludes_future_fixtures():
    """read_events(match_id=None) walks the WHOLE schedule. One gameweek into
    26/27 that meant driving a browser through all 380 fixtures to collect
    events from the 10 played -- slow, visible, and the kind of traffic that
    gets a scraper blocked.
    """
    now = datetime(2026, 8, 25, 12, 0)
    kickoff_of = {
        101: datetime(2026, 8, 21, 19, 0),   # played, days ago
        102: datetime(2026, 8, 24, 19, 0),   # played, yesterday
        103: datetime(2026, 8, 28, 19, 0),   # future -- GW2
        104: datetime(2026, 9, 4, 19, 0),    # future -- GW3
    }

    assert played_match_ids(kickoff_of, now=now) == [101, 102]


def test_played_match_ids_excludes_a_match_still_in_progress():
    """A match that kicked off 30 minutes ago is not over, and its event
    stream is incomplete. MATCH_SETTLE_HOURS covers stoppage, half-time and
    WhoScored's own publishing lag."""
    now = datetime(2026, 8, 25, 20, 0)
    kickoff_of = {
        201: datetime(2026, 8, 25, 19, 30),  # 30 min ago -- in progress
        202: datetime(2026, 8, 25, 15, 0),   # 5h ago -- done
    }

    assert played_match_ids(kickoff_of, now=now) == [202]


def test_played_match_ids_excludes_undated_rows():
    """An unknown kickoff cannot be shown to be in the past, and
    assign_gameweek could not place its events anyway."""
    now = datetime(2026, 8, 25, 12, 0)
    kickoff_of = {301: None, 302: datetime(2026, 8, 21, 19, 0)}

    assert played_match_ids(kickoff_of, now=now) == [302]


def test_played_match_ids_empty_when_season_has_not_started():
    now = datetime(2026, 8, 1, 12, 0)
    kickoff_of = {401: datetime(2026, 8, 21, 19, 0)}

    assert played_match_ids(kickoff_of, now=now) == []


def test_ingest_skips_the_browser_when_there_are_no_rows_to_patch(monkeypatch):
    """This ingest only UPDATEs rows FBref wrote. With none for the season,
    opening a browser computes counts that are then discarded.

    Reachable in practice: on 2026-08-25 the FBref ingest refused 2026-27
    outright (soccerdata season-code collision) and run_weekly.py called this
    straight afterwards anyway, since every step is warn-and-continue.
    """
    import data.ingestors.whoscored as ws_module

    monkeypatch.setattr(ws_module, "_existing_event_rows", lambda season: 0)

    def _boom(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("must not construct a browser session")

    monkeypatch.setitem(__import__("sys").modules, "soccerdata", type(
        "sd", (), {"WhoScored": staticmethod(_boom)},
    ))

    assert ws_module.ingest_whoscored_season("2026-27") == (0, 0)
