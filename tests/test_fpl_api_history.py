"""data/ingestors/fpl_api.py — per-fixture gameweek history rows.

The live ingest used to pre-sum a gameweek into ONE row and hardcode
``opponent_team_id`` to a sentinel, leaving ``team_id_season``/``was_home``
NULL on every live-season row. That pre-summing was a workaround for the OLD
3-column unique key; the constraint has since been widened to
``(player_id, gameweek, season, opponent_team_id)``, which is exactly what
``scripts/backfill_history.py`` already writes against — one row per fixture.

Consequence of the old shape, found on 2026-08-28: every odds and FDR join in
``projection/features.py`` keys on
``team_id_season``/``opponent_team_id``/``was_home``, so for 2026-27 they could
never match and fell through to their COALESCE defaults —
``my_cs_prob``/``opp_cs_prob`` pinned at 0.2 and ``over25_prob`` at 0.5 on all
571 rows, against real variation in the data the model was fitted on. The same
three columns feed the Dixon-Coles fit and DGW fixture pairing.
"""

from __future__ import annotations

from data.ingestors.fpl_api import (
    _NO_OPPONENT_SENTINEL,
    _history_rows,
    _resolve_team_id_season,
)


def test_single_fixture_gameweek_carries_stats_and_fixture_context():
    history = [{
        "round": 10, "fixture": 95, "opponent_team": 12, "was_home": False,
        "total_points": 6, "minutes": 90, "selected": 500, "value": 55,
    }]
    rows = _history_rows(history)
    assert list(rows) == [(10, 12)]
    row = rows[(10, 12)]
    assert row["total_points"] == 6
    assert row["minutes"] == 90
    assert row["selected"] == 500
    assert row["value"] == 5.5
    assert row["opponent_team_id"] == 12
    assert row["was_home"] is False
    assert row["fpl_fixture_id"] == 95


def test_double_gameweek_yields_one_row_per_fixture_not_a_sum():
    """The whole point of the widened constraint: both fixtures survive."""
    history = [
        {"round": 12, "fixture": 120, "opponent_team": 3, "was_home": True,
         "total_points": 6, "minutes": 90, "goals_scored": 1, "bps": 30,
         "selected": 500, "value": 55},
        {"round": 12, "fixture": 121, "opponent_team": 8, "was_home": False,
         "total_points": 2, "minutes": 90, "goals_scored": 0, "bps": 10,
         "selected": 510, "value": 56},
    ]
    rows = _history_rows(history)

    assert sorted(rows) == [(12, 3), (12, 8)]
    assert rows[(12, 3)]["total_points"] == 6
    assert rows[(12, 3)]["bps"] == 30
    assert rows[(12, 3)]["was_home"] is True
    assert rows[(12, 8)]["total_points"] == 2
    assert rows[(12, 8)]["bps"] == 10
    assert rows[(12, 8)]["was_home"] is False


def test_snapshot_fields_are_per_row_not_summed():
    """selected/value are point-in-time squad snapshots, never per-fixture sums."""
    history = [
        {"round": 12, "fixture": 120, "opponent_team": 3, "was_home": True,
         "total_points": 6, "selected": 500, "value": 55},
        {"round": 12, "fixture": 121, "opponent_team": 8, "was_home": False,
         "total_points": 2, "selected": 510, "value": 56},
    ]
    rows = _history_rows(history)
    assert rows[(12, 3)]["selected"] == 500
    assert rows[(12, 3)]["value"] == 5.5
    assert rows[(12, 8)]["selected"] == 510
    assert rows[(12, 8)]["value"] == 5.6


def test_repeated_entries_for_one_fixture_are_summed():
    """Defensive: same (round, opponent) twice must not violate the unique key."""
    history = [
        {"round": 5, "fixture": 50, "opponent_team": 9, "was_home": True,
         "total_points": 3, "minutes": 45},
        {"round": 5, "fixture": 50, "opponent_team": 9, "was_home": True,
         "total_points": 2, "minutes": 45},
    ]
    rows = _history_rows(history)
    assert list(rows) == [(5, 9)]
    assert rows[(5, 9)]["total_points"] == 5
    assert rows[(5, 9)]["minutes"] == 90


def test_missing_round_is_skipped():
    history = [{"round": None, "total_points": 6},
               {"round": 5, "fixture": 50, "opponent_team": 9, "total_points": 3}]
    rows = _history_rows(history)
    assert list(rows) == [(5, 9)]


def test_missing_opponent_falls_back_to_the_sentinel():
    """Never NULL: SQLite treats two NULLs as distinct, so ON CONFLICT would
    insert a fresh duplicate row on every re-run."""
    history = [{"round": 5, "total_points": 3}]
    rows = _history_rows(history)
    assert list(rows) == [(5, _NO_OPPONENT_SENTINEL)]
    assert rows[(5, _NO_OPPONENT_SENTINEL)]["opponent_team_id"] == _NO_OPPONENT_SENTINEL


def test_resolve_team_id_season_picks_the_players_own_side():
    sides = {95: (13, 12)}  # (team_h_id, team_a_id)
    assert _resolve_team_id_season(sides, 95, was_home=True) == 13
    assert _resolve_team_id_season(sides, 95, was_home=False) == 12


def test_resolve_team_id_season_is_none_when_unresolvable():
    """A fixture we have not ingested, or an entry with no home/away flag."""
    assert _resolve_team_id_season({}, 95, was_home=True) is None
    assert _resolve_team_id_season({95: (13, 12)}, 95, was_home=None) is None
    assert _resolve_team_id_season({95: (13, 12)}, None, was_home=True) is None
