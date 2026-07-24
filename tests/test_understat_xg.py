"""Understat per-match xG ingest — pure parsers + key-passes aggregation."""

from __future__ import annotations

from datetime import datetime

from data.ingestors.fbref import aggregate_xg_rows
from data.ingestors.understat_xg import parse_game_date, understat_row_to_xg


def test_parse_game_date():
    assert parse_game_date("2025-08-15 Liverpool-Bournemouth") == datetime(2025, 8, 15)
    assert parse_game_date("") is None
    assert parse_game_date("not a date") is None


def test_understat_row_to_xg():
    out = understat_row_to_xg({"xg": 0.42, "xa": 0.18, "shots": 3, "key_passes": 2})
    assert out == {"xg": 0.42, "npxg": 0.42, "xa": 0.18, "shots": 3, "key_passes": 2}
    # npxg mirrors xg (no penalty split); missing fields → 0
    assert understat_row_to_xg({"xg": 0.5}) == {
        "xg": 0.5, "npxg": 0.5, "xa": 0.0, "shots": 0, "key_passes": 0
    }


def test_aggregate_sums_key_passes_dgw():
    per_match = [
        (1, 5, {"xg": 0.4, "xa": 0.1, "npxg": 0.4, "shots": 2, "key_passes": 1}),
        (1, 5, {"xg": 0.2, "xa": 0.3, "npxg": 0.2, "shots": 1, "key_passes": 2}),
    ]
    agg = aggregate_xg_rows(per_match)
    assert agg[(1, 5)]["key_passes"] == 3
    assert agg[(1, 5)]["xg"] == 0.6
    assert agg[(1, 5)]["xgi"] == 1.0   # xg 0.6 + xa 0.4
