"""P11 prior-league ingest — pure per-90 + row-mapping helpers (network-free)."""

from __future__ import annotations

from data.ingestors import fbref_prior as fp


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
