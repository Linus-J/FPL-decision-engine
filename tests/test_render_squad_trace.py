"""Regression test for scripts/render_squad_trace.py's captain-doubling
display bug, found 2026-07-28 (user's own review of the rendered report):
the table showed every player's RAW actual points including the captain,
so summing the displayed column looked like it was missing the captain's
doubled contribution -- even though the underlying backtest scoring
(scripts/backtest.py::_score_squad) already applies it correctly. This
tests the display only; _score_squad's own correctness is a separate,
pre-existing concern (unaffected by this fix)."""

from __future__ import annotations

from scripts.render_squad_trace import _squad_table


def _player(**overrides) -> dict:
    base = {
        "id": 1, "web_name": "Haaland", "position": "FWD", "now_cost": 14.0,
        "xpts": 8.0, "is_starting": True, "is_captain": False,
        "is_vice_captain": False, "bench_order": -1, "actual_pts": 16,
    }
    base.update(overrides)
    return base


def test_captain_shown_at_double_points_normally():
    table = _squad_table([_player(is_captain=True, actual_pts=16)], chip=None)
    assert "| 32 |" in table
    assert "(C x2)" in table


def test_captain_shown_at_triple_points_on_triple_captain_chip():
    table = _squad_table([_player(is_captain=True, actual_pts=16)], chip="3xc")
    assert "| 48 |" in table
    assert "(C x3)" in table


def test_non_captain_shown_at_raw_points():
    table = _squad_table([_player(is_captain=False, actual_pts=9)], chip=None)
    assert "| 9 |" in table


def test_displayed_total_matches_official_scoring():
    # The exact bug: naive-sum-of-displayed-column must now equal what
    # _score_squad would actually credit (raw sum + one extra captain
    # multiple), not just the raw sum.
    squad = [
        _player(id=1, web_name="Haaland", is_captain=True, actual_pts=16),
        _player(id=2, web_name="Semenyo", is_captain=False, actual_pts=9,
                is_vice_captain=True),
    ]
    table = _squad_table(squad, chip=None)
    displayed_total = sum(
        int(line.split("|")[-2].strip())
        for line in table.splitlines()[2:]
    )
    assert displayed_total == 16 * 2 + 9  # matches _score_squad's own doubling
