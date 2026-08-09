"""scripts/site_export/writer.py"""

from __future__ import annotations

import json

from scripts.site_export import writer


def test_write_run_file_creates_dir_and_file(tmp_path):
    out_dir = tmp_path / "simulations"
    payload = {"schema_version": 1, "gameweek": 3, "squad": []}

    path = writer.write_run_file(out_dir, gw=3, payload=payload)

    assert path == out_dir / "gw3.json"
    assert json.loads(path.read_text()) == payload


def test_update_index_creates_new_index_when_absent(tmp_path):
    out_dir = tmp_path / "simulations"

    path = writer.update_index(out_dir, gw=3, label="GW3 — 3 Aug", generated_at="2026-08-03T06:00:00Z")

    index = json.loads(path.read_text())
    assert index == {
        "schema_version": 1,
        "runs": [{"id": "gw3", "gameweek": 3, "label": "GW3 — 3 Aug", "generated_at": "2026-08-03T06:00:00Z"}],
    }


def test_update_index_replaces_existing_entry_for_same_gw(tmp_path):
    out_dir = tmp_path / "simulations"
    writer.update_index(out_dir, gw=3, label="GW3 — old label", generated_at="2026-08-03T06:00:00Z")

    path = writer.update_index(out_dir, gw=3, label="GW3 — 3 Aug", generated_at="2026-08-03T07:00:00Z")

    index = json.loads(path.read_text())
    assert len(index["runs"]) == 1
    assert index["runs"][0]["label"] == "GW3 — 3 Aug"


def test_update_index_keeps_runs_sorted_most_recent_gw_first(tmp_path):
    out_dir = tmp_path / "simulations"
    writer.update_index(out_dir, gw=2, label="GW2", generated_at="t2")
    writer.update_index(out_dir, gw=4, label="GW4", generated_at="t4")
    path = writer.update_index(out_dir, gw=3, label="GW3", generated_at="t3")

    index = json.loads(path.read_text())
    assert [r["gameweek"] for r in index["runs"]] == [4, 3, 2]
