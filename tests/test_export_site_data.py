"""scripts/export_site_data.py"""

from __future__ import annotations

from unittest.mock import MagicMock

import scripts.export_site_data as cli


def test_run_wires_payload_write_index_and_commit_in_order(monkeypatch, tmp_path):
    calls = []
    fake_payload = {"gameweek": 3, "label": "GW3 — 3 Aug", "generated_at": "t"}

    def fake_write_run_file(out_dir, gw, payload):
        calls.append(("write", gw))
        path = tmp_path / f"gw{gw}.json"
        path.write_text("{}")
        return path

    monkeypatch.setattr(cli, "get_session", lambda: MagicMock())
    monkeypatch.setattr(
        cli, "build_run_payload",
        lambda db, team_id: (calls.append(("build", team_id)), fake_payload)[1],
    )
    monkeypatch.setattr(cli, "write_run_file", fake_write_run_file)
    monkeypatch.setattr(
        cli, "update_index",
        lambda out_dir, gw, label, generated_at: (
            calls.append(("index", gw)), tmp_path / "index.json"
        )[1],
    )
    monkeypatch.setattr(
        cli, "commit_and_push",
        lambda repo_root, data_dir, message, push: (calls.append(("commit", push)), True)[1],
    )
    monkeypatch.setattr(cli.settings, "fpl_team_id", 99999)

    cli.run(no_push=True)

    assert calls == [("build", 99999), ("write", 3), ("index", 3), ("commit", False)]
