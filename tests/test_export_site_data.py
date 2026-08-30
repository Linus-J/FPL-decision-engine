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


# --- jsDelivr purge after publishing (2026-08-30) ------------------------
#
# Pushing does not make the site current: jsDelivr pins a mutable branch ref
# for up to seven days, so the site served a five-day-old gw2.json while
# origin/v2 had the corrected squad. The export now invalidates the two
# files it just published.


def _wire(monkeypatch, tmp_path, calls, *, committed=True):
    fake_payload = {"gameweek": 3, "label": "GW3 — 3 Aug", "generated_at": "t"}

    def fake_write_run_file(out_dir, gw, payload):
        path = tmp_path / f"gw{gw}.json"
        path.write_text("{}")
        return path

    monkeypatch.setattr(cli, "get_session", lambda: MagicMock())
    monkeypatch.setattr(cli, "build_run_payload", lambda db, team_id: fake_payload)
    monkeypatch.setattr(cli, "write_run_file", fake_write_run_file)
    monkeypatch.setattr(
        cli, "update_index",
        lambda out_dir, gw, label, generated_at: tmp_path / "index.json",
    )
    monkeypatch.setattr(
        cli, "commit_and_push", lambda repo_root, data_dir, message, push: committed
    )
    monkeypatch.setattr(
        cli, "purge_cdn",
        lambda **kwargs: (calls.append(kwargs), True)[1],
    )
    monkeypatch.setattr(cli.settings, "fpl_team_id", 99999)


def test_run_purges_the_index_and_the_run_file_after_pushing(monkeypatch, tmp_path):
    calls = []
    _wire(monkeypatch, tmp_path, calls)

    cli.run(no_push=False)

    assert len(calls) == 1
    assert calls[0]["files"] == ["index.json", "gw3.json"], (
        "a fresh gw3.json behind a stale index.json is still a stale page"
    )


def test_run_purges_the_ref_the_site_actually_fetches(monkeypatch, tmp_path):
    """The purge key is the request URL, so these must match the constants in
    the site's assets/panels/fpl.js. The repo was renamed from FPL-26-27-bot
    and GitHub 301s the old name, but jsDelivr caches per spelling -- purging
    the old name clears a key the site never requests."""
    calls = []
    _wire(monkeypatch, tmp_path, calls)

    cli.run(no_push=False)

    assert calls[0]["repo"] == "Linus-J/FPL-decision-engine"
    assert calls[0]["ref"] == "refs/heads/v2"
    assert calls[0]["path"] == "data/simulations"


def test_run_does_not_purge_when_nothing_was_pushed(monkeypatch, tmp_path):
    calls = []
    _wire(monkeypatch, tmp_path, calls, committed=False)

    cli.run(no_push=False)

    assert calls == []


def test_run_does_not_purge_in_no_push_mode(monkeypatch, tmp_path):
    """--no-push leaves the commit local, so the CDN has nothing new to fetch
    and a purge would only cost a round trip."""
    calls = []
    _wire(monkeypatch, tmp_path, calls)

    cli.run(no_push=True)

    assert calls == []
