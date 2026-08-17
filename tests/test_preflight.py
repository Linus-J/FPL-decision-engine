"""scripts/preflight.py — the decision-surface guard.

The DB-backed checks are exercised against the live database by running the
script; what is pinned here is the logic that decides pass/fail, because a
guard that cannot fail is worse than no guard -- it manufactures confidence.
"""

from __future__ import annotations

import json

import pytest

from scripts import preflight


def test_result_records_failures_and_keeps_going():
    """One run must report every problem. Stopping at the first turns a
    thorough pass into a game of whack-a-mole."""
    r = preflight.Result()
    r.check(True, "fine")
    r.check(False, "broken one")
    r.check(False, "broken two", "detail")
    assert r.failures == ["broken one", "broken two"]


def test_baseline_drift_is_reported_per_field(tmp_path, monkeypatch, capsys):
    """The check that would have caught every self-inflicted defect on
    2026-08-17: the tests passed, the gate passed, and the ANSWER changed."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"captain": "B.Fernandes", "cost": 100.0}))
    monkeypatch.setattr(preflight, "BASELINE_PATH", baseline)

    r = preflight.Result()
    preflight.compare_to_baseline({"captain": "Haaland", "cost": 100.0}, r, update=False)

    assert r.failures, "a changed captain must fail the run"
    out = capsys.readouterr().out
    assert "B.Fernandes" in out and "Haaland" in out, "must show was/now, not just 'changed'"
    assert "cost" not in out.split("DRIFT")[1], "unchanged fields must not be reported"


def test_an_unchanged_surface_passes(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    snapshot = {"captain": "B.Fernandes", "squad": ["a", "b"]}
    baseline.write_text(json.dumps(snapshot))
    monkeypatch.setattr(preflight, "BASELINE_PATH", baseline)

    r = preflight.Result()
    preflight.compare_to_baseline(dict(snapshot), r, update=False)
    assert r.failures == []


def test_a_new_or_removed_field_counts_as_drift(tmp_path, monkeypatch):
    """Adding a field to the snapshot changes what is being guarded, so it
    has to be accepted deliberately rather than slipping in."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"captain": "B.Fernandes"}))
    monkeypatch.setattr(preflight, "BASELINE_PATH", baseline)

    r = preflight.Result()
    preflight.compare_to_baseline({"captain": "B.Fernandes", "new_field": 1}, r, update=False)
    assert r.failures


def test_update_baseline_writes_the_snapshot(tmp_path, monkeypatch):
    baseline = tmp_path / "nested" / "baseline.json"
    monkeypatch.setattr(preflight, "BASELINE_PATH", baseline)

    r = preflight.Result()
    preflight.compare_to_baseline({"captain": "Haaland"}, r, update=True)
    assert r.failures == []
    assert json.loads(baseline.read_text()) == {"captain": "Haaland"}


def test_missing_baseline_is_not_a_failure(tmp_path, monkeypatch):
    """A fresh clone has no baseline. That should prompt, not block."""
    monkeypatch.setattr(preflight, "BASELINE_PATH", tmp_path / "absent.json")
    r = preflight.Result()
    preflight.compare_to_baseline({"captain": "x"}, r, update=False)
    assert r.failures == []


def test_the_committed_baseline_matches_the_squad_in_the_checklist():
    """The checklist is what gets typed into FPL. It has been stale once
    already -- listing a bench decision_log did not contain -- so tie the two
    together rather than trusting them to be updated in step."""
    from pathlib import Path

    baseline = json.loads(preflight.BASELINE_PATH.read_text())
    checklist = (Path(__file__).resolve().parents[1] / "docs" / "gw1-checklist.md").read_text()

    for name in baseline["starting_xi"]:
        assert name in checklist, f"{name} is in the recorded XI but not the checklist"
    for name in baseline["bench_order"]:
        assert name in checklist, f"{name} is on the recorded bench but not the checklist"
    assert baseline["captain"] in checklist


@pytest.mark.parametrize("quota", [preflight.POSITION_QUOTA])
def test_squad_rules_match_fpl(quota):
    """Guards against a well-meaning edit to the constants themselves."""
    assert quota == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert sum(quota.values()) == preflight.SQUAD_SIZE == 15
    assert preflight.MAX_PER_CLUB == 3
    assert preflight.BUDGET == 100.0
