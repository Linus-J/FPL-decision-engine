"""scripts/run_weekly.py -- FBref must run headed, not its own headless default.

Real bug found 2026-08-01, live-testing on the user's machine:
scrape_fbref.py defaults to headless (FBREF_HEADED unset), but FBref sits
behind Cloudflare and headless mode cannot clear its CAPTCHA -- the real
run hit "CAPTCHA detected... attempting to solve" and failed. run_weekly.py
never overrode the default, so its "automatic" weekly refresh was
guaranteed to fail every time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_weekly as run_weekly


def test_fbref_scrape_is_forced_headed_by_default(monkeypatch):
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(args, cwd=None, env=None):
        calls.append((args, env))
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.delenv("FBREF_HEADED", raising=False)
    monkeypatch.setattr(run_weekly, "_current_gameweek", lambda: None)

    monkeypatch.setattr(sys, "argv", ["run_weekly.py", "--dry-run"])
    run_weekly.main()

    fbref_call = next(c for c in calls if "scripts/scrape_fbref.py" in c[0])
    assert fbref_call[1] is not None
    assert fbref_call[1]["FBREF_HEADED"] == "1"


def test_fbref_scrape_respects_an_explicit_headed_override(monkeypatch):
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(args, cwd=None, env=None):
        calls.append((args, env))
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setenv("FBREF_HEADED", "0")  # user explicitly wants headless
    monkeypatch.setattr(run_weekly, "_current_gameweek", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_weekly.py", "--dry-run"])
    run_weekly.main()

    fbref_call = next(c for c in calls if "scripts/scrape_fbref.py" in c[0])
    assert fbref_call[1]["FBREF_HEADED"] == "0"


def test_skip_match_events_never_invokes_fbref(monkeypatch):
    calls: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(args, cwd=None, env=None):
        calls.append(args)
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(run_weekly, "_current_gameweek", lambda: None)
    monkeypatch.setattr(
        sys, "argv", ["run_weekly.py", "--dry-run", "--skip-match-events"]
    )
    run_weekly.main()

    assert not any("scripts/scrape_fbref.py" in c for c in calls)
