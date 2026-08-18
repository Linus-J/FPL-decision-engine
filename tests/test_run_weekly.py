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
    # This test is about the headed-mode env var; pretend the season is under
    # way so the pre-season guard does not skip the scrape out from under it.
    monkeypatch.setattr(run_weekly, "_season_has_started", lambda season: True)

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
    monkeypatch.setattr(run_weekly, "_season_has_started", lambda season: True)
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


def test_match_event_scrapes_are_skipped_before_the_season_starts(monkeypatch):
    """Regression, 2026-08-18.

    The match-event scrapers exist to collect what happened in matches. Run
    against a season with no played gameweeks they have nothing to fetch, and
    FBref does not simply return empty -- asked for a season it has no match
    reports for, the scrape wandered off pulling unrelated archive pages (a
    1926-1927 one was observed), spending browser time against Cloudflare on
    data that could only ever be discarded.

    Nothing was ever written -- the ingest keys on the requested season, so the
    junk had nowhere to land -- but a step whose only outcomes are "no-op" and
    "wrong" should not run.
    """
    calls: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(args, cwd=None, env=None):
        calls.append(args)
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(run_weekly, "_current_gameweek", lambda: None)
    monkeypatch.setattr(run_weekly, "_season_has_started", lambda season: False)
    monkeypatch.setattr(sys, "argv", ["run_weekly.py", "--dry-run"])
    run_weekly.main()

    invoked = " ".join(a for call in calls for a in call)
    assert "scrape_fbref.py" not in invoked
    assert "scrape_whoscored.py" not in invoked
    assert "scrape_setpieces.py" not in invoked
    # The decision itself must still be made -- this is a pre-season SKIP of
    # data collection, not of the weekly run.
    assert "run_agent.py" in invoked


def test_match_event_scrapes_run_once_the_season_is_under_way(monkeypatch):
    """The other half: the guard must open again by itself, with no flag to
    remember to unset."""
    calls: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(args, cwd=None, env=None):
        calls.append(args)
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(run_weekly, "_current_gameweek", lambda: None)
    monkeypatch.setattr(run_weekly, "_season_has_started", lambda season: True)
    monkeypatch.setattr(sys, "argv", ["run_weekly.py", "--dry-run"])
    run_weekly.main()

    invoked = " ".join(a for call in calls for a in call)
    assert "scrape_fbref.py" in invoked
    assert "scrape_whoscored.py" in invoked


def test_an_unreadable_database_does_not_silently_skip_collection(monkeypatch):
    """Failing to answer "has the season started" must default to RUNNING the
    scrapes. Pre-season the cost of being wrong is a wasted scrape; in-season
    it is a week of stale DefCon and bonus data feeding a real decision."""
    monkeypatch.setattr(
        run_weekly, "season_has_played_history", None, raising=False
    )

    def _boom(season):
        raise RuntimeError("database is locked")

    import projection.pipeline as pipeline

    monkeypatch.setattr(pipeline, "season_has_played_history", _boom)
    assert run_weekly._season_has_started("2026-27") is True
