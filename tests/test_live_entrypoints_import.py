"""Real bug found 2026-07-30 (the user's own live-smoke-test request):
agent/decision_engine.py has imported a function (`get_dgw_coverage`) that
never existed anywhere in the codebase since its very first commit -- an
ImportError at module load, meaning the live agent could never run at all.
Undetected because the whole test suite is backtest-focused and nothing
ever imports agent.decision_engine or scripts.run_agent.

This doesn't test BEHAVIOUR -- just that the modules the real cron/live
path depends on still import cleanly, so a broken import in a module nothing
else touches can't silently sit undetected again the way this one did.
"""

from __future__ import annotations

import importlib


def test_agent_decision_engine_imports_cleanly():
    importlib.import_module("agent.decision_engine")


def test_agent_fpl_client_imports_cleanly():
    importlib.import_module("agent.fpl_client")


def test_agent_notifier_imports_cleanly():
    importlib.import_module("agent.notifier")


def test_scripts_run_agent_imports_cleanly():
    importlib.import_module("scripts.run_agent")
