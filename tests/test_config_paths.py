"""Configuration must not depend on where the process was started.

Both fixes here are the same bug in two places, and it has bitten this project
twice for real: a relative path resolved against the cwd, with a silent
fallback rather than an error. The live crontab spent five weeks reading a
stale, wrong-schema database (2026-07-31), and the odds ingest silently
skipped — leaving every fixture on a flat league-average scoreline — when run
from a worktree three days before the 2026-27 GW1 deadline.
"""

from pathlib import Path

from config.settings import _ENV_FILE, _REPO_ROOT
from data.db import _resolve_db_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_file_is_absolute_and_at_the_repo_root():
    """Every Settings field has a default, so a missed .env raises nothing —
    the process just runs with an empty API key and the wrong database."""
    assert _ENV_FILE.is_absolute()
    assert _ENV_FILE == REPO_ROOT / ".env"
    assert _REPO_ROOT == REPO_ROOT


def test_relative_db_path_anchors_to_the_repo_root_not_the_cwd():
    resolved = _resolve_db_path("fpl_bot_v2.db")
    assert Path(resolved).is_absolute()
    assert resolved == str(REPO_ROOT / "fpl_bot_v2.db")


def test_absolute_db_path_is_left_alone():
    """An explicit absolute path is the clearest thing to put in .env and must
    survive untouched — including the one the test suite itself injects."""
    assert _resolve_db_path("/var/lib/fpl/live.db") == "/var/lib/fpl/live.db"


def test_sqlite_special_forms_are_not_treated_as_paths():
    assert _resolve_db_path(":memory:") == ":memory:"
