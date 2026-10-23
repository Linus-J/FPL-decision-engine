"""Run manifests: which code, strategy configuration and inputs produced a run.

A decision, a backtest and a simulation batch each record one ``RunManifest``
row before their results are written. The decision and simulation writers
then stamp their rows with the manifest id, taken from a context variable so
the call sites in between do not need to change.

Settings are deliberately not recorded. They hold secrets and the team id, and
the strategy configuration is what determines a decision.
"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import config.strategy as strategy
from data.db import get_session
from data.models import RunManifest

REPO_ROOT = Path(__file__).resolve().parents[1]

_current_manifest_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "current_manifest_id", default=None
)


def current_manifest_id() -> int | None:
    """The manifest id of the run in progress, or None outside a recorded run."""
    return _current_manifest_id.get()


@contextmanager
def manifest_context(manifest_id: int | None) -> Iterator[None]:
    """Make ``manifest_id`` the current run for the duration of the block."""
    token = _current_manifest_id.set(manifest_id)
    try:
        yield
    finally:
        _current_manifest_id.reset(token)


def strategy_config() -> dict[str, Any]:
    """Every upper-case strategy constant in ``config/strategy.py``, as plain data.

    Dataclass instances are expanded field by field. Anything that is not a
    plain value or a dataclass is left out rather than guessed at, so the
    record holds only what it can represent exactly.
    """
    out: dict[str, Any] = {}
    for name, value in vars(strategy).items():
        if name.startswith("_") or not name.isupper():
            continue
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            out[name] = dataclasses.asdict(value)
        elif isinstance(value, (bool, int, float, str, list, tuple, dict)) or value is None:
            out[name] = value
    return out


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_state() -> tuple[str | None, bool]:
    """(HEAD sha, whether tracked files differ from it). Both are None or False
    when git is unavailable, and the manifest says so rather than guessing."""
    sha = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    return sha, bool(status)


def record_run(
    kind: str,
    *,
    season: str,
    gameweek: int | None = None,
    seed: int | None = None,
    archive_path: Path | None = None,
    output_path: Path | None = None,
    notes: str | None = None,
    config: dict[str, Any] | None = None,
) -> int:
    """Write one manifest row and return its id. Call before the run's results
    are recorded, so every result can be matched to it."""
    resolved = strategy_config() if config is None else config
    blob = json.dumps(resolved, sort_keys=True, default=str)
    sha, dirty = git_state()
    db = get_session()
    try:
        row = RunManifest(
            kind=kind,
            season=season,
            gameweek=gameweek,
            git_sha=sha,
            git_dirty=dirty,
            config_sha256=hashlib.sha256(blob.encode()).hexdigest(),
            config_json=blob,
            seed=seed,
            archive_path=str(archive_path) if archive_path else None,
            output_path=str(output_path) if output_path else None,
            notes=notes,
            created_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        return row.id
    finally:
        db.close()
