"""Raw input archive: the FPL responses a run was made from, kept as received.

Every derived table in the database is a rendering of these responses, and
several of them are overwritten in place. The archive is the only copy of
what FPL said at a given moment, so a past gameweek can be replayed exactly
rather than rebuilt from derived tables.

Each capture is a directory named by its UTC timestamp, holding the raw JSON
and a manifest that records a SHA-256 for each file. Nothing here is ever
rewritten: a second capture gets its own directory. The archive lives at the
repository root, outside git, next to the database.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = REPO_ROOT / "archive"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def _encode(payload: Any) -> bytes:
    # sort_keys makes the bytes a function of the content alone, so the hash
    # does not change if FPL reorders keys within an object.
    return json.dumps(payload, indent=2, sort_keys=True, default=str).encode()


def write_input_archive(
    season: str,
    captured_at: datetime,
    payloads: dict[str, Any],
    gameweek_context: int | None,
    root: Path = ARCHIVE_ROOT,
) -> Path:
    """Write each payload as ``<name>.json`` under ``root/season/<stamp>/``.

    Raises ``FileExistsError`` if the capture directory already exists rather
    than overwriting it: two captures in the same microsecond are a bug to
    surface, not a file to clobber.
    """
    if not payloads:
        raise ValueError("refusing to archive an empty capture")
    stamp = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
    out = root / season / stamp
    out.mkdir(parents=True, exist_ok=False)

    files: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        body = _encode(payload)
        (out / f"{name}.json").write_bytes(body)
        files[name] = {
            "file": f"{name}.json",
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "captured_at": captured_at.isoformat(),
        "gameweek_context": gameweek_context,
        "files": files,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def verify_input_archive(path: Path) -> bool:
    """True when every file listed in the manifest exists and still matches its
    recorded SHA-256. A missing or altered file makes the capture unusable for
    replay, and this is how that is detected before anyone relies on it."""
    manifest = json.loads((path / MANIFEST_NAME).read_text())
    for entry in manifest["files"].values():
        target = path / entry["file"]
        if not target.exists():
            return False
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            return False
    return True
