"""Test-session bootstrap.

Two jobs, both of which used to be done by accident.

**1. Make the project importable.** ``tool.uv.package = false`` means the
project's own packages are never installed into the venv, so a bare ``pytest``
collected nothing -- 55 ``ModuleNotFoundError`` collection errors -- while
``python -m pytest`` (which puts cwd on ``sys.path``) passed. ``pythonpath``
in ``pyproject.toml`` now covers this; this file's presence also anchors
pytest's rootdir.

**2. Keep the suite off the real database.** ``data/db.py`` builds its engine
from ``settings.db_path`` at IMPORT time, and ``get_session()`` resolves
``SessionLocal`` from module globals at CALL time. Tests that wanted isolation
monkeypatched one module's ``get_session`` (e.g.
``decision_engine.get_session``), but any code reached from there which imported
``get_session`` itself -- ``data/ingestors/ownership.py::load_latest_ownership``
is the one that surfaced this -- opened a session against whatever
``settings.db_path`` pointed at. In a checkout with a populated
``fpl_bot_v2.db`` those tests therefore PASSED BY READING THE LIVE DATABASE,
and would have written to it had the code under test done a write. The same
tests failed anywhere the file was absent, which is how it was found.

Setting ``DB_PATH`` here, before anything imports ``config.settings``, points
the entire suite at a throwaway database. Environment variables take priority
over ``.env`` in pydantic-settings, so this wins regardless of local config.
Tests that build their own engine and monkeypatch are unaffected; this is the
floor, not a replacement for them.
"""

import os
import tempfile
from pathlib import Path

_TMP_DB_DIR = tempfile.mkdtemp(prefix="fpl-bot-tests-")
os.environ["DB_PATH"] = str(Path(_TMP_DB_DIR) / "test.db")

# Imported only AFTER DB_PATH is set — data.db binds its engine at import time.
from data.db import init_db  # noqa: E402

init_db()
