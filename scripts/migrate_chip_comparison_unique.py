"""Add ``uq_chip_comparison`` to a ``chip_comparison_log`` that lacks it.

``init_db`` calls ``create_all``, which builds MISSING tables in full but never
alters an existing one, and ``_add_missing_columns`` is deliberately ADD COLUMN
only -- it says outright that constraints need a real migration. This is that
migration, for the one table that needs it: ``chip_comparison_log`` was created
without the constraint earlier today, so every database that ran the agent
between then and now has the old shape.

The migration itself (``migrate``) lives in ``data/migrations.py`` -- a library
module, imported by ``data.db.init_db`` on every startup, which is what stops a
fresh clone, a backtest database or another machine's copy drifting in silence.
This script is the CLI entry point for pointing that same migration at any
OTHER database: ``--db PATH`` migrates it directly; with no argument it targets
``DB_PATH``, the live database, exactly as ``init_db`` would.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import Engine, create_engine  # noqa: E402

from data.migrations import migrate  # noqa: E402

logger = logging.getLogger(__name__)


def _default_engine() -> Engine:
    """``DB_PATH``'s engine, imported lazily.

    ``data.db.init_db`` calls ``migrate``, so importing ``data.db`` at module
    scope here would be a cycle. Resolving it at call time also means an
    explicit ``--db`` never touches the live database at all.
    """
    from data.db import engine

    return engine


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        help="SQLite file to migrate. Defaults to DB_PATH, the live database.",
    )
    args = parser.parse_args(argv)
    if args.db is not None:
        # SQLite's own answer to a missing file is to silently CREATE an empty
        # one -- the same failure mode `data/db.py::_resolve_db_path` exists to
        # avoid for DB_PATH itself. Without this check, a typo'd --db path
        # would "succeed" against a fresh empty database with nothing to
        # migrate, rather than the caller's real one.
        if not Path(args.db).exists():
            parser.error(f"--db {args.db!r} does not exist")
        engine = create_engine(f"sqlite:///{args.db}")
    else:
        engine = _default_engine()
    migrate(engine)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
