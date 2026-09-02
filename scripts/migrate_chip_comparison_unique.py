"""One-off: put ``uq_chip_comparison`` on an existing ``chip_comparison_log``.

``init_db`` calls ``create_all``, which builds MISSING tables in full but never
alters an existing one, and ``_add_missing_columns`` is deliberately ADD COLUMN
only -- it says outright that constraints need a real migration. This is that
migration, for the one table that needs it: ``chip_comparison_log`` was created
without the constraint earlier today, so every database that ran the agent
between then and now has the old shape.

SQLite cannot add a constraint in place, so the table is rebuilt: create the new
shape under a temporary name, copy every row across, drop the old, rename. Rows
are preserved exactly -- re-running a gameweek and getting a different verdict is
the measurement this table exists to capture, and collapsing those rows would
destroy it.

Idempotent: does nothing when the constraint is already present, or when the
table does not exist yet (``init_db`` will then create it with the constraint).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import MetaData, inspect, text  # noqa: E402
from sqlalchemy.schema import CreateTable  # noqa: E402

from data.db import engine  # noqa: E402
from data.models import ChipComparisonLog, SimManager  # noqa: E402

logger = logging.getLogger(__name__)

_TMP = "chip_comparison_log_new"


def _new_table_ddl() -> str:
    """The CREATE TABLE the model itself implies, under a temporary name.

    Generated rather than hand-written so the rebuilt table cannot drift from
    ``ChipComparisonLog`` -- and so SQLAlchemy can reflect its constraint back,
    which is what makes the idempotency check below work.
    """
    metadata = MetaData()
    # The sim_managers table comes along only so the foreign key has something
    # to resolve against; it is never created here.
    SimManager.__table__.to_metadata(metadata)
    table = ChipComparisonLog.__table__.to_metadata(metadata, name=_TMP)
    return str(CreateTable(table).compile(engine))


def migrate() -> bool:
    """Returns True when the table was rebuilt, False when nothing was needed."""
    inspector = inspect(engine)
    if "chip_comparison_log" not in inspector.get_table_names():
        logger.info("chip_comparison_log does not exist yet; init_db will create it")
        return False
    names = {c["name"] for c in inspector.get_unique_constraints("chip_comparison_log")}
    if "uq_chip_comparison" in names:
        logger.info("uq_chip_comparison already present; nothing to do")
        return False

    columns = ", ".join(c.name for c in ChipComparisonLog.__table__.columns)
    with engine.begin() as conn:
        conn.execute(text(_new_table_ddl()))
        conn.execute(text(
            f"INSERT INTO {_TMP} ({columns}) SELECT {columns} FROM chip_comparison_log"
        ))
        conn.execute(text("DROP TABLE chip_comparison_log"))
        conn.execute(text(f"ALTER TABLE {_TMP} RENAME TO chip_comparison_log"))
    logger.info("chip_comparison_log rebuilt with uq_chip_comparison")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    migrate()
