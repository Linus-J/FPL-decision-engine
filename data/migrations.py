"""Library-side schema migrations for databases that predate a constraint.

Split out of ``scripts/migrate_chip_comparison_unique.py`` (2026-09-02, Task
7d item 5). ``data/db.py::init_db`` is a library entry point with 12 call
sites across the repo, and it was importing ``migrate`` FROM ``scripts/`` --
which re-ran that module's own ``sys.path.insert(0, repo_root)`` line as a
side effect of every single ``init_db()`` call, purely so the script could
also be invoked directly as ``python scripts/migrate_....py``. A library
should not depend on a scripts directory for its own logic, and importing it
should not mutate global interpreter state (``sys.path``) just to make that
import work. ``migrate`` now lives here with no ``sys.path`` involvement at
all; ``scripts/migrate_chip_comparison_unique.py`` keeps working as a CLI by
importing it from here instead of defining it.
"""
from __future__ import annotations

import logging

from sqlalchemy import Engine, MetaData, inspect, text
from sqlalchemy.schema import CreateTable

from data.models import ChipComparisonLog, SimManager

logger = logging.getLogger(__name__)

_TMP = "chip_comparison_log_new"


def _new_table_ddl(engine: Engine) -> str:
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


def migrate(engine: Engine) -> bool:
    """Add ``uq_chip_comparison`` to ``chip_comparison_log`` if it's missing it.

    SQLite cannot add a constraint in place, so the table is rebuilt: create
    the new shape under a temporary name, copy every row across, drop the
    old, rename. Rows are preserved exactly -- re-running a gameweek and
    getting a different verdict is the measurement this table exists to
    capture, and collapsing those rows would destroy it.

    Idempotent: does nothing when the constraint is already present, or when
    the table does not exist yet (``init_db``'s own ``create_all`` will then
    create it with the constraint). Every caller passes its own ``engine``
    explicitly -- unlike the script this was split from, there is no implicit
    default here; resolving "which database" is a CLI/caller concern, not a
    library one.

    Returns True when the table was rebuilt, False when nothing was needed.
    """
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
        conn.execute(text(_new_table_ddl(engine)))
        conn.execute(text(
            f"INSERT INTO {_TMP} ({columns}) SELECT {columns} FROM chip_comparison_log"
        ))
        conn.execute(text("DROP TABLE chip_comparison_log"))
        conn.execute(text(f"ALTER TABLE {_TMP} RENAME TO chip_comparison_log"))
    logger.info("chip_comparison_log rebuilt with uq_chip_comparison")
    return True
