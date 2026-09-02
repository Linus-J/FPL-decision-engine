import logging
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from data.models import Base

logger = logging.getLogger(__name__)


def _enable_wal_mode(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _resolve_db_path(raw: str) -> str:
    """A relative DB_PATH is anchored to the repository root, not the cwd.

    The live crontab spent five weeks in 2026 reading a stale, wrong-schema
    database because a relative path resolved against whatever directory the
    scheduler happened to start in, and SQLite's answer to a missing file is
    to create an empty one rather than fail. An absolute path in .env is
    still the clearest thing to write; this makes the relative form safe
    instead of silently wrong.
    """
    path = Path(raw)
    if path.is_absolute() or raw.startswith(("file:", ":memory:")):
        return raw
    return str((Path(__file__).resolve().parents[1] / path).resolve())


engine = create_engine(
    f"sqlite:///{_resolve_db_path(settings.db_path)}",
    connect_args={"check_same_thread": False},
)

event.listen(engine, "connect", _enable_wal_mode)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _add_missing_columns() -> list[str]:
    """Add columns the models declare but the database is missing.

    ``create_all`` creates missing TABLES but never alters existing ones, so
    adding a field to a model silently did nothing to a database that
    already had the table -- reads then failed with "no such column" against
    the real DB while passing every test (tests build their schema fresh).

    Deliberately narrow: ADD COLUMN only, and only for columns that are
    nullable or carry a scalar default, which is all SQLite can do in place
    anyway. Type changes, drops and renames are NOT handled and never will
    be here -- those need a real migration. Anything it cannot add is logged
    and left alone rather than guessed at. Idempotent.

    Returns the ``table.column`` names it added, for logging and tests.
    """
    from sqlalchemy import inspect

    added: list[str] = []
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # create_all will build it in full
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                default = column.default.arg if column.default is not None else None
                if not column.nullable and default is None:
                    logger.warning(
                        "%s.%s is missing and is NOT NULL with no default — needs a "
                        "real migration; leaving it alone",
                        table.name, column.name,
                    )
                    continue
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column.type}"
                if default is not None and not callable(default):
                    literal = repr(default) if isinstance(default, str) else default
                    ddl += f" DEFAULT {literal}"
                conn.execute(text(ddl))
                added.append(f"{table.name}.{column.name}")

    if added:
        logger.info("Added missing columns: %s", ", ".join(added))
    return added


def _seed_data_checked_from_finished(added: list[str]) -> None:
    """One-time, only on the run that ADDS ``gameweeks.data_checked``.

    ``ALTER TABLE ... ADD COLUMN`` can only fill a constant, so every existing
    row would land on the column default (False). For gameweeks already stored
    as ``finished`` — the five backfilled historical seasons — that is simply
    wrong: their data has been settled for years, and
    ``backfill_decision_outcomes`` now requires ``data_checked`` before it will
    score anything, so historical re-scoring would silently stop working.

    Seeding from ``finished`` is right for exactly those rows. Live 2026-27
    gameweeks get FPL's real flag from the next bootstrap ingest, which
    overwrites this on conflict.
    """
    if "gameweeks.data_checked" not in added:
        return
    with engine.begin() as conn:
        result = conn.execute(text("UPDATE gameweeks SET data_checked = finished"))
    logger.info(
        "Seeded gameweeks.data_checked from finished on %s existing rows "
        "(one-time, on column creation)", result.rowcount,
    )


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    added = _add_missing_columns()
    _seed_data_checked_from_finished(added)
    _migrate_chip_comparison_unique()


def _migrate_chip_comparison_unique() -> None:
    """Add ``uq_chip_comparison`` to a ``chip_comparison_log`` created without
    it (2026-09-02).

    ``create_all`` builds MISSING tables in full but never alters an existing
    one, and ``_add_missing_columns`` is ADD COLUMN only, so a database that ran
    the agent before the constraint existed keeps the old shape forever. The
    migration is idempotent and does nothing once the constraint is present, so
    running it on every startup costs one reflection and is what stops a fresh
    clone, a backtest database or another machine's copy drifting in silence.

    Imported inside the function because the script imports ``data.db``.
    """
    from scripts.migrate_chip_comparison_unique import migrate

    migrate(engine)


def get_session() -> Session:
    return SessionLocal()
