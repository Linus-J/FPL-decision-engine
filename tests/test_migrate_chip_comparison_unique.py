"""``uq_chip_comparison`` migration: targetable, idempotent, and automatic.

``chip_comparison_log`` was created without its unique constraint, so every
database that ran the agent before it existed keeps the old shape --
``create_all`` builds missing tables in full but never alters an existing one.
The migration rebuilds the table, which SQLite requires for adding a
constraint.

Two properties make it safe to call from ``init_db`` on every startup, and both
are asserted here: running it a second time does nothing, and rows survive the
rebuild (a re-run that flipped the comparison's verdict IS the measurement this
table exists to capture).
"""

from __future__ import annotations

import sqlite3

from sqlalchemy import create_engine, inspect, text

from data.models import Base, ChipComparisonLog
from scripts.migrate_chip_comparison_unique import migrate


def _old_shape_db(path) -> None:
    """``chip_comparison_log`` as it was created before the constraint existed."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE chip_comparison_log (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            season VARCHAR(7) NOT NULL,
            gameweek INTEGER NOT NULL,
            sim_manager_id INTEGER,
            option VARCHAR(16) NOT NULL,
            horizon_xpts FLOAT NOT NULL,
            detail VARCHAR NOT NULL,
            chosen_live BOOLEAN NOT NULL,
            chosen_shadow BOOLEAN NOT NULL,
            created_at DATETIME
        )
        """
    )
    conn.execute(
        "INSERT INTO chip_comparison_log "
        "(season, gameweek, sim_manager_id, option, horizon_xpts, detail, "
        " chosen_live, chosen_shadow, created_at) "
        "VALUES ('2026-27', 3, NULL, 'free_hit', 288.49, 'free hit', 1, 1, "
        "'2026-09-01 15:22:34.930524')"
    )
    conn.commit()
    conn.close()


def _constraint_names(engine) -> set[str]:
    return {
        c["name"] for c in inspect(engine).get_unique_constraints("chip_comparison_log")
    }


def test_migrating_twice_is_a_no_op(tmp_path):
    db = tmp_path / "old.db"
    _old_shape_db(db)
    engine = create_engine(f"sqlite:///{db}")

    assert "uq_chip_comparison" not in _constraint_names(engine)
    assert migrate(engine) is True
    assert "uq_chip_comparison" in _constraint_names(engine)

    # The second run finds the constraint already present and reports that it
    # changed nothing -- which is what makes calling it from init_db cheap.
    assert migrate(engine) is False
    assert "uq_chip_comparison" in _constraint_names(engine)

    with engine.begin() as conn:
        rows = conn.execute(text("SELECT option, horizon_xpts FROM chip_comparison_log")).all()
    assert rows == [("free_hit", 288.49)]


def test_migrating_a_table_that_does_not_exist_yet_does_nothing(tmp_path):
    """``init_db`` runs ``create_all`` first, so a fresh database already has
    the constraint and there is nothing to rebuild."""
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    assert migrate(engine) is False


def test_a_freshly_created_table_already_carries_the_constraint(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    ChipComparisonLog.__table__.create(bind=engine, checkfirst=True)
    assert "uq_chip_comparison" in _constraint_names(engine)
    assert migrate(engine) is False


def test_init_db_runs_the_migration(monkeypatch, tmp_path):
    """The point of calling it from ``init_db``: an existing database in the old
    shape is corrected on startup, with no operator step."""
    import data.db as db

    target = tmp_path / "live.db"
    _old_shape_db(target)
    engine = create_engine(f"sqlite:///{target}")
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(Base.metadata, "create_all", lambda **kwargs: None)
    monkeypatch.setattr(db, "_add_missing_columns", lambda: [])
    monkeypatch.setattr(db, "_seed_data_checked_from_finished", lambda added: None)

    db.init_db()

    assert "uq_chip_comparison" in _constraint_names(engine)
