"""T2.5 acceptance gate — per-season schema keys + cross-season player code.

Self-contained (throwaway temp DB). Proves the Phase-1 M1/M3 fixes:
  - Gameweek is keyed on (season, id): the same GW-number can exist in two
    seasons, but (season, id) collides.
  - Fixture is keyed on (season, fpl_id): the same FPL fixture id can recur
    across seasons, but (season, fpl_id) collides.
  - Player.code is unique (cross-season-stable) and nullable.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from data.models import Base, Fixture, Gameweek, Player


@pytest.fixture
def Session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'schema.db'}")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _gw(gw_id: int, season: str) -> Gameweek:
    return Gameweek(id=gw_id, season=season, name=f"GW{gw_id}", deadline_time=datetime(2025, 8, 1))


def _fixture(fpl_id: int, season: str) -> Fixture:
    return Fixture(fpl_id=fpl_id, season=season, gameweek=1, team_h_id=1, team_a_id=2)


def _player(fpl_id: int, code: int | None) -> Player:
    return Player(
        fpl_id=fpl_id, code=code, first_name="F", second_name="L",
        web_name=f"P{fpl_id}", team_id=1, position="MID", now_cost=5.0,
    )


def test_init_db_adds_new_columns(Session):
    insp = inspect(Session().bind)
    gw_cols = {c["name"] for c in insp.get_columns("gameweeks")}
    assert "season" in gw_cols
    fx_cols = {c["name"] for c in insp.get_columns("fixtures")}
    assert "season" in fx_cols
    pl_cols = {c["name"] for c in insp.get_columns("players")}
    assert "code" in pl_cols


def test_data_checked_is_seeded_from_finished_when_the_column_is_added(tmp_path, monkeypatch):
    """Regression, 2026-08-18 (engine review §8).

    ``backfill_decision_outcomes`` now requires ``data_checked`` before it will
    score a gameweek. ``ALTER TABLE ... ADD COLUMN`` can only fill a constant,
    so every pre-existing row would land on the default (False) — including the
    five backfilled historical seasons, whose data has been settled for years.
    Historical re-scoring would have silently stopped working.
    """
    from sqlalchemy import text

    from data import db as db_module

    engine = create_engine(f"sqlite:///{tmp_path / 'seed.db'}")
    monkeypatch.setattr(db_module, "engine", engine)

    # A database as it existed BEFORE data_checked: build the schema, then
    # drop the column back off so init_db has to re-add it.
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE gameweeks DROP COLUMN data_checked"))
        for gw_id, finished in ((1, 1), (2, 0)):
            conn.execute(
                text(
                    "INSERT INTO gameweeks (id, season, name, deadline_time, finished, "
                    "is_current, is_next, average_entry_score, highest_score, "
                    "is_dgw, is_bgw) "
                    "VALUES (:id, '2025-26', :name, '2025-08-15 17:30:00', :finished, "
                    "0, 0, 0, 0, 0, 0)"
                ),
                {"id": gw_id, "name": f"GW{gw_id}", "finished": finished},
            )

    db_module.init_db()

    with engine.begin() as conn:
        seeded = dict(
            conn.execute(text("SELECT id, data_checked FROM gameweeks")).fetchall()
        )
    # A finished historical gameweek is settled; an unfinished one is not.
    assert seeded == {1: 1, 2: 0}


def test_gameweek_same_number_two_seasons_ok(Session):
    s = Session()
    s.add_all([_gw(1, "2025-26"), _gw(1, "2026-27")])
    s.commit()
    assert s.query(Gameweek).count() == 2
    s.close()


def test_gameweek_duplicate_season_id_rejected(Session):
    s = Session()
    s.add(_gw(1, "2025-26"))
    s.commit()
    s.add(_gw(1, "2025-26"))
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()
    s.close()


def test_fixture_same_fpl_id_two_seasons_ok(Session):
    s = Session()
    s.add_all([_fixture(100, "2025-26"), _fixture(100, "2026-27")])
    s.commit()
    assert s.query(Fixture).count() == 2
    s.close()


def test_fixture_duplicate_season_fpl_rejected(Session):
    s = Session()
    s.add(_fixture(100, "2025-26"))
    s.commit()
    s.add(_fixture(100, "2025-26"))
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()
    s.close()


def test_player_code_unique(Session):
    s = Session()
    s.add(_player(1, 12345))
    s.commit()
    s.add(_player(2, 12345))  # same code, different fpl_id
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()
    s.close()


def test_player_code_nullable_allows_multiple(Session):
    s = Session()
    s.add_all([_player(1, None), _player(2, None)])
    s.commit()  # multiple NULL codes are allowed
    assert s.query(Player).count() == 2
    s.close()
