"""simulation/personas.py"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, SimManager
from simulation.personas import (
    _CHIP_AGGRESSIVENESS_RANGE,
    _MAX_OWNERSHIP_DIFFERENTIAL_RANGE,
    _RISK_MODES,
    _VARIANCE_WEIGHT_RANGE,
    generate_personas,
    load_or_create_personas,
)


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'personas.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def test_generate_personas_is_deterministic():
    a = generate_personas("2026-27", n=20, seed=42)
    b = generate_personas("2026-27", n=20, seed=42)
    assert a == b


def test_generate_personas_different_seed_differs():
    a = generate_personas("2026-27", n=20, seed=42)
    b = generate_personas("2026-27", n=20, seed=43)
    assert a != b


def test_generate_personas_values_within_configured_ranges():
    personas = generate_personas("2026-27", n=50, seed=1)
    for p in personas:
        assert p["risk_mode"] in _RISK_MODES
        assert _VARIANCE_WEIGHT_RANGE[0] <= p["variance_weight"] <= _VARIANCE_WEIGHT_RANGE[1]
        assert (
            _MAX_OWNERSHIP_DIFFERENTIAL_RANGE[0]
            <= p["max_ownership_differential"]
            <= _MAX_OWNERSHIP_DIFFERENTIAL_RANGE[1]
        )
        assert (
            _CHIP_AGGRESSIVENESS_RANGE[0]
            <= p["chip_aggressiveness"]
            <= _CHIP_AGGRESSIVENESS_RANGE[1]
        )
        assert p["season"] == "2026-27"
        assert p["label"]


def test_generate_personas_count_and_labels_are_unique():
    personas = generate_personas("2026-27", n=30, seed=7)
    assert len(personas) == 30
    assert len({p["label"] for p in personas}) == 30


def test_load_or_create_personas_persists_once(session):
    first = load_or_create_personas(session, "2026-27", n=10, seed=5)
    assert len(first) == 10
    assert session.query(SimManager).filter_by(season="2026-27").count() == 10

    second = load_or_create_personas(session, "2026-27", n=10, seed=5)
    assert [p.id for p in second] == [p.id for p in first]
    assert session.query(SimManager).filter_by(season="2026-27").count() == 10


def test_load_or_create_personas_does_not_regenerate_with_different_args(session):
    """Once personas exist for a season, later calls return the SAME rows
    even if n/seed differ -- persona identity must never change mid-season."""
    first = load_or_create_personas(session, "2026-27", n=10, seed=5)
    second = load_or_create_personas(session, "2026-27", n=999, seed=999)
    assert len(second) == len(first) == 10


def test_load_or_create_personas_scoped_per_season(session):
    load_or_create_personas(session, "2026-27", n=5, seed=1)
    load_or_create_personas(session, "2027-28", n=7, seed=1)
    assert session.query(SimManager).filter_by(season="2026-27").count() == 5
    assert session.query(SimManager).filter_by(season="2027-28").count() == 7
