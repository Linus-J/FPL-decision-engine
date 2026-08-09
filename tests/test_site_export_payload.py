"""scripts/site_export/payload.py"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, ProjectionSample
from scripts.site_export import payload as payload_module


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'export.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def _add_samples(session, player_id: int, gw: int, season: str, values: list[float]) -> None:
    session.add_all([
        ProjectionSample(player_id=player_id, gameweek=gw, season=season, scenario_id=i, xpts=v)
        for i, v in enumerate(values)
    ])
    session.commit()


def test_get_projection_distributions_returns_summary_per_player(session):
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[2.0, 4.0, 6.0, 8.0, 10.0])
    _add_samples(session, player_id=2, gw=3, season="2026-27", values=[1.0, 1.0, 1.0, 1.0, 1.0])

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert set(dist.keys()) == {1, 2}
    assert dist[1]["mean"] == 6.0
    assert dist[1]["median"] == 6.0
    assert dist[2]["mean"] == 1.0
    assert dist[2]["p10"] == 1.0
    assert dist[2]["p90"] == 1.0


def test_get_projection_distributions_ignores_other_gameweeks_and_seasons(session):
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[5.0, 5.0])
    _add_samples(session, player_id=1, gw=4, season="2026-27", values=[99.0])
    _add_samples(session, player_id=1, gw=3, season="2025-26", values=[99.0])

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert dist[1]["mean"] == 5.0


def test_get_projection_distributions_empty_when_no_samples(session):
    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")
    assert dist == {}
