"""scripts/site_export/payload.py"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Gameweek, ProjectionSample, Team
from scripts.site_export import payload as payload_module


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'export.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def _add_samples(session, player_id: int, gw: int, season: str, values: list[float], created_at: datetime | None = None) -> None:
    session.add_all([
        ProjectionSample(player_id=player_id, gameweek=gw, season=season, scenario_id=i, xpts=v, created_at=created_at)
        for i, v in enumerate(values)
    ])
    session.commit()


def test_get_projection_distributions_returns_summary_per_player(session):
    batch_ts = datetime(2026, 8, 1, 6, 0)
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[2.0, 4.0, 6.0, 8.0, 10.0], created_at=batch_ts)
    _add_samples(session, player_id=2, gw=3, season="2026-27", values=[1.0, 1.0, 1.0, 1.0, 1.0], created_at=batch_ts)

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert set(dist.keys()) == {1, 2}
    assert dist[1]["mean"] == 6.0
    assert dist[1]["median"] == 6.0
    assert dist[2]["mean"] == 1.0
    assert dist[2]["p10"] == 1.0
    assert dist[2]["p90"] == 1.0


def test_get_projection_distributions_ignores_other_gameweeks_and_seasons(session):
    batch_ts = datetime(2026, 8, 1, 6, 0)
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[5.0, 5.0], created_at=batch_ts)
    _add_samples(session, player_id=1, gw=4, season="2026-27", values=[99.0], created_at=batch_ts)
    _add_samples(session, player_id=1, gw=3, season="2025-26", values=[99.0], created_at=batch_ts)

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert dist[1]["mean"] == 5.0


def test_get_projection_distributions_empty_when_no_samples(session):
    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")
    assert dist == {}


def test_get_projection_distributions_uses_only_latest_batch(session):
    batch1_ts = datetime(2026, 8, 1, 6, 0)
    batch2_ts = datetime(2026, 8, 3, 6, 0)
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[1.0, 1.0, 4.0], created_at=batch1_ts)
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[7.0, 8.0, 9.0], created_at=batch2_ts)

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert dist[1]["mean"] == 8.0


def _add_team(session, team_id: int, short_name: str) -> None:
    session.add(Team(id=team_id, name=short_name, short_name=short_name))
    session.commit()


def test_team_short_names_maps_id_to_short_name(session):
    _add_team(session, 1, "ARS")
    _add_team(session, 2, "MCI")

    names = payload_module._team_short_names(session)

    assert names == {1: "ARS", 2: "MCI"}


def test_label_for_gw_uses_deadline_time(session):
    session.add(Gameweek(
        id=3, season="2026-27", name="Gameweek 3",
        deadline_time=datetime(2026, 8, 3, 10, 30),
    ))
    session.commit()

    label = payload_module._label_for_gw(session, "2026-27", 3)

    assert label == "GW3 — 3 Aug"


def test_label_for_gw_falls_back_when_gameweek_missing(session):
    label = payload_module._label_for_gw(session, "2026-27", 99)
    assert label == "GW99"
