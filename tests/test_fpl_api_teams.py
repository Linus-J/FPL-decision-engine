"""data/ingestors/fpl_api.py -- upsert_teams' team_season_strength write,
and ingest_player_history's conflict-key fix.

Both are real bugs found 2026-07-30, live-smoke-testing the actual
2026-27 season path for the first time this session.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import data.ingestors.fpl_api as fpl_api
from data.models import Base, Player, PlayerGameweekStats, Team, TeamSeasonStrength


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'fpl_api.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fpl_api, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _bootstrap_team(team_id: int) -> dict:
    return {
        "id": team_id, "name": f"Team{team_id}", "short_name": f"T{team_id}", "code": team_id,
        "strength_overall_home": 1250, "strength_overall_away": 1150,
        "strength_attack_home": 1300, "strength_attack_away": 1200,
        "strength_defence_home": 1280, "strength_defence_away": 1180,
    }


def test_upsert_teams_also_writes_team_season_strength(session):
    # Real gap found 2026-07-30 (the user's own follow-up on the live smoke
    # test): team_season_strength was ONLY ever written by
    # scripts/backfill_history.py for its fixed list of PAST seasons --
    # nothing in the live sync path ever wrote a row for the CURRENT
    # season, so every live FDR feature silently fell back to the neutral
    # 1200 default, permanently, for the whole live-serving path.
    bootstrap = {"teams": [_bootstrap_team(1), _bootstrap_team(2)]}
    fpl_api.upsert_teams(bootstrap, season="2026-27")

    teams = session.execute(select(Team)).scalars().all()
    assert len(teams) == 2
    assert teams[0].strength_overall_home == 1250

    strengths = session.execute(select(TeamSeasonStrength)).scalars().all()
    assert len(strengths) == 2
    row = next(r for r in strengths if r.team_id == 1)
    assert row.season == "2026-27"
    assert row.strength_overall_home == 1250
    assert row.strength_attack_away == 1200


def test_upsert_teams_season_strength_is_idempotent(session):
    bootstrap = {"teams": [_bootstrap_team(1)]}
    fpl_api.upsert_teams(bootstrap, season="2026-27")
    updated = _bootstrap_team(1)
    updated["strength_overall_home"] = 1400
    fpl_api.upsert_teams({"teams": [updated]}, season="2026-27")

    strengths = session.execute(select(TeamSeasonStrength)).scalars().all()
    assert len(strengths) == 1  # updated in place, not duplicated
    assert strengths[0].strength_overall_home == 1400


@pytest.mark.asyncio
async def test_ingest_player_history_rerun_updates_not_duplicates(session, monkeypatch):
    # Real regression found 2026-07-30: this session's own earlier fix
    # widened PlayerGameweekStats' unique constraint to include
    # opponent_team_id (so a real DGW's two fixtures can both be stored),
    # but this ingestor never set opponent_team_id at all -- every row got
    # the column's NULL default, and SQLite never treats two NULLs as
    # equal for uniqueness, so a second run would insert a FRESH duplicate
    # row instead of updating the first.
    session.add(Player(
        fpl_id=100, code=100, first_name="A", second_name="A", web_name="Test",
        team_id=1, position="MID", now_cost=5.0,
    ))
    session.add(Team(id=1, name="Team1", short_name="T1"))
    session.commit()
    player_db_id = session.query(Player.id).filter_by(fpl_id=100).scalar()

    history = [{"round": 5, "total_points": 6, "minutes": 90, "selected": 500, "value": 55}]

    async def _fake_fetch(fpl_id):
        return {"history": history}

    monkeypatch.setattr(fpl_api, "fetch_player_summary", _fake_fetch)

    await fpl_api.ingest_player_history(100, player_db_id, season="2026-27")
    await fpl_api.ingest_player_history(100, player_db_id, season="2026-27")  # rerun

    rows = session.execute(
        select(PlayerGameweekStats).filter_by(player_id=player_db_id, gameweek=5)
    ).scalars().all()
    assert len(rows) == 1  # updated in place, not duplicated
    assert rows[0].total_points == 6
