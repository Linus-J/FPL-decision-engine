"""T3b gate — per-season team strengths + context-based FDR.

Pure parsing helpers + an integration test proving load_fixture_difficulty
now yields season-accurate, non-default strengths from the fixture context on
the stat row (team_id_season / opponent_team_id / was_home).
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Player, PlayerGameweekStats, TeamSeasonStrength
from projection import features
from scripts import backfill_history as bh


def _teams_df():
    return pd.DataFrame([
        {"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS",
         "strength_overall_home": 1350, "strength_overall_away": 1340,
         "strength_attack_home": 1390, "strength_attack_away": 1400,
         "strength_defence_home": 1310, "strength_defence_away": 1300},
        {"id": 2, "code": 8, "name": "Chelsea", "short_name": "CHE",
         "strength_overall_home": 1250, "strength_overall_away": 1240,
         "strength_attack_home": 1290, "strength_attack_away": 1280,
         "strength_defence_home": 1210, "strength_defence_away": 1150},
    ])


def test_team_strength_rows():
    rows = bh.team_strength_rows(_teams_df(), "2025-26")
    assert len(rows) == 2
    ars = next(r for r in rows if r["team_id"] == 1)
    assert ars["season"] == "2025-26" and ars["code"] == 3
    assert ars["strength_attack_home"] == 1390


def test_team_name_to_id():
    m = bh.team_name_to_id(_teams_df())
    assert m["Arsenal"] == 1 and m["ARS"] == 1 and m["Chelsea"] == 2


def test_resolve_team_id():
    m = {"Arsenal": 1}
    assert bh._resolve_team_id(5, m) == 5           # numeric id passthrough
    assert bh._resolve_team_id("Arsenal", m) == 1   # by name
    assert bh._resolve_team_id(None, m) is None
    assert bh._resolve_team_id(float("nan"), m) is None


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'fdr.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(features, "get_session", lambda: Local())
    return Local


def _seed_strengths(s, season="2025-26"):
    for r in _teams_df().itertuples():
        s.add(TeamSeasonStrength(
            season=season, team_id=int(r.id), code=int(r.code),
            strength_overall_home=r.strength_overall_home,
            strength_overall_away=r.strength_overall_away,
            strength_attack_home=r.strength_attack_home,
            strength_attack_away=r.strength_attack_away,
            strength_defence_home=r.strength_defence_home,
            strength_defence_away=r.strength_defence_away,
        ))


def test_load_fixture_difficulty_uses_season_context(temp_session):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A",
                     web_name="P", team_id=1, position="MID", now_cost=5.0))
        s.commit()
        pid = s.query(Player.id).filter_by(fpl_id=1).scalar()
        _seed_strengths(s)
        # Arsenal (team 1) at HOME vs Chelsea (team 2)
        s.add(PlayerGameweekStats(player_id=pid, gameweek=5, season="2025-26",
                                  minutes=90, total_points=6,
                                  team_id_season=1, opponent_team_id=2, was_home=True))
        s.commit()
    finally:
        s.close()

    fdr = features.load_fixture_difficulty("2025-26")
    row = fdr.iloc[0]
    assert row["is_home"] == 1
    # home player → opponent's AWAY defence/attack; own HOME attack/defence
    assert row["opp_defence_strength"] == 1150   # Chelsea defence_away (non-default)
    assert row["opp_attack_strength"] == 1280     # Chelsea attack_away
    assert row["own_attack_strength"] == 1390     # Arsenal attack_home
    assert row["own_defence_strength"] == 1310    # Arsenal defence_home


def _seed_placeholder_strengths(s, season):
    """What FPL's bootstrap actually serves before a season starts: attack and
    defence 0, overall on the 1-5 tier. Stored as-is because 0 is this
    project's 'not published' signal (see features._PLAUSIBLE_STRENGTH_FLOOR
    and cold_start.load_current_defence_strength)."""
    for r in _teams_df().itertuples():
        s.add(TeamSeasonStrength(
            season=season, team_id=int(r.id), code=int(r.code),
            strength_overall_home=4, strength_overall_away=3,
            strength_attack_home=0, strength_attack_away=0,
            strength_defence_home=0, strength_defence_away=0,
        ))


def test_lookahead_strengths_are_not_zero_when_fpl_serves_placeholders(temp_session):
    """Regression, 2026-08-18 (engine review §1).

    ``_usable()`` was applied to the six single-fixture strength columns but
    NOT inside the two lookahead subqueries, which averaged the raw columns.
    ``AVG(0,0,0)`` is 0, not NULL, so ``COALESCE(..., 1200)`` never fired and
    both features read 0 for a whole pre-publication season -- against ~1200
    in every training season, making them a perfect "this row is the current
    season" split for a boosted tree.

    This is the exact failure mode the file's own comment describes ("NaN is
    missing, 0 is present"), and it is invisible in any season with real
    strengths, so this test IS the guard.
    """
    season = "2026-27"
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A",
                     web_name="P", team_id=1, position="MID", now_cost=5.0))
        s.commit()
        pid = s.query(Player.id).filter_by(fpl_id=1).scalar()
        _seed_placeholder_strengths(s, season)
        # A run of gameweeks so the 3-GW lookahead subquery has rows to average.
        for gw in (5, 6, 7, 8):
            s.add(PlayerGameweekStats(
                player_id=pid, gameweek=gw, season=season, minutes=90,
                total_points=6, team_id_season=1, opponent_team_id=2,
                was_home=True,
            ))
        s.commit()
    finally:
        s.close()

    fdr = features.load_fixture_difficulty(season)
    row = fdr[fdr["gameweek"] == 5].iloc[0]

    # The bug: these came back 0.0 because AVG over placeholders is not NULL.
    assert row["next_3gw_avg_opp_defence"] == 1200
    assert row["next_3gw_avg_opp_attack"] == 1200
    # The six direct columns were already guarded — NULL, for add_fdr_features
    # to fill — and must stay that way.
    assert pd.isna(row["opp_defence_strength"])
    assert pd.isna(row["own_overall_strength"])
