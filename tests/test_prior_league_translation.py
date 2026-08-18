"""P11 — cross-league translation-factor calibration (config lookup + pure
hold-out/factor-computation math). No live scrape needed."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.strategy import PRIOR_LEAGUE
from data.models import Base, Player, PlayerGameweekStats, PriorLeagueStats
from projection import prior_league_translation as plt


def test_prior_league_rules_covers_all_five_leagues():
    leagues = ["ENG-Championship", "ESP-La Liga", "ITA-Serie A",
               "GER-Bundesliga", "FRA-Ligue 1"]
    for league in leagues:
        assert PRIOR_LEAGUE.translation_factor(league) > 0
        assert PRIOR_LEAGUE.translation_variance(league) > 0


def test_every_league_is_discounted_and_the_championship_most():
    """Changed 2026-08-18. The top-5 factors used to be exactly 1.0 -- a claim
    that a Ligue 1 goal is worth precisely a Premier League goal. They now
    carry a modest discount ordered by a transfer-based league-strength study
    (La Liga strongest, Bundesliga weakest of the five), with the Championship
    still far below any of them.

    The exact values are a prior, not a measurement -- a direct calibration was
    rejected for survivorship bias -- so this locks the ORDERING and the fact
    that no league translates one-for-one, not the numbers."""
    top5 = [
        "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1",
    ]
    for league in top5:
        factor = PRIOR_LEAGUE.translation_factor(league)
        assert 0.5 < factor < 1.0, f"{league} should be discounted but not crushed"

    championship = PRIOR_LEAGUE.translation_factor("ENG-Championship")
    assert championship < min(PRIOR_LEAGUE.translation_factor(x) for x in top5)
    # La Liga is the strongest of the five, Bundesliga the weakest.
    assert PRIOR_LEAGUE.translation_factor("ESP-La Liga") == max(
        PRIOR_LEAGUE.translation_factor(x) for x in top5
    )
    assert PRIOR_LEAGUE.translation_factor("GER-Bundesliga") == min(
        PRIOR_LEAGUE.translation_factor(x) for x in top5
    )


def test_prior_league_rules_covers_every_registered_league():
    # regression guard: PRIOR_LEAGUES (fbref_prior.py), PriorLeagueRules's
    # two lookup dicts (config/strategy.py), and calibrate_prior_league_
    # factors.py's _FIELD_SUFFIX separately enumerate the same 5 league
    # strings -- a drift would raise a bare KeyError inside the live GW1
    # build path (build_initial_squad) with nothing today catching it.
    from data.ingestors.fbref_prior import PRIOR_LEAGUES

    for league in PRIOR_LEAGUES:
        PRIOR_LEAGUE.translation_factor(league)
        PRIOR_LEAGUE.translation_variance(league)


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'plt.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(plt, "get_session", lambda: Local())
    return Local


def _seed_holdout_player(Local, *, code, fpl_id, prior_goals90, prior_assists90,
                          prior_minutes, pl_goals, pl_assists, pl_points, pl_minutes,
                          prior_season, pl_season):
    s = Local()
    try:
        s.add(Player(fpl_id=fpl_id, code=code, first_name="P", second_name=str(fpl_id),
                     web_name=f"p{fpl_id}", team_id=1, position="FWD", now_cost=5.0))
        s.commit()
        s.add(PriorLeagueStats(
            player_name=f"p{fpl_id}", team="Leeds", league="ENG-Championship",
            season=prior_season, code=code, position="FW",
            minutes=prior_minutes, matches=prior_minutes // 90,
            goals90=prior_goals90, assists90=prior_assists90, npxg90=prior_goals90,
            xa90=prior_assists90,
        ))
        pid = s.query(Player.id).filter_by(fpl_id=fpl_id).scalar()
        s.add(PlayerGameweekStats(
            player_id=pid, gameweek=1, season=pl_season,
            minutes=pl_minutes, goals_scored=pl_goals, assists=pl_assists,
            total_points=pl_points,
        ))
        s.commit()
    finally:
        s.close()


def test_build_holdout_pools_across_season_transitions(temp_session):
    # one qualifying player in each of two different season-transitions
    _seed_holdout_player(
        temp_session, code=1, fpl_id=1, prior_goals90=0.5, prior_assists90=0.1,
        prior_minutes=1000, pl_goals=9, pl_assists=2, pl_points=100, pl_minutes=1000,
        prior_season="2021-2022", pl_season="2022-23",
    )
    _seed_holdout_player(
        temp_session, code=2, fpl_id=2, prior_goals90=0.4, prior_assists90=0.2,
        prior_minutes=900, pl_goals=5, pl_assists=1, pl_points=60, pl_minutes=900,
        prior_season="2023-2024", pl_season="2024-25",
    )
    holdout = plt.build_holdout("ENG-Championship")
    assert len(holdout) == 2
    assert set(holdout["code"]) == {1, 2}
    row1 = holdout[holdout["code"] == 1].iloc[0]
    assert row1["realized_goals90"] == pytest.approx(9 / 1000 * 90)
    assert row1["realized_points90"] == pytest.approx(100 / 1000 * 90)


def test_build_holdout_excludes_players_below_the_minutes_bar(temp_session):
    # below MIN_HOLDOUT_MINUTES on the PL side -- must not count
    _seed_holdout_player(
        temp_session, code=1, fpl_id=1, prior_goals90=0.5, prior_assists90=0.1,
        prior_minutes=1000, pl_goals=1, pl_assists=0, pl_points=5, pl_minutes=50,
        prior_season="2021-2022", pl_season="2022-23",
    )
    holdout = plt.build_holdout("ENG-Championship")
    assert holdout.empty


def test_build_holdout_empty_league_returns_empty_frame(temp_session):
    holdout = plt.build_holdout("GER-Bundesliga")
    assert holdout.empty
    assert list(holdout.columns) == [
        "code", "prior_goals90", "prior_assists90",
        "realized_goals90", "realized_assists90", "realized_points90",
    ]


def test_compute_league_stats_below_min_samples_returns_none():
    holdout = pd.DataFrame({
        "code": [1], "prior_goals90": [0.5], "prior_assists90": [0.1],
        "realized_goals90": [0.6], "realized_assists90": [0.1],
        "realized_points90": [8.0],
    })
    factor, variance, n = plt.compute_league_stats(holdout)
    assert (factor, variance, n) == (None, None, 1)


def test_compute_league_stats_ratio_of_medians():
    # 20 rows (>= MIN_CALIBRATION_SAMPLES=15): prior median 0.50, realized median 0.40
    # -> factor 0.80. Interleaved so the median lands exactly on those values.
    prior = [0.4] * 10 + [0.6] * 10
    realized = [0.3] * 10 + [0.5] * 10
    holdout = pd.DataFrame({
        "code": range(20),
        "prior_goals90": prior, "prior_assists90": [0.0] * 20,
        "realized_goals90": realized, "realized_assists90": [0.0] * 20,
        "realized_points90": [5.0] * 10 + [7.0] * 10,
    })
    factor, variance, n = plt.compute_league_stats(holdout)
    assert n == 20
    assert factor == pytest.approx(0.4 / 0.5)
    assert variance == pytest.approx(pd.Series([5.0] * 10 + [7.0] * 10).var(ddof=1))
