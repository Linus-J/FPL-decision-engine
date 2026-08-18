"""T7 gate — GW1 cold-start projections + departure gate.

Self-contained (temp DB). Proves the contract: every candidate gets a
non-default projection source (prior-season or position/price prior — never a
silent 0.0), and confirmed leavers (status 'u') are dropped.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.strategy import OptimiserConfig
from data.models import (
    Base,
    Fixture,
    Player,
    PlayerGameweekStats,
    PriorLeagueStats,
    Team,
    TeamSeasonStrength,
)
from projection import cold_start as cs


def test_prior_season_of():
    assert cs.prior_season_of("2026-27") == "2025-26"
    assert cs.prior_season_of("2024-25") == "2023-24"


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'cs.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(cs, "get_session", lambda: Local())
    return Local


def _seed(Local):
    s = Local()
    try:
        # p1: established (10 prior appearances), p2: promoted/new (no prior),
        # p3: confirmed leaver (status 'u').
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Estab",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.add(Player(fpl_id=2, code=2, first_name="B", second_name="B", web_name="NewSign",
                     team_id=1, position="FWD", now_cost=6.5, status="a"))
        s.add(Player(fpl_id=3, code=3, first_name="C", second_name="C", web_name="Leaver",
                     team_id=1, position="DEF", now_cost=5.0, status="u"))
        s.commit()
        p1 = s.query(Player.id).filter_by(fpl_id=1).scalar()
        # 10 prior-season appearances for p1, avg 6 pts when played, all starts
        for gw in range(1, 11):
            s.add(PlayerGameweekStats(player_id=p1, gameweek=gw, season="2025-26",
                                      minutes=90, total_points=6))
        s.commit()
        return {"p1": p1}
    finally:
        s.close()


def test_load_prior_season_features(temp_session):
    ids = _seed(temp_session)
    feats = cs.load_prior_season_features("2025-26")
    row = feats[feats["player_id"] == ids["p1"]].iloc[0]
    assert row["appearances"] == 10
    assert row["ppg_played"] == pytest.approx(6.0)
    assert row["starts_rate"] == pytest.approx(1.0)


def test_departure_gate_drops_confirmed_leaver(temp_session):
    _seed(temp_session)
    players = cs.apply_departure_gate(cs.load_current_players())
    names = set(players["web_name"])
    assert "Leaver" not in names          # status 'u' dropped
    assert {"Estab", "NewSign"} <= names


def test_projection_sources_and_no_silent_zero(temp_session):
    _seed(temp_session)
    players = cs.apply_departure_gate(cs.load_current_players())
    prior = cs.load_prior_season_features("2025-26")
    proj = cs.project_cold_start(players, prior)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    src = proj.set_index("name")["proj_source"].to_dict()

    assert src["Estab"] == "prior_season"
    assert src["NewSign"] == "position_price_prior"   # no prior PL data
    # the deliverable's core contract: no silent 0.0, every slot has a source
    assert (proj["xpts"] > 0).all()
    assert proj["proj_source"].notna().all()
    # extended to variance (plan/risk-aware-cold-start-v1.md): never
    # undefined/negative even with no raw_appearances supplied at all
    assert proj["xpts_var"].notna().all()
    assert (proj["xpts_var"] >= 0).all()
    # established player's xpts tracks prior ppg
    estab = proj[proj["name"] == "Estab"].iloc[0]
    assert estab["xpts"] == pytest.approx(6.0)
    assert estab["start_probability"] == pytest.approx(1.0)


def test_price_prior_monotonic_in_price():
    assert cs._price_prior("MID", 10.0) > cs._price_prior("MID", 5.0)
    assert cs._price_prior("FWD", 4.0) >= cs._MIN_XPTS


# --- real variance (plan/risk-aware-cold-start-v1.md, 2026-07-31) ----------


def _seed_variance_pool(Local):
    """One established player with KNOWN varying per-GW points (for an
    exact hand-computed own-variance check), five established MID peers
    all priced at exactly £8.0m forming a real (position, price-bucket)
    pool of 25 pooled appearances (>= _MIN_BUCKET_SAMPLES=20), and a new
    signing at that same price with no prior data of their own."""
    s = Local()
    try:
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Varied",
                     team_id=1, position="MID", now_cost=9.0, status="a"))
        for i in range(5):
            s.add(Player(
                fpl_id=10 + i, code=10 + i, first_name="P", second_name=str(i),
                web_name=f"Peer{i}", team_id=1, position="MID", now_cost=8.0, status="a",
            ))
        s.add(Player(fpl_id=99, code=99, first_name="N", second_name="N", web_name="NewMid",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()

        varied_id = s.query(Player.id).filter_by(fpl_id=1).scalar()
        for gw, pts in enumerate([2, 4, 6, 8, 10], start=1):
            s.add(PlayerGameweekStats(player_id=varied_id, gameweek=gw, season="2025-26",
                                      minutes=90, total_points=pts))

        peer_values = [2, 4, 6, 8, 10]
        for i, value in enumerate(peer_values):
            peer_id = s.query(Player.id).filter_by(fpl_id=10 + i).scalar()
            for gw in range(1, 6):
                s.add(PlayerGameweekStats(player_id=peer_id, gameweek=gw, season="2025-26",
                                          minutes=90, total_points=value))
        s.commit()
        return {"varied_id": varied_id}
    finally:
        s.close()


def test_established_player_gets_real_own_variance(temp_session):
    _seed_variance_pool(temp_session)
    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    proj = cs.project_cold_start(players, prior, raw_appearances=raw)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    row = proj[proj["name"] == "Varied"].iloc[0]

    assert row["proj_source"] == "prior_season"
    assert row["xpts"] == pytest.approx(6.0)  # mean of [2,4,6,8,10]
    # sample variance (ddof=1) of [2,4,6,8,10]: sum((x-6)^2)/(5-1) = 40/4
    assert row["xpts_var"] == pytest.approx(10.0)


def test_new_signing_gets_pooled_peer_bucket_stats(temp_session):
    _seed_variance_pool(temp_session)
    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    proj = cs.project_cold_start(players, prior, raw_appearances=raw)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    row = proj[proj["name"] == "NewMid"].iloc[0]

    assert row["proj_source"] == "peer_bucket_prior"
    # pool = five 2s, five 4s, five 6s, five 8s, five 10s (25 samples)
    assert row["xpts"] == pytest.approx(6.0)
    # sum((x-6)^2) = 5*(16+4+0+4+16) = 200; /(25-1) = 8.3333...
    assert row["xpts_var"] == pytest.approx(200 / 24)


def test_sparse_position_falls_back_to_synthetic_prior(temp_session):
    """A position with far fewer than _MIN_BUCKET_SAMPLES pooled
    appearances (even widened to position-only) must still get a real,
    non-null xpts/xpts_var from the last-resort synthetic prior -- never
    crash, never leave it undefined."""
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="OnlyGK",
                     team_id=1, position="GKP", now_cost=8.0, status="a"))
        s.commit()
        gk_id = s.query(Player.id).filter_by(fpl_id=1).scalar()
        for gw in range(1, 6):
            s.add(PlayerGameweekStats(player_id=gk_id, gameweek=gw, season="2025-26",
                                      minutes=90, total_points=5))
        s.add(Player(fpl_id=2, code=2, first_name="B", second_name="B", web_name="NewGK",
                     team_id=1, position="GKP", now_cost=4.5, status="a"))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    proj = cs.project_cold_start(players, prior, raw_appearances=raw)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    row = proj[proj["name"] == "NewGK"].iloc[0]

    assert row["proj_source"] == "position_price_prior"
    # P0.1: the synthetic prior is a per-APPEARANCE figure, so it is weighted
    # by the assumed availability of a player we know nothing about before it
    # becomes a per-gameweek expectation.
    expected_xpts, expected_var = cs.unconditional_moments(
        cs.NEW_PLAYER_APPEARANCE_PROB, cs._price_prior("GKP", 4.5), cs._FALLBACK_VAR
    )
    assert row["xpts"] == pytest.approx(expected_xpts)
    assert row["xpts_var"] == pytest.approx(expected_var)


def test_new_signing_with_matched_prior_league_row_gets_prior_league_prior(temp_session):
    _seed(temp_session)  # p2 = NewSign, code=2, position FWD, price 6.5
    players = cs.apply_departure_gate(cs.load_current_players())
    prior = cs.load_prior_season_features("2025-26")
    lookup = {
        2: {"league": "ENG-Championship", "goals90": 0.6, "assists90": 0.2,
            "npxg90": 0.5, "xa90": 0.15, "minutes": 3000, "matches": 34},
    }
    proj = cs.project_cold_start(players, prior, prior_league_lookup=lookup)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    row = proj[proj["name"] == "NewSign"].iloc[0]

    assert row["proj_source"] == "prior_league_prior"
    # P0.1: both sides of this comparison are now per-gameweek expectations,
    # not per-appearance figures -- the prior-league tier is weighted by
    # matches/season-length (34/46 for the Championship), the synthetic
    # fallback by the unknown-player default.
    fallback_xpts, _ = cs.unconditional_moments(
        cs.NEW_PLAYER_APPEARANCE_PROB, cs._price_prior("FWD", 6.5), cs._FALLBACK_VAR
    )
    assert row["xpts"] > fallback_xpts
    expected_xpts, expected_var = cs.unconditional_moments(
        34 / 46,
        row["xpts"] / (34 / 46),  # back out the per-match mean this tier built
        cs.PRIOR_LEAGUE.translation_variance("ENG-Championship"),
    )
    assert row["xpts_var"] == pytest.approx(expected_var)
    # nailed-on Championship starter (3000/34/90 ~= 0.98 share) blended 50/50
    # with the flat 0.6 default -> higher than the flat default alone.
    assert row["start_probability"] > cs.NEW_PLAYER_START_PROB


def test_prior_league_prior_never_scores_below_the_peer_bucket_floor(temp_session):
    _seed_variance_pool(temp_session)  # MID peer pool (mean 6.0) + a code=99 "NewMid"
    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    # a matched prior-league row with a low, unremarkable attacking record --
    # the prior-league tier models attacking output only, so unfloored this
    # would score well below the peer-bucket pool's real mean of 6.0.
    lookup = {
        99: {"league": "ENG-Championship", "goals90": 0.02, "assists90": 0.01,
             "npxg90": 0.02, "xa90": 0.01, "minutes": 3000, "matches": 34},
    }
    proj = cs.project_cold_start(players, prior, raw_appearances=raw, prior_league_lookup=lookup)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    row = proj[proj["name"] == "NewMid"].iloc[0]

    # the fallback floor binds here, so the reported source must be the
    # fallback's own source, not the mismatched "prior_league_prior" label.
    assert row["proj_source"] == "peer_bucket_prior"
    # must never score below what the peer-bucket cascade alone would give
    # the same player (pool mean 6.0 from _seed_variance_pool's fixture)
    assert row["xpts"] >= 6.0


def test_new_signing_with_no_prior_league_match_falls_through_unchanged(temp_session):
    # regression guard: passing an EMPTY lookup must behave exactly like
    # passing none at all (today's existing peer_bucket_prior cascade).
    _seed_variance_pool(temp_session)
    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    proj = cs.project_cold_start(players, prior, raw_appearances=raw, prior_league_lookup={})

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    row = proj[proj["name"] == "NewMid"].iloc[0]
    assert row["proj_source"] == "peer_bucket_prior"


def test_load_prior_league_lookup_reads_matched_rows_for_the_right_prior_season(
    temp_session,
):
    s = temp_session()
    try:
        s.add(PriorLeagueStats(
            player_name="Prolific Striker", team="Leeds", league="ENG-Championship",
            season="2025-2026", code=42, position="FW", minutes=3000, matches=34,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.add(PriorLeagueStats(
            player_name="Stale Season", team="Leeds", league="ENG-Championship",
            season="2024-2025", code=43, position="FW", minutes=3000, matches=34,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.add(PriorLeagueStats(
            player_name="One Cameo", team="Leeds", league="ENG-Championship",
            season="2025-2026", code=44, position="FW", minutes=90, matches=1,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.commit()
    finally:
        s.close()

    lookup = cs.load_prior_league_lookup("2026-27")
    assert set(lookup.keys()) == {42}
    assert lookup[42]["league"] == "ENG-Championship"
    assert lookup[42]["npxg90"] == pytest.approx(0.5)


def test_load_prior_league_lookup_empty_when_nothing_ingested(temp_session):
    assert cs.load_prior_league_lookup("2026-27") == {}


def _seed_full_pool(Local):
    """A large-enough pool (2 GKP, 5 DEF, 5 MID, 3+ FWD per club-limit) for
    optimise_squad to actually build a legal 15, one confirmed leaver mixed
    in, so the departure gate's effect is visible in the final squad too."""
    positions = ["GKP"] * 4 + ["DEF"] * 8 + ["MID"] * 8 + ["FWD"] * 5
    s = Local()
    try:
        for i, position in enumerate(positions):
            s.add(Player(
                fpl_id=i + 1, code=i + 1, first_name="P", second_name=str(i + 1),
                web_name=f"Leaver{i}" if i == 0 else f"p{i}",
                team_id=1 + (i % 8), position=position, now_cost=4.5,
                status="u" if i == 0 else "a",
            ))
        s.commit()
    finally:
        s.close()


def test_build_initial_squad_uses_injected_players_not_live_bootstrap(temp_session, monkeypatch):
    # 2026-07-30 (user's own request: "we need to have and test a method to
    # start from GW1... for the realtime 26/27 season which is
    # approaching"). build_initial_squad must be usable against a POINT-IN-
    # TIME historical snapshot (e.g. a completed season's real GW1 roster),
    # not just the live bootstrap -- otherwise the only way to validate it
    # is by waiting for the real season to start.
    _seed_full_pool(temp_session)

    def _boom():
        raise AssertionError("load_current_players() must not be called when players is given")

    monkeypatch.setattr(cs, "load_current_players", _boom)

    s = temp_session()
    try:
        injected = pd.read_sql(
            "SELECT id, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    solution, projections = cs.build_initial_squad("2026-27", players=injected)
    assert len(solution.squad) == 15
    assert "Leaver0" not in solution.squad["web_name"].tolist()  # departure gate still applied
    assert not projections.empty


def test_build_initial_squad_applies_curse_shrinkage(temp_session, monkeypatch):
    """Regression, 2026-08-18 (engine review §3).

    ``apply_curse_shrinkage`` ran in ``projection/pipeline.py`` for every
    in-season gameweek and never here, because the decision engine calls
    ``build_initial_squad`` directly and bypasses the pipeline. So the one
    decision made from the noisiest projections in the system — the initial
    15, locked in for weeks — was the one made with no correction for
    selecting on noise.

    ``xpts_raw`` is the tell: it only exists once shrinkage has run, and it
    preserves the pre-shrinkage value.
    """
    import dataclasses

    from config.strategy import OPTIMISER

    _seed_full_pool(temp_session)
    s = temp_session()
    try:
        injected = pd.read_sql(
            "SELECT id, code, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    _, projections = cs.build_initial_squad("2026-27", players=injected)
    # `xpts_raw` exists only once shrinkage has run, and holds the
    # pre-shrinkage value. (The shrinkage arithmetic itself is covered in
    # test_curse_shrinkage.py; this test is about the wiring, since the defect
    # was that the function was never reached on this path at all.)
    assert "xpts_raw" in projections.columns, "cold start must apply curse shrinkage"

    # A player the departure gate zeroed must not be handed points back.
    zeroed = projections[projections["xpts_raw"] == 0.0]
    assert (zeroed["xpts"] == 0.0).all()

    # And the flag still disables it exactly, as it does in-season.
    _, unshrunk = cs.build_initial_squad(
        "2026-27", players=injected,
        config=dataclasses.replace(OPTIMISER, curse_shrinkage_enabled=False),
    )
    assert "xpts_raw" not in unshrunk.columns


def test_load_current_players_applies_team_overrides(temp_session, monkeypatch, tmp_path):
    import yaml

    from data import overrides as ov

    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=111, first_name="A", second_name="A", web_name="Moved",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    overrides_path = tmp_path / "transfer_overrides.yaml"
    overrides_path.write_text(yaml.safe_dump({"confirmed": [{"code": 111, "team_id": 99}]}))
    monkeypatch.setattr(ov, "OVERRIDES_PATH", overrides_path)

    players = cs.load_current_players()
    assert players.loc[players["web_name"] == "Moved", "team_id"].iloc[0] == 99


def test_build_initial_squad_passes_config_through_to_optimise_squad(temp_session, monkeypatch):
    """Simulation-engine entry point: run_for_persona cold-starts a persona
    via build_initial_squad(config=...) -- must actually reach the internal
    optimise_squad call, not be silently dropped."""
    from config.strategy import OptimiserConfig

    _seed_full_pool(temp_session)
    s = temp_session()
    try:
        injected = pd.read_sql(
            "SELECT id, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    persona_config = OptimiserConfig(risk_level=1.0)
    received = {}

    import optimiser.squad as squad_module
    real_fn = squad_module.optimise_squad

    def _spy(*args, **kwargs):
        received["config"] = kwargs.get("config")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(squad_module, "optimise_squad", _spy)
    cs.build_initial_squad("2026-27", players=injected, config=persona_config)
    assert received["config"] is persona_config


def test_build_initial_squad_discounts_rumoured_player(temp_session, monkeypatch, tmp_path):
    import yaml

    from data import overrides as ov

    _seed_full_pool(temp_session)
    s = temp_session()
    try:
        rumoured_id = s.query(Player.id).filter_by(fpl_id=2).scalar()  # a non-leaver "p1"
        rumoured_code = s.query(Player.code).filter_by(fpl_id=2).scalar()
        injected = pd.read_sql(
            "SELECT id, code, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    overrides_path = tmp_path / "transfer_overrides.yaml"
    overrides_path.write_text(yaml.safe_dump({
        "rumoured": [
            {"code": rumoured_code, "p_leave": 0.9, "reason": "x", "as_of": "2026-08-10"},
        ],
    }))
    monkeypatch.setattr(ov, "OVERRIDES_PATH", overrides_path)
    # data/overrides.py resolves rumoured codes against the live players
    # table via its own `get_session` binding (separate from cs.get_session,
    # which the `temp_session` fixture above already patches) -- same
    # isolation fix tests/test_overrides.py's own `temp_session` fixture
    # applies, otherwise this reads the real DB instead of the seeded one.
    monkeypatch.setattr(ov, "get_session", lambda: temp_session())

    _, projections = cs.build_initial_squad("2026-27", players=injected)
    row = projections[projections["player_id"] == rumoured_id]
    assert (row["xpts"] == 0.0).all()  # p_leave=0.9 -> stay-probability multiplier 0.0


def _seed_two_team_fixture(Local, season="2026-27", gw=1, def_home=1000.0, def_away=1400.0):
    """Team 1 (weak defence, 1000) hosts Team 2 (strong defence, 1400) at
    ``gw`` for ``season``. Player p1 is on Team 1 (an easy home fixture vs a
    weak defence); p2 is on Team 2 (a hard away fixture vs a strong one)."""
    s = Local()
    try:
        s.add_all([
            Team(id=1, name="Weak", short_name="WEA"),
            Team(id=2, name="Strong", short_name="STR"),
        ])
        s.add(TeamSeasonStrength(
            season=season, team_id=1, code=101,
            strength_defence_home=def_home, strength_defence_away=def_home,
        ))
        s.add(TeamSeasonStrength(
            season=season, team_id=2, code=202,
            strength_defence_home=def_away, strength_defence_away=def_away,
        ))
        s.add(Fixture(fpl_id=1, season=season, gameweek=gw, team_h_id=1, team_a_id=2))
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Home",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.add(Player(fpl_id=2, code=2, first_name="B", second_name="B", web_name="Away",
                     team_id=2, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()


def test_load_horizon_fixtures_resolves_opponent_and_home_away(temp_session):
    _seed_two_team_fixture(temp_session)
    players = cs.load_current_players()
    out = cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=1)

    home_id = players.loc[players["web_name"] == "Home", "id"].iloc[0]
    away_id = players.loc[players["web_name"] == "Away", "id"].iloc[0]
    home_row = out[out["player_id"] == home_id].iloc[0]
    away_row = out[out["player_id"] == away_id].iloc[0]
    assert home_row["gameweek"] == 1
    assert bool(home_row["was_home"]) is True
    assert home_row["opp_defence_strength"] == pytest.approx(1400.0)
    assert bool(away_row["was_home"]) is False
    assert away_row["opp_defence_strength"] == pytest.approx(1000.0)


def test_load_horizon_fixtures_warns_when_a_real_team_has_zero_fixtures(temp_session, caplog):
    _seed_two_team_fixture(temp_session)  # teams 1 and 2, fixture at GW1
    s = temp_session()
    try:
        s.add(Team(id=3, name="Orphan", short_name="ORP"))
        s.add(Player(fpl_id=3, code=3, first_name="C", second_name="C", web_name="NoFixture",
                     team_id=3, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    with caplog.at_level(logging.WARNING):
        cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=1)
    assert "team_id 3 has no fixtures" in caplog.text


def test_load_horizon_fixtures_prior_season_fallback_when_current_is_zero(temp_session):
    # Current season (2026-27): both teams' defence strength unpublished (0,
    # the real pre-season state as of 2026-08-10). Prior season (2025-26)
    # has real values, joined on the stable `code`.
    s = temp_session()
    try:
        s.add_all([
            Team(id=1, name="Weak", short_name="WEA"),
            Team(id=2, name="Strong", short_name="STR"),
        ])
        s.add(TeamSeasonStrength(season="2026-27", team_id=1, code=101,
                                  strength_defence_home=0, strength_defence_away=0))
        s.add(TeamSeasonStrength(season="2026-27", team_id=2, code=202,
                                  strength_defence_home=0, strength_defence_away=0))
        s.add(TeamSeasonStrength(season="2025-26", team_id=1, code=101,
                                  strength_defence_home=1000, strength_defence_away=1000))
        s.add(TeamSeasonStrength(season="2025-26", team_id=2, code=202,
                                  strength_defence_home=1400, strength_defence_away=1400))
        s.add(Fixture(fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Home",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    out = cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=1)
    home_id = players.loc[players["web_name"] == "Home", "id"].iloc[0]
    home_row = out[out["player_id"] == home_id].iloc[0]
    assert home_row["opp_defence_strength"] == pytest.approx(1400.0)  # from 2025-26, via code=202


def test_load_horizon_fixtures_degrades_to_none_when_no_strength_data_at_all(temp_session):
    s = temp_session()
    try:
        s.add_all([
            Team(id=1, name="Weak", short_name="WEA"),
            Team(id=2, name="Strong", short_name="STR"),
        ])
        s.add(Fixture(fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Home",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    out = cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=1)
    home_id = players.loc[players["web_name"] == "Home", "id"].iloc[0]
    home_row = out[out["player_id"] == home_id].iloc[0]
    assert home_row["opp_defence_strength"] is None


def test_load_horizon_fixtures_spans_multiple_gws(temp_session):
    s = temp_session()
    try:
        s.add_all([
            Team(id=1, name="Weak", short_name="WEA"),
            Team(id=2, name="Strong", short_name="STR"),
            Team(id=3, name="Mid", short_name="MID"),
        ])
        s.add(TeamSeasonStrength(season="2026-27", team_id=1, code=101,
                                  strength_defence_home=1000, strength_defence_away=1000))
        s.add(TeamSeasonStrength(season="2026-27", team_id=2, code=202,
                                  strength_defence_home=1400, strength_defence_away=1400))
        s.add(TeamSeasonStrength(season="2026-27", team_id=3, code=303,
                                  strength_defence_home=1200, strength_defence_away=1200))
        s.add(Fixture(fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
        s.add(Fixture(fpl_id=2, season="2026-27", gameweek=2, team_h_id=3, team_a_id=1))
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Home",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    out = cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=2)
    pid = players.loc[players["web_name"] == "Home", "id"].iloc[0]
    rows = out[out["player_id"] == pid].sort_values("gameweek")
    assert list(rows["gameweek"]) == [1, 2]
    assert rows.iloc[0]["opp_defence_strength"] == pytest.approx(1400.0)  # GW1 vs Strong
    assert rows.iloc[1]["opp_defence_strength"] == pytest.approx(1200.0)  # GW2 vs Mid


def test_load_horizon_fixtures_empty_players_or_gws_returns_empty(temp_session):
    empty = pd.DataFrame(columns=["id", "team_id"])
    out = cs.load_horizon_fixtures(empty, "2026-27", target_gw=1, horizon=1)
    assert out.empty
    assert list(out.columns) == ["player_id", "gameweek", "opp_defence_strength", "was_home"]


def test_project_cold_start_horizon_1_is_byte_identical_to_default(temp_session):
    _seed(temp_session)
    players = cs.apply_departure_gate(cs.load_current_players())
    prior = cs.load_prior_season_features("2025-26")

    default_proj = cs.project_cold_start(players, prior)
    explicit_proj = cs.project_cold_start(players, prior, horizon=1, season="2026-27")
    pd.testing.assert_frame_equal(
        default_proj.sort_values("player_id").reset_index(drop=True),
        explicit_proj.sort_values("player_id").reset_index(drop=True),
    )


def test_project_cold_start_horizon_emits_one_row_per_gw_with_distinct_xpts(temp_session):
    _seed_two_team_fixture(temp_session, gw=1)
    s = temp_session()
    try:
        s.add(Fixture(fpl_id=2, season="2026-27", gameweek=2, team_h_id=2, team_a_id=1))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    home_id = players.loc[players["web_name"] == "Home", "id"].iloc[0]

    proj = cs.project_cold_start(players, prior, target_gw=1, horizon=2, season="2026-27")
    home_rows = proj[proj["player_id"] == home_id].sort_values("gameweek")

    assert list(home_rows["gameweek"]) == [1, 2]
    gw1_xpts, gw2_xpts = home_rows["xpts"].tolist()
    # GW1: home vs weak Team 2... wait -- Home is Team 1, opponent Team 2 is
    # the STRONG defence in _seed_two_team_fixture (1400) at GW1 (home), and
    # Team 2 (still strong, now at home) hosts Team 1 (away) at GW2 -- both
    # legs are against the same strong opponent, but home/away differs, so
    # the multipliers (and therefore xpts) must differ between the two rows.
    assert gw1_xpts != gw2_xpts
    from projection.fixture_adjust import fixture_multiplier
    base_xpts = home_rows["xpts"].iloc[0] / fixture_multiplier(1400.0, True)
    assert gw2_xpts == pytest.approx(base_xpts * fixture_multiplier(1400.0, False))


def test_project_cold_start_horizon_var_scales_with_multiplier_squared(temp_session):
    _seed_variance_pool(temp_session)
    s = temp_session()
    try:
        s.add_all([
            Team(id=1, name="T1", short_name="T1_"), Team(id=2, name="T2", short_name="T2_"),
        ])
        s.add(TeamSeasonStrength(season="2026-27", team_id=1, code=1001,
                                  strength_defence_home=1000, strength_defence_away=1000))
        s.add(TeamSeasonStrength(season="2026-27", team_id=2, code=1002,
                                  strength_defence_home=1000, strength_defence_away=1000))
        s.add(Fixture(fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    base_proj = cs.project_cold_start(players, prior, raw_appearances=raw)
    horizon_proj = cs.project_cold_start(
        players, prior, raw_appearances=raw, target_gw=1, horizon=1, season="2026-27"
    )

    from projection.fixture_adjust import fixture_multiplier
    varied_id = players.loc[players["web_name"] == "Varied", "id"].iloc[0]
    base_row = base_proj[base_proj["player_id"] == varied_id].iloc[0]
    horizon_row = horizon_proj[horizon_proj["player_id"] == varied_id].iloc[0]
    mult = fixture_multiplier(1000.0, True)  # Varied is on team_id=1, home
    assert horizon_row["xpts_var"] == pytest.approx(base_row["xpts_var"] * mult ** 2)


def test_project_cold_start_horizon_with_no_fixture_data_repeats_base_row_neutrally(temp_session):
    _seed(temp_session)  # no Fixture/TeamSeasonStrength rows seeded at all
    players = cs.apply_departure_gate(cs.load_current_players())
    prior = cs.load_prior_season_features("2025-26")

    proj = cs.project_cold_start(players, prior, target_gw=1, horizon=3, season="2026-27")
    estab_id = players.loc[players["web_name"] == "Estab", "id"].iloc[0]
    rows = proj[proj["player_id"] == estab_id].sort_values("gameweek")
    assert list(rows["gameweek"]) == [1, 2, 3]
    # same base value every GW, no crash -- plain equality (not
    # pytest.approx) since this is a pandas Series comparison, and
    # pytest.approx's __eq__ does not broadcast correctly against a Series
    # (it silently returns all-False rather than raising, a known
    # pandas/pytest interaction quirk); exact equality is safe here since
    # the degrade path is a literal copy of the base row with no arithmetic.
    assert rows["xpts"].tolist() == [6.0, 6.0, 6.0]


def test_build_initial_squad_uses_horizon_sum_not_single_gw(temp_session):
    _seed_full_pool(temp_session)
    s = temp_session()
    try:
        # give every club a real GW1-5 fixture list against itself in a
        # round-robin so load_horizon_fixtures has something to resolve
        # (content doesn't matter here -- only that >1 distinct GW exists).
        club_ids = list(range(1, 9))
        fpl_id = 1
        for gw in range(1, 6):
            for i in range(0, len(club_ids), 2):
                s.add(Fixture(fpl_id=fpl_id, season="2026-27", gameweek=gw,
                              team_h_id=club_ids[i], team_a_id=club_ids[i + 1]))
                fpl_id += 1
        s.commit()
        injected = pd.read_sql(
            "SELECT id, code, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    solution_5gw, proj_5gw = cs.build_initial_squad("2026-27", players=injected)
    solution_1gw, proj_1gw = cs.build_initial_squad(
        "2026-27", players=injected,
        config=OptimiserConfig(cold_start_lookahead_gws=1),
    )

    assert proj_5gw["gameweek"].nunique() == 5
    assert proj_1gw["gameweek"].nunique() == 1
    assert len(solution_5gw.squad) == 15
    assert len(solution_1gw.squad) == 15


def test_build_initial_squad_regression_single_gw_config_matches_pre_feature_a_shape(
    temp_session,
):
    """Regression guard (spec's explicit requirement): a caller pinning
    cold_start_lookahead_gws=1 must get a projections frame shaped exactly
    like the pre-Feature-A single-GW shape -- one row per player, all at
    target_gw=1. Only the SHAPE is guaranteed here, not byte-identical
    values: with real fixture data present, xpts/xpts_var are fixture-
    scaled even at horizon=1 (that's correct, intended behaviour) -- the
    byte-identical-values path is season=None, not horizon=1."""
    _seed_full_pool(temp_session)
    s = temp_session()
    try:
        injected = pd.read_sql(
            "SELECT id, code, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    _, projections = cs.build_initial_squad(
        "2026-27", players=injected, config=OptimiserConfig(cold_start_lookahead_gws=1),
    )
    assert (projections["gameweek"] == 1).all()
    assert projections["player_id"].nunique() == len(projections)  # exactly one row per player


# --- P0.1 availability weighting (2026-08-16,
# plan/decision-engine-recovery-plan.md) -----------------------------------
#
# Cold-start xpts used to be points-per-APPEARANCE while the in-season
# engine (projection/assemble.py) produces an unconditional scenario mean,
# so a rotation risk scoring 6/appearance was valued identically to a nailed
# player scoring 6/appearance. These pin the corrected semantics.


def test_appearance_probability_windows_from_first_appearance():
    """A January arrival who then played every week is a NAILED player, not
    a 50%-availability one -- the denominator is the window from their first
    appearance to the end of the season, not the whole season."""
    # played GW20-38 inclusive (19 GWs) having arrived in January
    assert cs.appearance_probability(19, 20, 38) == pytest.approx(1.0)
    # played GW1-19 then injured out for the rest -- genuine availability risk
    assert cs.appearance_probability(19, 1, 38) == pytest.approx(19 / 38)
    # ever-present
    assert cs.appearance_probability(38, 1, 38) == pytest.approx(1.0)


def test_appearance_probability_degrades_safely():
    assert cs.appearance_probability(0, None, 38) == 0.0
    assert cs.appearance_probability(5, 1, 0) == 0.0
    # more appearance-gameweeks than the window can hold: clamped, never > 1
    assert cs.appearance_probability(10, 35, 38) == pytest.approx(1.0)


def test_unconditional_moments_matches_the_mixture_decomposition():
    # E[X] = p*m ; Var(X) = p*v + p(1-p)*m^2
    mean, var = cs.unconditional_moments(0.5, 6.0, 4.0)
    assert mean == pytest.approx(3.0)
    assert var == pytest.approx(0.5 * 4.0 + 0.5 * 0.5 * 36.0)


def test_unconditional_moments_is_identity_for_an_ever_present_player():
    """p=1 must leave both moments untouched -- the whole point is that a
    nailed player is unaffected and only availability risk is priced."""
    mean, var = cs.unconditional_moments(1.0, 6.0, 4.0)
    assert mean == pytest.approx(6.0)
    assert var == pytest.approx(4.0)


def test_unconditional_moments_adds_variance_for_a_rotation_risk():
    """Same per-appearance return, different availability: the rotation risk
    must come out with a LOWER mean and a HIGHER variance. Previously the
    optimiser could see neither difference."""
    nailed_mean, nailed_var = cs.unconditional_moments(1.0, 6.0, 4.0)
    rotated_mean, rotated_var = cs.unconditional_moments(0.5, 6.0, 4.0)
    assert rotated_mean < nailed_mean
    assert rotated_var > nailed_var


def test_load_prior_season_features_emits_p_appear(temp_session):
    ids = _seed(temp_session)
    feats = cs.load_prior_season_features("2025-26")
    row = feats[feats["player_id"] == ids["p1"]].iloc[0]
    # _seed gives p1 GW1-10 of a 10-gameweek season: ever-present
    assert row["p_appear"] == pytest.approx(1.0)


def test_rotation_risk_is_discounted_against_an_equal_per_appearance_peer(temp_session):
    """The headline behaviour change: two players with IDENTICAL points per
    appearance, one ever-present and one playing half the time, must no
    longer project the same."""
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Nailed",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.add(Player(fpl_id=2, code=2, first_name="B", second_name="B", web_name="Rotated",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
        nailed = s.query(Player.id).filter_by(fpl_id=1).scalar()
        rotated = s.query(Player.id).filter_by(fpl_id=2).scalar()
        for gw in range(1, 21):
            s.add(PlayerGameweekStats(player_id=nailed, gameweek=gw, season="2025-26",
                                      minutes=90, total_points=6))
            # same 6 pts when he plays, but only every other gameweek.
            # Starts at GW1 so his window is the full 20 -- starting at GW2
            # would (correctly) give a 19-gameweek window instead.
            s.add(PlayerGameweekStats(
                player_id=rotated, gameweek=gw, season="2025-26",
                minutes=90 if gw % 2 == 1 else 0, total_points=6 if gw % 2 == 1 else 0,
            ))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    proj = cs.project_cold_start(players, prior, raw_appearances=raw)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    nailed_row = proj[proj["name"] == "Nailed"].iloc[0]
    rotated_row = proj[proj["name"] == "Rotated"].iloc[0]

    assert nailed_row["proj_source"] == rotated_row["proj_source"] == "prior_season"
    assert nailed_row["xpts"] == pytest.approx(6.0)          # ever-present: unchanged
    assert rotated_row["xpts"] == pytest.approx(3.0)         # 10 of 20 gameweeks
    assert rotated_row["xpts_var"] > nailed_row["xpts_var"]  # and genuinely riskier


def test_project_cold_start_without_p_appear_column_is_unweighted(temp_session):
    """Backwards compatibility: a prior_features frame predating P0.1 must
    reproduce the old per-appearance behaviour exactly, not silently deflate
    every projection by a default weight."""
    ids = _seed(temp_session)
    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26").drop(columns=["p_appear"])
    proj = cs.project_cold_start(players, prior)
    row = proj[proj["player_id"] == ids["p1"]].iloc[0]
    assert row["xpts"] == pytest.approx(6.0)


def test_prior_league_falls_back_to_raw_output_when_expected_goals_are_absent():
    """The FBref prior-league scrape only populated the basic stats: measured
    2026-08-16, goals90/assists90 have 7,413/7,097 non-zero rows and
    npxg90/xa90/sca90 have ZERO. Using the expected-goals inputs regardless
    projected every league-tier player at the floor, making a genuinely good
    foreign signing invisible to the optimiser."""
    row = {"league": "ESP-La Liga", "npxg90": 0.0, "xa90": 0.0,
           "goals90": 0.5, "assists90": 0.3, "minutes": 2700, "matches": 30}
    xpts, _var, _sp = cs._prior_league_projection("FWD", row)
    assert xpts > cs._MIN_XPTS * 2, "raw output must reach the projection"


def test_prior_league_prefers_expected_goals_when_they_exist():
    """npxG/xA are smoother and luck-adjusted; one season's raw output is a
    small, high-variance sample. The fallback is for absent data only."""
    smooth = {"league": "ESP-La Liga", "npxg90": 0.6, "xa90": 0.4,
              "goals90": 0.0, "assists90": 0.0, "minutes": 2700, "matches": 30}
    raw = {"league": "ESP-La Liga", "npxg90": 0.0, "xa90": 0.0,
           "goals90": 0.6, "assists90": 0.4, "minutes": 2700, "matches": 30}
    assert cs._prior_league_projection("FWD", smooth)[0] == pytest.approx(
        cs._prior_league_projection("FWD", raw)[0]
    )


def test_prior_league_with_no_output_at_all_still_floors_safely():
    row = {"league": "ESP-La Liga", "npxg90": 0.0, "xa90": 0.0,
           "goals90": 0.0, "assists90": 0.0, "minutes": 900, "matches": 10}
    xpts, var, sp = cs._prior_league_projection("DEF", row)
    assert xpts >= 0.0 and var >= 0.0 and 0.0 <= sp <= 1.0


# --- penalty duty in the cold start (2026-08-17) ----------------------------
def test_penalty_bonus_skips_a_taker_who_was_already_on_penalties():
    """The double-count guard. An established taker's prior-season points
    already contain his penalties; adding the duty again would pay him twice
    for the same spot-kicks."""
    from projection.cold_start import penalty_bonus

    assert penalty_bonus("FWD", 0.0806, prior_penalty_xg=3.2) == (0.0, 0.0)


def test_penalty_bonus_applies_to_a_taker_new_to_the_duty():
    """The case it exists for: on penalties now, none last season, so the
    prior-season record carries no penalty component at all."""
    from projection.cold_start import penalty_bonus

    mean, var = penalty_bonus("FWD", 0.0806, prior_penalty_xg=0.0)
    assert mean == pytest.approx(0.0806 * 4)
    assert var == pytest.approx(0.0806 * 16)
    assert var > mean, "Poisson goal points are over-dispersed in points terms"


def test_penalty_bonus_scales_with_position_scoring():
    """A defender's goal is worth more than a forward's, so the same duty is
    worth more to him."""
    from projection.cold_start import penalty_bonus

    fwd, _ = penalty_bonus("FWD", 0.0806, 0.0)
    dfd, _ = penalty_bonus("DEF", 0.0806, 0.0)
    assert dfd > fwd


def test_penalty_bonus_is_nothing_without_a_duty():
    from projection.cold_start import penalty_bonus

    assert penalty_bonus("MID", 0.0, prior_penalty_xg=0.0) == (0.0, 0.0)


def test_cold_start_lifts_a_newly_appointed_taker_and_no_one_else():
    """End to end through project_cold_start: the bonus reaches xpts, and a
    player without the duty is byte-for-byte unchanged."""
    import pandas as pd

    from projection.cold_start import project_cold_start

    players = pd.DataFrame([
        {"id": 1, "position": "FWD", "now_cost": 90.0, "code": 111},
        {"id": 2, "position": "FWD", "now_cost": 90.0, "code": 222},
    ])
    prior = pd.DataFrame([
        {"player_id": 1, "appearances": 30, "ppg_played": 5.0,
         "starts_rate": 0.9, "p_appear": 1.0},
        {"player_id": 2, "appearances": 30, "ppg_played": 5.0,
         "starts_rate": 0.9, "p_appear": 1.0},
    ])

    without = project_cold_start(players, prior)
    with_duty = project_cold_start(players, prior, penalty_duty={1: 0.0806})

    x_without = without.set_index("player_id")["xpts"]
    x_with = with_duty.set_index("player_id")["xpts"]

    assert x_with[1] == pytest.approx(x_without[1] + 0.0806 * 4)
    assert x_with[2] == x_without[2], "a non-taker must not move"
