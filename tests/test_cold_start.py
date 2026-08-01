"""T7 gate — GW1 cold-start projections + departure gate.

Self-contained (temp DB). Proves the contract: every candidate gets a
non-default projection source (prior-season or position/price prior — never a
silent 0.0), and confirmed leavers (status 'u') are dropped.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Player, PlayerGameweekStats, PriorLeagueStats
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
    assert row["xpts"] == pytest.approx(cs._price_prior("GKP", 4.5))
    assert row["xpts_var"] == pytest.approx(cs._FALLBACK_VAR)


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
    assert row["xpts"] > cs._price_prior("FWD", 6.5)
    assert row["xpts_var"] == pytest.approx(
        cs.PRIOR_LEAGUE.translation_variance("ENG-Championship")
    )
    # nailed-on Championship starter (3000/34/90 ~= 0.98 share) blended 50/50
    # with the flat 0.6 default -> higher than the flat default alone.
    assert row["start_probability"] > cs.NEW_PLAYER_START_PROB


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
