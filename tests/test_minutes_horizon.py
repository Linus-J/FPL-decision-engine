"""A8 — per-gameweek minutes bands across the projection horizon.

Real defect: ``predict_minutes_bands`` collapsed its feature frame with
``groupby("player_id").last()`` and returned ``dict[player_id -> bands]``.
``assemble_gw_projections`` called it ONCE, outside the ``for gw in target_gws``
loop, and then read the same scalar on every iteration — so a player's
``start_probability`` was byte-identical across the whole horizon. Measured on
the live database before this change: 0 of 615 players showed any variation
across GW4-GW8.

That made the A6 congestion features unreachable in the one place they were
built for. A post-international break at GW6 gives every club 14 days' rest;
a Champions League tie the Tuesday before GW7 takes three days off it. Neither
could move a projection that was decided once, at GW3's fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projection import assemble
from projection import minutes_model as mm

SEASON = "2026-27"
TEAMS = {1: 10, 2: 20, 3: 30}
HISTORY_GWS = [1, 2, 3]
TARGET_GW = 4
HORIZON = 3


# --- fixtures -------------------------------------------------------------

def _played() -> pd.DataFrame:
    """A ``player_gw_stats``-shaped history frame, strictly before TARGET_GW."""
    return pd.DataFrame([
        {
            "player_id": pid, "season": SEASON, "gameweek": gw,
            # distinct per player so a test can tell the frozen rolling block
            # apart between players as well as between gameweeks
            "minutes": 60 + 10 * pid, "total_points": 5, "goals_scored": 0, "assists": 0,
            "clean_sheets": 0, "goals_conceded": 0, "saves": 0,
            "yellow_cards": 0, "red_cards": 0, "bonus": 0, "bps": 10,
            "xg": 0.3, "xa": 0.1, "key_passes": 1, "dribbles": 0,
            "position": "MID", "was_home": True, "opponent_team_id": 99,
            "team_id_season": team, "team_id": team,
            "now_cost": 5.0, "selected_by_percent": 1.0,
            "status": "a", "chance_of_playing_next_round": 100,
        }
        for pid, team in TEAMS.items()
        for gw in HISTORY_GWS
    ])


def _horizon_fixtures() -> pd.DataFrame:
    """Player 1's club hosts player 2's club every horizon gameweek; player 3's
    club is unpaired and so is never sampled (it only needs to exist)."""
    rows = []
    for gw in range(TARGET_GW, TARGET_GW + HORIZON):
        rows.append({"player_id": 1, "gameweek": gw, "team_id_season": 10,
                     "opponent_team_id": 20, "was_home": True})
        rows.append({"player_id": 2, "gameweek": gw, "team_id_season": 20,
                     "opponent_team_id": 10, "was_home": False})
    return pd.DataFrame(rows)


def _match_odds() -> pd.DataFrame:
    return pd.DataFrame([
        {"gameweek": gw, "home_team_id": 10, "away_team_id": 20,
         "home_win_prob": 0.5, "draw_prob": 0.25, "away_win_prob": 0.25,
         "over25_prob": 0.5}
        for gw in range(TARGET_GW, TARGET_GW + HORIZON)
    ])


# Rest days per horizon gameweek: a normal week, a European week, then a
# post-international break. The three are far enough apart that any model
# reading ``days_since_last_match`` must produce three different answers.
REST_BY_GW = {TARGET_GW: 3.0, TARGET_GW + 1: 7.0, TARGET_GW + 2: 14.0}


def _congestion() -> pd.DataFrame:
    """Covers the horizon gameweeks AND the history ones — ``load_congestion``
    returns the whole calendar, not a slice, and the history rows are what the
    old collapse-to-last path was reading."""
    rows = []
    for team in TEAMS.values():
        for gw in [*HISTORY_GWS, *REST_BY_GW]:
            rest = REST_BY_GW.get(gw, 5.0)
            rows.append({
                "season": SEASON, "team_id": team, "gameweek": gw,
                "days_since_last_match": rest,
                "days_to_next_match": rest,
                "matches_in_prev_14d": 1.0,
                "euro_match_in_prev_7d": 0.0,
                "euro_competition_tier": 0.0,
                "is_post_international_break": 1.0 if rest >= 14.0 else 0.0,
            })
    return pd.DataFrame(rows)


class _RestSensitiveModel:
    """P(60+) rises linearly with rest days. Not a plausible minutes model —
    a probe. It exists so that if the served frame carries per-gameweek
    congestion, the bands MUST differ per gameweek, and if it does not, they
    cannot."""

    classes_ = [0, 1, 2]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rest = np.clip(X["days_since_last_match"].to_numpy() / 14.0, 0.0, 1.0)
        return np.column_stack([1.0 - rest, np.zeros_like(rest), rest])


def _empty_keyed(*extra: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=["player_id", "gameweek", "season", "opponent_team_id", *extra]
    )


def _fake_sample_fixture(rng, home_players, away_players, lam_home, lam_away, n, shares):
    return {p["player_id"]: np.full(n, 4.0) for p in [*home_players, *away_players]}


@pytest.fixture
def hermetic(monkeypatch):
    """No database. Every loader ``_build_features`` reaches for is stubbed;
    only the congestion table carries real per-gameweek structure."""
    monkeypatch.setattr(mm, "load_congestion", lambda: _congestion())
    monkeypatch.setattr(mm, "load_fixture_difficulty", lambda: _empty_keyed())
    monkeypatch.setattr(mm, "load_fixture_odds", lambda: _empty_keyed(
        "my_cs_prob", "opp_cs_prob", "over25_prob"))
    # The live FDR/odds readers hit the database directly (there is no
    # ``team_matches``-style single load point to stub upstream of them) --
    # stub them here too so this fixture stays fully hermetic even for tests
    # that pass ``fixture_context``/``match_odds`` through to
    # ``predict_minutes_bands_by_gameweek``.
    monkeypatch.setattr(mm, "load_live_fixture_difficulty", lambda *a, **k: _empty_keyed(
        "is_home", "is_bgw", "opp_defence_strength", "opp_attack_strength",
        "own_attack_strength", "own_defence_strength", "own_overall_strength",
        "next_3gw_avg_opp_defence", "next_3gw_avg_opp_attack"))
    monkeypatch.setattr(mm, "load_live_fixture_odds", lambda *a, **k: _empty_keyed(
        "my_cs_prob", "opp_cs_prob", "over25_prob"))
    monkeypatch.setattr(mm, "load_player_enrichment", lambda: pd.DataFrame([
        {
            "player_id": pid, "season": SEASON, "gameweek": gw,
            "is_penalty_taker": 0.0, "penalty_xg_per_game": 0.0,
            "is_set_piece_taker": 0.0, "key_passes_per_game": 0.0,
            "injury_severity": 0.0, "price_momentum": 0.0,
            "transfer_velocity": 0.0,
        }
        for pid in TEAMS
        for gw in [*HISTORY_GWS, *REST_BY_GW]
    ]))
    monkeypatch.setattr(assemble, "sample_fixture", _fake_sample_fixture)


# --- the defect -----------------------------------------------------------

def test_start_probability_is_not_flat_across_the_horizon(hermetic):
    """The acceptance gate. Three horizon gameweeks with 3, 7 and 14 rest days
    must not produce one repeated start_probability."""
    out = assemble.assemble_gw_projections(
        history=_played(), all_stats=_horizon_fixtures(),
        minutes_model=_RestSensitiveModel(),
        target_gw=TARGET_GW, horizon=HORIZON, match_odds=_match_odds(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1, season=SEASON,
    )
    assert not out.empty
    per_player = out.groupby("player_id")["start_probability"].nunique()
    assert (per_player == HORIZON).all(), (
        f"start_probability is flat across the horizon: {out.to_dict('records')}"
    )

    # and flat in the RIGHT direction -- more rest, more likely to start
    p1 = out[out["player_id"] == 1].sort_values("gameweek")["start_probability"]
    assert p1.is_monotonic_increasing


def test_the_horizon_reads_its_own_gameweeks_congestion(hermetic):
    """Not merely 'varies' — each gameweek's value must be the one the calendar
    gives THAT gameweek. The probe model makes P(60+) == rest/14 exactly."""
    out = assemble.assemble_gw_projections(
        history=_played(), all_stats=_horizon_fixtures(),
        minutes_model=_RestSensitiveModel(),
        target_gw=TARGET_GW, horizon=HORIZON, match_odds=_match_odds(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1, season=SEASON,
    )
    got = out[out["player_id"] == 1].set_index("gameweek")["start_probability"]
    for gw, rest in REST_BY_GW.items():
        assert got.loc[gw] == pytest.approx(rest / 14.0), f"gameweek {gw}"


# --- the asymmetry, pinned rather than left implicit -----------------------

class _RecordingModel:
    """Keeps every feature matrix it is asked to score, so a test can assert
    what actually varied across the horizon."""

    classes_ = [0, 1, 2]

    def __init__(self) -> None:
        self.seen: list[pd.DataFrame] = []

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.seen.append(X.copy())
        n = len(X)
        return np.column_stack([np.zeros(n), np.zeros(n), np.ones(n)])


def test_only_the_calendar_features_vary_across_the_horizon(hermetic):
    """The train/serve asymmetry, asserted rather than described in prose.

    Congestion is recomputed per gameweek because ``team_matches`` knows the
    fixture calendar in advance. The rolling history block is FROZEN at its
    as-of-target_gw value, because GW5's rolling average is a function of GW4's
    outcome and that does not exist yet. Forecasting it would be a leak in the
    backtest and a fabrication live."""
    model = _RecordingModel()
    mm.predict_minutes_bands_by_gameweek(
        _played(), model, list(REST_BY_GW), season=SEASON
    )
    assert len(model.seen) == 1
    x = model.seen[0]
    assert len(x) == len(TEAMS) * len(REST_BY_GW)

    frozen = [c for c in mm.FEATURE_COLS if c not in mm.CONGESTION_FEATURE_COLS]
    # every horizon row for a given player is the SAME on the frozen block --
    # len(TEAMS) distinct rows, not len(TEAMS) * len(REST_BY_GW)
    assert len(x[frozen].drop_duplicates()) == len(TEAMS)
    # ...and the calendar block genuinely moves
    assert x["days_since_last_match"].nunique() == len(set(REST_BY_GW.values()))


def test_fdr_and_odds_vary_across_the_horizon_given_a_fixture_context(hermetic, monkeypatch):
    """Item 5 / the FDR-odds-serve-freeze fix. Without ``fixture_context`` /
    ``match_odds``, FDR and odds are frozen at the as-of row's value (the
    previous test pins exactly that). Given them, they must vary per
    gameweek same as congestion already does."""
    fdr_by_gw = {
        TARGET_GW: 1000.0, TARGET_GW + 1: 1200.0, TARGET_GW + 2: 1400.0,
    }
    cs_by_gw = {
        TARGET_GW: 0.10, TARGET_GW + 1: 0.30, TARGET_GW + 2: 0.50,
    }

    def _fake_fdr(fixture_context, season):
        rows = []
        for gw in fixture_context["gameweek"].unique():
            for pid in fixture_context.loc[
                fixture_context["gameweek"] == gw, "player_id"
            ].unique():
                rows.append({
                    "player_id": pid, "gameweek": gw, "season": season,
                    "opponent_team_id": 99, "opp_defence_strength": fdr_by_gw[gw],
                })
        return pd.DataFrame(rows)

    def _fake_odds(fixture_context, match_odds, season):
        rows = []
        for gw in fixture_context["gameweek"].unique():
            for pid in fixture_context.loc[
                fixture_context["gameweek"] == gw, "player_id"
            ].unique():
                rows.append({
                    "player_id": pid, "gameweek": gw, "season": season,
                    "opponent_team_id": 99, "my_cs_prob": cs_by_gw[gw],
                    "opp_cs_prob": 0.2, "over25_prob": 0.5,
                })
        return pd.DataFrame(rows)

    monkeypatch.setattr(mm, "load_live_fixture_difficulty", _fake_fdr)
    monkeypatch.setattr(mm, "load_live_fixture_odds", _fake_odds)

    fixture_context = pd.DataFrame([
        {"player_id": pid, "gameweek": gw, "team_id_season": team,
         "opponent_team_id": 99, "was_home": True}
        for pid, team in TEAMS.items()
        for gw in REST_BY_GW
    ])
    match_odds = _match_odds()

    model = _RecordingModel()
    mm.predict_minutes_bands_by_gameweek(
        _played(), model, list(REST_BY_GW), season=SEASON,
        fixture_context=fixture_context, match_odds=match_odds,
    )
    x = model.seen[0]
    assert x["opp_defence_strength"].nunique() == len(fdr_by_gw)
    assert x["my_cs_prob"].nunique() == len(cs_by_gw)
    # ``predict_proba`` only ever sees ``FEATURE_COLS`` (no player_id/gameweek
    # to key on) -- but the frame is built one gameweek-block at a time, in
    # ``target_gws`` order, each block exactly ``len(TEAMS)`` rows long, so
    # position alone recovers which gameweek a row belongs to.
    n = len(TEAMS)
    for i, (gw, expected) in enumerate(fdr_by_gw.items()):
        block = x.iloc[i * n:(i + 1) * n]
        assert block["opp_defence_strength"].unique().tolist() == [expected], f"gameweek {gw}"
        assert block["my_cs_prob"].unique().tolist() == [cs_by_gw[gw]], f"gameweek {gw}"


def test_the_scalar_api_still_answers_for_the_next_match_only(hermetic):
    """``predict_minutes_bands`` is kept, unchanged, for callers that want
    'what happens next' rather than a horizon. Reverting A8 is one import."""
    bands = mm.predict_minutes_bands(_played(), _RestSensitiveModel())
    assert set(bands) == set(TEAMS)
    assert all(isinstance(k, int) for k in bands)
    # A10: the as-of row is TARGET_GW -- the match about to be played, on its
    # own 3.0 rest days. It used to be GW3, the match already finished, on the
    # 5.0-day default; "the next match" answered with the last one's calendar
    # and with rolling form a gameweek stale. See test_minutes_asof_staleness.
    assert bands[1][2] == pytest.approx(REST_BY_GW[TARGET_GW] / 14.0)


# --- the double gameweek the per-gameweek key had to not break ------------

def _dgw_fixtures() -> pd.DataFrame:
    """Player 1's club plays TWICE in the target gameweek: at home to player
    2's club, then away at player 3's."""
    return pd.DataFrame([
        {"player_id": 1, "gameweek": TARGET_GW, "team_id_season": 10,
         "opponent_team_id": 20, "was_home": True},
        {"player_id": 2, "gameweek": TARGET_GW, "team_id_season": 20,
         "opponent_team_id": 10, "was_home": False},
        {"player_id": 3, "gameweek": TARGET_GW, "team_id_season": 30,
         "opponent_team_id": 10, "was_home": True},
        {"player_id": 1, "gameweek": TARGET_GW, "team_id_season": 10,
         "opponent_team_id": 30, "was_home": False},
    ])


def test_a_double_gameweek_still_combines_to_p_at_least_one(hermetic):
    """assemble.py combines a DGW player's two fixtures as
    ``1 - (1-p)(1-p)``. Under the (player_id, gameweek) key BOTH fixtures now
    look up the same gameweek, so they must draw the SAME p -- the combination
    is unchanged, not silently applied to two different numbers."""
    odds = pd.DataFrame([
        {"gameweek": TARGET_GW, "home_team_id": 10, "away_team_id": 20,
         "home_win_prob": 0.5, "draw_prob": 0.25, "away_win_prob": 0.25,
         "over25_prob": 0.5},
        {"gameweek": TARGET_GW, "home_team_id": 30, "away_team_id": 10,
         "home_win_prob": 0.4, "draw_prob": 0.3, "away_win_prob": 0.3,
         "over25_prob": 0.5},
    ])
    out = assemble.assemble_gw_projections(
        history=_played(), all_stats=_dgw_fixtures(),
        minutes_model=_RestSensitiveModel(),
        target_gw=TARGET_GW, horizon=1, match_odds=odds,
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1, season=SEASON,
    )
    assert out["player_id"].tolist().count(1) == 1, "one row per player per gw"
    p = REST_BY_GW[TARGET_GW] / 14.0
    dgw = out[out["player_id"] == 1].iloc[0]
    single = out[out["player_id"] == 2].iloc[0]
    assert dgw["start_probability"] == pytest.approx(1.0 - (1.0 - p) ** 2)
    assert single["start_probability"] == pytest.approx(p)


# --- availability is read NOW, not at the last played deadline -------------

def _flagged_doubtful_last_deadline() -> pd.DataFrame:
    """Player 1 was 'd' at 75% before the last played gameweek's deadline --
    the point-in-time snapshot ``load_all_stats`` joins onto that row."""
    played = _played()
    last = (played["player_id"] == 1) & (played["gameweek"] == HISTORY_GWS[-1])
    played.loc[last, "status"] = "d"
    played.loc[last, "chance_of_playing_next_round"] = 75
    return played


def _availability(**by_pid: tuple[str, int | None]) -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": pid, "status": status, "chance_of_playing_next_round": cop}
        for pid, (status, cop) in ((int(k[1:]), v) for k, v in by_pid.items())
    ])


def test_a_recovered_player_is_not_scaled_by_last_deadlines_doubt(hermetic):
    """Real defect, 2026-27 GW6: Joao Pedro was 'd' 75% before the GW5
    deadline, then cleared to 'a' 100% before GW6. The as-of row is a copy of
    his GW5 row, so the 75% scaling was still applied to GW6-GW10."""
    bands = mm.predict_minutes_bands_by_gameweek(
        _flagged_doubtful_last_deadline(), _RestSensitiveModel(), [TARGET_GW],
        season=SEASON, current_availability=_availability(p1=("a", 100)),
    )
    assert bands[(1, TARGET_GW)][2] == pytest.approx(REST_BY_GW[TARGET_GW] / 14.0)


def test_a_newly_doubtful_player_is_scaled_by_todays_chance(hermetic):
    """The other direction: fit at the last deadline, 'd' 50% now."""
    bands = mm.predict_minutes_bands_by_gameweek(
        _played(), _RestSensitiveModel(), [TARGET_GW],
        season=SEASON, current_availability=_availability(p2=("d", 50)),
    )
    assert bands[(2, TARGET_GW)][2] == pytest.approx(0.5 * REST_BY_GW[TARGET_GW] / 14.0)


def test_a_player_missing_from_current_availability_keeps_the_asof_flag(hermetic):
    bands = mm.predict_minutes_bands_by_gameweek(
        _flagged_doubtful_last_deadline(), _RestSensitiveModel(), [TARGET_GW],
        season=SEASON, current_availability=_availability(p2=("a", 100)),
    )
    assert bands[(1, TARGET_GW)][2] == pytest.approx(0.75 * REST_BY_GW[TARGET_GW] / 14.0)


def test_assemble_passes_current_availability_through(hermetic):
    out = assemble.assemble_gw_projections(
        history=_flagged_doubtful_last_deadline(), all_stats=_horizon_fixtures(),
        minutes_model=_RestSensitiveModel(),
        target_gw=TARGET_GW, horizon=1, match_odds=_match_odds(),
        defcon_events=pd.DataFrame(), defcon_field_shares={"DEF": {}, "MID_FWD": {}},
        n_scenarios=4, seed=1, season=SEASON,
        current_availability=_availability(p1=("a", 100)),
    )
    p1 = out[out["player_id"] == 1].iloc[0]
    assert p1["start_probability"] == pytest.approx(REST_BY_GW[TARGET_GW] / 14.0)
