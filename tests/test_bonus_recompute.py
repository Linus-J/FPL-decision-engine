"""T5b gate — 26/27 bonus recompute pipeline + FBref mapper + sanity harness.

Network-free: the recompute core operates on the DB and plain mappings, and the
FBref column-mappers are pure. The live FBref scrape (browser-only) is exercised
separately when an event-capable environment is available.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.ingestors import fbref
from data.models import Base, Player, PlayerMatchEvents, RecomputedBonus
from projection import bonus_recompute as br

# --- fixture events (hand-computable under 26/27 BPS_WEIGHTS) ---------------
_FWD_BRACE = {"position": "FWD", "minutes": 90, "goals": 2}            # 6+48 = 54
_MID_GA = {"position": "MID", "minutes": 90, "goals": 1, "assists": 1}  # 6+18+9 = 33
_DEF_CS = {"position": "DEF", "minutes": 90, "clean_sheet": 1}          # 6+12 = 18


# --- pure recompute ---------------------------------------------------------
def test_recompute_fixture_bps_and_bonus():
    result = br.recompute_fixture({1: _FWD_BRACE, 2: _MID_GA, 3: _DEF_CS})
    assert result[1] == (54, 3)
    assert result[2] == (33, 2)
    assert result[3] == (18, 1)


def test_event_to_mapping_round_trips_through_sim():
    row = PlayerMatchEvents(
        player_id=1, season="2025-26", game_id="g", position="FWD",
        minutes=90, goals=2,
    )
    # attribute defaults are None pre-flush, so mimic a flushed row's ints
    for f in br.EVENT_FIELDS:
        if getattr(row, f) is None:
            setattr(row, f, 0)
    ev = br.event_to_mapping(row)
    assert ev["position"] == "FWD"
    assert br.recompute_fixture({1: ev})[1] == (54, 3)


# --- DB-backed recompute (temp DB) ------------------------------------------
@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'br.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    for pid in (1, 2, 3, 4):
        s.add(Player(id=pid, fpl_id=pid, code=pid, first_name="P", second_name=str(pid),
                     web_name=f"p{pid}", team_id=1, position="MID", now_cost=5.0))
    s.commit()
    yield s
    s.close()


def _seed_match(s, game_id, gw, players):
    for pid, ev in players.items():
        s.add(PlayerMatchEvents(player_id=pid, season="2025-26", gameweek=gw,
                                game_id=game_id, **ev))
    s.commit()


def test_recompute_season_writes_and_covers(session):
    _seed_match(session, "g1", 5, {1: _FWD_BRACE, 2: _MID_GA, 3: _DEF_CS})
    matches, written = br.recompute_season(session, "2025-26")
    assert matches == 1
    assert written == 3

    rows = {r.player_id: (r.bps_2627, r.bonus_2627)
            for r in session.query(RecomputedBonus).all()}
    assert rows == {1: (54, 3), 2: (33, 2), 3: (18, 1)}
    assert session.query(RecomputedBonus).first().gameweek == 5
    assert br.recomputed_bonus_coverage(session, "2025-26") == 1.0


def test_recompute_is_idempotent(session):
    _seed_match(session, "g1", 5, {1: _FWD_BRACE, 2: _MID_GA, 3: _DEF_CS})
    br.recompute_season(session, "2025-26")
    br.recompute_season(session, "2025-26")  # rerun: upsert, no duplicates
    assert session.query(RecomputedBonus).count() == 3


def test_dgw_player_gets_a_row_per_match(session):
    # player 1 plays twice in GW5 (two game_ids) → two recomputed rows.
    _seed_match(session, "g1", 5, {1: _FWD_BRACE, 2: _MID_GA})
    _seed_match(session, "g2", 5, {1: _MID_GA, 3: _DEF_CS})
    br.recompute_season(session, "2025-26")
    p1_rows = session.query(RecomputedBonus).filter_by(player_id=1).all()
    assert len(p1_rows) == 2
    assert {r.game_id for r in p1_rows} == {"g1", "g2"}


# --- old-rules sanity harness -----------------------------------------------
def test_oldrules_reproduction_perfect_agreement():
    events = {"g1": {1: _FWD_BRACE, 2: _MID_GA, 3: _DEF_CS}}
    actual = {"g1": {1: 3, 2: 2, 3: 1}}  # matches the ranking under old rules too
    m = br.oldrules_reproduction(events, actual)
    assert m["n_matches"] == 1.0
    assert m["slot_exact_rate"] == 1.0
    assert m["recipient_jaccard"] == 1.0


def test_oldrules_reproduction_detects_disagreement():
    events = {"g1": {1: _FWD_BRACE, 2: _MID_GA, 3: _DEF_CS}}
    # FPL actually gave the DEF the 3 (Opta metrics we lack) → we disagree
    actual = {"g1": {3: 3, 1: 2, 2: 1}}
    m = br.oldrules_reproduction(events, actual)
    assert m["slot_exact_rate"] < 1.0
    # recipients overlap (same 3 players got bonus) even when ranks differ
    assert m["recipient_jaccard"] == 1.0


def test_scoring_slot_rate_ignores_the_trivially_correct_zeros():
    """2026-08-18 (engine review §6). ``slot_exact_rate`` is dominated by
    players who got no bonus and were correctly predicted to get none — about
    nine in ten slots in a real match. On the live 2025-26 recompute it reads
    0.889 while the same comparison restricted to slots that ACTUALLY scored
    reads 0.342, and the headline number was being cited as evidence the
    simulator ranks players correctly.

    Here four of five slots are non-scoring and predicted right; the one
    scoring slot is wrong. The headline rate looks healthy, the honest one is
    zero.
    """
    events = {
        "g1": {
            1: _FWD_BRACE, 2: _MID_GA, 3: _DEF_CS,
            4: {"position": "MID", "minutes": 5},
            5: {"position": "DEF", "minutes": 5},
        }
    }
    # Only player 3 scored, and the sim ranks the FWD brace top instead.
    actual = {"g1": {3: 3}}
    m = br.oldrules_reproduction(events, actual)

    assert m["n_scoring_slots"] == 1.0
    assert m["scoring_slot_exact_rate"] == 0.0
    assert m["slot_exact_rate"] > m["scoring_slot_exact_rate"]


# --- FBref pure mappers ------------------------------------------------------
def test_map_summary_row_available_fields_and_derivations():
    """Corrected 2026-08-18. This used to feed ``"Passes Att"``,
    ``"Passes Cmp%"`` and ``"Performance Blocks"`` -- columns FBref's
    match-summary table does not contain -- and assert they mapped. Inventing
    the input is how the defect survived: in production those names matched
    nothing, were silently omitted, and the ORM default of 0 stood in for data
    the codebase believed it was collecting.

    Every key below now comes from soccerdata's real parsed frame.
    """
    raw = {
        "min": 90, "Performance Gls": 1, "Performance Ast": 0,
        "Performance CrdY": 1, "Performance TklW": 3, "Performance Int": 2,
        "Take-Ons Succ": 2,
        "Performance Fls": 2, "Performance Off": 1, "Performance Crs": 4,
        "Performance Sh": 4, "Performance SoT": 1,          # → 3 off target
        "Performance PKatt": 1, "Performance PK": 0,        # → 1 missed
    }
    out = fbref.map_summary_row(raw)
    assert out["minutes"] == 90
    assert out["goals"] == 1
    assert out["tackles"] == 3
    assert out["dribbles"] == 2
    assert out["fouls"] == 2
    assert out["offsides"] == 1
    assert out["open_play_crosses"] == 4
    assert out["shots_off_target"] == 3
    assert out["penalties_missed"] == 1
    # Genuinely unavailable metrics are omitted (ORM default 0 applies), not
    # guessed. `clearances` and `blocks` come from WhoScored's event stream;
    # passing volume lives in FBref's separate "passing" stat type.
    assert "clearances" not in out
    assert "blocks" not in out
    assert "passes" not in out
    assert "big_chances_created" not in out


def test_map_keeper_row():
    out = fbref.map_keeper_row({"Shot Stopping Saves": 5, "Penalty Kicks PKsv": 1})
    assert out == {"saves": 5, "penalties_saved": 1}


def test_map_xg_row_extracts_only_what_fbref_published():
    out = fbref.map_xg_row({
        "Expected xG": 0.42, "Expected npxG": 0.31, "Expected xAG": 0.18,
        "Performance Sh": 3,
    })
    assert out == {"xg": 0.42, "npxg": 0.31, "xa": 0.18, "shots": 3}
    # A missing column is OMITTED, never defaulted to 0 (2026-09-02): these
    # rows are upserted with set_=fields, so a 0 here overwrites the real
    # value Understat stored. FBref publishes no Expected columns at all for
    # 2026-27, which zeroed the season's xG down to 19 non-zero rows.
    assert fbref.map_xg_row({"Expected xG": 0.5}) == {"xg": 0.5}
    assert fbref.map_xg_row({"Performance Sh": 4}) == {"shots": 4}
    assert fbref.map_xg_row({}) == {}


def test_aggregate_xg_rows_sums_dgw():
    # player 1 plays two matches in GW5 (DGW) → summed into one player-GW row
    per_match = [
        (1, 5, {"xg": 0.4, "xa": 0.1, "npxg": 0.3, "shots": 2}),
        (1, 5, {"xg": 0.2, "xa": 0.3, "npxg": 0.2, "shots": 1}),
        (2, 5, {"xg": 0.9, "xa": 0.0, "npxg": 0.9, "shots": 4}),
    ]
    agg = fbref.aggregate_xg_rows(per_match)
    assert agg[(1, 5)]["xg"] == pytest.approx(0.6)
    assert agg[(1, 5)]["shots"] == 3
    assert agg[(1, 5)]["xgi"] == pytest.approx(0.6 + 0.4)   # xg + xa
    assert agg[(2, 5)]["npxg"] == pytest.approx(0.9)


def test_aggregate_xg_rows_drops_fields_no_source_supplied():
    """A shots-only source must not write xg/xa/npxg/xgi at all.

    This is the 2026-27 FBref shape: shots present, no Expected columns.
    Emitting zeros for the rest overwrote Understat's real values, because
    _write_xg_rows sets exactly the keys the aggregate hands it.
    """
    agg = fbref.aggregate_xg_rows([(1, 5, {"shots": 2}), (1, 5, {"shots": 3})])
    assert agg[(1, 5)] == {"shots": 5}
    assert "xg" not in agg[(1, 5)]
    assert "xgi" not in agg[(1, 5)]


def test_normalize_position():
    assert fbref.normalize_position("GK") == "GK"
    assert fbref.normalize_position("DF,MF") == "DEF"
    assert fbref.normalize_position("FW") == "FWD"
    assert fbref.normalize_position("MF") == "MID"
    assert fbref.normalize_position(None) == "MID"


def test_ingest_fbref_season_needs_optional_extra(monkeypatch):
    """Without soccerdata the adapter fails with an actionable message, not an
    opaque ImportError deep in the call stack."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "soccerdata":
            raise ImportError("no soccerdata")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError, match="soccerdata"):
        fbref.ingest_fbref_season("2025-26")
