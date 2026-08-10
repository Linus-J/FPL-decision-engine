"""overrides.py — manual transfer/rumour corrections (plan: cold-start
fixture lookahead + transfer overrides, 2026-08-10). Every loader must
degrade safely (empty/no-op, warning log) on a missing file, empty file, or
unmatched code -- never crash the pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data import overrides as ov
from data.models import Base, Player


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'overrides.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(ov, "get_session", lambda: Local())
    return Local


@pytest.fixture
def overrides_file(tmp_path, monkeypatch):
    path = tmp_path / "transfer_overrides.yaml"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", path)
    return path


def _write(path, data: dict):
    path.write_text(yaml.safe_dump(data))


def test_load_team_overrides_missing_file_is_empty(overrides_file):
    assert ov.load_team_overrides() == {}


def test_load_team_overrides_reads_confirmed_list(overrides_file):
    _write(overrides_file, {"confirmed": [
        {"code": 111, "team_id": 5, "reason": "x", "as_of": "2026-08-10"},
    ]})
    assert ov.load_team_overrides() == {111: 5}


def test_apply_team_overrides_replaces_matched_code(overrides_file):
    _write(overrides_file, {"confirmed": [{"code": 111, "team_id": 5}]})
    players = pd.DataFrame([
        {"id": 1, "code": 111, "team_id": 1, "web_name": "Moved"},
        {"id": 2, "code": 222, "team_id": 2, "web_name": "Unaffected"},
    ])
    out = ov.apply_team_overrides(players)
    assert out.loc[out["code"] == 111, "team_id"].iloc[0] == 5
    assert out.loc[out["code"] == 222, "team_id"].iloc[0] == 2
    # original untouched (returns a copy)
    assert players.loc[players["code"] == 111, "team_id"].iloc[0] == 1


def test_apply_team_overrides_noop_when_file_missing(overrides_file):
    players = pd.DataFrame([{"id": 1, "code": 111, "team_id": 1}])
    out = ov.apply_team_overrides(players)
    pd.testing.assert_frame_equal(out, players)


def test_apply_team_overrides_noop_when_no_code_column(overrides_file):
    _write(overrides_file, {"confirmed": [{"code": 111, "team_id": 5}]})
    players = pd.DataFrame([{"id": 1, "team_id": 1}])
    out = ov.apply_team_overrides(players)
    pd.testing.assert_frame_equal(out, players)


def test_load_p_leave_overrides_resolves_code_to_player_id(temp_session, overrides_file):
    s = temp_session()
    try:
        s.add(Player(id=42, fpl_id=42, code=999, first_name="A", second_name="B",
                     web_name="Rumoured", team_id=1, position="MID", now_cost=8.0))
        s.commit()
    finally:
        s.close()
    _write(overrides_file, {"rumoured": [
        {"code": 999, "p_leave": 0.4, "reason": "linked", "as_of": "2026-08-10"},
    ]})
    assert ov.load_p_leave_overrides() == {42: 0.4}


def test_load_p_leave_overrides_skips_unmatched_code_without_crashing(
    temp_session, overrides_file, caplog
):
    temp_session()  # ensure players table exists, empty
    _write(overrides_file, {"rumoured": [{"code": 999, "p_leave": 0.4}]})
    with caplog.at_level(logging.WARNING):
        result = ov.load_p_leave_overrides()
    assert result == {}
    assert "999" in caplog.text


def test_load_rumoured_overrides_missing_file_is_empty(overrides_file):
    assert ov.load_rumoured_overrides() == {}


def test_log_rumoured_squad_members_logs_matched_squad_member(
    temp_session, overrides_file, caplog
):
    s = temp_session()
    try:
        s.add(Player(id=42, fpl_id=42, code=999, first_name="A", second_name="B",
                     web_name="Rumoured", team_id=1, position="MID", now_cost=8.0))
        s.commit()
    finally:
        s.close()
    _write(overrides_file, {"rumoured": [
        {"code": 999, "p_leave": 0.4, "reason": "linked", "as_of": "2026-08-10"},
    ]})
    players = pd.DataFrame([{"id": 42, "web_name": "Rumoured"}, {"id": 7, "web_name": "Other"}])
    with caplog.at_level(logging.WARNING):
        ov.log_rumoured_squad_members([42, 7], players)
    assert "Rumoured" in caplog.text
    assert "0.40" in caplog.text


def test_log_rumoured_squad_members_noop_when_no_rumours(overrides_file, caplog):
    players = pd.DataFrame([{"id": 1, "web_name": "X"}])
    with caplog.at_level(logging.WARNING):
        ov.log_rumoured_squad_members([1], players)
    assert caplog.text == ""


def test_apply_team_overrides_result_respected_by_max_players_per_club_constraint(overrides_file):
    """Integration proof (spec's explicit testing requirement): the
    override must be visible to optimise_squad's own max-3-per-club
    constraint, not just a cosmetic column change nothing downstream reads.

    Team 1 already has 3 members (id1 GKP, id2 DEF, id3 MID) filling the
    cap. Player code=999 (id4, a MID) starts on team 2, where nothing
    competes with them -- overriding their code to team_id=1 forces the
    solver to choose between id3 (score 10) and id4 (score 6) for team 1's
    now-contested 3rd slot, since keeping both would push team 1 to 4
    members. If the override were a no-op, id4 would stay attributed to
    team 2 and would certainly be picked (nothing there outscores them);
    the assertion that id4 is EXCLUDED only holds if the override actually
    changed what the ILP sees."""
    from config.strategy import SQUAD
    from optimiser.squad import optimise_squad

    _write(overrides_file, {"confirmed": [{"code": 999, "team_id": 1}]})

    rows = [
        {"id": 1, "code": 1, "position": "GKP", "now_cost": 4.5, "team_id": 1,
         "status": "a", "web_name": "gk1"},
        {"id": 2, "code": 2, "position": "DEF", "now_cost": 4.5, "team_id": 1,
         "status": "a", "web_name": "def1"},
        {"id": 3, "code": 3, "position": "MID", "now_cost": 4.5, "team_id": 1,
         "status": "a", "web_name": "mid1"},
        {"id": 4, "code": 999, "position": "MID", "now_cost": 4.5, "team_id": 2,
         "status": "a", "web_name": "target"},
    ]
    # Fill out a legal 16-candidate pool (one MORE than squad_size needs per
    # position, in MID specifically -- see below) across teams 3-9, cheap
    # and unremarkable so the solver has no reason to prefer them on merit
    # alone, only when a hard constraint forces a choice.
    fillers = ["GKP"] * 1 + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 3
    next_id = 5
    for i, pos in enumerate(fillers):
        rows.append({
            "id": next_id, "code": next_id, "position": pos, "now_cost": 4.5,
            "team_id": 3 + (i % 7), "status": "a", "web_name": f"filler{next_id}",
        })
        next_id += 1
    # Pool per position: GKP=2 (exact fit, both forced), DEF=5 (exact fit,
    # all forced), FWD=3 (exact fit, all forced) -- MID=6 candidates for 5
    # slots (id3=10, id4=6, four fillers=2 each) is the ONLY position with
    # slack, which is what lets the solver drop id4 instead of being forced
    # into an infeasible 4-player team 1.

    players = pd.DataFrame(rows)
    players = ov.apply_team_overrides(players)
    assert players.loc[players["code"] == 999, "team_id"].iloc[0] == 1  # override took effect
    players["start_probability"] = 1.0

    proj_rows = [
        {"player_id": r.id, "gameweek": 1,
         "xpts": 10.0 if r.id in (1, 2, 3) else (6.0 if r.id == 4 else 2.0)}
        for r in players.itertuples()
    ]
    projections = pd.DataFrame(proj_rows)

    solution = optimise_squad(projections=projections, players=players, budget=100.0, horizon=1)

    team_counts = solution.squad["team_id"].value_counts()
    assert team_counts.get(1, 0) <= SQUAD.max_players_per_club
    # target excluded by the (corrected) team cap
    assert 4 not in solution.squad["id"].tolist()
    # the higher-scoring incumbent wins the contested slot
    assert 3 in solution.squad["id"].tolist()
