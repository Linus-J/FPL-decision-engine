"""scripts/backfill_decision_outcomes.py

Covers: a normal finished GW gets actual_outcome written (captain doubled,
bench excluded); a genuine double-gameweek player's two player_gw_stats rows
are summed; an unfinished GW and a row that already has actual_outcome are
both left untouched; a bench_boost GW includes bench points."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import scripts.backfill_decision_outcomes as backfill_module
from data.models import Base, DecisionLog, Gameweek, PlayerGameweekStats, SimDecisionLog, SimManager


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'backfill.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(backfill_module, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _gw(session, gw_id: int, finished: bool, data_checked: bool | None = None) -> None:
    """``data_checked`` defaults to ``finished`` — a settled gameweek is the
    normal case for these tests. It is separable because scoring requires BOTH
    (engine review §8): ``finished`` means the last ball was kicked, while
    bonus and DefCon stay provisional until FPL marks the data checked."""
    session.add(Gameweek(
        id=gw_id, season="2026-27", name=f"GW{gw_id}",
        deadline_time=datetime(2026, 9, 1), finished=finished,
        data_checked=finished if data_checked is None else data_checked,
    ))
    session.commit()


def _stats(session, player_id: int, gw: int, points: int, opponent: int = 1) -> None:
    session.add(PlayerGameweekStats(
        player_id=player_id, gameweek=gw, season="2026-27",
        total_points=points, opponent_team_id=opponent,
    ))
    session.commit()


def _lineup(session, gw: int, squad_ids, starting_ids, captain_id, vice_captain_id=None) -> int:
    entry = DecisionLog(
        gameweek=gw, decision_type="lineup",
        details=json.dumps({
            "squad_ids": squad_ids, "starting_ids": starting_ids,
            "captain_id": captain_id, "vice_captain_id": vice_captain_id,
        }),
        projected_gain=0.0, dry_run=True,
    )
    session.add(entry)
    session.commit()
    return entry.id


def test_unfinished_gw_is_skipped(session):
    _gw(session, 10, finished=False)
    _lineup(session, 10, [1, 2], [1], captain_id=1)
    n = backfill_module.backfill("2026-27")
    assert n == 0
    row = session.query(DecisionLog).one()
    assert row.actual_outcome is None


def test_played_but_unchecked_gw_is_skipped(session):
    """Regression, 2026-08-18 (engine review §8).

    ``finished`` only means the last ball was kicked. Bonus points and
    defensive contributions are still provisional after it, and 26/27 moved
    the gameweek lockdown from ~1 hour after the final whistle to 09:00 the
    NEXT DAY — so the provisional window is now twelve hours or more.

    ``run_weekly.py`` scores last gameweek before deciding this one, so a run
    inside that window used to write provisional points into ``decision_log``
    and ``sim_decision_log``: the calibration instrument and the persona
    ranking. The scorer only ever revisits UNSCORED rows, so those numbers
    would have been wrong permanently.
    """
    _gw(session, 10, finished=True, data_checked=False)
    _stats(session, 1, 10, points=10)
    _lineup(session, 10, [1, 2], [1], captain_id=1)

    assert backfill_module.backfill("2026-27") == 0
    assert session.query(DecisionLog).one().actual_outcome is None


def test_finished_gw_computes_captain_doubled_starting_only(session):
    _gw(session, 11, finished=True)
    _stats(session, 1, 11, points=10)  # captain -> doubled
    _stats(session, 2, 11, points=4)   # starting
    _stats(session, 3, 11, points=99)  # bench, should not count
    _lineup(session, 11, squad_ids=[1, 2, 3], starting_ids=[1, 2], captain_id=1)

    n = backfill_module.backfill("2026-27")

    assert n == 1
    row = session.query(DecisionLog).one()
    assert row.actual_outcome == 24  # 10*2 + 4


def test_dgw_player_stats_rows_are_summed(session):
    _gw(session, 12, finished=True)
    _stats(session, 1, 12, points=5, opponent=1)
    _stats(session, 1, 12, points=8, opponent=2)  # same player, 2nd fixture (DGW)
    _lineup(session, 12, squad_ids=[1], starting_ids=[1], captain_id=1)

    backfill_module.backfill("2026-27")

    row = session.query(DecisionLog).one()
    assert row.actual_outcome == 26  # (5 + 8) * 2 (captain)


def test_already_backfilled_row_is_left_untouched(session):
    _gw(session, 13, finished=True)
    _stats(session, 1, 13, points=10)
    entry_id = _lineup(session, 13, squad_ids=[1], starting_ids=[1], captain_id=1)
    session.query(DecisionLog).filter_by(id=entry_id).update({"actual_outcome": 999})
    session.commit()

    n = backfill_module.backfill("2026-27")

    assert n == 0
    row = session.query(DecisionLog).one()
    assert row.actual_outcome == 999


def test_bench_boost_includes_bench_points(session):
    _gw(session, 14, finished=True)
    _stats(session, 1, 14, points=6)
    _stats(session, 2, 14, points=3)  # bench, counted because of bench boost
    session.add(DecisionLog(
        gameweek=14, decision_type="chip",
        details=json.dumps({"chip": "bboost", "reason": "test"}),
        projected_gain=0.0, dry_run=True,
    ))
    session.commit()
    _lineup(session, 14, squad_ids=[1, 2], starting_ids=[1], captain_id=1)

    backfill_module.backfill("2026-27")

    row = session.query(DecisionLog).filter_by(decision_type="lineup").one()
    assert row.actual_outcome == 15  # 6*2 (captain) + 3 (bench, boosted)


def _add_sim_manager(session, manager_id: int) -> None:
    session.add(SimManager(
        id=manager_id, season="2026-27", label=f"sim-{manager_id}",
        risk_level=0.0, max_ownership_differential=0.5,
        chip_aggressiveness=1.0,
    ))
    session.commit()


def _sim_lineup(session, sim_manager_id: int, gw: int, squad_ids, starting_ids, captain_id) -> int:
    entry = SimDecisionLog(
        sim_manager_id=sim_manager_id, gameweek=gw, decision_type="lineup",
        details=json.dumps({
            "squad_ids": squad_ids, "starting_ids": starting_ids, "captain_id": captain_id,
        }),
        projected_gain=0.0,
    )
    session.add(entry)
    session.commit()
    return entry.id


def test_sim_decision_log_gets_backfilled(session):
    _gw(session, 15, finished=True)
    _add_sim_manager(session, 1)
    _stats(session, 1, 15, points=7)
    _stats(session, 2, 15, points=2)
    _sim_lineup(session, 1, 15, squad_ids=[1, 2], starting_ids=[1, 2], captain_id=1)

    n = backfill_module.backfill("2026-27")

    assert n == 1
    row = session.query(SimDecisionLog).one()
    assert row.actual_outcome == 16  # 7*2 (captain) + 2


def test_sim_decision_log_chip_history_is_isolated_per_manager(session):
    """Manager 1 played bench boost this GW; manager 2 did not -- manager
    2's bench points must NOT be counted even though both share a
    gameweek."""
    _gw(session, 16, finished=True)
    _add_sim_manager(session, 1)
    _add_sim_manager(session, 2)
    _stats(session, 1, 16, points=5)
    _stats(session, 2, 16, points=9)  # bench for both managers

    session.add(SimDecisionLog(
        sim_manager_id=1, gameweek=16, decision_type="chip",
        details=json.dumps({"chip": "bboost", "reason": "test"}), projected_gain=0.0,
    ))
    session.commit()

    _sim_lineup(session, 1, 16, squad_ids=[1, 2], starting_ids=[1], captain_id=1)
    _sim_lineup(session, 2, 16, squad_ids=[1, 2], starting_ids=[1], captain_id=1)

    backfill_module.backfill("2026-27")

    manager1_row = session.query(SimDecisionLog).filter_by(
        sim_manager_id=1, decision_type="lineup"
    ).one()
    manager2_row = session.query(SimDecisionLog).filter_by(
        sim_manager_id=2, decision_type="lineup"
    ).one()
    assert manager1_row.actual_outcome == 19  # 5*2 (captain) + 9 (bench, boosted)
    assert manager2_row.actual_outcome == 10  # 5*2 (captain) only -- no boost


def test_backfill_covers_both_real_and_sim_logs_in_one_call(session):
    _gw(session, 17, finished=True)
    _add_sim_manager(session, 1)
    _stats(session, 1, 17, points=4)
    _lineup(session, 17, squad_ids=[1], starting_ids=[1], captain_id=1)
    _sim_lineup(session, 1, 17, squad_ids=[1], starting_ids=[1], captain_id=1)

    n = backfill_module.backfill("2026-27")

    assert n == 2
    assert session.query(DecisionLog).filter_by(decision_type="lineup").one().actual_outcome == 8
    assert session.query(SimDecisionLog).one().actual_outcome == 8


# --- P2.1 / P2.2 (2026-08-16): outcomes must be what a manager really scored


def _stats_with_minutes(
    session, player_id: int, gw: int, points: int, minutes: int, opponent: int = 1
) -> None:
    session.add(PlayerGameweekStats(
        player_id=player_id, gameweek=gw, season="2026-27",
        total_points=points, minutes=minutes, opponent_team_id=opponent,
    ))
    session.commit()


def _full_squad_details(**overrides) -> dict:
    """A legal 15 with a 1-4-4-2 XI: 1 GKP, 4 DEF, 4 MID, 2 FWD starting;
    bench is a GK plus one of each outfield position, so an autosub is
    always formation-legal."""
    positions = {
        1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF", 5: "DEF",
        6: "MID", 7: "MID", 8: "MID", 9: "MID", 10: "FWD", 11: "FWD",
        12: "GKP", 13: "DEF", 14: "MID", 15: "FWD",
    }
    details = {
        "squad_ids": list(range(1, 16)),
        "starting_ids": list(range(1, 12)),
        "captain_id": 10,
        "vice_captain_id": 11,
        "positions": {str(k): v for k, v in positions.items()},
        "bench_order": {"12": 0, "13": 1, "14": 2, "15": 3},
    }
    details.update(overrides)
    return details


def test_autosub_replaces_a_blanking_starter(session):
    """The defect: `_score_squad` supports autosubs but needs minutes,
    positions AND bench_order together, and the lineup decision recorded
    none of the latter two -- so a starter who didn't play scored 0 with no
    substitute, understating every persona exactly where the bench matters."""
    _gw(session, 5, finished=True)
    for pid in range(1, 12):
        # player 9 (a starting MID) blanks; everyone else plays and scores 2
        _stats_with_minutes(session, pid, 5, points=0 if pid == 9 else 2,
                            minutes=0 if pid == 9 else 90)
    _stats_with_minutes(session, 12, 5, points=5, minutes=90)   # bench GK
    _stats_with_minutes(session, 13, 5, points=7, minutes=90)   # bench DEF, first up
    _stats_with_minutes(session, 14, 5, points=9, minutes=90)
    _stats_with_minutes(session, 15, 5, points=9, minutes=90)

    session.add(DecisionLog(
        gameweek=5, decision_type="lineup",
        details=json.dumps(_full_squad_details()), projected_gain=0.0, dry_run=True,
    ))
    session.commit()

    assert backfill_module.backfill("2026-27") == 1
    outcome = session.query(DecisionLog).one().actual_outcome
    # 10 players scoring 2 = 20, captain (player 10) doubled = +2, and the
    # blanking MID is replaced by the first bench player who played (13, 7pts)
    assert outcome == 20 + 2 + 7


def test_vice_captain_is_promoted_when_the_captain_blanks(session):
    _gw(session, 5, finished=True)
    for pid in range(1, 12):
        _stats_with_minutes(session, pid, 5, points=0 if pid == 10 else 2,
                            minutes=0 if pid == 10 else 90)
    for pid in (12, 13, 14, 15):
        _stats_with_minutes(session, pid, 5, points=1, minutes=90)

    session.add(DecisionLog(
        gameweek=5, decision_type="lineup",
        details=json.dumps(_full_squad_details()), projected_gain=0.0, dry_run=True,
    ))
    session.commit()

    backfill_module.backfill("2026-27")
    outcome = session.query(DecisionLog).one().actual_outcome
    # ten starters played and scored 2 each (20); the blanking captain (10,
    # a FWD) is replaced by bench DEF 13 for 1 more; and the armband passes
    # to the vice (11, 2pts), whose points are then doubled (+2).
    assert outcome == 20 + 1 + 2


def test_hits_are_deducted_from_the_recorded_outcome(session):
    """Hits are booked on the separate `transfers` decision, so the lineup
    row recorded GROSS points -- but actual_outcome is what the season
    analysis compares personas on, so it has to be net."""
    _gw(session, 5, finished=True)
    for pid in range(1, 16):
        _stats_with_minutes(session, pid, 5, points=2, minutes=90)

    session.add(DecisionLog(
        gameweek=5, decision_type="lineup",
        details=json.dumps(_full_squad_details(hits_taken=2)),
        projected_gain=0.0, dry_run=True,
    ))
    session.commit()

    backfill_module.backfill("2026-27")
    outcome = session.query(DecisionLog).one().actual_outcome
    assert outcome == 22 + 2 - 8  # 11 starters@2 + captain bonus, minus two 4pt hits


def test_decisions_recorded_before_p2_1_still_score_without_autosubs(session, caplog):
    """Older rows carry no positions/bench_order. They must keep their
    existing basis and say so, not be silently rescored."""
    _gw(session, 5, finished=True)
    for pid in range(1, 12):
        _stats_with_minutes(session, pid, 5, points=0 if pid == 9 else 2,
                            minutes=0 if pid == 9 else 90)
    _stats_with_minutes(session, 13, 5, points=7, minutes=90)

    legacy = _full_squad_details()
    del legacy["positions"]
    del legacy["bench_order"]
    session.add(DecisionLog(
        gameweek=5, decision_type="lineup",
        details=json.dumps(legacy), projected_gain=0.0, dry_run=True,
    ))
    session.commit()

    with caplog.at_level("WARNING"):
        backfill_module.backfill("2026-27")
    outcome = session.query(DecisionLog).one().actual_outcome
    assert outcome == 20 + 2, "no substitute applied"
    assert "WITHOUT auto-substitutions" in caplog.text


# --- re-runs must not multiply a gameweek (2026-08-17) ----------------------
def test_only_the_last_lineup_of_a_gameweek_is_scored(session):
    """Re-running a gameweek APPENDS a lineup row -- every read elsewhere
    takes the latest, so that is correct storage. The scorer, though, picked up
    every unscored row: seven re-runs of GW1 left 8 rows in decision_log and
    689 in sim_decision_log for 90 real (persona, gameweek) pairs. Those would
    have counted as independent observations in the season analysis, skewing
    the persona ranking and the calibration sample with decisions that were
    never live.
    """
    _gw(session, 12, finished=True)
    _stats(session, 1, 12, points=10)
    _stats(session, 2, 12, points=4)
    _stats(session, 3, 12, points=7)

    superseded = _lineup(session, 12, [1, 2, 3], [1, 2], captain_id=1)
    stood = _lineup(session, 12, [1, 2, 3], [1, 3], captain_id=3)

    assert backfill_module.backfill("2026-27") == 1, "one gameweek, one score"

    rows = {r.id: r.actual_outcome for r in session.query(DecisionLog).all()}
    assert rows[superseded] is None, "a superseded decision was never live"
    assert rows[stood] == 10 + 7 + 7, "the decision that stood is the one scored"


def test_each_persona_still_gets_its_own_gameweek_scored(session):
    """The de-duplication is per (persona, gameweek). Taking the latest row
    globally would score one persona out of ninety."""
    _gw(session, 13, finished=True)
    _stats(session, 1, 13, points=6)
    for sim_id in (1, 2):
        session.add(SimManager(
            id=sim_id, season="2026-27", label=f"p{sim_id}",
            risk_level=0.0, max_ownership_differential=0.5, chip_aggressiveness=1.0,
        ))
    session.commit()
    for sim_id in (1, 2):
        for _ in range(3):            # three re-runs each
            session.add(SimDecisionLog(
                sim_manager_id=sim_id, gameweek=13, decision_type="lineup",
                details=json.dumps({"squad_ids": [1], "starting_ids": [1],
                                    "captain_id": 1, "vice_captain_id": None}),
                projected_gain=0.0,
            ))
        session.commit()

    backfill_module.backfill("2026-27")

    scored = [r for r in session.query(SimDecisionLog).all()
              if r.actual_outcome is not None]
    assert len(scored) == 2, "exactly one scored row per persona"
    assert {r.sim_manager_id for r in scored} == {1, 2}


def test_unscoped_sim_scoring_is_refused_rather_than_scoring_one_persona(session):
    """The subquery resolves MAX(created_at) within its scope, so an unscoped
    call would silently collapse ninety personas to one."""
    with pytest.raises(ValueError, match="one persona at a time"):
        backfill_module._backfill_table(session, "2026-27", "sim_decision_log")
