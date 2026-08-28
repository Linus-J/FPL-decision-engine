"""The gate check that would have caught the dead live-season xG feed.

2026-08-28: ``player_xg_stats`` for 2026-27 held 309 rows, 147 of them with
``shots > 0``, and only TWO with a non-zero ``xg`` -- the "shots-only interim"
shape that data/ingestors/understat_xg.py was written to replace. Nothing
reported it. ``projection/assemble.py`` LEFT JOINs the table and COALESCEs a
miss to 0, so a shots-only row is indistinguishable from a genuine zero, and
the attacking signal was simply switched off for the live season.

``check_stat_column_not_dead`` already existed and was already wired to
``player_projections`` columns; no caller ever pointed it at ``player_xg_stats``.
``run_understat_coverage_check`` defaults to the PRIOR season and needs the
network, so it could not cover this either.

Eligible set is rows with ``shots > 0`` rather than all rows: a player who
took no shot has no xG to report, so including him would put a
roster-composition number in the denominator. Every player who DID shoot must
carry some xG. Healthy seasons sit at ~100% on this measure (2026-27 read
147/147 once the ingest was fixed); the broken state read 2/147.
"""

from __future__ import annotations

from sqlalchemy import text

from data.db import get_session
from data.models import Player, Team
from scripts.data_quality_gate import run_xg_liveness_check

_SEASON = "2099-00"


def _seed(rows: list[tuple[int, int, float]]) -> None:
    """(player_id, shots, xg) -> player_xg_stats rows for a throwaway season."""
    db = get_session()
    try:
        db.execute(
            text("DELETE FROM player_xg_stats WHERE season = :s"), {"s": _SEASON}
        )
        # players.team_id -> teams, and player_xg_stats.player_id -> players.
        if not db.execute(text("SELECT 1 FROM teams WHERE id = 1")).fetchone():
            db.add(Team(id=1, name="XG Test FC", short_name="XGT"))
            db.flush()
        # player_xg_stats.player_id is a real FK.
        existing = {
            r[0] for r in db.execute(text("SELECT id FROM players")).fetchall()
        }
        for player_id, _shots, _xg in rows:
            if player_id in existing:
                continue
            db.add(Player(
                id=player_id, fpl_id=player_id, code=player_id,
                first_name="xg", second_name=f"test{player_id}",
                web_name=f"xg-test-{player_id}", team_id=1, position="MID",
                now_cost=5.0, cost_change_start=0.0, status="a", news="",
            ))
            existing.add(player_id)
        db.flush()
        for i, (player_id, shots, xg) in enumerate(rows):
            db.execute(
                text(
                    "INSERT INTO player_xg_stats "
                    "(player_id, gameweek, season, xg, xa, xgi, npxg, shots, key_passes) "
                    "VALUES (:p, :gw, :s, :xg, 0, 0, :xg, :shots, 0)"
                ),
                {"p": player_id, "gw": 1 + i % 3, "s": _SEASON, "xg": xg, "shots": shots},
            )
        db.commit()
    finally:
        db.close()


def test_flags_a_shots_only_feed():
    """The real broken shape: shots present, xg absent."""
    _seed([(i, 2, 0.0) for i in range(1, 50)] + [(50, 2, 0.4)])
    issues = run_xg_liveness_check(_SEASON)
    assert len(issues) == 1
    assert "player_xg_stats.xg" in issues[0].message
    assert issues[0].severity == "error"


def test_passes_a_healthy_feed():
    """Every shot-taker carries xG."""
    _seed([(i, 2, 0.3) for i in range(1, 50)])
    assert run_xg_liveness_check(_SEASON) == []


def test_players_who_took_no_shots_are_not_in_the_denominator():
    """A squad full of non-shooters must not read as a dead column."""
    _seed([(i, 0, 0.0) for i in range(1, 50)] + [(50, 3, 0.5)])
    assert run_xg_liveness_check(_SEASON) == []


def test_a_season_with_no_rows_is_not_flagged():
    """Pre-season, and any season we simply have not ingested."""
    _seed([])
    assert run_xg_liveness_check(_SEASON) == []
