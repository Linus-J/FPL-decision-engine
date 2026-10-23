"""Replay a past gameweek: rewind a copy of the database to its deadline.

The decision engine reads the database as it stands. A replay therefore needs a
database that stands where the deadline stood. ``rewind_database`` makes one:
it copies the live file and then, inside the copy, removes everything stamped
at or after the deadline and restores every mutable value to its as-of
snapshot. The engine then runs unchanged against the copy.

Two things are not recoverable and are left as the rewind finds them, and both
are named in the report the rewind returns:

- Season-cumulative counters on ``players`` (goals, minutes, points) are not
  rewound. The decision path does not read them, but a reader that does would
  see the end-of-season figure.
- Players with no snapshot before the deadline keep their current row. In
  practice this is only players whose first snapshot postdates the deadline.

The copy is never the live file. ``rewind_database`` refuses to write there.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

REPLAY_ENV = "FPL_REPLAY"

# Mutable player fields restored from player_state_snapshots. Mirrors the
# columns the snapshot model carries; injury_severity is recomputed from the
# rewound news by the injury parser, not from a snapshot.
_PLAYER_FIELDS = (
    "now_cost", "status", "chance_of_playing_this_round", "chance_of_playing_next_round",
    "selected_by_percent", "form", "ict_index", "influence", "creativity", "threat",
    "news", "news_added", "transfers_in_event", "transfers_out_event",
)

_TEAM_FIELDS = (
    "strength_overall_home", "strength_overall_away",
    "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
)

_SETPIECE_FIELDS = (
    "penalty_order", "freekick_order", "corner_order",
    "is_penalty_taker", "penalty_xg_per_game", "is_set_piece_taker", "source",
)


def replay_active() -> bool:
    """True inside a replay child process. Live code paths check this to avoid
    network calls that would return the present, not the past."""
    return os.environ.get(REPLAY_ENV) == "1"


def _deadline(con: sqlite3.Connection, season: str, gameweek: int) -> str:
    row = con.execute(
        "SELECT deadline_time FROM gameweeks WHERE season = ? AND id = ?",
        (season, gameweek),
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"no deadline for {season} GW{gameweek} in this database")
    return row[0]


def rewind_database(src: Path, dst: Path, season: str, gameweek: int) -> dict[str, int]:
    """Write ``dst`` as ``src`` stood at the deadline of ``season`` GW ``gameweek``.

    Returns the number of rows changed per step, for the report. Raises before
    writing anything if ``dst`` is the source or already exists.
    """
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst == src:
        raise ValueError("refusing to rewind the live database in place")
    if dst.exists():
        raise FileExistsError(f"{dst} exists; replays are written fresh")
    dst.parent.mkdir(parents=True, exist_ok=True)

    source = sqlite3.connect(src)
    try:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)  # online copy, consistent under WAL
        finally:
            target.close()
    finally:
        source.close()

    _upgrade_schema(dst)
    con = sqlite3.connect(dst)
    counts: dict[str, int] = {}
    try:
        D = _deadline(con, season, gameweek)
        N = gameweek
        counts.update(_coverage(con, season, D))
        with con:
            counts.update(_rewind_calendar(con, season, N))
            counts.update(_rewind_history(con, season, N, D))
            counts.update(_rewind_snapshots(con, D))
            counts.update(_rewind_decisions_and_projections(con, D))
            counts.update(_restore_players(con, season, D))
            counts.update(_restore_teams(con, season, D))
            counts.update(_restore_setpieces(con, season, D))
    finally:
        con.close()
    return counts


def _upgrade_schema(path: Path) -> None:
    """Bring a copy up to the current models: create missing tables and add missing
    nullable columns. The live database gets the same change from ``init_db`` on
    its next run; a replay must not depend on that having happened first.
    """
    from sqlalchemy import create_engine, inspect

    from data.models import Base

    engine = create_engine(f"sqlite:///{path}")
    try:
        Base.metadata.create_all(engine)
        insp = inspect(engine)
        with engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                present = {c["name"] for c in insp.get_columns(table.name)}
                for column in table.columns:
                    if column.name in present:
                        continue
                    if not column.nullable:
                        raise RuntimeError(
                            f"{table.name}.{column.name} is missing and NOT NULL; "
                            "it needs a real migration before a replay can run"
                        )
                    ctype = column.type.compile(dialect=engine.dialect)
                    conn.exec_driver_sql(
                        f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ctype}'
                    )
    finally:
        engine.dispose()


def _coverage(con, season: str, D: str) -> dict[str, int]:
    """1 when the season has snapshots stamped before the deadline, else 0.

    A rewind with no coverage cannot say what a value was, so it keeps the
    current one and says so in the report, rather than inventing a default.
    """
    def has(table: str) -> int:
        return int(con.execute(
            f"SELECT EXISTS(SELECT 1 FROM {table} WHERE season = ? AND snapshot_ts < ?)",
            (season, D),
        ).fetchone()[0])

    return {
        "coverage_player_state": has("player_state_snapshots"),
        "coverage_team_strength": has("team_strength_snapshots"),
        "coverage_setpieces": has("player_setpiece_snapshots"),
    }


def _changes(cur: sqlite3.Cursor) -> int:
    return cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0


def _rewind_calendar(con, season: str, N: int) -> dict[str, int]:
    # Other seasons' flags must not compete with this one's: the engine picks
    # "the current gameweek" by flag across the whole table.
    other = _changes(con.execute(
        "UPDATE gameweeks SET is_current = 0, is_next = 0 WHERE season != ?", (season,)
    ))
    # Mid-season a decision is made in the gap between finishing N-1 and N's
    # deadline: N-1 is current, N is next, and only gameweeks before N are played.
    own = _changes(con.execute(
        """UPDATE gameweeks SET
               is_current = (id = :n - 1),
               is_next = (id = :n),
               finished = (id < :n),
               data_checked = (id < :n)
           WHERE season = :season""",
        {"n": N, "season": season},
    ))
    fixtures = _changes(con.execute(
        """UPDATE fixtures SET finished = 0, team_h_score = NULL, team_a_score = NULL
           WHERE season = :season AND gameweek >= :n""",
        {"season": season, "n": N},
    ))
    return {"gameweeks_other_seasons": other, "gameweeks": own, "fixtures_unplayed": fixtures}


def _rewind_history(con, season: str, N: int, D: str) -> dict[str, int]:
    out = {}
    out["player_gw_stats"] = _changes(con.execute(
        "DELETE FROM player_gw_stats WHERE season = ? AND gameweek >= ?", (season, N)))
    for table in ("player_xg_stats", "player_match_events", "recomputed_bonus"):
        out[table] = _changes(con.execute(
            f"DELETE FROM {table} WHERE season = ? AND gameweek >= ?", (season, N)))
    # team_matches is the season's fixture schedule, which is public before the
    # deadline, so it is kept whole. Removing future kickoffs would leave the
    # congestion features with no anchor for the very gameweek being decided.
    for table in ("entry_gameweek", "chip_usage", "chip_usage_sync"):
        out[table] = _changes(con.execute(
            f"DELETE FROM {table} WHERE season = ? AND synced_at >= ?", (season, D)))
    return out


def _rewind_snapshots(con, D: str) -> dict[str, int]:
    out = {}
    for table, col in (
        ("player_state_snapshots", "snapshot_ts"),
        ("team_strength_snapshots", "snapshot_ts"),
        ("player_setpiece_snapshots", "snapshot_ts"),
        ("ownership_snapshots", "snapshot_ts"),
    ):
        out[table] = _changes(con.execute(
            f"DELETE FROM {table} WHERE {col} >= ?", (D,)))
    out["fixture_odds"] = _changes(con.execute(
        "DELETE FROM fixture_odds WHERE fetched_at >= ?", (D,)))
    return out


def _rewind_decisions_and_projections(con, D: str) -> dict[str, int]:
    out = {}
    for table in ("decision_log", "player_projections", "projection_samples",
                  "chip_comparison_log", "sim_decision_log"):
        out[table] = _changes(con.execute(
            f"DELETE FROM {table} WHERE created_at >= ?", (D,)))
    return out


def _restore_players(con, season: str, D: str) -> dict[str, int]:
    """Set each player's mutable fields to the latest snapshot before the deadline.

    Only players with such a snapshot are touched. The rest keep their current
    row, which the report counts.
    """
    has_snapshot = (
        "EXISTS (SELECT 1 FROM player_state_snapshots ps WHERE ps.player_id = players.id "
        "AND ps.season = :season AND ps.snapshot_ts < :deadline)"
    )
    assignments = ", ".join(
        f"{col} = (SELECT ps.{col} FROM player_state_snapshots ps "
        f"WHERE ps.player_id = players.id AND ps.season = :season "
        f"AND ps.snapshot_ts < :deadline ORDER BY ps.snapshot_ts DESC LIMIT 1)"
        for col in _PLAYER_FIELDS
    )
    restored = _changes(con.execute(
        f"UPDATE players SET {assignments} WHERE {has_snapshot}",
        {"season": season, "deadline": D},
    ))
    untouched = con.execute(
        "SELECT COUNT(*) FROM players WHERE NOT " + has_snapshot,
        {"season": season, "deadline": D},
    ).fetchone()[0]
    return {"players_restored": restored, "players_without_snapshot": untouched}


def _restore_teams(con, season: str, D: str) -> dict[str, int]:
    out = {}
    for table in ("teams", "team_season_strength"):
        assignments = ", ".join(
            f"{col} = (SELECT ts.{col} FROM team_strength_snapshots ts "
            f"WHERE ts.team_id = {table}.{'id' if table == 'teams' else 'team_id'} "
            f"AND ts.season = :season AND ts.snapshot_ts < :deadline "
            f"ORDER BY ts.snapshot_ts DESC LIMIT 1)"
            for col in _TEAM_FIELDS
        )
        team_key = "id" if table == "teams" else "team_id"
        scope = "" if table == "teams" else "season = :season AND "
        has_snapshot = (
            f"EXISTS (SELECT 1 FROM team_strength_snapshots ts "
            f"WHERE ts.team_id = {table}.{team_key} AND ts.season = :season "
            f"AND ts.snapshot_ts < :deadline)"
        )
        out[table] = _changes(con.execute(
            f"UPDATE {table} SET {assignments} WHERE {scope}{has_snapshot}",
            {"season": season, "deadline": D},
        ))
    return out


def _restore_setpieces(con, season: str, D: str) -> dict[str, int]:
    """Set duty on every roles row to the as-of snapshot, defaults where there is
    none. Key passes are not snapshotted and stay as they are.

    Only when the season has set-piece snapshots before the deadline. Without
    them, "no snapshot" would mean "no duty" for every player, which is a claim
    about the past that nothing supports, so the current table is kept instead.
    """
    covered = con.execute(
        "SELECT EXISTS(SELECT 1 FROM player_setpiece_snapshots "
        "WHERE season = ? AND snapshot_ts < ?)", (season, D),
    ).fetchone()[0]
    if not covered:
        return {"player_setpiece_roles": 0}
    latest = (
        "(SELECT sp.{col} FROM player_setpiece_snapshots sp "
        "WHERE sp.player_id = player_setpiece_roles.player_id AND sp.season = :season "
        "AND sp.snapshot_ts < :deadline ORDER BY sp.snapshot_ts DESC LIMIT 1)"
    )
    defaults = {
        "penalty_order": "NULL", "freekick_order": "NULL", "corner_order": "NULL",
        "is_penalty_taker": "0", "penalty_xg_per_game": "0.0",
        "is_set_piece_taker": "0", "source": "''",
    }
    assignments = ", ".join(
        f"{col} = COALESCE({latest.format(col=col)}, {defaults[col]})"
        for col in _SETPIECE_FIELDS
    )
    changed = _changes(con.execute(
        f"UPDATE player_setpiece_roles SET {assignments} WHERE season = :season",
        {"season": season, "deadline": D},
    ))
    return {"player_setpiece_roles": changed}


def verify_rewind(path: Path, season: str, gameweek: int) -> list[str]:
    """Violations of the rewind's promise, as human-readable strings.

    Empty means nothing stamped at or after the deadline is visible in ``path``.
    A replay must not run on a database that fails this.
    """
    con = sqlite3.connect(path)
    try:
        D = _deadline(con, season, gameweek)
        checks = {
            "player_gw_stats past the deadline": (
                "SELECT COUNT(*) FROM player_gw_stats WHERE season = ? AND gameweek >= ?",
                (season, gameweek)),
            "decision_log after the deadline": (
                "SELECT COUNT(*) FROM decision_log WHERE created_at >= ?", (D,)),
            "fixture_odds after the deadline": (
                "SELECT COUNT(*) FROM fixture_odds WHERE fetched_at >= ?", (D,)),
            "player snapshots after the deadline": (
                "SELECT COUNT(*) FROM player_state_snapshots WHERE snapshot_ts >= ?", (D,)),
            "team strength snapshots after the deadline": (
                "SELECT COUNT(*) FROM team_strength_snapshots WHERE snapshot_ts >= ?", (D,)),
            "set-piece snapshots after the deadline": (
                "SELECT COUNT(*) FROM player_setpiece_snapshots WHERE snapshot_ts >= ?", (D,)),
            "fixtures finished from the deadline on": (
                "SELECT COUNT(*) FROM fixtures WHERE season = ? AND gameweek >= ? "
                "AND finished = 1", (season, gameweek)),
            "a gameweek other than the decided one is next": (
                "SELECT COUNT(*) FROM gameweeks WHERE season = ? AND is_next = 1 "
                "AND id != ?", (season, gameweek)),
        }
        violations = []
        for label, (sql, params) in checks.items():
            count = con.execute(sql, params).fetchone()[0]
            if count:
                violations.append(f"{label}: {count}")
        return violations
    finally:
        con.close()
