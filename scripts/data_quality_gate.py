#!/usr/bin/env python
"""data_quality_gate.py — run the reusable checks in data/quality_checks.py
against live data. Meant to be run periodically (e.g. alongside the normal
ingestion cadence) so the bug classes found in the 2026-07-28
data-completeness audit get caught automatically instead of surfacing later
as a captaincy-monopoly or a suspiciously-low backtest number.

    DB_PATH=fpl_bot_v2.db uv run --extra events python scripts/data_quality_gate.py

Exits 1 if any check reports an "error"-severity issue, 0 otherwise
(warnings are printed but don't fail the run).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def run_team_id_freshness_check() -> list:
    """Any player still in today's live FPL feed whose stored team_id
    disagrees with it right now (the Penders/Anselmino/Garnacho/Targett
    staleness found during the audit -- fixed by re-running the normal
    ingestion pipeline, not a code change)."""
    import httpx

    from data.db import get_session
    from data.models import Player
    from data.quality_checks import check_team_id_matches_live

    live = httpx.get(FPL_BOOTSTRAP_URL, timeout=15).json()
    live_player_team = {e["code"]: e["team"] for e in live["elements"]}

    db = get_session()
    try:
        rows = (
            db.query(Player.code, Player.web_name, Player.team_id)
            .filter(Player.code.isnot(None))
            .all()
        )
    finally:
        db.close()

    player_team_ids = {code: (web, tid) for code, web, tid in rows}
    return check_team_id_matches_live(player_team_ids, live_player_team)


def run_understat_coverage_check(season: str = "2025-26") -> list:
    """Name-match coverage against Understat's per-match feed for `season`.
    Would have caught the season-wide xG gap (only 14/524 players had any
    nonzero xg) immediately instead of it surfacing later as a captaincy
    monopoly. Best-effort: skipped (with a warning) if soccerdata isn't
    installed or the network call fails."""
    from data.quality_checks import check_name_match_coverage

    try:
        import soccerdata as sd

        from data.ingestors.fbref import _build_name_map, _match_player
        from data.ingestors.understat_xg import SEASON_MAP
    except ImportError:
        logger.warning("soccerdata not installed -- skipping Understat coverage check")
        return []

    yr = SEASON_MAP.get(season)
    if not yr:
        logger.warning("No Understat season mapping for %r -- skipping", season)
        return []

    try:
        name_map = _build_name_map()
        us = sd.Understat(leagues="ENG-Premier League", seasons=yr)
        names = us.read_player_match_stats().reset_index()["player"].unique()
    except Exception as exc:  # pragma: no cover - live network
        logger.warning("Understat coverage check failed to fetch live data: %s", exc)
        return []

    matched = sum(1 for n in names if _match_player(str(n), name_map) is not None)
    return check_name_match_coverage(f"understat/{season}", matched, len(names))



# --- Checks added 2026-08-16 ------------------------------------------------
#
# The suite proves the code does what it was written to do. These ask a
# different question: is the DATA the code is running on actually there, and
# are the numbers it produces plausible? Two of data/quality_checks.py's four
# checks had no caller at all -- including check_stat_column_not_dead, which
# is precisely the check that would have flagged cs_probability being 0.0 for
# every row ever written.


def run_source_coverage_checks(season: str = "2025-26") -> list:
    """How much of the football we model does each source actually see?

    Weighted by MINUTES, not player count: a source missing thirty fringe
    players is noise; one missing three everpresents is a hole in every
    projection they appear in.
    """
    from sqlalchemy import text

    from data.db import get_session
    from data.quality_checks import check_source_coverage

    issues = []
    db = get_session()
    try:
        total = db.execute(
            text(
                "SELECT COALESCE(SUM(minutes), 0) FROM player_gw_stats "
                "WHERE season = :s"
            ),
            {"s": season},
        ).scalar() or 0
        for label, table in (
            ("understat xg", "player_xg_stats"),
            ("match events", "player_match_events"),
            ("recomputed bonus", "recomputed_bonus"),
        ):
            covered = db.execute(
                text(
                    f"SELECT COALESCE(SUM(s.minutes), 0) FROM player_gw_stats s "
                    f"WHERE s.season = :s AND EXISTS ("
                    f"  SELECT 1 FROM {table} x "
                    f"  WHERE x.player_id = s.player_id AND x.season = s.season)"
                ),
                {"s": season},
            ).scalar() or 0
            issues += check_source_coverage(
                f"{label}/{season}", float(covered), float(total)
            )
    finally:
        db.close()
    return issues


def run_dead_column_checks(season: str = "2026-27") -> list:
    """Columns that are supposed to carry real per-player data.

    Catches the FBref dead-mapping bug class: a column that exists, is
    written on every row, and is always the same default -- which reads as
    "no signal" downstream rather than as an error.
    """
    from sqlalchemy import text

    from data.db import get_session
    from data.quality_checks import check_stat_column_not_dead
    from projection.pipeline import season_has_played_history

    # Cold-start projections legitimately carry defaults for anything that
    # needs a fixture lambda: projection/cold_start.py works from prior-season
    # points per appearance and never computes one, so cs_probability is
    # genuinely absent pre-season rather than broken. Checking it then would
    # fail the gate every week of pre-season for a correct state — the same
    # cry-wolf failure as running the gate before the ingest.
    lambda_derived = {"cs_probability"}
    in_season = season_has_played_history(season)

    issues = []
    db = get_session()
    try:
        for column in ("xpts", "xpts_var", "start_probability", "cs_probability"):
            if column in lambda_derived and not in_season:
                continue
            total = db.execute(
                text("SELECT COUNT(*) FROM player_projections WHERE gameweek IS NOT NULL")
            ).scalar() or 0
            nonzero = db.execute(
                text(
                    f"SELECT COUNT(*) FROM player_projections WHERE {column} != 0"
                )
            ).scalar() or 0
            issues += check_stat_column_not_dead(
                f"player_projections.{column}", int(nonzero), int(total)
            )
    finally:
        db.close()
    return issues


def run_copied_column_checks() -> list:
    """Columns that should differ from each other but do not.

    npxg was a verbatim copy of xg for all 11,306 rows: real, plausible,
    non-zero values that happened to be the same numbers. Nothing flagged it
    because nothing was empty or out of range — it only surfaced when a
    decomposition built on the pair (non-penalty xG + penalty duty) turned
    out to be double-counting.
    """
    from sqlalchemy import text

    from data.db import get_session
    from data.quality_checks import check_column_is_not_a_copy

    db = get_session()
    try:
        total = db.execute(text("SELECT COUNT(*) FROM player_xg_stats")).scalar() or 0
        differing = db.execute(
            text("SELECT COUNT(*) FROM player_xg_stats WHERE npxg < xg - 1e-9")
        ).scalar() or 0
    finally:
        db.close()
    return check_column_is_not_a_copy(
        "player_xg_stats.npxg vs xg", int(differing), int(total)
    )


def run_referential_integrity_checks() -> list:
    """Rows pointing at a player that does not exist. An orphan is silent --
    it simply never joins, so that player's data vanishes from projections
    rather than raising."""
    from sqlalchemy import text

    from data.db import get_session
    from data.quality_checks import check_referential_integrity

    issues = []
    db = get_session()
    try:
        for table in (
            "player_gw_stats", "player_xg_stats", "player_match_events",
            "player_projections", "player_setpiece_roles", "recomputed_bonus",
        ):
            total = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
            orphans = db.execute(
                text(
                    f"SELECT COUNT(*) FROM {table} t "
                    f"WHERE NOT EXISTS (SELECT 1 FROM players p WHERE p.id = t.player_id)"
                )
            ).scalar() or 0
            issues += check_referential_integrity(table, int(orphans), int(total))
    finally:
        db.close()
    return issues


def run_projection_sanity_checks() -> list:
    """Are the numbers plausible at all?

    Unit tests prove the arithmetic matches what was written; they cannot
    say the answer is sane. Every real defect this project has had showed up
    first as a number outside its plausible range.
    """
    from sqlalchemy import text

    from data.db import get_session
    from data.quality_checks import check_projection_sanity

    db = get_session()
    try:
        rows = db.execute(
            text(
                "SELECT xpts FROM player_projections WHERE gameweek = "
                "(SELECT MIN(gameweek) FROM player_projections)"
            )
        ).fetchall()
    finally:
        db.close()
    if not rows:
        return []  # nothing projected yet is a legitimate pre-season state
    values = [float(r[0]) for r in rows]
    # A per-player, per-gameweek expectation across the WHOLE pool, most of
    # whom are fringe. Sub-0.5 means a scale error (per-appearance mistaken
    # for per-gameweek, or availability applied twice); above 4 means the
    # pool average is scoring like a premium, which no pool does.
    return check_projection_sanity(
        "player_projections.xpts (pool mean)", values, low=0.5, high=4.0
    )


def run_odds_feature_liveness_check(season: str = "2026-27") -> list:
    """Do the odds features carry real values in the season being PLAYED?

    They are the only model inputs whose two sources differ by season:
    ``historical_fixture_odds`` is a backfill of finished seasons, so it can
    only ever end before the current one, while ``fixture_odds`` is written
    live. When the live leg is missing, ``load_fixture_odds`` still returns a
    full frame -- every row on its COALESCE default. The model then applies
    coefficients fitted on five seasons of real variation to three constants,
    and nothing anywhere raises, because a defaulted column is indistinguishable
    from a populated one at every point downstream.

    That is not hypothetical: it was the live state on 2026-08-17, with the
    backfill ending at 2025-26 and the bot about to play 2026-27.
    """
    from data.quality_checks import QualityIssue
    from projection.features import ODDS_FEATURE_COLS, load_fixture_odds
    from projection.pipeline import season_has_played_history

    # Pre-season there are no player-gameweek rows to carry odds at all; the
    # cold start does not read these features. Checking then would fail the
    # gate every week for a correct state.
    if not season_has_played_history(season):
        return []

    df = load_fixture_odds(season)
    if df.empty:
        return []

    issues = []
    for column in ODDS_FEATURE_COLS:
        if df[column].nunique() > 1:
            continue
        issues.append(
            QualityIssue(
                check="odds_feature_liveness",
                severity="error",
                message=(
                    f"{column} is the same value ({df[column].iloc[0]}) on all "
                    f"{len(df)} {season} rows — the live fixture_odds leg is not "
                    f"reaching the minutes model, so this feature is inert for "
                    f"the whole season while the model was fitted on real "
                    f"variation. Check that scripts/run_weekly.py is ingesting "
                    f"odds and that fixtures resolve to the right team pair."
                ),
            )
        )
    return issues


def main() -> int:
    issues = []
    issues += run_team_id_freshness_check()
    issues += run_understat_coverage_check()
    issues += run_source_coverage_checks()
    issues += run_dead_column_checks()
    issues += run_copied_column_checks()
    issues += run_referential_integrity_checks()
    issues += run_projection_sanity_checks()
    issues += run_odds_feature_liveness_check()

    if not issues:
        logger.info("data quality gate: all checks passed")
        return 0

    has_error = False
    for issue in issues:
        level = logging.ERROR if issue.severity == "error" else logging.WARNING
        logger.log(level, "[%s] %s", issue.check, issue.message)
        has_error = has_error or issue.severity == "error"

    logger.info(
        "data quality gate: %d issue(s) found (%s)",
        len(issues),
        "FAILING" if has_error else "warnings only",
    )
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
