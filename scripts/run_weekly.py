#!/usr/bin/env python
"""Manual weekly kickoff: refreshes match-event + ownership data for the
gameweek that just finished, THEN runs the real agent decision, THEN the
simulation batch -- in that order, so the decision is made against
up-to-date DefCon/bonus/ownership data, not whatever was last scraped.

FBref -> WhoScored -> set-pieces -> ownership ->
run_agent.py -> backfill_decision_outcomes.py -> data_quality_gate.py ->
export_site_data.py -> preflight.py -> run_simulations.py.

The outcome backfill scores LAST gameweek's decisions (for the real bot and
all 90 personas). It runs AFTER run_agent.py because the ingest it depends on
lives inside that script; nothing in the decision path reads actual_outcome,
so the ordering costs nothing. It runs against
complete match data.
FBref must run before WhoScored (WhoScored only PATCHES rows FBref's
ingest already created, never inserts new -- see scrape_whoscored.py).
Each data step also declares a POST-CONDITION -- the data it must leave
behind. The exit code cannot see an empty result (these steps exit 0 on one
by design), and twice on 2026-08-28 a step reported success having written
nothing. Failures are logged where they happen and re-reported together at
the end of the run.

Every step degrades gracefully (logs a warning, continues) except the
final two, whose own exit codes are reported but never block each other
-- the simulation batch must always run regardless of the agent's exit
code (it legitimately exits 1 on the benign pre-season "no_projections"
case).

Match-event scraping needs a real browser (Chrome/Chromium) and can hit
Cloudflare -- pass --skip-match-events to skip FBref/WhoScored on a run
where that's not available or not worth the time.

Flags are passed straight through to scripts/run_agent.py -- pass nothing
to use its own .env-based DRY_RUN default, same as the systemd service.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _run(args: list[str], env: dict[str, str] | None = None) -> int:
    logger.info("Running: %s", " ".join(args))
    return subprocess.run(args, cwd=REPO_ROOT, env=env).returncode


# Steps that report success having written nothing (2026-08-28). Twice in one
# day a warn-and-continue step exited 0 with an empty result and the run read
# as healthy: scrape_understat_xg.py treated an unreadable schedule as "season
# not published" and skipped the refresh, leaving player_xg_stats at 2 non-zero
# xg rows out of 309; and before that the Understat step was missing from this
# pipeline altogether, which this module's own comments record as 11,495 rows
# for 2025-26 with 87 non-zero xg, "left in place because nothing ever ran it".
#
# The exit code cannot see any of that -- these steps exit 0 on an empty
# result by design. A post-condition asks the other question: is the data this
# step is responsible for actually there?
#
# Post-conditions, not before/after deltas: these are idempotent upserts, so a
# healthy re-run legitimately changes no counts, and warning on "nothing
# changed" would fire every week. Collected and re-reported together at the
# end, because a warning 200 lines up the log is a warning nobody reads.
_POSTCONDITION_WARNINGS: list[str] = []


def _postcondition_warning(step_name: str, label: str, count: int | None) -> str | None:
    """The warning this step's post-condition earns, or None if it is fine.

    ``count is None`` means the check itself could not run. That is warned
    rather than passed: a check that silently fails open is the same failure
    class it was written to catch.
    """
    if count is None:
        return (
            f"{step_name}: post-condition ({label}) could not be checked -- "
            f"treat this run's output for that step as unverified"
        )
    if count == 0:
        return (
            f"{step_name}: exited cleanly but {label} is EMPTY -- the step "
            f"reported success having written nothing"
        )
    return None


def _count_rows(sql: str, params: dict) -> int | None:
    """Row count for a post-condition, or None if the query cannot be run."""
    try:
        from sqlalchemy import text

        from data.db import get_session
        db = get_session()
        try:
            return int(db.execute(text(sql), params).scalar() or 0)
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 -- an unrunnable check is reported, not raised
        logger.warning("Post-condition query failed: %s", exc)
        return None


def _run_or_warn(
    step_name: str,
    args: list[str],
    env: dict[str, str] | None = None,
    *,
    postcondition: tuple[str, str, dict] | None = None,
) -> None:
    """Run a best-effort step. ``postcondition`` is (label, sql, params) naming
    the data the step must leave behind."""
    code = _run(args, env=env)
    if code != 0:
        logger.warning(
            "%s exited with code %d -- continuing with whatever data is already "
            "in the DB (this step is best-effort, never blocks the rest of the run)",
            step_name, code,
        )
    if postcondition is None:
        return
    label, sql, params = postcondition
    warning = _postcondition_warning(step_name, label, _count_rows(sql, params))
    if warning:
        logger.warning(warning)
        _POSTCONDITION_WARNINGS.append(warning)


def _season_has_started(season: str) -> bool:
    """Has a single gameweek of ``season`` actually been played?

    The match-event scrapers exist to collect what happened in matches. Before
    a season's first kickoff there are no matches, so running them is at best a
    slow no-op -- and at worst not a no-op at all: asked for a season FBref has
    no match reports for, the scrape wanders off fetching pages that have
    nothing to do with the current season (a 1926-1927 archive page was
    observed on 2026-08-18), burning browser time against Cloudflare for data
    that could only ever be discarded.

    Nothing was written -- the ingest keys on the requested season, so the junk
    had nowhere to land -- but a step whose only possible outcomes are "no-op"
    and "wrong" should not run at all.
    """
    from projection.pipeline import season_has_played_history

    try:
        return season_has_played_history(season)
    except Exception as exc:
        # Unreadable DB: assume it HAS started, so a real in-season week is
        # never silently skipped. Pre-season the cost of being wrong is a
        # wasted scrape; in-season it is stale DefCon and bonus data.
        logger.warning("Could not tell whether %s has started (%s) — not skipping", season, exc)
        return True


def _current_gameweek() -> int | None:
    """The gameweek whose deadline has just passed -- the one to sample a
    fresh ownership snapshot for (see ingest_ownership.py's own caveat:
    sampling before a GW's deadline gets zero ranked entries)."""
    from projection.pipeline import _get_current_and_next_gw

    try:
        current_gw, _ = _get_current_and_next_gw()
        return current_gw
    except Exception as exc:
        logger.warning("Could not determine the current gameweek for ownership: %s", exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Accepted and ignored (2026-08-18). This engine has no submission "
             "path, so every run is what --dry-run used to mean. Kept so "
             "documented commands and the systemd unit keep working.",
    )
    parser.add_argument("--chip", default=None, help="Force a specific chip")
    parser.add_argument("--season", default="2026-27")
    parser.add_argument(
        "--skip-match-events", action="store_true",
        help="Skip FBref/WhoScored (no browser available, or not worth the time this week)",
    )
    args = parser.parse_args()

    if args.skip_match_events:
        logger.info("Skipping FBref/WhoScored match-event refresh (--skip-match-events)")
    elif not _season_has_started(args.season):
        logger.info(
            "Skipping FBref/WhoScored/set-piece refresh — %s has no played "
            "gameweeks yet, so there are no match events to collect. Resumes "
            "by itself once GW1 has been played. To refresh a PRIOR season's "
            "events, run scripts/scrape_fbref.py with that season directly.",
            args.season,
        )
    else:
        # scrape_fbref.py's own default is headless, but FBref sits behind
        # Cloudflare and headless mode cannot clear its CAPTCHA (confirmed
        # 2026-08-01: a real headless run on the user's machine hit
        # "CAPTCHA detected... attempting to solve" and failed) -- force
        # headed unless the user has already set an explicit override.
        fbref_env = {**os.environ, "FBREF_HEADED": os.environ.get("FBREF_HEADED", "1")}
        _run_or_warn(
            "scripts/scrape_fbref.py",
            [sys.executable, "scripts/scrape_fbref.py", args.season],
            env=fbref_env,
            postcondition=(
                f"{args.season} match events",
                "SELECT COUNT(*) FROM player_match_events WHERE season = :s",
                {"s": args.season},
            ),
        )
        _run_or_warn(
            "scripts/scrape_whoscored.py",
            [sys.executable, "scripts/scrape_whoscored.py", args.season],
            # WhoScored only PATCHES rows FBref already created, so a row
            # count would pass on FBref's work alone. The defensive columns
            # are the ones only this step can fill -- the DefCon/bonus gap it
            # exists to close.
            postcondition=(
                f"{args.season} match events carrying defensive actions",
                "SELECT COUNT(*) FROM player_match_events "
                "WHERE season = :s AND (clearances > 0 OR interceptions > 0 "
                "OR recoveries > 0)",
                {"s": args.season},
            ),
        )
        # P3.7: penalty/set-piece duty. Shares the browser requirement above,
        # so it lives behind the same --skip-match-events guard. Duty moves
        # with transfers and managers, so this is refreshed rather than
        # scraped once -- write_setpiece_roles updates in place.
        _run_or_warn(
            "scripts/scrape_setpieces.py",
            [sys.executable, "scripts/scrape_setpieces.py", args.season],
            env=fbref_env,
            postcondition=(
                f"{args.season} set-piece roles",
                "SELECT COUNT(*) FROM player_setpiece_roles WHERE season = :s",
                {"s": args.season},
            ),
        )

    # 2026-08-25: xG/xA/npxg/key-passes. NOT behind --skip-match-events, and
    # deliberately not grouped with the three scrapes above: Understat is
    # browserless (soccerdata's TLS client, no Chrome, no API key), so the
    # reason those are optional does not apply to it.
    #
    # This step did not exist, and the omission was silent and expensive.
    # player_xg_stats had 11,495 rows for 2025-26 and only 87 of them carried a
    # non-zero xg -- the "shots-only interim" that data/ingestors/understat_xg.py
    # was written to replace, left in place because nothing ever ran it after
    # the initial backfill. 2026-27 had no rows at all. projection/assemble.py
    # LEFT JOINs the table and COALESCEs to 0, so a missing row is
    # indistinguishable from a genuine zero: every projection was being built
    # with the attacking signal switched off, and nothing reported it.
    #
    # Warn-only like its neighbours: a failed xG refresh degrades the
    # projection, it does not invalidate the week's decision.
    if _season_has_started(args.season):
        _run_or_warn(
            "scripts/scrape_understat_xg.py",
            [sys.executable, "scripts/scrape_understat_xg.py", args.season],
            # xg > 0, not a row count: the failure mode this exists to catch
            # wrote 309 rows carrying shots and no xG at all.
            postcondition=(
                f"{args.season} xG rows carrying real xg",
                "SELECT COUNT(*) FROM player_xg_stats WHERE season = :s AND xg > 0",
                {"s": args.season},
            ),
        )

    gw = _current_gameweek()
    if gw:
        _run_or_warn(
            "scripts/ingest_ownership.py",
            [sys.executable, "scripts/ingest_ownership.py", str(gw)],
            postcondition=(
                "ownership snapshots",
                "SELECT COUNT(*) FROM ownership_snapshots",
                {},
            ),
        )

    agent_args = [sys.executable, "scripts/run_agent.py", "--season", args.season]
    if args.chip:
        agent_args.extend(["--chip", args.chip])

    agent_code = _run(agent_args)
    logger.info("run_agent.py exited with code %d", agent_code)

    # P2.3 (2026-08-16): score the gameweek that just finished, for the real
    # bot and every persona. Without this the season produces a full record of
    # decisions and no record of how any of them turned out, which is the one
    # thing the live walk-through is for.
    #
    # Moved to AFTER run_agent on 2026-08-25. It ran before, and silently did
    # nothing all season: the scorer gates on gameweeks.finished AND
    # data_checked and needs player_gw_stats for the gameweek, and BOTH are
    # written by run_full_ingest -- which happens inside run_agent.py, after
    # this used to run. On the first live attempt the local DB still showed
    # GW1 unfinished with zero stats rows, so the gate correctly refused, and
    # the run reported success having scored nothing.
    #
    # Ordering it after the decision is safe: nothing in the decision path
    # reads `actual_outcome` (agent/decision_engine.py mentions it only in a
    # docstring and a comment). The scoring is a record of the PREVIOUS
    # gameweek either way, and this is the earliest point at which the data it
    # needs actually exists. Same argument the data-quality gate below already
    # makes for running after run_agent rather than before.
    _run_or_warn(
        "scripts/backfill_decision_outcomes.py",
        [sys.executable, "scripts/backfill_decision_outcomes.py", "--season", args.season],
    )
    if agent_code != 0:
        logger.warning(
            "Real agent run reported an error (exit %d) -- check the log above "
            "before assuming your GW decision went through", agent_code,
        )

    # P3.9 (2026-08-16): data-integrity checks, AFTER run_agent rather than
    # before it. The freshest ingest of players/teams/fixtures happens INSIDE
    # run_agent.py (run_full_ingest is the first thing it does), so running
    # the gate first checked last week's data and reported staleness that was
    # about to be fixed seconds later -- confirmed on the first scheduled run,
    # which flagged four team_ids that matched the live feed immediately
    # afterwards. Run here it validates the data this week's decision was
    # ACTUALLY made on, which is the question worth answering. Warn-only: a
    # blocked week is worse than a week decided on slightly stale data.
    _run_or_warn(
        "scripts/data_quality_gate.py",
        [sys.executable, "scripts/data_quality_gate.py"],
    )

    # 2026-08-18: regenerate the site export BEFORE preflight, because
    # preflight asserts the published site matches decision_log -- and nothing
    # in this pipeline produced it. So every legitimate re-decision left that
    # check failing until someone remembered to run the export by hand, which
    # trains you to ignore a failing preflight. Either the pipeline owns the
    # artefact or the check should not exist; it owns it.
    #
    # --no-push keeps the weekly run from touching a remote on its own; the
    # commit is local and the checklist covers publishing.
    _run_or_warn(
        "scripts/export_site_data.py",
        [sys.executable, "scripts/export_site_data.py", "--no-push"],
    )

    # P4 (2026-08-17): the decision SURFACE, after the gate has checked the
    # data it was built from. The gate answers "is the data plausible"; this
    # answers "is the answer legal, and did it change". Five defects that day
    # were introduced by fixes made the same day, every one passing both the
    # suite and the gate, because each altered stored state that some other
    # consumer read. Drift against the committed baseline is the only signal
    # that catches that class. Warn-only for the same reason as the gate: a
    # blocked week is worse than a week that needs a look.
    _run_or_warn(
        "scripts/preflight.py",
        [sys.executable, "scripts/preflight.py"],
    )

    sim_code = _run([sys.executable, "scripts/run_simulations.py", "--season", args.season])
    logger.info("run_simulations.py exited with code %d", sim_code)
    if sim_code != 0:
        logger.warning(
            "Simulation batch reported an error (exit %d) -- check the log above", sim_code
        )

    # Re-reported together: a warning 200 lines up a scrape log is a warning
    # nobody reads, and these are precisely the failures that look like
    # success at the exit-code level.
    if _POSTCONDITION_WARNINGS:
        logger.warning(
            "%d step(s) finished without producing their data:",
            len(_POSTCONDITION_WARNINGS),
        )
        for warning in _POSTCONDITION_WARNINGS:
            logger.warning("  - %s", warning)
    else:
        logger.info("All step post-conditions satisfied")


if __name__ == "__main__":
    main()
