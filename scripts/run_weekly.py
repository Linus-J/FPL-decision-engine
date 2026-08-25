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


def _run_or_warn(step_name: str, args: list[str], env: dict[str, str] | None = None) -> None:
    code = _run(args, env=env)
    if code != 0:
        logger.warning(
            "%s exited with code %d -- continuing with whatever data is already "
            "in the DB (this step is best-effort, never blocks the rest of the run)",
            step_name, code,
        )


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
        )
        _run_or_warn(
            "scripts/scrape_whoscored.py",
            [sys.executable, "scripts/scrape_whoscored.py", args.season],
        )
        # P3.7: penalty/set-piece duty. Shares the browser requirement above,
        # so it lives behind the same --skip-match-events guard. Duty moves
        # with transfers and managers, so this is refreshed rather than
        # scraped once -- write_setpiece_roles updates in place.
        _run_or_warn(
            "scripts/scrape_setpieces.py",
            [sys.executable, "scripts/scrape_setpieces.py", args.season],
            env=fbref_env,
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
        )

    gw = _current_gameweek()
    if gw:
        _run_or_warn(
            "scripts/ingest_ownership.py",
            [sys.executable, "scripts/ingest_ownership.py", str(gw)],
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


if __name__ == "__main__":
    main()
