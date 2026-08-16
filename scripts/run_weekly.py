#!/usr/bin/env python
"""Manual weekly kickoff: refreshes match-event + ownership data for the
gameweek that just finished, THEN runs the real agent decision, THEN the
simulation batch -- in that order, so the decision is made against
up-to-date DefCon/bonus/ownership data, not whatever was last scraped.

FBref -> WhoScored -> ownership -> backfill_decision_outcomes.py ->
run_agent.py -> run_simulations.py.

The outcome backfill scores LAST gameweek's decisions (for the real bot and
all 100 personas) before this gameweek's are made, so it always runs against
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
    parser.add_argument("--live", action="store_true", help="Force live submission")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run (no submission)")
    parser.add_argument("--chip", default=None, help="Force a specific chip")
    parser.add_argument("--season", default="2026-27")
    parser.add_argument(
        "--skip-match-events", action="store_true",
        help="Skip FBref/WhoScored (no browser available, or not worth the time this week)",
    )
    args = parser.parse_args()

    if args.skip_match_events:
        logger.info("Skipping FBref/WhoScored match-event refresh (--skip-match-events)")
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

    gw = _current_gameweek()
    if gw:
        _run_or_warn(
            "scripts/ingest_ownership.py",
            [sys.executable, "scripts/ingest_ownership.py", str(gw)],
        )

    # P2.3 (2026-08-16): score the gameweek that just finished, for the real
    # bot and every persona, BEFORE new decisions are made. This step existed
    # but was never wired into any scheduled run, so `actual_outcome` was
    # never populated at all -- the season would have produced a full record
    # of decisions with no record of how any of them turned out, which is the
    # one thing the live walk-through is for. Runs after the match-event and
    # ownership refresh above so it scores against complete data.
    _run_or_warn(
        "scripts/backfill_decision_outcomes.py",
        [sys.executable, "scripts/backfill_decision_outcomes.py", "--season", args.season],
    )

    agent_args = [sys.executable, "scripts/run_agent.py", "--season", args.season]
    if args.live:
        agent_args.append("--live")
    if args.dry_run:
        agent_args.append("--dry-run")
    if args.chip:
        agent_args.extend(["--chip", args.chip])

    agent_code = _run(agent_args)
    logger.info("run_agent.py exited with code %d", agent_code)
    if agent_code != 0:
        logger.warning(
            "Real agent run reported an error (exit %d) -- check the log above "
            "before assuming your GW decision went through", agent_code,
        )

    sim_code = _run([sys.executable, "scripts/run_simulations.py", "--season", args.season])
    logger.info("run_simulations.py exited with code %d", sim_code)
    if sim_code != 0:
        logger.warning(
            "Simulation batch reported an error (exit %d) -- check the log above", sim_code
        )


if __name__ == "__main__":
    main()
