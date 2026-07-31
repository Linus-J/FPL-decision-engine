#!/usr/bin/env python
"""Manual weekly kickoff: runs the real agent decision, then the simulation
batch, in that order -- the simulation batch always runs regardless of the
agent's exit code (it legitimately exits 1 on the benign pre-season
"no_projections" case; see deploy/fpl-bot.service's ExecStart chain for the
same reasoning). Built for running this by hand on a machine that isn't
always on, rather than relying solely on the systemd timer.

Flags are passed straight through to scripts/run_agent.py -- pass nothing
to use its own .env-based DRY_RUN default, same as the systemd service.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _run(args: list[str]) -> int:
    logger.info("Running: %s", " ".join(args))
    return subprocess.run(args, cwd=REPO_ROOT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Force live submission")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run (no submission)")
    parser.add_argument("--chip", default=None, help="Force a specific chip")
    parser.add_argument("--season", default="2026-27")
    args = parser.parse_args()

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
