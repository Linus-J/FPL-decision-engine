#!/usr/bin/env python
"""CLI entrypoint for the weekly simulation batch (plan/simulation-engine-v1.md).

Invoked right after scripts/run_agent.py in the Fri/Sat/Sun systemd job, as
a SEPARATE process -- a crash here can never block or corrupt the real
agent's already-completed run. Every persona's own failure is already
caught inside simulation.engine.run_all_personas; only a catastrophic
failure (e.g. the DB itself being unreachable) would raise out of main()
here, and that should be visible in the logs, not swallowed.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.engine import run_all_personas

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    args = parser.parse_args()

    results = run_all_personas(args.season)
    n_ok = sum(1 for r in results.values() if "error" not in r)
    n_err = len(results) - n_ok
    logger.info(
        "Simulation batch complete: %d ok, %d failed (season=%s)", n_ok, n_err, args.season
    )
    print(json.dumps({"season": args.season, "ok": n_ok, "failed": n_err}, indent=2))


if __name__ == "__main__":
    main()
