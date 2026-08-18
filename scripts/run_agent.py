#!/usr/bin/env python
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from agent import decision_engine, fpl_client, notifier
from config.settings import settings
from config.strategy import OPTIMISER
from data.ingestors.fpl_api import run_full_ingest
from data.ingestors.injury_parser import run_injury_parser
from data.ingestors.odds_api import ingest_odds_sync, log_odds_coverage
from data.ingestors.understat import run_understat_ingest
from optimiser.chips import Chip


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FPL autonomous agent")
    p.add_argument("--dry-run", action="store_true", default=None, help="Override DRY_RUN=true")
    p.add_argument(
        "--live", action="store_true", default=False,
        help="Force live submission (overrides DRY_RUN)",
    )
    p.add_argument(
        "--chip", choices=[c.value for c in Chip], default=None,
        help="Force a specific chip",
    )
    p.add_argument("--season", default="2026-27", help="Season string (default: 2026-27)")
    p.add_argument("--json-out", type=Path, default=None, help="Write decision JSON to this path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.live:
        dry_run = False
    elif args.dry_run:
        dry_run = True
    else:
        dry_run = settings.dry_run

    force_chip = Chip(args.chip) if args.chip else None

    logging.getLogger().info(
        "Starting FPL agent — season=%s dry_run=%s chip=%s",
        args.season, dry_run, force_chip,
    )

    # 2026-07-30 (user's own live-smoke-test follow-up): this script used
    # to run injury/press signals straight into a decision without ever
    # refreshing players/teams/fixtures/gameweeks first -- the deployed
    # timer (deploy/fpl-bot.timer) only schedules THIS script, so nothing
    # anywhere ever kept that base data fresh. run_full_ingest first so a
    # single run always decides against current data.
    try:
        asyncio.run(run_full_ingest(args.season))
    except Exception as exc:
        logging.getLogger().warning("Full FPL ingest skipped: %s", exc)

    try:
        ingest_odds_sync()
        # P3.11: make the odds window visible. Bookmakers price only
        # near-term fixtures, so the planning horizon is routinely longer
        # than the odds window — and an uncovered gameweek silently projects
        # on a flat league-average scoreline instead of real fixture
        # difficulty. Nothing here widens the window; it stops the horizon's
        # real information content being an assumption.
        log_odds_coverage(args.season, OPTIMISER.projection_horizon_gws)
    except Exception as exc:
        logging.getLogger().warning("Odds ingest skipped: %s", exc)

    try:
        asyncio.run(run_understat_ingest([args.season]))
    except Exception as exc:
        logging.getLogger().warning("Understat ingest skipped: %s", exc)

    try:
        run_injury_parser()
    except Exception as exc:
        logging.getLogger().warning("Injury parser skipped: %s", exc)

    # P3.8 (2026-08-16): press-conference signals are NOT ingested here any
    # more. `player_press_signals` was written every run and read by nothing
    # but the model definition -- a shadow layer that was never compared to
    # anything, so it was pure weekly cost. The ingestor
    # (data/ingestors/press_conference.py) and its table are kept intact;
    # re-enable this once there is a way to measure whether the signal helps
    # (the obvious route is a swept persona axis, so the cohort answers it
    # before the real bot ever uses it -- see the recovery plan P3.8).

    decision = decision_engine.run(
        season=args.season,
        force_chip=force_chip,
        dry_run=dry_run,
    )

    if "error" in decision:
        logging.getLogger().error("Decision engine error: %s", decision["error"])
        sys.exit(1)

    submission = fpl_client.submit(
        squad=decision["squad"],
        captain_id=decision["captain_id"],
        vice_captain_id=decision["vice_captain_id"],
        transfers_in=decision["transfers_in"],
        transfers_out=decision["transfers_out"],
        hits_taken=decision["hits_taken"],
        chip=decision["chip"],
        free_transfers=max(0, 1 - len(decision["transfers_in"])),
        dry_run=dry_run,
    )

    output = {**decision, "submission": submission}

    notifier.notify_sync(decision)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2, default=str))
        logging.getLogger().info("Decision written to %s", args.json_out)

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
