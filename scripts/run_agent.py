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

from agent import decision_engine, notifier, team_sheet
from config.strategy import OPTIMISER
from data.db import init_db
from data.ingestors.fpl_api import run_full_ingest
from data.ingestors.injury_parser import run_injury_parser
from data.ingestors.odds_api import ingest_odds_sync, log_odds_coverage
from data.ingestors.understat import run_understat_ingest
from data.run_manifest import manifest_context, record_run
from optimiser.chips import Chip


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build this gameweek's FPL decision. Nothing is submitted."
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Accepted and ignored. This engine never submits; kept so existing "
             "commands and the systemd unit do not break.",
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

    force_chip = Chip(args.chip) if args.chip else None

    logging.getLogger().info(
        "Building GW decision — season=%s chip=%s (no submission path exists)",
        args.season, force_chip,
    )

    # 2026-07-30 (user's own live-smoke-test follow-up): this script used
    # to run injury/press signals straight into a decision without ever
    # refreshing players/teams/fixtures/gameweeks first -- the deployed
    # timer (deploy/fpl-bot.timer) only schedules THIS script, so nothing
    # anywhere ever kept that base data fresh. run_full_ingest first so a
    # single run always decides against current data.
    # 2026-09-09: this script is the deployed timer's only entry point
    # (deploy/fpl-bot.timer) and was the one runner in the repo that never
    # called init_db -- every scraper does. A table added since the live
    # database was built therefore never got created on it, and the code
    # reading it degraded quietly instead of failing, which is how a feature
    # ships and then does nothing. create_all builds missing tables and never
    # alters existing ones, so this is safe to run every week.
    init_db()

    archive_path = None
    try:
        archive_path = asyncio.run(run_full_ingest(args.season))
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

    # Recorded before the decision is made, so every decision row carries the
    # id of the run that wrote it. A null archive_path means this run decided
    # without a raw copy of its inputs (the ingest above was skipped).
    manifest_id = record_run("decision", season=args.season, archive_path=archive_path)
    with manifest_context(manifest_id):
        decision = decision_engine.run(
            season=args.season,
            force_chip=force_chip,
        )

    if "error" in decision:
        logging.getLogger().error("Decision engine error: %s", decision["error"])
        sys.exit(1)

    sheet = team_sheet.build(
        squad=decision["squad"],
        captain_id=decision["captain_id"],
        vice_captain_id=decision["vice_captain_id"],
        transfers_in=decision["transfers_in"],
        transfers_out=decision["transfers_out"],
        chip=decision["chip"],
    )

    output = {**decision, "team_sheet": sheet}

    notifier.notify_sync(decision)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2, default=str))
        logging.getLogger().info("Decision written to %s", args.json_out)

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
