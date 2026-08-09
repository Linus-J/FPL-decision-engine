#!/usr/bin/env python
"""Export the current squad, top-15 projections, and decision history for
the portfolio site's $ fpl status panel. Run manually, after reviewing the
week's decision -- see README.md > Running > "Site data export"."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.settings import settings
from data.db import get_session
from scripts.site_export.git_sync import commit_and_push
from scripts.site_export.payload import build_run_payload
from scripts.site_export.writer import update_index, write_run_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "simulations"

logger = logging.getLogger(__name__)


def run(*, no_push: bool) -> None:
    db = get_session()
    try:
        payload = build_run_payload(db, settings.fpl_team_id)
    finally:
        db.close()

    gw = payload["gameweek"]
    run_path = write_run_file(DATA_DIR, gw, payload)
    logger.info("Wrote %s (%d bytes)", run_path, run_path.stat().st_size)

    index_path = update_index(DATA_DIR, gw, payload["label"], payload["generated_at"])
    logger.info("Updated %s", index_path)

    committed = commit_and_push(
        REPO_ROOT, DATA_DIR, f"export: GW{gw} site data", push=not no_push
    )
    if committed:
        logger.info("Committed%s", " (not pushed)" if no_push else " and pushed")
    else:
        logger.info("No changes to commit")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export current squad/projections/history for the portfolio site"
    )
    parser.add_argument(
        "--no-push", action="store_true", help="Write and commit locally without pushing"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(no_push=args.no_push)


if __name__ == "__main__":
    main()
