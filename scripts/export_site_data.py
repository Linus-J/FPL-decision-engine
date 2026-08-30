#!/usr/bin/env python
"""Export the current squad, top-15 projections, and decision history for
the portfolio site's $ fpl status panel. Run manually, after reviewing the
week's decision -- see README.md > Running > "Site data export"."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from data.db import get_session
from scripts.site_export.cdn import purge as purge_cdn
from scripts.site_export.git_sync import commit_and_push
from scripts.site_export.payload import build_run_payload
from scripts.site_export.writer import update_index, write_run_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "simulations"

# How the portfolio site addresses this data on jsDelivr. These must match
# the `repo`, `ref` and `path` constants in that site's
# assets/panels/fpl.js: a jsDelivr purge is keyed on the request URL, so
# purging any other spelling clears an entry nobody asks for. The repo was
# renamed from FPL-26-27-bot; GitHub 301s the old name but the CDN caches
# the two spellings as separate entries, so this has to be the name the
# site actually requests, not whatever `origin` happens to be set to.
CDN_REPO = "Linus-J/FPL-decision-engine"
CDN_REF = "refs/heads/v2"
CDN_PATH = "data/simulations"

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

    # Pushing is not enough on its own: jsDelivr pins a mutable branch ref
    # for up to seven days, so without this the site kept serving a
    # five-day-old squad (2026-08-30). Both files matter -- a fresh
    # gw{N}.json behind a stale index.json is still a stale page.
    if committed and not no_push:
        purge_cdn(
            repo=CDN_REPO, ref=CDN_REF, path=CDN_PATH,
            files=["index.json", run_path.name],
        )


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
