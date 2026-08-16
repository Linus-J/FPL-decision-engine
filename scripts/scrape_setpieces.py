#!/usr/bin/env python
"""scrape_setpieces.py — penalty and set-piece takers into
``player_setpiece_roles`` (P3.7).

``projection/features.py`` has always LEFT JOINed this table and COALESCEd
every field to zero, so penalty duty was silently absent from every
projection the model has ever made. This fills it.

Needs soccerdata + a Chromium/Chrome browser, same as the FBref scrape --
FBref sits behind Cloudflare and headless is often blocked, so pass
``--headed`` if a headless run fails:

    DB_PATH=fpl_bot_v2.db uv run --with soccerdata \
        python scripts/scrape_setpieces.py 2026-27 --headed

By default the EVIDENCE comes from the prior season (pre-season has no
current-season record to read). Once the season is under way, refresh from
live evidence with ``--source-season 2026-27``.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", nargs="?", default="2026-27")
    parser.add_argument(
        "--source-season", default=None,
        help="Season to read duties FROM (default: the season before `season`)",
    )
    parser.add_argument("--headed", action="store_true", help="Run the browser headed")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--path-to-browser", default=None)
    args = parser.parse_args()

    from data.db import init_db
    from data.ingestors.setpiece import ingest_setpiece_roles

    init_db()
    written, unmatched = ingest_setpiece_roles(
        args.season,
        source_season=args.source_season,
        no_cache=args.no_cache,
        path_to_browser=args.path_to_browser,
        headless=not args.headed,
    )
    logger.info("Wrote %d set-piece roles (%d players unmatched)", written, unmatched)
    # Unmatched players are expected (foreign signings, spelling variants) and
    # are reported, not fatal -- an empty write is the real failure.
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
