#!/usr/bin/env python
"""scrape_whoscored.py — patch clearances/blocks/interceptions/tackles/
recoveries/dribbles onto player_match_events from WhoScored's raw event
stream (P10 follow-up — FBref's summary table structurally lacks these).

Browser-only, same reasoning as scrape_fbref.py — run it on a workstation
with Chrome/Chromium. Run scrape_fbref.py FIRST: this only UPDATEs rows
FBref's ingest already created (it never inserts new ones), then re-runs the
26/27 BPS recompute so recomputed_bonus/bonus_2627 pick up the richer counts.

Usage (from the repo root, with a browser available):

    DB_PATH=fpl_bot_v2.db WHOSCORED_HEADED=1 \
        uv run --with soccerdata python scripts/scrape_whoscored.py 2025-26

WhoScored blocks scrapers more aggressively than FBref — headed mode
(WHOSCORED_HEADED=1) is the default here (unlike FBref's headless default)
because that's what worked in the feasibility probe (scripts/probe_whoscored.py).

Defaults to 2025-26 (the Phase-2 exit-gate / P-RS season) when no season given.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from data.db import get_session, init_db
from data.ingestors.whoscored import ingest_whoscored_season
from projection.bonus_recompute import recompute_season, recomputed_bonus_coverage

logger = logging.getLogger(__name__)

DEFAULT_SEASONS = ["2025-26"]


def main(seasons: list[str]) -> None:
    path_to_browser = os.environ.get("WHOSCORED_BROWSER") or None
    headless = os.environ.get("WHOSCORED_HEADED", "1") != "1"
    init_db()
    for season in seasons:
        logger.info("=== WhoScored scrape: %s (headless=%s) ===", season, headless)
        updated, unmatched = ingest_whoscored_season(
            season, path_to_browser=path_to_browser, headless=headless
        )
        logger.info(
            "%s: %d player-GW rows updated with clearances/blocks/interceptions/"
            "tackles/recoveries/dribbles, %d unmatched",
            season, updated, unmatched,
        )

        db = get_session()
        try:
            matches, rows = recompute_season(db, season)
            coverage = recomputed_bonus_coverage(db, season)
        finally:
            db.close()
        logger.info(
            "%s: recomputed bonus for %d matches (%d rows), coverage %.1f%%",
            season, matches, rows, 100 * coverage,
        )
    logger.info("WhoScored scrape + bonus recompute complete")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_SEASONS)
