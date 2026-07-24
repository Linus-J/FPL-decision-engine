#!/usr/bin/env python
"""scrape_fbref.py — run the OPTIONAL FBref event scrape (T5b live path).

Browser-only: soccerdata drives a headless-incompatible Selenium/Chrome stack to
get past FBref's Cloudflare wall, so this cannot run in CI/background — run it on
a workstation with Chrome. Populates ``player_match_events`` and then recomputes
``recomputed_bonus`` under the 26/27 BPS rules.

Usage (from the repo root, with a browser available):

    uv pip install soccerdata            # the optional 'events' dependency
    DB_PATH=fpl_bot_v2.db uv run python scripts/scrape_fbref.py 2025-26
    # multiple seasons:
    DB_PATH=fpl_bot_v2.db uv run python scripts/scrape_fbref.py 2025-26 2024-25

Defaults to 2025-26 (the Phase-2 exit-gate / P-RS season) when no season is given.
Players are matched to the DB by name, so run this AFTER the player roster is
current so departed players still resolve. Re-runnable: event writes are
idempotent on (player_id, season, game_id).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from data.db import get_session, init_db
from data.ingestors.fbref import ingest_fbref_season
from projection.bonus_recompute import recompute_season, recomputed_bonus_coverage

logger = logging.getLogger(__name__)

DEFAULT_SEASONS = ["2025-26"]


def main(seasons: list[str]) -> None:
    init_db()
    for season in seasons:
        logger.info("=== FBref scrape: %s ===", season)
        written, unmatched = ingest_fbref_season(season)
        logger.info("%s: %d event rows written, %d players unmatched", season, written, unmatched)

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
    logger.info("FBref scrape + bonus recompute complete")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_SEASONS)
