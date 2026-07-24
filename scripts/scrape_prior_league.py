#!/usr/bin/env python
"""scrape_prior_league.py — FBref season stats for non-PL leagues (P11 prior).

Browser-only (soccerdata + Chromium, headed recommended). Season-level, so it's
a handful of requests per league — far lighter than the PL match scrape.

Usage (from repo root, with Chromium):

    # top-5 leagues (registered in soccerdata out of the box):
    FBREF_HEADED=1 DB_PATH=fpl_bot_v2.db uv run --with soccerdata \
        python scripts/scrape_prior_league.py "ESP-La Liga" 2025-2026

    # all default prior leagues for 2025-2026 (omit the league arg):
    FBREF_HEADED=1 DB_PATH=fpl_bot_v2.db uv run --with soccerdata \
        python scripts/scrape_prior_league.py

ENG-Championship is NOT in soccerdata by default — register it once in
~/soccerdata/config/league_dict.json (see the snippet this script prints if the
league is unknown), then re-run with "ENG-Championship" as the league arg.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from data.db import init_db
from data.ingestors.fbref_prior import PRIOR_LEAGUES, ingest_prior_league_season

logger = logging.getLogger(__name__)

DEFAULT_SEASON = "2025-2026"

_CHAMP_SNIPPET = """\
ENG-Championship needs a one-time league_dict entry. Create/merge into
~/soccerdata/config/league_dict.json:

{
  "ENG-Championship": {
    "FBref": "EFL Championship",
    "season_start": "Aug",
    "season_end": "May"
  }
}

The FBref value must be "EFL Championship" (the exact competition name on
fbref.com/en/comps/, id 10) — "Championship" alone does not match and yields
"No objects to concatenate". Then re-run with "ENG-Championship" as the arg."""


def main(argv: list[str]) -> None:
    league = argv[0] if argv else None
    season = argv[1] if len(argv) > 1 else DEFAULT_SEASON
    leagues = [league] if league else list(PRIOR_LEAGUES)

    path_to_browser = os.environ.get("FBREF_BROWSER") or None
    headless = os.environ.get("FBREF_HEADED", "") == ""
    init_db()
    for lg in leagues:
        logger.info("=== prior-league season stats: %s %s (headless=%s) ===", lg, season, headless)
        try:
            written = ingest_prior_league_season(
                lg, season, path_to_browser=path_to_browser, headless=headless
            )
            logger.info("%s %s: %d rows", lg, season, written)
        except Exception as exc:  # one league failing must not kill the rest
            logger.warning("%s %s failed: %s", lg, season, exc)
            if "hampionship" in lg:
                logger.warning("%s", _CHAMP_SNIPPET)
    logger.info("Prior-league scrape complete")


if __name__ == "__main__":
    main(sys.argv[1:])
