#!/usr/bin/env python
"""scrape_understat_xg.py — per-match xG/xA/key-passes from Understat (FREE).

Browserless (soccerdata Understat uses a TLS client, no Chrome, no API key).
Populates player_xg_stats with real per-GW xg/xa/key_passes — the P3/P4
attacking signal. Runnable headless.

    DB_PATH=fpl_bot_v2.db uv run --with soccerdata python scripts/scrape_understat_xg.py 2025-26
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from data.db import init_db
from data.ingestors.understat_xg import ingest_understat_xg_season

DEFAULT_SEASONS = ["2025-26"]


def main(seasons: list[str]) -> None:
    init_db()
    for season in seasons:
        written, unmatched = ingest_understat_xg_season(season)
        logging.getLogger(__name__).info(
            "%s: %d xG player-GW rows written, %d unmatched", season, written, unmatched
        )


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_SEASONS)
