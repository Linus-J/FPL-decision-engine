#!/usr/bin/env python
"""ingest_ownership.py — sample top-10k effective ownership for a gameweek.

Browserless (plain aiohttp against FPL's public API, same as the rest of
data/ingestors/fpl_api.py) — runnable headless/on a schedule. Samples the
"Overall" classic league standings + a sample of those managers' picks to
build top10k_selected_pct/captaincy_pct_top10k (P3-2).

    DB_PATH=fpl_bot_v2.db python scripts/ingest_ownership.py <gw> [sample_size]

⚠️ UNVERIFIED AGAINST LIVE DATA as of authoring (2026-07-26) — the 2026-27
season has zero played gameweeks, so the Overall league has zero ranked
entries right now. Run this for real once GW1's picks lock (after the
deadline, so captaincy/ownership reflect final squads) — see
data/ingestors/ownership.py's module docstring for the full caveat.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from data.db import init_db
from data.ingestors.ownership import DEFAULT_SAMPLE_SIZE, ingest_ownership_snapshot

logger = logging.getLogger(__name__)


async def main(gw: int, sample_size: int) -> None:
    init_db()
    written, sampled = await ingest_ownership_snapshot(gw, sample_size=sample_size)
    logger.info(
        "GW%d ownership snapshot: %d players written from %d sampled entries",
        gw, written, sampled,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <gameweek> [sample_size={DEFAULT_SAMPLE_SIZE}]")
        sys.exit(1)
    target_gw = int(sys.argv[1])
    size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SAMPLE_SIZE
    asyncio.run(main(target_gw, size))
