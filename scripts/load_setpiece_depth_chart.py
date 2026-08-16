#!/usr/bin/env python
"""load_setpiece_depth_chart.py — load a published penalty/set-piece taker
list into ``player_setpiece_roles``.

Preferred over scripts/scrape_setpieces.py at the start of a season: FBref
has not populated the new season yet, and last season's attempt share cannot
survive a summer transfer window. A published list also carries taker ORDER,
which attempt share only approximates.

Expects a ``Team | Penalties | Free Kicks | Corners`` table, one team per
line, each cell a comma-separated list in depth-chart order:

    DB_PATH=fpl_bot_v2.db python scripts/load_setpiece_depth_chart.py \
        setPiece2627GW1.txt --season 2026-27 --source allaboutfpl
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
    parser.add_argument("path", type=Path)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--source", default="depth-chart")
    args = parser.parse_args()

    from data.db import init_db
    from data.ingestors.setpiece import ingest_depth_chart

    init_db()
    written, unresolved = ingest_depth_chart(
        args.season, args.path.read_text(), args.source
    )
    logger.info("Wrote %d roles; %d names unresolved", written, len(unresolved))
    # Unresolved names are reported, not fatal -- but an empty write means the
    # table was not parsed at all, which is.
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
