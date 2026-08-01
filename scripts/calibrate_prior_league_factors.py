#!/usr/bin/env python
"""calibrate_prior_league_factors.py — P11: compute real per-league
translation factors + variances from the made-the-jump hold-out, and print
the config/strategy.py values to hand-copy in.

Needs prior_league_stats already populated for every prior season in
projection.prior_league_translation.SEASON_TRANSITIONS (run
scripts/scrape_prior_league.py once per (league, season) -- see
plan/p11-prior-league-cold-start.md section 1) with identity mapping
already applied (scrape_prior_league.py runs the backfill automatically).

Usage: uv run python scripts/calibrate_prior_league_factors.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.strategy import PRIOR_LEAGUE
from data.ingestors.fbref_prior import PRIOR_LEAGUES
from projection.prior_league_translation import (
    MIN_CALIBRATION_SAMPLES,
    build_holdout,
    compute_league_stats,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)

# league label -> config/strategy.py PriorLeagueRules field-name suffix.
_FIELD_SUFFIX = {
    "ENG-Championship": "championship",
    "ESP-La Liga": "la_liga",
    "ITA-Serie A": "serie_a",
    "GER-Bundesliga": "bundesliga",
    "FRA-Ligue 1": "ligue_1",
}

def main() -> None:
    print("# Paste into config/strategy.py's PriorLeagueRules if these look sane:")
    for league in PRIOR_LEAGUES:
        holdout = build_holdout(league)
        factor, variance, n = compute_league_stats(holdout)
        suffix = _FIELD_SUFFIX[league]
        if factor is None:
            # reads the live config rather than a second hardcoded copy, so
            # a sparse league's printed line always matches what's actually
            # live instead of silently proposing to revert a real
            # calibration back to a stale placeholder.
            default_factor = PRIOR_LEAGUE.translation_factor(league)
            default_variance = PRIOR_LEAGUE.translation_variance(league)
            logger.warning(
                "%s: hold-out too sparse (n=%d < %d) -- keeping literature default %.2f",
                league, n, MIN_CALIBRATION_SAMPLES, default_factor,
            )
            factor, variance = default_factor, default_variance
        else:
            logger.info("%s: n=%d, factor=%.3f, variance=%.3f", league, n, factor, variance)
        print(f"    translation_factor_{suffix}: float = {factor:.3f}  # n={n}")
        print(f"    translation_variance_{suffix}: float = {variance:.3f}")


if __name__ == "__main__":
    main()
