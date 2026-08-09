"""Real bug found 2026-08-09 (manual scrape_fbref.py run): soccerdata's
season-code parser keeps only the last two digits of each year, so
'2026-2027' and '1926-1927' both normalize to the same internal code
'2627'. Since the 2026-27 season hadn't started, FBref's site had no
row for it yet, and the lookup silently fell back to the 1926-27 First
Division archive instead of raising.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.ingestors.fbref import _validate_schedule_season


def test_validate_schedule_season_accepts_matching_year():
    schedule = pd.DataFrame({"date": ["2026-08-21", "2026-08-28"]})
    _validate_schedule_season(schedule, "2026-27")  # must not raise


def test_validate_schedule_season_accepts_season_crossing_new_year():
    schedule = pd.DataFrame({"date": ["2026-08-21", "2027-01-15"]})
    _validate_schedule_season(schedule, "2026-27")  # must not raise


def test_validate_schedule_season_rejects_century_collision():
    schedule = pd.DataFrame({"date": ["1926-09-01", "1927-01-15"]})
    with pytest.raises(ValueError, match="season-code collision"):
        _validate_schedule_season(schedule, "2026-27")


def test_validate_schedule_season_skips_empty_schedule():
    _validate_schedule_season(pd.DataFrame({"date": []}), "2026-27")  # must not raise


def test_validate_schedule_season_skips_missing_date_column():
    _validate_schedule_season(pd.DataFrame({"other": [1, 2]}), "2026-27")  # must not raise
