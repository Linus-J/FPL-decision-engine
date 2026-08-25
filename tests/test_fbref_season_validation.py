"""Real bug found 2026-08-09 (manual scrape_fbref.py run): soccerdata's
season-code parser keeps only the last two digits of each year, so
'2026-2027' and '1926-1927' both normalize to the same internal code
'2627'. The lookup silently fell back to the 1926-27 First Division
archive instead of raising.

Corrected 2026-08-25: the original note here blamed FBref for not listing
the season. It does list it. soccerdata resolves a season against a CACHED
copy of FBref's seasons index, de-duplicating the shared code and keeping
the first entry; while the current season is listed it wins, because FBref
lists newest first. The failure needs a cache taken BEFORE the season was
added -- on 2026-08-25 that was a copy from 2026-07-24, holding 1926-1927
and no 2026-2027 at all. Hence refresh_stale_seasons_cache, tested below.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.ingestors.fbref import _validate_schedule_season, cached_seasons


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


_SEASONS_HTML = """
<html><body><table id="seasons">
<tr><th data-stat="year_id"><a href="/x">{newest}</a></th></tr>
<tr><th data-stat="year_id"><a href="/y">2024-2025</a></th></tr>
<tr><th data-stat="year_id"><a href="/z">1926-1927</a></th></tr>
</table></body></html>
"""


def test_cached_seasons_reads_the_listed_labels(tmp_path):
    f = tmp_path / "seasons_ENG-Premier League.html"
    f.write_text(_SEASONS_HTML.format(newest="2025-2026"))

    assert cached_seasons(f) == ["2025-2026", "2024-2025", "1926-1927"]


def test_cached_seasons_missing_file_is_no_information(tmp_path):
    """Absent must never read as "the season is not listed" -- that would make
    every fresh checkout delete a file it had not looked at."""
    assert cached_seasons(tmp_path / "nope.html") == []


def test_stale_seasons_cache_is_removed(tmp_path, monkeypatch):
    """The 2026-08-25 failure: a seasons index cached 2026-07-24 listed
    1926-1927 but not 2026-2027, so the shared '2627' code resolved to the
    1926-27 season and the ingest pulled century-old fixtures.
    """
    import data.ingestors.fbref as fbref_module

    d = tmp_path / "FBref"
    d.mkdir()
    f = d / f"seasons_{fbref_module.FBREF_LEAGUE}.html"
    f.write_text(_SEASONS_HTML.format(newest="2025-2026"))
    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)

    assert fbref_module.refresh_stale_seasons_cache("2026-2027") is True
    assert not f.exists(), "the stale index must be dropped so it re-fetches"


def test_fresh_seasons_cache_is_kept(tmp_path, monkeypatch):
    """Once FBref lists the season it sorts newest-first, so soccerdata's
    keep="first" dedupe picks it over the century-old twin. Nothing to do."""
    import data.ingestors.fbref as fbref_module

    d = tmp_path / "FBref"
    d.mkdir()
    f = d / f"seasons_{fbref_module.FBREF_LEAGUE}.html"
    f.write_text(_SEASONS_HTML.format(newest="2026-2027"))
    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)

    assert fbref_module.refresh_stale_seasons_cache("2026-2027") is False
    assert f.exists(), "a current index must not be thrown away"


def test_missing_cache_is_not_treated_as_stale(tmp_path, monkeypatch):
    import data.ingestors.fbref as fbref_module

    (tmp_path / "FBref").mkdir()
    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)

    assert fbref_module.refresh_stale_seasons_cache("2026-2027") is False
