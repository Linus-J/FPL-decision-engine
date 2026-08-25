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


def test_season_code_matches_soccerdata_cache_naming():
    from data.ingestors.fbref import season_code

    assert season_code("2025-2026") == "2526"
    assert season_code("2026-2027") == "2627"


_GOOD_STATS_HTML = (
    "<html><body><!-- div_stats_shooting --><table id='stats_shooting'>"
    "</table></body></html>"
)
# What FBref actually served on 2026-08-16 and soccerdata cached as if it were
# the page: a consent wall with no stats table anywhere in it.
_BLOCK_PAGE_HTML = (
    "<html><head><title>Close this consent banner</title></head>"
    "<body>Please enable JavaScript and cookies to continue</body></html>"
)


def _stats_cache(tmp_path, league, stat_type, body):
    d = tmp_path / "FBref"
    d.mkdir(exist_ok=True)
    f = d / f"players_{league}_2526_{stat_type}.html"
    f.write_text(body)
    return f


def test_poisoned_stats_cache_is_purged(tmp_path, monkeypatch):
    """A cached Cloudflare interstitial fails deep inside soccerdata with
    "not enough values to unpack (expected 1, got 0)" -- its xpath for
    div_stats_<stat_type> matching nothing. Re-running never clears it, because
    the bad page IS the cache. Found 2026-08-25 on the set-piece scrape.
    """
    import data.ingestors.fbref as fbref_module

    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)
    bad = _stats_cache(tmp_path, fbref_module.FBREF_LEAGUE, "shooting", _BLOCK_PAGE_HTML)

    assert fbref_module.purge_unusable_stats_cache("2025-2026", ("shooting",)) == ["shooting"]
    assert not bad.exists()


def test_valid_stats_cache_is_kept(tmp_path, monkeypatch):
    """The marker checked is the same string soccerdata looks for, so a page
    that passes here is one it can parse. Never throw those away."""
    import data.ingestors.fbref as fbref_module

    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)
    good = _stats_cache(tmp_path, fbref_module.FBREF_LEAGUE, "shooting", _GOOD_STATS_HTML)

    assert fbref_module.purge_unusable_stats_cache("2025-2026", ("shooting",)) == []
    assert good.exists()


def test_absent_stats_cache_is_not_an_error(tmp_path, monkeypatch):
    import data.ingestors.fbref as fbref_module

    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)
    (tmp_path / "FBref").mkdir()

    assert fbref_module.purge_unusable_stats_cache(
        "2025-2026", ("shooting", "passing", "passing_types")
    ) == []


def test_purge_checks_each_stat_type_independently(tmp_path, monkeypatch):
    import data.ingestors.fbref as fbref_module

    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)
    lg = fbref_module.FBREF_LEAGUE
    good = _stats_cache(tmp_path, lg, "passing", _GOOD_STATS_HTML.replace("shooting", "passing"))
    bad = _stats_cache(tmp_path, lg, "shooting", _BLOCK_PAGE_HTML)

    purged = fbref_module.purge_unusable_stats_cache(
        "2025-2026", ("shooting", "passing")
    )

    assert purged == ["shooting"]
    assert good.exists() and not bad.exists()


def test_modal_year_ignores_incidental_recent_dates():
    """A 1926-27 schedule page still carries recent dates in its chrome, so
    the presence of a 2026 date proves nothing. The year most fixtures fall
    in is what identifies the season."""
    from data.ingestors.fbref import modal_year

    page = " ".join(["1926-09-04"] * 20 + ["1927-03-12"] * 18 + ["2026-08-25"])

    assert modal_year(page) == 1926


def test_modal_year_none_when_no_dates():
    from data.ingestors.fbref import modal_year

    assert modal_year("<html>no dates here</html>") is None


def _season_cache(tmp_path, league, kind, body):
    d = tmp_path / "FBref"
    d.mkdir(exist_ok=True)
    f = d / f"{kind}_{league}_2627.html"
    f.write_text(body)
    return f


def test_wrong_season_schedule_cache_is_purged(tmp_path, monkeypatch):
    """Repairing the seasons index only fixes which URL is RESOLVED. A schedule
    already downloaded under the ambiguous 2627 code is still served from cache,
    so the ingest keeps reading 1926-27 without touching the network -- which is
    why re-running changed nothing on 2026-08-25.
    """
    import data.ingestors.fbref as fbref_module

    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)
    old = _season_cache(
        tmp_path, fbref_module.FBREF_LEAGUE, "schedule",
        " ".join(["1926-09-04"] * 20 + ["1927-03-12"] * 18),
    )

    assert fbref_module.purge_wrong_season_caches("2026-2027") == ["schedule"]
    assert not old.exists()


def test_right_season_schedule_cache_is_kept(tmp_path, monkeypatch):
    import data.ingestors.fbref as fbref_module

    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)
    good = _season_cache(
        tmp_path, fbref_module.FBREF_LEAGUE, "schedule",
        " ".join(["2026-08-21"] * 20 + ["2027-05-10"] * 18),
    )

    assert fbref_module.purge_wrong_season_caches("2026-2027") == []
    assert good.exists()


def test_block_page_cache_is_purged_even_without_dates(tmp_path, monkeypatch):
    """teams_ENG-Premier League_2627.html was a consent wall with no dates at
    all, so the year check cannot judge it. The interstitial signature can."""
    import data.ingestors.fbref as fbref_module

    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)
    bad = _season_cache(
        tmp_path, fbref_module.FBREF_LEAGUE, "teams",
        "<title>Close this consent banner</title>",
    )

    assert fbref_module.purge_wrong_season_caches("2026-2027") == ["teams"]
    assert not bad.exists()


def test_purge_wrong_season_caches_tolerates_absent_files(tmp_path, monkeypatch):
    import data.ingestors.fbref as fbref_module

    monkeypatch.setattr(fbref_module, "SD_DATA_DIR", tmp_path)
    (tmp_path / "FBref").mkdir()

    assert fbref_module.purge_wrong_season_caches("2026-2027") == []
