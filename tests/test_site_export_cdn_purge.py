"""scripts/site_export/cdn.py

The portfolio site reads the exported JSON through jsDelivr, pinned to a
mutable branch ref. jsDelivr caches a branch ref for up to seven days
(`cache-control: max-age=604800, s-maxage=43200`), so between 25 and 30
August 2026 the site served a five-day-old gw2.json -- still listing
Gibbs-White and Pedro Porro -- while origin/v2 had the corrected squad all
along. Pushing is therefore not enough on its own; the CDN copy has to be
invalidated too.
"""

from __future__ import annotations

import pytest

from scripts.site_export import cdn


def test_purge_urls_cover_the_index_and_the_run_file():
    urls = cdn.purge_urls(repo="Linus-J/FPL-decision-engine", ref="refs/heads/v2",
                          path="data/simulations", files=["index.json", "gw2.json"])

    assert urls == [
        "https://purge.jsdelivr.net/gh/Linus-J/FPL-decision-engine@refs/heads/v2"
        "/data/simulations/index.json",
        "https://purge.jsdelivr.net/gh/Linus-J/FPL-decision-engine@refs/heads/v2"
        "/data/simulations/gw2.json",
    ]


def test_purge_url_matches_the_url_the_site_fetches():
    """The purge key is the request URL. assets/stats-panel.js builds
    `cdn.jsdelivr.net/gh/<repo>@<ref>/<path>/<file>`; purging any other
    spelling -- the repo's pre-rename name, say -- clears a key nobody asks
    for and leaves the stale entry serving."""
    fetched = ("https://cdn.jsdelivr.net/gh/Linus-J/FPL-decision-engine@refs/heads/v2"
               "/data/simulations/gw2.json")
    purged = cdn.purge_urls(repo="Linus-J/FPL-decision-engine", ref="refs/heads/v2",
                            path="data/simulations", files=["gw2.json"])[0]

    assert purged == fetched.replace("cdn.jsdelivr.net", "purge.jsdelivr.net")


def test_purge_requests_every_url(monkeypatch):
    called = []
    monkeypatch.setattr(cdn, "_get", lambda url, timeout: called.append(url))

    ok = cdn.purge(repo="r/r", ref="refs/heads/v2", path="p", files=["a.json", "b.json"])

    assert ok is True
    assert called == [
        "https://purge.jsdelivr.net/gh/r/r@refs/heads/v2/p/a.json",
        "https://purge.jsdelivr.net/gh/r/r@refs/heads/v2/p/b.json",
    ]


def test_purge_failure_is_reported_but_not_raised(monkeypatch):
    """The data is already pushed and correct by this point. A CDN that is
    down should cost the site freshness for a few hours, not fail the export
    and leave the operator thinking the week did not publish."""
    def boom(url, timeout):
        raise OSError("jsdelivr unreachable")

    monkeypatch.setattr(cdn, "_get", boom)

    assert cdn.purge(repo="r/r", ref="refs/heads/v2", path="p", files=["a.json"]) is False


def test_purge_continues_after_one_url_fails(monkeypatch):
    called = []

    def flaky(url, timeout):
        called.append(url)
        if url.endswith("a.json"):
            raise OSError("nope")

    monkeypatch.setattr(cdn, "_get", flaky)

    assert cdn.purge(repo="r/r", ref="refs/heads/v2", path="p",
                     files=["a.json", "b.json"]) is False
    assert len(called) == 2, "a failure on the first file must not skip the rest"


@pytest.mark.parametrize("ref", ["refs/heads/v2", "main"])
def test_purge_url_does_not_double_up_separators(ref):
    url = cdn.purge_urls(repo="r/r", ref=ref, path="data/simulations/",
                         files=["gw2.json"])[0]

    assert "//data" not in url.removeprefix("https://")
    assert url.endswith("/data/simulations/gw2.json")
