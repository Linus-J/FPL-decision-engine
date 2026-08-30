"""jsDelivr cache invalidation for the exported site data.

The portfolio site fetches the JSON through jsDelivr pinned to a mutable
branch ref, and jsDelivr caches a branch ref for up to seven days
(``cache-control: max-age=604800, s-maxage=43200``). Between 25 and 30
August 2026 that meant the site served a five-day-old ``gw2.json`` -- still
listing the two players the GW2 transfers had already moved out -- while
``origin/v2`` carried the corrected squad the whole time. Pushing publishes
the data; only a purge makes the site see it.
"""

from __future__ import annotations

import logging
import urllib.request

PURGE_ORIGIN = "https://purge.jsdelivr.net/gh"
_TIMEOUT_SECONDS = 15

logger = logging.getLogger(__name__)


def purge_urls(*, repo: str, ref: str, path: str, files: list[str]) -> list[str]:
    """The purge endpoint for each file, in the given order.

    The purge key is the request URL, so this must spell the repo and ref
    exactly as the site does in ``assets/stats-panel.js``. Purging any other
    spelling -- the repo's pre-rename name, a bare branch instead of the
    fully-qualified ref -- clears a key nobody requests and leaves the stale
    entry serving.
    """
    prefix = f"{PURGE_ORIGIN}/{repo}@{ref}/{path.strip('/')}"
    return [f"{prefix}/{file}" for file in files]


def _get(url: str, timeout: int) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        response.read()


def purge(*, repo: str, ref: str, path: str, files: list[str]) -> bool:
    """Ask jsDelivr to drop its cached copy of each file. True if every
    request succeeded.

    Best-effort by design: the data is already pushed and correct by the
    time this runs, so a jsDelivr outage should cost the site its freshness
    for a few hours rather than fail the export and leave the operator
    thinking the week never published. One failure does not skip the rest --
    a stale ``index.json`` and a stale ``gw2.json`` are separate problems.
    """
    ok = True
    for url in purge_urls(repo=repo, ref=ref, path=path, files=files):
        try:
            _get(url, _TIMEOUT_SECONDS)
            logger.info("Purged %s", url)
        except (OSError, ValueError) as exc:
            logger.warning("Could not purge %s: %s", url, exc)
            ok = False
    return ok
