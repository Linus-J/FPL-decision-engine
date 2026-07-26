"""ownership.py — effective ownership (EO) ingestion (P3-2, v2-build-plan §3.2).

Feeds Phase 3's rank-aware objective differential-value term: your_pts minus
EO-weighted field_pts, where the field that actually matters for RANK is the
top-10k (the players you're actually competing against), not the whole
~11M-manager player base.

Real overall ownership (``selected_by_percent``) already comes free from FPL's
bootstrap-static endpoint and is already ingested elsewhere (data/ingestors/
fpl_api.py) — this module adds the genuinely missing piece: top-10k ownership
+ captaincy, built by sampling the "Overall" classic league's (id 314)
standings for entry IDs, then those managers' picks for the target gameweek
(exactly the "aggregating a top-10k mini-league sample" the plan calls for).
``captaincy_pct_overall`` has no free population-wide source and is left
unpopulated — a documented gap, not a silent default.

⚠️ UNVERIFIED AGAINST LIVE DATA (authored 2026-07-26): the 2026-27 season has
zero played gameweeks — the Overall league has zero ranked entries and
standings/picks endpoints return empty right now, so this was built and
tested against the well-documented, stable (but currently unprobeable) shape
of FPL's public API, not a real populated response. Re-verify at GW1.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import aiohttp
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.ingestors.fpl_api import _get
from data.models import OwnershipSnapshot, Player

logger = logging.getLogger(__name__)

OVERALL_LEAGUE_ID = 314
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_CONCURRENCY = 10


def _extract_entries(standings_response: dict) -> tuple[list[int], bool]:
    """(entry_ids on this page, has_next) from a raw standings API response.
    Pure — the only part of the pagination loop worth unit-testing without
    a live/mocked HTTP session."""
    standings = standings_response.get("standings", {})
    results = standings.get("results", [])
    entries = [int(r["entry"]) for r in results]
    return entries, bool(standings.get("has_next"))


def aggregate_ownership(picks_by_entry: list[list[dict]]) -> dict[int, dict[str, float]]:
    """Per-player (keyed by FPL element id) ownership + captaincy % across a
    sample of managers' picks. Pure — no network/DB. ``picks_by_entry``: one
    list of pick-dicts (each ``{"element": ..., "is_captain": ..., ...}``)
    per successfully-fetched manager (callers filter out failed fetches
    before calling this)."""
    n = len(picks_by_entry)
    if n == 0:
        return {}
    owned_count: dict[int, int] = {}
    captain_count: dict[int, int] = {}
    for picks in picks_by_entry:
        for pick in picks:
            element = int(pick["element"])
            owned_count[element] = owned_count.get(element, 0) + 1
            if pick.get("is_captain"):
                captain_count[element] = captain_count.get(element, 0) + 1
    return {
        element: {
            "selected_pct": 100.0 * count / n,
            "captaincy_pct": 100.0 * captain_count.get(element, 0) / n,
        }
        for element, count in owned_count.items()
    }


async def fetch_standings_page(  # pragma: no cover - live network
    session: aiohttp.ClientSession, league_id: int, page: int
) -> dict:
    return await _get(session, f"/leagues-classic/{league_id}/standings/?page_standings={page}")


async def sample_top_entries(  # pragma: no cover - live network
    session: aiohttp.ClientSession, n: int, league_id: int = OVERALL_LEAGUE_ID
) -> list[int]:
    """Entry IDs of the top ``n`` managers in the classic league, paging
    through its standings (FPL's fixed 50/page). Stops early if the league
    runs out of pages before reaching ``n`` — e.g. pre-season, when zero
    gameweeks have been played and the league has no ranked entries at all
    (the exact situation as of this module's authoring)."""
    entries: list[int] = []
    page = 1
    while len(entries) < n:
        data = await fetch_standings_page(session, league_id, page)
        page_entries, has_next = _extract_entries(data)
        if not page_entries:
            break
        entries.extend(page_entries)
        if not has_next:
            break
        page += 1
    return entries[:n]


async def fetch_entry_picks(  # pragma: no cover - live network
    session: aiohttp.ClientSession, entry_id: int, gw: int
) -> dict | None:
    try:
        return await _get(session, f"/entry/{entry_id}/event/{gw}/picks/")
    except aiohttp.ClientResponseError as exc:
        # a private/deleted entry, or a manager who didn't set a team that
        # week -- skip rather than fail the whole sample over one entry
        logger.debug("Skipping entry %d GW%d: %s", entry_id, gw, exc)
        return None


async def ingest_ownership_snapshot(  # pragma: no cover - live network
    gw: int,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[int, int]:
    """Samples the top ``sample_size`` managers in the Overall classic league
    and their GW picks, aggregates top-N ownership/captaincy %, and writes
    one ``OwnershipSnapshot`` row per player (``overall_selected_pct`` from
    the already-ingested ``players`` table; ``top10k_*`` from this sample).
    Returns ``(players_written, entries_sampled)``.

    No ``season`` parameter: FPL's picks endpoint (``/entry/{id}/event/{gw}/
    picks/``) is scoped to the CURRENT live season only (confirmed by probing
    it directly — there is no way to address a past season's picks through
    this API), so "season" isn't a meaningful axis for this ingestor the way
    it is for the historical-backfill ingestors elsewhere in this codebase.
    """
    async with aiohttp.ClientSession() as http:
        entry_ids = await sample_top_entries(http, sample_size)
        if not entry_ids:
            logger.warning(
                "No ranked entries found in the Overall league (season not "
                "started yet?) — cannot sample top-%d EO for GW%d.",
                sample_size, gw,
            )
            return 0, 0

        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch(entry_id: int) -> list[dict] | None:
            async with semaphore:
                data = await fetch_entry_picks(http, entry_id, gw)
                return data.get("picks") if data else None

        results = await asyncio.gather(*(_fetch(e) for e in entry_ids))
        picks_by_entry = [p for p in results if p]

    if not picks_by_entry:
        logger.warning("No valid picks fetched for GW%d — writing nothing", gw)
        return 0, len(entry_ids)

    agg = aggregate_ownership(picks_by_entry)

    db = get_session()
    try:
        players = db.query(Player).all()
        fpl_to_db = {p.fpl_id: p.id for p in players}
        overall_pct = {p.fpl_id: p.selected_by_percent for p in players}

        snapshot_ts = datetime.utcnow()
        written = 0
        for fpl_id, stats in agg.items():
            player_id = fpl_to_db.get(fpl_id)
            if not player_id:
                continue
            stmt = insert(OwnershipSnapshot).values(
                player_id=player_id,
                snapshot_ts=snapshot_ts,
                overall_selected_pct=float(overall_pct.get(fpl_id, 0.0)),
                top10k_selected_pct=stats["selected_pct"],
                captaincy_pct_top10k=stats["captaincy_pct"],
                sample_size=len(picks_by_entry),
            ).on_conflict_do_nothing()
            db.execute(stmt)
            written += 1
        db.commit()
    finally:
        db.close()

    logger.info(
        "Ownership snapshot GW%d: %d players written from a %d-manager sample "
        "(%d entries attempted)",
        gw, written, len(picks_by_entry), len(entry_ids),
    )
    return written, len(entry_ids)
