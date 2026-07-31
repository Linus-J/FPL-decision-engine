"""Thin client for FPL's public (no-auth) endpoints, used by the dashboard to
show the real live squad regardless of dry-run mode or manual in-app changes.
"""

import asyncio
import logging

import aiohttp

from data.ingestors.fpl_api import FPL_BASE

logger = logging.getLogger(__name__)


async def _fetch_picks(team_id: int, gameweek: int) -> dict:
    url = f"{FPL_BASE}/entry/{team_id}/event/{gameweek}/picks/"
    async with aiohttp.ClientSession() as session, session.get(url) as resp:
        resp.raise_for_status()
        return await resp.json()


def get_picks(team_id: int, gameweek: int) -> dict:
    """Returns the raw FPL picks payload (``{"picks": [...], "entry_history": {...}}``),
    or ``{}`` if the team/gameweek isn't available yet (e.g. pre-deadline, or
    the network call fails)."""
    try:
        return asyncio.run(_fetch_picks(team_id, gameweek))
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning(
            "FPL public picks fetch failed for team=%s gw=%s: %s", team_id, gameweek, exc
        )
        return {}
