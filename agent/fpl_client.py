import asyncio
import logging
from typing import Any

import aiohttp

from config.settings import settings

logger = logging.getLogger(__name__)

LOGIN_URL = "https://users.premierleague.com/accounts/login/"
BASE_URL = "https://fantasy.premierleague.com/api"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://fantasy.premierleague.com/",
    "Origin": "https://fantasy.premierleague.com",
    "Content-Type": "application/x-www-form-urlencoded",
}


async def _login(session: aiohttp.ClientSession) -> None:
    payload = {
        "login": settings.fpl_email,
        "password": settings.fpl_password,
        "app": "plfpl-web",
        "redirect_uri": "https://fantasy.premierleague.com/",
    }
    async with session.post(LOGIN_URL, data=payload, headers=HEADERS) as resp:
        if resp.status not in (200, 302):
            raise RuntimeError(f"FPL login failed: HTTP {resp.status}")
        logger.info("Logged in to FPL as %s", settings.fpl_email)


async def _get_my_team(session: aiohttp.ClientSession) -> dict[str, Any]:
    url = f"{BASE_URL}/my-team/{settings.fpl_team_id}/"
    async with session.get(url, headers=HEADERS) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _submit_transfers(
    session: aiohttp.ClientSession,
    transfers_in: list[dict],
    transfers_out: list[dict],
    chip: str | None,
    free_transfers: int,
    wildcard_active: bool,
) -> dict[str, Any]:
    if not transfers_in and not chip:
        logger.info("No transfers to submit")
        return {}

    entry_id = settings.fpl_team_id
    current_team = await _get_my_team(session)

    transfer_payload: list[dict] = []
    for t_in, t_out in zip(transfers_in, transfers_out):
        in_player = t_in["player_id"]
        out_player = t_out["player_id"]
        selling_price = next(
            (p["selling_price"] for p in current_team.get("picks", []) if p["element"] == out_player),
            int(t_out.get("cost", 0) * 10),
        )
        purchase_price = int(t_in.get("cost", 0) * 10)
        transfer_payload.append({
            "element_in": in_player,
            "element_out": out_player,
            "purchase_price": purchase_price,
            "selling_price": selling_price,
        })

    payload = {
        "confirmed": True,
        "entry": entry_id,
        "event": current_team.get("helper", {}).get("next_gw", 1),
        "transfers": transfer_payload,
        "wildcard": wildcard_active,
        "freehit": chip == "freehit",
    }

    url = f"{BASE_URL}/transfers/"
    async with session.post(url, json=payload, headers={**HEADERS, "Content-Type": "application/json"}) as resp:
        if resp.status not in (200, 201):
            body = await resp.text()
            raise RuntimeError(f"Transfer submission failed: HTTP {resp.status} — {body}")
        result = await resp.json()
        logger.info("Transfers submitted successfully")
        return result


_STARTING_POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _build_picks(squad: list[dict], captain_id: int, vice_captain_id: int) -> list[dict]:
    """FPL's picks payload requires each of the 15 ``position`` values to be
    a UNIQUE integer 1-15 (1 = the starting GK; the API rejects duplicates).
    Real bug found 2026-08-01 (user's own repo-cleanup request): the old
    per-player helper always returned a fixed slot per position (e.g. every
    starting DEF got ``position: 2``), so any squad with more than one
    starter in a position would submit duplicate values -- untested and
    never exercised live (dry-run only so far).

    Starters are ordered GKP/DEF/MID/FWD then assigned 1..11 sequentially;
    bench players are ordered by their own pre-computed ``bench_order``
    (GK bench = 0, remaining 3 outfield by xPts descending -- see
    optimiser/squad.py) and assigned 12..15."""
    starters = sorted(
        (p for p in squad if p.get("is_starting")),
        key=lambda p: _STARTING_POSITION_ORDER.get(p["position"], 99),
    )
    bench = sorted(
        (p for p in squad if not p.get("is_starting")),
        key=lambda p: p.get("bench_order", 99),
    )

    picks = []
    for slot, player in enumerate(starters + bench, start=1):
        pid = player["id"]
        picks.append({
            "element": pid,
            "position": slot,
            "is_captain": pid == captain_id,
            "is_vice_captain": pid == vice_captain_id,
        })
    return picks


async def _submit_lineup(
    session: aiohttp.ClientSession,
    squad: list[dict],
    captain_id: int,
    vice_captain_id: int,
    chip: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"picks": _build_picks(squad, captain_id, vice_captain_id)}
    if chip:
        payload["chip"] = chip

    url = f"{BASE_URL}/my-team/{settings.fpl_team_id}/"
    async with session.patch(url, json=payload, headers={**HEADERS, "Content-Type": "application/json"}) as resp:
        if resp.status not in (200, 201):
            body = await resp.text()
            raise RuntimeError(f"Lineup submission failed: HTTP {resp.status} — {body}")
        result = await resp.json()
        logger.info("Lineup submitted: captain=%d vc=%d chip=%s", captain_id, vice_captain_id, chip)
        return result


async def submit_decisions(
    squad: list[dict],
    captain_id: int,
    vice_captain_id: int,
    transfers_in: list[dict],
    transfers_out: list[dict],
    hits_taken: int,
    chip: str | None,
    free_transfers: int,
    dry_run: bool = True,
) -> dict[str, Any]:
    if dry_run:
        logger.info(
            "[DRY RUN] Would submit: %d transfers, chip=%s, captain=%d",
            len(transfers_in), chip, captain_id,
        )
        return {
            "dry_run": True,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "captain_id": captain_id,
            "vice_captain_id": vice_captain_id,
            "chip": chip,
        }

    if not settings.fpl_email or not settings.fpl_password:
        raise RuntimeError("FPL_EMAIL and FPL_PASSWORD must be set for live submission")

    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar()) as session:
        await _login(session)

        wildcard_active = chip == "wildcard"

        if transfers_in:
            await _submit_transfers(
                session,
                transfers_in=transfers_in,
                transfers_out=transfers_out,
                chip=chip,
                free_transfers=free_transfers,
                wildcard_active=wildcard_active,
            )

        result = await _submit_lineup(
            session,
            squad=squad,
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            chip=chip if not transfers_in else None,
        )

    return result


def submit(
    squad: list[dict],
    captain_id: int,
    vice_captain_id: int,
    transfers_in: list[dict],
    transfers_out: list[dict],
    hits_taken: int,
    chip: str | None,
    free_transfers: int,
    dry_run: bool = True,
) -> dict[str, Any]:
    return asyncio.run(
        submit_decisions(
            squad=squad,
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            transfers_in=transfers_in,
            transfers_out=transfers_out,
            hits_taken=hits_taken,
            chip=chip,
            free_transfers=free_transfers,
            dry_run=dry_run,
        )
    )
