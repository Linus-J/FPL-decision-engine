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


async def _submit_lineup(
    session: aiohttp.ClientSession,
    squad: list[dict],
    captain_id: int,
    vice_captain_id: int,
    chip: str | None,
) -> dict[str, Any]:
    picks = []
    bench_order = 12
    for player in squad:
        pid = player["id"]
        is_starting = player.get("is_starting", False)
        is_cap = pid == captain_id
        is_vc = pid == vice_captain_id

        if is_starting:
            position = _playing_position(player["position"], squad)
        else:
            position = bench_order
            bench_order += 1

        picks.append({
            "element": pid,
            "position": position,
            "is_captain": is_cap,
            "is_vice_captain": is_vc,
        })

    payload: dict[str, Any] = {"picks": picks}
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


def _playing_position(position: str, squad: list[dict]) -> int:
    order = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    starters = [p for p in squad if p.get("is_starting")]
    same_pos = [p for p in starters if p["position"] == position]
    idx = next(
        (i for i, p in enumerate(same_pos) if p.get("is_captain") or p.get("is_vice_captain") or True),
        0,
    )
    base = {"GKP": 1, "DEF": 2, "MID": 6, "FWD": 10}[position]
    return base


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
