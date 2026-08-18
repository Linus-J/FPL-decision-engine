"""team_sheet.py — turn a decision into the team sheet a human enters.

Replaces ``agent/fpl_client.py`` (removed 2026-08-18). That module existed to
log into the Premier League site and PUT transfers and a lineup; this project
no longer does that, and the capability is gone rather than disabled. There is
no flag to turn it back on, no credentials to leak, and nothing that can submit
a squad because a default was wrong somewhere.

What is left is the part that was always doing the useful work: putting fifteen
players in the order the entry form wants them.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_STARTING_POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def build_picks(squad: list[dict], captain_id: int, vice_captain_id: int) -> list[dict]:
    """The fifteen players in slot order, 1-15.

    Slot 1 is the starting goalkeeper, 2-11 the rest of the XI by position,
    and 12-15 the bench in the order it should be listed: reserve keeper
    first, then the three outfield substitutes by the priority
    ``optimiser/squad.py`` assigned them. That order is not cosmetic — it is
    the order automatic substitutions consult, so entering it wrongly quietly
    changes who comes on when somebody does not play.

    Real bug this shape exists to prevent (2026-08-01): the previous
    per-player helper returned a fixed slot per POSITION, so every starting
    defender got slot 2. Any squad with two starters in a position produced
    duplicate slots. It went unnoticed because nothing ever consumed the
    output for real.
    """
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


def build(
    squad: list[dict],
    captain_id: int,
    vice_captain_id: int,
    transfers_in: list[dict],
    transfers_out: list[dict],
    chip: str | None,
) -> dict[str, Any]:
    """The decision as a team sheet: what to enter, and what changed."""
    logger.info(
        "Team sheet ready: %d transfers, chip=%s, captain=%s",
        len(transfers_in), chip, captain_id,
    )
    return {
        "picks": build_picks(squad, captain_id, vice_captain_id),
        "transfers_in": transfers_in,
        "transfers_out": transfers_out,
        "captain_id": captain_id,
        "vice_captain_id": vice_captain_id,
        "chip": chip,
    }
