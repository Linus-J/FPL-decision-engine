#!/usr/bin/env python
"""sync_played_team.py -- record the team FPL says was played, when it differs
from the bot's last logged decision for that gameweek.

    DB_PATH=fpl_bot_v2.db uv run python scripts/sync_played_team.py [gw] [--apply]

``_load_squad_state`` takes the NEWEST lineup row in ``decision_log`` as next
week's squad, bank and purchase prices. That row is whatever the bot last
recommended -- not what was submitted. Found 2026-10-10 (GW6): the manager
started Joao Pedro over Thomas, and a later rerun had logged a squad with two
players that were never bought, so GW7 would have planned transfers out of a
team the entry did not own.

Run after the deadline (FPL only publishes picks then). Default is a dry run
that prints the differences; ``--apply`` appends one ``transfers`` and one
``lineup`` row built from FPL's picks, marked ``"source": "fpl_actual"``.
Nothing is updated or deleted, so removing the two rows undoes it. When FPL
already matches the bot's last row, nothing is written.

``projected_gain`` is copied from the bot's row: it is the bot's estimate for
ITS lineup, and the script does not re-solve one for the played lineup.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from config.settings import settings  # noqa: E402
from config.strategy import OPTIMISER  # noqa: E402
from data.db import get_session  # noqa: E402
from optimiser.chips import Chip  # noqa: E402

API = "https://fantasy.premierleague.com/api"
_COMPARED = ("squad_ids", "starting_ids", "captain_id", "vice_captain_id",
             "bench_order", "chip_considered")


def _get(path: str):
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def played_team_rows(
    picks: dict,
    transfers: list[dict],
    positions: Mapping[int, str],
    prior_purchase_prices: Mapping[int, float],
    bot_lineup: dict,
) -> tuple[dict, dict] | None:
    """(transfers details, lineup details) for the team FPL says was played,
    or ``None`` when it already matches ``bot_lineup`` on everything
    ``_load_squad_state`` and the outcome scorer read.

    ``picks`` is ``/entry/{id}/event/{gw}/picks/``; ``transfers`` that
    gameweek's entries from ``/entry/{id}/transfers/``. Both must already be
    mapped from FPL element ids to internal player ids.
    """
    ordered = sorted(picks["picks"], key=lambda p: p["position"])
    squad = [int(p["element"]) for p in ordered]
    chip = picks.get("active_chip")
    hits = int(picks["entry_history"]["event_transfers_cost"]) // 4

    played = {
        "squad_ids": sorted(squad),
        "starting_ids": sorted(squad[:11]),
        "captain_id": next(int(p["element"]) for p in ordered if p["is_captain"]),
        "vice_captain_id": next(int(p["element"]) for p in ordered if p["is_vice_captain"]),
        # Pick 12 is the reserve keeper (slot 0), 13-15 the outfield queue --
        # the same encoding as optimise_starting_xi's bench_order.
        "bench_order": {str(pid): i for i, pid in enumerate(squad[11:])},
        "chip_considered": chip,
    }
    if all(_same(played[k], bot_lineup.get(k)) for k in _COMPARED):
        return None

    prices = {int(k): float(v) for k, v in prior_purchase_prices.items()}
    if chip != Chip.FREE_HIT.value:
        # Bought players at what FPL charged; sold players dropped. A Free Hit
        # squad is handed back, so the owned squad's prices stay as they were.
        for t in transfers:
            prices.pop(int(t["element_out"]), None)
            prices[int(t["element_in"])] = t["element_in_cost"] / 10
        prices = {pid: p for pid, p in prices.items() if pid in squad}

    def _move(pid: int, cost: float) -> dict:
        return {"player_id": pid, "cost": cost, "position": positions.get(pid)}

    transfers_details = {
        "transfers_in": [_move(int(t["element_in"]), t["element_in_cost"] / 10)
                         for t in transfers],
        "transfers_out": [_move(int(t["element_out"]), t["element_out_cost"] / 10)
                          for t in transfers],
        "hits_taken": hits,
        "source": "fpl_actual",
    }
    lineup_details = {
        **bot_lineup,
        **played,
        "positions": {str(pid): positions.get(pid) for pid in squad},
        "hits_taken": hits,
        "bank": picks["entry_history"]["bank"] / 10,
        "purchase_prices": {str(pid): p for pid, p in prices.items()},
        "source": "fpl_actual",
    }
    return transfers_details, lineup_details


def _same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return sorted(a) == sorted(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return {str(k): v for k, v in a.items()} == {str(k): v for k, v in b.items()}
    return a == b


def _latest(db, gw: int, kind: str) -> tuple[dict, float] | None:
    row = db.execute(text(
        "SELECT details, projected_gain FROM decision_log "
        "WHERE gameweek = :gw AND decision_type = :kind ORDER BY created_at DESC LIMIT 1"
    ), {"gw": gw, "kind": kind}).fetchone()
    return (json.loads(row[0]), float(row[1])) if row else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("gw", nargs="?", type=int,
                        help="gameweek (default: FPL's current one)")
    parser.add_argument("--apply", action="store_true", help="append the rows")
    args = parser.parse_args()

    from agent.decision_engine import _load_squad_state

    entry = settings.fpl_team_id
    if not entry:
        print("FPL_TEAM_ID is unset; nothing to read.")
        return 1
    boot = _get("bootstrap-static/")
    gw = args.gw or next(e["id"] for e in boot["events"] if e["is_current"])
    code_of = {e["id"]: e["code"] for e in boot["elements"]}

    db = get_session()
    try:
        # Join on players.code, never fpl_id: element ids are reissued per
        # season (see players.code).
        by_code = {int(c): (int(i), p) for i, c, p in
                   db.execute(text("SELECT id, code, position FROM players")).fetchall()}

        def to_id(element: int) -> int:
            return by_code[code_of[element]][0]

        picks = _get(f"entry/{entry}/event/{gw}/picks/")
        for p in picks["picks"]:
            p["element"] = to_id(p["element"])
        transfers = [t for t in _get(f"entry/{entry}/transfers/") if t["event"] == gw]
        for t in transfers:
            t["element_in"], t["element_out"] = to_id(t["element_in"]), to_id(t["element_out"])

        bot = _latest(db, gw, "lineup")
        if bot is None:
            print(f"No GW{gw} lineup in decision_log -- nothing to compare against.")
            return 1
        bot_lineup, lineup_gain = bot
        bot_t = _latest(db, gw, "transfers")
        prior = _load_squad_state(None, entry, OPTIMISER, decided_gw=gw)

        rows = played_team_rows(
            picks, transfers, {i: p for i, p in by_code.values()},
            prior.purchase_prices, bot_lineup,
        )
        if rows is None:
            print(f"GW{gw}: FPL matches the bot's last logged lineup. Nothing to do.")
            return 0
        transfers_details, lineup_details = rows
        names = dict(db.execute(text("SELECT id, web_name FROM players")).fetchall())
        print(f"GW{gw}: FPL differs from the bot's last logged lineup:")
        for k in _COMPARED:
            if not _same(lineup_details[k], bot_lineup.get(k)):
                print(f"  {k}:\n    bot    {_named(bot_lineup.get(k), names)}"
                      f"\n    played {_named(lineup_details[k], names)}")
        moved = {side: [names.get(m["player_id"]) for m in transfers_details[side]]
                 for side in ("transfers_in", "transfers_out")}
        print(f"  transfers played: in {moved['transfers_in']} out {moved['transfers_out']}"
              f", hits {transfers_details['hits_taken']}")
        print(f"  bank {lineup_details['bank']}")
        if not args.apply:
            print("Dry run. Re-run with --apply to append the played team.")
            return 0

        now = datetime.utcnow()
        for kind, details, gain in (
            ("transfers", transfers_details, bot_t[1] if bot_t else 0.0),
            ("lineup", lineup_details, lineup_gain),
        ):
            db.execute(text(
                "INSERT INTO decision_log (gameweek, decision_type, details, "
                "projected_gain, actual_outcome, dry_run, created_at) "
                "VALUES (:gw, :kind, :details, :gain, NULL, 1, :now)"
            ), {"gw": gw, "kind": kind, "details": json.dumps(details),
                "gain": gain, "now": now})
        db.commit()
        print(f"Appended the played GW{gw} team to decision_log.")
        return 0
    finally:
        db.close()


def _named(value, names: Mapping[int, str]):
    if isinstance(value, list):
        return sorted(names.get(int(v), v) for v in value)
    if isinstance(value, dict):
        return [names.get(int(k), k) for k, _ in sorted(value.items(), key=lambda kv: kv[1])]
    if isinstance(value, int):
        return names.get(value, value)
    return value


if __name__ == "__main__":
    sys.exit(main())
