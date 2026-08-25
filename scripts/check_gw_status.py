#!/usr/bin/env python
"""check_gw_status.py -- can the weekly cycle score last gameweek yet?

    DB_PATH=fpl_bot_v2.db python scripts/check_gw_status.py [gw]

The scorer (``backfill_decision_outcomes._gw_finished``) requires FPL's
EVENT-level ``finished`` AND ``data_checked``, not merely that every fixture
has been played. Those flags lag the final whistle -- in 26/27 the gameweek
lockdown moved to 09:00 the day AFTER the last match, so there is a window
of twelve hours or more in which every match is over, bonus is applied, and
the gameweek is still not settled as far as the gate is concerned.

Sitting in that window looks identical to "something is broken" from the
outside. This says which it is: it reports the live flags, the fixture
state, and what is already in the local DB, then gives the verdict.

Read-only. Hits the public FPL API and opens the DB for reads only.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import urllib.request
from pathlib import Path

API = "https://fantasy.premierleague.com/api"


def _get(path: str):
    with urllib.request.urlopen(f"{API}/{path}", timeout=30) as r:
        return json.load(r)


def _tick(ok: bool) -> str:
    return "yes" if ok else "NO "


def main(argv: list[str]) -> int:
    boot = _get("bootstrap-static/")
    events = {e["id"]: e for e in boot["events"]}

    if len(argv) > 1:
        gw = int(argv[1])
    else:
        # The gameweek we would be scoring: FPL's current one. Before the
        # first deadline there is none, in which case nothing is scorable.
        cur = [e for e in boot["events"] if e["is_current"]]
        if not cur:
            print("No current gameweek -- the season has not started.")
            return 0
        gw = cur[0]["id"]

    ev = events[gw]
    print(f"=== GW{gw} ({ev['name']}) -- live FPL API ===")
    print(f"  deadline        {ev['deadline_time']}")
    print(f"  finished        {_tick(ev['finished'])}")
    print(f"  data_checked    {_tick(ev['data_checked'])}   <- the scorer's gate")
    print(f"  average score   {ev['average_entry_score']}")

    fixtures = _get(f"fixtures/?event={gw}")
    played = sum(1 for f in fixtures if f["finished"])
    prov = sum(1 for f in fixtures if f["finished_provisional"])
    print(f"  fixtures        {played}/{len(fixtures)} finished, "
          f"{prov}/{len(fixtures)} provisionally settled (bonus applied)")

    live = _get(f"event/{gw}/live/")["elements"]
    with_bonus = sum(1 for e in live if e["stats"]["bonus"] > 0)
    print(f"  bonus awarded   {with_bonus} players")

    # --- local DB ---------------------------------------------------------
    db_path = os.environ.get("DB_PATH", "fpl_bot_v2.db")
    season = os.environ.get("SEASON", "2026-27")
    print(f"\n=== local DB ({db_path}, season {season}) ===")
    if not Path(db_path).exists():
        print(f"  MISSING -- {db_path} does not exist")
        return 1

    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row = c.execute(
        "SELECT finished, data_checked FROM gameweeks WHERE season = ? AND id = ?",
        (season, gw),
    ).fetchone()
    if row is None:
        print(f"  gameweeks       no row for GW{gw} -- run an ingest")
        local_gate = False
    else:
        local_gate = bool(row[0] and row[1])
        print(f"  gameweeks       finished={_tick(bool(row[0]))} "
              f"data_checked={_tick(bool(row[1]))}")
        if (bool(row[0]), bool(row[1])) != (ev["finished"], ev["data_checked"]):
            print("                  STALE vs the live API -- re-ingest to refresh")

    stats = c.execute(
        "SELECT count(*) FROM player_gw_stats WHERE season = ? AND gameweek = ?",
        (season, gw),
    ).fetchone()[0]
    print(f"  player_gw_stats {stats} rows")

    scored = c.execute(
        "SELECT count(*), sum(actual_outcome IS NOT NULL) FROM decision_log "
        "WHERE gameweek = ? AND decision_type = 'lineup'",
        (gw,),
    ).fetchone()
    print(f"  decision_log    {scored[0]} lineup rows, {scored[1] or 0} scored")

    sim = c.execute(
        "SELECT count(*), sum(actual_outcome IS NOT NULL) FROM sim_decision_log "
        "WHERE gameweek = ? AND decision_type = 'lineup'",
        (gw,),
    ).fetchone()
    print(f"  sim_decision_log {sim[0]} lineup rows, {sim[1] or 0} scored")
    c.close()

    # --- verdict ----------------------------------------------------------
    print("\n=== verdict ===")
    if ev["finished"] and ev["data_checked"]:
        if local_gate:
            print("  SCORABLE. Run: uv run python scripts/run_weekly.py --dry-run")
        else:
            print("  FPL has settled the gameweek but the local DB has not caught")
            print("  up. run_weekly.py re-ingests first, so just run it:")
            print("    uv run python scripts/run_weekly.py --dry-run")
    else:
        print("  NOT YET SCORABLE -- FPL has not set both flags.")
        if played == len(fixtures):
            print("  Every fixture is played and bonus is applied, so this is the")
            print("  normal post-match settling window, not a fault. Expect the")
            print("  flags around 09:00 the day after the last match.")
        else:
            print(f"  {len(fixtures) - played} fixture(s) still to play.")
        print("\n  You can still run the weekly cycle now: only the scoring step")
        print("  self-skips. The GW decision itself does not need these flags.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
