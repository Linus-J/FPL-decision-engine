#!/usr/bin/env python
"""Replay one gameweek's decision against the data as it stood at its deadline.

Rewinds a copy of the live database to the deadline (``data/replay.py``),
verifies the copy holds nothing from after it, then runs the decision engine on
the copy in a child process. The child points at the copy through ``DB_PATH``
and sets ``FPL_REPLAY=1``, so nothing it does reaches the live database and no
live FPL call returns the present in place of the past.

Usage:
    uv run python scripts/replay_gameweek.py --season 2026-27 --gw 6

The replay is written to ``archive/replays/<season>-gw<N>.db`` and its decision
to ``<same stem>.json``. The report compares the replayed squad with the squad
the bot logged before that deadline, which is the nearest thing to a control.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.replay import REPLAY_ENV, rewind_database, verify_rewind  # noqa: E402

REPLAY_DIR = REPO_ROOT / "archive" / "replays"


def _live_db_path() -> Path:
    from config.settings import settings
    from data.db import _resolve_db_path

    return Path(_resolve_db_path(settings.db_path))


def _logged_squad_before_deadline(live_db: Path, season: str, gw: int, deadline: str):
    """The lineup the bot logged for ``gw`` before its deadline, from the LIVE log.

    Read-only. This is what the engine actually recommended at the time, so the
    replay can be compared with it.
    """
    import sqlite3

    con = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT details, created_at FROM decision_log "
            "WHERE decision_type = 'lineup' AND gameweek = ? AND created_at < ? "
            "ORDER BY created_at DESC LIMIT 1",
            (gw, deadline),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    details = json.loads(row[0])
    return {
        "squad_ids": sorted(details["squad_ids"]),
        "starting_ids": sorted(details["starting_ids"]),
        "captain_id": details["captain_id"],
        "logged_at": row[1],
    }


def _apply_overrides(pairs: list[str]) -> dict:
    """Replace OPTIMISER fields for this replay only. Returns what was changed.

    The engine reads OPTIMISER at call time, so swapping the module attribute is
    enough. Nothing is written back to config/strategy.py.
    """
    import dataclasses

    import agent.decision_engine as decision_engine
    from config.strategy import OPTIMISER

    updates = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        current = getattr(OPTIMISER, key, None)
        if current is None or isinstance(current, (tuple, list, dict)):
            raise SystemExit(f"--set {pair!r}: not a scalar OPTIMISER field")
        if isinstance(current, bool):
            updates[key] = raw.lower() in ("1", "true", "yes")
        else:
            updates[key] = type(current)(raw)
    if updates:
        decision_engine.OPTIMISER = dataclasses.replace(OPTIMISER, **updates)
    return updates


def _child(args: argparse.Namespace) -> None:
    """Runs inside the replay process: DB_PATH and FPL_REPLAY are already set."""
    import dataclasses

    import agent.decision_engine as decision_engine
    from data.ingestors.injury_parser import run_injury_parser
    from data.run_manifest import manifest_context, record_run, strategy_config

    # The injury severity is a function of the rewound news text, so it is
    # recomputed here exactly as the live run does, rather than read from a copy
    # that may have been written later.
    run_injury_parser()

    overrides = _apply_overrides(args.set or [])
    config = strategy_config()
    config["OPTIMISER"] = dataclasses.asdict(decision_engine.OPTIMISER)

    manifest_id = record_run(
        "replay",
        season=args.season,
        gameweek=args.gw,
        archive_path=None,
        output_path=Path(args.json_out),
        config=config,
        notes=(
            f"rewound copy of the live database at the GW{args.gw} deadline"
            + (f"; overrides {overrides}" if overrides else "")
        ),
    )
    with manifest_context(manifest_id):
        decision = decision_engine.run(season=args.season)

    if "error" in decision:
        raise SystemExit(f"replay decision failed: {decision['error']}")

    squad = decision["squad"]
    # The engine's squad records carry the player id under "id".
    squad_ids = [int(p["id"]) for p in squad]
    out = {
        "manifest_id": manifest_id,
        "overrides": overrides,
        "squad_ids": sorted(squad_ids),
        "captain_id": decision["captain_id"],
        "vice_captain_id": decision["vice_captain_id"],
        "chip": decision.get("chip"),
        "transfers_in": decision.get("transfers_in", []),
        "transfers_out": decision.get("transfers_out", []),
    }
    Path(args.json_out).write_text(json.dumps(out, indent=2, default=str) + "\n")


def _report(replay: dict, logged: dict | None) -> str:
    lines = [
        f"replayed squad: {len(replay['squad_ids'])} players, captain {replay['captain_id']}",
        f"chip: {replay['chip']}; transfers in: {replay['transfers_in']}",
    ]
    if logged is None:
        lines.append("no lineup was logged for this gameweek before its deadline")
        return "\n".join(lines)
    same_squad = set(replay["squad_ids"]) & set(logged["squad_ids"])
    lines += [
        f"logged at {logged['logged_at']}: captain {logged['captain_id']}",
        f"squad overlap with the logged decision: {len(same_squad)}/15",
        f"captain agrees: {replay['captain_id'] == logged['captain_id']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument(
        "--set", action="append", default=[], metavar="FIELD=VALUE",
        help="override one OPTIMISER field for this replay, e.g. --set risk_level=0.5",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--db", help=argparse.SUPPRESS)
    parser.add_argument("--json-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        _child(args)
        return

    stem = f"{args.season}-gw{args.gw}"
    if args.set:
        stem += "-" + "-".join(pair.replace("=", "") for pair in args.set)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    db_path = REPLAY_DIR / f"{stem}.db"
    json_out = REPLAY_DIR / f"{stem}.json"
    # A replay is a derived file, so the previous attempt at this gameweek is
    # replaced. The live database is never in this directory.
    db_path.unlink(missing_ok=True)
    json_out.unlink(missing_ok=True)

    live = _live_db_path()
    counts = rewind_database(live, db_path, args.season, args.gw)
    violations = verify_rewind(db_path, args.season, args.gw)
    if violations:
        db_path.unlink(missing_ok=True)
        raise SystemExit("rewind leaked post-deadline data; replay discarded:\n  "
                         + "\n  ".join(violations))

    import sqlite3

    con = sqlite3.connect(db_path)
    deadline = con.execute(
        "SELECT deadline_time FROM gameweeks WHERE season = ? AND id = ?",
        (args.season, args.gw),
    ).fetchone()[0]
    con.close()

    env = {**os.environ, "DB_PATH": str(db_path), REPLAY_ENV: "1"}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child",
         "--season", args.season, "--gw", str(args.gw),
         "--json-out", str(json_out),
         *[item for pair in args.set for item in ("--set", pair)]],
        env=env, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(f"replay child exited with {result.returncode}")

    replay = json.loads(json_out.read_text())
    logged = _logged_squad_before_deadline(live, args.season, args.gw, deadline)
    changed = sum(v for k, v in counts.items() if not k.startswith("coverage_"))
    print(f"rewound to {deadline} ({changed} rows changed)")
    for key, label in (
        ("coverage_player_state", "player state"),
        ("coverage_team_strength", "team strength"),
        ("coverage_setpieces", "set pieces"),
    ):
        if not counts[key]:
            print(f"WARNING: no {label} snapshots before the deadline; "
                  f"those values were kept as the current table holds them")
    print(_report(replay, logged))
    print(f"replay database: {db_path}")
    print(f"decision: {json_out}")
    print(f"ran at {datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()
