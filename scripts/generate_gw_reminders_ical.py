#!/usr/bin/env python
"""generate_gw_reminders_ical.py — one calendar reminder per real GW deadline.

The bot is kicked off manually (plan/manual-weekly-kickoff, 2026-07-31: no
systemd timer since the host PC isn't always on) — the user runs
`scripts/run_weekly.py --live` themselves ahead of each deadline. This
generates a .ics file with one VALARM'd event per gameweek, scheduled a
lead time before the REAL deadline (read from the gameweeks table, which
must already be synced — run scripts/run_agent.py or any ingest first).

Usage:
    uv run python scripts/generate_gw_reminders_ical.py [--season 2026-27]
        [--lead-hours 24] [--out fpl_gw_reminders.ics]
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from data.db import get_session

_DT_FMT = "%Y%m%dT%H%M%SZ"


def _fold(line: str) -> str:
    """RFC 5545 line folding: continuation lines start with a single space,
    max 75 octets per physical line (ASCII-only content here, so len==octets)."""
    if len(line) <= 75:
        return line
    parts = [line[:75]]
    rest = line[75:]
    while rest:
        parts.append(" " + rest[:74])
        rest = rest[74:]
    return "\r\n".join(parts)


def load_gameweek_deadlines(season: str) -> list[tuple[int, str, datetime]]:
    db = get_session()
    try:
        rows = db.execute(
            text("""
                SELECT id, name, deadline_time FROM gameweeks
                WHERE season = :season ORDER BY id
            """),
            {"season": season},
        ).fetchall()
    finally:
        db.close()
    return [
        (r.id, r.name, datetime.fromisoformat(r.deadline_time).replace(tzinfo=UTC))
        for r in rows
    ]


def build_ical(
    gameweeks: list[tuple[int, str, datetime]], lead_hours: float, season: str
) -> str:
    now = datetime.now(UTC).strftime(_DT_FMT)
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//fpl-26-27-bot//gw-reminders//EN",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:FPL Bot - run before deadline ({season})",
    ]
    for gw_id, name, deadline in gameweeks:
        start = deadline - timedelta(hours=lead_hours)
        end = start + timedelta(minutes=30)
        gw_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{season}-gw{gw_id}")
        uid = f"fpl-{season}-gw{gw_id}-{gw_uuid}@fpl-bot"
        deadline_str = deadline.strftime("%a %d %b %Y, %H:%M UTC")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now}",
            f"DTSTART:{start.strftime(_DT_FMT)}",
            f"DTEND:{end.strftime(_DT_FMT)}",
            _fold(f"SUMMARY:Run FPL bot - {name} deadline in {lead_hours:g}h"),
            _fold(
                "DESCRIPTION:Deadline: " + deadline_str
                + r"\n\nRun: uv run python scripts/run_weekly.py --live"
                + r"\n(dry-run first if unsure: uv run python scripts/run_weekly.py)"
            ),
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            _fold(f"DESCRIPTION:Run the FPL bot - {name} deadline in {lead_hours:g}h"),
            "TRIGGER:PT0M",
            "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument(
        "--lead-hours", type=float, default=24.0,
        help="How long before each real deadline the reminder fires (default: 24h)",
    )
    parser.add_argument("--out", type=Path, default=Path("fpl_gw_reminders.ics"))
    args = parser.parse_args()

    gameweeks = load_gameweek_deadlines(args.season)
    if not gameweeks:
        raise SystemExit(
            f"No gameweeks found for season {args.season} -- sync first "
            "(e.g. run scripts/run_agent.py once)."
        )
    ical = build_ical(gameweeks, args.lead_hours, args.season)
    args.out.write_text(ical, newline="")
    print(f"Wrote {len(gameweeks)} reminders to {args.out}")


if __name__ == "__main__":
    main()
