#!/usr/bin/env python
"""generate_gw_reminders_ical.py — one calendar reminder per real GW deadline.

The bot is kicked off manually (plan/manual-weekly-kickoff, 2026-07-31: no
systemd timer since the host PC isn't always on) — the user runs
`scripts/run_weekly.py --live` themselves ahead of each deadline. This
generates .ics reminders, one VALARM'd event per gameweek, scheduled a
lead time before the REAL deadline (read from the gameweeks table, which
must already be synced — run scripts/run_agent.py or any ingest first).

Two output modes:
- Combined (default): one .ics with all 38 events. Works with calendar
  apps that support multi-event import.
- Split (--split-dir): one .ics per gameweek, each a complete single-event
  calendar. Needed for apps that only support single-event .ics import
  (confirmed 2026-08-01: OneCalendar rejects multi-event .ics files).

Usage:
    uv run python scripts/generate_gw_reminders_ical.py [--season 2026-27]
        [--lead-hours 24] [--out fpl_gw_reminders.ics]
    uv run python scripts/generate_gw_reminders_ical.py --split-dir fpl_gw_reminders
        [--zip fpl_gw_reminders.zip]
"""

from __future__ import annotations

import argparse
import sys
import uuid
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from data.db import get_session

_DT_FMT = "%Y%m%dT%H%M%SZ"

_CALENDAR_HEADER = [
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "PRODID:-//fpl-26-27-bot//gw-reminders//EN",
    "CALSCALE:GREGORIAN",
]


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


def _vevent_lines(
    gw_id: int, name: str, deadline: datetime, lead_hours: float, season: str
) -> list[str]:
    now = datetime.now(UTC).strftime(_DT_FMT)
    start = deadline - timedelta(hours=lead_hours)
    end = start + timedelta(minutes=30)
    gw_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{season}-gw{gw_id}")
    uid = f"fpl-{season}-gw{gw_id}-{gw_uuid}@fpl-bot"
    deadline_str = deadline.strftime("%a %d %b %Y, %H:%M UTC")
    return [
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


def build_ical(
    gameweeks: list[tuple[int, str, datetime]], lead_hours: float, season: str
) -> str:
    """Combined .ics: all gameweeks as separate VEVENTs in one calendar.
    Only works with calendar apps that support multi-event .ics import."""
    lines = [*_CALENDAR_HEADER, f"X-WR-CALNAME:FPL Bot - run before deadline ({season})"]
    for gw_id, name, deadline in gameweeks:
        lines += _vevent_lines(gw_id, name, deadline, lead_hours, season)
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def build_single_ical(
    gw_id: int, name: str, deadline: datetime, lead_hours: float, season: str
) -> str:
    """One complete calendar containing exactly one VEVENT -- needed for
    calendar apps that reject multi-event .ics import (OneCalendar,
    confirmed 2026-08-01)."""
    lines = [*_CALENDAR_HEADER, f"X-WR-CALNAME:{name} reminder"]
    lines += _vevent_lines(gw_id, name, deadline, lead_hours, season)
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument(
        "--lead-hours", type=float, default=24.0,
        help="How long before each real deadline the reminder fires (default: 24h)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("fpl_gw_reminders.ics"),
        help="Combined output path (ignored if --split-dir is given)",
    )
    parser.add_argument(
        "--split-dir", type=Path, default=None,
        help="Write one single-event .ics per gameweek into this directory instead",
    )
    parser.add_argument(
        "--zip", type=Path, default=None,
        help="With --split-dir, also bundle the per-gameweek files into this zip",
    )
    args = parser.parse_args()

    gameweeks = load_gameweek_deadlines(args.season)
    if not gameweeks:
        raise SystemExit(
            f"No gameweeks found for season {args.season} -- sync first "
            "(e.g. run scripts/run_agent.py once)."
        )

    if args.split_dir is not None:
        args.split_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for gw_id, name, deadline in gameweeks:
            ical = build_single_ical(gw_id, name, deadline, args.lead_hours, args.season)
            path = args.split_dir / f"gw{gw_id:02d}_reminder.ics"
            path.write_text(ical, newline="")
            paths.append(path)
        print(f"Wrote {len(paths)} single-event reminders to {args.split_dir}/")
        if args.zip is not None:
            with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in paths:
                    zf.write(path, arcname=path.name)
            print(f"Zipped into {args.zip}")
    else:
        ical = build_ical(gameweeks, args.lead_hours, args.season)
        args.out.write_text(ical, newline="")
        print(f"Wrote {len(gameweeks)} reminders to {args.out}")


if __name__ == "__main__":
    main()
