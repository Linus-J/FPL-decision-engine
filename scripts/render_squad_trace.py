#!/usr/bin/env python
"""render_squad_trace.py — human-readable squad-evolution report (2026-07-28).

Runs run_backtest with `trace` enabled and renders a Markdown report of the
squad's full week-by-week evolution: transfers (named, with cost/xpts),
captain/vice, full 15-man squad with starting-XI/bench split, and actual vs
predicted points — so a real FPL manager can eyeball it for weird choices
that pure summary statistics (mean bias, correlation) don't surface.

    DB_PATH=fpl_bot_v2.db uv run python scripts/render_squad_trace.py \
        --season 2025-26 --start-gw 6 --end-gw 38 --out /tmp/squad_trace.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from scripts.backtest import run_backtest

POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _fmt_transfer_list(transfers: list[dict]) -> str:
    if not transfers:
        return "—"
    return ", ".join(f"{t['web_name']} (£{t['cost']:.1f}m)" for t in transfers)


def _squad_table(squad: list[dict]) -> str:
    rows = sorted(
        squad,
        key=lambda p: (
            p["bench_order"] == -1 and 0 or 1,  # starters first
            POSITION_ORDER.get(p["position"], 9),
            -p["xpts"],
        ),
    )
    lines = ["| | Pos | Player | Cost | xPts | Actual |", "|---|---|---|---|---|---|"]
    for p in rows:
        tag = ""
        if p["is_captain"]:
            tag = "(C)"
        elif p["is_vice_captain"]:
            tag = "(VC)"
        bench_marker = "🪑 " if p["bench_order"] != -1 else ""
        lines.append(
            f"| {bench_marker}| {p['position']} | {p['web_name']} {tag} | "
            f"£{p['now_cost']:.1f}m | {p['xpts']:.1f} | {p['actual_pts']} |"
        )
    return "\n".join(lines)


def render_markdown(trace: list[dict], season: str) -> str:
    out = [f"# Squad evolution trace — {season} (v2 bot, full decision engine)\n"]
    out.append(
        "Legend: **(C)** captain, **(VC)** vice-captain, 🪑 bench. "
        "\"xPts\" is this gameweek's projection for that player at decision time "
        "(after the optimiser's-curse shrinkage); \"Actual\" is real points scored "
        "(26/27-rescored bonus).\n"
    )

    out.append("## Summary\n")
    out.append(
        "| GW | Chip | Transfers in | Transfers out | Hits | Squad £ | Predicted | Actual | Net |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for row in trace:
        transfers_in = _fmt_transfer_list(row["transfers_in"])
        transfers_out = _fmt_transfer_list(row["transfers_out"])
        out.append(
            f"| {row['gameweek']} | {row['chip'] or '—'} | "
            f"{transfers_in} | {transfers_out} | "
            f"{row['hits']} | £{row['squad_cost']:.1f}m | {row['predicted_xpts']:.1f} | "
            f"{row['actual_pts']} | {row['net_pts']} |"
        )
    out.append("")

    out.append("## Gameweek-by-gameweek detail\n")
    for row in trace:
        gw = row["gameweek"]
        out.append(f"### GW{gw}" + (f" — {row['chip']}" if row["chip"] else ""))
        if gw == trace[0]["gameweek"]:
            out.append(f"Initial squad built (£{row['squad_cost']:.1f}m).\n")
        else:
            out.append(
                f"**In:** {_fmt_transfer_list(row['transfers_in'])}  \n"
                f"**Out:** {_fmt_transfer_list(row['transfers_out'])}  \n"
                f"**Hits taken:** {row['hits']}\n"
            )
        out.append(
            f"Predicted {row['predicted_xpts']:.1f} → "
            f"Actual {row['actual_pts']} (net {row['net_pts']})\n"
        )
        out.append(_squad_table(row["squad"]))
        out.append("")

    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--season", default="2025-26")
    p.add_argument("--start-gw", type=int, default=6)
    p.add_argument("--end-gw", type=int, default=38)
    p.add_argument("--out", default="squad_trace.md")
    args = p.parse_args()

    trace: list[dict] = []
    run_backtest(
        season=args.season, start_gw=args.start_gw, end_gw=args.end_gw,
        score_2627=True, trace=trace,
    )

    md = render_markdown(trace, args.season)
    Path(args.out).write_text(md)
    logging.getLogger(__name__).info("Wrote %d gameweeks to %s", len(trace), args.out)


if __name__ == "__main__":
    main()
