#!/usr/bin/env python
"""One-off feasibility probe for WhoScored's raw Opta event stream (NOT a
permanent ingest — just answers: can we reach it, and what does it contain).

Run with a browser layered on, same pattern as the FBref scrapes:
    uv run --with soccerdata python scripts/probe_whoscored.py

Env overrides (same convention as scrape_fbref.py):
    WHOSCORED_BROWSER=/usr/bin/chromium   # path to Chrome/Chromium binary
    WHOSCORED_HEADED=1                    # 1 = show the browser window
"""
import os
import sys

import soccerdata as sd

browser = os.environ.get("WHOSCORED_BROWSER")
headless = os.environ.get("WHOSCORED_HEADED", "1") != "1"

ws_kwargs = {"leagues": "ENG-Premier League", "seasons": "2025-2026", "headless": headless}
if browser:
    ws_kwargs["path_to_browser"] = browser

print(f"Connecting to WhoScored (headless={headless}, browser={browser or 'auto-detect'})...")
ws = sd.WhoScored(**ws_kwargs)

print("Fetching schedule...")
schedule = ws.read_schedule()
print(f"{len(schedule)} games found in the schedule.")
played = schedule.reset_index()
game_id = int(played.iloc[0]["game_id"])
print(f"Probing one match, game_id={game_id}...")

events = ws.read_events(match_id=game_id, output_fmt="events")
print(f"\n{len(events)} events retrieved for this match.")
print("\nTop 30 event types by count:")
print(events["type"].value_counts().head(30).to_string())

print("\nSample rows for a few key types (if present):")
for t in ("Tackle", "Clearance", "BallRecovery", "Interception", "BlockedPass", "Foul"):
    subset = events[events["type"] == t]
    if not subset.empty:
        print(f"\n--- {t} ({len(subset)} rows) ---")
        print(subset[["minute", "player", "team", "outcome_type"]].head(3).to_string())
    else:
        print(f"\n--- {t}: NONE FOUND ---")

print("\nSample qualifiers from a random pass event (to check for KeyPass/Cross flags):")
pass_events = events[events["type"] == "Pass"]
if not pass_events.empty:
    for q in pass_events["qualifiers"].head(5):
        print(q)

if len(sys.argv) > 1 and sys.argv[1] == "--save-csv":
    events.to_csv("/tmp/whoscored_probe_events.csv", index=False)
    print("\nSaved full event dump to /tmp/whoscored_probe_events.csv")
