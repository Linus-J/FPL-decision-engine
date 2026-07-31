# FPL 2026/27 Autonomous Bot

A fully autonomous Fantasy Premier League agent. On each run it ingests live data, projects player points, optimises squad and transfers via integer linear programming, decides chip usage, submits decisions to the FPL API, and sends a Telegram summary — all without human input.

---

## Architecture

```
config/
  settings.py       — credentials and runtime flags (loaded from .env)
  strategy.py       — ALL season-variable rules: scoring, chips, squad structure, DGW params

data/
  models.py         — SQLAlchemy ORM: 11 tables including PlayerSetPieceRole, PlayerPressSignal
  db.py             — SQLite engine (WAL mode), session factory, init_db()
  ingestors/
    fpl_api.py      — async FPL bootstrap + fixtures + per-player GW history (incl. transfer counts)
    understat.py    — xG/xA via Understat AJAX API; derives set piece / penalty-taker roles
    odds_api.py     — The Odds API h2h odds → CS probabilities → fixture_odds table
    injury_parser.py — regex parser on players.news → injury_severity (0–3) stored on players table
    midweek.py      — derives midweek fixture flags from fixtures table (2 games in 7d with ≥3d gap)
    press_conference.py — Guardian API sentiment scraper → PlayerPressSignal table

projection/
  features.py       — FDR features (8), enrichment features (7), cross-season form decay features
  minutes_model.py  — GBT classifier: P(60+ minutes), calibrated; cross-season rolling features
  points_model.py   — GBT regressor: expected points; cross-season form decay ratio
  cs_model.py       — GBT classifier: P(clean sheet) per team per GW; CV Brier ≈ 0.113
  pipeline.py       — batch prediction orchestrator → player_projections table (deduped per run)

optimiser/
  squad.py          — PuLP ILP: 15-man squad picker + XI selector; bench ordered by priority
  transfers.py      — evaluates transfer scenarios (0..max_hits), picks best net xPts gain
  chips.py          — priority-ordered chip recommender; WC H1/H2 boundary from DB

agent/
  decision_engine.py — main orchestrator: ingest → project → chip → transfers → lineup → log
  fpl_client.py      — async FPL API client: login, transfer submission, lineup PATCH
  notifier.py        — Telegram notification: starting XI, bench in priority order, transfers

scripts/
  run_agent.py       — CLI entrypoint (--dry-run / --live / --chip / --json-out)
  train_models.py    — trains and saves all 3 ML models to models/
  backfill_history.py — one-time load of vaastav historical CSVs into DB
  backtest.py        — walk-forward backtester: retrain per GW, score actual decisions
  plot_analysis.py   — generates 5 analysis plots to results/plots/

deploy/
  fpl-bot.service   — systemd user service unit
  fpl-bot.timer     — runs 06:00 Fri/Sat/Sun (covers all FPL deadline days)
  install.sh        — symlinks units, enables timer
```

---

## Data Sources

| Source | What it provides | Key |
|---|---|---|
| FPL API | Players, fixtures, GW history, team strengths | Free |
| vaastav CSV archive | Historical GW stats 2021–25 (backfill) | Free |
| Understat AJAX API | xG, xA, npxG, shots, key passes per player per GW | Free |
| The Odds API | h2h odds → CS probability estimates | Free tier (500 req/month) |
| Guardian API | Press conference text → injury/availability signals | Free (`api-key=test`) |

---

## ML Models

| Model | Type | Target | CV score |
|---|---|---|---|
| Minutes model | GBT classifier (calibrated) | P(60+ minutes) | Brier 0.085 |
| Points model | GBT regressor | Expected points | MAE 2.00 |
| CS model | GBT classifier | P(clean sheet) | Brier 0.113 |

All three use 15 FDR features (fixture difficulty, home/away, attack/defence ratios) and 7 enrichment features (penalty taker, set piece taker, injury severity, press sentiment, price momentum). Cross-season form decay features (`avg_pts_5gw_global`, `form_decay_ratio`) prevent historically strong players like Salah from being over-projected during poor-form spells.

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set FPL_EMAIL, FPL_PASSWORD, FPL_TEAM_ID at minimum
```

| Variable | Required | Description |
|---|---|---|
| `FPL_EMAIL` | Live only | FPL account email |
| `FPL_PASSWORD` | Live only | FPL account password |
| `FPL_TEAM_ID` | Yes | Your FPL team ID (from the URL on the FPL site) |
| `THE_ODDS_API_KEY` | Optional | The Odds API key for CS probability estimates |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram bot token for notifications |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat ID to send notifications to |
| `DRY_RUN` | — | `true` (default) = no live submissions |
| `DB_PATH` | — | SQLite database path (default: `fpl_bot.db`) |

### 3. Initialise the database and backfill history

```bash
uv run python -c "from data.db import init_db; init_db()"
uv run python scripts/backfill_history.py
```

This loads vaastav historical CSVs (2021–25) matched by FPL player ID — no fuzzy name matching required.

### 4. Train the models

```bash
uv run python scripts/train_models.py
```

Models are saved to `models/` (gitignored — retrain after a fresh clone).

### 5. Seed your current squad

Before the first live run, seed your GW37 squad (or whichever was your last non-Free Hit gameweek) into the decision log so the transfer planner has a starting point:

```python
from data.db import get_session
from data.models import DecisionLog
import json, datetime

db = get_session()
db.add(DecisionLog(
    gameweek=37,
    decision_type="lineup",
    details=json.dumps({
        "squad_ids": [<your 15 internal player IDs>],
        "budget": 104.2,          # (value + bank) / 10 from FPL history API
        "free_transfers": 5,
    }),
    projected_gain=0.0,
    dry_run=True,
    created_at=datetime.datetime.now(datetime.UTC),
))
db.commit()
```

Internal player IDs can be looked up via `SELECT id, fpl_id, web_name FROM players WHERE fpl_id IN (...)`.

### 6. Run a dry-run decision

```bash
uv run python scripts/run_agent.py --dry-run
```

### 7. Scheduler (optional — disabled by default, see below)

```bash
bash deploy/install.sh
```

Fires at 06:00 Fri/Sat/Sun — well before FPL's typical 11:00/11:30 deadlines.
**Disabled as of 2026-07-31**: the host machine isn't always on, so a
systemd timer can silently miss a deadline entirely. Manual weekly kickoff
(below) is the current workflow — re-enable with
`systemctl --user enable --now fpl-bot.timer` if that changes.

---

## Running

```bash
uv run python scripts/run_agent.py --dry-run              # safe preview, no submission
uv run python scripts/run_agent.py --live                 # live submission
uv run python scripts/run_agent.py --chip wildcard        # force a specific chip
uv run python scripts/run_agent.py --json-out out.json    # save full decision to file
```

**Manual weekly kickoff** (real decision + all simulation personas, in the
right order — see plan/simulation-engine-v1.md):
```bash
uv run python scripts/run_weekly.py                       # uses .env's DRY_RUN default
uv run python scripts/run_weekly.py --live                # live submission
```
Runs `run_simulations.py` regardless of `run_agent.py`'s exit code — it
legitimately exits 1 on the benign pre-season "no_projections" case, which
must never block the simulation batch.

Backtest (walk-forward, retrains per GW):
```bash
uv run python scripts/backtest.py --season 2026-27 --start-gw 6 --end-gw 38 --out results/backtest.csv
```

Analysis plots:
```bash
uv run python scripts/plot_analysis.py
# Saves to results/plots/: value_map.png, top_picks.png, cs_by_team.png,
#                           start_prob_violin.png, points_heatmap.png
```

---

## Key Design Decisions

- **`DRY_RUN=true` by default** — the bot never submits live without `--live` or `DRY_RUN=false` in `.env`
- **`config/strategy.py` is the single update point** for all season rule changes (scoring, chip counts, squad structure, DGW multipliers). Nothing else needs editing between seasons.
- **Cross-season form decay** — `avg_pts_5gw_global` and `form_decay_ratio` features cross season boundaries so the model correctly downgrades players who were historically strong but are currently in poor form (e.g. avoids over-projecting expensive players carrying a bad run into a new season).
- **Batch prediction** — features are built once per GW on the full player DataFrame (~841 players × 5 GWs). Previously per-player; now ~100× faster (~10s total).
- **Deduped projections** — `get_latest_projections()` uses a correlated subquery to return only the most recent run's rows per player-GW, avoiding duplicate squad picks across repeated runs.
- **Bench priority order** — GK bench slot = position 0, remaining 3 bench outfield ordered by xPts descending. This ordering is surfaced in the decision JSON and Telegram notification.
- **SQLite WAL mode** — sufficient for local single-machine use; no concurrency issues; back up by copying the `.db` file.
- **Understat AJAX API** — Understat's old HTML-embedded JSON approach broke; uses `POST /main/getPlayersStats` with `X-Requested-With: XMLHttpRequest`. Response is `text/javascript`, parsed with `.text()` not `.json()`.
- **Guardian API free tier** (`api-key=test`) for press signals — PL website and BBC Sport are SPAs not scrapeable without a browser.
- **WC H1/H2 boundary** — derived dynamically from `floor(total_gws / 2)` queried from the DB, with fallback to `strategy.py`.

---

## What Needs Doing Before Season Start

### Critical

- **Seed current squad into decision_log** — done for 2026-27 (GW37 squad, £104.2m budget, 5 FTs). For future seasons, repeat the seeding step above after your final non-FH gameweek.

- **Run Understat ingest once live data appears** — Understat doesn't have 2026-27 data yet (season not started). Once the season begins, `run_agent.py` will automatically ingest xG/xA stats and set piece role derivations on each run.

- **Verify FPL API submission format** — the FPL API occasionally changes its transfer and lineup payload format between seasons. Before going live in GW1, run `--dry-run`, inspect the payload logged, and cross-check against the current API (browser devtools on the FPL site).

- **Retrain models after first few GWs** — models are trained on prior-season data at season start. Retrain with `scripts/train_models.py` after GW3–5 once current-season form data accumulates.

### Important

- **Cold-start prior for new signings** — the model has no historical data for new arrivals (e.g. a foreign signing with no prior PL stats). Currently their rolling averages are zero and the model projects them at replacement level. A planned improvement is a team-level xG floor for players with <3 GWs of data who are starting regularly.

- **Tune chip thresholds** — `config/strategy.py` `ChipTimingThresholds` values are heuristic. After backtesting, adjust `wildcard_pts_gain_threshold`, `bench_boost_min_bench_xpts` etc. to values calibrated to actual historical performance.

- **Captain differential logic** — the captain selector always picks highest single-GW xPts. Doesn't account for ownership (captaining a 60%-owned player is low-leverage). An ownership-weighted captain score is a planned improvement.

- **Guardian API rate limits** — `api-key=test` allows low-volume requests. During the season with frequent press conferences, consider upgrading to a paid Guardian API key if signals start returning empty.
