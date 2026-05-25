# FPL 2026/27 Autonomous Bot

A fully autonomous Fantasy Premier League agent. On each run it ingests live data, projects player points, optimises squad and transfers via integer linear programming, decides chip usage, submits decisions to the FPL API, and sends a Telegram summary — all without human input.

---

## Architecture

```
config/
  settings.py       — credentials and runtime flags (loaded from .env)
  strategy.py       — ALL season-variable rules: scoring, chips, squad structure, DGW params

data/
  models.py         — SQLAlchemy ORM: teams, players, fixtures, gameweeks, stats, projections, decision_log
  db.py             — SQLite engine (WAL mode), session factory, init_db()
  ingestors/
    fpl_api.py      — async FPL bootstrap + fixtures + per-player GW history
    understat.py    — xG/xA scraper (native aiohttp, no understat lib)
    odds_api.py     — The Odds API h2h odds → CS probabilities → fixture_odds table

projection/
  minutes_model.py  — GBT classifier: P(60+ minutes), calibrated probabilities
  points_model.py   — GBT regressor: expected points given recent form + xG/xA
  pipeline.py       — orchestrates feature building → both models → player_projections table

optimiser/
  squad.py          — PuLP ILP: 15-man squad picker + starting XI selector
  transfers.py      — evaluates transfer scenarios (0..max_hits), picks best net xPts gain
  chips.py          — priority-ordered chip recommender (TC → BB → FH → WC)

agent/
  decision_engine.py — main orchestrator: ingest → project → chip → transfers → lineup → log
  fpl_client.py      — async FPL API client: login, transfer submission, lineup PATCH
  notifier.py        — Telegram notification with formatted decision summary

scripts/
  run_agent.py       — CLI entrypoint (--dry-run / --live / --chip / --json-out)
  train_models.py    — trains and saves both ML models to models/
  backfill_history.py — one-time load of vaastav historical CSVs into DB
  backtest.py        — walk-forward backtester: replays historical GWs, scores decisions

deploy/
  fpl-bot.service   — systemd user service unit
  fpl-bot.timer     — runs at 06:00 Fri/Sat/Sun (covers all FPL deadline days)
  install.sh        — symlinks units and enables the timer
```

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
| `FPL_TEAM_ID` | Yes | Your FPL team ID (from the URL) |
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

### 4. Train the models

```bash
uv run python scripts/train_models.py
```

### 5. Run a dry-run decision

```bash
uv run python scripts/run_agent.py --dry-run
```

### 6. Install the scheduler (optional)

```bash
bash deploy/install.sh
```

---

## Running

```bash
uv run python scripts/run_agent.py --dry-run              # safe preview, no submission
uv run python scripts/run_agent.py --live                 # live submission
uv run python scripts/run_agent.py --chip wildcard        # force a specific chip
uv run python scripts/run_agent.py --json-out out.json    # save full decision to file
```

Backtest:
```bash
uv run python scripts/backtest.py --season 2026-27 --start-gw 10 --end-gw 38 --out results/backtest.csv
```

---

## Key Design Decisions

- **`DRY_RUN=true` by default** — the bot will never submit live without explicitly setting `--live` or `DRY_RUN=false` in `.env`
- **`config/strategy.py` is the single update point** for all season rule changes (scoring, chip counts, squad structure, DGW multipliers). Nothing else needs editing between seasons.
- **DGW multiplier = 1.85** (not 2.0) — discounts rotation and injury risk for double-gameweek players
- **No `understat` library** — incompatible with aiohttp ≥ 3.9; replaced with native aiohttp scraper parsing embedded JSON
- **SQLite with WAL mode** — sufficient for local single-machine use; simple to back up (just copy the `.db` file)

---

## What Needs Doing Before Season Start

### Critical

- **Backfill current squad into decision_log** — on first run the bot has no saved squad and treats it as a blank slate (triggers a WC). Before GW1, manually insert a `lineup` decision log entry with your actual FPL squad IDs, budget, and free transfers. See the schema in `data/models.py` (`decision_log` table).

- **Run the backtest and tune chip thresholds** — `config/strategy.py` contains `ChipTimingThresholds` values that were set heuristically. Run the backtest over a full historical season and adjust `wildcard_pts_gain_threshold`, `bench_boost_min_bench_xpts`, etc. to values that would have made correct chip calls historically.

- **Verify FPL API submission format** — the FPL API occasionally changes its transfer and lineup payload format between seasons. Before going live in GW1, run `--dry-run`, inspect the payload logged, and cross-check against the current API (browser devtools on the FPL site). Pay particular attention to `purchase_price` / `selling_price` fields.

- **Retrain models on 26/27 data** — after a few GWs of the new season, re-run `scripts/train_models.py` so the models are trained on current-season player values, team strengths, and form. The models shipped at the start of the season are trained on prior-season data only.

### Important

- **Improve historical backfill data** — the vaastav CSV backfill (`scripts/backfill_history.py`) only loaded partial data for 2022-25 seasons. A denser historical dataset would improve model calibration. Consider loading all players from the vaastav CSVs, or supplementing with FPL's own historical endpoints.

- **Add fixture difficulty ratings (FDR) as a model feature** — currently the projection models don't use opponent strength. Adding FDR or team defensive/attack ratings from the `teams` table as features would improve xPts estimates for players facing strong/weak opponents.

- **Improve CS probability estimation** — `odds_api.py` derives CS probability from h2h odds using a simple heuristic (`cs ≈ draw_prob + opponent_win_prob * 0.3`). This is a rough approximation. A dedicated over/under 0.5 goals market, or a trained CS model using historical team clean sheet rates vs opponent attack strength, would be more accurate.

- **Captain differential logic** — the current captain selector always picks the highest single-GW xPts player. It doesn't account for ownership (captaining a 60%-owned player is low-leverage). Add an ownership-weighted captain score, especially for aggressive/contrarian team strategies.

- **Wildcard timing near the half-season boundary** — `chips.py` checks whether WC1 has expired using `CHIPS.wildcard_first_half_deadline_gw`. This hardcoded boundary needs verifying each season (FPL sometimes moves the H1/H2 split).

### Nice to Have

- **FPLReview / FPLForm integration** — the architecture supports plugging in crowd-sourced projections as an additional feature. Adding FPLReview xPts as a weighted input alongside the model's own projections would likely improve accuracy.

- **Price change prediction** — `OPTIMISER.use_price_change_signals = True` is set but the signal isn't implemented yet. Use transfer volume from the FPL API (`transfers_in`, `transfers_out` in `player_gw_stats`) to predict imminent price rises/falls and factor into squad selection.

- **Bench ordering optimisation** — the ILP currently picks the optimal squad and XI but doesn't optimise bench order. The first bench player (emergency sub) matters most; they should be the highest-xPts outfield player not in the XI, not just whoever falls outside the starting positions.

- **Telegram interactive commands** — extend `notifier.py` to support inbound Telegram commands (`/status`, `/squad`, `/force_chip wildcard`) via a webhook or polling loop, allowing manual overrides without touching the CLI.

- **Test coverage** — the `tests/` directory exists but is empty. At minimum, unit tests for the ILP constraints (valid squad structure, budget, per-club limit) and the chip recommender logic would catch regressions when tuning thresholds.
