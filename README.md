# FPL 2026/27 Autonomous Bot

A Fantasy Premier League system with three parts:

1. **The autonomous agent** — each run ingests live data, projects player points via a distributional Monte-Carlo pipeline, optimises squad and transfers via integer linear programming, decides chip usage, submits decisions to the FPL API, and sends a Telegram summary — all without human input (submission is dry-run by default).
2. **A read-only dashboard** (Streamlit) — one place to check your live squad + projected points, fixtures/DGW exposure, injury news, past-decision history (projected vs actual), and the bot's next planned chip/transfer.
3. **A live simulation engine** — 90 "shadow" managers, each varying one parameter of the decision engine, stepped forward through the exact same decision logic as the real bot every scheduled run. Never submitted to the real FPL app — it is the project's validation instrument, and the design is a one-factor-at-a-time experiment so a single season's results are interpretable.

> **📖 [How the decision engine works](docs/decision-engine.md)** — high-level
> explanation of the approach and the ideas behind it, then a call-by-call
> reference. Start there.
>
> Note: submission to the FPL API is **not** in scope — the bot decides, you
> enter the team. See the doc for what else is deliberately excluded.

---

## Architecture

```
config/
  settings.py       — credentials and runtime flags (loaded from .env)
  strategy.py       — ALL season-variable rules: scoring, chips, squad structure,
                       DGW params, risk-aware optimiser config (risk_level/mu_baseline/mu_range)

data/
  models.py         — SQLAlchemy ORM: 22 tables (players/fixtures/projections/
                       decision_log + sim_managers/sim_decision_log for the simulation engine)
  db.py             — SQLite engine (WAL mode), session factory, init_db()
  overrides.py      — reads config/transfer_overrides.yaml: applies confirmed team_id
                       corrections to the live candidate pool, feeds rumoured-departure
                       p_leave into the optimiser's discount gate — see "Transfer Overrides"
                       below
  ingestors/
    fpl_api.py           — async FPL bootstrap + fixtures + per-player GW history
    understat_xg.py      — real per-match xG/xA/shots/key passes (soccerdata's Understat reader)
    whoscored.py         — real per-match defensive actions (tackles/interceptions/clearances/
                            blocks/recoveries) — the free-tier fix for DefCon + bonus accuracy
    fbref.py / fbref_prior.py — BPS-relevant match events; Championship/top-5 prior-league
                            bridge for promoted-team players and new signings
    odds_api.py          — The Odds API h2h odds → CS probabilities → fixture_odds table
    ownership.py         — top-10k ownership snapshots (differential/template scoring input)
    injury_parser.py     — regex parser on players.news → injury_severity (0–3)
    midweek.py           — derives midweek fixture flags (2 games in 7d with ≥3d gap)
    press_conference.py  — Guardian API sentiment scraper → PlayerPressSignal table
    transfermarkt.py     — Transfermarkt scraper: auto-fills confirmed transfers, writes
                            reviewable rumour candidates — manual/on-demand, see
                            "Transfer Overrides" below

projection/
  assemble.py       — P10 Monte-Carlo assembly: the live projection pipeline (per-fixture
                       scenario draws for minutes/goals/assists/CS/saves/DefCon/bonus)
  minutes_model.py  — 3-way GBT classifier: P(0 / 1-59 / 60+ minutes), calibrated
  team_goals.py     — double-Poisson λ_home/λ_away solved from odds (1X2 + O/U2.5)
  goals.py / assists.py / clean_sheets.py / saves.py / defcon.py / bonus.py
                    — per-component MC samplers feeding assemble.py
  covariance.py     — shared per-fixture team-goal latent (real teammate correlation,
                       not independent per-player draws)
  cold_start.py     — GW1 projections with no current-season data: established players get
                       their own real per-GW variance from prior-season history; new
                       signings/promoted players get mean AND variance pooled from real
                       peers at the same position + price (not a synthetic formula);
                       fixture-difficulty-weighted lookahead across the next N GWs
                       (cold_start_lookahead_gws, default 5), not just single-GW xPts
  fixture_adjust.py — per-horizon-GW opponent scaling
  rescore.py        — 26/27 BPS re-scoring of historical actuals for backtesting
  pipeline.py       — orchestrates assemble.py, persists to player_projections

optimiser/
  squad.py          — PuLP ILP: 15-man squad picker + XI selector; bench ordered by priority
  transfers.py      — multi-period ILP transfer planner (0..max_hits), best net xPts gain
  chips.py          — priority-ordered chip recommender (scenario-EV gated); WC H1/H2 from DB
  scoring.py        — risk-adjusted objective: continuous risk_level [-1, 1] drives both
                       ownership-differential weight (lambda) and variance-awareness (mu)
  captaincy.py      — scenario-based captaincy using real joint MC draws for team-total variance

agent/
  decision_engine.py — shared decision core (_run_decision_cycle) used by BOTH the real
                        bot (run()) and every simulation persona (run_for_persona()) —
                        same logic, different config + storage, never a forked copy
  fpl_client.py      — async FPL API client: login, transfer submission, lineup PATCH
  notifier.py        — Telegram notification: starting XI, bench in priority order, transfers

simulation/
  personas.py       — generates ~100 personas (risk_level, max_ownership_differential,
                       chip_aggressiveness), seeded + persisted once per season
  engine.py         — steps every persona forward one GW via decision_engine.run_for_persona,
                       each isolated by try/except so one failure can't affect another

dashboard/
  Home.py, pages/   — Streamlit multi-page app: Squad, Fixtures & DGW, Injury News,
                       Decision History, Chip Plan, Simulations leaderboard
  data/             — pure DB-in/DataFrame-out query functions behind each page

scripts/
  run_agent.py                   — CLI entrypoint (--dry-run / --live / --chip / --json-out)
  run_simulations.py             — runs the weekly simulation batch (all personas)
  run_weekly.py                  — manual kickoff: run_agent.py then run_simulations.py,
                                    in order, regardless of the first's exit code
  backfill_decision_outcomes.py  — fills in actual (vs projected) points once a GW finishes,
                                    for both the real decision_log and every persona
  backtest.py                    — walk-forward backtester (pinned to zero-variance scoring
                                    so the exit-gate number stays comparable across changes)
  scrape_understat_xg.py / scrape_whoscored.py / scrape_fbref.py / scrape_prior_league.py
                                  — data ingestion runners for the sources above
  plot_analysis.py               — generates analysis plots to results/plots/

deploy/
  fpl-bot.service   — systemd user service (chains run_agent.py + run_simulations.py)
  fpl-bot.timer     — Fri/Sat/Sun 06:00 — disabled by default, see "Running" below
  install.sh        — symlinks units, enables the timer if you want it
```

---

## Data Sources

All free tier. The free-tier defensive-action gap (clearances/blocks/recoveries missing from
FBref) turned out to be the single biggest lever on projection accuracy this project found —
see `docs/superpowers/plans/phase-2-xpts-engine.md` for the investigation.

| Source | What it provides | Key |
|---|---|---|
| FPL API | Players, fixtures, GW history, team strengths, live squad picks | Free |
| vaastav CSV archive | Historical GW stats 2021–25 (backfill) | Free |
| Understat (via `soccerdata`) | Real per-match xG/xA/npxG/shots/key passes | Free, browserless |
| WhoScored (via `soccerdata`) | Real per-match tackles/interceptions/clearances/blocks/recoveries — fixes the DefCon + bonus accuracy gap | Free, browser required |
| FBref (via `soccerdata`) | Match events for BPS re-scoring; Championship/top-5 prior-league bridge | Free, browser required |
| The Odds API | h2h + O/U2.5 odds → team-goal Poisson λ, CS probability | Free tier (500 req/month) |
| Guardian API | Press conference text → injury/availability signals | Free (`api-key=test`) |
| Transfermarkt (scraped) | Confirmed transfers (auto-fills team_id corrections), transfer rumours (credibility-scored candidates for manual review) | Free, no key — manual/on-demand only, see "Transfer Overrides" below |

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
| `GUARDIAN_KEY` | Optional | Guardian API key for press-conference signals (defaults to the low-volume `test` key) |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram bot token for notifications |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat ID to send notifications to |
| `DRY_RUN` | — | `true` (default) = no live submissions |
| `DB_PATH` | — | SQLite database path (default: `fpl_bot_v2.db`) |

### 3. Initialise the database and backfill history

```bash
uv run python -c "from data.db import init_db; init_db()"
uv run python scripts/backfill_history.py
```

This loads vaastav historical CSVs (2021–25) matched by FPL player ID — no fuzzy name matching required.

### 4. Seed your current squad

Before the first live run, seed your last-known squad into the decision log so the transfer
planner has a starting point (skip this for a true GW1 cold start — `cold_start.py` builds
an initial squad automatically when no prior decision exists):

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
        "starting_ids": [<your 11 starting internal player IDs>],
        "captain_id": <internal player id>,
        "vice_captain_id": <internal player id>,
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

### 5. Run a dry-run decision

```bash
uv run python scripts/run_agent.py --dry-run
```

### 6. Scheduler (optional — disabled by default)

```bash
bash deploy/install.sh
```

Fires at 06:00 Fri/Sat/Sun — well before FPL's typical 11:00/11:30 deadlines.
**Disabled by default**: the host machine isn't always on, so a systemd timer can silently
miss a deadline entirely. Manual weekly kickoff (below) is the current workflow — re-enable
with `systemctl --user enable --now fpl-bot.timer` if you move to an always-on host.

---

## Running

```bash
uv run python scripts/run_agent.py --dry-run              # safe preview, no submission
uv run python scripts/run_agent.py --live                 # live submission
uv run python scripts/run_agent.py --chip wildcard        # force a specific chip
uv run python scripts/run_agent.py --json-out out.json    # save full decision to file
```

**Manual weekly kickoff** (real decision + all simulation personas, in the right order):
```bash
uv run python scripts/run_weekly.py                       # uses .env's DRY_RUN default
uv run python scripts/run_weekly.py --live                # live submission
```
Runs `run_simulations.py` regardless of `run_agent.py`'s exit code — it legitimately exits 1
on the benign pre-season "no_projections" case, which must never block the simulation batch.

**Dashboard**:
```bash
uv run streamlit run dashboard/Home.py
```
Local-only, read-only — never triggers ingestion or submissions. Opens at `localhost:8501`.

**Backfill actual outcomes** (once a gameweek finishes, for the projected-vs-actual view in
the Decision History / Simulations dashboard pages):
```bash
uv run python scripts/backfill_decision_outcomes.py --season 2026-27
```

**Site data export** (after reviewing/overruling a `run_agent.py --dry-run` decision, to publish
the current squad + top-15 xPts + transfer/chip history to the portfolio site's `$ fpl status`
panel — see `linus-j.github.io`'s own repo for the display side):
```bash
uv run python scripts/export_site_data.py            # writes, commits, and pushes
uv run python scripts/export_site_data.py --no-push   # writes + commits locally only, for review
```
Writes `data/simulations/gw{N}.json` and updates `data/simulations/index.json`. Not run
automatically by anything — a deliberate manual step, run once you're happy with the week's
decision (matches why the systemd timer below is disabled by default).

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

## Transfer Overrides

FPL's own bootstrap data can lag reality — a confirmed transfer takes a few days to update
`team_id`, and there's no signal at all for a rumoured-but-not-yet-confirmed departure.
`config/transfer_overrides.yaml` (hand-edited, version-controlled) fixes both, keyed by each
player's stable `code` (not `id`/`fpl_id` — those get reassigned across seasons/transfers,
`code` doesn't):

```yaml
confirmed:
  - code: 123456        # corrects team_id ahead of FPL's own update
    team_id: 1
    reason: "Signed from Newcastle, not yet reflected in FPL team_id"
    as_of: "2026-08-10"

rumoured:
  - code: 234567         # discounts (never excludes) projected points by (1 - p_leave)
    p_leave: 0.35
    reason: "Strongly linked to a January move per <source>"
    as_of: "2026-08-10"
```

**`confirmed`** is read by every candidate-pool load (cold-start and in-season) — a corrected
`team_id` is visible to the max-3-per-club constraint and the fixture-difficulty lookahead
immediately. **`rumoured`** feeds the optimiser's existing departure-risk discount
(`optimiser/departure_risk.py`) — the bot still considers the player, just at a
`p_leave`-scaled discount, and logs a warning if a rumoured player still makes the final squad.

### Editing it by hand

Just add an entry under `confirmed`/`rumoured` with the player's `code` (look it up via
`SELECT code, web_name FROM players WHERE web_name LIKE '%...%'`) and commit the file — no
code changes, no restart needed, the next decision cycle picks it up.

### Auto-filling it from Transfermarkt

`data/ingestors/transfermarkt.py` scrapes Transfermarkt's Premier League transfers + rumours
pages. Manual/on-demand only (not wired into `run_weekly.py`):

```bash
uv run python -c "from data.ingestors import transfermarkt as tm; tm.run(season='2026-27')"
```

- **Confirmed transfers are auto-applied** directly into `transfer_overrides.yaml`'s
  `confirmed` list, tagged `source: transfermarkt` so a hand-written entry is never touched —
  idempotent (safe to re-run) and self-cleaning (removes its own entries once FPL's own
  `team_id` catches up).
- **Rumours are never auto-applied.** They're written to a separate, gitignored
  `config/transfer_overrides_candidates.yaml` (regenerated fresh each run, sorted by
  Transfermarkt's own credibility "Assessment" score, floor 40%) for you to review and
  hand-copy any you trust into `transfer_overrides.yaml`'s `rumoured` list yourself.

---

## Key Design Decisions

- **`DRY_RUN=true` by default** — the bot never submits live without `--live` or `DRY_RUN=false` in `.env`
- **`config/strategy.py` is the single update point** for all season rule changes (scoring, chip counts, squad structure, DGW multipliers, risk posture). Nothing else needs editing between seasons.
- **Continuous risk_level, not a 3-way switch** — `risk_level` runs -1.0 (safe) to +1.0 (aggressive); the variance term has a non-zero baseline so 0.0 ("medium") carries real variance-awareness rather than none. The same formula governs every gameweek's real transfer/lineup decisions, not just cold start.
- **Risk-aware cold start** — GW1 projections have real variance (established players' own historical spread; new signings pooled from real position+price peers), so different risk postures genuinely produce different squads from day one, not just once ownership data exists mid-season.
- **Simulation engine shares the real decision loop** — `agent/decision_engine.py::run_for_persona` calls the exact same core as the real bot's `run()`, parameterized by config + storage, never a duplicated/forked copy. It never imports `fpl_client` — no submission path exists in that code at all.
- **Backtest is pinned to zero-variance scoring** (`scripts/backtest.py::_BACKTEST_CONFIG`) so the walk-forward exit-gate number stays a stable, comparable yardstick even as the live risk-scoring defaults evolve.
- **Cross-season form decay** — `avg_pts_5gw_global` and `form_decay_ratio` features cross season boundaries so the model correctly downgrades players who were historically strong but are currently in poor form.
- **Bench priority order** — GK bench slot ordered first, remaining 3 bench outfield ordered by xPts descending. Surfaced in the decision JSON, Telegram notification, and the live FPL submission payload (each of the 15 `picks` gets a genuinely unique 1-15 slot).
- **SQLite WAL mode** — sufficient for local single-machine use; no concurrency issues; back up by copying the `.db` file.
- **WC H1/H2 boundary** — derived dynamically from `floor(total_gws / 2)` queried from the DB, with fallback to `strategy.py`.

---

## What Needs Doing Before Season Start

### Critical

- **Seed current squad into decision_log** — done for 2026-27. For future seasons, repeat the seeding step above after your final non-Free-Hit gameweek.

- **Run Understat/WhoScored ingest once live GW1 data appears** — both are season-long scrapes; re-run after each real gameweek so DefCon/bonus accuracy keeps improving through the season.

- **Verify FPL API submission format before the first real `--live` run** — the FPL API occasionally changes its transfer/lineup payload format between seasons, and this codebase's live submission path has never been exercised outside dry-run. Run `--dry-run`, inspect the payload logged, and cross-check against the current API (browser devtools on the FPL site).

### Important

- **Tune risk-aware scoring constants** — `mu_baseline`/`mu_range` in `config/strategy.py::OptimiserConfig` are untuned starting values (like this project's other heuristic constants). Revisit after a season's worth of real risk_level-varied outcomes from the simulation engine.

- **Captain differential logic is partially addressed** — the default (non-scenario) captain pick already factors in ownership via the risk-adjusted objective, but the scenario-based MC override (`captaincy.pick_captain`, used once real fixture samples exist) only weighs mean + variance, not ownership. Worth unifying if scenario-based captaincy turns out to systematically override the differential-aware pick.

- **Guardian API rate limits** — `api-key=test` allows low-volume requests. During the season with frequent press conferences, consider upgrading to a paid Guardian API key if signals start returning empty.

- **Calibrate `mu_baseline`/`mu_range`/chip-timing thresholds together** — several of `config/strategy.py`'s heuristic constants (bonus GK-save scaling, chip panic-window shrink, the new risk constants) were set by judgment rather than backtesting; worth a joint calibration pass once enough live data exists.
