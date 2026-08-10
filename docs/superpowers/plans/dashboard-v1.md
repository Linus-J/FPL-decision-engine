# Dashboard v1 — Design

## Purpose

A local, read-only Streamlit dashboard giving one place to check: current live
squad + projected points, fixtures/DGW exposure, injury/availability news,
past-gameweek decisions (projected vs actual), and the bot's next planned
chip/transfer move. Scoped as phase 1 of a larger idea; a phase-2 multi-
simulation engine (many parallel "shadow" strategies run live through the
season, never submitted to FPL, compared post-season) is a separate future
project that will plug into this same dashboard once built.

## Decisions

- **Stack**: Streamlit (pure Python, no separate frontend/backend split).
- **Access**: local only (`streamlit run`, localhost). No remote/network exposure.
- **Data source**: reads the existing SQLite DB directly (same one `run_agent.py`
  writes to). No new ingestion triggered by the dashboard — freshness is
  whatever the last scheduled/manual agent run produced.
- **Current squad**: fetched live from FPL's public (no-auth) entry/picks
  endpoint (`/api/entry/{team_id}/event/{gw}/picks/`), not from `decision_log`
  — this stays correct even if the bot is in dry-run or the squad was changed
  manually in the FPL app.
- **Transfer rumours**: out of scope for v1 (no reliable free source exists).
  Noted as a placeholder / known future data-source gap, not built.
- **Chip/transfer plan**: display-only — reads the most recent `decision_log`
  rows, does not re-invoke the optimiser live.
- **Actual outcomes**: `DecisionLog.actual_outcome` exists in the schema but
  is never written anywhere today. Adding a backfill script so past decisions
  show projected-vs-actual, which is central to "track... decisions".

## Architecture

```
dashboard/
  Home.py                    — entrypoint: season/GW context, nav
  pages/
    1_Squad.py                — live squad (FPL picks) + xPts overlay
    2_Fixtures_DGW.py         — fixture list, DGW tracker
    3_Injury_News.py          — injury/status news, squad-scoped + league-wide
    4_Decision_History.py     — past decisions, projected vs actual
    5_Chip_Plan.py             — latest chip/transfer recommendation (display only)
  data/
    squad.py                  — live FPL picks fetch + xPts join
    fixtures.py                — fixture list + DGW coverage queries
    news.py                    — injury/status/news queries
    decisions.py               — decision_log reads, joins actual_outcome
  fpl_public.py                — thin httpx client for FPL's public entry/picks endpoint

scripts/
  backfill_decision_outcomes.py — computes actual realized points for past
                                   "lineup" DecisionLog rows once a GW's
                                   fixtures are all finished; writes
                                   DecisionLog.actual_outcome
```

`dashboard/data/*.py` functions take a DB session and return plain
dataclasses/DataFrames — no Streamlit imports — so they're unit-testable and
reusable later by the simulation-engine's own dashboard views.

## Data flow

- **Squad page**: `fpl_public.get_picks(team_id, gw)` → element IDs → join to
  `players` (by `fpl_id`) and `get_latest_projections()`.
- **Fixtures/DGW page**: `fixtures` table for a rolling window, plus
  `optimiser.transfers.get_dgw_coverage()` against the live squad.
- **Injury/News page**: `players` where `status != 'a'` or `news != ''`,
  split into "in your squad" vs "league-wide".
- **Decision History page**: `decision_log` ordered by GW desc, `details`
  JSON parsed per `decision_type` (`lineup` / `transfers` / `chip`), joined
  to `actual_outcome` once backfilled; running projected-vs-actual chart.
- **Chip Plan page**: most recent `chip`/`transfers` rows from `decision_log`.

### Backfill script (`scripts/backfill_decision_outcomes.py`)

For each `lineup` `DecisionLog` row with `actual_outcome IS NULL` whose GW's
fixtures are all `finished`:
1. Sum `player_gw_stats.total_points` per player for that (season, gameweek)
   — summed across rows to handle a genuine DGW player having two rows.
2. Reuse `scripts.backtest._score_squad(squad_ids, starting_ids, captain_id,
   actual_points, bench_boost, triple_captain, vice_captain_id)` — same
   function the backtester already uses — to get the realized total.
3. **Known simplification**: no autosub modelling. `_score_squad` supports
   autosubs when given `minutes`/`positions`/`bench_order`, but
   `bench_order` is never persisted on the `lineup` decision today, so this
   backfill calls it without those three (its documented no-autosub mode).
   Good enough for a projected-vs-actual sanity check; revisit if bench
   rotation appears to matter.
4. Write the result to `actual_outcome`; commit. Purely additive — never
   touches `dry_run` or any decision-making logic.

Run manually (`uv run python scripts/backfill_decision_outcomes.py`) or add
to the Fri/Sat/Sun schedule after `run_agent.py`.

## Error handling

- FPL public endpoint call wrapped with a short retry (matches existing
  `tenacity` usage elsewhere); on failure the Squad page shows a warning and
  falls back to the last `lineup` `DecisionLog` entry instead of crashing.
- Empty states for every page (no fixtures yet pre-season, no decision_log
  rows yet, no projections yet) — render a plain "nothing yet" message
  rather than an empty/broken table.

## Testing

- `dashboard/data/*.py` functions are pure (DB in, dataclass/DataFrame out)
  → unit tests with an in-memory/temp SQLite DB, same pattern as existing
  `tests/test_*.py`.
- `scripts/backfill_decision_outcomes.py`'s point-summing and `_score_squad`
  reuse get a unit test covering a normal GW and a DGW player.
- Streamlit page rendering itself is not unit-tested (no good tool for it in
  this stack); verified manually by running `streamlit run dashboard/Home.py`
  and checking each page loads without exception.

## Out of scope for v1

- Transfer rumours / market news (no source yet).
- Interactive what-if transfer/chip explorer (display-only for now).
- Autosub-aware actual-outcome scoring (needs `bench_order` persisted first).
- The 100-parallel-simulation engine (separate future project).
