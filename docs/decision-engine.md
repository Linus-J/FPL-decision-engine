# How the decision engine works

Two levels. Part 1 is the shape of the thing and the ideas behind it — read
this to understand what the bot is doing and why. Part 2 is the call-by-call
detail — read that when changing something.

Current as of 2026-08-16 (post P0–P3 of
`superpowers/plans/decision-engine-recovery-plan.md`).

---

# Part 1 — High level

## What it is

A **decision engine**, not a submission bot. Every week it reads the world,
projects how many points each player will score, and solves for the best
legal squad and lineup under FPL's rules. It writes its decision to a log and
notifies you. **Entering the team is manual** — there is no live submission
path, deliberately.

## The one-paragraph version

Ingest FPL, odds, match events and set-piece duty into SQLite. For each of the
next few gameweeks, simulate every fixture thousands of times: bookmaker odds
set how many goals each team scores, learned per-player rates decide who
scores them, and FPL's scoring rules turn simulated events into points. That
gives every player a distribution of points, not just a mean. Feed those into
an integer program that picks the squad, the XI, the captain and the transfers
that maximise points subject to budget, squad structure, and the free-transfer
rules. Log what it decided and what it declined. After the gameweek, score
what actually happened.

## The five stages

```
  ┌────────────┐   ┌─────────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐
  │ 1. INGEST  │──▶│ 2. PROJECT  │──▶│ 3. OPTIMISE  │──▶│ 4. DECIDE│──▶│ 5. MEASURE │
  │ the world  │   │ the points  │   │ the squad    │   │ & record │   │ the result │
  └────────────┘   └─────────────┘   └──────────────┘   └──────────┘   └────────────┘
    FPL API          Monte Carlo        Integer            decision       actual
    odds             per fixture        programming        log +          outcomes,
    match events     ↓                  (PuLP/CBC)         Telegram       calibration
    set-piece duty   xpts + variance
```

**1. Ingest.** FPL's own API is the spine: players, prices, teams, fixtures,
gameweek deadlines, injury status. Around it sit bookmaker odds (the single
most valuable external signal), FBref/WhoScored match events (for defensive
contributions and the bonus-point simulator), Understat xG, and a published
penalty/set-piece taker list.

**2. Project.** For each fixture, odds are converted into an expected goals
figure for each team. The fixture is then simulated thousands of times: draw
each team's goals, draw who scored and assisted them from learned per-player
shares, draw minutes, cards, saves, defensive actions. Each simulated match is
scored under FPL's 26/27 rules — including a bonus-point simulator that
reproduces BPS from the underlying events. The output per player is a **mean
and a variance**, not a point estimate.

**3. Optimise.** Squad selection is an integer program. Maximise projected
points subject to: £100m budget, 2/5/5/3 by position, max 3 per club, a legal
XI. Transfers are a *multi-period* program over the planning horizon, so it can
choose to bank a transfer this week to make two next week, and can take a
−4 hit when the gain justifies it.

**4. Decide and record.** Chips are evaluated in priority order against
thresholds. The decision is written to `decision_log` — squad, XI, captain,
bench order, bank, purchase prices, free transfers, **and the counterfactual**:
which chip was considered and why it didn't fire.

**5. Measure.** After the gameweek finishes, each decision is scored against
what actually happened — auto-substitutions applied, vice-captain promoted if
the captain blanked, hits deducted. That number is what everything is judged
on.

## The ideas that matter

**Distributions, not point estimates.** Almost every stage carries variance,
not just a mean. This is what makes it possible to ask "how likely is this
chip to pay off" rather than only "is the average good".

**Odds set the total; player rates set the attribution.** This split is the
core modelling decision. Bookmakers are extremely good at "how many goals will
Arsenal score" and say nothing about "will it be Saka or Gyökeres". So team
totals come from the market and the split across players is learned. It also
means anything the market already knows — including that a team is weak
defending corners — arrives automatically through the goal total, and does not
need modelling separately.

**Cold start is a separate problem.** At GW1 there is no current-season
history. A parallel path builds the initial squad from prior-season data,
translated non-PL leagues for new signings, and pooled peer data for genuine
unknowns. Its projections are on the same *scale* as the in-season ones
(expected points per gameweek, not per appearance), which is what lets the
same optimiser consume both.

**The optimiser's curse is real and corrected for.** Picking the highest
projections systematically selects players whose projections are too high.
Measured: the projection layer is roughly unbiased, but with the decision
layer switched on the bot over-predicted by ~8 points per gameweek. Two
mechanisms push back — projections are shrunk toward their group mean before
selection, and every transfer pays a fixed switching cost so a noise-sized
edge is not enough to trigger churn.

**Validation is a live season, not a backtest.** The historical harness is a
second implementation of the decision loop and kept diverging from the real
one. Instead, ~90 shadow personas run the *same* code every week with
different parameter settings. Because they share a season, comparisons between
them are paired and far less noisy than any absolute score.

## What it deliberately does not do

- **Submit to FPL.** Out of scope.
- **Predict price changes.** No model exists. Selling prices are tracked
  (you get half of any rise back), but price movement is not forecast.
- **Model set-piece defensive weakness.** Largely priced into the odds already.
- **Use effective ownership.** Plumbed in, but inert while the risk posture is
  neutral, and there is no data until GW1 locks.
- **Use press-conference sentiment.** Ingested historically, never validated,
  now switched off.

## Known limitations, honestly

| Limitation | Effect |
|---|---|
| Odds cover ~6/10 GW1 fixtures, none beyond | Unpriced fixtures fall back to a calibrated team-strength model rather than a flat scoreline, so the horizon still differentiates fixtures — but on a weaker signal than the market |
| Penalty duty is a published list, not observed | Right today, decays as duty changes; refresh in-season |
| No teammate covariance in the optimiser | Two players in the same match are treated as independent when choosing the squad. `captaincy.py` implements the covariance-aware version, but it is **dormant**: `mu_baseline` calibrated to 0, so it short-circuits to mean argmax |
| Chip thresholds are untuned | Fitted against a model that could not bank, hit, or see past one gameweek; the persona sweep is what will retune them |
| Cohort measures main effects only | One season cannot resolve interactions between parameters |

---

# Part 2 — Low level

## Entry points

| Command | Does |
|---|---|
| `scripts/run_weekly.py` | The whole weekly cycle, in order |
| `scripts/run_agent.py` | Ingest + one real decision + notify |
| `scripts/run_simulations.py` | Step every shadow persona forward one gameweek |
| `scripts/backfill_decision_outcomes.py` | Score finished gameweeks |
| `scripts/backtest.py` | Historical walk-forward (secondary; see caveat) |

`run_weekly.py` order, and why:

```
FBref → WhoScored → set-pieces → ownership → quality gate
      → backfill outcomes → run_agent → run_simulations
```

FBref before WhoScored (WhoScored only *patches* rows FBref created).
Outcomes are backfilled **before** new decisions so last week is scored
against complete data. Simulations run last and always, regardless of the
agent's exit code.

## Stage 1 — Ingest

| Source | Module | Lands in | Notes |
|---|---|---|---|
| FPL bootstrap | `data/ingestors/fpl_api.py` | `players`, `teams`, `fixtures`, `gameweeks`, `player_state_snapshots` | The spine. `code` is the stable cross-season player identity, not `fpl_id` |
| Bookmaker odds | `data/ingestors/odds_api.py` | `fixture_odds` | Append-only, one row per fetch; reads are as-of the deadline |
| FBref match events | `data/ingestors/fbref.py` | `player_match_events` | Needs a browser; Cloudflare blocks headless |
| WhoScored | `data/ingestors/whoscored.py` | patches `player_match_events` | Defensive actions for DefCon |
| Understat | `data/ingestors/understat.py` | `player_xg_stats`, `player_setpiece_roles` | |
| Set-piece depth chart | `data/ingestors/setpiece.py` | `player_setpiece_roles` | Published list; carries taker *order* |
| Top-10k ownership | `data/ingestors/ownership.py` | `ownership_snapshots` | Empty until GW1 locks |

**Leakage discipline.** Anything read for gameweek *N* is filtered to
information available before *N*'s deadline. Odds reads pick the latest fetch
at or before the deadline; rolling features are `shift(1)`-ed; player state
snapshots are joined as-of.

## Stage 2 — Projection

`projection/pipeline.py::run_projections` orchestrates; the work is in
`projection/assemble.py::assemble_gw_projections`.

Per fixture, per scenario (default a few thousand):

1. **Team goals.** `team_goals.py::team_goals_from_odds` inverts de-vigged
   1X2 + over/under-2.5 into `λ_home`, `λ_away`. No odds → `team_goals_from_strength`,
   a calibrated fit of λ to league-relative team strengths (R²≈0.58, 39% less
   error than the flat pair it replaced). Neither known → the league-average
   fixture, 1.51/1.22.
2. **Minutes.** `minutes_model.py::predict_minutes_bands` gives each player
   P(0 min), P(1–59), P(60+). A band is drawn per scenario; attacking output
   scales with the drawn minutes, not an average.
3. **Goal and assist attribution.** `covariance.py::split_multinomial`
   distributes the drawn team goals across players by `goal_weight`, scaled by
   drawn minutes. **`goal_weight` = rolling non-penalty xG + this season's
   penalty duty** — an exact decomposition, since `xg − npxg` *is* the penalty
   component. Without a depth chart it falls back to rolling `xg`.
4. **Everything else.** Clean sheets (`clean_sheets.py`) anchored on the same
   drawn goals conceded; saves; cards; defensive contributions (`defcon.py`);
   bonus via a BPS simulator (`bps_sim.py`, `bonus.py`) reproducing the 26/27
   weights.
5. **Reduce.** Mean → `xpts`, variance → `xpts_var`. Raw draws are persisted
   to `projection_samples` (used by chip payoff probabilities and scenario
   captaincy).

Then, in order: zero out unavailable players → apply an injury-severity
discount → apply **curse shrinkage** (`apply_curse_shrinkage`, shrink toward
the gameweek/position group mean) → persist.

`get_latest_projections(horizon=…)` reads them back. **The horizon matters**:
every consumer slices this frame rather than building its own, so
`OPTIMISER.projection_horizon_gws` must be ≥ every consumer's horizon.
`config/strategy.py::assert_horizons_consistent()` enforces it at import.

### Cold start

When the season has no played gameweeks, `run_projections` correctly returns
nothing and `projection/cold_start.py` takes over. Four tiers, in preference
order:

| Tier | Applies to | Source |
|---|---|---|
| `prior_season` | ≥5 prior PL appearances | Own prior-season points per appearance |
| `prior_league_prior` | Matched non-PL record | Translated npxG90/xA90 |
| `peer_bucket_prior` | Nobody knows them | Pooled real peers by position + price band |
| `position_price_prior` | Even the peer pool is sparse | Synthetic linear prior |

Every tier's output is converted from *per appearance* to *per gameweek* via
`unconditional_moments(p_appear, mean, var)`, so a rotation risk projects both
lower and more variable than a nailed player with the same per-appearance
return. `p_appear` is measured over the window from a player's first
appearance to season end, so a January arrival reads as nailed rather than
half-available.

## Stage 3 — Optimisation

All integer programs, PuLP over CBC.

### `optimiser/squad.py::optimise_squad` — build a squad from scratch

Maximise `Σ score·(starting + captain) + bench_value_weight·score·bench`
subject to 15 players, £100m, 2/5/5/3, ≤3 per club, a legal XI. Used by the
cold start, wildcards and Free Hits. The bench term exists because otherwise
bench players are worth exactly zero and the solver fills those slots with the
cheapest bodies available.

### `optimiser/squad.py::optimise_starting_xi` — pick the XI

Squad fixed; choose 11 + captain + vice within formation limits. Bench order is
GK first, then outfield by projected points.

### `optimiser/transfers.py::evaluate_transfers` — the multi-period program

The most intricate piece. Binary variables per (player, week) for squad
membership, starting, transfer in, transfer out, captain; integer variables per
week for free transfers, hits and transfer count; a continuous bank balance.

Key constraints:

```
squad[p,w]   = squad[p,w-1] + in[p,w] - out[p,w]      # continuity
ft[w+1]     ≤ ft[w] - n_transfers[w] + hits[w] + 1    # allowance carry
ft[w+1]     ≤ 5                                       # banking cap
hits[w]     ≥ n_transfers[w] - ft[w]                  # hits are the overflow
bank[w+1]   = bank[w] + Σ sell·out - Σ cost·in         # money
bank[w]     ≥ 0                                       # affordability
```

Objective: points from starters and captain, plus the bench term, plus a
terminal value on unspent free transfers, minus 4 per hit, minus a fixed
switching cost per transfer.

Three things worth understanding:

- **`hits[w]` in the carry constraint is what makes hits legal.** Without it,
  `n_transfers ≤ ft` is forced every week.
- **The bank flow replaces a simple budget cap** and generalises it exactly:
  with no purchase prices, every player sells at current cost and `bank ≥ 0`
  reduces to `Σ cost ≤ budget`.
- **Hits are rarer than intuition suggests, correctly.** Deferring a move to
  next week's banked transfer costs only *one gameweek* of that player's
  advantage. So a hit is rational only when the per-gameweek gain exceeds 4.

### `optimiser/chips.py::recommend_chip`

Priority order, first to clear wins. On an active double gameweek, Bench Boost
and Free Hit get first refusal (they can exploit the whole squad); otherwise
Triple Captain leads.

| Chip | Fires when |
|---|---|
| Triple Captain | Captain's own projected points clear a low bar |
| Bench Boost | Active DGW and bench projection clears a threshold |
| Free Hit | Active DGW, or ≥5 squad players blank *this* gameweek |
| Wildcard | Rebuild gains enough over the horizon, squad old enough |

One of each per half-season, no carryover. Thresholds shrink as a half's
deadline approaches (`_panic_shrink`) so a marginal chip is spent rather than
lost, and Triple Captain is force-played at the half boundary as a last
resort. Where MC samples exist, the gate is P(gain ≥ threshold), not just the
mean.

### Scoring the objective — `optimiser/scoring.py`

`score = xpts · (1 + λ·(1 − EO/100)) + μ · xpts_var`

`λ` from risk posture (positive chases differentials, negative hugs the
template); `μ` weights variance. **At the default `risk_level = 0` both are
zero**, so the real bot's objective is plain expected points. The persona
cohort is what tests whether that is right.

## Stage 4 — The decision cycle

`agent/decision_engine.py::_run_decision_cycle`, shared by the real bot and
every persona — behaviour differs only by config and which tables it reads
and writes.

```
resolve gameweek → refresh projections → apply rumour discounts
  → load squad state (ids, bank, purchase prices, free transfers)
  → cold start?  yes → build initial 15, record, done
  → recommend chip
  → free hit? → optimise_squad     else → evaluate_transfers
       (a wildcard takes the evaluate_transfers path with wildcard_active=True)
  → optimise_starting_xi
  → settle bank and purchase prices
  → record transfers + lineup (+ chip)
```

`SquadState` carries the ledger between weeks: squad ids, `bank`,
`purchase_prices` (needed because FPL returns only half of a price rise),
and `free_transfers` rolled forward by `roll_forward_free_transfers` — shared
with the backtest so the two cannot drift.

## Stage 5 — Measurement

`scripts/backfill_decision_outcomes.py` scores finished gameweeks into
`actual_outcome`: auto-substitutions applied, vice-captain promoted if the
captain blanked, hits deducted. Runs for the real bot and every persona.

`simulation/analysis.py` reads it:

| Function | Answers |
|---|---|
| `axis_effect` | Which *value* of a swept parameter did best |
| `persona_season_summary` | How each persona did, including the paired delta vs baseline |
| `calibration` | Predicted vs actual per gameweek — the live bias instrument |

The cohort (`simulation/personas.py`) is a one-factor-at-a-time design: a
baseline at the real configuration, then seven axes swept individually.
**Read `delta_vs_baseline` and `gws_better_than_baseline`, not `total_actual`**
— personas share a season's luck, so paired differences carry far less
variance than absolute scores.

## Configuration

`config/strategy.py` is the single source of season-dependent truth: scoring
rules, DefCon, BPS weights, chip rules, transfer rules, squad structure, chip
thresholds, optimiser behaviour, departure risk, prior-league factors.

Values fall in three classes, and the file says which is which:

- **Rules** (scoring, squad size, hit cost) — facts, change only when FPL does.
- **Calibrated** (`mu_baseline`) — fitted against data, with the fit recorded.
- **Untuned heuristics** (`transfer_switching_cost`, `ft_terminal_value`,
  `bench_value_weight`, chip thresholds) — starting values pending real
  measurement. These are exactly what the persona cohort sweeps.

## Testing

`pytest` (774 tests) and `ruff check` are the gates; mypy is not enforced.
Both genuinely run as written — until 2026-08-18 a bare `pytest` collected
nothing (55 import errors) and only `python -m pytest` worked, and eight tests
were passing by reading the LIVE database. `conftest.py` now pins the suite to
a throwaway DB, so no test can reach `fpl_bot_v2.db`.
Two conventions worth preserving:

- **Pure logic separated from live I/O.** Scrapers keep their mapping
  functions pure and unit-tested, with only the network/browser call excluded
  from coverage.
- **Regression tests name the bug.** Most tests document the specific defect
  they prevent, several with the date and the symptom.

## A caveat about the backtest

`scripts/backtest.py` is a **second implementation** of the decision loop — it
calls the optimiser primitives directly rather than `_run_decision_cycle`.
That divergence is how a whole class of live-only defects survived a green
test suite. It is retained for projection-quality work (its `--naive-xi` mode
isolates projection from decision quality cleanly), but it is **not** the
validation instrument. The live cohort is.
