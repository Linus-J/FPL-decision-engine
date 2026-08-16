# Pre-GW1 critical review — architecture, validation, transfer planning, robustness

Date: 2026-08-16. GW1 deadline: **2026-08-21 17:30** (5 days).
Baseline: `pytest -q` → **564 passed**. Every defect below is invisible to the suite.

---

## Executive answer

**The architecture is sound.** Ingest → MC-scenario projections → ILP optimiser →
decision engine → submission is the right shape, the projection layer is
genuinely good (odds-implied λ, per-fixture joint sampling, leak-free as-of
reads, curse shrinkage), and the individual modules are careful and
well-documented.

**The validation loop is what's broken**, and every serious defect found is a
direct consequence of it. Specifically:

1. There are **three implementations of the decision loop** — `scripts/backtest.py`,
   `agent/decision_engine.py`, and the simulation path (which reuses the decision
   engine). The backtest does **not** call the decision engine; it re-implements
   the same sequence against the optimiser primitives. So the thing that gets
   measured is not the thing that runs.
2. **Every backtest artifact in `results/` is dated 2026-05-30.** Everything since
   — P3-5 scenario chip gating, the chip-rule and panic fixes, TC rebasing, curse
   shrinkage, `bench_value_weight`, `transfer_switching_cost`, P10 distributional
   projections, P11 prior-league cold start, risk-aware cold start, the current
   FT/wildcard code — has never been backtested. `ft_terminal_value=2.0` was
   picked from a grid search on 2026-05-30 against a system that no longer
   exists.
3. I re-ran the backtest against current code (see **Current measurement**
   below). The accumulated work **has** helped — 54.97 net pts/GW vs the stale
   artifact's 52.27 — but nobody could have known that, and the re-run surfaced
   a serious new calibration problem that three months of not measuring hid.

Everything in Part B below exists in the live path and not in the backtest. That
is not a coincidence — it is the mechanism.

---

## Part A — the meta-problem

### A1. The validated path ≠ the live path

`scripts/backtest.py` imports `optimise_squad`, `optimise_starting_xi`,
`evaluate_transfers`, `recommend_chip` directly and orchestrates them itself
(lines 508–660). It never imports `agent.decision_engine`. Divergences that
follow from this, each verified:

| behaviour | backtest | live |
|---|---|---|
| FT roll-forward | `min(5, max(1, ft - made + 1))` | `max(0, ft - made)` → **B1** |
| wildcard | `optimise_squad(...)` directly | `evaluate_transfers(wildcard_active=True)` → **B2** |
| projection horizon fed to ILP | full `horizon` GWs | **1 GW** → **B5** |
| `squad_age_gws` | tracked and passed | never passed (defaults 99) → **B4** |
| BGW/DGW detection | derived from fixture counts | `Gameweek.is_dgw/is_bgw` |
| injury-severity discount | not applied | applied |
| budget | `current_cost` recomputed each GW | frozen at 100.0 → **B7** |

`test_live_entrypoints_import.py` exists, so the import-time breakage class is
covered — but no test ever runs a live decision cycle.

### A2. The backtest is stale, and cheap to re-run

`results/` last written 2026-05-30. Re-running
`scripts/backtest.py --season 2025-26 --start-gw 6 --end-gw 38 --score-2627`
against current code takes ~25 minutes unattended. Three months of work went
unmeasured for want of one command. This should be a merge gate.

### A4. The projection is now badly over-confident *(new — the most important
finding from the re-run)*

| run | avg net pts/GW | mean bias (predicted − actual) | corr(pred, actual) |
|---|---|---|---|
| `v6b` (2026-05-30) | 52.27 | **−4.90** | 0.344 |
| current code (2026-08-16) | **54.97** | **+7.98** | 0.420 |

The bot's team score improved by +2.7 pts/GW (≈ +89 pts/season), and the
predicted/actual correlation improved. But calibration **flipped by ~13 points**:
the model now predicts 62.95 and delivers 54.97, over-predicting in **24 of 33**
gameweeks (MAE 13.6). Excluding chip weeks the bias is still +7.65.

This is not a re-scoring artifact. `projection/rescore.py` only swaps bonus
(standard scoring and DefCon were both already in force in 25/26), so predicted
and actual are on one basis.

Why it matters beyond accuracy: **every decision threshold in the system is
denominated in absolute points** — `hit_cost_points = -4`,
`transfer_switching_cost = 1.5`, `ft_terminal_value = 2.0`,
`wildcard_pts_gain_threshold = 25.0`, `bench_boost_min_bench_xpts = 20.0`,
`free_hit_single_gw_gain_threshold = 12.0`, `triple_captain_min_gain = 4.0`.
All of them are being compared against systematically inflated gains, which
biases the bot toward acting: transferring every week, and firing chips. The
re-run played 5 chips (2× TC, 2× WC, 1× FH) where `v6b` played 1 — some of that
is the chip fixes working as intended, some is likely threshold inflation.

Diagnosing the source of the +8 should come before any threshold re-tuning; a
naive-XI run (`--naive-xi`, fixed squad, no transfers or chips) isolates
projection quality from decision quality and is the right first probe.

### A3. Parameter tuning rests on that stale run

`results/grid_search/grid_summary.csv` (the source of `ft_terminal_value=2.0`):

```
ft_terminal_value, horizon, avg_net
0.5, 3 → 54.2      0.5, 5 → 54.0
2.0, 3 → 55.0      2.0, 5 → 53.6
3.0, 3 → 54.3      3.0, 5 → 54.8
5.0, 3 → 54.6      5.0, 5 → 54.8
```

A 1.4-point spread across a 10× parameter range is noise, not signal. The
banking incentive as formulated has essentially no measurable effect.

The fresh run confirms it behaviourally: **the bot took a transfer in 30 of 33
gameweeks and never banked past 5** (and past GW12, never past 1 — it spends its
FT every single week). It also took **zero hits across all 33 gameweeks**, which
is B3 showing up in the measurement. So even once B1 is fixed, the current
parameterisation produces "spend the transfer every week" — precisely the
short-sighted behaviour worth worrying about. Bench Boost was never played at
all.

---

## Part B — live-path defects

### B1. Free transfers never accumulate, then lock at 0 *(blocker)*

`agent/decision_engine.py:400` — `max(0, free_transfers - len(transfers_in))`.
Missing the `+1` weekly allowance and the cap at 5.
`ft = LpVariable(lowBound=1, upBound=5)` + `ft[0] == 0` ⇒ **Infeasible**, caught
and returned as an empty plan behind a `logger.warning`. After the first week in
which a transfer is made, the bot never transfers again. Verified: FT=0 →
Infeasible; FT=1/2/5 → 1/2/5 transfers.

### B2. Wildcard is structurally infeasible; the chip is burned for nothing *(blocker)*

`optimiser/transfers.py:155,160` — `ft[0] == 15` against `upBound=5`. Verified
Infeasible → empty plan. The decision engine still logs the chip and
`fpl_client` still sends `wildcard=True`. The backtest never touches this path.

### B3. Hits are impossible *(blocker)*

`ft[w+1] <= ft[w] - n_trans[w] + 1` together with `ft[w+1] >= 1` forces
`n_trans <= ft` every week. Verified: three available players worth +29 xPts/GW
each, FT=1 → 1 transfer, 0 hits. Separately the hit penalty sits inside the
per-player loop (`optimiser/transfers.py:221-229`), so one hit costs `4 × N`
(~900+) rather than 4 — masked today, would bite the moment the constraint is
fixed. `should_take_hit` is dead code.

### B5. The live planning horizon is one gameweek *(blocker — new)*

`run_projections` correctly builds and persists 3 GWs, then
`_run_decision_cycle` throws that away and calls
`get_latest_projections()`, which is `WHERE pp.gameweek = :gw` for a single
`next_gw` (`projection/pipeline.py:360-393`).

Consequences, verified:

- `evaluate_transfers` computes `H = len(gws[:horizon]) = 1`. The entire
  multi-period structure — FT carry, `ft_terminal_value`, planning a transfer
  for next week — is unreachable live. There is no banking behaviour to fix
  until this is fixed.
- `recommend_chip` takes `gws[:wildcard_eval_horizon_gws]` = 1 GW and compares
  the gain to `wildcard_pts_gain_threshold = 25.0`, a threshold meant for 5 GWs.
  The wildcard can essentially never fire live.
- Feeds B6.

This is a one-line class of fix (pass `run_projections`' returned frame through,
or give `get_latest_projections` a horizon), and it is the highest-leverage
change in this document after B1.

### B6. Free Hit fires on a phantom blank *(new)*

`agent/decision_engine.py:280-288` counts squad players with zero projected
points in any BGW in the lookahead. With only `next_gw` in `projections`, any
BGW at `next_gw+1` or `+2` has **no rows at all**, so `.sum() == 0` is true for
every player. Verified: `bgw_affected = 15/15`, and the Free Hit gate is
`bgw_affected_count >= 5`.

Independently of B5, there is a logic error here: the count is taken over a
*lookahead window* but the chip is played *this* week. A blank two weeks out
should not trigger a Free Hit now.

### B7. Budget is frozen at £100.0m; no bank or selling-price tracking *(new)*

`_load_squad_state` reads `budget` from the last decision-log row, and
`_run_decision_cycle` writes back the same `available_budget` unchanged. It is
seeded at 100.0 by the cold start and never moves. Nothing anywhere reads FPL's
`transfers.bank` or `picks[].selling_price` — `agent/fpl_client.py:61` uses
selling price only to build the submission payload, never the optimiser's
constraint.

Real FPL budget is `Σ selling_price(sold) + bank`, and selling price lags
`now_cost` after a rise (you keep half). So the ILP's `Σ now_cost ≤ 100.0`
constraint is wrong in both directions: it under-spends as the squad appreciates
and, if the squad depreciates, proposes transfers FPL will reject.

### B4. Live wildcard age gate is inert

`recommend_chip(squad_age_gws=99)` by default; the backtest passes the real
value, the decision engine does not. `wildcard_min_managed_gws = 6` never binds
live.

### B8. Dry runs pollute live state *(new)*

`_record_decision` stores `dry_run` on the row, but neither `_load_squad_state`
nor `_load_own_decision_log` filters on it. So:

- a `--dry-run` rehearsal overwrites the bot's notion of "my current squad";
- a dry run that recommends a chip writes a `decision_type="chip"` row, and
  `chips_used_this_season` counts it — **marking the chip permanently used**.

The live `decision_log` currently holds 4 lineup rows, all `dry_run=1`, no chip
rows, so no damage yet. This becomes live-destructive the moment real submission
starts and any rehearsal run happens afterwards.

### B9. There is no initial-squad submission path *(new — GW1-specific)*

The cold-start branch returns `transfers_in: []`. `submit_decisions` then does
`if transfers_in:` → skips `_submit_transfers` entirely and only PATCHes
`/my-team/{id}/` with picks. A lineup PATCH reorders and captains players
*already in the team*; it cannot bring 15 new players in. And
`_submit_transfers` `zip`s in/out pairwise, so it could not express an
initial-15 selection even if it were called.

Also `FPL_PASSWORD` is absent from `.env`, so `submit_decisions` would raise
before reaching the network on a live run.

**Practical implication for GW1: plan to enter the squad manually on the FPL
site.** The Telegram notifier prints the full XI, bench order, captain and vice,
which is enough to do that accurately. Treat the bot as a decision engine this
week, not a submission bot.

---

## Part C — machinery that is inert at the default configuration

Not bugs exactly, but the system is smaller than it looks, and several weekly
costs buy nothing:

- **The whole risk layer is off.** `mu = mu_baseline + risk_level*mu_range =
  0.0 + 0×0.08 = 0.0` and `lam = risk_level × magnitude = 0.0`. So
  `risk_adjusted_score(x, v, eo, lam, mu) ≡ x` exactly. `xpts_var` influences
  nothing, and `scenario_based_captain` short-circuits at `mu == 0` without
  touching the DB — P3-4 is entirely dormant live. (`mu_baseline` was
  deliberately calibrated to 0.0, so this is a consequence of a real result, not
  an accident — but it means the live bot is a plain expected-points maximiser
  and should be described as one.)
- **Ownership data is never consumed.** `run_weekly.py` runs
  `scripts/ingest_ownership.py` every week, but neither `decision_engine` nor
  `backtest.py` ever passes `ownership=` to `optimise_squad` /
  `evaluate_transfers` / `optimise_starting_xi`. Combined with `lam = 0`, the EO
  layer is doubly dead.
- `OPTIMISER.use_price_change_signals = True` is **read nowhere**. There is no
  price-change modelling at all.
- `evaluate_transfers(dgw_gws=...)` is accepted and **never used** in the body.
  DGW/BGW-aware transfer preference does not exist.
- `DGW.lookahead_gws = 6` is unused; DGW detection actually uses
  `transfer_planning_horizon_gws` (3).
- `CHIP_TIMING.wildcard_eval_horizon_gws = 5` exceeds the number of GWs that
  exist in the live frame (see B5).
- **The 100-persona simulation has no feedback loop.** It writes
  `sim_decision_log` for the dashboard and nothing reads outcomes back to inform
  the real bot. It also runs through `_run_decision_cycle`, so every persona
  inherits B1/B5/B6/B7 — it cannot currently serve as a shadow A/B either.
- `scripts/data_quality_gate.py` is never invoked by `run_weekly.py`; there is no
  pre-decision data validation in the live pipeline.
- `_get_current_and_next_gw` / `_get_current_season` query `gameweeks` with **no
  season filter**. Benign today (only 26/27 rows carry `is_current`/`is_next`),
  but the same shape as the bug `_get_wc_half_boundary` already had to fix.

---

## Part D — squad robustness and cold start

### D1. No bench value in the in-season transfer objective

`optimiser/squad.py` applies `bench_value_weight` (0.15) on
`selected[i] - starting[i]`, which is why the GW1 build produces a sane £17.5m
bench. `optimiser/transfers.py`'s objective scores only `starting` and
`captain` — bench quality is worth exactly zero to every in-season transfer, so
it will erode to fodder over the season. This is the "strong XI, indefensible
bench" failure mode directly.

### D2. Cold-start xPts is on a different scale from in-season xPts

`projection/cold_start.py:489` sets `xpts = ppg_played` = points **per
appearance**. `projection/assemble.py` produces a scenario mean across minutes
bands — an **unconditional** expectation including 0-minute scenarios.
`start_probability` is used only as a hard 0.4 filter, never as a multiplier. A
rotation risk is therefore valued identically to a nailed player with the same
per-appearance return.

Measured on the live GW1 build:

| build | XI mean P(start) | bench mean P(start) | overlap with current |
|---|---|---|---|
| current (per-appearance) | 0.828 (min 0.68) | 0.55 | — |
| × `starts_rate` | 0.938 (min 0.79) | 0.79 | 6/15 |
| × `appearances/38` | 0.955 | — | 6/15 |

Nine of fifteen players change. The current XI carries four starters below 0.80
(Muñoz 0.68, Stach 0.71, Bruno G. 0.71, Dewsbury-Hall 0.76).

Caveats: `starts_rate` is `P(minutes ≥ 60)` while `ppg_played` conditions on
`minutes > 0`, so multiplying by `starts_rate` overcorrects; `appearances/38` is
the matching estimator. The peer-bucket and prior-league tiers pool
per-appearance values too and need the same treatment. And the availability-
weighted build drops Haaland — a real strategic call, not a mechanical fix.

### D3. Fragility is not modelled

No minimum-availability constraint on bench players, no penalty for
concentrating the XI in low-P(start) players, no autosub modelling in any
objective (`_apply_autosubs` exists only in `scripts/backtest.py`, for scoring).
`bench_value_weight` weights bench *points*, a proxy for bench quality but not
for the probability the bench is needed.

Cheapest real guard before GW1: a minimum-P(start) floor on bench players
(~0.5 for outfield) — a one-line constraint in `optimise_squad`.

---

## Recommended order

**Before the GW1 deadline (5 days):**

1. **B9 awareness** — decide now that GW1 goes in manually. No code needed.
2. **D2** — decide deliberately whether cold-start xPts becomes
   availability-weighted. This is the only change that alters the GW1 squad, so
   it is the only one that *must* be settled this week.
3. **D3** — optional bench-availability floor if D2 is deferred.

**Before GW2 (the bot's first real transfer decision):**

4. **B5** — restore the multi-GW projection frame. Nothing about banking works
   until this lands.
5. **B1** — FT accounting. Without it the bot stops transferring after week one.
6. **B2** — wildcard `upBound`.
7. **B3** — FT slack + move the hit term out of the per-player loop.
8. **B6** — BGW counting, and gate Free Hit on the blank week, not the current one.
9. **B7** — read `bank`/`selling_price` from `/my-team/` and constrain against
   real budget.
10. **B8** — filter `dry_run` out of state reads.
11. **D1** — bench value in the transfer objective.

**Structural, and the thing that actually prevents recurrence:**

12. Collapse `scripts/backtest.py` onto `_run_decision_cycle` so the backtest
    exercises the live code path, with the season-replay differences injected
    rather than re-implemented.
13. Re-run the backtest and refresh `results/` as a merge gate. Re-tune
    `ft_terminal_value`, `transfer_switching_cost`, `bench_value_weight` and the
    chip thresholds *after* B1/B3/B5 land — every current value was fitted
    against a model that could not bank, could not take hits, and could not see
    past one gameweek.
14. Regression tests for the untested surface: `free_transfers=0`, banking across
    three gameweeks, `wildcard_active=True`, a hit clearing 4 points, and one
    end-to-end `_run_decision_cycle` against a fixture DB.

**Deferred:** B4, C items, ownership wiring, price-change modelling.
