# Phase 3 — Decision layer (rank-aware)

Scoped 2026-07-26, grounded in a code-level survey (not assumptions) of `optimiser/*`,
`data/models.py`, `projection/pipeline.py`, and `projection/assemble.py`. Per
`plan/v2-build-plan.md` §5/§8: reuse `optimiser/transfers.py`'s multi-period ILP
structure, change the objective and inputs. **Exit gate:** walk-forward vs
benchmarks (avg manager ~55–57, frozen template, v1, top-10k ~63), reporting a
**simulated final-rank distribution**, not mean points.

## Current state (verified in code, not assumed)

| Area | State |
|---|---|
| `optimiser/squad.py` / `transfers.py` | Pure `Σ xpts·(starting+captain)` ILP. `xpts` is a single point estimate. No EO, no variance, no covariance term anywhere. Captaincy is linear argmax (doubling the top scorer is always weakly optimal under a linear objective) — not scenario-based. `transfers.py`'s multi-period structure (FT banking, hits, wildcard accounting) is real and worth keeping per the plan. |
| `optimiser/chips.py` | Threshold heuristics vs fixed constants (`CHIP_TIMING`). WC/FH paths call `optimise_squad` (mean-only). Zero scenario-EV machinery. |
| EO (effective ownership) | **100% unbuilt.** No `ownership_snapshots` table, no ingestor, no reads anywhere in the codebase. Was deferred into GW1 scope (v2-build-plan §8 decision 6) but never assigned a task or schema. |
| `projection/pipeline.py` (LIVE 2026-27 serving) | Still imports `points_model`/`minutes_model`/`cs_model` — **not** `assemble.py`. `_persist_projections` writes `xpts`/`start_probability`/`cs_probability` only; `xpts_mean`/`xpts_var` are left at the column default `0.0` on every live-written row. P10's MC engine is backtest-only (confirmed: `assemble` is imported by `scripts/backtest.py`, not `pipeline.py`). |
| `data/models.py::ProjectionSample` | Schema fully defined (`player_id, gameweek, season, scenario_id, xpts`), indexed for exactly this covariance use case (P0 built it for this). **Nothing writes to it** outside one synthetic round-trip test. `assemble.sample_fixture()` produces the right per-scenario arrays internally but `assemble_gw_projections()` collapses them to mean/var and discards the samples before any caller could persist them. |
| `config/strategy.py::OptimiserConfig` | Has `max_ownership_differential`/`risk_mode` fields already sitting there — **dead**, read nowhere. Looks like a placeholder from an earlier design pass. |
| Tests | None for `optimiser/*` at all — Phase 3 test coverage starts from zero. |

**Implication:** the live bot cannot use a variance/covariance-aware objective today even
if the optimiser math were rewritten — the only path that ever produces real
`xpts_var` (`assemble.py` via the backtest harness) isn't wired into live serving.
Phase 3 has a real prerequisite the plan text doesn't call out explicitly.

## Task graph (proposed)

### P3-0 — Wire `assemble.py` into `pipeline.py` (live serving)  ✅ DONE (`86e4d2b`)
`run_projections()` now calls `assemble.assemble_gw_projections` instead of the old
`points_model`/`cs_model`/`minutes_model.predict_batch` combo — real `xpts_mean`/
`xpts_var` are written, not the inert `0.0` default. Built the live-horizon plumbing
`assemble.py` needed but never had (P-FIX): `_build_live_fixture_context` (fixtures
table + players' current team, since an unplayed fixture has no `player_gw_stats`
row for the backtest-style lookup) and `_load_live_match_odds` (raw 1X2/O25 from
`fixture_odds`, latest fetch at/before each target GW's own deadline — same
leakage-free posture as `features.load_live_odds_asof`). Moved the shared
`load_all_stats` query into `assemble.py` so backtest and live can't diverge.
Removed the now-dead `_get_team_fixture_count`/`_apply_dgw_bgw_multipliers`/
`_precompute_cs_probabilities`/`_load_player_recent_stats`.
**Known, documented, NOT solved here:** GW1 cold start — an unplayed season has
zero `player_gw_stats` rows for `assemble.py`'s rolling history, so `run_projections`
detects this and returns empty rather than crashing (verified live against the real
DB, 2026-07-26, pre-season). Real GW1 projections still need T7/P11, tracked
separately. Suite 188/188 (6 new tests, temp-DB pattern).

### P3-1 — Persist MC scenario samples (`ProjectionSample`)  ✅ DONE (`2dca601`)
`assemble_gw_projections` gained `persist_samples`/`season` params — when on, each
fixture's raw per-scenario draws (previously computed then discarded after the
mean/var reduction) get bulk-written. **Design point verified by test:** each fixture
gets a disjoint `scenario_id` range within its gameweek (offset by `n_scenarios` per
fixture already assembled that GW) — the schema has no fixture-grouping column, so
without this a naive join on `(season, gameweek, scenario_id)` would accidentally
correlate two players from DIFFERENT matches who happen to share a raw scenario
index; only real teammates (same fixture) ever share a range now. Off by default
(the 33-GW backtest walk-forward calls this hundreds of times and doesn't want tens
of thousands of DB rows per call) — `pipeline.py` turns it on, tied to the same
`persist` flag callers already use. **Storage/retention policy flagged, not solved**
(same open item P0 originally noted — this table will grow fast under live use).
Suite 191/191 (3 new tests, monkeypatched `sample_fixture` to isolate the
persistence/offset wiring from MC-sampling correctness, already covered elsewhere).

### P3-2 — EO ingestion
New `ownership_snapshots` table (per v2-build-plan §3.2: `player_id, snapshot_ts,
overall_selected_pct, top10k_selected_pct, captaincy_pct_overall/top10k`). Source:
FPL API top-10k sampling or a LiveFPL scrape (both free, per the resolved budget
decision). Unlocks `differential_value ≈ your_pts − EO·field_pts`.

### P3-3 — Objective v1 rewrite
`optimiser/transfers.py` (keep its multi-period/FT-banking/wildcard structure) +
`optimiser/squad.py`: change the linear objective to
`E[pts] + λ·differential_value + μ·variance`, reading EO (P3-2) and `xpts_var`/
covariance (P3-1) as real inputs instead of unused config placeholders.

### P3-4 — Scenario-based captaincy
Replace the linear-argmax captain pick with selection over sampled scenarios
(biggest weekly variance lever per the plan) — needs P3-1's real samples.

### P3-5 — Chips rework
`chips.py` from threshold-vs-constant to scenario EV (`P(chip pays off)` over
sampled scenarios), reusing P3-1's persisted samples.

### Gate
Walk-forward vs benchmarks; report the simulated final-rank distribution, not mean
points — a risk-seeking bot can have a slightly lower mean with a fatter right tail
and still be the better rank-optimising choice.

## Open sequencing question
P3-0/P3-1 (live-serving + sample persistence) and P3-2 (EO ingestion) are independent —
neither blocks the other. P3-3 needs both. Objective v1 (linear, additive terms) is
mechanically straightforward once its inputs exist; objective v2 (true scenario-based
stochastic programming) is a bigger structural lift and explicitly framed as an
"upgrade" in the plan, not a v1 requirement.
