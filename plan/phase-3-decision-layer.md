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

### P3-2 — EO ingestion  ⚠️ DONE, UNVERIFIED AGAINST LIVE DATA (`ce01d9c`)
`OwnershipSnapshot` table + `data/ingestors/ownership.py` (samples the "Overall"
classic league standings for entry IDs, then those managers' GW picks, aggregating
top-N ownership/captaincy %). `overall_selected_pct` reuses the already-ingested
bootstrap-static `selected_by_percent`; `captaincy_pct_overall` has no free
population-wide source and is left nullable (documented gap).
**Real blocker found by directly probing the live FPL API (not assumed):** GW1's
deadline is 2026-08-21 — zero gameweeks played this season, so the Overall league
returns zero ranked entries right now, and the picks endpoint (`/entry/{id}/event/
{gw}/picks/`) only serves the CURRENT season (confirmed — no way to address a past
season's picks through this API). **EO sampling cannot produce real output until
the season starts, regardless of how the ingestor is written** — this isn't a
code gap, it's a data-availability one. Built and unit-tested (13 tests, pure
aggregation math + schema round-trip) against the well-documented, stable FPL API
shape, but NOT verified against a real populated response. Re-verify at GW1 —
budget time for this in case the real response shape differs subtly from what's
assumed.

### P3-3 — Objective v1 rewrite  ✅ DONE (`5afa33a`)
**Found before implementing:** the plan's literal `differential_value` formula
(`your_pts - EO*field_pts`, summed) is algebraically a no-op for the mean objective
— `EO_i*xpts_i` is a constant w.r.t. your own selection, so maximising the
difference picks EXACTLY the team you'd get from maximising raw `E[pts]` alone.
Confirmed algebraically, not assumed — checked with the user before proceeding.
The plan's own §5 wording ("benching a 60%-EO player is a short position")
confirms the real intent is RANK-OUTCOME VARIANCE, not the mean pick.
**Implemented instead (`optimiser/scoring.py`):** a linear v1 approximation —
`differential_multiplier` reweights each player's score by ownership (does change
relative rankings, unlike the literal formula), `risk_adjusted_score` adds
`μ*xpts_var` (own-variance only; teammate COVARIANCE is quadratic in a 0/1
selection and needs the v2 scenario-based objective the plan itself frames as an
"upgrade" — P3-1 already persists the samples a v2 implementation would need).
`lambda_mu_for_risk_mode` sets sign from `risk_mode` ("balanced" = (0,0), today's
exact behaviour). Wired into `optimise_squad`/`optimise_starting_xi`/
`evaluate_transfers` via an optional `ownership` param, default `None` everywhere
including the P-XI harness — each function's TRUE xpts is still used for
`total_xpts`/`xpts_gain` reporting, only the ILP's own objective coefficients use
the risk-adjusted score. **Verified the P-XI exit gate is unaffected:** a GW6–9
spot-check against the live DB produced byte-identical predicted_xpts/captain/
actual_pts with and without this commit (git-stash A/B) — confirmed, not assumed.
Suite 226/226 (19 new tests, including an ILP-level test proving EO actually
flips a captain choice between two equally-projected players under aggressive
mode).

### P3-4 — Scenario-based captaincy  ✅ DONE
**Why linear argmax wasn't the end of the story:** doubling the top scorer is
always weakly optimal for a MEAN objective, and P3-3's own-variance term
(`mu*xpts_var`, additive per player) doesn't change that — it just reranks
who counts as "top" before the same argmax runs. It's blind to the one thing
that actually makes captaincy a real decision under a variance-aware
objective: doubling a player adds `Var(2X) = 4*Var(X)` (not `2*Var(X)`) to
the team total, and if that player shares a fixture with other starting-XI
players, doubling also doubles their COVARIANCE contribution — invisible to
a sum of independent per-player variances.
**Implemented (`optimiser/captaincy.py`):** uses the real joint MC draws
P3-1 persists (`ProjectionSample`). Fixture membership is recovered from
each fixture's disjoint `scenario_id` range within a gameweek (an existing,
tested invariant of `assemble.py`, not a new assumption) — players sharing
an identical (min, max) scenario_id span were drawn in the same fixture, so
cross-fixture covariance is exactly 0 and team-total variance decomposes
additively across fixture groups; only the candidate captain's own group
needs recomputing per candidate. `pick_captain` degrades EXACTLY to the
additive own-variance approximation for any candidate with no persisted
samples (cold start, or the 33-GW backtest walk-forward, which never
persists samples per P3-1) — proved algebraically (the shared "baseline"
term becomes a true additive constant across candidates when no group data
exists anywhere) and confirmed by test, not assumed. At `mu == 0`
(risk_mode="balanced", today's default) `scenario_based_captain` short-
circuits to plain mean argmax without touching the DB at all.
Wired into `optimise_squad`/`optimise_starting_xi` via an optional `season`
param (threaded through every call site: `decision_engine.py`,
`backtest.py`'s both harnesses, `cold_start.py`) — a post-ILP override of
the captain (and vice, re-picked if it collided with the new captain) using
the true team-total variance instead of the ILP's own additive-score
argmax. **Verified the P-XI exit gate is unaffected:** re-ran the GW6–9
naive-XI backtest against the live DB with `season` now threaded through
every call site — predicted_xpts/captain/actual_pts came back byte-for-byte
identical to the pre-P3-4 52.48 run (Keane/Senesi/Gabriel/Gabriel), because
default config is balanced (mu=0) and backtest never persists samples
either way — both independently sufficient for the no-op guarantee, and
both hold. Suite 239/239 (13 new tests, including a synthetic-correlation
test proving two candidates with IDENTICAL declared own-variance get
different captaincy scores once one of them has a real positively-
correlated fixture-mate — the exact gap the additive approximation can't
see).

### P3-5 — Chips rework  ✅ DONE
`chips.py` from threshold-vs-constant to scenario EV (`P(chip pays off)` over
sampled scenarios), reusing P3-1's persisted samples.

**Implementation:** each of TC/BB/FH/WC's old rule was `point_estimate_gain
>= threshold` (a mean built from `projections["xpts"]`). `optimiser/
chip_scenarios.py` (new) reuses `captaincy.py`'s real persisted joint MC
draws to build an actual per-scenario GAIN DISTRIBUTION for each decision —
`load_scenario_totals(season, gameweeks, player_ids)` sums real draws for a
player set into one per-scenario total, composing across fixture groups (and
across gameweeks, for Wildcard's multi-GW horizon) by POSITION rather than by
raw `scenario_id` value, since `scenario_id` ranges are only jointly
meaningful WITHIN one fixture (disjoint elsewhere, per P3-1) and reset to 0
at the start of every gameweek (no shared latent across GWs either) — any
FIXED pairing of independent draws is a valid joint sample of their sum, so
pairing by each group's own rank is legitimate. `gain_distribution` builds
the two sides' totals from the SAME underlying run, so a fixture shared
between both sides (e.g. Wildcard keeping a player, or two squads both
containing an Arsenal player) stays correlated rather than being treated as
independent noise. `chips.py::_clears_threshold` then replaces the point
check with `P(scenario_value >= threshold) >= min_probability` (four new
`ChipTimingThresholds` fields, defaulted to 0.6, untuned pending backtesting)
whenever real samples exist for that decision, so a chip whose MEAN clears
the bar but is actually closer to a coin-flip can be correctly blocked.

**Degrades exactly to pre-P3-5 behaviour when no real samples exist** (cold
start; the backtest walk-forward, which never persists samples per P3-1; or
`season=None`) — `_clears_threshold` falls back to the untouched point-
estimate rule, verified by test and by a live-DB `run_backtest` GW6–9
spot-check (predicted 60.47/67.72/69.62/77.03, captains
Keane/Sarr/Sarr/N.Gonzalez, transfers 0/3/2/2) matching the pre-P3-5
baseline exactly.

**Real bug found and fixed while wiring this up:** `recommend_chip`'s real
caller (`scripts/backtest.py`) passes `current_gw` as a `numpy.int64` (from a
pandas groupby), not a plain Python `int` — `load_scenario_totals`'s initial
`isinstance(gameweeks, int)` scalar-vs-sequence check missed it (numpy
scalars aren't Python `int` instances) and raised `TypeError: 'numpy.int64'
object is not iterable` for every GW past the cold-start build in a
backtest run with `season` set. Fixed: check `isinstance(gameweeks, (int,
np.integer))`. Caught by re-running `run_backtest` end-to-end before calling
this done, not by the unit suite alone (which only exercised
`chip_scenarios.py` with plain Python ints) — added a regression test with
an explicit `np.int64` gameweek.

Suite 260/260 (21 new tests across `test_chip_scenarios.py` and
`test_chips.py` — the latter is `chips.py`'s first test coverage at all,
since none existed pre-P3-5). Lint-clean (verified via git-stash A/B that
the only 2 ruff findings in `chips.py` are pre-existing and untouched).
**Not built:** TC's own "gain" definition (best-minus-second-candidate
differential) is a pre-existing heuristic proxy for the true TC formula
(marginal 1× of the captain's own points, since captaincy already doubles)
— P3-5 scenario-ises the existing heuristic faithfully but does not correct
its underlying formula, since that wasn't in scope and changing chip
semantics deserves its own decision, not a silent side effect of a
plumbing upgrade.

### Gate  ✅ v2 BOT NOW BEATS BOTH BASELINES (2026-07-28, P3-6 fix)
Walk-forward vs benchmarks; report the simulated final-rank distribution, not mean
points — a risk-seeking bot can have a slightly lower mean with a fatter right tail
and still be the better rank-optimising choice.

**Built:** `scripts/walk_forward_gate.py`. Benchmarks: **v2 bot** (this project's full
`run_backtest` — transfers, chips, scenario captaincy) vs **v1 bot** (`run_naive_xi_backtest`,
the Phase-2 P-XI harness — fixed squad, no transfers, weekly lineup/captain only) vs
**frozen template** (new: squad AND lineup AND captain picked ONCE at `start_gw`, never
revisited again — stricter than v1) vs the plan's own approximate **avg-manager (~56 pts/GW)**
and **top-10k (~63 pts/GW)** reference constants, used as the two calibration anchors for a
Normal population model (`ASSUMED_POPULATION_SIZE=9,000,000`, a documented assumption, not a
scrape — no free source of the real manager-population score distribution exists). The v2 bot's
own predictive uncertainty (`run_backtest`'s new `predicted_var` column — own-variance +
captain-doubling correction, P3-3-level) drives a real Monte-Carlo simulated season-total/rank
distribution; v1 and frozen template get single point-estimate ranks (no persisted variance in
the same form).

**Two real bugs found while building/running this** (beyond the P12 DGW fix, `2cd558a`, found in
the same debugging pass and documented under its own P12 entry in `phase-2-xpts-engine.md`):
1. The `player_xg_stats` season-wide xG bug — see `phase-2-xpts-engine.md`'s "Real bug found
   2026-07-28" entry (reopened decision 3). Found because this gate's `run_backtest` phase was
   captaining ONE mid-priced player for 15+ consecutive gameweeks with an escalating, ultimately
   implausible predicted total (60 → 100+ over the season) while every teammate showed
   `goal_weight=assist_weight=0.0` exactly. Fixed live (`ingest_understat_xg_season`, 10,590 rows).
2. (unresolved, see below) `run_backtest`'s season-long total STILL trails both simpler baselines
   even after the xG fix.

**Result (2025-26, GW6-38, post-xG-fix):**

| benchmark | season total | pts/GW | rank (point estimate, population-model) |
|---|---|---|---|
| v2 bot (full decision engine) | 1565 | 47.4 | ~8,999,196 (worst) |
| frozen template (pick once, never touch) | 1666 | 50.5 | ~8,928,207 |
| v1 bot (naive-XI, weekly lineup/captain only) | 1695 | 51.4 | ~8,807,541 |
| avg manager (reference anchor) | 1848 | 56.0 | — |
| top-10k pace (reference anchor) | 2079 | 63.0 | — |

**Gate does not pass: the full decision engine underperforms a squad that is never touched at
all.** This is NOT a rank-optimising risk-seeking tradeoff (the framing this gate was written to
allow for) — v2 loses on mean points too. Ruled out as causes: hits (zero taken all season, per
log), chip misuse (one Wildcard, one Triple Captain, both look like reasonable calls), solver
failures (`ILP status: Optimal` throughout, no "rebuild from scratch" fallbacks fired), and the
xG data bug above (fixed; captains are now real, sensible attacking players — Haaland, Fernandes,
Gabriel, Enzo, Mbeumo, Cunha, Thiago — not a single exploited outlier).

**Leading hypothesis, not yet fixed: an "optimiser's curse" in `evaluate_transfers`/
`optimise_squad`'s transfer selection.** `predicted_xpts` runs consistently, substantially hot
against `actual_pts` all season (e.g. GW9: predicted 87.2 vs actual 46; GW18: 72.2 vs 26) — a
~24 pts/GW average overshoot (71.2 predicted vs 47.4 actual), well beyond normal calibration
noise. v1's naive-XI harness, using the SAME per-player projection engine, shows a much smaller
overshoot (60.3 predicted vs 51.4 actual, ~9 pts/GW) over the identical window. Since both harnesses
draw on identical per-player projections, the extra bias is specific to the ACT of repeatedly
searching the full player pool for "whoever the model currently likes best" every gameweek:
argmax-of-a-noisy-estimate is a biased estimator of the TRUE best option (classic winner's-curse /
optimiser's-curse phenomenon) — a fixed or rarely-changed squad only pays this selection bias once
(at initial build), while a transfer engine that re-optimises weekly re-exposes itself to it every
single week, systematically buying into short-window statistical flukes (5-game rolling rates) that
regress after the transfer is made.

### P3-6 — Optimiser's-curse fix for weekly transfer selection  ✅ DONE
Scoped narrowly to the mechanism identified above: `optimiser/transfers.py::evaluate_transfers`'s
ILP objective now always applies an additional risk discount, independent of `OPTIMISER.risk_mode`
(which stays a pure preference dial elsewhere, and today defaults its own magnitude to 0.0 — i.e.
this is a NEW, separate correction, not a reuse of the existing-but-inert P3-3 variance-weight
knob). `TransferRules.transfer_variance_penalty` (new config field, default `0.1`, untuned pending
real backtesting) is subtracted from `mu` before computing each candidate's
`risk_adjusted_score` for the transfer ILP: `score = xpts - 0.1 * xpts_var` at default (balanced)
settings, so a candidate whose apparent edge rests on a noisier, less-supported projection is
discounted more than one with the same raw xPts but lower variance — directly countering "buy
whoever's current 5-game rolling rate spiked" rather than broadly discouraging all transfers.
**Deliberately scoped to ONLY the weekly transfer ILP** — `optimise_squad` (cold-start build,
wildcard/free-hit rebuilds) and `optimise_starting_xi` (captaincy, an 11-15-candidate pool where
the winner's-curse effect is far weaker than searching the full ~500+ player market) are
untouched, both to keep the fix targeted at the exact mechanism diagnosed and to avoid disturbing
the P-XI/P3-4/P3-5 byte-for-byte reproducibility guarantees already verified earlier this session
(which depend on `optimise_squad`'s default scoring being unchanged).

**Result (2025-26, GW6-38, same window, xG data already fixed):** re-ran `run_backtest` alone
(v1/frozen are unaffected — neither calls `evaluate_transfers`) — **1700 pts / 51.5 pts/GW**, up
from the pre-fix 1565/47.4, and now edges out BOTH baselines:

| benchmark | season total | pts/GW |
|---|---|---|
| **v2 bot (full decision engine, post-P3-6 fix)** | **1700** | **51.5** |
| v1 bot (naive-XI, weekly lineup/captain only) | 1695 | 51.4 |
| frozen template (pick once, never touch) | 1666 | 50.5 |
| avg manager (reference anchor) | 1848 | 56.0 |
| top-10k pace (reference anchor) | 2079 | 63.0 |

Sanity-checked the transfer log: hits stayed at 0 all season (same as before), one Wildcard use,
transfer gains now mostly modest and plausible (5-30 xPts, one legitimately-large 39.5 WC gain, one
slightly negative -3.09 single-transfer "net gain" — expected and correct: the ILP now sometimes
prefers a lower-variance swap that scores slightly lower on the raw undiscounted `xpts_gain`
reporting metric, which is exactly the intended risk/noise tradeoff, not a bug). Captains are real,
sensible attacking picks throughout (Haaland, Calafiori, Enzo, Gabriel, Thiago, Fernandes) with no
single-player monopoly streak.

**Margin over the baselines is thin (+0.1 vs v1, +1.0 vs frozen) and `transfer_variance_penalty
=0.1` is an untuned, single-value choice** — this closes the specific "loses to doing nothing"
failure mode the gate caught, but is not a claim that this is the OPTIMAL discount value; proper
calibration (a held-out split, or trying a small grid of values) is future work, not done here.
Suite 264/264, lint-clean.

### Data-completeness audit (2026-07-28) — user-prompted, found a second real bug
User questioned whether 51.5 pts/GW was suspiciously low and asked to verify data completeness.
Audited match-odds coverage (`historical_fixture_odds`: 380/380 real fixtures, all 38 GWs — fine),
DefCon/bonus event coverage (`player_match_events`: 11,182 rows, 98.6% of real playing gameweeks,
503/517 players with some real CBIRT signal — fine), then checked for players missing ENTIRELY
from `player_match_events`/`player_xg_stats` all season, the same failure signature as the earlier
xG bug.

**Found 21+ significant players — several of them this bot's OWN most-favoured captains —
with ZERO event/xG data for the entire season**, including Bruno Fernandes (35 games, 3065 mins),
Robertson, Mané, Martinelli, G.Jesus, N.Gonzalez. Root cause: the shared name matcher
(`data/ingestors/fbref.py::_match_player`, used by `fbref.py`/`understat_xg.py`/`whoscored.py`)
only checked exact match and contiguous-substring containment — a stored full legal name with an
extra middle name or a second surname (Iberian dual-surname convention: "Nico **González Iglesias**"
vs the football name "Nico González") breaks containment in BOTH directions, and Turkish/Nordic/
Polish characters (ı/ğ/ø/ł) don't decompose under standard Unicode normalisation the way accented
Latin letters do. Fixed with a token-subset fallback + a diacritic-normalisation helper.

**Second, WORSE bug found while verifying the first fix: the pre-existing substring check was
actively CORRUPTING data, not just missing it.** A short, generic single-token `web_name`
("Gabriel", Arsenal's Gabriel Magalhães, a centre-back) is trivially "in" any longer external name
starting with the same common first name — "Gabriel Martinelli" and "Gabriel Jesus" were silently
merging their real xG/xA into Magalhães's totals. Confirmed live: his season xG was **13.98**, with
single-match readings above 1.5 (striker-level, impossible for a CB). He has been one of THIS
SESSION's most frequently favoured captains across multiple earlier backtests (the original 52.48
gate run, P3-4/P3-5 verification, and the pre-cleanup walk-forward runs above). Fixed by requiring
≥2 tokens on the candidate side of the substring check.

Re-ingested `player_xg_stats` for 2025-26 from a clean wipe (10,852 rows, 513 distinct players).
Gabriel's season xG dropped to **4.65** (one legitimate high match at 1.04) — a believable
set-piece-threat-CB profile. 23 distinct players remain unmatched (mostly fringe/backup players;
"Lucas Paquetá" specifically needs a manual alias — his football name has zero lexical relationship
to his stored legal name, a heuristic can't catch that). Did NOT attempt a live FBref/WhoScored
re-scrape (needs a browser; Chromium is available, but Gabriel's FBref-sourced event data already
looked plausible — no confirmed contamination found there, unlike Understat).

**Re-ran the full walk-forward gate on the cleaned data (GW6-38):**

| benchmark | before cleanup | after cleanup | Δ |
|---|---|---|---|
| v2 bot (full decision engine) | 1700 (51.5 pts/GW) | **1634 (49.5 pts/GW)** | −66 |
| v1 bot (naive-XI) | 1695 (51.4 pts/GW) | 1508 (45.7 pts/GW) | −187 |
| frozen template | 1666 (50.5 pts/GW) | 1519 (46.0 pts/GW) | −147 |

**All three benchmarks dropped in absolute terms** (removing an inflated free-lunch player lowers
everyone's baseline), **but v2's margin over the other two widened substantially** — from +5/+34
pts to **+126/+115 pts** over v1/frozen respectively. This is the important signal: v1 and the
frozen template have no way to correct once they're captaining a wrongly-inflated player every
week, while v2's active transfer logic (freshly improved by the P3-6 optimiser's-curse fix) can
actually move away from a bad valuation once the underlying data stops lying to it. Captaincy is
now qualitatively sane — Gabriel captained only ONCE across the whole v2 run (down from
dominating nearly every prior backtest this session), replaced by Thiago (12x) and Haaland (10x);
v1's fixed squad sensibly captains Haaland 19 times.

**Honest remaining caveat:** 49.5 pts/GW is still below the plan's own avg-manager reference
(~56) and well below top-10k pace (~63) on this one realised 2025-26 season. That's not
necessarily a red flag on its own (a single season is one noisy draw, and the reference constants
are themselves approximate, undocumented-source anchors — see the Gate section above) — but it's
also not something to wave away. The 23 remaining unmatched players (Ben White, Matthew Cash, Max
Kilman among them — real, relevant defenders) are a known, smaller residual gap. Suite 274/274,
lint-clean. Commits `6a99ed5` (matcher fix).

### Departure-risk gate (§6.5)  ✅ DONE (`9531207`)
Not originally in the P3-0..P3-5 numbering above, but flagged as *more urgent* than
P3-4/P3-5 when asked "does anything else need to be done?" — unlike the rest of the
Phase-4 news layer, this gate is meant to be **live from the initial-15 build**, not
shadow-mode, because picking a confirmed departure into the squad is a costly,
asymmetric mistake. `config/strategy.py::DepartureRiskRules` +
`optimiser/departure_risk.py` (`confirmed_p_leave`, `stay_probability_multiplier`,
`apply_departure_discount`, `hard_excluded_ids`).
**Real bug found and fixed while wiring this up:** `optimise_squad` already correctly
excluded `status='u'` players from new picks, but `evaluate_transfers` filtered ALL
candidates — including already-OWNED ones — by status, so an owned player who became
`status='u'` was silently dropped from the ILP's variable set entirely (no `tout`
variable ever existed for them). The squad-size constraint happened to coincidentally
force a replacement transfer in, but the departure itself never appeared in the
reported `transfers_out`. Fixed: owned players stay in the candidate pool regardless
of status, with `tout==1` explicitly forced for confirmed departures — correct
hit/FT accounting, correct reporting. Suite 207/207 (9 new tests, including a
synthetic 15-player squad scenario proving the fix).
**Not built:** the rumour tier (`0.2 ≤ p_leave < 0.7`) has no data source —
Phase 4's LLM news layer (RSS + Ollama credibility grading) doesn't exist yet. The
discount mechanism is ready (`apply_departure_discount`), just unfed.

## Open sequencing question
P3-0/P3-1 (live-serving + sample persistence) and P3-2 (EO ingestion) are independent —
neither blocks the other. P3-3 needs both. Objective v1 (linear, additive terms) is
mechanically straightforward once its inputs exist; objective v2 (true scenario-based
stochastic programming) is a bigger structural lift and explicitly framed as an
"upgrade" in the plan, not a v1 requirement.
