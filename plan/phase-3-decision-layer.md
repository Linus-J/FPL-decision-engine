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

### Data-completeness audit, part 2 (2026-07-28) — systematic ingestor sweep + automated gates

User pushed back ("this is a huge problem, check for data issues throughout the build") and asked
for three things: a systematic audit of every ingestor for the same bug classes, automated
data-quality gates going forward, and resolution of the 23 remaining unmatched Understat players.

**23 unmatched players, root-caused and mostly fixed (21/23).** Broke into distinct classes, each
verified against the real stored DB record before fixing (no guessing):
- **Hyphen-vs-space disagreement** (3 players: Smith-Rowe, Aït-Nouri, Ben *Gannon*-Doak) — sources
  disagree on hyphenation for compound names. Fixed generically: `_normalize_name` now flattens
  hyphens to spaces, so the existing token-subset fallback sees each part as its own token.
- **Reversed token-subset** (1 player: "Hamed **Junior** Traore") — the *external* source has an
  extra given name ours doesn't, the mirror image of the original Bruno Fernandes case. The
  token-subset fallback only checked one direction; added the reverse, with a uniqueness guard
  (ambiguous matches return `None` rather than guessing — the same collision-safety principle as
  the ≥2-token guard from the Gabriel fix).
- **Serbo-Croatian đ/Đ mistransliteration** (1 player: Djordje/**Đorđe** Petrović) — the existing
  translit table mapped đ→"d", dropping the "j" sound entirely ("Dorde" instead of "Djordje").
  Corrected to đ→"dj" (the standard latinization), which resolves the case with zero special-casing.
- **Nickname/spelling variants with no safe generic rule** (16 players: Ben *White*↔Benjamin, Matty
  *Cash*↔Matthew, Joe *Gomez*↔Joseph, Max *Kilman*↔Maximilian, Dan *Ballard*↔Daniel, Fer
  *López*↔Fernando, Yeremy *Pino*↔Yeremi, Yarmoliuk↔Yarmolyuk, Nayef *Aguerd*↔Naif, Abdukodir
  *Khusanov*↔Abduqodir, Trey *Nyoni*↔Treymaurice, Ollie *Scarles*↔Oliver, Alex *Jiménez*↔Alejandro,
  Josh *King*↔Joshua, Lucas Paquetá↔his full legal name, Lesley↔Chimuanya Ugochukwu). Added a small
  curated alias dict (`_KNOWN_ALIASES` in `fbref.py`) mapping each verified external spelling to the
  exact normalized form already in our name_map — deliberately NOT a fuzzy/edit-distance matcher,
  since a low-confidence nickname guess risks reintroducing exactly the Gabriel-collision class of
  bug this whole audit started from.
- **Left unmatched, on purpose** (2 players): "Jota Silva" — our DB's only "Jota" record is a
  *different* real player (Diogo Jota), and aliasing the two would risk a genuine collision, so it's
  left unmatched rather than guessed. "Mathis Cherki" — Understat's own data doesn't cleanly resolve
  to a real current PL player (our only "Cherki" is Rayan, a different first name); flagged as a
  source-side anomaly for manual review, not fixed blind.

New tests: `tests/test_fbref_name_matching.py` (+8: hyphen flattening, đ/Đ, reversed-subset match +
its ambiguity guard, alias resolution) plus regression coverage below.

**A second, actively-corrupting name-matcher bug found live in production code, not just the
backtest.** `data/ingestors/understat.py` — a second, OLDER Understat ingestor, separate from
`understat_xg.py` — carries its own local `_match_player`/`_build_fpl_name_map` with the *exact*
unguarded-substring collision the Gabriel fix addressed (no ≥2-token guard at all). Unlike
`understat_xg.py` (manual, soccerdata-only, never auto-run), **this module is wired into
`scripts/run_agent.py` — the bot's real live decision entrypoint** — so the collision risk is
live-production-facing, not a backtest artifact: this would silently re-corrupt captaincy data on
every single live agent cycle once the 26-27 season starts. Fixed by dropping the local matcher
entirely and reusing the shared, hardened one from `fbref.py` (`understat._match_player is
fbref._match_player`, `understat._build_fpl_name_map is fbref._build_name_map`).

**Same bug class, third instance, in `press_conference.py`'s news-sentiment matcher** (also wired
into `run_agent.py`). `_extract_player_signals` matched a bare short name (web_name/second_name) via
plain substring containment with zero guard, so a news sentence mentioning any "Gabriel" or "James"
would attribute sentiment to whichever player's name happened to iterate first in the dict — same
failure mode, different consumer (rotation/availability signal, not xG). Fixed: `_build_player_name_map`
now drops any name shared by more than one real player entirely (rather than resolving to an
arbitrary one), and `_extract_player_signals` uses word-boundary regex matching and checks the
longest candidate name first, so a full "first second" match wins over a shorter substring of the
same sentence. New tests: `tests/test_press_conference.py`, `tests/test_understat_legacy_matcher.py`.

**A fourth suspected bug, investigated hard, turned out NOT to be a bug — recorded here so it isn't
re-investigated from scratch.** `Team.id`/`Player.team_id` mirror FPL's own numeric team id, which
is NOT a stable cross-season identity (FPL reassigns ids 1-20 fresh every season based on that
season's actual 20 clubs; only `team.code` is stable). Right now `teams.id=7/11/12` are live-correctly
named Coventry/Hull/Ipswich (promoted for 26-27), while a handful of departed players (Cucurella,
Konaté, M.Salah — all confirmed absent from today's live bootstrap `elements`, i.e. genuinely no
longer in the league) still carry those same numeric ids frozen from when they meant Chelsea/
Liverpool. This LOOKED like a severe, systemic corruption (346/841 players' team_id disagreed with
the frozen 2025-26 historical snapshot) but a precise test — comparing only players still present in
TODAY's live feed against what live says RIGHT NOW — found just **4 genuinely stale rows** (Penders,
Anselmino, Garnacho, Targett), fixed by simply re-running the existing `upsert_teams`/`upsert_players`
pipeline (no code change needed; `team_id` already gets refreshed correctly on every live re-run for
anyone still in the league). The other 342 "mismatches" were normal reality: real summer transfers
plus the expected season-to-season id renumbering for departed players, not corruption. Also
independently confirmed the walk-forward backtest is fully insulated from any of this either way —
`team_season_strength`/`player_gw_stats` come from a completely separate, correctly season-scoped
historical pipeline (`scripts/backfill_history.py`, sourced from the vaastav GitHub archive, not the
live API) and never touch the raw `teams`/live `Player.team_id` at all. No architecture change made;
building one would have solved a problem that didn't actually exist.

**Automated data-quality gates, so this class of bug gets caught going forward instead of
surfacing later as a captaincy anomaly.** New `data/quality_checks.py`: four small, pure, unit-tested
checks — `check_name_match_coverage` (flags an external-source name-matching pass below a coverage
floor — would have caught the season-wide xG gap immediately), `check_stat_column_not_dead` (flags a
stat column that's almost always exactly zero — would have caught FBref's dead `Expected xG` mapping
directly instead of it surfacing as a captaincy monopoly two bugs later), `check_team_id_matches_live`
(the staleness check above, generalized), and `check_no_single_teammate_monopoly` (flags one player
holding ≥95% of a team's goal/assist weight while teammates have nonzero weight too — would have
caught the N.Gonzalez/Gabriel monopoly pattern generically, independent of root cause). New
`scripts/data_quality_gate.py` wires the live-checkable ones (team-id freshness, Understat coverage)
into a runnable gate (`DB_PATH=fpl_bot_v2.db uv run --extra events python scripts/data_quality_gate.py`,
exit 1 on any error-severity issue) — not yet wired into `run_agent.py`'s automatic cycle, a natural
next step.

**One residual, deliberately unverified item:** `data/ingestors/odds_api.py`'s `_match_fixture` now
requires BOTH team names to match (see below) rather than home-only, which removes most of the
practical risk, but it still can't be exercised end-to-end because `fixture_odds` has 0 rows and
verifying the live match would mean spending a real call against a paid, quota-limited API key
(`THE_ODDS_API_KEY`) just to confirm team-name-format compatibility (FPL's short names like "Man
City" vs. whatever The Odds API actually returns). Flagged for whoever runs live odds ingestion
first to sanity-check.

Re-ran the Understat xG ingest with the fixed matcher: 11,306 rows written (up from 10,852), 34 unmatched
*rows* (down from the prior run; only 2 distinct players now, Jota Silva and Mathis Cherki, both
deliberately left unmatched above). Refreshed the 4 stale `team_id` rows via a plain ingestion
re-run.

### Data-completeness audit, part 3 (2026-07-28) — parallel ingestor-audit agent, 6 more confirmed bugs

Dispatched a second, independent pass (an `Explore`-agent audit) across the remaining 8 ingestor
files while the walk-forward gate re-ran in the background, specifically hunting the same 6 bug
classes (dead source columns, name-matcher collisions, dedup/duplicate bugs, unvalidated API
responses, id/join mismatches, silent zero defaults). It came back with 6 confirmed defects,
evidence-checked against the live DB/schema, ranked by severity — 2 were already fixed earlier in
this session (independently re-confirmed, see below); 4 were new and are now fixed:

1. **`odds_api.py::_extract_h2h` silently swapped home/away win probability for ~50% of fixtures.**
   It sorted the two non-Draw outcome names ALPHABETICALLY and assigned the first to home, the
   second to away — with no reference to which team was actually home. Traced downstream to a real
   ML feature (`projection/features.py`'s `my_cs_prob`/`opp_cs_prob`, built from
   `home_cs_prob`/`away_cs_prob` via `CASE WHEN was_home`). Fixed: now keyed directly by the real
   home/away team names from the same odds-API event payload, no sorting at all.
2. **`odds_api.py::_match_fixture` ignored the away team and kickoff time entirely** — matched
   purely on home-team-name substring against ALL of that team's unfinished fixtures with no
   ordering, so a team with more than one unfinished home fixture in the response window got every
   odds snapshot attached to whichever fixture the unordered query happened to return first, while
   its other fixture(s) silently got zero coverage. Fixed: now requires both home AND away names to
   match, tie-breaking a genuine same-week double-header by nearest kickoff time.
3. **`fpl_api.py::ingest_player_history` — the P12 double-gameweek bug, unfixed in this ingestor.**
   `PlayerGameweekStats`'s unique key is `(player_id, gameweek, season)` with no fixture component,
   but a genuine DGW player's FPL history has two entries with the same `round`; the old
   `on_conflict_do_update` per-entry meant the second fixture's full stat line (goals/minutes/bps/
   points/...) silently overwrote the first's rather than summing — one match's entire contribution
   destroyed. Fixed via a new pure `_accumulate_gw_history` that sums genuinely-cumulative per-fixture
   fields across same-round entries while keeping the LATEST entry's `selected`/`value` (those are
   point-in-time snapshots, not per-fixture stats, so summing them would be wrong the other way).
4. **`odds_api.py`'s `btts_prob` hardcoded to a literal `0.0`, never actually sourced.** `MARKETS`
   never requests a BTTS market, so this was a dead-mapped fake "certain no BTTS" value, not a
   documented gap. Downstream (`projection/features.py`) only falls back to 0.5 on NULL/NaN, so a
   real live fixture would have fed a bogus "BTTS never happens" signal straight into projections
   the moment `fixture_odds` ever got populated. Fixed: `FixtureOdds.btts_prob` is now nullable
   (`data/models.py`), and `ingest_odds` writes `None` instead of `0.0` — the existing COALESCE/
   fillna(0.5) fallback now actually fires as designed. Table was empty (0 rows), so the schema
   change needed no data migration, just a drop+recreate.
5. **Independently re-confirmed already-fixed:** `understat.py`'s local matcher (verified live: DB
   web_name `"Onana"` is shared by two real, different players — Amadou Onana id 63 and André Onana
   id 536 — and `"Wilson"` by four) and `press_conference.py`'s matcher (same surnames). Both were
   already patched earlier in this session (part 2 above); re-verified live post-fix that
   `_match_player("Amadou Onana", ...)` and `_match_player("Andre Onana", ...)` now correctly resolve
   to 63/536 respectively, and that `press_conference._build_player_name_map()` correctly excludes
   `"onana"`/`"wilson"`/`"sarr"`/`"phillips"`/`"patterson"`/`"king"`/`"johnson"`/`"gray"` as ambiguous
   rather than silently picking one.
6. **Confirmed clean, no fix needed:** `fbref_prior.py` (tries multiple flattened-column candidates
   per field — the season-stats page genuinely has `Expected npxG`/`Expected xAG`, unlike the
   already-fixed per-match dead-column bug), `injury_parser.py` and `midweek.py` (pure internal-ID
   logic, no cross-source join), `ownership.py` (joins on `fpl_id`, same ID space both sides;
   documents its one real gap, `captaincy_pct_overall`, explicitly rather than silently defaulting).

New tests: `tests/test_odds_api.py` (+5, including one that itself caught a real
naive-vs-aware-datetime `TypeError` in the kickoff-tiebreak code before it shipped),
`tests/test_fpl_api_history.py` (+4).

Re-ran the full walk-forward gate (GW6-38) a second time on top of all part-2 and part-3 fixes:

| benchmark | part-2 result | part-3 result (this pass) | Δ |
|---|---|---|---|
| v2 bot (full decision engine) | 1634 (49.5 pts/GW) | **1642 (49.8 pts/GW)** | +8 |
| v1 bot (naive-XI) | 1508 (45.7 pts/GW) | 1617 (49.0 pts/GW) | +109 |
| frozen template | 1519 (46.0 pts/GW) | 1631 (49.4 pts/GW) | +112 |

**Important, slightly uncomfortable honest finding: v2's margin over the static baselines
collapsed** — from +126/+115 pts (v1/frozen) after part-2 to just **+25/+11 pts** now. None of the
part-3 code fixes (odds swap/matching, DGW-history overwrite, btts_prob) should have moved this
number at all — they're both live-only paths the 2025-26 backtest never touches (`fixture_odds` is
empty and unused historically; `ingest_player_history` writes season="2026-27" only, never the
2025-26 data `backfill_history.py`'s separate vaastav-CSV pipeline actually backtests against). The
real cause is almost certainly the 21 newly-resolved Understat players (mostly solid, unglamorous
defenders — Ben White, Matty Cash, Joe Gomez, Max Kilman, Dan Ballard among them): once their real
underlying signal existed, the ONE-TIME squad optimised at GW6 for v1/frozen apparently picked a
materially stronger fixed XI than before, while v2's own number barely moved. Captaincy stayed
qualitatively sane throughout (Haaland 23x, Thiago 13x, B.Fernandes 10x, Enzo 9x across the whole
run — no repeat of the 100+-week single-player monopoly from earlier bugs), so this isn't a new
name-collision-style bug; it reads as a genuine result: **completing the underlying data made the
static baselines catch up to v2 far more than it improved v2 itself.** Worth investigating further
in its own pass (is the GW6 squad-optimisation step somehow capturing most of the season's real
value in one shot, leaving little room for v2's ongoing transfer logic to add on top?) — flagged as
an open question, not resolved here.

Suite 306/306, lint-clean on all changed files. Commits: TBD.

### P3-7 — Optimiser's-curse shrinkage, generalised (2026-07-28) — root cause of the below-average-manager result

User pushed back hard on the 49.8 pts/GW result: "not possible to use this work if we are worse than
the average manager" (avg-manager reference ≈56 pts/GW). Root-caused rather than smoothed over.

**Diagnosis.** v2's own per-GW `predicted` vs `actual` numbers from the walk-forward gate showed a
**+12.6 pt/GW bias** (mean predicted 62.4 vs mean actual 49.8) with only **0.33 correlation** — v1's
simpler static approach was roughly half as biased (+5.6) and much better correlated (0.52). Rather
than assume the projection MODEL itself was broken, checked calibration across the whole player pool
for 3 sample gameweeks by calling the same `assemble_gw_projections` the optimiser uses, directly:

| slice | GW10 bias | GW20 bias | GW30 bias |
|---|---|---|---|
| All ~800 players | +0.01 | −0.03 | −0.12 |
| Players who actually played (minutes>0) | −0.43 | −0.60 | −0.81 |
| **Top-50 by projected xpts** (the pool an optimiser draws from) | **+0.43** | **+1.30** | **+1.20** |

The model is essentially unbiased in aggregate — even slightly conservative for players who actually
play. The bias appears, and grows over the season, specifically among whichever players the model is
CURRENTLY most excited about: textbook **optimiser's curse**. Picking the argmax of many noisy
estimates systematically overselects players whose estimate is inflated by noise, not true ability.

Traced the mechanism precisely: `optimiser/scoring.py::risk_adjusted_score` already has a
`mu * xpts_var` term, but at the default `risk_mode="balanced"`, `mu=0` — **zero curse correction**.
P3-6 (earlier the same day) added `mu -= transfer_variance_penalty` (0.1) but ONLY inside
`evaluate_transfers`, explicitly scoped there ("never to squad-building or captaincy") to avoid
disturbing other reproducibility guarantees. That left `optimise_squad` and `optimise_starting_xi` —
which pick the INITIAL squad, and EVERY WEEK's lineup and captain, for both v1 and v2 — completely
uncorrected. This explains why v1 still showed a real (smaller) bias, and why v2's is roughly double:
both inherit the same uncorrected squad-build/captaincy selection bias; v2 additionally compounds it
with weekly transfers that only got a token 0.1-magnitude discount, ~10-15x too small relative to the
measured ~1.2 pt/player bias.

**Fix (design-and-implement, user's explicit choice over a quick recalibration or write-up-only).**
New `projection/assemble.py::apply_curse_shrinkage(projections, players)`: empirical-Bayes shrinkage
of `xpts` toward its **(gameweek, position) group mean**, weighted by each player's own `xpts_var`
relative to the real between-player variance in that group — high-uncertainty players (rotation risk,
new signings, edge-case fixtures) shrink further toward the mean; stable, low-variance nailed-on
starters barely move. Applied ONCE at the projection-assembly boundary (in both `scripts/backtest.py`
and the live `projection/pipeline.py`, after the unavailable-player zeroing so a genuinely unavailable
player's xpts=0 doesn't get pulled back up), so every downstream consumer — squad-building,
starting-XI, captaincy, transfers, AND live serving — sees the corrected value automatically instead
of needing its own copy of the same fix. Original value preserved as `xpts_raw`; `xpts_mean`/
`xpts_var` (the simulator's own honest per-scenario summary) are left untouched since the shrink
factor is computed FROM `xpts_var`. Gated by a new `OPTIMISER.curse_shrinkage_enabled` flag
(default `True`; `False` is byte-identical to pre-fix behaviour, same convention as P3-6). This
**supersedes and removes** `TransferRules.transfer_variance_penalty` entirely — a single, generalised,
self-calibrating correction replacing the narrower, arbitrarily-sized one.

New tests: `tests/test_curse_shrinkage.py` (+9: variance-proportional shrinkage, `xpts_raw`
preservation, `xpts_mean`/`xpts_var` left untouched, minimum-group-size and zero-between-variance
no-ops, position/gameweek independence, graceful no-op when `players` lacks `position`). Two existing
tests (`tests/test_p0_projection_scaffold.py`) initially broke on a minimal test fixture missing
`position` — fixed by making the no-op path explicit rather than loosening the fixture, since a
caller with no position data genuinely has nothing to group by.

Suite 315/315, lint-clean. **First gate re-run showed the fix was badly miscalibrated**: predicted
flatlined at ~22-24 pts/GW regardless of squad (vs actual bouncing 20-86) — the James-Stein-style
per-player weighting by `xpts_var` was wrong. `xpts_var` is the MC simulator's OUTCOME variance (how
spiky a player's week-to-week returns are — a explosive-returns forward legitimately has high
`xpts_var` with a precisely-known mean), not the model's ESTIMATION uncertainty about that mean;
checked live and mean `xpts_var` (~3.3-4.5 across positions) is comparable to or LARGER than the real
between-player variance (~1.7-2.9), so the ratio shrank nearly every player toward the mean regardless
of confidence, destroying the ranking signal instead of correcting a bias. **Reverted to a simpler,
safer uniform shrinkage strength** (`CURSE_SHRINKAGE_STRENGTH = 0.15`, not weighted by `xpts_var` at
all) — the 10 tests in `tests/test_curse_shrinkage.py` were rewritten to match. Killed the broken gate
run mid-flight rather than let it finish and report a nonsense number.

**Second gate re-run, with the corrected uniform shrinkage:**

| benchmark | pre-P3-7 | post-P3-7 (uniform shrinkage) | Δ |
|---|---|---|---|
| v2 bot | 1642 (49.8 pts/GW) | 1617 (49.0 pts/GW) | −25 |
| v1 bot (naive-XI) | 1617 (49.0 pts/GW) | 1629 (49.4 pts/GW) | +12 |
| frozen template | 1631 (49.4 pts/GW) | 1631 (49.4 pts/GW) | 0 |

v2's own predicted-vs-actual bias fell from **+12.6 to +9.3 pts/GW** (a ~26% reduction) — real,
measurable progress on the diagnosed mechanism — but correlation between predicted and actual
*also* fell, from 0.33 to 0.20, which is not fully understood and is flagged rather than explained
away. **The headline, unresolved finding: all three approaches now cluster within 14 points of each
other (49.0-49.4 pts/GW) — v2 no longer has a distinguishable edge over a squad picked once at GW6
and never touched again, and none of the three comes close to the ~56 pts/GW avg-manager reference.**
Since even the frozen template (built via the SAME curse-corrected `optimise_squad` call, then never
revisited) sits at the same ~49.4, the remaining gap looks less like "the weekly optimiser makes bad
transfers" and more like something upstream of weekly decision-making — the INITIAL squad-build
quality, the projection model's absolute scale, or the avg-manager reference constant's own validity
(flagged as an approximate, undocumented-source anchor back in the Gate section) are all still-open
candidates, not yet distinguished from each other.

Suite 316/316 (10 in test_curse_shrinkage.py + no regressions elsewhere), lint-clean. Commits: TBD.
**This does not fully resolve the user's original concern** (still below the avg-manager reference)
— reported honestly as partial, real progress plus a clearly-scoped open question, not a fix.

### Squad-evolution trace (2026-07-28) — human-readable audit, found a captaincy anomaly

User asked for an easy, human-readable trace of the full season's squad/transfer/captaincy
evolution specifically so they (a real FPL manager) could eyeball it for weird choices no summary
statistic surfaces. Built `scripts/render_squad_trace.py` + a new optional `trace: list[dict] | None`
parameter on `run_backtest` (default `None`, zero behaviour change for existing callers/tests) that
captures the full 15-man squad, named transfers in/out, captain/vice, and predicted-vs-actual per
gameweek. Renders Markdown; also generated a styled, collapsible HTML version for this session's
delivery (dark "pitch-at-night" theme, `<details>`-based per-GW disclosure, anomaly flagging).

**Found a real, concrete anomaly by inspecting it directly.** Haaland stayed in the squad
continuously GW6→GW19 (only actually sold GW20) but was captained only through GW14. For 8 of the
following gameweeks the armband went to a defender or backup-role midfielder priced under £8m
instead — Nico O'Reilly (£5.2-5.3m DEF) three separate times, Gabriel Magalhães (£6.5-6.6m DEF)
twice, plus Thiaw and Mavropanos (both sub-£5.2m DEF) — while Haaland (£14-15m) started every one
of those weeks, uncaptained, sitting right there in the XI.

**Traced the actual cause, not just the symptom.** Checked the model's OWN raw numbers for these
exact gameweeks: the chosen captain's projected xPts (7.6-8.4) genuinely exceeded Haaland's own
projected xPts (5.0-7.1) in the model's output — this is not a selection-logic bug that ignores a
higher number, the underlying projection itself rates these defenders higher. Traced further into
`_build_rolling_features`: Gabriel's rolling `defcon_rate` (mean CBIT count/match) was 12.2, above
the `DefConRules.def_threshold` of 10 — meaning the model has him reliably crossing the 25/26
Defensive Contribution threshold most matches, stacking with Arsenal's strong clean-sheet
probability and Gabriel's genuine set-piece goal threat (a real, legitimate profile — his cleaned-up
season xG post the earlier name-collision fix was 4.65, mostly headers). This is a real, defensible
mechanism, not obviously a bug: a defender's DefCon+CS floor genuinely CAN look higher than a
misfiring striker's current-form point estimate. **Whether the model over-credits that floor
relative to a world-class striker's ceiling, or whether captaincy should weight ceiling more heavily
than the current mean-argmax approach does, is exactly the kind of judgement call flagged for the
user's own FPL-manager read** rather than resolved unilaterally here — delivered as an artifact with
the pattern pre-flagged, not buried in 33 gameweeks of tables.

Not yet fixed pending that read. If confirmed as a real miscalibration, the likely next step is
either recalibrating the DefCon-crossing probability model (currently unclear whether it treats
`defcon_rate` as a soft Poisson-style rate or a hard threshold-crossing indicator — not yet checked)
or making captaincy selection ceiling-aware rather than pure mean-argmax (the scenario-based
captaincy from P3-4 already has the machinery — real MC samples — to weight upside explicitly if
that's the right call).

Suite 316/316 unaffected (trace is additive/optional). Commits: TBD.

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

### P1-fix — Recent-absence override (2026-07-28) — the injury-reaction gap the user's own trace review found

User reviewed the squad-trace artifact personally (as a real FPL manager) and found the real bug the
statistical analysis alone hadn't surfaced: Gabriel Magalhães was captained GW12 while genuinely out
injured, Rice captained GW19 injured, Ekitiké captained GW20 injured — and, more importantly, kept
starting for MULTIPLE gameweeks into a confirmed absence rather than being dropped after the first
blank.

**Root-caused, not guessed.** Queried `player_state_snapshots` directly: **100% of 29,338 rows for
the 2025-26 season show `status='a'`** — zero exceptions, for the entire backtest. `merged_gw.csv`
(the vaastav-archive source `backfill_history.py` uses) structurally has no `status`/
`chance_of_playing`/`news` columns at all — confirmed by fetching its real header. This was already
flagged in `compute_snapshot_rows`'s own docstring ("status / chance_of_playing / news are NOT
recoverable from merged_gw and default... documented residual skew") but nobody had checked whether
that gap actually *mattered* until this review proved it does: `apply_availability_override` — the
existing, correctly-designed deterministic safety net that forces certain-DNP on `status in
('i','u','s')` — has literally never fired once in any backtest run this session, because the
signal it reads is constant.

A player's FIRST blank gameweek is genuinely unpredictable from any data source available here (no
historical injury-news archive exists to check in advance) — Gabriel's GW11→GW12 real minutes went
90→0 with zero prior signal, same for Rice/Ekitiké at their respective onset weeks. But the
FOLLOW-UP reaction is where it's concretely broken and fixable: checked Ekitiké's real minutes
(GW20=0, GW21=0) against what the model projected for GW22 — his xPts went **up** (4.2→6.2) with
zero new positive evidence, and he stayed in the starting XI for 4 straight gameweeks through the
whole blank.

**Fix:** `projection/minutes_model.py` — new `_trailing_dnp_streak` (counts consecutive
zero-minutes gameweeks ending at each row, leakage-free: computed on already-played history only,
then shifted by 1 before use) feeds a new `apply_recent_absence_override(p0, p1, p2, dnp_streak)`,
applied in `_bands_frame` alongside (before) the existing status override. 1 confirmed blank retains
50% of the ML-predicted playing mass (real uncertainty — could be rotation, a minor knock, or a
longer injury); 2+ retains only 15% (statistically unlikely to be a safe near-term pick). Untuned
starting values, same convention as other heuristic constants this session. This needs no new data
source — it's built entirely from real minutes already in `player_gw_stats`, exactly like the
existing rolling-average features, just applied as an explicit deterministic override instead of
relying on the GBM to implicitly learn it from a diluted 3-5 GW rolling mean (which was clearly too
slow: two full zero-minute gameweeks weren't enough to stop the projection from *rising*).

**Verified against the exact real cases the user flagged:** Ekitiké's GW22 decision (after 2
confirmed blanks) now shows P(DNP)=0.878, appearance-points contribution collapsed from a
near-certain-starter level to 0.173. Gabriel's GW13 decision (after 1 confirmed blank) shows
P(DNP)=0.788, appearance-points down to 0.385 from whatever produced the original 3.4 xPts. Both
directly address the reported cases.

New tests: `tests/test_p1_minutes.py` (+6: streak counting incl. per-player-group reset, both
retention tiers, the 2+-streak floor not scaling further with a longer streak). Suite 321/321,
lint-clean on new code (pre-existing E501 debt in `minutes_model.py` confirmed untouched via diff).

**Not yet addressed (raised in the same review, deferred by explicit user choice):** the second
finding — Bruno Fernandes sold GW10 despite 4 straight solid, nailed-on gameweeks, replaced by
Gakpo who proved less reliable (including his own real injury/rotation gap) — is a separate,
transfer-churn/premium-retention problem, not an injury-data problem. Confirmed with real numbers
(Fernandes 90 min/3-8-4-5 pts before the sale; Semenyo's sale looked more defensible, a real 2-game
dip; Haaland's sale more debatable, ordinary star-striker variance). Flagged for a follow-up pass —
likely a retention-inertia term in the transfer ILP objective, distinct from the curse-shrinkage fix
(P3-7) which addresses selection bias, not switching cost.

## Open sequencing question
P3-0/P3-1 (live-serving + sample persistence) and P3-2 (EO ingestion) are independent —
neither blocks the other. P3-3 needs both. Objective v1 (linear, additive terms) is
mechanically straightforward once its inputs exist; objective v2 (true scenario-based
stochastic programming) is a bigger structural lift and explicitly framed as an
"upgrade" in the plan, not a v1 requirement.
