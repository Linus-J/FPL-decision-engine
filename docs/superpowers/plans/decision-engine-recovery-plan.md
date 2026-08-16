# Decision-engine recovery plan (26/27 season)

Companion to `pre-gw1-transfer-planning-review.md` (the findings). This is the
work plan.

**Framing decisions taken 2026-08-16:**

- This is a **decision engine**. Live submission to FPL is out of scope; the
  squad goes in by hand. Everything about `agent/fpl_client.py` is deferred.
- **The backtest is not the validation vehicle.** Weekly points are dominated by
  noise, and the historical harness is a second implementation of the decision
  loop that keeps diverging from the live one. Validation is the live season
  walk-through: the real bot plus 100 shadow personas, scored weekly, analysed
  at season end.
- One exception, held loosely: the re-run measured the projection
  over-predicting team totals by +7.98 pts/GW with 24/33 weeks in the same
  direction. Mean points are noisy; a mean signed error is a different kind of
  statistic. But the harness that produced it is itself a second implementation
  of the decision loop with known divergences, so the number is a **hypothesis,
  not a measurement** — it could be a harness artifact as easily as a model
  problem. It matters enough to chase because every threshold in the engine is
  denominated in absolute points, and it is cheap to settle properly: see P4.

**Consequence of choosing live-walk-through validation:** the measurement layer
becomes load-bearing. It is currently not trustworthy (P2). That moves ahead of
almost everything else, because without it the season produces no usable
evidence and the year is wasted.

---

## P0 — Before the GW1 deadline (2026-08-21 17:30)

Exactly one decision cannot wait, because it is the only one that changes the
GW1 squad.

**P0.1 — Cold-start availability weighting.** `projection/cold_start.py:489`
sets `xpts = ppg_played` (points **per appearance**); `projection/assemble.py`
produces an unconditional expectation. Different scales, and
`start_probability` is only a 0.4 filter.

Measured on the live build: weighting by availability changes 9 of 15 players,
lifts XI mean P(start) 0.828 → 0.955, bench 0.55 → 0.79 — and drops Haaland.

Options:
- (a) Ship `xpts × (appearances/38)` for the `prior_season` tier and the
  equivalent for the peer-bucket/prior-league tiers. Correct on the merits;
  materially changes the squad five days out.
- (b) Leave the scale as-is and add only a bench-availability floor (P0.2).
  Conservative; keeps a known scale inconsistency into GW1.

Recommendation: (a), because the in-season pipeline is unconditional and the
cold start should match it — but this is a judgement call about the GW1 squad
and needs an explicit decision, not a silent change.

**P0.2 — Bench availability floor. NOT NEEDED — dropped after measuring P0.1.**
The plan was a constraint forcing outfield bench players above ~0.5 P(start).
With P0.1 shipped, the live GW1 bench comes out at mean P(start) 0.793,
minimum 0.60, and the two 0.60s are peer-bucket unknowns sitting exactly at
`NEW_PLAYER_START_PROB` — a default, not a measured rotation risk, which a
floor could not improve on anyway. The floor would bind on nothing, and it
carried a real infeasibility risk (it must exempt the backup GK, who is
sub-0.5 by definition). Pricing availability in the objective turned out to
be strictly better than constraining it. Revisit only if a future build
actually produces a weak bench.

---

## P1 — Decision-engine correctness — **COMPLETE 2026-08-16**

Landed in `6aea96a` (P1.1–P1.4, P1.7), `4fe3050` (P1.5, P1.8, P1.9, P3.10)
and `faabf31` (P1.6). Suite 564 → 603 passing;
`tests/test_transfer_banking.py` is the regression gate for the whole
surface, which previously had none.

Two findings worth carrying forward, both discovered by writing the tests:

- **Hits are rarer than expected, correctly.** Deferring a second move to
  next week's banked transfer costs only ONE gameweek of that player's
  advantage, so a hit is rational only when the per-gameweek gain exceeds
  the 4-point cost. Verified: the threshold sits exactly between 3.5 and 4.5
  per gameweek for a persistent gain, and exactly at 4.0 for a one-week
  spike. The bot will now bank in preference to hitting far more often than
  the old backtest's 0.5–0.9 hits/GW suggested — which is the correct
  behaviour, and it compounds with P4.1a's finding that the transfer step is
  where the selection bias enters.
- **`_chip_uses_remaining` counted decision-log rows**, so re-running a
  gameweek consumed chips. This, not the `dry_run` flag, was the real
  corruption vector behind the original B8 finding — no history-splitting
  filter was needed.

Ordered by dependency. Nothing about transfer planning works until P1.1 lands.

**P1.1 — Restore the multi-gameweek projection frame.** `run_projections` builds
and persists `transfer_planning_horizon_gws` GWs; `_run_decision_cycle` discards
that and calls `get_latest_projections()`, which is `WHERE pp.gameweek = :gw`
for a single `next_gw`. So `evaluate_transfers` runs at `H = 1` and the entire
multi-period structure is unreachable. Fix: give `get_latest_projections` a
horizon (or pass `run_projections`' returned frame straight through).
*Everything below depends on this.*

**P1.2 — Free-transfer accounting.** `agent/decision_engine.py:400` →
`min(5, max(1, ft - made + 1))`. Also clamp `free_transfers <= 0` to 1 inside
`evaluate_transfers` so a bad input degrades instead of producing an infeasible
model that is silently swallowed as "no transfers".

**P1.3 — Wildcard feasibility.** `optimiser/transfers.py:155` — raise `ft`'s
`upBound` to 15, or drop the FT/hit constraints entirely for `w == 0` when
`wildcard_active`.

**P1.4 — Make hits legal.** `ft[w+1] <= ft[w] - n_trans[w] + hit[w] + 1`, and
move the `- hit[w] * 4` term out of the per-player loop (it is currently
multiplied by the candidate-pool size).

**P1.5 — BGW/Free Hit logic.** Two separate errors: `bgw_affected` counts
players with no projection row as "blanked" (with P1.1 this stops being
trivially 15/15, but it should be computed from the fixture list, not from
missing rows), and the Free Hit gate uses a lookahead-window count to justify
playing the chip *this* week. Gate on the blank week itself.

**P1.6 — Real budget.** `available_budget` is seeded at 100.0 by the cold start
and written back unchanged forever. Even with no submission, the ILP's
affordability constraint must be right or every transfer it proposes is wrong.
Track squad value and bank explicitly: value at purchase, selling price
(`now_cost` minus half the unbanked rise), and carry the bank across weeks.
`player_state_snapshots` (96,840 rows) already has the per-GW `now_cost`
history needed to compute this.

**P1.7 — Bench value in the transfer objective.** Add
`bench_value_weight * score * (squad - starting)` to `evaluate_transfers`'
objective, matching `optimise_squad`. Without it every in-season transfer treats
bench quality as worthless and erodes it back to fodder.

**P1.8 — Dry-run isolation.** Filter `dry_run` out of `_load_squad_state` and
`_load_own_decision_log`. Today a rehearsal overwrites "my current squad", and a
dry run recommending a chip marks that chip permanently used.

**P1.9 — Pass `squad_age_gws`** to `recommend_chip` so `wildcard_min_managed_gws`
binds live.

---

## P2 — Make the measurement trustworthy — **COMPLETE 2026-08-16**

Landed in `62d2f35` (P2.1–P2.3), `a51ae9b` (P2.4, plus P3.5) and `e0624da`
(P2.5, P2.6). Suite 603 → 622 passing.

The live cohort was regenerated: 90 personas (backed up first; no outcomes
existed, GW1 being unplayed, so nothing measured was lost), all 90 ran GW1
successfully, and the baseline control's projection matches the real bot's
exactly — which is the check that the simulation path and the real path have
not drifted.

Still open before this is genuinely load-bearing:

- **The dashboard doesn't surface any of it yet.** `simulation/analysis.py`
  is importable and tested but `dashboard/pages/6_Simulations.py` still reads
  the raw tables. Low effort, worth doing before GW1 finishes so the first
  real read-out is visible.
- **`swept_axis` is a string on `SimManager`.** The analysis groups on it. If
  a future sweep renames an axis mid-season the grouping silently splits.
- One axis, `risk_level`, still partly drives an inert path (`lambda` needs
  ownership, see P3.2). Its `mu` half is real, so it is not wasted, but its
  result will under-state what a fully-wired version would show.

### Original scope (all landed)

The plan is to learn from a live season. That only works if what gets recorded
is correct. Current state:

**P2.1 — Persona outcomes ignore autosubs.** `scripts/backfill_decision_outcomes.py`
calls `_score_squad` without `minutes`/`positions`/`bench_order`, so no autosub
and no vice-captain fallback is applied. `_score_squad` supports all three; the
blocker is that `_record_decision`'s lineup `details` never persists
`bench_order` or positions. Fix: persist them, then pass them through.

This is not cosmetic. Un-modelled autosubs systematically understate every
persona, and they understate *most* exactly where the bench matters — so the
season's data would be blind to the bench-robustness question that motivated
this review.

**P2.2 — Hit costs are not deducted.** `actual_outcome` on the lineup row is raw
points; hits live on a separate `transfers` row and are never subtracted.
Currently moot (hits are impossible), live the moment P1.4 lands.

**P2.3 — Outcome backfill is never run.** `run_weekly.py` runs FBref →
WhoScored → ownership → agent → simulations. It never calls
`backfill_decision_outcomes.py`, so `actual_outcome` is never populated at all.
Add it as the first step of the weekly run (it scores the gameweek that just
finished, before new decisions are made).

**P2.4 — The persona sweep varies the wrong knobs.** `run_for_persona` overrides
only `risk_level`, `max_ownership_differential` and `chip_aggressiveness`. Of
those, `max_ownership_differential` is **completely inert** (see P3.2), and
`risk_level` drives `lam`/`mu`, which at the current baselines are near-zero
anyway. Meanwhile the knobs the review flagged as untuned and load-bearing —
`transfer_switching_cost`, `ft_terminal_value`, `bench_value_weight`,
`transfer_planning_horizon_gws`, `mu_baseline` — are held fixed at the real
bot's values across all 100 personas.

Redesign the persona space around the decisions actually in question. A season
of 100 personas is one experiment; it should be spent on parameters that matter.

**P2.5 — Season-level tracking and analysis.** Add cumulative per-persona
running totals, rank-within-cohort, chips played and when, transfers and hits
taken, and a per-persona season summary. The dashboard has a Simulations page;
extend it rather than starting over. Define the post-season analysis *now*, so
the data needed for it is being recorded from GW1 rather than reconstructed
later.

**P2.6 — Log the counterfactual.** Record, per gameweek, what the bot decided
*and* what it declined — the transfer plan's runner-up, the chip that nearly
cleared, the projected-vs-actual gap. Otherwise "what went wrong" at season end
is unanswerable beyond the score.

---

## P3 — Part C: inert machinery, and what to do about each

Each item: what it is, why it ended up dead, and the call.

**P3.1 — The risk layer is switched off.**
`mu = mu_baseline + risk_level·mu_range = 0.0 + 0×0.08 = 0`, and
`lam = risk_level × magnitude = 0`. So `risk_adjusted_score ≡ xpts` exactly:
`xpts_var` influences nothing, and `scenario_based_captain` short-circuits at
`mu == 0` without touching the DB. P3-3 and P3-4 are dormant.
*Why:* `mu_baseline` was calibrated 0.05 → 0.0 on 2026-07-31 against a reduced
GW6-20 window; the calibration was honest and 0.0 genuinely won.
*Call:* keep it, but **describe the bot accurately** as an expected-points
maximiser, and stop paying costs for the dormant path (P3-1 sample persistence
writes tens of thousands of rows per run to feed a captaincy path that
short-circuits). Make it explicit rather than incidental: a
`risk_enabled: bool` that gates the sample writes. Then use P2.4's persona sweep
to test whether a non-zero `mu_baseline` is actually worse, on live data, rather
than on one reduced backtest window.

**P3.2 — Ownership/EO is ingested and never consumed.**
`ownership_snapshots` is **empty (0 rows)** — the ingestor has never
successfully produced data, because the 26/27 Overall league has no ranked
entries until GW1 locks. Independently, **no call site anywhere passes
`ownership=`** to `optimise_squad` / `evaluate_transfers` /
`optimise_starting_xi`. So even once data exists it will not reach a decision.
*Why:* the parameter was threaded through the optimisers (P3-3) but the
decision engine was never updated to supply it; the empty table hid the gap.
*Call:* two steps, in order. (1) After the GW1 deadline, run
`scripts/ingest_ownership.py 1` and verify it writes real rows — this is its
first-ever live execution and is flagged UNVERIFIED in its own docstring. (2)
Only once data exists, wire `ownership=` into the decision engine. Note it does
nothing while `lam = 0` (P3.1), so this is paired with that decision, not
independent of it.

**P3.3 — `use_price_change_signals` is read nowhere.**
There is no price-change modelling of any kind. The config flag claims a feature
that does not exist.
*Why:* aspirational config written ahead of implementation, same pattern as the
`DGWStrategy` fields already deleted on 2026-08-01.
*Call:* **delete the flag now.** Price prediction is real alpha (team value
compounds over a season), but it is a project, not a fix, and it should not sit
in config pretending to be wired. If built later, `player_state_snapshots` has
the `now_cost` history to train on. Related: P1.6 needs selling-price mechanics
regardless, which is the useful half.

**P3.4 — `evaluate_transfers(dgw_gws=...)` is accepted and never used.**
DGW/BGW-aware transfer preference does not exist. The `DGWStrategy` knobs that
would have driven it were deleted as dead on 2026-08-01, which removed the
config but not the unused parameter.
*Why:* the parameter was added in anticipation; the feature never followed.
*Call:* this is the direct answer to "bank transfers ahead of an uncertain
period". Implement it properly after P1.1: scale `ft_terminal_value` per
gameweek so a DGW/BGW just beyond the horizon raises the value of arriving with
transfers in hand. Until then, **delete the parameter** so the signature stops
implying a capability.

**P3.5 — Horizon constants disagree with each other.**
`CHIP_TIMING.wildcard_eval_horizon_gws = 5` and
`OPTIMISER.cold_start_lookahead_gws = 5`, but
`transfer_planning_horizon_gws = 3` governs how many GWs are projected — and
live only 1 survives (P1.1). `DGW.lookahead_gws = 6` is unused; DGW detection
uses the transfer horizon instead. So the wildcard's 25-point threshold is
compared against a 1-to-3 GW gain.
*Why:* the constants were introduced at different times against different
assumptions about what the projection frame contained.
*Call:* make the projection horizon the maximum of everything downstream that
consumes it, and assert that relationship at startup so it cannot drift again.

**P3.6 — `should_take_hit` is dead code.** Never called. Delete, or call it.

**P3.7 — Set-piece and penalty roles: schema and query exist, data does not.**
`player_setpiece_roles` has **0 rows**. `projection/features.py:132-169` already
LEFT JOINs it and `COALESCE(..., 0)`s every field, so `is_penalty_taker`,
`penalty_xg_per_game` and `is_set_piece_taker` are silently zero for every
player, always. No ingestor writes the table — `PlayerSetpieceRole` appears
only in `data/models.py`.
*Why:* the plumbing was built first and the ingestor never followed; the
COALESCE made its absence invisible.
*Call:* **this is the largest missing source of real alpha in the system.**
Penalties are worth several points a season per taker and are highly
predictable. Build the ingestor. Even a hand-maintained YAML of the 20 clubs'
penalty takers — the same override pattern `transfer_overrides.yaml` already
uses — would beat zero, and could ship before GW1.

**P3.8 — Press-conference signals are ingested and discarded.**
`player_press_signals` has 13 rows, written by
`data/ingestors/press_conference.py`, and read by **nothing** but `models.py`.
*Why:* deliberately built as a shadow-mode layer pending A/B validation that
never happened.
*Call:* decide. Either wire it into the minutes model as a start-probability
adjustment and measure it via the persona sweep, or stop running it weekly. A
shadow layer that is never compared to anything is pure cost.

**P3.9 — `scripts/data_quality_gate.py` is never invoked.**
No pre-decision data validation runs in the live pipeline.
*Why:* written as a manual audit tool during the 2026-07-28 audit, never wired.
*Call:* run it at the top of `run_weekly.py` and make it warn loudly (not
block — a blocked week is worse than a week decided on slightly stale data).

**P3.10 — Season-unscoped gameweek queries.**
`_get_current_and_next_gw` and `_get_current_season` query `gameweeks` with no
season filter across a 6-season DB. Benign today (only 26/27 rows carry
`is_current`/`is_next`), but this is the exact shape of the bug
`_get_wc_half_boundary` already had to fix on 2026-07-29.
*Call:* add the filter. One line, removes a whole failure class.

**P3.11 — Odds coverage is thin.**
`fixture_odds` holds **6 of 10 GW1 fixtures and nothing beyond GW1**. In-season
projections fall back to `lam_home=1.35, lam_away=1.15` for everything else, so
the GW2/GW3 half of the planning horizon is odds-blind — which directly weakens
the multi-period planning P1.1 restores.
*Why:* bookmakers price near-term fixtures; the ingest window is narrow.
*Call:* measure how far ahead odds are actually available once the season
starts, widen the ingest window to match, and record coverage per run so the
horizon's real information content is visible rather than assumed.

---

## P4 — Calibration

**P4.1 — Measure calibration on the live path, weekly, and stop relying on the
harness to tell us.** The backtest reported predicted 62.95 vs actual 54.97 over
33 GWs (over-predicting in 24), where the 2026-05-30 code *under*-predicted by
4.90. That is worth chasing, but the harness is a known-divergent second
implementation, so treat the number as a lead rather than a result.

The better instrument is already implied by P2: **every live gameweek produces a
predicted-vs-actual pair for the real bot and for all 100 personas, on the code
path that actually runs.** Once P2.1–P2.3 land, calibration is measured for free
every week, with 101 observations per gameweek instead of one, and with no
harness in the loop. Record the signed error per gameweek from GW1 and plot the
running mean; four or five gameweeks will settle whether an ~8-point bias is
real far more convincingly than re-running the backtest would.

**P4.1a — Probe result (2026-08-16): the bias is decision-induced, not a
projection problem.** Ran naive-XI (`--naive-xi`, fixed squad, no transfers, no
chips) against the same projections and the same scoring code:

| run | mean bias | corr | MAE |
|---|---|---|---|
| naive XI (no decisions) | **+0.96** | 0.64 | 10.37 |
| full bot (transfers + chips) | **+7.98** | 0.42 | 13.60 |

The projection layer is essentially **unbiased** over 33 gameweeks. The entire
+8 appears only once the decision loop is switched on. Because both runs share
the projection and scoring code and differ only in the decision layer, the
*difference* is a clean within-harness comparison — far more robust than either
absolute number, and it survives the objection that the harness itself is
unrepresentative.

This is the optimiser's curse, undisguised. Each week the transfer ILP moves
into whichever available player projects highest, which is precisely the
operation that maximises selection bias: it systematically picks the players
whose projections are highest *because they are overestimated*. The correlation
drop (0.64 → 0.42) is the same story — churn is adding noise, not signal.

Implications, which redirect several other items:

- **No projection-layer calibration work is needed.** The earlier suspect list
  (curse shrinkage, DefCon, injury discount, BPS weights) is cleared.
- `apply_curse_shrinkage` was built for exactly this and is evidently
  under-powered against the transfer step. It shrinks toward a
  (gameweek, position) group mean and was calibrated on the observed top-50
  bias; the transfer ILP selects harder than that.
- `transfer_switching_cost = 1.5` was introduced for the same reason (the Bruno
  Fernandes case) and is likewise under-powered.
- This substantially raises the value of P1.7 (bench value in the transfer
  objective) and of the P4.2 re-tune, and it is a strong argument for keeping
  the bar for a transfer *high* — which is the same conclusion the banking work
  (P1.1/P1.2) points at from the other direction.

Do not re-tune anything on the strength of the backtest's absolute numbers
alone — but the naive-XI-vs-full-bot *gap* is solid enough to act on.

**P4.2 — Re-tune thresholds only after P1 and P4.1.** `hit_cost_points` is fixed
by the rules, but `transfer_switching_cost`, `ft_terminal_value`,
`bench_value_weight`, `wildcard_pts_gain_threshold`,
`bench_boost_min_bench_xpts`, `free_hit_single_gw_gain_threshold` and
`triple_captain_min_gain` are all absolute-point thresholds being compared
against inflated gains. Every current value was fitted against a model that
could not bank, could not take hits, could not see past one gameweek, and
over-predicted by 8. Re-tune via the persona sweep (P2.4), not the backtest.

**P4.3 — Behavioural targets, not just point targets.** The fresh run took a
transfer in 30 of 33 gameweeks, never banked past 1 after GW12, took zero hits
all season, and never played Bench Boost. Those are checkable properties
independent of noise. Set explicit expectations (e.g. "banks 2+ transfers at
least N times a season", "plays 6+ of 8 chips") and track them weekly.

---

## Sequencing summary

| when | items |
|---|---|
| **before 2026-08-21** | P0.1 (decision required), P0.2, optionally P3.7 penalty-taker YAML |
| **before 2026-08-28 (GW2)** | P1.1 → P1.9, P2.1 → P2.3 |
| **first weeks of the season** | P2.4 → P2.6, P3.2 step 1, P3.9, P3.10 |
| **ongoing** | P4.1, P4.2, P3.7 ingestor, P3.11 |
| **cleanup, anytime** | P3.3, P3.4, P3.6 deletions; P3.5 assertion |
| **explicit decisions needed** | P0.1, P3.1 (risk layer), P3.8 (press signals) |

The single highest-leverage item is **P2** — not because it scores points, but
because the chosen validation strategy is a live season, a season only happens
once, and right now the season would produce data that is wrong in exactly the
place the review cares about.
