# Decision-engine review — 2026-08-18

A review of methods, modelling choices, logic and data quality ahead of the
GW1 deadline (2026-08-21 17:30). Distinct from `db-audit-2026-08-16.md`: that
audit profiled the stored data, this one interrogates the *modelling* and the
26/27 **rule set**, and re-derives the evidence behind claims the project
already makes about itself.

Every finding below was verified first-hand — code read, and where a number is
quoted, a query or a computation run against the live `fpl_bot_v2.db`. Where a
hypothesis did not survive checking it is recorded as refuted rather than
dropped, because several plausible-sounding concerns turned out to be sound
design.

**Nineteen defects, three of them in the same failure class the last audit
closed.** Sections 1–9 cover projection, ingest and measurement; sections 10–17
cover the optimiser, transfer and chip layer.

> **Status, 2026-08-18: all nineteen are addressed.** Fifteen are fixed in
> code; §7 (goalkeeper bonus re-calibration) is blocked on 26/27 BPS that does
> not exist yet and is flagged in place; §16 was resolved as a documentation
> correction, since the dormant risk layer is a defensible configuration and
> only its description was wrong. Fixing §3 exposed a further defect, live in
> the in-season path too: curse shrinkage was resurrecting players it had
> zeroed. The GW1 squad, XI, bench order, captain and cost are **unchanged**
> after all of it. See `docs/superpowers/plans/engine-fix-plan-2026-08-18.md`.

---

## 1. Two model features will read 0 for the whole season

`projection/features.py` wraps six single-fixture strength columns in
`_usable()` (line 37), which turns FPL's placeholders into NULL so the
median/1200 fallback can fire. The file's own comment says exactly why:

> *"add_fdr_features' fillna could not help: NaN is missing, 0 is present."*

The two **subquery-derived** lookahead columns never got that treatment:

```sql
COALESCE((SELECT AVG(CASE WHEN s2.was_home THEN t2.strength_defence_away
                          ELSE t2.strength_defence_home END) ...), 1200)
  AS next_3gw_avg_opp_defence      -- features.py:91-98
  AS next_3gw_avg_opp_attack       -- features.py:100-107
```

They average the raw strength columns. In the live database:

| season | `strength_defence_home` | `strength_attack_away` |
|---|---|---|
| 2025-26 | 1000 – 1310 | 1050 – 1390 |
| **2026-27** | **0 – 0** | **0 – 0** |

`AVG(0,0,0)` is `0`, not NULL, so the `COALESCE(…, 1200)` never fires. Both
columns will be **0 on every 2026-27 row**, against ~1200 in all five training
seasons. Both are live model features (`features.py:131-132, 382-383`) and
neither is in preflight's `degenerate` pin-list, so nothing reports it.

This is defect 14's exact shape — *a row full of zeros is present, not missing*
— and the fix reached the six direct columns while missing the two subquery
ones. It self-corrects when FPL publishes real strengths, but bites from GW2
until then, and a column that is ~1200 for five seasons and 0 only for 2026-27
is a perfect "this row is the current season" split for a boosted tree.

**Fix:** apply `_usable()` inside both subqueries, or wrap as
`NULLIF(AVG(...), 0)`.

## 2. There is no fixture signal beyond the current gameweek

`fixture_odds` holds **6 fixtures, all in GW1**. GW2–38: zero.

```
gameweek | fixtures | with_odds
       1 |       10 |         6
       2 |       10 |         0   … through GW38
```

`assemble.py:623` falls back to a flat `lam_home, lam_away = 1.35, 1.15` for
any fixture without odds. There is no team-strength fallback — unlike
`cold_start.py`, which resolves prior-season defence strength by team `code`
and applies `fixture_multiplier`.

Measured on the six fixtures that *do* have odds:

| quantity | real range | flat fallback |
|---|---|---|
| `lambda_home` | 1.30 – 2.61 | 1.35 |
| `lambda_away` | 0.55 – 1.88 | 1.15 |
| clean-sheet prob | **0.153 – 0.574** | 0.317 |

So across the 3-gameweek transfer horizon and the 5-gameweek wildcard horizon,
every fixture looks identical: a defender's clean sheet is worth the same
whoever they play. This nullifies the engine's stated core modelling decision,
*"odds set the total"*, for all but the current week.

Bookmakers only price one or two weeks ahead, so GW+1 fills in naturally —
GW+2…+5 never will. The wildcard's 25-point gain threshold is therefore
compared against a five-gameweek number of which four gameweeks are
fixture-blind.

**Fix:** fall back to team-strength-derived lambdas. The machinery already
exists in `cold_start.load_prior_defence_strength_by_code`.

## 3. The GW1 squad gets no optimiser's-curse correction

`apply_curse_shrinkage` is called from exactly two places:
`projection/pipeline.py:291` and `scripts/backtest.py:208`.

`decision_engine.py:351` branches on `season_has_played_history` and calls
`cold_start.build_initial_squad` directly, bypassing `pipeline.py` entirely.

So the highest-stakes decision of the season — the initial 15, locked in for
weeks — is made from unshrunk projections, even though the cold-start tiers
(prior-season, translated prior-league, peer bucket, synthetic) are the
noisiest estimates the system produces. The curse correction exists precisely
because selecting on noisy projections over-selects noise, and it is switched
off exactly where the noise is highest.

Secondary: the shrinkage target is the `(gameweek, position)` group mean over
**all** players, including the zeroed-out unavailable ones, so how hard a good
player is shrunk depends on how many irrelevant players happen to be in that
week's frame.

## 4. `player_match_events.source` is wrong on all 11,182 rows

Every row says `source='fbref'`. But `clearances` (4.72/90 for DEF) and
`recoveries` (3.63/90) are populated, and `fbref.py`'s own docstring states
those are **not available** from FBref's summary table. `whoscored.py:160`
UPDATEs the row without touching `source`.

Provenance is therefore unauditable — which is what let defect 5 below hide,
and what makes it hard to measure now.

## 5. Two sources write `tackles` under different definitions

- `fbref.py:63` maps `tackles` → `"Performance TklW"`, tackles **won**.
- `whoscored.py:74`: *"A dribble only counts if the TakeOn was won; **every
  other field is a plain per-type row count**"* — so `Tackle` events are
  counted regardless of outcome, i.e. **attempted**.
- WhoScored runs after FBref and overwrites.

Splitting on a provenance proxy (`recoveries>0 OR clearances>0`), minutes ≥ 60,
2025-26:

| position | fbref-only | whoscored-patched | inflation |
|---|---|---|---|
| DEF | 0.92 tkl/90 | **1.69** | +84% |
| MID | 1.07 tkl/90 | **1.88** | +76% |

Consistent with a ~55-60% tackle success rate. It matters twice:

- **BPS** — `successful_tackle` is +2, the highest-weight defensive action.
- **DefCon** — CBIT ≥ 10 is a hard threshold, so a scale error on one of its
  four components does not wash out; it shifts P(threshold met) systematically.

*Caveat:* the fbref-only sample is small (72 DEF / 112 MID rows) and may be
biased toward matches WhoScored did not cover. The mechanism is confirmed from
the code; the magnitude is corroborated, not proven.

## 6. The BPS validation figure is much weaker than it reads

The documented claim — bonus recompute matches FPL on 88.9% of rows, mean error
+0.012 — reproduces exactly (11,176 rows, 0.8893, +0.0115). But that number is
dominated by the ~90% of rows where both values are zero.

**Conditional on FPL actually awarding bonus (1,155 rows): 34.2% exact.**

Recomputed vs real bonus recipients, by position:

| position | real | recomputed | ratio |
|---|---|---|---|
| **GKP** | 80 | 32 | **0.40×** |
| DEF | 352 | 275 | 0.78× |
| MID | 515 | 646 | 1.25× |
| FWD | 208 | 267 | 1.28× |

MID/FWD being up is plausibly the 26/27 rule change working as intended — it
deliberately shifted bonus toward attacking players. **GKP at 0.40× goes the
wrong way**: 26/27 explicitly *buffed* goalkeepers.

The cause is a data gap. `saves_in_box` and `big_chances_saved` are zero on all
11,182 rows — and those are precisely the two 26/27 goalkeeper buffs.
`passes` and `pass_completion_pct` are mapped in `FBREF_SUMMARY_MAP` (lines
67-68) but are also zero on every row; pass-completion is worth up to +6, the
largest single positive BPS component available to an outfielder.

Note the reduced-BPS approximation in `bonus.py` is deliberate, documented, and
instrumented by `reduced_full_agreement` — that is not the problem. The problem
is that the *full* recompute used to validate it is itself missing its largest
inputs, so 88.9% was never evidence that the simulator ranks players correctly.

## 7. The goalkeeper bonus calibration targets the superseded rule set

`bonus.py:43`, `GK_BONUS_SAVE_SCALE = 0.45`, was calibrated on 2026-07-26 to
reproduce **25/26 outcome rates** — GK P(bonus>0) of 11.0%. That target checks
out against the database (GKP 10.95%, DEF 9.11%, matching the code comment).

But it is applied under **26/27 weights**, and 26/27 deliberately raised
goalkeepers' bonus prospects (flat 2 BPS per save, +1 inside the box, +1 for a
big chance saved) while cutting defenders' CBI rate from 1-per-2 to 1-per-3.
Tuning the new model to hit the old season's rate mechanically undoes the rule
change, and it stacks on top of the data gap in §6.

## 8. Outcomes can be scored before bonus and DefCon are final

`backfill_decision_outcomes.py:71` gates on `gameweeks.finished`. FPL's
`data_checked` flag — the one that actually means the data is final — is
neither stored (it is not in the `gameweeks` schema) nor referenced anywhere in
the codebase.

This is newly dangerous in 26/27. The gameweek lockdown moved from ~1 hour
after the final whistle to **09:00 the day after**, widening the provisional
window from about an hour to twelve or more. `run_weekly.py` scores last
gameweek *before* deciding this one, so a run inside that window writes
provisional bonus and DefCon into `decision_log` and `sim_decision_log` — the
calibration instrument and the persona ranking, which the project names as its
primary validation instrument.

**Fix:** ingest `data_checked` and gate on it.

## 9. The documented test gate does not run

`docs/decision-engine.md` names `pytest` and `ruff check` as the gates.

- `uv run pytest` → **55 collection errors**, `ModuleNotFoundError` on the
  project's own packages (no `conftest.py`, and `tool.uv.package = false`).
- `uv run python -m pytest` → **755 passed**.
- `uv run ruff check .` → **117 errors** (55 E402, 53 E501).

The suite is healthy; the documented way to invoke it is not. Worth fixing
because a CI or cron invocation that collects nothing exits non-zero in a way
that is easy to mistake for an environment problem — and a variant that
swallowed the exit code would look like success.

---

# The optimiser, transfer and chip layer

## 10. A wildcard throws away every banked free transfer

`optimiser/transfers.py:52-53`:

```python
if wildcard_played:
    return trules.free_transfers_per_gw      # 1
```

FPL's actual rule, confirmed against the Premier League's own guidance and
Fantasy Football Scout: **saved free transfers are retained through a Wildcard
or a Free Hit.** Two saved before a GW6 wildcard means three available in GW7 —
the two saved plus GW7's allotment. You simply do not get an extra one in the
week you play it.

The Free Hit branch immediately below gets this right (it zeroes
`transfers_made` and lets the normal carry run). The wildcard branch does not.
A wildcard played on a full bank drops the engine from 5 free transfers to 1 —
**four transfers destroyed, up to 16 points of avoidable hits, twice a
season.** `ft[0]` in the multi-period ILP is seeded from this value, so every
subsequent week plans against an allowance that is wrong.

**Fix:** treat the wildcard exactly like the free hit — one line.

## 11. The wildcard rebuild is taxed by the anti-churn switching cost

`transfers.py:360` subtracts `transfer_switching_cost * n_trans[w]` from the
objective for **every** week, including `w = 0` under an active wildcard.

`transfer_switching_cost` (1.5) exists for a specific reason, documented at
length in `strategy.py:186-198`: to stop a noise-sized edge triggering churn
*within the free allowance*. On a wildcard that rationale is void — unlimited
transfers are the entire point of the chip, and it is scarce (one per half).

Charging it anyway means a 10-player rebuild is docked 15 points against its
own objective, so the solver systematically under-uses the wildcard it just
decided to play. The hit term is correctly disabled under a wildcard
(`hit[0] == 0`, line 329); the switching cost was missed.

## 12. The squad that justifies the wildcard is not the squad that gets built

`chips.py::_try_wc` decides whether to play the chip by calling
`optimise_squad(..., free_transfers=15, horizon=5)` — a single-period build,
untaxed, unconstrained by the bank.

`decision_engine.py:538` then *executes* the wildcard through
`evaluate_transfers(wildcard_active=True)` — a different optimiser: multi-period,
switching-cost-taxed (§11), and constrained by real bank and purchase prices.

So the gain that cleared the 25-point bar is not the gain that will be
realised. Two optimisers with different objectives decide and act.

`docs/decision-engine.md` also describes this incorrectly — it says
*"wildcard/free hit? → optimise_squad"*, but only Free Hit takes that path.

## 13. Chip gains are computed over 15 players, ignoring the XI and the captain

`transfers._squad_xpts` gets this right — `nlargest(11)` plus a captain bonus.
`chips.py` does not. Both `_try_wc` and `_try_fh` use raw 15-man sums:

```python
projections[...isin(wc_squad_ids)]["xpts"].sum()      # chips.py:343-347
```

A Free Hit plays eleven players, not fifteen. So the Free Hit gain credits four
bench players who will not play, and neither chip credits captaincy at all —
two errors pulling in opposite directions, against thresholds (25.0, 12.0)
that were never calibrated against either definition. Bench Boost is the one
case where a 15-man sum is right.

## 14. The payoff-probability gate does not do what its config says

`config/strategy.py:283-288` states the intent precisely:

> *"minimum **P(gain >= 0)** over real persisted MC scenarios required, **IN
> ADDITION to** the point-estimate thresholds above, before a chip is
> recommended."*

`_clears_threshold` (chips.py:22-38) implements neither half:

```python
if scenario_values.empty:
    return point_value >= threshold
return float((scenario_values >= threshold).mean()) >= min_probability
```

Two divergences from the stated design:

1. It tests `P(value >= threshold)`, not `P(gain >= 0)`.
2. It applies that **instead of** the point-estimate test, not in addition —
   `point_value` is discarded entirely whenever samples exist.

The consequence is a far stricter bar than either the config or the constants
intend. "Mean gain ≥ 25" and "60% chance of a gain ≥ 25" are wildly different
tests; for a right-skewed FPL distribution the latter is much harder. A
wildcard requiring a 60% probability of a ≥25-point five-gameweek gain will
essentially never fire.

And it matters most because **samples exist in live serving and never in the
backtest** (P3-1 does not persist them there), so the gate that runs in
production is not the gate that runs in any tuning run. That is a live/backtest
divergence pointing in exactly the direction of the original complaint that
chips go unused all season — and it is a plausible primary cause of it.

## 15. Bench Boost and Free Hit cannot fire in the first half of the season

`_try_bb` returns `None` unless `dgw_active_now`. `_try_fh` returns `None`
unless `bgw_affected_count >= 5 or dgw_active_now`.

The 26/27 fixture list currently has **zero** doubles and blanks — all 38
gameweeks hold exactly 10 fixtures. Doubles and blanks only arise from
postponements, which cluster in the second half of the season.

`_panic_shrink` lowers *thresholds* as the half expires, but it does not relax
these hard structural gates, and the panic force-play at the boundary covers
**only Triple Captain**. So if no DGW or 5-blank BGW materialises before GW19 —
the normal case — two of the four first-half chips expire unused by
construction, and nothing reports it.

**Resolved 2026-08-18, and the original framing here was too soft.** Waiting
for a double gameweek is not merely *risky*, it is unsound, because the chip
being saved cannot be carried over: each half issues its own set of four and
destroys whatever is left at the boundary. A chip held back for a double that
never comes is not saved, it is thrown away — and a bench scores *something*
every week, so playing Bench Boost on an ordinary gameweek strictly beats not
playing it at all.

Both preconditions are removed. A double gameweek now expresses itself where it
belongs, in the *number*: a doubled bench scores roughly twice as much and
clears `bench_boost_min_bench_xpts` easily, while an ordinary bench clears it
only when genuinely strong. That is a preference, not a gate.

Backing it is a budget rule, `must_play_a_chip_now`. Only one chip may be
played per gameweek, so a half's remaining gameweeks are *slots* and its unused
chips are *items*. Once the items reach the slots, declining today makes it
arithmetically impossible to play them all, and the engine forces the best
available chip instead of letting the calendar bin one. Triple Captain, Bench
Boost and Free Hit all quantify a one-gameweek point gain, so under a forced
play they are directly comparable and the largest wins; the wildcard's gain is a
multi-gameweek figure and is the fallback rather than a rival.

This replaces a narrower rule that force-played only Triple Captain, and only
in a half's last two gameweeks — leaving the other three to evaporate in
silence.

## 16. The entire risk and variance layer is inert at the default config

`mu_baseline` was calibrated to **0.0** (`strategy.py:393`), and
`mu = mu_baseline + risk_level * mu_range` with `risk_level = 0` gives `mu = 0`.
Consequences, all confirmed by reading the code paths:

- `risk_adjusted_score` reduces to plain `xpts` — no variance term.
- `differential_multiplier` is 1 for everyone — no EO effect.
- `scenario_based_captain` short-circuits at `mu == 0.0`
  (`captaincy.py:176`) and returns a plain mean argmax **without touching the
  database**.

That last one matters: `captaincy.py` is a genuinely sophisticated piece of
work — it recovers fixture groups from `scenario_id` spans and computes true
joint team-total variance under each captaincy choice. It never runs.

Two pieces of documentation now assert the opposite. `captaincy.py`'s own
docstring says *"mu is no longer 0 by default … so this short-circuit is now
the exception rather than the common case"*, and `decision-engine.md`'s
limitations table says teammate covariance is unmodelled in the optimiser
*"(captaincy does model it)"*. Both were true when written and were falsified
by the `mu_baseline` calibration.

This is not necessarily a bug — 0.0 won its calibration sweep. But the engine
is carrying three layers of unused machinery whose documentation claims they
are live.

## 17. Triple Captain will very likely be spent in the first few gameweeks

`_try_tc` is **first** in the evaluation order on any non-DGW week
(chips.py:370). Its threshold is the captain's *absolute* projected points
against `triple_captain_min_gain = 4.0`. The `triple_captain_dgw_wait_multiplier`
that would raise that bar only applies when `dgw_visible_ahead` — and no DGW is
visible, because none exist in the fixture list yet (§15).

So from GW2 the gate reduces to *P(captain scores ≥ 4) ≥ 0.6*, which a premium
captain clears comfortably in a good fixture. There is no mechanism to hold the
chip for a better week other than a visible DGW.

The rebasing from "gap over the second-best captain" to "absolute points" was
correct — TC really is worth one extra copy of the captain's points. But the
scarcity side of the trade (two uses all season) is now guarded only by a DGW
lookahead that is empty for most of the first half. Mark this SUSPECTED on
magnitude: it depends on the realised sample distribution, which does not exist
until GW2. It is worth instrumenting before it fires.

## 18. Smaller items in this layer

- `max_hits_per_gw = 2` (`transfers.py:331`) is **not** an FPL rule — FPL
  allows unlimited hits. Rarely binding, but it is enforced as though it were a
  rule.
- `_bench_player_ids` approximates the bench as "outside the top 11 by xpts",
  ignoring formation legality, so `bench_xpts` can name a player who would
  actually start. Documented as an approximation and consistent with
  `_bench_xpts`, but it gates a real chip decision.

---

## 19. The curse correction cannot change who gets picked

Found while verifying §3, and it reframes what that fix achieves.

`apply_curse_shrinkage` computes `xpts' = (1 - s)·xpts + s·m_p`, where `m_p` is
the mean of the player's own `(gameweek, position)` group. That is an **affine
transform with a per-group offset**, which has two consequences:

- **Within a position, the ranking is exactly preserved.** Which defender or
  midfielder is best cannot change, at any shrinkage strength.
- **Squad quotas are fixed at 2/5/5/3**, so `Σ s·m_p` over the fifteen is the
  same constant for every feasible squad.

The only channels left are the starting XI's positional shape and the captain's
position, where composition varies. Measured on the live cold start, neither
bites:

| | shrinkage on | shrinkage off |
|---|---|---|
| squad | identical | identical |
| starting XI | identical | identical |
| cost | £100.0m | £100.0m |

Spearman correlation between shrunk and raw GW1 projections: **0.9996** across
568 rows (below 1.0 only because the comparison spans positions with different
offsets).

So the initial squad being unchanged after §3 is not reassurance that the fix
was safe — it is a structural property. **A uniform within-group shrink cannot
correct a selection bias**, because selection within a group is exactly what it
leaves alone. What it does correct is the *level*: the reported GW1 total drops
59.97 → 54.74, which is the honest number and feeds calibration.

It is not inert everywhere. In-season, gains are compared against absolute bars
— the 1.5 switching cost, the 4-point hit — so compressing a gap by 15% can flip
those comparisons and genuinely suppress churn. That is where the correction
earns its place.

But the optimiser's curse proper — the tendency to over-select players whose
estimate is inflated by noise — needs a correction that is *non-uniform*,
shrinking each estimate by its own estimation uncertainty. The docstring already
records that the James-Stein version was tried and reverted, because `xpts_var`
is outcome variance rather than estimation variance. Until a real
estimation-uncertainty signal exists (multi-seed reassembly variance is the
obvious candidate), the selection half of this correction is unimplemented, and
should not be described as done.

---

## 20. Every rolling rate discards the most recent gameweek, and GW2 has none at all

The worst of the late findings, and nothing in 780 tests or the preflight
baseline would have shown it — GW1 is a cold start, so it only bites from the
first in-season decision onward.

`_build_rolling_features` computed each rate as
`x.shift(1).rolling(5, min_periods=1).mean()`. The `shift(1)` came, as the
docstring says, from the "same pattern as `points_model._build_features`" — an
ML feature builder, where the frame legitimately contains the row being
predicted and shifting is the only thing standing between you and a leak.

Here the frame is `history`, already strictly prior to the target gameweek. The
shift was guarding against a leak the truncation had already prevented, and it
cost the newest and most informative gameweek every week.

Measured by giving one player a distinct CBIT each gameweek, so the resulting
rate says unambiguously which gameweeks built it:

| played | CBIT | engine rate | if all used | if newest dropped |
|---|---|---|---|---|
| GW1 | 10 | **0.00** | 10.00 | — |
| GW1–2 | 10, 20 | **10.00** | 15.00 | 10.00 |
| GW1–3 | 10, 20, 30 | **15.00** | 20.00 | 15.00 |
| GW1–4 | 10, 20, 30, 40 | **20.00** | 25.00 | 20.00 |

The engine matched "newest dropped" exactly, at every length.

Two consequences, the first much worse than the second:

**At GW2, every rate is zero.** `shift(1)` on a one-row group is NaN, and the
`fillna(0.0)` at the end turns that into a confident zero — the project's own
recurring failure shape, a fallback that makes missing data indistinguishable
from a real measurement. So on the first in-season decision of the season,
`goal_weight`, `assist_weight`, `defcon_rate`, `key_pass_rate`, `dribble_rate`
and both card rates are all 0. Team goals get split by all-zero weights,
DefCon's Poisson rate is 0 so the CBIT ≥ 10 threshold is unreachable, and two
BPS channels are dead. GW2 projections collapse to appearance points plus clean
sheets and saves — and GW2 is the first week transfers are made.

**All season, form is a week stale.** The five-gameweek window is really
gameweeks n−5…n−1. That is worst exactly where recency matters most: the week
after an injury return, a positional change, or a transfer, the one informative
match is the one ignored.

**Fix:** drop the shift, and give the function `target_gw` so it owns the
leakage boundary itself rather than trusting callers to have pre-truncated —
the same "a comment cannot hold an invariant that spans two files" reasoning
that produced the shared `team_goals` derivation after audit defect 8. Three
regression tests: rates use every played gameweek, a single gameweek yields a
real rate rather than zero, and an untruncated frame still cannot see the
target gameweek.

---

## Structural, not a defect: nothing has been validated for 26/27

`decision_log` and `sim_decision_log` contain **zero scored rows** (180 sim rows
across 90 personas, none scored). The backtest is deprecated as a validation
instrument by design; the live persona cohort is the replacement, and it cannot
produce a signal until several gameweeks have been played.

So the engine enters GW1 with no working validation, while its behaviour is
governed substantially by constants the config file itself marks as untuned:
`transfer_switching_cost`, `ft_terminal_value`, `bench_value_weight`,
`mu_range`, every chip threshold, and the prior-league translation factors.
This is a known and deliberate posture, but it is worth stating plainly: the
first four or five gameweeks are the measurement, not the payoff.

## Smaller items

- `BPS_WEIGHTS.penalty_saved = 7`. Fantasy Football Scout reports penalty saves
  **unchanged at 8** for 26/27. The code assumes `big_chance_saved` stacks to
  net 8 and self-flags this `UNVERIFIED-STACKING`. Low impact; confirmable from
  observed BPS in the first weeks.
- `PRIOR_LEAGUE` translation factors are **1.0** for La Liga, Serie A,
  Bundesliga and Ligue 1 — top-5 leagues treated as Premier-League-equivalent.
  Optimistic, and it systematically over-projects foreign signings, which is
  exactly the cold-start population feeding the GW1 squad. (A real calibration
  was previously rejected for survivorship bias; the defaults are the fallback,
  not a measurement.)
- The cold-start tier (`source`) is not persisted, so it is impossible to audit
  after the fact which tier produced each GW1 pick. Same provenance gap as §4.
- `.env` has `DB_PATH=fpl_bot_v2.db`, a **relative** path resolved against the
  working directory. This project has already lost five weeks to a wrong-DB
  incident; the filename is right now, but the fragility is unchanged.
- `DEFCON.cap_per_match` is defined but never read — `compute_defcon_points`
  returns `rules.points`, which happens to equal the cap.

## Checked and found sound

Recorded so they are not re-investigated:

- **`selling_price`** — integer tenths, floor division on the rise, full fall.
  Matches FPL's rule exactly.
- **Chip half-boundary** — `current_gw <= half_boundary` treats GW19 as
  inclusive, which matches the official "must be played before the GW19
  deadline" wording. No off-by-one.
- **DefCon minutes scaling** — `assemble.py:161` scales the action rate by the
  drawn minutes band before the threshold test, so cameos correctly almost
  never trigger.
- **`key_passes`** — genuinely sourced from Understat (`player_xg_stats`, 4,313
  non-zero rows) and drawn per scenario. Not a dead field, despite being zero
  in `player_match_events`.
- **Minutes-model rolling features** — season-scoped (no cross-season leakage),
  with cross-season `avg_minutes_5gw_global` and a career average carrying
  signal across the boundary.
- **Live odds wiring** — `pipeline._load_live_match_odds` correctly reads
  `fixture_odds` as-of the deadline. `assemble.load_match_odds`
  (`historical_fixture_odds`) is confined to the backtest and walk-forward
  gate. Defect 9's lesson held; the problem in §2 is coverage, not wiring.
- **DefCon's Poisson assumption** — overdispersion is real (variance/mean 1.52
  median across 110 defenders with ≥10 matches, against Poisson's 1.0), but the
  aggregate effect on P(CBIT ≥ 10) is only 1.05× (empirical 0.314 vs Poisson
  0.299). A refinement, not a defect.
- **26/27 rule constants** — scoring table, DefCon thresholds (10 CBIT / 12
  CBIRT, cap 2), CBI at 1-per-3, removal of the being-tackled penalty, and the
  chip structure (four chips per half, no Assistant Manager) all verified
  against Premier League and Fantasy Football Scout sources.
- **Fixture data** — 380 fixtures, 38 gameweeks, 10 per gameweek, no doubles or
  blanks yet, GW1 deadline 2026-08-21 17:30. Correct.

In the optimiser layer specifically:

- **The free-transfer carry constraint** —
  `ft[w+1] <= ft[w] - n_trans[w] + hit[w] + 1`, floored at 1 and capped at 5.
  Algebraically correct in all three regimes (bank, spend exactly, take hits).
- **The bank flow** — `bank[w+1] = bank[w] + Σ sell·out − Σ cost·in` with
  `bank ≥ 0`, and it does collapse to the old `Σ cost ≤ budget` when no
  purchase prices are supplied, exactly as claimed.
- **Squad and formation constraints** — 2/5/5/3, max 3 per club, XI of 11 with
  exactly 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD. Matches FPL, including that 5-2-3
  is legal.
- **Captaincy in the ILP** — a second copy of the score with
  `captain <= starting` and exactly one captain per week. Correct doubling.
- **`_squad_xpts`** — `nlargest(11)` plus captain bonus, the right definition
  of squad value (which is what makes §13 visible as a divergence).
- **`chips_used_this_season`** — de-duplicates on `(chip, gameweek)`, so
  re-running a gameweek no longer consumes a chip.
- **`_get_wc_half_boundary`** — season-scoped, so the multi-season database no
  longer yields a boundary of 113.
- **`load_latest_samples`** — filters to a single `created_at`, so a re-run's
  fresh draws are never paired scenario-by-scenario with another run's.

---

## Sequenced remediation

See `docs/superpowers/plans/engine-fix-plan-2026-08-18.md`. In short: only §3
changes the GW1 squad, so it is the only item that must be decided before
Friday's deadline. §10 and §11 are one-line fixes with rule-verified answers.
§1 is two lines of SQL. Everything else is post-GW1 work.

## The pattern

The last audit's closing note was that its defects came from *"a change to
stored state altering behaviour in a consumer nobody was looking at"*, and that
`preflight.py` now guards the decision surface against that.

The defects here are a different class, and preflight would not catch any of
them. §1, §2, §6 and §8 are all cases where **a value is present, plausible,
and wrong** — a zero that means "unpublished", a flat constant standing in for
a missing market, an agreement statistic dominated by trivially-correct zeros,
a "finished" flag that does not mean finished. The baseline diff cannot see
them because they do not change the answer *today*; they change what the answer
is made of.

The instrument that would catch this class is not a diff against yesterday but
a check that each input is **on the distribution the model was fitted on**.
Preflight already computes `degenerate` for constant features; the natural
extension is a per-feature train/serve distribution comparison that fails when
a live column's range falls outside its training range. That single check would
have caught §1 outright, and defects 9, 12 and 14 from the previous audit.

The optimiser layer fails differently, in two shapes of its own.

**Two implementations of one concept, drifting.** §12 (one optimiser decides
the wildcard, another executes it), §13 (two definitions of "squad points" in
adjacent modules), §10 (wildcard and free hit handled inconsistently in the
same function). This is the same root cause the project already named when it
retired the backtest as a validation instrument — *"a second implementation of
the decision loop… that divergence is how a whole class of live-only defects
survived a green test suite"* — reappearing inside the optimiser rather than
beside it. The countermeasure is the one already used for
`roll_forward_free_transfers`: one shared function, so the two paths cannot
drift.

**A guard whose premise has expired.** §16 (three risk layers inert because a
calibration set `mu_baseline` to 0, with docstrings still asserting they are
live), §14 (thresholds written as means, now serving as probabilities), §17 (a
scarcity guard that depends on DGW data which does not yet exist), §15 (chip
gates conditioned on fixture structure that has not materialised). Each was
correct when written. What changed was the world around it, and nothing
re-checks the premise.

That second shape has no test-shaped answer, because nothing is broken — the
code does exactly what it says. What would catch it is asserting the
*consequence* rather than the mechanism: a preflight check that each chip is
reachable given current data, and that the risk layer is either active or
declared inactive. A chip that cannot fire under any input this season is a
fact worth failing on, and today nothing computes it.
