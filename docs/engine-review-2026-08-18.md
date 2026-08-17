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

**Nine defects, two of them in the same failure class the last audit closed.**

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

---

## Suggested order of work before the deadline

Only §1 and §2 plausibly change the GW1 squad, and §2 only via four of ten
fixtures. §3 changes it directly.

1. **§1** — a two-line SQL fix, and it silently degrades every week until FPL
   publishes strengths.
2. **§3** — decide whether the cold start should shrink. It changes the squad
   you are about to enter.
3. **§8** — cheap, and it protects the only validation instrument that will
   exist by October.
4. **§2** — the largest modelling gain, but a real change; better done in the
   first international break than four days before GW1.
5. **§5/§4** — re-scrape tackles from one source, or take WhoScored's outcome
   type into account, and fix `source` so this is measurable next time.
6. **§6/§7** — re-calibrate goalkeeper bonus once 26/27 BPS is observable.

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
