# Engine fix plan — 2026-08-18

Remediation for the seventeen findings in `docs/engine-review-2026-08-18.md`.
Section numbers below refer to that document.

> **Completed 2026-08-18.** All tiers landed in one pass rather than being
> staged, at the user's direction, with the GW1 re-run deferred to the end so
> the fixes could be checked as a set rather than one at a time. Outcome: the
> GW1 squad, XI, bench order, captain and cost are **identical** to the
> committed baseline; the only baseline movement is two new chip-reachability
> keys. Suite 774 passing, `ruff check` down from 117 findings to 53
> (line-length only). §7 remains open by necessity — it needs 26/27 BPS that
> does not exist yet.

**Hard constraint:** the GW1 deadline is **Friday 2026-08-21 17:30**. Exactly
one finding (§3) changes the GW1 squad. Everything else either bites from GW2
or is measurement work. The plan is sequenced around that, not around severity.

Every code change lands with `pytest` green — which now genuinely runs, see
§9 — and a `scripts/preflight.py` run, which diffs the
decision surface against `config/preflight_baseline.json`. Where a fix is
*expected* to change the answer, the baseline is updated in the same commit
with the diff quoted in the message — changing the answer is allowed, changing
it silently is not.

---

## Tier 0 — before Friday's deadline

### T0.1 — §3 (curse shrinkage at cold start) — DONE, and it found another defect

The only item that could change the squad about to be entered, so it was the
one genuine judgment call in the set.

`apply_curse_shrinkage` runs in `projection/pipeline.py:291` and never on the
cold-start path, because `decision_engine.py:351` calls
`cold_start.build_initial_squad` directly.

**Argument for switching it on:** the optimiser's curse is a property of
selecting on noisy estimates, and cold-start estimates (prior-season, translated
prior-league, peer bucket, synthetic) are the noisiest the system produces. The
correction is currently absent exactly where it should be largest.

**Argument for leaving it:** the 0.15 strength was fitted against *in-season*
projections, where the measured top-50 bias was +1.2–1.3 pts/player. Cold start's
bias is probably larger but is unmeasured, so 0.15 is not the right number — it
is only more right than 0. Cold start also carries its own conservatism already
(the peer-bucket floor, the `unconditional_moments` conversion).

**What it would actually do:** a uniform shrink toward the (gameweek, position)
group mean does *not* reorder players within a position. It compresses the
spread, which under a budget constraint makes premiums relatively less
attractive and spreads money down the squad. So expect a real change in shape,
not in who the best midfielder is.

**Outcome:** shrinkage is now applied on the cold-start path, behind the same
`OPTIMISER.curse_shrinkage_enabled` flag as in-season. The GW1 squad did not
change at all — same 15, same XI, same bench order, same captain, same £100.0m.
Only the projected total moved, 59.97 → 54.74 xPts, which is exactly the 15%
compression toward the group means doing its job: a more honest number for the
same team.

Wiring it in immediately failed a departure-gate test, which turned out to be a
real defect in the shrinkage itself and live in the in-season path too:
**shrinkage was resurrecting players it had zeroed.** A zero in that frame means
the player will not feature — the unavailable are zeroed and confirmed departures
discounted to 0.0 before shrinkage runs — and pulling them toward a positive
group mean gave a leaver 0.30 xPts and made them selectable again. They also
dragged the mean down, making every real player's correction depend on how many
non-participants were in that week's frame. Both fixed by shrinking only rows
with `xpts > 0`.

### T0.2 — §10, wildcard free-transfer loss (one line, rule-verified)

`optimiser/transfers.py:52-53`. Cannot affect GW1 (no wildcard at cold start),
but it is a verified rule error with a trivially correct fix, so land it while
it is fresh.

```python
# before
if wildcard_played:
    return trules.free_transfers_per_gw
if free_hit_played:
    transfers_made = 0

# after — FPL retains saved free transfers through BOTH chips
if wildcard_played or free_hit_played:
    transfers_made = 0
carried = free_transfers - transfers_made + trules.free_transfers_per_gw
return min(trules.max_banked_free_transfers, max(trules.free_transfers_per_gw, carried))
```

Verify against the Premier League's own worked example: two saved before a GW6
wildcard must give **three** in GW7. Add that as a named regression test
alongside the existing `roll_forward_free_transfers` cases.

### T0.3 — §1, the two zeroed FDR features (two lines of SQL)

`projection/features.py:91-107`. Does not affect GW1 (cold start does not use
the minutes model) but bites from GW2, and it is the cheapest fix in the set.

Wrap the aggregate's input in the existing `_usable()` helper. `AVG` ignores
NULLs, so an all-placeholder window returns NULL and the existing
`COALESCE(…, 1200)` fires as designed:

```sql
SELECT AVG({_usable("CASE WHEN s2.was_home THEN t2.strength_defence_away "
                    "ELSE t2.strength_defence_home END")})
```

and the same for the attack column. Test: assert both columns are 1200, not 0,
for a season whose `team_season_strength` rows are all placeholders — that test
is the actual guard, since the bug is invisible in any season with real data.

---

## Tier 1 — before GW2 (the first in-season decision)

GW2 is when `assemble.py`, the minutes model, the transfer ILP and the chip
gates all run for real for the first time this season. Everything here should
land in that window.

### T1.1 — §14, the payoff gate (highest expected value in the set)

Make the implementation match the documented intent in `strategy.py:283-288`:
`P(gain >= 0)` **in addition to** the point-estimate test, not
`P(value >= threshold)` instead of it.

```python
def _clears_threshold(point_value, threshold, scenario_values, min_probability):
    if point_value < threshold:
        return False
    if scenario_values.empty:
        return True
    return float((scenario_values >= 0.0).mean()) >= min_probability
```

Note this changes the meaning of the `scenario_values` each caller passes: for
Wildcard and Free Hit they are already *gain* distributions
(`gain_distribution`), which is correct. For Triple Captain and Bench Boost they
are *totals* (`load_scenario_totals`), for which `>= 0` is trivially true and
the gate becomes inert. Either pass gain distributions for those two as well, or
gate them on the point estimate only and delete the unused parameter — do not
leave a gate that silently always passes.

This is the fix most likely to restore intended chip behaviour, and it should
land before any chip threshold is retuned, because the thresholds cannot be
tuned while the gate misreports what they mean.

### T1.2 — §11 and §12, the wildcard execution path

Two changes, same commit:

- **§11:** zero the switching cost on a wildcard week. In the objective
  (`transfers.py:360`), use
  `0.0 if (wildcard_active and w == 0) else trules.transfer_switching_cost`.
  The hit term is already zeroed the same way at line 329; this is the missing
  half.
- **§12:** make the decision and the execution use the same optimiser. The
  cheapest honest version is to have `chips.py::_try_wc` evaluate the wildcard
  with the *same* call the engine will execute —
  `evaluate_transfers(wildcard_active=True)` — and read its gain, rather than
  calling `optimise_squad`. That costs an extra ILP solve in the chip
  evaluation, which is acceptable at weekly cadence.

Also correct `docs/decision-engine.md`, which claims wildcard takes the
`optimise_squad` path. It does not.

### T1.3 — §13, one definition of squad points

`transfers._squad_xpts` is the correct one (top 11 + captain). Move it to a
shared location and have `chips.py::_try_wc` and `_try_fh` call it instead of
raw 15-man sums. Leave Bench Boost summing all fifteen — that is right for that
chip, and the asymmetry should be commented so it is not "fixed" later.

Expect chip gains to move materially. Do **not** retune thresholds in the same
commit; land the correction, observe, then tune.

### T1.4 — §8, score outcomes only on final data

Add `data_checked` to the `gameweeks` model and the FPL bootstrap ingest, and
gate `backfill_decision_outcomes._gw_finished` on it. Cheap, and it protects the
calibration and persona tables — the only validation instrument that will exist
by October — from provisional bonus written during 26/27's much wider
post-match window.

### T1.5 — §15 and §17, make unfireable chips visible

Not a code fix so much as an instrument. Add to `scripts/preflight.py` a
per-chip reachability report: for each chip and each half, whether any input
state reachable from current data could trigger it. Today it would report that
first-half Bench Boost and Free Hit are unreachable (no DGW or 5-blank BGW in
the fixture list) and that Triple Captain's hold-back multiplier is inert.

That converts §15 and §17 from arguments into a weekly number, which is what
lets you decide whether to relax the DGW gate before the chips expire rather
than after.

---

## Tier 2 — the first international break

### T2.1 — §2, fixture signal beyond the current gameweek

The largest single modelling gain and the largest change, hence the break
rather than mid-season.

Give `assemble.py` a team-strength fallback for fixtures with no odds, instead
of flat `1.35 / 1.15`. `cold_start.load_prior_defence_strength_by_code` plus
`fixture_multiplier` already do exactly this and can be reused. Calibrate the
strength→λ mapping against the seasons in `historical_fixture_odds`, where both
sides are observable: fit λ from strength, check it reproduces the odds-implied
λ within tolerance, then use it only where odds are absent.

Success criterion: on GW2–5 projections, clean-sheet probability should span a
real range rather than the single value 0.317.

### T2.2 — §4 and §5, tackle provenance and definition

- Set `source` correctly when WhoScored patches a row (`whoscored.py:160`), or
  add a per-field provenance column. Without this, §5 cannot be measured.
- Decide one definition of `tackles` and enforce it. FPL's DefCon and BPS both
  count tackles **won**, so filter WhoScored's `Tackle` events on
  `outcome_type == "Successful"`, matching what it already does for `TakeOn`.
- Then re-derive `defcon_rate` and re-check the DEF bonus residual: the
  acknowledged "DEF over-credited, 14.4% modelled vs 9.1% real" may substantially
  be this.

### T2.3 — §6 and §7, goalkeeper bonus

Once four or five gameweeks of 26/27 BPS are observable:

- Re-calibrate `GK_BONUS_SAVE_SCALE` against **26/27** outcomes, not 25/26.
  The current 0.45 targets a rate the rules deliberately changed.
- Decide whether to source `saves_in_box` and `big_chances_saved`. They are the
  two 26/27 GK buffs and are currently zero everywhere, so the engine models the
  defender nerf without either compensating buff.
- Re-report the bonus agreement **conditional on bonus being awarded**, by
  position. The unconditional 88.9% is not a useful instrument.

### T2.4 — §16, resolve the inert risk layer

Decide explicitly, and make the code and docs agree either way:

- If `mu_baseline = 0` is right, say so, and mark `captaincy.py`'s
  scenario-based path and the EO machinery as dormant rather than active.
  Correct the two docstrings that claim otherwise.
- If covariance-aware captaincy is wanted, it needs a non-zero `mu` — which
  means re-running the calibration that chose 0.0, this time over GW6–38 as
  that calibration's own note recommends, and separately for the captaincy
  path, since a `mu` that is wrong for squad selection may be right for
  captaincy.

Do not leave it as is: three documented capabilities that do not run is the
failure mode §16 describes.

---

## Tier 3 — measurement-dependent, not before GW6

- **§18 / `max_hits_per_gw`** — decide whether the self-imposed cap of 2 should
  exist at all. It is not an FPL rule.
- **Prior-league translation factors** (review §J) — all top-5 leagues at 1.0
  over-projects foreign signings. A real calibration was rejected for
  survivorship bias; the honest interim is a literature-informed discount
  (~0.85–0.95) rather than parity, but it changes cold start, so it belongs
  with next season's cold start rather than mid-season.
- **Chip threshold retuning** — only after T1.1 and T1.3, and only with persona
  evidence. Retuning against a misreporting gate would bake the error in.
- **§9 / test gate** — add a `conftest.py` or set `pythonpath` in
  `pyproject.toml` so the documented `pytest` invocation collects, and either
  fix or explicitly ignore the 117 ruff findings. Hygiene, but it is what makes
  everything above verifiable.

---

## Sequencing summary

| When | Items | Nature |
|---|---|---|
| Before Fri 21 Aug | §3 decision, §10, §1 | One judgment call, two safe fixes |
| Before GW2 | §14, §11, §12, §13, §8, §15/§17 instrument | Correctness, live from GW2 |
| International break | §2, §4, §5, §6, §7, §16 | Modelling and calibration |
| From GW6 | thresholds, prior-league, hygiene | Needs real data |

The single highest-value item is **§14** — it is a confirmed
implementation/intent mismatch, it plausibly explains a season-long symptom the
project has already complained about twice, and it must be fixed before any
chip tuning is meaningful. The most urgent is **§3**, only because the deadline
forces it.
