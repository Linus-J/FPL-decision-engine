# Risk-Aware Cold Start v1 — Design

## Purpose

Cold-start squad building (`projection/cold_start.py`) feeds the optimiser a
single deterministic point-estimate xPts per player with no variance at
all — so every persona in the simulation engine picked an identical GW1
squad regardless of its risk configuration (verified live 2026-07-31, see
[[simulation-engine-v1]]). The fix is not per-persona random sampling; it's
giving cold-start projections real variance, since `optimise_squad` already
has a fully-built risk-aware scoring mechanism (`risk_mode`/`variance_weight`,
built for the simulation engine) that silently does nothing when
`xpts_var` doesn't exist. This also fixes a second, related weakness: the
current 3-way `risk_mode` switch (`safe`/`balanced`/`aggressive`) makes
"balanced" mean *exactly zero* variance-awareness by construction — not a
genuine "medium" setting.

**Scope note**: `lambda_mu_for_risk_mode` is called by every gameweek's
transfer/lineup optimisation all season, not just cold start. This redesign
changes that formula globally — a deliberate choice (a risk dial that means
nothing at its own midpoint is wrong all season, not only at GW1), not a
side effect to be surprised by later.

## Decisions

1. **`risk_mode: str` → `risk_level: float`** (-1.0 safe … 0.0 medium …
   +1.0 aggressive), threaded through every place `risk_mode` currently
   exists.
2. **New variance-term formula**: `mu = mu_baseline + risk_level * mu_range`
   (`mu_baseline > 0`, both new `OptimiserConfig` fields, marked untuned
   like this project's other heuristic constants). `risk_level=0` now
   means real, moderate variance-awareness; `-1` can go net risk-averse;
   `+1` leans further into upside variance.
3. **Differential weight (`lambda`) keeps its current sign-based shape**
   (`lambda = risk_level * lambda_magnitude`, zero at the midpoint) —
   chasing ownership differentials is a separate "taste" axis from risk,
   not something "medium" should default to non-zero on.
4. **Cold-start projections gain real `xpts_var`**:
   - Established players (≥5 real prior appearances): the real sample
     variance of their own per-GW points from `player_gw_stats`.
   - New signings / promoted players (no top-flight history): both `xpts`
     *and* `xpts_var` come from pooling real peers in the same (position,
     price-band) bucket — replacing today's synthetic linear formula
     (`base + slope*price`) with actual observed outcomes. Falls back to a
     wider bucket if too few peers; never crashes on sparse data.
   - No changes needed to `optimise_squad`/`build_initial_squad` — they
     already consume `xpts_var` via the config-threading work already
     shipped for the simulation engine.
5. **Real squad**: `risk_level = 0.0` going forward ("medium," now
   meaningful).
6. **Data consequence**: today's 100 `sim_managers` rows (string
   `risk_mode`, all picked an identical cold-start squad) are obsolete
   under this redesign — cleared and regenerated, not migrated. This is
   disposable pre-season verification data, not real decision history.

## Architecture / touch points

```
config/strategy.py
  OptimiserConfig.risk_mode: str            -> risk_level: float = 0.0
  OptimiserConfig.variance_weight            -> renamed mu_range (spread)
  + OptimiserConfig.mu_baseline: float        (new, untuned, > 0)
  # max_ownership_differential unchanged (lambda's magnitude, sign-based)

optimiser/scoring.py
  lambda_mu_for_risk_mode(risk_mode: str, ...) -> lambda_mu_for_risk_level(
      risk_level: float, lambda_magnitude: float, mu_baseline: float, mu_range: float
  ) -> (lam, mu)
      lam = risk_level * lambda_magnitude                      # unchanged shape
      mu  = mu_baseline + risk_level * mu_range                # new shape
  _RISK_MODE_SIGN dict deleted (no longer a categorical lookup)

optimiser/squad.py, optimiser/transfers.py
  cfg.risk_mode -> cfg.risk_level (mechanical); calls to
  lambda_mu_for_risk_mode -> lambda_mu_for_risk_level with mu_baseline/mu_range

projection/cold_start.py
  project_cold_start() gains real xpts_var per player (see Decision 4)
  new: _peer_bucket_stats() -- pools (position, price-band) peers' real
  per-GW points for players with no usable prior season history

data/models.py
  SimManager.risk_mode: Mapped[str] -> risk_level: Mapped[float]

simulation/personas.py
  _RISK_MODES tuple deleted; risk_level sampled via rng.uniform(-1.0, 1.0)

agent/decision_engine.py
  run_for_persona: dataclasses.replace(OPTIMISER, risk_mode=persona.risk_mode, ...)
                -> dataclasses.replace(OPTIMISER, risk_level=persona.risk_level, ...)

dashboard/data/simulations.py, dashboard/pages/6_Simulations.py
  "Risk mode" column -> "Risk level" (float, e.g. rounded to 2dp)

tests
  every risk_mode="aggressive"/"safe"/"balanced" literal across
  test_p3_objective.py, test_chips.py, test_simulation_personas.py,
  test_dashboard_simulations.py, test_decision_engine_sim_storage.py
  becomes a risk_level float
```

## Cold-start variance computation, precisely

**Established players** (≥`MIN_PRIOR_APPEARANCES` real appearances in the
prior season): query the raw per-GW `total_points` for minutes>0 rows
(not just the aggregate `load_prior_season_features` already computes),
compute `xpts = mean`, `xpts_var = sample variance`, same as today's mean
but now with a real spread alongside it.

**New/promoted players**: bucket every ESTABLISHED player from the same
prior season by `(position, price rounded to nearest £1.0m)`. For a new
player's own `(position, price)`, look up that bucket's pooled
(`mean`, `variance`) of real per-appearance points across every player and
every appearance in the bucket. If a bucket has fewer than a minimum
sample size (e.g. 20 player-appearances), widen to position-only (drop the
price constraint) before falling back to today's synthetic linear formula
as a last resort (never silently zero, matching cold_start.py's existing
"no silent 0.0" contract).

## Testing

- `optimiser/scoring.py`: unit tests for the new `lambda_mu_for_risk_level`
  — `risk_level=0` yields `mu=mu_baseline` (not zero); `-1`/`+1` yield the
  expected baseline∓range; `lambda` stays sign-based and zero at 0.
- `projection/cold_start.py`: established player's `xpts_var` matches a
  hand-computed sample variance from seeded fixture data; new-signing
  variance/mean matches the expected bucket pooling; bucket-widening
  fallback triggers correctly on sparse data; the "no silent 0.0" contract
  extended to variance (no player ever gets `xpts_var` undefined/NaN).
  Real gate: `optimise_squad` fed these real-variance cold-start
  projections under different `risk_level` values must pick genuinely
  different squads on the live player pool (spot-checked, not just unit
  fixtures) — closes the loop on the original finding.
- `simulation/personas.py`, `data/models.py`, dashboard, decision_engine:
  mechanical updates to existing tests (rename literals), no new behaviour
  to newly cover beyond what's already tested for the risk-parameter
  plumbing itself.
- Full suite must stay green throughout; live dry-run smoke test after
  each major step, matching this session's established verification
  pattern for anything touching the real decision path.

## Out of scope for v1

- Per-persona genuine Monte-Carlo sampling on top of the variance-aware
  mean (layering literal stochastic draws over the distributional
  approach) — the variance-aware deterministic optimisation already
  answers the diversity problem; revisit only if personas still converge
  too much in practice once real variance is flowing.
- Structural diversity knobs unrelated to risk (forced differential picks,
  formation tilts, budget-allocation bias) — a separate persona-design
  axis, not part of this fix.
- Retuning `mu_baseline`/`mu_range`/`lambda_magnitude` against real
  backtested outcomes — ships with untuned starting values, same
  convention as this project's other heuristic constants.
