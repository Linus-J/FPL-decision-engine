# Live Simulation Engine v1 — Design

## Purpose

Run ~100 parallel "shadow" FPL managers through the live 2026-27 season,
each varying in risk posture (existing `risk_mode`/`variance_weight`/
`max_ownership_differential`/`hit_min_gain_buffer` knobs) and chip-timing
aggressiveness. None are ever submitted to the real FPL app — this is
purely to compare strategies post-season and improve next season's model.
Phase 2 of the dashboard work (`plan/dashboard-v1.md`); results surface in
a new dashboard page.

## Decisions

- **Persona axes**: `risk_mode` ∈ {safe, balanced, aggressive},
  `variance_weight`, `max_ownership_differential`, `hit_min_gain_buffer`
  (all existing `OptimiserConfig` fields), plus a new `chip_aggressiveness`
  multiplier applied to `ChipTimingThresholds`' four point-thresholds.
  Other `OptimiserConfig` fields (horizon, form window, curse shrinkage,
  bench value weight) stay fixed at their real defaults for every persona —
  varying them would confound strategy comparison with pipeline-tuning
  noise.
- **Personas are generated once per season** (seeded RNG, reproducible) and
  persisted to `sim_managers`. Every later run loads the same rows — a
  given sim's identity/config never changes mid-season.
- **Decision-loop reuse**: `agent/decision_engine.py::run()`'s body is
  extracted into a private `_run_decision_cycle(...)`. `run()`'s public
  signature is completely unchanged (still called identically from
  `scripts/run_agent.py`); a new `run_for_persona(persona, season)` calls
  the same core with a persona's config overrides and sim-scoped storage.
  This guarantees sims always run the actual current decision logic, not a
  frozen fork of it.
- **No submission path exists for sims** — `run_for_persona` never imports
  `agent/fpl_client.py`. Not a disabled flag; the capability isn't present.
- **Storage**: new `sim_managers` + `sim_decision_log` tables in the same
  `fpl_bot_v2.db`, `sim_decision_log` an exact mirror of `decision_log`'s
  shape plus a `sim_manager_id` FK.
- **Scheduling**: `scripts/run_simulations.py`, invoked right after
  `scripts/run_agent.py` in the existing Fri/Sat/Sun systemd job, as a
  separate process (a sim crash can't propagate into or block the already-
  completed real run). Reuses that run's already-computed
  projections/players/DGW data — no recomputation.
- **Per-persona isolation**: each persona's step is wrapped in try/except
  in `simulation/engine.py`'s loop; one failure is logged and skipped, never
  aborts the batch.
- **Dashboard**: leaderboard (cumulative actual outcome per persona, ranked,
  vs. the real squad) + drill-down reusing the existing Decision History
  page's query functions with an added `sim_manager_id` parameter
  (`None` = real log, matching the same additive-parameter pattern used
  throughout this codebase).

## Architecture

```
simulation/
  __init__.py
  personas.py    — generate_personas(n, seed) -> list[SimPersona];
                    load_or_create_personas(db, season) -> persists once
  engine.py      — run_all_personas(season, players, projections, dgw_gws,
                    bgw_gws): loads/creates personas, calls
                    agent.decision_engine.run_for_persona() per persona,
                    each wrapped in try/except

data/models.py   — + SimManager, SimDecisionLog

scripts/
  run_simulations.py            — CLI entrypoint, called after run_agent.py
  backfill_decision_outcomes.py — extended: also backfills
                                   sim_decision_log.actual_outcome per sim

dashboard/
  pages/6_Simulations.py — leaderboard + drill-down
  data/simulations.py    — leaderboard query; decisions.py's functions gain
                            an optional sim_manager_id param
```

## Config threading (the actual diff, traced against current code)

- `optimiser/squad.py::optimise_squad` / `optimise_starting_xi`: add
  `config: OptimiserConfig | None = None`; `cfg = config or OPTIMISER`;
  replace the direct `OPTIMISER.x` reads (3 sites) with `cfg.x`.
- `optimiser/transfers.py::evaluate_transfers`: add
  `config: OptimiserConfig | None = None` and
  `transfer_rules: TransferRules | None = None` (the separate `TRANSFERS`
  global governing hit costs/banking), same default-to-global pattern.
- `optimiser/chips.py::recommend_chip`: add
  `chip_timing: ChipTimingThresholds | None = None`. `_try_tc/_try_bb/
  _try_fh/_try_wc` are nested closures *inside* `recommend_chip`, so one
  local rename (`timing = chip_timing or CHIP_TIMING`, then `CHIP_TIMING.x`
  → `timing.x` throughout the function body) covers all four automatically.
  The one standalone helper it calls, `_panic_shrink`, gets its own new
  parameter and two call-site updates.
- `projection/cold_start.py::build_initial_squad`: one passthrough
  `config: OptimiserConfig | None = None` to its internal `optimise_squad`
  call.
- **Every new parameter defaults to `None` → today's global config.**
  `agent/decision_engine.py::run()` is never modified to pass anything new
  to these functions, so real-bot behaviour is provably unchanged.

## Decision-loop refactor (`agent/decision_engine.py`)

```python
def _load_squad_state(sim_manager_id: int | None, team_id: int) -> tuple[list[int], float, int]: ...
def _load_own_decision_log(sim_manager_id: int | None) -> pd.DataFrame: ...
def _record_decision(sim_manager_id: int | None, gameweek, decision_type, details, projected_gain, dry_run) -> None: ...

def _run_decision_cycle(
    season: str, dry_run: bool, force_chip: Chip | None,
    config: OptimiserConfig, chip_timing: ChipTimingThresholds,
    team_id: int | None, sim_manager_id: int | None,
) -> dict:
    # today's run() body, with OPTIMISER/CHIP_TIMING -> config/chip_timing,
    # and _load_my_squad/_load_decision_log/_log_decision -> the three
    # sim_manager_id-aware helpers above.

def run(season="2026-27", force_chip=None, dry_run=None) -> dict:
    # UNCHANGED signature and call sites (scripts/run_agent.py untouched)
    dry_run = settings.dry_run if dry_run is None else dry_run
    return _run_decision_cycle(season, dry_run, force_chip, OPTIMISER, CHIP_TIMING,
                                settings.fpl_team_id, sim_manager_id=None)

def run_for_persona(persona, season: str) -> dict:
    cfg = dataclasses.replace(OPTIMISER, risk_mode=persona.risk_mode,
                               variance_weight=persona.variance_weight,
                               max_ownership_differential=persona.max_ownership_differential,
                               hit_min_gain_buffer=persona.hit_min_gain_buffer)
    timing = dataclasses.replace(CHIP_TIMING,
        wildcard_pts_gain_threshold=CHIP_TIMING.wildcard_pts_gain_threshold * persona.chip_aggressiveness,
        free_hit_single_gw_gain_threshold=CHIP_TIMING.free_hit_single_gw_gain_threshold * persona.chip_aggressiveness,
        bench_boost_min_bench_xpts=CHIP_TIMING.bench_boost_min_bench_xpts * persona.chip_aggressiveness,
        triple_captain_min_gain=CHIP_TIMING.triple_captain_min_gain * persona.chip_aggressiveness)
    return _run_decision_cycle(season, dry_run=True, force_chip=None, config=cfg,
                                chip_timing=timing, team_id=None, sim_manager_id=persona.id)
```

## Storage schema

```python
class SimManager(Base):
    __tablename__ = "sim_managers"
    id: int  # primary key
    season: str
    label: str
    risk_mode: str
    variance_weight: float
    max_ownership_differential: float
    hit_min_gain_buffer: float
    chip_aggressiveness: float
    created_at: datetime

class SimDecisionLog(Base):
    __tablename__ = "sim_decision_log"
    id: int  # primary key, autoincrement
    sim_manager_id: int  # FK -> sim_managers.id
    gameweek: int
    decision_type: str  # "lineup" | "transfers" | "chip", same as decision_log
    details: str  # JSON, same shape as decision_log.details
    projected_gain: float
    actual_outcome: float | None
    created_at: datetime
    # index on (sim_manager_id, gameweek)
```

## Persona generation

`simulation/personas.py::generate_personas(n=100, seed=<fixed constant>)`:
a `numpy.random.default_rng(seed)` samples, per persona:
- `risk_mode`: uniform choice from {safe, balanced, aggressive}
- `variance_weight`: uniform magnitude around the real default (0.0)
- `max_ownership_differential`: uniform magnitude around the real default (0.5)
- `hit_min_gain_buffer`: uniform around the real default (2.0); lower = more
  hit-happy
- `chip_aggressiveness`: uniform multiplier (e.g. 0.5–1.5); lower = plays
  chips more eagerly, higher = more patient

Exact ranges are tuning constants in `personas.py`, not load-bearing
design — adjustable after observing a season's worth of behaviour.
`load_or_create_personas(db, season)`: if `sim_managers` already has rows
for `season`, return them unchanged; only generate+persist on the season's
first-ever call.

## Dashboard

- **Leaderboard** (`pages/6_Simulations.py`): one row per persona — label,
  risk_mode, key params, cumulative `actual_outcome` (summed across its
  `lineup` rows in `sim_decision_log`), rank, delta vs. the real squad's own
  cumulative actual outcome (`decision_log`, once backfilled).
  `dashboard/data/simulations.py` holds the query.
- **Drill-down**: `dashboard/data/decisions.py`'s functions gain an
  optional `sim_manager_id: int | None = None` parameter — `None` reads
  `decision_log` (today's behaviour, unchanged), a value reads
  `sim_decision_log` filtered to that manager. No new rendering code needed
  in the Decision History page beyond a persona selector.

## Error handling

- No sim ever calls `agent/fpl_client.py` — the capability doesn't exist in
  `run_for_persona`'s code path, not a flag.
- `simulation/engine.py`'s per-persona loop: try/except around each
  `run_for_persona` call, logs the persona id and exception, continues to
  the next persona.
- `scripts/run_simulations.py` is invoked as a separate process after
  `scripts/run_agent.py` completes (not imported into it) — a crash
  anywhere in the simulation script cannot block or corrupt the real run.
- `SIM_COUNT = 100` (default, in `simulation/personas.py`) is a config
  constant, not hard-coded — tunable if the weekly batch's runtime ever
  becomes a real constraint (unlikely: a single cold-start squad build
  measured at ~0.25s).

## Testing

- `_run_decision_cycle` extraction: existing tests touching
  `agent.decision_engine.run` must pass unmodified; add a byte-for-byte
  comparison test (identical inputs before/after the refactor → identical
  output dict).
- `optimise_squad`/`optimise_starting_xi`/`evaluate_transfers`/
  `recommend_chip`: for each, a test asserting the no-override call and the
  override-set-to-today's-global call produce identical output (proves the
  default path is a true no-op).
- `simulation/personas.py`: determinism test (same seed → byte-identical
  personas) + range-coverage test.
- `simulation/engine.py`: a deliberately-broken persona in the batch proves
  the loop continues and other personas still get recorded.
- `scripts/backfill_decision_outcomes.py` extension: a test covering the
  `sim_manager_id`-scoped path alongside the existing real-log tests.
- Dashboard: same headless `AppTest` smoke-test pattern as the other 5
  pages.

## Out of scope for v1

- Interactive persona editing from the dashboard (personas are generated
  once, seeded — not user-tunable per-sim yet).
- Cross-persona covariance/portfolio analysis (e.g. "which parameter most
  explains variance in outcome") — v1 is a leaderboard, not a stats engine.
- Reducing `sim_decision_log`'s duplication with `decision_log`'s schema
  into a single shared table — kept as two tables for now since merging
  would require a `sim_manager_id NULL = real` convention throughout every
  existing `decision_log` query in the codebase, out of scope for this pass.
