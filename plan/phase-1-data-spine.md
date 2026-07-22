# Phase 1 — Data Spine (leakage fix + train/serve consistency). Granular task plan.

Companion to `v2-build-plan.md` §1, §3, §6.5, §7 and Appendix A. **Blocking phase — nothing downstream is trustworthy until this lands.**
Branch: `v2` (in-place, per decision #1). No code merged to `master` until the exit gate passes.

**Revised 2026-07-22 after the plan-critic pass** (findings C1–C3, M1–M4, all verified in code). The original T3/T4/T6 specs were unsound: they each passed their own acceptance gate while being wrong end-to-end, because the historical-backfill path (vaastav per-GW CSV) and the live path (FPL bootstrap) populate the *same* snapshot columns with *different quantities*, and the as-of machinery (per-season deadlines, historical fixtures) that T4/T6 read against does not exist in the schema. This revision adds the missing plumbing (T2.5, T3a), reconciles column semantics + adds a parity test (T3), turns the exit gate into a real pass/fail (T4), and makes GW1 cold-start — the actual mid-Aug deliverable — an explicit task (T7).

**Phase-1 exit gate (the one number that matters):** re-run the backtest **leakage-free AND train/serve-consistent** on 25/26 and record the *honest* baseline. Gate is pass/fail (see T4), not "record a number." Tasks 0–4 deliver it. T5–T7 unblock Phase 2 / the GW1 deliverable and can run in parallel after T4.

---

## Leak & skew inventory (grounded in current code — verified 2026-07-22)

| # | Location | Defect | Fix task |
|---|---|---|---|
| L1 | `scripts/backtest.py::_load_players_snapshot` | `players.status, form, ict_index, influence, creativity, threat, selected_by_percent, chance_of_playing_next_round` read at inference as *latest*, not as-of GW | T4 |
| L2 | `scripts/backtest.py::_load_all_stats` | same `players.*` columns joined into **training** rows | T4 |
| L3 | `projection/features.py::load_player_enrichment` | current `transfers_in/out_event, selected_by_percent, injury_severity` broadcast onto all historical rows | T4 |
| L4 | `data/models.py::FixtureOdds` `fixture_id UNIQUE` + `features.py::load_fixture_odds` | one mutable odds row per fixture; no as-of read | T6 |
| **C1** | `fpl_api.py::write_player_snapshots` (live) vs T3 (backfill) | **train/serve skew**: live writes bootstrap **cumulative** ICT/influence/creativity/threat + `selected_by_percent` (0–100); vaastav `merged_gw` carries **per-GW** ICT + `selected` (raw count). Same columns, different quantities. Backtest passes (all-vaastav) while live is corrupted. | T3 |
| **C2** | T6 odds stamping vs `load_fixture_odds` `< deadline` | FPL deadline ≈ first kickoff − 90 min; odds stamped `kickoff − ε` are *after* the deadline, so `< deadline` excludes them all → `COALESCE(…, 0.2/0.5)` defaults. Odds anchor silently untested. | T6 |
| **M1** | `data/models.py::Gameweek` | no `season` column → one row per GW-number → `deadline(gw)` returns the wrong season's calendar in a historical backtest | T2.5 |
| **M2** | `scripts/backfill_history.py` | writes only `PlayerGameweekStats`; no historical `fixtures`/`gameweeks` → T3 kickoff anchor + all FDR/odds JOINs default for past seasons | T3a |
| **M3** | `data/models.py::Player` (`fpl_id` only) + `backfill_history.py::_build_fpl_id_map` | FPL reassigns `element` ids per season → multi-season backfill by element id mis-maps players (wrong-player rows) | T2.5 + T3 |
| **M4** | T4 exit gate | "record a number, expect <50" cannot fail; grep guard omits L3 columns | T4 |

Point-in-time SAFE (do not touch): `player_gw_stats`, `player_xg_stats`, and rolling `.shift(1)` features built off them.

---

## T0 — Branch + config truth ✅ DONE (`6ce6f1d`)
ScoringRules clean-sheet fix (6→4), DefConRules, BPSWeights (Appendix A.3), singletons; `tests/test_scoring_rules.py` (9, synthetic). ruff clean.

## T1 — Snapshot schema ✅ DONE (`d6b639f`)
`PlayerStateSnapshot` (append-only, `UNIQUE(player_id, snapshot_ts)`, ts-index, FK→players). Additive, idempotent.

## T2 — Snapshot-WRITE ingest ✅ DONE (`ad1fd10`)
`write_player_snapshots(bootstrap, snapshot_ts, season)` append-only via `on_conflict_do_nothing`; wired into `run_full_ingest`. **Semantics note (C1):** the live path writes bootstrap **cumulative-to-date** ICT/influence/creativity/threat and `selected_by_percent` (0–100). This is self-consistent for the live season; **T2 stands** — the reconciliation burden is on T3's backfill to reproduce the *same* cumulative-as-of-deadline quantities.

---

## T2.5 — Schema hardening (NEW; fixes M1, M3) — precondition for as-of reads

**Scope**
- `data/models.py::Gameweek`: add `season: str` (String(7)); primary/unique key becomes `(season, id)` where `id` is the GW number. Nothing FKs to `gameweeks.id` (both `PlayerGameweekStats` and `Fixture` carry a plain `gameweek` int), so this restructure is safe. Rebuild the table via `init_db` on a fresh spine (Phase-1 DB is regenerated from ingest+backfill).
- `data/models.py::Fixture`: add `season: str` (String(7)); change `fpl_id UNIQUE` → `UNIQUE(season, fpl_id)`; add index on `(season, gameweek)`. Feature JOINs must key on `(season, gameweek)`, not `gameweek` alone.
- `data/models.py::Player`: add `code: int` (FPL's cross-season-stable player code), unique. `fpl_api.py::upsert_players` reads `p["code"]`.
- `fpl_api.py::upsert_gameweeks`/`upsert_fixtures`: accept + write `season`.

**Acceptance gate**
- `init_db()` on a fresh DB builds the new columns/constraints; unit test asserts `Gameweek` composite key and `Fixture (season, fpl_id)` uniqueness reject cross-season collisions correctly.
- `upsert_players` populates `Player.code` from a sample bootstrap element.

---

## T3a — Per-season deadlines + code crosswalk ✅ DONE (fixes M1-boundary, M3) — precondition for T3/T4

**Delivered (`scripts/backfill_history.py`)**
- `compute_gw_deadlines(fixtures_df)` + `upsert_gameweek_deadlines(season, …)`: populate `Gameweek` per season with `deadline_time = first kickoff of (season, gw) − 90 min` from `{VAASTAV_BASE}/{season}/fixtures.csv`. **Team-agnostic** (only needs `event` + `kickoff_time`) — this is all the T3/T4 as-of boundary requires.
- `element_code_map(players_raw_df)` + `build_code_to_dbid_map()` + `resolve_player_id(…)`: map vaastav `element → code → players.id` via `players_raw.csv`. Rewired `_ingest_dataframe` off the season-unstable `fpl_id` (fixes M3 in the *existing* stats backfill too). Players who left the league (no current `players.code`) are skipped, not misjoined.

**Gate:** `tests/test_backfill_deadlines.py` (7, synthetic + temp DB) — deadline math, code mapping, season-scoped idempotent writes. *(Live-run coverage verification is folded into T3's run, which needs network + the live DB.)*

**⚠️ Scope split discovered during T3a — deferred to T3b (a T6 dependency, NOT on the T4 critical path):** full historical `Fixture` rows with correct `team_h/team_a`. vaastav fixture team ids are per-season, but `teams` is single-season (current bootstrap) and `Fixture.team_h_id` is an FK to it; relegated teams don't exist in the current table. Storing historical fixtures correctly needs **season-aware `teams`** (add `season` + stable `code`, per-season backfill from `teams.csv`, and a season-aware fixture→team reference) — a T2.5-sized change. Only FDR/odds JOIN *quality* for past seasons depends on it; the honest baseline (a leaky-vs-leakfree comparison) does not.

## T3b — Season-aware teams + historical fixtures (NEW; the deferred M2 remainder) — gates T6 odds JOINs
- `Team`: add `season` + stable `code`; per-season backfill from `{season}/teams.csv`.
- Historical `Fixture` rows with `(season, team refs, kickoff, difficulty)` resolved via the team crosswalk; `load_fixture_difficulty`/`load_fixture_odds` JOIN on `(season, gameweek)` return non-default rows.
- **Gate:** ≥95% of `player_gw_stats` (season, gw) rows resolve to a `Fixture` for the player's team; FDR/odds non-default for a backfilled season.

---

## T3 — Snapshot backfill, reconciled semantics + parity test ✅ DONE (fixes C1, M3)

**Delivered (`scripts/backfill_history.py`):** `compute_snapshot_rows(df, deadlines)` (pure) + `write_snapshot_rows(...)`, wired into `backfill()`. For a snapshot informing GW g (stamped `deadline(g) − 1min`): ICT/influence/creativity/threat = **cumulative through GW g-1** (matches bootstrap after g-1 GWs; DGWs summed per GW); `now_cost`/transfers from the GW-g row; `selected_by_percent = selected / (Σselected/15) × 100`; `form` = prior-window mean points. Mapped element→code→players.id (M3); departed players skipped.

**Gate:** `tests/test_snapshot_backfill.py` (8) — incl. the **C1 parity test** (snapshot GW g ICT == Σ per-GW ICT over GWs < g), DGW aggregation, ownership scale, idempotent writes. Suite 34/34.

**⚠️ Residual train/serve skew (documented, Phase-2 to close):** `status`, `chance_of_playing_*`, `news` are not recoverable from `merged_gw` → backfill defaults them (`'a'`/None/`''`) while live has real values. `form` is a prior-window proxy, not the exact bootstrap 30-day form. Parity test asserts only the *exactly* reconcilable features (ICT cumulative, ownership). Phase 2 should move `form`/ICT to rate features computed identically on both paths, and source a historical availability signal if one becomes free.

### T3 (original spec retained below for reference)

**Scope**
- `backfill_history.py`: emit `PlayerStateSnapshot` rows from `merged_gw.csv`, **reconciled to the live bootstrap semantics** (C1):
  - `ict_index, influence, creativity, threat`: store **cumulative-to-date as-of the GW deadline** (running sum of per-GW values over prior GWs of that season), NOT the per-GW value. Matches what live bootstrap holds at that moment.
  - `selected_by_percent`: convert vaastav `selected` (raw count) → percent using the season's total entries (documented per-season constant, or the archived total). Never store the raw count in this column.
  - `form`: FPL form = mean points over prior ~30 days; compute from prior GWs (do **not** trust `merged_gw.form` until verified pre-GW — open question from the critic).
  - `now_cost` from `value/10`; `transfers_in/out_event`, `status`, `news` as-of.
  - Map `element → code → players.id` via the T3a crosswalk (M3); never join on raw `element`.
  - `snapshot_ts = deadline(season, gw) − ε`; `gameweek_context = gw`.
- Seasons 21/22–24/25 + 25/26.

**Acceptance gate (the parity test is the point)**
- **Backfill-vs-live feature-parity test** (NEW, the C1 guard): the per-feature *distribution* of a real backfilled 25/26 snapshot matches a synthetic bootstrap-derived snapshot within tolerance (quantile/KS check on `ict_index`, `influence`, `selected_by_percent`). Fails loudly on a units/semantics mismatch.
- Snapshot coverage ≥90% of `player_gw_stats` rows per backfilled season.
- Spot-check 3 named players' cumulative `ict_index`/`selected_by_percent` as-of a GW vs a hand-computed value from the CSV.

---

## T4 — Leakage-free reads + honest baseline ✅ DONE (fixes L1–L3, M4, M5)

**HONEST BASELINE (2026-07-22, `results/backtest_2526_v1_leakfree.csv`):** leakage-free walk-forward on **2025-26**, GW6–38 → **43.0 actual pts/GW** (total 1420), predicted xPts **41.4** ≈ actual (well-calibrated, no look-ahead inflation). The old **leaky** v1 baseline was **~50 pts/GW** → removing the leak dropped it **~7 pts/GW**, exactly the predicted direction. **Exit gate met:** leakfree < leaky.

**Delivered:**
- `backtest.py::_load_all_stats` / `_load_players_snapshot` read dynamic columns from `player_state_snapshots` as-of `deadline(season, gw)`; static (position/team) from `players`. Call-site simplified.
- `features.py::load_player_enrichment` as-of per (player, gw, season) from snapshots (L3 killed); `add_enrichment_features` re-keyed; `add_fdr_features` fill made robust to all-NaN (no historical fixtures yet → constant default, not a leak).
- `minutes_model`/`points_model` `_load_training_data` sourced from snapshots too.
- `tests/test_no_leakage.py`: grep guard (4 files, incl. L3 cols per M4) + as-of canary + numpy-int regression guard.

**Findings surfaced during the live run (all fixed):**
- **M5** — `PlayerGameweekStats` unique was `(player_id, gameweek)` (no season) → multi-season backfill collided, dropping ~70% of recent-season actuals. Fixed to `(player_id, gameweek, season)` + both conflict targets; DB rebuilt → 2025-26 now 100% coverage (29,338 rows).
- **numpy-int** — `available_gws` are numpy int64; SQLite silently matched no `gameweeks.id` → every GW skipped. Cast to `int` in the as-of read; regression-guarded.
- **FDR all-NaN fill** — `add_fdr_features` median-fill was a no-op when every row is NaN (no historical fixtures) → `NaN in X` at model fit. Falls back to the constant default.

*(Original T4 spec retained below.)*

### T4 (original spec)
Leakage-free reads + REAL pass/fail exit gate (REVISED; fixes L1–L3, M4)

**Scope**
- Rewrite `backtest.py::_load_players_snapshot` + `_load_all_stats` to source the dynamic `p.*` columns from `player_state_snapshots`, latest row with `snapshot_ts < deadline(season, gw)` (deadline resolved via T2.5's `(season, gw)`).
- Fix `features.py::load_player_enrichment` (L3): read `transfers_in/out_event, selected_by_percent, injury_severity` from the as-of snapshot, or drop them from the historical training path.
- Pin training determinism (seed the minutes/points estimators) per the reproducibility rule.

**Acceptance gate (pass/fail, not record-only)**
- Grep guard `tests/test_no_leakage.py`: no `p.form|p.ict_index|p.influence|p.creativity|p.threat|p.selected_by_percent|p.status|p.chance_of_playing|p.transfers_in_event|p.transfers_out_event|p.injury_severity` in any training/backtest read (L3 columns **added** to the pattern per M4).
- **Synthetic-leak canary:** inject a future-dated value into a snapshot; assert the as-of read excludes it.
- **Comparison gate:** `pts_leakfree < pts_leaky − margin` (removing leakage must *reduce* the flattered ~50). Record `results/backtest_2526_v1_leakfree.csv` + a one-line honest-baseline note here.
- `python scripts/backtest.py --season 2025-26 --start-gw 6 --end-gw 38` runs end-to-end.

---

## T5 — Event data + 26/27 BPS recompute (§3.3–3.4) — unblocks Phase 2

*Larger; starts once T2.5 lands. Full scope retained (Monte-Carlo BPS sim kept — decision 2026-07-22).*

**Scope**
- `data/ingestors/fbref.py` (new) via `soccerdata` → `player_match_events` (shots, shots_in_box, CBI, tackles, recoveries, saves, saves_in_box, key_passes, npxg, xa …). Rate-limited, cached.
- `projection/bps_sim.py` (new): deterministic 26/27 BPS from `BPS_WEIGHTS` over event rows → per-player-fixture BPS; rank top-3 → bonus.
- `recomputed_bonus` table (§3.4): `bps_2627, bonus_2627` for historical GWs.

**Acceptance gate**
- BPS sim reproduces **old-rules** awarded bonus within tolerance under old-rules weights (sanity harness) before switching to 26/27.
- DefCon and BPS computed independently (no shared CBI term) — asserted by test.
- `recomputed_bonus` coverage ≥95% of finished fixtures with event data.

---

## T6 — Odds history as-of, stamped ≤ deadline (REVISED; fixes L4, C2)

**Scope**
- `data/models.py::FixtureOdds`: `fixture_id UNIQUE` → `UNIQUE(fixture_id, fetched_at)` (append-only). `features.py::load_fixture_odds` reads latest `fetched_at <= deadline(season, gw)`; the JOIN keys on `(season, gameweek)` (T2.5).
- Backfill historical closing odds (1X2 + O/U 2.5) from **football-data.co.uk** → `fixture_odds` stamped `fetched_at = deadline(season, gw) − ε` (NOT `kickoff − ε`, per C2).

**Acceptance gate**
- **Post-filter** non-default odds coverage ≥95% of backtest player-GWs (rows that *survive* the `<= deadline` filter — not mere row existence).
- `load_fixture_odds` returns as-of values (test a fixture with 2 snapshots).
- Live `odds_api.py` writes append-only rows.

---

## T7 — GW1 cold-start harness (NEW; fixes C3) — the actual mid-Aug deliverable

**Problem.** Every gate above runs `--start-gw 6` (5 GWs of current-season data). The real deliverable is the **GW1 squad with zero current-season rows**: `history = all_stats[gameweek < 1]` is empty (models untrainable), snapshots empty, and promoted-team/new-signing players have no history → they collapse to `sp=0.5, xp=0.0` defaults.

**Scope**
- Backtest mode that builds the **initial 15 at GW1 of a held-out season** using only pre-season-available data: prior-season features carried into the new season (via `Player.code`), a promoted-team / new-signing prior (position + price + prior-league or prior-season proxy), and the §6.5 departure gate applied to the pool.
- Prior-season→new-season feature bridge in the feature builder.

**Acceptance gate**
- A held-out-season GW1 initial-15 is constructed with no current-season leakage; every squad slot has a non-default projection source (prior-season or explicit prior — no silent `0.0`).
- No confirmed leaver (§6.5, `status='u'`) appears in the GW1 squad.

*(The promoted-team/new-signing prior model itself is Phase-2 work; T7 defines the harness + gate and the carryover so Phase 2 has a target.)*

---

## Dependency order (revised)
```
T0 ✅ ─ T1 ✅ ─ T2 ✅ ─ T2.5 ✅ ─ T3a ✅ ─┬─ T3 ─ T4  ► EXIT GATE (honest, consistent baseline)
                                          ├─ T5  (parallel, gates Phase 2)
                                          ├─ T3b ─ T6  (season-aware teams → odds JOINs)
                                          └─ T7  (needs T3 + Player.code; gates GW1 deliverable)
```

## Deferred-to-Phase-2 notes (raised by critic, not Phase-1 blocking)
- Per-**team** DGW/BGW (current `Fixture.is_dgw` is a GW-level count heuristic; `DGWStrategy` multipliers need per-team).
- Cumulative ICT is a poor raw feature (volume≠rate); Phase 2 should move to per-90 / rolling-rate features — Phase 1 prioritises train/serve *consistency* over feature quality.
- Price-change model/data feed for `OptimiserConfig.use_price_change_signals` is undefined.
- Postponement/reschedule handling (fixtures moving between GWs changes `(season, gw)` joins).

## Cross-cutting
- Every task on `v2`; conventional commits; one commit per task min.
- Reproducibility: seed sampling + estimator training (T4); record backfill source hashes.
- No `test.skip`/stubs count as done (repo failure-mode guard).
- **GW1 full-scope retained (decision 2026-07-22):** distributional xPts, scenario ILP, MC BPS, top-10k EO all in scope for GW1; the ~3-week deadline risk is accepted knowingly.
