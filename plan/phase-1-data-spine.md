# Phase 1 — Data Spine (leakage fix). Granular task plan.

Companion to `v2-build-plan.md` §1, §3, §7 and Appendix A. **Blocking phase — nothing downstream is trustworthy until this lands.**
Branch: `v2` (in-place, per decision #1). No code merged to `master` until the exit gate passes.

**Phase-1 exit gate (the one number that matters):** re-run the v1 backtest leakage-free on 25/26 and record the *honest* baseline (expect it to drop from ~50). Tasks 0–4 deliver this. Tasks 5–6 unblock Phase 2 and can run in parallel after Task 4.

---

## Leak inventory (grounded in current code — this is what Phase 1 must kill)

| # | Location | Leaked columns | Fix task |
|---|---|---|---|
| L1 | `scripts/backtest.py::_load_players_snapshot` (L62–84) | `players.status, form, ict_index, influence, creativity, threat, selected_by_percent, chance_of_playing_next_round` read at inference as *latest*, not as-of GW | T4 |
| L2 | `scripts/backtest.py::_load_all_stats` (L42–44) | same `players.*` columns joined into **training** rows | T4 |
| L3 | `projection/features.py::load_player_enrichment` (L127–129) | current `transfers_in/out_event, selected_by_percent, injury_severity` broadcast onto all historical rows | T4 |
| L4 | `data/models.py::FixtureOdds` `UNIQUE(fixture_id)` + `features.py::load_fixture_odds` | one mutable odds row per fixture; no as-of read | T6 |

Point-in-time SAFE (do not touch): `player_gw_stats`, `player_xg_stats`, and the rolling `.shift(1)` features built off them.

---

## T0 — Branch + config truth (quick, fully specified by Appendix A)

**Scope**
- `git checkout -b v2`.
- `config/strategy.py::ScoringRules`: `points_cs_gk` 6→**4**, `points_cs_def` 6→**4** (rest verified correct).
- Add `@dataclass(frozen=True) DefConRules`: `def_threshold=10` (CBIT), `mid_fwd_threshold=12` (CBIRT), `points=2`, `cap_per_match=2`.
- Add `@dataclass(frozen=True) BPSWeights` encoding Appendix A.3 (all metric weights + the 26/27 deltas: CBI per-3, tackled removed, save +2/+1 in-box/+1 big-chance, penalty-save +7). Include a `# UNVERIFIED-STACKING` comment on penalty-save per Appendix A note.
- Export singletons `DEFCON`, `BPS_WEIGHTS`.

**Acceptance gate**
- `python -c "from config.strategy import SCORING,DEFCON,BPS_WEIGHTS; assert SCORING.points_cs_def==4"` passes.
- New unit test `tests/test_scoring_rules.py`: reconstruct total FPL points for 3 known 25/26 player-GW lines (a CS defender, a returning mid, a hauling fwd) from `player_gw_stats` component columns using `ScoringRules` → matches recorded `total_points ± bonus`.
- `ruff check config/` clean.

---

## T1 — Snapshot schema (§3.1) — additive migration

**Scope**
- `data/models.py`: add `PlayerStateSnapshot` — `id, player_id (FK), snapshot_ts (UTC, indexed), season, gameweek_context, now_cost, status, chance_of_playing_this/next_round, selected_by_percent, form, ict_index, influence, creativity, threat, news, news_added, transfers_in_event, transfers_out_event`, `UNIQUE(player_id, snapshot_ts)`.
- `data/db.py::init_db` creates it; **no existing table altered/dropped**.

**Acceptance gate**
- `init_db()` on a copy of `fpl_bot.db` adds exactly one table; `sqlite_master` diff shows no change to existing tables.
- Re-running `init_db()` is idempotent (no error, no dupe).

---

## T2 — Snapshot-WRITE ingest path (§3) — never UPDATE

**Scope**
- `data/ingestors/fpl_api.py`: add `write_player_snapshots(bootstrap, snapshot_ts)` that INSERTs one `PlayerStateSnapshot` per element (append-only; `on_conflict_do_nothing` on the unique key). Keep `upsert_players` as-is for current-state convenience, but the snapshot table becomes the source of truth for features.
- Call it from `run_full_ingest` (and the live scheduler pre-deadline).

**Acceptance gate**
- Run ingest twice with different `snapshot_ts` → 2 snapshot rows/player; `players` still 1 row/player.
- Log line reports snapshot rows written; `SELECT COUNT(DISTINCT snapshot_ts)` == number of captures.

---

## T3 — Historical snapshot backfill (§3, from vaastav)

**Scope**
- `scripts/backfill_history.py`: extend `_ingest_dataframe` to also emit `PlayerStateSnapshot` rows from `merged_gw.csv` per-GW columns (`value, selected, transfers_in/out, ict_index, influence, creativity, threat`; `form` if present). `snapshot_ts = fixture kickoff − ε` for that GW (best available proxy for "as-of deadline"); `gameweek_context = GW`.
- Seasons 21/22–24/25 (existing list) + 25/26 when available.

**Acceptance gate**
- Snapshot coverage ≥ 90% of player-GW rows that exist in `player_gw_stats` for each backfilled season.
- Spot-check: 3 named players' GW `value`/`ict_index` in snapshot == vaastav CSV value for that GW.

---

## T4 — Leakage-free reads (§7) — THE fix; delivers the honest baseline

**Scope**
- Rewrite `backtest.py::_load_players_snapshot` and `_load_all_stats` to source the dynamic `p.*` feature columns from `player_state_snapshots`, selecting the latest row with `snapshot_ts < deadline(gw)` (deadline from `gameweeks.deadline_time`). `player_gw_stats`/`xg_stats` reads unchanged.
- Fix `features.py::load_player_enrichment` (L3): either read `transfers_in/out_event, selected_by_percent` from the as-of snapshot, or drop them from the historical training path. Set-piece roles keyed by season are fine.
- Add a **guard test** asserting no leaked-column read survives.

**Acceptance gate**
- Grep gate (CI): no `p.form|p.ict_index|p.influence|p.creativity|p.threat|p.selected_by_percent|p.status|p.chance_of_playing` in any read used by training/backtest. `tests/test_no_leakage.py` enforces it.
- `python scripts/backtest.py --season 2025-26 --start-gw 6 --end-gw 38` runs end-to-end.
- **Record the honest v1 baseline** (pts/GW) in `results/backtest_2526_v1_leakfree.csv` + a one-line note in this file. Expect < the old ~50.

---

## T5 — Event data + 26/27 BPS recompute (§3.3–3.4) — unblocks Phase 2

*Larger; can start once T1 lands. Not required for the T4 honest-baseline gate but required before Phase 2's BPS sim.*

**Scope**
- `data/ingestors/fbref.py` (new) via `soccerdata` → `player_match_events` table (§3.3: shots, shots_in_box, CBI, tackles, recoveries, saves, saves_in_box, key_passes, npxg, xa …). Rate-limit-respecting, cached.
- `projection/bps_sim.py` (new): deterministic 26/27 BPS calc from `BPS_WEIGHTS` (Appendix A.3) over event rows → per-player-fixture BPS; rank top-3 → bonus.
- `recomputed_bonus` table (§3.4): `bps_2627, bonus_2627` for all historical GWs.

**Acceptance gate**
- BPS sim reproduces **old-rules** awarded bonus within tolerance when fed old-rules weights (sanity harness), proving the engine before switching to 26/27 weights.
- DefCon and BPS computed independently (no shared CBI term) — asserted by test.
- `recomputed_bonus` coverage ≥ 95% of finished fixtures with event data.

---

## T6 — Odds history as-of (§3.5) — kills L4

**Scope**
- `data/models.py::FixtureOdds`: `UNIQUE(fixture_id)` → `UNIQUE(fixture_id, fetched_at)` (append-only). `features.py::load_fixture_odds` reads latest `fetched_at < deadline`.
- Backfill historical closing odds (match 1X2 + O/U 2.5) from **football-data.co.uk** CSVs → `fixture_odds` rows with `fetched_at = kickoff − ε`.

**Acceptance gate**
- ≥ 95% of historical fixtures have an odds row; `load_fixture_odds` returns as-of values (test on a fixture with 2 snapshots).
- Live `odds_api.py` writes append-only rows (no more single-row overwrite).

---

## Dependency order
```
T0 ─┬─ T1 ─┬─ T2 ─┐
    │      ├─ T3 ─┼─ T4  ► EXIT GATE (honest baseline)
    │      └─ T5 (parallel, gates Phase 2)
    └─ T6 (parallel)
```

## Cross-cutting
- Every task on `v2`; conventional commits; one commit per task min.
- Reproducibility: seed any sampling; record snapshot backfill source hashes.
- No `test.skip`/stubs count as done (per repo failure-mode guard).
```
