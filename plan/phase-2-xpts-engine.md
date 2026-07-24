# Phase 2 — xPts Engine (decomposed + distributional). Granular task plan.

Companion to `v2-build-plan.md` §4 and §7. **Phase 1 is complete** (`plan/phase-1-data-spine.md`): the spine is leakage-free and train/serve-consistent, the honest baseline is recorded, and the 26/27 BPS simulator + recompute pipeline exist. Phase 2 replaces the weak xPts model — nothing downstream (the Phase-3 decision layer) can rescue a bad projection.

Branch: `v2` (in-place).

**Revised 2026-07-23 after the plan-critic pass** (findings C1–C3, M1–M4, verified in code). The original draft repeated the Phase-1 failure mode: several tasks passed their own gate while the *phase* was unsound end-to-end. The decomposition never reconnected to the 26/27 **bonus** (the exit gate's scoring basis needs event-derived `recomputed_bonus`, which is the deferred FBref scrape); "team covariance" cannot emerge from summing independent per-player marginals; P8's bonus MC consumed ~15 event-count distributions no component produced; and the ≥57 gate compared two different harnesses on two different scoring bases. This revision adds the missing plumbing (P-RS re-score, P-COV joint sampling, P-XI harness, P-FIX forward fixtures), reframes odds anchoring (bivariate-Poisson, not the capped heuristic), scopes P8 honestly, and fixes the dependency graph.

**Phase-2 exit gate (the one number that matters):** a **naive best-XI** (precisely defined in P-XI), walk-forward on **25/26**, with **both the predicted and the actual sides scored under 26/27 BPS on the identical harness** (P-RS), ≥ **57 pts/GW**. The 40.2 Phase-1 baseline must be **re-run on this same harness + scoring basis** so "+17" is like-for-like (M2). Pass/fail, plus **per-component calibration** (predicted vs realised: minutes-band, CS rate, goal involvement, assist rate, DefCon rate, bonus). If below 57, iterate inside Phase 2 (open decision 3).

---

## Current model deficiencies (grounded in code — verified 2026-07-23)

| # | Location | Defect | Fix task |
|---|----------|--------|----------|
| D1 | `points_model.py::train` | Monolithic `GradientBoostingRegressor` → single **point** xPts. No variance, no components, no distribution. | P0/P10 |
| D2 | `minutes_model.py` | **Binary** P(start) (`predict_proba[:,1]`), not 3-way P({0, 1–59, 60+}). | P1 |
| D3 | `backtest.py::_build_gw_projections` | Broadcasts the **same xp to every horizon GW** — not fixture-specific. | P0 |
| D4 | `features.py` cumulative ICT + T3 `form` proxy | Volume≠rate; `form` not identical train/serve. | P2 |
| D5 | `pipeline.run_projections` | Point × CS × DGW/BGW multiplier — no MC, no covariance. | P10 |
| D6 | `DGWStrategy` / `Fixture.is_dgw` | GW-level DGW heuristic, not per-team. | P12 |
| D7 | backfill (`status`/`chance_of_playing`/`news` defaulted for history) | Minutes model has no historical availability signal. | P1 (M1 resolution) |

## Plan-critic findings folded in (verified 2026-07-23)

| # | Where | Defect | Resolution |
|---|-------|--------|-----------|
| **C1** | `backtest.py:115 _actual_gw_points` returns old-rules `total_points`; gate claims 26/27 | Exit gate's actual side isn't 26/27-scored, and needs `recomputed_bonus` ← events (deferred scrape). Baseline 40.2 is old-rules → not like-for-like. | **P-RS** (new) + FBref-scrape-first (decision 2) |
| **C2** | Decision 1 / P10 | Summing independent per-player marginal samples ⇒ team covariance ≈ 0, the one structure P10 exists to produce. | **P-COV** (new): shared per-fixture team latent per scenario |
| **C3** | `bps_sim.py:30-102` vs P3–P7 | P8 bonus MC needs ~15 event-count inputs (key passes, big chances, crosses, dribbles, pass-completion tier, …) no component produces. | **P8 rescoped**: reduced-BPS over modelled events + measured bias vs `recomputed_bonus` |
| **M1** | `minutes_model.py:119-120` on defaulted 25/26 `status` | Availability features ~constant in training/gate ⇒ gate can't detect they're broken live. | **P1**: availability via deterministic override (as `pipeline.py:305-311` already does) + synthetic non-'a' injection gate |
| **M2** | gate def | "naive best-XI" under-defined + different harness than the 40.2 baseline. | **P-XI** (new): precise harness def + re-baseline |
| **M3** | `backfill_odds.py:101-106` capped CS heuristic; only 1X2 + O/U2.5 stored | P3/P5 overstate "real odds-implied CS"; no per-team goals market. | **P3/P5 reframed**: bivariate-Poisson from 1X2 supremacy + O/U2.5 → team goals → CS=P(opp=0) |
| **M4** | `features.py:18-62` FDR only from `player_gw_stats` rows | No forward-fixture FDR source for the **live** horizon (works only in backtest via later stat rows). | **P-FIX** (new): forward-fixture FDR from season-aware `fixtures` |

Point-in-time SAFE inputs available: snapshots (T3), real FDR historical (T3b), historical odds (T6), 26/27 BPS sim + recompute (T5a/T5b), cold-start harness (T7).

---

## Event data status (decision 2)

**PL 2025-26 scrape DONE** (2026-07-24, headed Chromium via `scripts/scrape_fbref.py`): `player_match_events` = 11,182 rows across all 380 matches; `recomputed_bonus` populated (11,182). So **P-RS** (exit-gate scoring basis), **P7** (DefCon), **P8** (Bonus) are unblocked. Earlier PL seasons (24/25…) can be scraped the same way if more training history is wanted.

**P11 needs additional scrapes** — ENG-Championship + top-5, 2025-26 season (the incomers' prior league). Same headed-Chromium + cache-resume workflow (Cloudflare re-challenges every few hundred requests; re-run to resume from the on-disk page cache).

---

## Task graph

### P0 — Per-GW fixture-specific projection scaffold + MC output contract  *(unblocks all)*
- Fix `_build_gw_projections` (D3): one projection **per (player, future GW)**, conditioned on that GW's opponent/home.
- Extend `PlayerProjection`: `xpts_mean`, `xpts_var`, + a `projection_samples` side table carrying a **shared `scenario_id`** (so P-COV can correlate players drawn in the same scenario). Fix sample count `N` and a retention/pruning policy (Minor: storage cost). Keep `xpts` as `xpts_mean` alias during migration.
- **Gate:** two horizon GWs with different opponents get different xPts; schema migration idempotent; `projection_samples` round-trips a scenario index.

### P-FIX — Forward-fixture FDR plumbing (live horizon)  *(M4)*
- Build the forward-fixture FDR path from the season-aware `fixtures` table for the **live 2026-27** horizon (the backtest gets future opponents from the player's own later `player_gw_stats` rows; the live path currently has none). Add season filter to `pipeline._get_team_fixture_count` (`pipeline.py:150`).
- **Gate:** a live GW1 projection conditions each horizon GW on the real scheduled opponent, not a default.

### P2 — Rate-feature refactor  *(D4)*  ✅ DONE (`features.CUMULATIVE_BANNED_FEATURES` + guard)
- **Retired** cumulative `ict_index/influence/creativity/threat` + the `form` proxy from both models' `FEATURE_COLS`. They were volume (season-cumulative), and read from *different* sources per path (snapshot as-of in train vs the mutable players row in serve) → train/serve skew. The rolling `avg_*_{n}gw` features already carry the signal as clean per-GW rates, computed identically on both paths (shift(1) as-of, from per-GW data present on both).
- `features.assert_rate_only(FEATURE_COLS)` runs at import in both models — a banned feature now fails fast.
- **ICT-rate deferred (data-layer):** a true ICT-rate needs per-GW ICT or cumulative exposure (minutes/appearances) stored alongside the cumulative ICT; the spine has neither (snapshots store only cumulative ICT; `player_xg_stats` has xG, not ICT). Options later: (a) reconstruct per-GW ICT by differencing consecutive snapshots; (b) add cumulative minutes to the snapshot. Not blocking — the xG rolling rates are the stronger attacking signal anyway.
- **Gate:** `tests/test_p2_rate_features.py` (4) — no banned col in either FEATURE_COLS, guard raises on a banned col, rolling rates still present; both models retrain clean on the live DB. Suite 100/100.

### P1 — Minutes model → 3-way multiclass  ✅ DONE (biggest lever, D2/D7/M1)
- **Delivered:** calibrated **P({0, 1–59, 60+})** (`minutes_band`; multiclass `CalibratedClassifierCV` via `_fit_calibrated`, robust to small/sparse per-GW backtest slices). `predict_minutes_bands` → the full band vector (P5 conditions CS on P(60+); `expected_appearance_points` = 1·P1 + 2·P2 for P10); `predict_batch` kept as the back-compat scalar = P(60+). Learned features are the P2 rolling rates + FDR/odds/enrichment.
- **Availability = deterministic override (M1), not learned:** `is_available`/`cop_next` removed from `FEATURE_COLS` (they're ~constant in backfilled history so unlearnable); `apply_availability_override` sets i/u/s → certain DNP and scales the 'd' (doubtful) playing mass by chance-of-playing, applied to the model's bands at predict time. News-override hook lands in Phase 4.
- **Gate MET:** out-of-sample (train GW<20, test GW≥20 on 25/26) P(60+) reliability is near-diagonal (pred 0.03/0.29/0.50/0.71/0.88 vs actual 0.03/0.31/0.46/0.70/0.88); 3-way log-loss 0.4510 on 14,322 rows. `tests/test_p1_minutes.py` (7) — band bucketing, absent-class handling, the M1 non-'a' injection (i/u/s→DNP, 'd' scaling), expected points, features-removed. Suite 107/107.
- **⚠️ European/cup congestion feature DEFERRED (needs data):** the per-team "in Europe" flag + rolling all-competitions fixture-density would sharpen rotation/fatigue, but the all-comps fixture list (UCL/UEL/UECL + FA/EFL Cup) isn't ingested — same FBref infra can pull it. Deferred as a P1 follow-up; the 3-way + override is the core lever and is complete without it.

### P3 — Goals component  ✅ DONE (shots-based; xG-quality deferred)
- **Team-goals anchor** (`projection/team_goals.py`): `team_goals_from_odds` recovers (λ_home, λ_away) from the T6 de-vigged 1X2 + O/U2.5 via a double-Poisson least-squares fit (scipy). Verified: recovers known λ within 0.06; real 0.84 favourite → λ_home 2.91. `clean_sheet_prob` = exp(−λ_opp) also serves P5.
- **Player distribution** (`projection/goals.py`): `distribute_team_goals` splits team λ among players by attacking **weight × minutes_frac** (anchor-conserving: Σ = λ; the odds carry finishing/quality, the weight allocates *who*). `expected_goal_points` (26/27 SCORING) + `sample_goals` (Poisson) for P10.
- **⚠️ xG-quality DEFERRED — weight is per-90 SHOTS, not npxG.** Real per-match xG proved unavailable from free sources: `player_xg_stats` was empty (Understat never ingested); soccerdata's FBref match API is Performance-only (no xG) and the cached match HTML lacks the JS-comment xG blocks; Understat changed its page-embed format (only a PROMOTION var parses). We DO have real per-GW **shots** (from the FBref summary `Performance Sh`, 5,191 player-GW rows, sensible: Haaland 126 / Fernandes 113). `weight` is a generic input, so swapping in npxG later is a drop-in change. `player_xg_stats.npxg/xg/xa` remain 0 (unpopulated); only `shots` is real.
- **Gate:** `tests/test_team_goals.py` (8) + `tests/test_goals.py` (6) — λ round-trip, conservation, shot-share ordering, benched→0, Poisson mean. Suite 123/123.

### P4 — Assists component  ✅ DONE (`projection/assists.py`)
- Team assists ≈ team λ (P3 anchor) × `ASSIST_FRACTION` (0.75, the FPL-vs-Opta calibration knob), distributed by creativity **weight** × minutes_frac (reuses `goals.distribute_team_goals` — same anchor-conserving split). `expected_assist_points` (×3) + `sample_assists` for P10.
- **⚠️ interim weight = rolling actual assists** — per-90 key-passes/xA aren't in the free feed (same xG wall; both 0 in our data). Swappable to xA when the paid feed lands, no interface change.
- **Gate:** `tests/test_clean_sheets_assists.py` (P4 half) — conservation × assist_fraction, weight ordering, points, sample mean. Suite 130/130.

> **Set-piece takers (raised 2026-07-24).** Penalty/corner/free-kick duty is a big assist+goal driver and *changes pre-season* (a new signing takes over; the prior taker leaves) — prior-season data won't catch 26/27 changes. Sources: penalties are derivable from Understat/FBref shot data (pen taker); corner/FK takers from FBref pass-types (`CK`, dead-ball) — not yet ingested; **confirmed 26/27 changes need the news layer (Phase 4)** or a manual pre-season override. `PlayerSetPieceRole` already exists (currently a crude derivation). Task: a pre-season set-piece-taker snapshot (FBref pass-types + a manual/News override for confirmed changes) feeding P3/P4 weights. Deferred to a P4 follow-up + Phase-4 news.

### P5 — Clean-sheet component  ✅ DONE (`projection/clean_sheets.py`)
- CS = **P(opponent goals = 0) = exp(−λ_opp)** from the P3 odds anchor (`expected_cs_points`) — retires the capped `min(…,0.6)` heuristic. **Conditional on 60+** via P1's P(60+). Plus `expected_concede_points` (GK/DEF −1 per 2 conceded = E[⌊X/2⌋]·−1 over Poisson(λ_opp)) and `sample_clean_sheet_points` (one shared opponent-goals draw → CS bonus needs CS ∧ 60+) for P10.
- **Gate:** `tests/test_clean_sheets_assists.py` (P5 half) — CS prob/points by position+minutes, concede negativity/monotonicity, sample mean == expectation. Not truncated for heavy favourites. Suite 130/130.

### P6 — Saves component (GK)
- saves/shot × opponent shot volume. **Gate:** predicted vs realised GK save points.

### P7 — DefCon component  *(needs events; basis-match note)*
- per-90 CBIRT rates × opponent context → P(threshold met); `player_match_events` (T5b) + `compute_defcon_points` (T5a). **Note:** 25/26 `total_points` already includes 25/26 DefCon; the actual side (P-RS) must recompute DefCon under the same 26/27 `DefConRules` used in prediction, or predicted-vs-actual is a basis mismatch.
- **Gate:** predicted vs realised DefCon-point rate.

### P8 — Bonus component (Monte-Carlo, reduced-BPS)  *(C3; needs events; depends on P1,P3–P7)*
- MC the 26/27 BPS formula (`bps_sim`, T5a) over **the events actually modelled** (appearance from P1, goals P3, assists P4, CS P5, saves P6, CBIRT P7) — a **documented reduced-BPS approximation**, since key-passes/big-chances/crosses/dribbles/pass-completion have no component. Measure the reduced-vs-full bias against `recomputed_bonus` (T5b) and record it.
- **Gate:** predicted vs realised (26/27-recomputed) bonus within the documented tolerance; top-3-per-fixture ranking sane for attacking mids despite the reduced inputs.

### P9 — Cards / other — static priors. Low effort.

### P-RS — 26/27 re-score of 25/26 actuals + calibration basis  *(C1; needs events)*
- Add a backtest scoring mode that re-scores `player_gw_stats` actuals under 26/27 rules: standard scoring (unchanged per Appendix A) + `recomputed_bonus.bonus_2627` + 26/27 DefCon. `_actual_gw_points` (`backtest.py:115`) currently returns old-rules `total_points`.
- **Gate:** actuals re-scored deterministically; the predicted and actual sides share one scoring basis (no definitional bias in the bonus/DefCon calibration).

### P-XI — Naive best-XI harness + re-baseline  *(M2)*
- Precisely define the gate harness: **fixed initial 15** (T7 cold-start), optimise the legal starting XI + captain (armband = argmax component mean; auto-subs by bench order) each GW, **GW6–38** (match the baseline window), DGW handled per P12. No transfers, no chips.
- **Re-run the Phase-1 40.2 baseline on this identical harness + 26/27 scoring** so the comparison is like-for-like.
- **Gate:** harness reproducible (seeded); baseline number reproduced within noise.

### P-COV — Joint sampling + team covariance  *(C2)*
- Per fixture, draw **shared team latents once per scenario** (team goals-for/against, CS indicator); condition each player's goals/assists/CS/minutes draws on the shared latent + common `scenario_id`. Shrinkage on the covariance estimate (~33 GW/season is noisy).
- **Gate:** corr(CS points) between two same-team defenders > 0.5 in the sampled output (a summed-marginal model gives ≈0).

### P10 — Monte-Carlo assembly + exit gate  *(D1/D5)*
- Sum component samples via P-COV joint scheme → per-player xPts distribution (`xpts_mean`, `xpts_var`, samples) + team covariance for Phase 3. Delete the monolithic regressor.
- **Gate:** the **≥57 pts/GW** exit gate runs here on the P-XI harness with P-RS scoring; MC seeded.

### P11 — Promoted-team / new-signing prior model  *(Phase-1 T7 deferred; the biggest alpha source)*
The 102 brand-new 26/27 codes (foreign signings + promoted-team players) have **no PL history**, so T7's cold-start collapses them to a weak position+price prior — i.e. the most-mispriced group gets the worst projection. P11 gives them a real prior from their **prior-league** 2025-26 stats.

- **Prior-league ingest (FBref, same infra as T5b):** scope = **ENG-Championship + top-5** (La Liga, Serie A, Bundesliga, Ligue 1). Top-5 are in soccerdata's default league dict; Championship needs a one-line `~/soccerdata/config/league_dict.json` entry. Season = 2025-26 (the just-finished season the incomers played). Same headed-Chromium + cache-resume workflow (`scripts/scrape_fbref.py`, generalise it to take a league). Store into `player_match_events`/an analogous prior-league stats table keyed by `code`/name.
- **Cross-league translation factors** (the hard part — where alpha vs noise is decided): per-league attacking-output scalars mapping prior-league per-90 (npxG, xA, minutes) to PL-equivalent (Championship→PL ≈ 0.6–0.7, top-5 higher/nearer 1.0). Config'd in `strategy.py` (season-tunable), calibrated against players who actually made the jump (players with both a prior-league season and a subsequent PL season in our data — a natural hold-out).
- **Identity mapping:** fuzzy-match prior-league FBref names → the 26/27 FPL entries (the 102 new codes), reusing/extending the fbref adapter's name matcher; hand-alias the ambiguous ones.
- **Promoted-team fixture context:** promoted clubs' PL fixtures are harder than their Championship ones — the prior must not over-extrapolate Championship output into PL returns (the translation factor handles magnitude; FDR handles opponent).
- **European/cup workload delta caveat:** a signing's prior per-90 was accumulated while carrying their old club's European/cup load; their new PL club's load may differ sharply (leaving a UCL side for a non-European PL club → fresher, more minutes, and vice-versa). Cross-reference the P1 European-participation flag for both the source and destination club so the minutes prior isn't naively carried over.
- Feeds the T7 initial-15 harness (GW1) and the January-incomer re-plan.

- **Gate:** (i) every one of the 102 new players gets a translated prior-league projection, not the flat position/price fallback; (ii) translation factors calibrated on the made-the-jump hold-out (predicted vs realised first-PL-season output, by source league); (iii) a known promoted-team standout from a prior season is ranked sensibly above a squad filler (sanity check).

### P12 — Per-team DGW/BGW  *(D6)*
- Per-team fixture counts per GW; `DGWStrategy` multipliers per-team.
- **Gate:** a synthetic per-team DGW doubles only the doubling team's players.

---

## Dependency order (revised)
```
FBref scrape (browser env) ─────────────► P-RS ─┐  P7 ─┐
                                                 │      │
P0 ─ P2 ─┬─ P1 ──────────────────────────┐      │      │
         ├─ P3 ─ P4                        │      ├─ P8 ─┤
         ├─ P3 ─ P5 (needs P1 minutes)     ├─ P-COV ─ P10  ► EXIT GATE (≥57, P-XI harness, P-RS scoring)
         ├─ P6                             │                  (+ P11, P12 fold in)
         └─ P9 ─────────────────────────── ┘
P0 ─ P-FIX (live horizon)      P-XI (harness) ─ re-baseline ─► gate
```

## Deferred beyond Phase 2 (→ Phase 3/4)
- Decision layer: EO-aware objective, scenario ILP, dynamic risk posture, captaincy, chips (§5, Phase 3).
- News layer: typed signals, start-prob override (P1 leaves the hook), departure-risk gate live (§6/§6.5, Phase 4).
- Price-change model/data feed for `use_price_change_signals` (undefined; Phase 3).
- Postponement/reschedule handling (fixtures moving between GWs).
- Full event-count distribution sub-models (key passes, big chances, crosses, dribbles, pass-completion) that would let P8 become full-BPS rather than reduced.

## Decisions (2026-07-23)
1. **Distributional representation = Monte-Carlo samples** (unblocks Phase-3 scenario optimiser + P8; seeded).
2. **Event data = FBref scrape first** (browser env) before P-RS/P7/P8; interim reduced-scoring fallback only if no browser env.

## Open decision (revisit during execution)
3. **Gate realism** — ≥57 is +17 over baseline. If components plateau below it, decide then whether to reduce the bar or extend scope (minutes/odds anchoring) before Phase 3.

## Cross-cutting
- Every task on `v2`; conventional commits; one commit per task min.
- Reproducibility: seed MC sampling + estimator training; record data-source hashes. Run gates against `fpl_bot_v2.db`.
- No `test.skip`/stubs count as done (repo failure-mode guard).
- Real gates are `pytest` + `ruff` (mypy configured, not enforced).
