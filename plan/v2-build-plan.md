# FPL 26/27 Bot — v2 Build Plan (Engineering Handoff)

Companion to the design spec (`~/Downloads/fpl-2627-bot-plan.md`). That doc says *what* v2 should be.
This doc grounds it in the **current v1 codebase**: what to keep, what to rip out, what to build, in what order, with acceptance gates. Read the design spec first; this is the execution layer.

Status: proposed. Not yet started. v1 lives on `master` (commit `9e19927`).

---

## 0. TL;DR

v1 is a **mean-points maximiser with a monolithic xPts regressor and a leaky backtest**. The design spec asks for the opposite on all three axes: a *decomposed, distributional* xPts engine, a *rank-aware, variance-seeking* decision layer, and a *point-in-time* data spine that makes any of it trustworthy.

The v1 module skeleton (`data / projection / optimiser / agent / scripts`) is sound and worth keeping. The **models inside it are not** — the points model and the decision objective both need replacing, and the data layer needs a snapshot spine before any backtest number can be believed.

**Do these in order. Do not skip #1.**

1. **Fix backtest leakage (data spine).** Until this is done, every v1 backtest number — including the "~50 pts/GW" baseline — is suspect. See §2.1.
2. **Decompose + distributionalise the xPts engine.** Replace the monolithic regressor. Hit the ≥57 pts/GW naive-baseline gate on *leakage-free* 25/26. See §4.
3. **Rank-aware decision layer.** Add EO, variance, scenario sampling. See §5.
4. **News layer + live ops.** Typed signals, shadow-mode A/B. See §6–7.

---

## 1. Critical finding: backtest leakage (fix before trusting any number)

**Symptom.** `scripts/backtest.py::_load_players_snapshot()` pulls player attributes from the `players` table. `data/models.py::Player` stores `form`, `ict_index`, `influence`, `creativity`, `threat`, `selected_by_percent`, `status`, `chance_of_playing_next_round` as **single mutable columns** (`updated_at ... onupdate=datetime.utcnow`) — overwritten on every ingest, no history.

**Consequence.** `projection/minutes_model.py` and `projection/points_model.py` both use these columns as features (`FEATURE_COLS` includes `form`, `ict_index`, `selected_by_percent`, `is_available`, `cop_next`, `influence`, `creativity`, `threat`). During walk-forward, the model scoring GW`t` sees these **as of the latest refresh**, not as of GW`t-1`. That is look-ahead leakage. It flatters v1's backtest and silently mis-ranks players (form/ICT are among the strongest features).

**What is already point-in-time (safe):** `player_gw_stats` (per-GW `minutes`, `total_points`, `value`, `selected`, `transfers_in/out`, event counts) and `player_xg_stats`. The rolling features built off these (`avg_*_3gw/5gw`, `starts_rate`, decay ratios) are correctly `.shift(1)`-ed and leakage-free. The leak is specifically the **`players.*` current-state columns**.

**Fix (Phase 1, blocking):** every dynamic player attribute must be stored as a **timestamped snapshot row**, and every training/backtest read must filter `snapshot_ts < deadline(gw)`. See §3 for the schema. This is design-spec §3's "single non-negotiable rule" and it is currently violated.

---

## 2. v1 inventory — keep / replace / build

| Component | File(s) | Verdict | Why |
|---|---|---|---|
| Package layout | `data/ projection/ optimiser/ agent/ scripts/` | **Keep** | Clean separation, matches target architecture. |
| FPL API ingest | `data/ingestors/fpl_api.py` | **Keep, extend** | Solid async bootstrap/fixtures/history. Add snapshot-write path (§3). |
| Historical backfill | `scripts/backfill_history.py` | **Keep, extend** | vaastav loader is fine; add per-GW ICT/ownership snapshot backfill where the archive has it. |
| Odds ingest | `data/ingestors/odds_api.py` | **Keep, promote** | Exists but underused. Make odds a *first-class* CS/goals anchor, not a feature (§4). |
| Understat / xG | `data/ingestors/understat.py` | **Keep, extend** | Need event-level (shots, CBI, tackles, saves) for BPS sim + DefCon, not just xG/xA. |
| DB schema | `data/models.py` | **Extend (major)** | Add snapshot tables + event-level + EO + BPS-recompute tables (§3). Keep existing tables. |
| Minutes model | `projection/minutes_model.py` | **Replace** | Binary P(60+) → 3-way distribution {0, 1–59, 60+}. It's the most important model; give it its own feature set + calibration report. |
| Points model | `projection/points_model.py` | **Delete + rebuild** | Monolithic GBT on `total_points` (bakes in old-rules bonus, point estimate only). Design spec §4 explicitly forbids this. Replace with component models (§4). |
| CS model | `projection/cs_model.py` | **Refactor** | Keep as one component; re-anchor on odds-implied CS + Poisson (§4). |
| BPS / bonus | — | **Build (new)** | No BPS simulator exists. Required for 26/27 rules. §4.7. |
| DefCon | — | **Build (new)** | Not modelled at all. §4.5. |
| Squad ILP | `optimiser/squad.py` | **Extend** | Good PuLP bones. Objective is `Σ xpts(start+captain)` — a pure mean maximiser. Add EO / variance / differential terms (§5). |
| Transfer ILP | `optimiser/transfers.py` | **Keep, re-objective** | Genuinely good multi-period model (FT banking, hits, terminal value). Swap the objective for the rank-aware one; feed per-GW distributional xPts. |
| Chips | `optimiser/chips.py` | **Rework** | Threshold heuristics → scenario-based placement (§5). |
| Backtest | `scripts/backtest.py` | **Rebuild** | Fix leakage (§1); score under 26/27 BPS; report **rank distribution**, not mean only; per-component calibration; ablations. §7. |
| Agent / client / notifier | `agent/*` | **Keep** | Orchestration, FPL submission, Telegram all reusable. Upgrade digest to design-spec §9. |
| Deploy | `deploy/*` | **Keep** | systemd timer fine. Adjust cadence for the 26/27 9am-next-day lockdown. |
| News layer | `injury_parser.py`, `press_conference.py` | **Replace** | Regex + Guardian sentiment → typed-signal extraction + deterministic fusion + shadow-mode logging (§6). |

**Config audit (do early, cheap):** `config/strategy.py::ScoringRules` looks non-standard — `points_cs_gk/def = 6` (FPL standard is 4), `points_goal_gk = 10`. Verify every constant against the official 26/27 rules before it silently corrupts the BPS simulator and scoring labels. `SCORING` also has no BPS *formula*, only `points_bps_first/second/third` — the 26/27 BPS rework (§1 of design spec) needs the full BPS point table encoded here.

---

## 3. Data layer redesign (Phase 1 — blocking)

Goal: **every fact the models read is reproducible as-of any past deadline.** Additive migration — do not drop existing tables.

### 3.1 New: player state snapshots
```
player_state_snapshots
  id, player_id, snapshot_ts (UTC, indexed), season, gameweek_context,
  now_cost, status, chance_of_playing_this/next_round,
  selected_by_percent, form, ict_index, influence, creativity, threat,
  news, news_added, transfers_in_event, transfers_out_event
  UNIQUE(player_id, snapshot_ts)
```
Ingest writes a new row per capture (daily + always pre-deadline). Never `UPDATE`. All feature reads become "latest snapshot with `snapshot_ts < deadline`". This single table kills the §1 leak.

### 3.2 New: effective ownership
```
ownership_snapshots
  player_id, snapshot_ts, overall_selected_pct, top10k_selected_pct,
  captaincy_pct_overall, captaincy_pct_top10k
```
Feeds the rank-aware objective (§5). Top-10k EO requires scraping (e.g. an EO provider or aggregating a top-10k mini-league sample) — see open decisions §8.

### 3.3 New: event-level match data (for BPS sim + DefCon)
```
player_match_events
  player_id, fixture_id, season, gameweek,
  shots, shots_in_box, big_chances, key_passes, xg, npxg, xa,
  cbi (clearances+blocks+interceptions), tackles, recoveries,
  saves, saves_in_box, penalties_saved, ... (full BPS input set)
```
Sourced from Understat/FBref/Opta-derived. This is what the BPS simulator consumes.

### 3.4 New: BPS recomputation (26/27 rules over history)
Recompute BPS and bonus for **all historical GWs** under the 26/27 formula, store as leakage-free labels:
```
recomputed_bonus
  player_id, fixture_id, season, gameweek, bps_2627, bonus_2627
```
**Never train the bonus model on historical *awarded* bonus** (old rules). Headline backtest scores use `bonus_2627`; report old-rules score as a sanity check.

### 3.5 Odds as first-class
`fixture_odds` exists; extend to store anytime-goalscorer and over/under lines per player/fixture where available, all with `fetched_at`. Odds are the CS/goals *anchor* in §4, not a garnish feature.

---

## 4. xPts engine (Phase 2) — decomposed + distributional

**Delete `points_model.py`'s monolithic regressor.** Build components; each outputs a **distribution** (mean + variance, ideally samples). Sum via Monte Carlo to get per-player xPts distribution and the **team covariance structure** the risk layer needs.

1. **Minutes (highest priority).** 3-way P({0, 1–59, 60+}). Reuse v1's calibrated-classifier scaffold but as multiclass; features from point-in-time snapshots only + rotation/congestion + pre-season minutes + news override hook (§6). This is the single biggest lever — v1's own diagnosis is that minutes is where it lost.
2. **Goals.** per-90 npxG (shot volume × quality, shrunk finishing) × odds-implied team goals. Anchor on odds.
3. **Assists.** per-90 xA / key passes × team attack context. Calibrate FPL-assist ≠ Opta-assist.
4. **Clean sheet.** odds-implied CS directly; Poisson on odds-implied goals-conceded otherwise. Conditional on 60+ (uses the minutes distribution).
5. **DefCon (new).** per-90 CBI+tackle+recovery rates × opponent possession/game-state → P(threshold). Requires §3.3.
6. **Saves.** saves/shot × opponent shot volume.
7. **Bonus (new).** Monte-Carlo the **26/27 BPS formula** over the component event distributions. Not a regressor. Requires §3.3–3.4.
8. **Cards/other.** static priors.

**Output contract:** `PlayerProjection` gains `xpts_mean`, `xpts_var`, and an optional samples blob (or a `projection_samples` side table), per player per **future** GW — fixture-specific, not the flat constant v1 currently broadcasts across the horizon (`backtest.py::_build_gw_projections` assigns the same xp to every horizon GW — fix this; the horizon optimisation is cosmetic without per-GW fixtures).

**Gate (design spec §4):** naive "best XI each week, no transfer logic" on **leakage-free** 25/26 (26/27 BPS) ≥ **57 pts/GW**, with per-component calibration (predicted vs realised CS rate, goal involvement, minutes-band). If below, iterate here — nothing downstream rescues a weak xPts model.

---

## 5. Decision layer (Phase 3) — rank-aware

Reuse the `optimiser/transfers.py` multi-period ILP; **change the objective and the inputs.**

- **EO as first-class input** (§3.2). Rank value ≈ `your_pts − EO·field_pts`. Benching a 60%-EO player is a short position; the optimiser must see it.
- **Objective v1 (ship first):** `E[pts] + λ·differential_value + μ·variance` with λ/μ by season state. `differential_value` uses EO; `variance` uses §4's `xpts_var` + teammate covariance.
- **Objective v2 (upgrade):** scenario-based stochastic programming — sample scenarios from the xPts distributions, maximise `P(beating an EO-weighted field team)`.
- **Dynamic risk posture:** risk ↑ when behind target rank, ↓ when ahead. Feed live rank percentile.
- **Captaincy** chosen inside the scenario framework (biggest weekly variance lever) — not v1's argmax-xPts.
- **Chips:** scenario-based placement against the fixture calendar; first set before GW19. Rework `chips.py` from thresholds to scenario EV.
- Keep the horizon, FT-banking (max 5), and −4 hit modelling from v1's ILP — they're already correct.

**Gate:** walk-forward vs benchmarks (avg manager ~55–57, frozen template, v1, top-10k ~63). Report **simulated final-rank distribution**, not mean points — a risk-seeking bot can have slightly lower mean with a fatter right tail.

---

## 6. News layer (Phase 4) — typed signals, deterministic fusion

Replace regex/sentiment with the design-spec §5 contract. LLM **reads and scores credibility**; numeric fusion into models is **deterministic code**, never LLM-edited xPts.

Typed signals (all timestamped, source-URL'd, point-in-time): `start_probability_override`, `injury_flag`, `new_signing_prior`, `set_piece_change`, `transfer_rumour`, `departure_risk`, `manager_change`. Primary injection point is the minutes model's start-prob override.

**Shadow-mode A/B from GW1:** log decisions-with-news vs decisions-without; measure counterfactual points delta. Promote the layer only once it demonstrates positive value. Historical replay only where archives are inherently timestamped (FFS archive, press-conf dates, Wayback).

**Exception — `departure_risk` is live from GW1, not shadow.** Its confirmed tier is FPL ground-truth (`status='u'` / element removed), not model inference, and the cost of picking a rumoured leaver into the initial 15 is asymmetric. See §6.5.

---

## 6.5 Transfer windows & departure risk (squad-construction gate)

**Problem.** xPts and the optimiser assume a player stays a PL player for the whole planning horizon. A player sold out of the league (summer window through GW1; January window) scores 0 after departure, and picking a rumoured leaver into the **initial 15** is a high-cost mistake no in-season transfer fully recovers. The §6 news layer *observes* `transfer_rumour`/`new_signing_prior` but is shadow-mode at GW1 — as specified it would not prevent the pick. This section closes that gap.

**Ground truth (free).** Confirmed departures are authoritative in the FPL bootstrap: a departed player drops out of `elements` or flips to `status='u'`. Phase-1 snapshots already capture `status`/`news`, so the spine carries the confirmed signal with **no schema change**. Only *pre-confirmation rumours* need the §6 LLM layer (Ollama over free transfer-news RSS).

**`departure_risk` signal.** A typed §6 signal (point-in-time, source-URL'd) carrying `p_leave ∈ [0,1]` per player, fused deterministically:
- FPL `status='u'` or element removed → `p_leave = 1.0` (confirmed).
- Ollama over free transfer-news RSS → graded `p_leave` for rumours, decayed by source credibility and staleness.

**Graduated handling (resolved 2026-07-22).**
- **Confirmed / near-confirmed** — `p_leave ≥ 0.7`, or FPL `status='u'` / dropped from `elements` → **hard-exclude** from the candidate pool (initial-15 and every transfer step).
- **Rumoured** — `0.2 ≤ p_leave < 0.7` → multiply each remaining horizon GW's xPts by `P(stays) = 1 − p_leave` (mechanically identical to the DGW/BGW multipliers in `DGWStrategy`).
- **`p_leave < 0.2`** → no effect.

**Promotion exception.** The departure gate is **live (acted on) from the initial-15 build**, unlike the general news A/B (§6). The confirmed tier is FPL ground-truth; the rumour tier's calibration still feeds the shadow A/B so its discount can be tuned before being trusted more heavily.

**January window as a re-plan trigger.** Treat the January window (config'd GW range) as a mini-preseason: (a) refresh the candidate pool so **incoming** PL signings enter with `new_signing_prior` (cold-start, no PL history); (b) apply the departure gate to shed/avoid **outgoing** players; (c) let the chip/transfer planner spend accumulated FTs / a wildcard against the refreshed pool.

**Config (Phase 3, `strategy.py::DepartureRiskRules`).** `hard_exclude_p_leave=0.7`, `rumour_floor_p_leave=0.2`, `january_window_start_gw`, `january_window_end_gw`. Season-tunable, hence co-located with the other strategy params.

**Gates.** (i) a confirmed leaver (`status='u'`) never appears in a constructed squad (candidate-filter unit test); (ii) the rumour discount reduces horizon xPts monotonically in `p_leave` (unit test); (iii) the January re-plan admits a synthetic incoming signing into the pool (integration test).

**Phase-1 impact: none.** The spine already carries `status`/`news`; the `departure_risk` signal table and the optimiser gate are Phase 3–4 work. Recorded here so T3/T4 proceed unblocked.

---

## 7. Backtest harness (rebuild alongside Phase 1–2)

- **Point-in-time walk-forward:** at each historical deadline, read only `snapshot_ts < deadline`. Kills §1.
- **Scoring:** 26/27 BPS headline (`recomputed_bonus`), old-rules sanity check.
- **Benchmarks:** avg manager, frozen template, v1 bot, top-10k pace.
- **Report distribution over seeds/scenarios**, and a **simulated final-rank distribution** — not one mean.
- **Ablations:** odds features, BPS sim, DefCon, news layer, risk objective — each toggled independently with per-GW delta.
- Also run 23/24, 24/25 with rule-era caveats.

---

## 8. Phased plan + open decisions

| Phase | Scope | Exit gate |
|---|---|---|
| **1. Data spine** | Snapshot tables (§3.1–3.5); ingest writes snapshots not updates; backfill; 26/27 BPS recompute; leakage-free backtest reads | Re-run v1 backtest leakage-free; record the *honest* v1 baseline (expect it to drop from ~50). |
| **2. xPts engine** | Minutes 3-way → components → BPS sim → distributional output; per-GW fixture projection | ≥57 pts/GW naive baseline + component calibration. |
| **3. Decision layer** | EO ingest; λ/μ objective → scenario objective; captaincy/chips in-framework | Beats template + v1 on rank distribution walk-forward. |
| **4. News + live ops** | Typed-signal pipeline; deterministic fusion; shadow A/B; **live `departure_risk` gate (§6.5)**; digest §9; deadline automation | Shadow-mode positive counterfactual; departure-gate unit/integration tests green; live digest dry-run before GW1. |

**Sequencing note:** Phases 1–3 are the GW1 must-haves (~mid-Aug 2026). News layer goes live in shadow mode at GW1, promoted once it proves out — **except the `departure_risk` gate (§6.5), which is live from the initial-15 build**. The January window is a first-class re-plan trigger (§6.5).

**Open decisions:**
1. **Rewrite strategy:** in-place on a `v2` branch, module-by-module (recommended), vs. parallel `v2/` package. Recommend branch + in-place given the skeleton is being kept. — *OPEN*
2. **Data-source access/budget:** ✅ **RESOLVED (2026-07-22).** Budget ceiling $100; committed spend $0. All sources free:
   - **Historical odds (backtest anchor):** `football-data.co.uk` CSVs (free closing match odds + O/U 2.5). *Not* The Odds API historical endpoint (10 credits/call, paid).
   - **Live odds:** The Odds API **free tier** (500 credits/mo, `h2h,totals`) — existing `data/ingestors/odds_api.py`.
   - **Goals anchor:** derive from FBref **npxG × odds-implied team goals**. No player props (not backfillable; marginal value given penalty-taker detection already exists). Reconsider only if a clear gap shows.
   - **Event data (§3.3, BPS sim + DefCon):** **FBref via `soccerdata`** (free) — shots, CBI, tackles, saves, xG/xA/key passes.
   - **Top-10k EO (§3.2):** FPL API top-10k sampling / LiveFPL scrape (free). No paid EO provider.
   - **FFS subscription:** **skipped.** DIY reproduces rotation (FBref), news (§6 layer), set-piece/penalty roles (already have). Only real gap = pre-season friendly minutes / expert lineups for GW1–3 cold-start, which decays fast. Cheap late add (~$38/yr) if shadow testing shows weak early-GW minutes.
3. **BPS formula source of truth:** ✅ **RESOLVED (2026-07-22).** Confirmed against [official PL BPS article](https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system). Full tables in **Appendix A**. Config bugs found: `points_cs_gk`/`points_cs_def` = 6 must be **4**; DefCon missing entirely; BPS metric weights missing entirely.
4. **LLM for news layer:** ✅ **RESOLVED (2026-07-22).** **Local Ollama** (free). News layer is shadow-mode A/B from GW1 (§6) — measure counterfactual value, swap to Anthropic Haiku 4.5 (~$10–25/season) only if local extraction proves to be the bottleneck.
5. **Transfer-window / departure risk:** ✅ **RESOLVED (2026-07-22).** Graduated gate (§6.5): hard-exclude confirmed/near-confirmed departures (`p_leave ≥ 0.7` or FPL `status='u'`/removed), soft-discount rumours (`0.2 ≤ p_leave < 0.7`) via `P(stays)` xPts multiplier. Live from the initial-15 build; January window is a first-class re-plan trigger. Free data (FPL bootstrap ground-truth + Ollama rumour grading); no Phase-1 schema change.
6. **GW1 scope vs 3-week runway:** ✅ **RESOLVED (2026-07-22).** **Full scope retained** for GW1 (distributional xPts, scenario ILP, Monte-Carlo BPS, top-10k EO). The plan-critic flagged real deadline risk and recommended a cut; user accepted the risk knowingly rather than descope. Revisit only if a phase gate slips.

**Plan-critic pass (2026-07-22).** A separate critic review of both plan docs (verified against code) surfaced a cluster of train/serve-skew + missing-plumbing defects in the *original* T3/T4/T6 — all folded into the revised `phase-1-data-spine.md`: **C1** vaastav-per-GW vs bootstrap-cumulative feature skew (same columns, different quantities); **C2** odds stamped after the deadline → anchor collapses to defaults; **C3** GW1 cold-start undefined (now T7); **M1** `Gameweek` has no `season`; **M2** historical `fixtures`/deadlines never backfilled (now T3a); **M3** cross-season FPL element-id remap needs the stable `Player.code` (now T2.5); **M4** exit gate was record-only (now pass/fail + leak canary). New Phase-1 order: T2.5 → T3a → T3 (+parity test) → T4; T5/T6/T7 parallel. Standing process: critic-on-entry + verifier-on-exit per phase.

---

## Appendix A — 26/27 scoring + BPS (source of truth for §2 config, §4.7 BPS sim)

Confirmed 2026-07-22 against the [official PL BPS article](https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system) and PL scoring basics. Standard scoring is **unchanged** from 25/26; only BPS + DefCon interactions changed.

### A.1 Standard scoring — fixes for `config/strategy.py::ScoringRules`
| Field | Current | Correct 26/27 | Action |
|---|---|---|---|
| `points_cs_gk` | 6 | **4** | 🔴 fix |
| `points_cs_def` | 6 | **4** | 🔴 fix |
| `points_goal_gk` / `_def` / `_mid` / `_fwd` | 10 / 6 / 5 / 4 | same | ok |
| `points_assist` | 3 | 3 | ok |
| `points_cs_mid` | 1 | 1 | ok |
| saves 3→1 · pen_save 5 · pen_miss −2 · conceded 2→−1 · yellow −1 · red −3 · own_goal −2 | — | same | ok |

### A.2 Defensive Contribution (DefCon) — **new fields to add** (missing entirely)
| Position | Threshold | Metric | Points | Cap/match |
|---|---|---|---|---|
| DEF | 10 | CBIT (clear+block+intercept+tackle) | +2 | 2 |
| MID / FWD | 12 | CBIRT (adds recoveries) | +2 | 2 |

*(Note: full-back DefCon threshold was flagged "under review"; treat 10 as current until FPL confirms otherwise.)*

### A.3 26/27 BPS metric table — **new** (code only had the 3/2/1 split)
| Metric | 26/27 BPS | Δ vs 25/26 |
|---|---|---|
| Play 1–60 min | +3 | — |
| Play 60+ min | +6 | — |
| Goal — GK/DEF | +12 | — |
| Goal — MID | +18 | — |
| Goal — FWD | +24 | — |
| Assist | +9 | — |
| Clean sheet — GK/DEF | +12 | — |
| Save (any) | +2 | 🔧 was 3 in-box / 2 out-box; now flat +2 any save |
| Save from inside box | +1 additional | 🔧 out-of-box metric removed |
| Big chance saved | +1 additional | 🆕 new |
| Penalty saved | +7 | 🔧 was 8; nets 8 with big-chance-saved +1 (penalty = big chance) |
| Big chance created | +3 | — |
| Key pass / open-play cross / dribble | +1 each | — |
| Successful tackle | +2 | — |
| CBI | +1 per **3** | 🔧 was per 2 |
| Recoveries | +1 per 3 | — |
| Pass completion 70–79 / 80–89 / 90%+ (30+ passes) | +2 / +4 / +6 | — |
| Winning goal | +3 | — |
| **Being tackled** | **0 (removed)** | 🔧 was −1 |
| Conceding penalty | −3 | — |
| Missing penalty | −6 | — |
| Yellow / red | −3 / −9 | — |
| Own goal | −6 | — |
| Missing big chance | −3 | — |
| Error → goal / → shot | −3 / −1 | — |
| Foul / offside / shot off target | −1 each | — |

**BPS sim modeling notes (§4.7):** (1) penalty-save stacking — encode penalty save as +7 and let big-chance-saved +1 apply (net 8); validate whether the flat +2 save / +1 in-box also stack on penalties against early-season observed BPS. (2) DefCon and BPS now deliberately de-overlap (CBI per-3, tackle penalty removed) — the sim must compute them independently, not share a CBI term.
