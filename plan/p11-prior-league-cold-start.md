# P11 — Promoted-team / new-signing cold-start prior (completion)

**Status:** design approved 2026-08-01, ready for implementation planning.

## Why

`plan/phase-2-xpts-engine.md`'s P11 task is only ~25% built. `data/ingestors/fbref_prior.py`
and the `prior_league_stats` table exist and can ingest a (league, season) of FBref
season-aggregate per-90 stats, but nothing ever reads that table. The 102 brand-new
26/27 codes (foreign signings + promoted-team players with no PL history) still fall
through `projection/cold_start.py`'s generic `peer_bucket_prior`/`position_price_prior`
cascade — the most-mispriced group in the pool gets the least-informed projection. This
finishes the three missing pieces: identity mapping, translation-factor calibration, and
cold-start wiring.

## Current state (verified in code)

- `data/models.py::PriorLeagueStats` — table exists, keyed by
  `(player_name, team, league, season)`, `code` column present but always `NULL` (never
  written by anything).
- `data/ingestors/fbref_prior.py::ingest_prior_league_season` — scrapes one
  (league, season) via `soccerdata.FBref.read_player_season_stats`, writes
  goals90/assists90/npxg90/xa90/minutes/matches. Never populates `code`.
- `scripts/scrape_prior_league.py` — CLI runner for the above.
- `projection/cold_start.py::project_cold_start` — the `else` branch (no PL prior)
  currently tries `_peer_bucket_stats` (pooled real per-appearance points from
  established players in the same position+price bucket), falling back to
  `_price_prior` (synthetic position+price linear formula). Neither uses
  `prior_league_stats` at all.
- `player_gw_stats` has PL seasons 2021-22 through 2025-26 (5 seasons) — enough to build
  4 valid season-transition hold-outs per league (e.g. Championship 2023-24 → PL 2024-25).
- `data/ingestors/fbref.py::_match_player` — an existing, hardened name matcher
  (exact → normalized substring with a 2+ token guard → unique token-subset fallback,
  plus a hand-verified alias table for nickname/transliteration variants). Built and
  scarred by real matching bugs during the 2026-07-28 data-completeness audit. Reused
  here rather than writing new fuzzy-matching logic.
- **Real bug found + fixed 2026-08-01: Serie A ingestion was silently broken, not a
  network/Cloudflare issue.** `~/soccerdata/config/league_dict.json`'s `"ITA-Serie A"`
  entry mapped to FBref's raw label `"Serie A"`, but FBref has since relabelled the
  page `"Serie A (M)"` (disambiguating from a newer women's competition) — the
  already-cached `leagues.html` genuinely contains the Serie A row (confirmed via
  direct grep, 12 hits), but `soccerdata.FBref.read_leagues()`'s name-translation step
  silently dropped it since `"Serie A"` no longer matched anything in the page, while
  Championship/La Liga/Bundesliga/Ligue 1 (whose labels didn't change) kept working.
  Not a soccerdata code bug — a stale local machine config left over from whenever
  Serie A was first registered. **Fixed** by updating the local config's `"FBref"`
  value to `"Serie A (M)"`; verified `read_leagues()` now resolves the league from the
  existing cache with no new network call. This is a machine-local config file (not
  git-tracked) — re-verify the same fix is needed on whatever machine actually runs
  the P11 scrape, since `~/soccerdata/config/league_dict.json` isn't shared via this
  repo. Manual HTML download (the user's fallback offer) should no longer be
  necessary now that the real cause is fixed, but keep it in mind as a fallback if
  Serie A's actual season/player-stats pages turn out to have their own separate
  issues once a live scrape is attempted.

## Scope for this pass

1. Identity mapping: match every `prior_league_stats` row to a `players.code`.
2. Empirical per-league translation factors, calibrated against a real hold-out.
3. Wire the translated prior into `project_cold_start` as a new top-priority tier.
4. Minor: inform `NEW_PLAYER_START_PROB` from prior-league minutes-share instead of a
   flat 0.6 for every new signing.

**Explicitly out of scope** (documented gaps, not oversights):
- European/cup workload-delta cross-referencing between a signing's old and new club —
  affects a small population of high-profile transfers, no clean data source, disproportionate
  effort for this pass.
- Separate promoted-team fixture-difficulty discounting — handled implicitly, since the
  translation factor is calibrated against players' *actual subsequent PL performance*,
  which already nets out average PL fixture difficulty.

## Design

### 1. Prior-league data ingest (calibration + live)

Scrape, via `scripts/scrape_prior_league.py` (already exists, browser-only):
- **Calibration seasons:** each of the 5 leagues (`ENG-Championship`, `ESP-La Liga`,
  `ITA-Serie A`, `GER-Bundesliga`, `FRA-Ligue 1`) × 4 prior-season pairs
  (2021-22, 2022-23, 2023-24, 2024-25) — 20 scrapes.
- **Live season:** each of the 5 leagues × 2025-26 (the season this year's actual new
  26/27 signings just played) — 5 scrapes.
- All season-aggregate (`read_player_season_stats`), same light call the existing
  ingestor already makes — not the heavier 380-match-per-season grind of the live
  match-event scrape.
- **Manual-HTML fallback (kept in reserve):** `soccerdata`'s `.get(url, filepath)`
  returns the cached file's contents unread from the network whenever `filepath`
  already exists (confirmed by reading its source) — so if any league's live scrape
  still fails for a reason that isn't the Serie A config bug above, the fix is to
  manually download the relevant FBref page and drop it at the exact cache path
  `soccerdata` expects (`~/soccerdata/data/FBref/seasons_<league>.html` and
  `players_<league>_<season>_standard.html`), then re-run the ingest — it reads the
  seeded file instead of hitting the network, no code change needed.

### 2. Identity mapping

New function `data/ingestors/fbref_prior.py::backfill_prior_league_codes()`:
- Build a name→`code` map from the current `players` table (same shape as
  `fbref.py::_build_name_map`, but keyed to `code` instead of `id`).
- For every `prior_league_stats` row with `code IS NULL`, run
  `fbref.py::_match_player` against that map; write `code` back on a match.
- Unmatched rows stay `NULL` — those players get no P11 prior and fall through to the
  existing peer-bucket cascade, exactly as today. No fuzzy guess ever silently merges
  two different players' stats (matches `_match_player`'s existing philosophy).
- Re-run this after every prior-league scrape (idempotent — only touches `NULL` rows).

### 3. Translation-factor calibration

New module `projection/prior_league_translation.py`:
- **Hold-out construction:** for each (league, prior_season) pair with `code` populated,
  join to `player_gw_stats` for the immediately-following PL season on that `code`.
  A player qualifies for the hold-out if they have >= `MIN_PRIOR_APPEARANCES`-equivalent
  minutes in BOTH the prior-league season and the following PL season (reuses
  `cold_start.MIN_PRIOR_APPEARANCES`'s existing bar, translated to a minutes threshold
  since prior-league data is season-aggregate, not per-appearance).
- **Factor computation:** one scalar per league =
  `median(realized PL per-90 goal contribution) / median(prior-league per-90 goal
  contribution)` across that league's pooled hold-out (all 4 season-transitions
  together, not one factor per season — maximizes sample size). "Goal contribution"
  = `goals90 + assists90` as the single combined metric the factor is fit on (npxG90/xA90
  move by the same factor at application time — computing 4 independent factors per
  league would fragment an already-small hold-out into unreliably small slices).
  Median over mean: robust to the occasional flop-or-breakout outlier, consistent with
  `cold_start.py`'s existing preference for simple, real-data-driven stats over
  model-fit ones.
- **Sparse-league fallback:** if a league's hold-out has fewer than
  `_MIN_CALIBRATION_SAMPLES` (proposed: 15) qualifying players even after pooling all 4
  season-transitions, fall back to the plan's original literature-style default for that
  league only (Championship 0.65, top-5 leagues 1.0) rather than trusting a factor fit
  on a handful of players — logged clearly so it's visible which leagues got a real vs.
  fallback factor.
- Factors are computed once (a small offline script/cache, not recomputed live every
  run) and stored in a new `config/strategy.py` constant
  (`PRIOR_LEAGUE_TRANSLATION_FACTORS: dict[str, float]`), same pattern as other
  calibrated constants in this file (e.g. `mu_baseline`) — with a comment recording
  when/how it was calibrated, per this project's existing convention.

### 4. Cold-start wiring

In `projection/cold_start.py::project_cold_start`'s `else` branch (player has no PL
prior), insert a new top-priority check ahead of the peer-bucket cascade:

```
if player's code has a matched prior_league_stats row:
    translated_goals90 = row.goals90 * factor[row.league]
    translated_assists90 = row.assists90 * factor[row.league]
    translated_npxg90 = row.npxg90 * factor[row.league]
    translated_xa90 = row.xa90 * factor[row.league]
    xpts = expected GW points from the translated per-90s via the same
           scoring constants (SCORING.points_per_goal etc.) goals.py/assists.py
           already use, scaled to a 90-minute basis
    xpts_var = variance of realized PL per-90 points among that league's hold-out
               (real-data variance, same principle as _peer_bucket_stats)
    proj_source = "prior_league_prior"
else:
    ...existing peer_bucket_prior / position_price_prior cascade, unchanged...
```

`NEW_PLAYER_START_PROB` becomes a function of the player's prior-league minutes-share
(`minutes / (matches * 90)`) when a prior-league match exists, discounted toward the
existing flat 0.6 (a nailed-on Championship starter shouldn't get full PL-starter
confidence immediately) — exact discount curve decided during implementation, not
locked into this spec.

### 5. Testing / gate

- Pure unit tests for the translation-factor calibration math (hold-out construction,
  median-ratio computation, sparse-league fallback) — no live scrape needed, same
  pattern as `tests/test_prior_league.py`'s existing pure-function tests.
- `tests/test_cold_start.py` extended: a player with a matched, translated prior-league
  row gets `proj_source == "prior_league_prior"`; a player with no match still gets the
  existing cascade untouched (regression guard).
- Sanity check (plan's own gate criterion iii): a known strong historical
  promoted-team-to-PL performer (identified from the calibration hold-out itself) ranks
  above a generic squad-filler peer at the same price when both are run through
  `project_cold_start`.
- Full existing suite stays green throughout.

## Open items for the implementation plan (not decided here)

- Exact discount curve for the minutes-share-informed start probability.
- Whether `_MIN_CALIBRATION_SAMPLES = 15` is the right threshold (revisit once the real
  hold-out sizes are known after scraping).
- Whether to persist the calibration hold-out itself (for future re-calibration as more
  PL seasons accumulate) or treat it as a one-off computation whose result is just the
  cached factor.
