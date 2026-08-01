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

## Open items — resolved during implementation planning

- **Start-probability discount curve:** a flat 50/50 blend —
  `start_prob = 0.5 * NEW_PLAYER_START_PROB + 0.5 * prior_minutes_share` — where
  `prior_minutes_share = min(1.0, minutes / (matches * 90))`. Deliberately simple and
  not itself backtested; a reasonable starting point, tunable later without touching
  any other part of the design.
- **`MIN_CALIBRATION_SAMPLES = 15`:** kept as specified. Revisit only if the real
  scrape shows a league's hold-out landing just under 15 — not a blocker for writing
  the code now.
- **Hold-out persistence:** NOT persisted as a table. `build_holdout()` is a pure
  function computed fresh from `prior_league_stats` + `player_gw_stats` each time
  `scripts/calibrate_prior_league_factors.py` runs — cheap to recompute (season-
  aggregate rows, not per-match), and re-running gets more accurate for free as more
  PL seasons accumulate in future years. Matches this session's `mu_baseline`
  calibration precedent (`scripts/calibrate_risk_constants.py`): compute once
  offline, hand-copy the result into `config/strategy.py`.
- **xpts basis for the prior-league tier (not pinned down in the design):** built
  from translated **npxG90/xA90** (the smoother, luck-adjusted quality metrics), not
  raw goals90/assists90 — one prior season's raw counting stats is a small,
  high-variance sample; npxG/xA is the standard way to avoid overfitting to a
  hot/cold streak. The calibration factor itself is still fit against **realized raw
  goal+assist output** (the actual ground truth being predicted) — only the
  application-time xpts calculation uses the smoother inputs.

---

# P11 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish P11 so the 102 brand-new 26/27 codes get a real, data-informed
cold-start projection instead of falling straight to the generic position/price
fallback.

**Architecture:** Five additive pieces, each independently testable: (1) an identity-
mapping backfill reusing the existing `fbref.py` name matcher, (2) a new
`PriorLeagueRules` config block holding per-league translation factors/variances,
(3) a pure calibration module that builds a real historical hold-out and computes
those factors, (4) an offline CLI script that runs the calibration and prints the
config values to hand-copy in, and (5) a new top-priority tier inside
`project_cold_start`'s existing no-PL-history branch that consumes it. Nothing here
touches the `prior_season`/`peer_bucket_prior`/`position_price_prior` branches —
purely additive, verified via existing regression tests staying green.

**Tech Stack:** Python, pandas, SQLAlchemy Core (`text()` queries, matching this
codebase's existing convention in `cold_start.py`), pytest, SQLite (temp DB fixtures
via `sqlalchemy.create_engine("sqlite:///...")`, same pattern as `test_cold_start.py`).

## Global Constraints

- Every new/changed function must degrade gracefully to today's existing behavior
  when its new optional input is missing (`None`/empty) — never crash, never a
  silent 0.0/undefined value (this file's own existing contract, `cold_start.py`'s
  module docstring).
- No fuzzy/guessed identity matches: reuse `data/ingestors/fbref.py::_match_player`
  exactly as-is (unique-match-or-nothing philosophy) — do not write a new, looser
  matcher.
- Reuse `cold_start.MIN_PRIOR_APPEARANCES` (translated to a minutes threshold) rather
  than inventing a second, inconsistent "enough data" bar.
- Full existing test suite (448 tests as of this session) and `ruff check .` must
  stay green after every task.
- All new DB access goes through `data.db.get_session()`, called unqualified inside
  each module (not `data.db.get_session()`) so tests can `monkeypatch.setattr` it,
  matching every existing ingestor/cold_start test fixture in this repo.

---

### Task 1: Identity mapping — `backfill_prior_league_codes()`

**Files:**
- Modify: `data/ingestors/fbref_prior.py`
- Modify: `scripts/scrape_prior_league.py`
- Test: `tests/test_prior_league.py`

**Interfaces:**
- Produces: `data.ingestors.fbref_prior.backfill_prior_league_codes() -> int` (rows
  newly matched). Consumed by Task 4's calibration data (needs `code` populated) and
  Task 5's cold-start wiring (looks up by `code`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prior_league.py` (it currently has no DB fixture — add one, same
pattern as `tests/test_cold_start.py`'s `temp_session`):

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Player, PriorLeagueStats


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'prior.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(fp, "get_session", lambda: Local())
    return Local


def test_backfill_prior_league_codes_matches_established_and_leaves_unmatched_null(
    temp_session,
):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=42, first_name="Prolific", second_name="Striker",
                     web_name="Striker", team_id=1, position="FWD", now_cost=6.5))
        s.add(PriorLeagueStats(
            player_name="Prolific Striker", team="Leeds", league="ENG-Championship",
            season="2025-2026", position="FW", minutes=3000, matches=34,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.add(PriorLeagueStats(
            player_name="Nobody Matches This Name", team="Leeds",
            league="ENG-Championship", season="2025-2026", position="FW",
            minutes=1000, matches=15, goals90=0.1, assists90=0.0, npxg90=0.1, xa90=0.0,
        ))
        s.commit()
    finally:
        s.close()

    matched = fp.backfill_prior_league_codes()
    assert matched == 1

    s = temp_session()
    try:
        rows = {r.player_name: r.code for r in s.query(PriorLeagueStats).all()}
    finally:
        s.close()
    assert rows["Prolific Striker"] == 42
    assert rows["Nobody Matches This Name"] is None


def test_backfill_prior_league_codes_is_idempotent(temp_session):
    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=42, first_name="Prolific", second_name="Striker",
                     web_name="Striker", team_id=1, position="FWD", now_cost=6.5))
        s.add(PriorLeagueStats(
            player_name="Prolific Striker", team="Leeds", league="ENG-Championship",
            season="2025-2026", code=42, position="FW", minutes=3000, matches=34,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.commit()
    finally:
        s.close()

    matched = fp.backfill_prior_league_codes()
    assert matched == 0  # already has a code -- nothing left to backfill
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_prior_league.py -v`
Expected: the two new tests FAIL with `AttributeError: module 'data.ingestors.fbref_prior' has no attribute 'backfill_prior_league_codes'`

- [ ] **Step 3: Implement `backfill_prior_league_codes()`**

In `data/ingestors/fbref_prior.py`, change the top-level import line

```python
from data.models import PriorLeagueStats
```

to

```python
from data.models import Player, PriorLeagueStats
```

then add, after `row_to_prior_stats` and before `ingest_prior_league_season`:

```python
def backfill_prior_league_codes() -> int:
    """Match every code-less prior_league_stats row to a players.code via the
    existing, hardened fbref.py name matcher (exact -> normalized substring
    -> unique token-subset, plus its hand-verified alias table) rather than
    writing new fuzzy-matching logic. Idempotent -- only ever touches rows
    where code IS NULL, so it's safe to call after every scrape. Returns the
    number of rows newly matched this call."""
    from data.ingestors.fbref import _match_player, _normalize_name

    db = get_session()
    try:
        name_map: dict[str, int] = {}
        for p in db.query(Player).filter(Player.code.isnot(None)).all():
            name_map[_normalize_name(f"{p.first_name} {p.second_name}")] = p.code
            name_map[_normalize_name(p.web_name)] = p.code

        unmatched = db.query(PriorLeagueStats).filter(PriorLeagueStats.code.is_(None)).all()
        matched = 0
        for row in unmatched:
            code = _match_player(row.player_name, name_map)
            if code is not None:
                row.code = code
                matched += 1
        db.commit()
        return matched
    finally:
        db.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_prior_league.py -v`
Expected: all 6 tests PASS (4 existing + 2 new)

- [ ] **Step 5: Wire the backfill into the scrape script**

In `scripts/scrape_prior_league.py`, change

```python
from data.ingestors.fbref_prior import PRIOR_LEAGUES, ingest_prior_league_season
```

to

```python
from data.ingestors.fbref_prior import (
    PRIOR_LEAGUES,
    backfill_prior_league_codes,
    ingest_prior_league_season,
)
```

and in `main()`, after the `for lg in leagues:` loop ends (right before
`logger.info("Prior-league scrape complete")`), add:

```python
    matched = backfill_prior_league_codes()
    logger.info("Identity mapping: %d prior_league_stats rows matched to a players.code", matched)
```

- [ ] **Step 6: Run the full suite + lint**

Run: `uv run python -m pytest -q && uv run ruff check .`
Expected: all tests pass, `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add data/ingestors/fbref_prior.py scripts/scrape_prior_league.py tests/test_prior_league.py
git commit -m "feat(p11): identity-map prior_league_stats rows to players.code

Reuses fbref.py's existing hardened name matcher rather than writing new
fuzzy-matching logic. Idempotent (only touches code IS NULL rows), wired
into scrape_prior_league.py so it runs automatically after every scrape."
```

---

### Task 2: `PriorLeagueRules` config

**Files:**
- Modify: `config/strategy.py`
- Test: `tests/test_prior_league_translation.py` (new file, also used by Task 3)

**Interfaces:**
- Produces: `config.strategy.PRIOR_LEAGUE: PriorLeagueRules` singleton, with
  `.translation_factor(league: str) -> float` and
  `.translation_variance(league: str) -> float`. Consumed by Task 3's calibration
  script (prints values to paste in here) and Task 5's cold-start wiring (reads it
  live).

- [ ] **Step 1: Write the failing test**

Create `tests/test_prior_league_translation.py`:

```python
"""P11 — cross-league translation-factor calibration (config lookup + pure
hold-out/factor-computation math). No live scrape needed."""

from __future__ import annotations

from config.strategy import PRIOR_LEAGUE


def test_prior_league_rules_covers_all_five_leagues():
    leagues = ["ENG-Championship", "ESP-La Liga", "ITA-Serie A",
               "GER-Bundesliga", "FRA-Ligue 1"]
    for league in leagues:
        assert PRIOR_LEAGUE.translation_factor(league) > 0
        assert PRIOR_LEAGUE.translation_variance(league) > 0


def test_championship_factor_discounted_below_top5():
    # the plan's own literature-style prior: Championship output doesn't
    # translate 1:1 to the PL, top-5 leagues roughly do.
    assert PRIOR_LEAGUE.translation_factor("ENG-Championship") < 1.0
    assert PRIOR_LEAGUE.translation_factor("ESP-La Liga") == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_prior_league_translation.py -v`
Expected: FAIL with `ImportError: cannot import name 'PRIOR_LEAGUE' from 'config.strategy'`

- [ ] **Step 3: Add `PriorLeagueRules` to `config/strategy.py`**

Add this new dataclass right after `DepartureRiskRules` (before the block of
singleton instantiations at the bottom of the file):

```python
@dataclass(frozen=True)
class PriorLeagueRules:
    """P11: cross-league translation factors + variance for the cold-start
    prior-league prior tier (plan/p11-prior-league-cold-start.md). Defaults
    are the plan's literature-style guess (Championship discounted, top-5
    treated as roughly PL-equivalent) -- replace with
    scripts/calibrate_prior_league_factors.py's real output once the
    historical hold-out has actually been scraped (needs a browser)."""

    translation_factor_championship: float = 0.65
    translation_factor_la_liga: float = 1.0
    translation_factor_serie_a: float = 1.0
    translation_factor_bundesliga: float = 1.0
    translation_factor_ligue_1: float = 1.0

    # Deliberately unremarkable variance guess (mirrors cold_start.py's own
    # _FALLBACK_VAR reasoning) until a real hold-out replaces it.
    translation_variance_championship: float = 6.0
    translation_variance_la_liga: float = 6.0
    translation_variance_serie_a: float = 6.0
    translation_variance_bundesliga: float = 6.0
    translation_variance_ligue_1: float = 6.0

    def translation_factor(self, league: str) -> float:
        return {
            "ENG-Championship": self.translation_factor_championship,
            "ESP-La Liga": self.translation_factor_la_liga,
            "ITA-Serie A": self.translation_factor_serie_a,
            "GER-Bundesliga": self.translation_factor_bundesliga,
            "FRA-Ligue 1": self.translation_factor_ligue_1,
        }[league]

    def translation_variance(self, league: str) -> float:
        return {
            "ENG-Championship": self.translation_variance_championship,
            "ESP-La Liga": self.translation_variance_la_liga,
            "ITA-Serie A": self.translation_variance_serie_a,
            "GER-Bundesliga": self.translation_variance_bundesliga,
            "FRA-Ligue 1": self.translation_variance_ligue_1,
        }[league]
```

Then add `PRIOR_LEAGUE = PriorLeagueRules()` to the singleton block at the bottom,
alongside `DEPARTURE_RISK = DepartureRiskRules()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_prior_league_translation.py -v`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add config/strategy.py tests/test_prior_league_translation.py
git commit -m "feat(p11): add PriorLeagueRules config (literature-default translation factors)"
```

---

### Task 3: Translation-factor calibration module

**Files:**
- Create: `projection/prior_league_translation.py`
- Test: `tests/test_prior_league_translation.py` (extend from Task 2)

**Interfaces:**
- Consumes: `projection.cold_start.MIN_PRIOR_APPEARANCES` (int),
  `data.ingestors.fbref.SEASON_MAP` (dict, our-season-format -> soccerdata-format),
  `data.db.get_session`.
- Produces: `build_holdout(league: str) -> pd.DataFrame` (columns: `code`,
  `prior_goals90`, `prior_assists90`, `realized_goals90`, `realized_assists90`,
  `realized_points90`); `compute_league_stats(holdout: pd.DataFrame) -> tuple[float | None, float | None, int]`
  (factor, variance, sample size — `(None, None, n)` when `n < MIN_CALIBRATION_SAMPLES`).
  Consumed by Task 4's calibration script.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prior_league_translation.py`:

```python
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, Player, PlayerGameweekStats, PriorLeagueStats
from projection import prior_league_translation as plt


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'plt.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(plt, "get_session", lambda: Local())
    return Local


def _seed_holdout_player(Local, *, code, fpl_id, prior_goals90, prior_assists90,
                          prior_minutes, pl_goals, pl_assists, pl_points, pl_minutes,
                          prior_season, pl_season):
    s = Local()
    try:
        s.add(Player(fpl_id=fpl_id, code=code, first_name="P", second_name=str(fpl_id),
                     web_name=f"p{fpl_id}", team_id=1, position="FWD", now_cost=5.0))
        s.commit()
        s.add(PriorLeagueStats(
            player_name=f"p{fpl_id}", team="Leeds", league="ENG-Championship",
            season=prior_season, code=code, position="FW",
            minutes=prior_minutes, matches=prior_minutes // 90,
            goals90=prior_goals90, assists90=prior_assists90, npxg90=prior_goals90,
            xa90=prior_assists90,
        ))
        pid = s.query(Player.id).filter_by(fpl_id=fpl_id).scalar()
        s.add(PlayerGameweekStats(
            player_id=pid, gameweek=1, season=pl_season,
            minutes=pl_minutes, goals_scored=pl_goals, assists=pl_assists,
            total_points=pl_points,
        ))
        s.commit()
    finally:
        s.close()


def test_build_holdout_pools_across_season_transitions(temp_session):
    # one qualifying player in each of two different season-transitions
    _seed_holdout_player(
        temp_session, code=1, fpl_id=1, prior_goals90=0.5, prior_assists90=0.1,
        prior_minutes=1000, pl_goals=9, pl_assists=2, pl_points=100, pl_minutes=1000,
        prior_season="2021-2022", pl_season="2022-23",
    )
    _seed_holdout_player(
        temp_session, code=2, fpl_id=2, prior_goals90=0.4, prior_assists90=0.2,
        prior_minutes=900, pl_goals=5, pl_assists=1, pl_points=60, pl_minutes=900,
        prior_season="2023-2024", pl_season="2024-25",
    )
    holdout = plt.build_holdout("ENG-Championship")
    assert len(holdout) == 2
    assert set(holdout["code"]) == {1, 2}
    row1 = holdout[holdout["code"] == 1].iloc[0]
    assert row1["realized_goals90"] == pytest.approx(9 / 1000 * 90)
    assert row1["realized_points90"] == pytest.approx(100 / 1000 * 90)


def test_build_holdout_excludes_players_below_the_minutes_bar(temp_session):
    # below MIN_HOLDOUT_MINUTES on the PL side -- must not count
    _seed_holdout_player(
        temp_session, code=1, fpl_id=1, prior_goals90=0.5, prior_assists90=0.1,
        prior_minutes=1000, pl_goals=1, pl_assists=0, pl_points=5, pl_minutes=50,
        prior_season="2021-2022", pl_season="2022-23",
    )
    holdout = plt.build_holdout("ENG-Championship")
    assert holdout.empty


def test_build_holdout_empty_league_returns_empty_frame(temp_session):
    holdout = plt.build_holdout("GER-Bundesliga")
    assert holdout.empty
    assert list(holdout.columns) == [
        "code", "prior_goals90", "prior_assists90",
        "realized_goals90", "realized_assists90", "realized_points90",
    ]


def test_compute_league_stats_below_min_samples_returns_none():
    holdout = pd.DataFrame({
        "code": [1], "prior_goals90": [0.5], "prior_assists90": [0.1],
        "realized_goals90": [0.6], "realized_assists90": [0.1],
        "realized_points90": [8.0],
    })
    factor, variance, n = plt.compute_league_stats(holdout)
    assert (factor, variance, n) == (None, None, 1)


def test_compute_league_stats_ratio_of_medians():
    # 20 rows (>= MIN_CALIBRATION_SAMPLES=15): prior median 0.50, realized median 0.40
    # -> factor 0.80. Interleaved so the median lands exactly on those values.
    prior = [0.4] * 10 + [0.6] * 10
    realized = [0.3] * 10 + [0.5] * 10
    holdout = pd.DataFrame({
        "code": range(20),
        "prior_goals90": prior, "prior_assists90": [0.0] * 20,
        "realized_goals90": realized, "realized_assists90": [0.0] * 20,
        "realized_points90": [5.0] * 10 + [7.0] * 10,
    })
    factor, variance, n = plt.compute_league_stats(holdout)
    assert n == 20
    assert factor == pytest.approx(0.4 / 0.5)
    assert variance == pytest.approx(pd.Series([5.0] * 10 + [7.0] * 10).var(ddof=1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_prior_league_translation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'projection.prior_league_translation'`

- [ ] **Step 3: Implement `projection/prior_league_translation.py`**

```python
"""prior_league_translation.py — P11 cross-league translation-factor
calibration: how much a prior-league (non-PL) per-90 attacking output
scales to its PL-equivalent, fit against a real hold-out of players who
actually made that jump in a past season (not asserted from literature).

Deliberately NOT persisted as a table -- build_holdout() is cheap to
recompute (season-aggregate rows, not per-match) and gets more accurate
for free as more PL seasons accumulate in future years. Compute once via
scripts/calibrate_prior_league_factors.py, hand-copy the result into
config/strategy.py's PriorLeagueRules -- same precedent as this session's
scripts/calibrate_risk_constants.py.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from data.db import get_session
from data.ingestors.fbref import SEASON_MAP
from projection.cold_start import MIN_PRIOR_APPEARANCES

# MIN_PRIOR_APPEARANCES full 90-minute appearances' worth, translated from
# cold_start.py's per-appearance bar since prior_league_stats is
# season-aggregate, not per-appearance.
MIN_HOLDOUT_MINUTES = MIN_PRIOR_APPEARANCES * 90

# (prior-league season, immediately-following PL season) -- every transition
# we have BOTH sides of real data for (player_gw_stats covers 2021-22..2025-26).
SEASON_TRANSITIONS: list[tuple[str, str]] = [
    ("2021-22", "2022-23"),
    ("2022-23", "2023-24"),
    ("2023-24", "2024-25"),
    ("2024-25", "2025-26"),
]

MIN_CALIBRATION_SAMPLES = 15

_HOLDOUT_COLUMNS = [
    "code", "prior_goals90", "prior_assists90",
    "realized_goals90", "realized_assists90", "realized_points90",
]


def _prior_side(league: str, prior_season: str) -> pd.DataFrame:
    """Matched (code populated), qualifying prior-league rows for one season."""
    db = get_session()
    try:
        soccerdata_season = SEASON_MAP.get(prior_season, prior_season)
        query = text("""
            SELECT code, goals90, assists90, minutes
            FROM prior_league_stats
            WHERE league = :league AND season = :season
              AND code IS NOT NULL AND minutes >= :min_minutes
        """)
        return pd.read_sql(query, db.bind, params={
            "league": league, "season": soccerdata_season,
            "min_minutes": MIN_HOLDOUT_MINUTES,
        })
    finally:
        db.close()


def _realized_pl_side(codes: list[int], pl_season: str) -> pd.DataFrame:
    """Real PL per-90 output (goals, assists, total points) for a set of
    codes in one PL season, summed across every gameweek/fixture that
    season (a genuine DGW player has two rows for one gameweek -- summing
    is correct here, not a double-count, since we want the season total)."""
    empty = pd.DataFrame(columns=["code", "pl_minutes", "pl_goals", "pl_assists", "pl_points"])
    if not codes:
        return empty
    db = get_session()
    try:
        placeholders = ", ".join(f":code{i}" for i in range(len(codes)))
        params = {f"code{i}": c for i, c in enumerate(codes)}
        params["season"] = pl_season
        query = text(f"""
            SELECT p.code AS code,
                   SUM(g.minutes) AS pl_minutes,
                   SUM(g.goals_scored) AS pl_goals,
                   SUM(g.assists) AS pl_assists,
                   SUM(g.total_points) AS pl_points
            FROM player_gw_stats g
            JOIN players p ON p.id = g.player_id
            WHERE g.season = :season AND p.code IN ({placeholders})
            GROUP BY p.code
        """)
        result = pd.read_sql(query, db.bind, params=params)
        return result if not result.empty else empty
    finally:
        db.close()


def build_holdout(league: str) -> pd.DataFrame:
    """Pooled made-the-jump hold-out for one prior league across every
    available historical season-transition: one row per qualifying player
    with both their prior-league per-90s and their realized PL per-90s."""
    rows = []
    for prior_season, pl_season in SEASON_TRANSITIONS:
        prior = _prior_side(league, prior_season)
        if prior.empty:
            continue
        realized = _realized_pl_side(prior["code"].tolist(), pl_season)
        if realized.empty:
            continue
        realized = realized[realized["pl_minutes"] >= MIN_HOLDOUT_MINUTES]
        if realized.empty:
            continue
        merged = prior.merge(realized, on="code", how="inner")
        if merged.empty:
            continue
        merged["realized_goals90"] = merged["pl_goals"] / merged["pl_minutes"] * 90
        merged["realized_assists90"] = merged["pl_assists"] / merged["pl_minutes"] * 90
        merged["realized_points90"] = merged["pl_points"] / merged["pl_minutes"] * 90
        merged = merged.rename(
            columns={"goals90": "prior_goals90", "assists90": "prior_assists90"}
        )
        rows.append(merged[_HOLDOUT_COLUMNS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=_HOLDOUT_COLUMNS)


def compute_league_stats(holdout: pd.DataFrame) -> tuple[float | None, float | None, int]:
    """(translation_factor, realized_points90_variance, hold-out sample
    size). Both are None when the sample is too sparse to trust -- caller
    falls back to a literature-style default for that league."""
    n = len(holdout)
    if n < MIN_CALIBRATION_SAMPLES:
        return None, None, n
    prior_median = (holdout["prior_goals90"] + holdout["prior_assists90"]).median()
    realized_median = (holdout["realized_goals90"] + holdout["realized_assists90"]).median()
    factor = float(realized_median / prior_median) if prior_median > 0 else None
    variance = float(holdout["realized_points90"].var(ddof=1)) if n > 1 else None
    return factor, variance, n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_prior_league_translation.py -v`
Expected: all tests PASS (2 from Task 2 + 5 new)

- [ ] **Step 5: Run the full suite + lint**

Run: `uv run python -m pytest -q && uv run ruff check .`
Expected: all tests pass, `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add projection/prior_league_translation.py tests/test_prior_league_translation.py
git commit -m "feat(p11): translation-factor calibration module (pure, hold-out-driven)"
```

---

### Task 4: Calibration script

**Files:**
- Create: `scripts/calibrate_prior_league_factors.py`

**Interfaces:**
- Consumes: `data.ingestors.fbref_prior.PRIOR_LEAGUES` (dict),
  `projection.prior_league_translation.{build_holdout, compute_league_stats,
  MIN_CALIBRATION_SAMPLES}`.
- Produces: printed `config/strategy.py`-ready field values (stdout only — no DB
  writes, no return value consumed elsewhere).

No automated test for this script: it is a thin, real-DB-driven CLI (same category
as `scripts/scrape_fbref.py`/`scripts/calibrate_risk_constants.py`) whose actual
output depends on real historical prior-league data that does not exist in this
environment yet (needs the browser-based scrape from
`plan/p11-prior-league-cold-start.md`'s section 1, run on a machine with Chromium).
Its logic is already fully covered by Task 3's tests (`build_holdout`/
`compute_league_stats` are the only real computation it does).

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""calibrate_prior_league_factors.py — P11: compute real per-league
translation factors + variances from the made-the-jump hold-out, and print
the config/strategy.py values to hand-copy in.

Needs prior_league_stats already populated for every prior season in
projection.prior_league_translation.SEASON_TRANSITIONS (run
scripts/scrape_prior_league.py once per (league, season) -- see
plan/p11-prior-league-cold-start.md section 1) with identity mapping
already applied (scrape_prior_league.py runs the backfill automatically).

Usage: uv run python scripts/calibrate_prior_league_factors.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from data.ingestors.fbref_prior import PRIOR_LEAGUES
from projection.prior_league_translation import (
    MIN_CALIBRATION_SAMPLES,
    build_holdout,
    compute_league_stats,
)

logger = logging.getLogger(__name__)

# league label -> config/strategy.py PriorLeagueRules field-name suffix.
_FIELD_SUFFIX = {
    "ENG-Championship": "championship",
    "ESP-La Liga": "la_liga",
    "ITA-Serie A": "serie_a",
    "GER-Bundesliga": "bundesliga",
    "FRA-Ligue 1": "ligue_1",
}

# The literature-style fallback already checked into PriorLeagueRules --
# reused here so a sparse league's printed line matches what's already live
# rather than silently proposing something different.
_CURRENT_DEFAULT_FACTOR = {
    "ENG-Championship": 0.65, "ESP-La Liga": 1.0, "ITA-Serie A": 1.0,
    "GER-Bundesliga": 1.0, "FRA-Ligue 1": 1.0,
}
_CURRENT_DEFAULT_VARIANCE = 6.0


def main() -> None:
    print("# Paste into config/strategy.py's PriorLeagueRules if these look sane:")
    for league in PRIOR_LEAGUES:
        holdout = build_holdout(league)
        factor, variance, n = compute_league_stats(holdout)
        suffix = _FIELD_SUFFIX[league]
        if factor is None:
            logger.warning(
                "%s: hold-out too sparse (n=%d < %d) -- keeping literature default %.2f",
                league, n, MIN_CALIBRATION_SAMPLES, _CURRENT_DEFAULT_FACTOR[league],
            )
            factor, variance = _CURRENT_DEFAULT_FACTOR[league], _CURRENT_DEFAULT_VARIANCE
        else:
            logger.info("%s: n=%d, factor=%.3f, variance=%.3f", league, n, factor, variance)
        print(f"    translation_factor_{suffix}: float = {factor:.3f}  # n={n}")
        print(f"    translation_variance_{suffix}: float = {variance:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it at least imports and runs against an empty DB**

Run: `uv run python scripts/calibrate_prior_league_factors.py`
Expected: prints all 5 leagues with a `hold-out too sparse (n=0 < 15)` warning each
and the literature-default lines (since no real prior-league scrape has been run in
this environment) — confirms the script is wired correctly end-to-end even without
real data.

- [ ] **Step 3: Commit**

```bash
git add scripts/calibrate_prior_league_factors.py
git commit -m "feat(p11): offline calibration script for prior-league translation factors"
```

---

### Task 5: Cold-start wiring

**Files:**
- Modify: `projection/cold_start.py`
- Test: `tests/test_cold_start.py`

**Interfaces:**
- Consumes: `config.strategy.PRIOR_LEAGUE` (Task 2), `projection.goals.expected_goal_points`,
  `projection.assists.expected_assist_points`, `config.strategy.SCORING`.
- Produces: `load_prior_league_lookup(season: str) -> dict[int, dict]` (new);
  `project_cold_start(..., prior_league_lookup: dict[int, dict] | None = None)` (new
  optional param, backward-compatible default `None`); `proj_source` value
  `"prior_league_prior"` (new, joining the existing `"prior_season"` /
  `"peer_bucket_prior"` / `"position_price_prior"` set).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cold_start.py` (after the existing `_seed_variance_pool`-based
tests, before `_seed_full_pool`):

```python
from data.models import PriorLeagueStats


def test_new_signing_with_matched_prior_league_row_gets_prior_league_prior(temp_session):
    _seed(temp_session)  # p2 = NewSign, code=2, position FWD, price 6.5
    players = cs.apply_departure_gate(cs.load_current_players())
    prior = cs.load_prior_season_features("2025-26")
    lookup = {
        2: {"league": "ENG-Championship", "goals90": 0.6, "assists90": 0.2,
            "npxg90": 0.5, "xa90": 0.15, "minutes": 3000, "matches": 34},
    }
    proj = cs.project_cold_start(players, prior, prior_league_lookup=lookup)

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    row = proj[proj["name"] == "NewSign"].iloc[0]

    assert row["proj_source"] == "prior_league_prior"
    assert row["xpts"] > cs._price_prior("FWD", 6.5)
    assert row["xpts_var"] == pytest.approx(
        cs.PRIOR_LEAGUE.translation_variance("ENG-Championship")
    )
    # nailed-on Championship starter (3000/34/90 ~= 0.98 share) blended 50/50
    # with the flat 0.6 default -> higher than the flat default alone.
    assert row["start_probability"] > cs.NEW_PLAYER_START_PROB


def test_new_signing_with_no_prior_league_match_falls_through_unchanged(temp_session):
    # regression guard: passing an EMPTY lookup must behave exactly like
    # passing none at all (today's existing peer_bucket_prior cascade).
    _seed_variance_pool(temp_session)
    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    proj = cs.project_cold_start(players, prior, raw_appearances=raw, prior_league_lookup={})

    by_name = players.set_index("id")["web_name"].to_dict()
    proj["name"] = proj["player_id"].map(by_name)
    row = proj[proj["name"] == "NewMid"].iloc[0]
    assert row["proj_source"] == "peer_bucket_prior"


def test_load_prior_league_lookup_reads_matched_rows_for_the_right_prior_season(
    temp_session,
):
    s = temp_session()
    try:
        s.add(PriorLeagueStats(
            player_name="Prolific Striker", team="Leeds", league="ENG-Championship",
            season="2025-2026", code=42, position="FW", minutes=3000, matches=34,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.add(PriorLeagueStats(
            player_name="Stale Season", team="Leeds", league="ENG-Championship",
            season="2024-2025", code=43, position="FW", minutes=3000, matches=34,
            goals90=0.6, assists90=0.2, npxg90=0.5, xa90=0.15,
        ))
        s.commit()
    finally:
        s.close()

    lookup = cs.load_prior_league_lookup("2026-27")
    assert set(lookup.keys()) == {42}
    assert lookup[42]["league"] == "ENG-Championship"
    assert lookup[42]["npxg90"] == pytest.approx(0.5)


def test_load_prior_league_lookup_empty_when_nothing_ingested(temp_session):
    assert cs.load_prior_league_lookup("2026-27") == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_cold_start.py -v`
Expected: the 4 new tests FAIL — `TypeError: project_cold_start() got an unexpected
keyword argument 'prior_league_lookup'` and `AttributeError: module
'projection.cold_start' has no attribute 'load_prior_league_lookup'`

- [ ] **Step 3: Implement the wiring in `projection/cold_start.py`**

Add to the imports at the top of the file:

```python
from config.strategy import PRIOR_LEAGUE, SCORING
from projection.assists import expected_assist_points
from projection.goals import expected_goal_points
```

Change `load_current_players`'s query to also select `code`:

```python
def load_current_players() -> pd.DataFrame:
    """Candidate universe for the initial squad: the current bootstrap players."""
    db = get_session()
    try:
        query = text("""
            SELECT id, code, web_name, position, now_cost, status, team_id
            FROM players
        """)
        return pd.read_sql(query, db.bind)
    finally:
        db.close()
```

Add this new constant near the top, next to `NEW_PLAYER_START_PROB`:

```python
# Blend weight toward a matched prior-league player's own minutes-share when
# setting their start probability (P11) -- 0.5 is a deliberately moderate
# starting choice, not itself backtested.
_PRIOR_LEAGUE_START_PROB_WEIGHT = 0.5
```

Add this new loader function, right after `load_current_players`:

```python
def load_prior_league_lookup(season: str) -> dict[int, dict]:
    """code -> matched prior_league_stats row for the season immediately
    before ``season`` (e.g. season="2026-27" reads 2025-26 prior-league
    data) -- the P11 translated prior for players with no PL history.
    Empty dict (never crashes) if nothing has been ingested yet."""
    from data.ingestors.fbref import SEASON_MAP

    prior_season = prior_season_of(season)
    soccerdata_season = SEASON_MAP.get(prior_season, prior_season)
    db = get_session()
    try:
        query = text("""
            SELECT code, league, goals90, assists90, npxg90, xa90, minutes, matches
            FROM prior_league_stats
            WHERE season = :season AND code IS NOT NULL
        """)
        df = pd.read_sql(query, db.bind, params={"season": soccerdata_season})
    finally:
        db.close()
    if df.empty:
        return {}
    # a code should map to exactly one league per season; if a mid-season
    # transfer somehow produced two rows, keep the one with more minutes.
    df = df.sort_values("minutes", ascending=False).drop_duplicates(subset="code", keep="first")
    return df.set_index("code").to_dict("index")
```

Add this new helper function, right after `_price_prior`:

```python
def _prior_league_projection(position: str, pl_row: dict) -> tuple[float, float, float]:
    """(xpts, xpts_var, start_probability) for a matched prior-league
    player (P11). xpts is built from translated npxG90/xA90 (the smoother,
    luck-adjusted quality metrics -- one prior season's raw goals/assists is
    a small, high-variance sample) plus a flat appearance-points constant.
    The translation factor itself is still fit against realized RAW
    goal+assist output (the actual ground truth being predicted) --
    projection/prior_league_translation.py -- only this application uses
    the smoother inputs. Clean sheets/bonus/cards are NOT estimated:
    prior_league_stats has no defensive data for these players, an honest
    limitation, not an oversight."""
    factor = PRIOR_LEAGUE.translation_factor(pl_row["league"])
    translated_npxg90 = pl_row["npxg90"] * factor
    translated_xa90 = pl_row["xa90"] * factor
    xpts = max(
        _MIN_XPTS,
        expected_goal_points(translated_npxg90, position)
        + expected_assist_points(translated_xa90)
        + SCORING.points_full_appearance,
    )
    xpts_var = PRIOR_LEAGUE.translation_variance(pl_row["league"])
    prior_minutes_share = min(1.0, pl_row["minutes"] / max(1, pl_row["matches"] * 90))
    start_prob = (
        (1 - _PRIOR_LEAGUE_START_PROB_WEIGHT) * NEW_PLAYER_START_PROB
        + _PRIOR_LEAGUE_START_PROB_WEIGHT * prior_minutes_share
    )
    return xpts, xpts_var, start_prob
```

Change `project_cold_start`'s signature and its `else` branch:

```python
def project_cold_start(
    players: pd.DataFrame,
    prior_features: pd.DataFrame,
    target_gw: int = 1,
    raw_appearances: pd.DataFrame | None = None,
    prior_league_lookup: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """GW1 xPts + xpts_var + start probability per player, tagged with its
    source.

    proj_source is 'prior_season' (established players, real own-variance),
    'prior_league_prior' (new signings/promoted players matched to a
    translated non-PL prior-season record, P11), 'peer_bucket_prior' (no PL
    or prior-league match, pooled real peer data by position+price), or
    'position_price_prior' (last-resort synthetic fallback). Neither xpts
    nor xpts_var is ever left 0.0/undefined by default -- the gate depends
    on it (plan/risk-aware-cold-start-v1.md, extended to variance).

    ``raw_appearances`` (optional, from ``load_prior_season_appearances``):
    powers the real variance computation. ``None`` (or empty) degrades
    every player straight to the synthetic fallback for BOTH xpts and
    xpts_var -- never crashes.

    ``prior_league_lookup`` (optional, from ``load_prior_league_lookup``):
    code -> translated prior-league row (P11). ``None`` (or a code with no
    entry) falls through to the existing peer_bucket_prior /
    position_price_prior cascade, unchanged.
    """
    if raw_appearances is None:
        raw_appearances = pd.DataFrame(columns=["player_id", "total_points"])
    if prior_league_lookup is None:
        prior_league_lookup = {}

    merged = players.merge(
        prior_features, left_on="id", right_on="player_id", how="left"
    )
    merged["appearances"] = merged["appearances"].fillna(0).astype(int)

    peer_buckets = _build_peer_buckets(players, prior_features, raw_appearances)
    own_appearances = raw_appearances.groupby("player_id")["total_points"]

    rows: list[dict] = []
    for r in merged.itertuples():
        has_prior = r.appearances >= MIN_PRIOR_APPEARANCES
        if has_prior:
            xpts = max(_MIN_XPTS, float(r.ppg_played))
            if r.id in own_appearances.groups:
                own_points = own_appearances.get_group(r.id)
                xpts_var = float(own_points.var(ddof=1)) if len(own_points) > 1 else 0.0
            else:
                xpts_var = 0.0
            start_prob = float(r.starts_rate)
            source = "prior_season"
        else:
            code = getattr(r, "code", None)
            pl_row = (
                prior_league_lookup.get(int(code))
                if code is not None and not pd.isna(code)
                else None
            )
            if pl_row is not None:
                xpts, xpts_var, start_prob = _prior_league_projection(r.position, pl_row)
                source = "prior_league_prior"
            else:
                peer_stats = _peer_bucket_stats(r.position, float(r.now_cost), peer_buckets)
                if peer_stats is not None:
                    xpts, xpts_var = peer_stats
                    xpts = max(_MIN_XPTS, xpts)
                    source = "peer_bucket_prior"
                else:
                    xpts = _price_prior(r.position, float(r.now_cost))
                    xpts_var = _FALLBACK_VAR
                    source = "position_price_prior"
                start_prob = NEW_PLAYER_START_PROB
        rows.append({
            "player_id": int(r.id),
            "gameweek": target_gw,
            "xpts": xpts,
            "xpts_var": xpts_var,
            "start_probability": start_prob,
            "proj_source": source,
        })
    return pd.DataFrame(rows)
```

Finally, wire it into `build_initial_squad` (in the body, right after
`raw_appearances = load_prior_season_appearances(prior_season)`):

```python
    prior_league_lookup = load_prior_league_lookup(season)
    projections = project_cold_start(
        players, prior, raw_appearances=raw_appearances,
        prior_league_lookup=prior_league_lookup,
    )
```

(replacing the existing `projections = project_cold_start(players, prior,
raw_appearances=raw_appearances)` line).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_cold_start.py -v`
Expected: all tests PASS (12 existing + 4 new)

- [ ] **Step 5: Run the full suite + lint**

Run: `uv run python -m pytest -q && uv run ruff check .`
Expected: all tests pass (452+ total), `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add projection/cold_start.py tests/test_cold_start.py
git commit -m "feat(p11): wire the translated prior-league prior into project_cold_start

New top-priority tier ahead of peer_bucket_prior/position_price_prior for
players with a matched, translated non-PL prior-season record. xpts built
from translated npxG90/xA90 + a flat appearance-points constant; start
probability blended 50/50 with the player's own prior-league minutes
share. Falls through to the existing cascade unchanged when there's no
match -- purely additive, all existing tests still pass."
```

---

### Task 6: Docs + final verification

**Files:**
- Modify: `plan/phase-2-xpts-engine.md`

- [ ] **Step 1: Update the P11 status line**

In `plan/phase-2-xpts-engine.md`, change:

```
### P11 — Promoted-team / new-signing prior model  *(Phase-1 T7 deferred; the biggest alpha source)*
```

to:

```
### P11 — Promoted-team / new-signing prior model  ⚠️ CODE COMPLETE, CALIBRATION PENDING (see plan/p11-prior-league-cold-start.md)
```

and add, directly under that heading, before the existing body text:

```
**2026-08-01:** identity mapping, translation-factor calibration module, and
cold-start wiring are all built and tested (see
plan/p11-prior-league-cold-start.md). `config/strategy.py`'s
`PriorLeagueRules` currently holds the plan's literature-style default
factors (Championship 0.65, top-5 1.0) -- the REAL calibration
(`scripts/calibrate_prior_league_factors.py`) needs the historical
prior-league scrape (5 leagues x 4 past seasons + the current season, all
browser-only) run on a machine with Chromium before it produces real
numbers. Until that scrape runs, new 26/27 signings/promoted players with a
name-matched prior-league record still get a real per-player projection
(not the flat fallback) -- just with an uncalibrated discount factor.
```

- [ ] **Step 2: Run the full suite + lint one more time**

Run: `uv run python -m pytest -q && uv run ruff check .`
Expected: all tests pass, `All checks passed!`

- [ ] **Step 3: Commit**

```bash
git add plan/phase-2-xpts-engine.md
git commit -m "docs(p11): mark P11 code-complete, calibration pending real scrape"
```

---

## Post-implementation manual step (not part of this plan's automated tasks)

Once this plan is fully implemented, run on a machine with Chromium
(same limitation as this session's FBref/WhoScored live-scrape steps —
cannot be done in this sandboxed environment):

1. `uv run python scripts/scrape_prior_league.py <league> <season>` for each of the
   5 leagues × 5 seasons (4 historical calibration pairs + the current 2025-2026
   season) — 25 runs total. Each call also runs the identity-mapping backfill
   automatically.
2. `uv run python scripts/calibrate_prior_league_factors.py` — hand-copy its printed
   values into `config/strategy.py`'s `PriorLeagueRules`.
3. Re-run the full test suite once more after the config change (a pure data
   substitution — no code path changes, but worth confirming nothing assumed the
   literature defaults specifically).
