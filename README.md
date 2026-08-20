# FPL Decision Engine — 2026/27

A Fantasy Premier League decision engine. It ingests live data, projects every
player's points as a **distribution** rather than a number, and solves for the
squad, transfers, captain and chip timing that maximise expected points over a
rolling horizon. Then it explains itself and hands you a team sheet.

---

## What it actually does

**Projects distributions, not point estimates.** 150 Monte Carlo scenarios per
fixture. Team goals are drawn once per fixture-scenario and every dependent
quantity conditions on that draw, so a goalkeeper and his own centre-back share
a clean sheet the way they do in reality, and bonus points are a fixture-relative
BPS ranking rather than a per-player average. Expected goals come from de-vigged
bookmaker odds where they exist, and from a Dixon-Coles model fitted to five
seasons of match results where they do not.

**Corrects for its own optimism.** Repeatedly picking whoever looks best
oversamples players whose estimate is inflated by noise — measured here at
+1.2 to +1.3 points per player among the top 50 by projection. Per-player
empirical-Bayes shrinkage by estimation uncertainty, banded by price so an
unknown regresses toward players at his own price rather than toward the league.

**Discounts what it cannot know.** A five-gameweek horizon is weighted
`0.85^n`. Bookmakers price about one round ahead, so on a typical pre-season
frame only ~22% of a squad's projected points rest on real odds and the rest on
a model — weighting them equally would put full confidence in the least
reliable numbers in the system.

**Prices contingencies at their probability.** The first substitute is reached
by an automatic substitution about half the time, the third almost never, so
bench slots are weighted 0.53 / 0.15 / 0.03 rather than uniformly. The vice
captaincy is worth P(captain does not feature) — about 0.17 — rather than
nothing.

**Explains the decision.** `scripts/explain_squad.py` reports, for every pick,
what the squad loses if that player is banned and it re-solves; how much of the
projection is measured versus modelled; which clubs sit at the selection cap;
and which players have never actually played for the club they are now at. With
`--pool N` it generates the N best distinct squads via no-good cuts, so you can
see whether a pick is a conviction or a coin toss.

**Validates against itself.** A walk-forward backtest, a decision-surface
baseline that fails when the answer changes for any reason, ~860 tests, and a
cohort of ~90 shadow managers each varying one parameter through the identical
decision code.

---

## The three parts

1. **The decision engine** — ingest, project, optimise, explain, notify.
2. **A read-only dashboard** (Streamlit) — live squad and projected points,
   fixture and double-gameweek exposure, injury news, projected-versus-actual
   history, and the next planned chip or transfer.
3. **A simulation cohort** — ~90 shadow managers, each varying one parameter,
   stepped through the same decision logic every run. One factor at a time, so
   a single season's results are interpretable rather than a fog.

---

## Where human judgement goes in

The engine has no way to know that a club has changed manager, or that a
signing is competing for a place. It projects minutes from a player's own
record, so a settled record at a *previous* club reads exactly like a settled
record at this one.

`config/transfer_overrides.yaml` is where you tell it, keyed by each player's
stable `code` (not `id`/`fpl_id`, which get reassigned):

| Tier | Effect |
|---|---|
| `confirmed` | Corrects a `team_id` FPL has not caught up on |
| `rumoured` | Discounts a player by P(leaves) |
| `rotation_risk` | Caps start probability — discounts, but the optimiser may still pick him |
| `exclude` | Hard veto — out of the pool, force-sold if owned, blocked from returning |

The two lower tiers are genuinely different in practice: on a live frame, two
of five *capped* players were selected anyway, which is correct behaviour for a
doubt and the wrong behaviour for a decision already made.

A blanket "new signing" discount was measured and **rejected** — across 1,149
player-seasons, prior-season regulars who changed club retained 95.6–97.2% of
the minutes share that stayers retained. Whether a move costs minutes depends
on who else plays there, so it is entered per player, with a reason and a date.

---

## Data sources

All free tier. The free-tier defensive-action gap — clearances, blocks and
recoveries missing from FBref — turned out to be the single biggest lever on
projection accuracy this project found.

| Source | What it provides | Key |
|---|---|---|
| FPL API | Players, fixtures, GW history, team strengths, squad picks | Free |
| vaastav CSV archive | Historical GW stats 2021–25 (backfill) | Free |
| Understat (via `soccerdata`) | Per-match xG/xA/npxG/shots/key passes | Free, browserless |
| WhoScored (via `soccerdata`) | Per-match tackles/interceptions/clearances/blocks/recoveries — closes the DefCon and bonus gap | Free, browser required |
| FBref (via `soccerdata`) | Match events for BPS re-scoring; prior-league bridge | Free, browser required |
| The Odds API | h2h + O/U 2.5 odds → team-goal Poisson λ | Free tier (500 req/month) |
| Guardian API | Press-conference text → availability signals | Free (`api-key=test`) |
| Transfermarkt (scraped) | Confirmed transfers and credibility-scored rumours | Free, on-demand only |

---

## Setup

```bash
uv sync
cp .env.example .env          # set FPL_TEAM_ID at minimum
```

| Variable | Required | Description |
|---|---|---|
| `FPL_TEAM_ID` | Yes | Your FPL team ID, from the URL on the FPL site |
| `DB_PATH` | Recommended | SQLite path. Relative paths resolve against the repo root |
| `THE_ODDS_API_KEY` | Optional | Odds for team-goal λ. Without it, everything falls back to the fitted strength model |
| `GUARDIAN_KEY` | Optional | Press-conference signals (defaults to the low-volume `test` key) |
| `TELEGRAM_BOT_TOKEN` | Optional | Notifications |
| `TELEGRAM_CHAT_ID` | Optional | Notifications |

There is no `FPL_PASSWORD` and no `DRY_RUN`. Neither exists any more; extra
keys in an old `.env` are ignored.

Initialise and backfill:

```bash
uv run python -c "from data.db import init_db; init_db()"
uv run python scripts/backfill_history.py
```

For a true GW1 cold start nothing else is needed — an initial squad is built
from prior-season data automatically. Mid-season, seed your existing squad into
`decision_log` first so the transfer planner has a starting point.

---

## Running

```bash
uv run python scripts/run_weekly.py         # the whole cycle: ingest, decide, simulate
uv run python scripts/run_agent.py          # just this gameweek's decision
uv run python scripts/run_agent.py --chip wildcard
uv run python scripts/run_agent.py --json-out out.json
```

`--dry-run` is accepted and ignored on both. It is kept only so existing
commands and the systemd unit keep working; every run is what it used to mean.

**Before trusting a decision:**

```bash
uv run python scripts/preflight.py                    # checks rules + diffs the decision surface
uv run python scripts/explain_squad.py --pool 10      # why this squad, and how close it was
```

Preflight diffs the whole decision surface against a committed baseline and
fails on any drift. That matters more than it sounds: of nineteen defects found
in one pre-season audit, five were introduced by fixes made the same day, each
passing the tests and the data gate. Accept an intended change with
`--update-baseline`.

**Dashboard**, local and read-only:

```bash
uv run streamlit run dashboard/Home.py
```

**After a gameweek finishes:**

```bash
uv run python scripts/backfill_decision_outcomes.py --season 2026-27
```

**Validation:**

```bash
uv run python scripts/backtest.py --season 2025-26 --start-gw 6 --end-gw 38
uv run python scripts/walk_forward_gate.py --season 2025-26
uv run python scripts/benchmark_strength_models.py   # Dixon-Coles vs the published-strength fallback
```

**Scheduler** (optional, disabled by default — a timer on a machine that is not
always on can silently miss a deadline):

```bash
bash deploy/install.sh
```

---

## Design decisions worth knowing

- **`config/strategy.py` is the single update point** for season rule changes:
  scoring, chip counts, squad structure, risk posture, decay. Nothing else
  should need editing between seasons.
- **Reported expected points are never the optimised quantity.** The objective
  is decayed, risk-adjusted and bench-weighted; `total_xpts` is the true
  undiscounted sum. They are deliberately separate, because a decision aid must
  not quietly restate the thing it is predicting. One consequence: `total_xpts`
  is *not* monotonic under added constraints, so it is the wrong number for
  costing an override.
- **Continuous `risk_level`, not a three-way switch** — −1.0 to +1.0, with
  one-sided semi-deviation so risk-seeking means wanting big good weeks and
  risk-averse means wanting few bad ones. Those are different players.
- **The simulation cohort shares the real decision loop** — parameterised by
  config and storage, never a forked copy.
- **The backtest is pinned to zero-variance scoring** so the walk-forward gate
  stays a comparable yardstick as live defaults evolve.
- **Bench order is load-bearing**, not cosmetic: it is the order automatic
  substitutions consult, and it is surfaced in the team sheet accordingly.
- **SQLite in WAL mode** — sufficient for single-machine use; back up by
  copying the file.

---

## Limitations

- **Squad-level correlation is priced in selection, not in the solver.** The
  projections model it correctly — team goals are drawn once per
  fixture-scenario, so a keeper and his own centre-back share one clean sheet.
  The MILP objective cannot see it: teammate covariance is quadratic in a 0/1
  selection vector and a linear solver has no way to express it, so the
  per-player risk term sums semi-deviations, which assumes teammates move in
  perfect lockstep. `optimiser/joint_risk.py` closes the gap by re-ranking the
  best N squads on the raw scenarios, where the correlation is empirical rather
  than assumed. Its limit is the pool: it can reorder squads the mean objective
  already liked, but it cannot find one that only looks good under the joint
  measure — splitting a keeper from his own defence, say. Local search over
  single-player swaps is the next step if that bound starts to bind.
- **The risk term is inert at the shipped defaults.** `mu = mu_baseline +
  risk_level * mu_range`, and with `mu_baseline = 0.0` and `risk_level = 0` the
  live objective is plain expected points. The variance term, the ownership
  weighting, and covariance-aware captaincy and squad selection are all gated
  behind a non-zero `mu`; the simulation cohort exercises them along its
  `risk_level` axis, the live bot does not. That is a measured position rather
  than an oversight — see below.
- **Covariance-aware selection did not replicate out of sample.** Calibrating
  `mu` on 2025-26 favoured −0.25 by +3.91 actual points per gameweek, paired on
  identical pools and Monte Carlo draws. On 2024-25 the same value is worth
  +0.06, the more negative values are worse than zero, and the concentration it
  is supposed to reduce moves the wrong way. Pooled across both seasons the
  effect is +1.98 pts/GW at t=1.39. An effect whose sign flips between seasons
  is not an effect, so `mu` stays at zero and the machinery stays dormant
  behind it. Deciding this needs a third season, not a re-run of these two.
- **The unpriced-gameweek model carries most of the weight.** Bookmakers price
  one round ahead. Everything beyond that is the fitted Dixon-Coles model, and
  it is the largest single source of uncertainty in any multi-gameweek claim.
- **Several constants are judgement, not calibration** — the decay base is the
  field's convention, and the bench slot weights are static where they should
  tighten as a squad becomes more predictable. A full-season A/B found decay
  worth about +65 points and the bench weights worth about +6, which is noise.
- **Everything depends on the minutes model.** Start probabilities drive
  projections, the candidate filter and the bench weights. Errors there
  propagate further than anywhere else.
- **No price-change or team-value modelling.** Team value compounds over a
  season and this does nothing with it.
