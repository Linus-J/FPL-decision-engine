# GW1 checklist — 2026/27

**Deadline: Friday 21 August 2026, 17:30 BST.**

Several steps are manual by design (submission is out of scope; see
[docs/decision-engine.md](decision-engine.md)). This is what needs doing and
when.

---

## Before the deadline

### 1. Enter the squad on the FPL site

The bot decides; you enter. This is the recorded decision as of 2026-08-16
(GW1, cold start, £100.0m spent, £0.0m bank):

**Starting XI** — 1-4-5-1

| Pos | Player | Price |
|---|---|---|
| GKP | Raya **(V)** | £6.0m |
| DEF | Gabriel | £8.0m |
| DEF | Senesi | £6.0m |
| DEF | Tarkowski | £6.0m |
| DEF | Guéhi | £6.0m |
| MID | B.Fernandes **(C)** | £12.0m |
| MID | Semenyo | £8.5m |
| MID | Gibbs-White | £8.0m |
| MID | Wilson | £6.5m |
| MID | Anderson | £6.5m |
| FWD | Thiago | £8.0m |

**Bench, in order** — 1. Kelleher (GKP) £5.0m · 2. Disasi £4.5m ·
3. Furo £4.5m · 4. Scarlett £4.5m

Bench order matters: it decides which substitute comes on if a starter
blanks, and the outcome scorer replays FPL's real auto-substitution rules
against it.

### 2. Re-run if anything changes

Prices move and injuries land right up to the deadline. Re-running is safe
and idempotent:

```bash
DB_PATH=fpl_bot_v2.db python scripts/run_weekly.py --season 2026-27 \
    --dry-run --skip-match-events
```

Re-running the same gameweek no longer burns a chip (chip usage is
de-duplicated per gameweek), and the outcome scorer keeps only the decision
that stood.

Drop `--skip-match-events` if you have a display available — that enables the
FBref/WhoScored scrapes, which are blocked headless by Cloudflare.

### 3. Push the squad to the site

```bash
DB_PATH=fpl_bot_v2.db python scripts/export_site_data.py
```

Writes `data/simulations/gw{N}.json`, commits it, and pushes. Add
`--no-push` to inspect first.

---

## Immediately after the deadline

### 4. Sample effective ownership — first ever live run

The Overall league has no ranked entries until GW1 locks, so this has never
run against real data and is flagged UNVERIFIED in its own docstring. Run it
once the deadline passes and check it actually writes rows:

```bash
DB_PATH=fpl_bot_v2.db python scripts/ingest_ownership.py 1
```

```sql
SELECT COUNT(*) FROM ownership_snapshots;   -- expect ~600+, not 0
```

Ownership is already wired into the decision cycle, so it starts feeding the
objective as soon as rows exist. It stays inert for the real bot while
`risk_level = 0` (that makes λ zero), but the cohort's `risk_level` axis
exercises it.

---

## After GW1 finishes

### 5. Run the weekly cycle

```bash
DB_PATH=fpl_bot_v2.db python scripts/run_weekly.py --season 2026-27 --dry-run
```

This scores GW1 for the real bot and all 90 personas before making GW2's
decision. First real exercise of the measurement layer.

### 6. Check the numbers came out

```bash
python -c "
from simulation.analysis import calibration, persona_season_summary
print(calibration('2026-27'))
print(persona_season_summary('2026-27').head())"
```

Or open the dashboard's **Simulations** page, which shows the same thing plus
per-axis effects.

**What to look for:** the `bias` column. The backtest suggested the decision
layer over-predicts by ~8 points per gameweek, but that number came from a
harness with known divergences. This is the same measurement on the real code
path with ~90 observations per gameweek. Four or five gameweeks will settle
it. If the bias is real and persistent, every points-denominated threshold in
the engine (hit cost, chip thresholds, switching cost) is being compared
against inflated gains and needs re-tuning.

---

## During the season

### 7. Refresh the penalty depth chart

Duty moves with transfers, managers and form. The published list loaded on
2026-08-16 is a snapshot.

```bash
DB_PATH=fpl_bot_v2.db python scripts/load_setpiece_depth_chart.py \
    <updated-file>.txt --season 2026-27 --source <source>
```

Format is `Team | Penalties | Free Kicks | Corners`, one team per line, each
cell comma-separated in depth-chart order, `*` to flag a doubt. Unresolved
names are reported rather than guessed at.

The weekly FBref scrape defers to this for any player with a published duty,
so it cannot overwrite the list with last season's attempt share.

### 8. Watch the odds window

Every run logs coverage per gameweek. As of 2026-08-16 it is **6/10 for GW1
and 0/10 beyond**, so GW2–5 of the planning horizon project on a flat
league-average scoreline rather than real fixture difficulty. Expect this to
improve once the season starts and bookmakers price further ahead; if it does
not, the multi-week horizon is weaker than its length suggests.

---

## Known open decisions

Not blocking, but worth a call at some point:

- **Cold-start penalty duty.** Penalty attribution reaches the in-season
  engine (`assemble.py`) but not the cold start, which projects from
  prior-season points per appearance. So the GW1 squad was built without it.
  The clean case is new signings with no PL penalty history, where adding
  duty is purely additive; for established takers the penalties are already
  inside their prior-season points and adding more would double-count.
- **Cold-start distributions on the site.** `p10`/`median`/`p90` collapse to
  the same value pre-season, because they come from `projection_samples` and
  the cold start produces no Monte Carlo draws. It does carry `xpts_var`, so
  a normal approximation would give a real spread. Display only — no decision
  reads it.
- **`swept_axis` is an unversioned string.** The season analysis groups on
  it, so renaming an axis mid-season would silently split the grouping.
