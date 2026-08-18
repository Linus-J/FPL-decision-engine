# GW1 checklist — 2026/27

**Deadline: Friday 21 August 2026, 17:30 BST.**

Several steps are manual by design (submission is out of scope; see
[docs/decision-engine.md](decision-engine.md)). This is what needs doing and
when.

---

## The final run (2026-08-19, verified 2026-08-18)

State as checked on the 18th: `v2` already carries every change from the
engine review except the vice-captain fix, and merging the rest is a real
merge with **zero conflicts** — the `data/simulations/*.json` clashes from
earlier in the day are already resolved. `v2` is 17 commits ahead of
`origin/v2` and unpushed.

```bash
# 1. one commit left to merge; no conflicts expected
git merge worktree-engine-review

# 2. full cycle. Re-ingests odds, which matters: prices move on team news
#    right up to kickoff, and the team-name matcher fix means all 10 GW1
#    fixtures now price (it was 6 of 10 before).
uv run python scripts/run_weekly.py --dry-run

# 3. preflight WILL fail on decision-surface drift. That is the guard doing
#    its job -- the squad changed. Read the diff, then accept it.
uv run python scripts/preflight.py --update-baseline

# 4. see the reasoning behind the squad before typing it in
uv run python scripts/explain_squad.py --pool 10 --out /tmp/why.md
```

Then enter the squad by hand (§1 below). `DRY_RUN=true`, so nothing submits
itself.

Worth doing at some point: `git push origin v2`. Seventeen commits of the
review exist only on this machine.

**What to look at in the explainer**, in order of how much it tends to change
your mind:

- *Who has actually played for this club* — the shortlist for a team-news
  check. Anyone with no minutes at his current club is rated on someone
  else's team, and the model cannot know it.
- *Margins* — a pick worth under about half a point over its replacement is
  not really a decision. Those are the free slots for a hunch of your own.
- *How much is measured vs modelled* — only GW1 has bookmaker odds. Any
  reasoning about gameweeks three or more out rests on the modelled part.
- *Correlated exposure* — clubs at the three-player cap, and any keeper
  sharing a clean sheet with his own defender. The objective is blind to
  both.

**Overrides.** `config/transfer_overrides.yaml` has two tiers. `exclude` is a
hard veto: out of the pool, force-sold if ever owned, and blocked from
returning via a transfer. `rotation_risk` is a soft cap on start probability
that discounts but lets the optimiser decide anyway — on the live frame two of
five capped players were still selected. Currently eight vetoes and no caps.
Both are hand-edited and both want a reason and a date.

A managerial change is the case neither tier can infer and only you can enter:
it breaks the assumption the minutes model rests on — that a player's own
recent record predicts his next start — and nothing in the data marks one.

## Before the deadline

### 1. Enter the squad on the FPL site

The bot decides; you enter. This is the recorded decision as of 2026-08-17
(GW1, cold start, £100.0m spent, £0.0m bank).

**Regenerate this table from the database rather than trusting it.** An
earlier version of this file listed a bench that `decision_log` did not
contain, which would have put three wrong players on it:

```bash
DB_PATH=fpl_bot_v2.db .venv/bin/python -c "
import sqlite3, json
c = sqlite3.connect('fpl_bot_v2.db')
d = json.loads(c.execute(\"SELECT details FROM decision_log WHERE decision_type='lineup' ORDER BY created_at DESC LIMIT 1\").fetchone()[0])
info = {r[0]: r for r in c.execute('SELECT id, web_name, now_cost, position FROM players')}
print('XI :', [(info[i][1], info[i][3]) for i in d['starting_ids']])
print('BEN:', [info[int(p)][1] for p, _ in sorted(d['bench_order'].items(), key=lambda kv: kv[1])])
print('C  :', info[d['captain_id']][1], '| V:', info[d['vice_captain_id']][1])"
```

**Starting XI** — 1-4-5-1

| Pos | Player | Price |
|---|---|---|
| GKP | Raya **(V)** | £6.0m |
| DEF | Gabriel **(C)** | £8.0m |
| DEF | Virgil | £6.5m |
| DEF | Tarkowski | £6.0m |
| DEF | Guéhi | £6.0m |
| MID | B.Fernandes | £12.0m |
| MID | Semenyo | £8.5m |
| MID | Gibbs-White | £8.0m |
| MID | Rice | £7.5m |
| MID | Anderson | £6.5m |
| FWD | Thiago | £8.0m |

**Bench, in order** — 1. Dubravka (GKP) £4.0m · 2. Targett (DEF) £4.0m ·
3. Mheuka (FWD) £4.5m · 4. Furo (FWD) £4.5m

> Changed 2026-08-18. The cold start now prices fixtures from bookmakers'
> odds where they exist, instead of a strength ratio whose entire GW1 range
> was 0.89–1.10 against the market's 0.46–1.72. Arsenal at home to Coventry
> is the swing: Gabriel becomes the highest-projected player in the league
> for GW1 (8.50 xPts) and takes the armband.
>
> **The captain is a defender.** That follows from maximising expected points
> with `risk_level = 0` — Gabriel's mean beats Haaland's 7.50, though his
> ceiling is far lower. If you would rather captain for upside, that is the
> `risk_level` / `mu_baseline` dial, not a bug.

Bench order matters: it decides which substitute comes on if a starter
blanks, and the outcome scorer replays FPL's real auto-substitution rules
against it.

### 1b. Verify before you type it in

```bash
DB_PATH=fpl_bot_v2.db .venv/bin/python scripts/preflight.py
```

Checks the squad against FPL's own rules (15 players, 2/5/5/3, £100m, max 3
per club, legal XI shape, captain in the XI), confirms nothing leaked past the
deadline, that the site export matches `decision_log`, that the fallbacks are
engaging, and — most importantly — **diffs the whole decision surface against
`config/preflight_baseline.json`**.

That last part is the one that matters. Five defects on 2026-08-17 were
introduced by fixes made the same day; every one passed the tests and the data
gate, because each changed stored state that some other consumer read. Drift
against the baseline is the only signal that catches that class.

If it reports DRIFT, read the was/now and decide whether you meant it. If you
did:

```bash
DB_PATH=fpl_bot_v2.db .venv/bin/python scripts/preflight.py --update-baseline
```

Re-running a gameweek is expected and does **not** trip it — superseded rows
are reported as harmless, not failed.

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

All three items previously listed here were closed on 2026-08-17:

- **Cold-start penalty duty** — implemented, for takers *new* to the duty
  only. Established takers are excluded because their prior-season points
  already contain their penalties. It did not change the GW1 squad.
- **Cold-start distributions on the site** — fixed in `089e9de`; the interval
  is derived from `xpts_var` by normal approximation when there are no Monte
  Carlo draws.
- **`swept_axis` is an unversioned string** — a rename is now caught at
  cohort load rather than at season's end.

What genuinely remains is a data limitation rather than a decision:

- **Odds beyond the near term.** Bookmakers price roughly one gameweek ahead,
  so the back half of the planning horizon projects on a flat league-average
  scoreline. Tracked by the coverage log in step 8 above.
- **The prior-league tier cannot see penalty duty.** A foreign signing's
  goals already include penalties taken abroad, and nothing in the data says
  whether he was on them, so no correction is safe either way.
- **Prior-league expected goals: solved, via Understat rather than FBref.**
  Your `--no-cache` run proved the flag works and that FBref simply does not
  serve xG on that endpoint — the freshly-fetched page is the right page and
  has no xG field at all. Do not re-run it expecting `npxg90`.

  Understat has the data and needs no browser. Refresh it with:

  ```bash
  DB_PATH=fpl_bot_v2.db .venv/bin/python -c "
  from data.ingestors.understat_prior import ingest_prior_league_expected, UNDERSTAT_PRIOR_LEAGUES
  for lg in UNDERSTAT_PRIOR_LEAGUES:
      print(lg, ingest_prior_league_expected(lg, '2025-2026'))"
  ```

  `updated` is the only number that means the data arrived. Serie A needs a
  one-time `{"Understat": "Serie A"}` entry in
  `~/soccerdata/config/league_dict.json` (already added).

  The **Championship** has no expected-goals data from any of these sources
  and projects from `npg90` (non-penalty goals) instead — which is penalty-free
  and was the part that actually mattered.

- **Team strengths are neutral until FPL publishes.** FPL serves pre-season
  placeholders; storing them verbatim would have put every FDR feature
  off-scale for the season, so they are held at the neutral 1200 and carry no
  signal. They self-correct on the first ingest after FPL publishes real
  values — the gate fails if they ever land off-scale again.
