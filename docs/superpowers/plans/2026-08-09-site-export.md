# FPL Site Data Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/export_site_data.py`, a manually-triggered command that reads the current squad, top-15 xPts projections, and decision history from the bot's local database and writes/commits/pushes them as compact JSON to `data/simulations/` for the portfolio site to fetch.

**Architecture:** A small `scripts/site_export/` package with three single-responsibility modules — `payload.py` (query DB → build the JSON-shaped dict), `writer.py` (write run file + maintain `index.json`), `git_sync.py` (scoped git add/commit/push) — wired together by the thin `scripts/export_site_data.py` CLI. No new automation triggers it; it's a command the user runs by hand after reviewing each gameweek's decision.

**Tech Stack:** Python 3.12, SQLAlchemy ORM (existing `data.models`), pandas, pytest, uv. No new dependencies.

**Companion spec:** `linus-j.github.io/docs/superpowers/specs/2026-08-09-fpl-site-integration-design.md` (Part 1). This plan covers only the bot-repo half; the site repo has its own separate plan.

## Global Constraints

- No GitHub Actions / CI automation of any kind for this feature — every step here is triggered manually by the user (per the design doc's confirmed scope; the decision pipeline itself, `run_agent.py`/`run_weekly.py`, is unchanged and out of scope for this plan).
- No new FPL/Odds/Guardian credentials — `export_site_data.py` only reads the local DB that a prior manual `run_agent.py --dry-run` already populated.
- No raw per-simulation draws ever written to the export — `projection_samples` rows are aggregated down to `{p10, median, mean, p90}` per player before they leave the database.
- JSON output: `schema_version` field on both `data/simulations/{gw}.json` and `data/simulations/index.json`, currently `1`.
- **Test invocation:** this repo's `pytest` entry-point script does not put the repo root on `sys.path` (there's no `tests/__init__.py`, so pytest's default "prepend" import mode inserts `tests/` itself, not its parent — a pre-existing, repo-wide issue unrelated to this feature). Always run tests as `uv run python -m pytest ...`, never bare `uv run pytest ...` — `python -m` puts the cwd on `sys.path[0]`, which resolves it. Every test command in this plan uses that form.
- Follow the existing test convention exactly (see `tests/test_dashboard_squad.py`, `tests/test_dashboard_decisions.py`): a `session(tmp_path)` pytest fixture backed by a real temporary SQLite file (`create_engine` + `Base.metadata.create_all`), and `monkeypatch.setattr(module, "name", ...)` on the *consuming* module's namespace for anything that hits the network or opens its own DB session internally.
- `git_sync.commit_and_push` must only ever `git add` the `data/simulations/` path, never the whole working tree — this runs on the user's real machine against their real repo.

---

## File Structure

```
scripts/
  export_site_data.py          new — CLI entrypoint
  site_export/
    __init__.py                 new — empty, makes this a package
    payload.py                  new — DB → JSON-shaped dict
    writer.py                   new — write run file + index.json
    git_sync.py                 new — scoped git add/commit/push

tests/
  test_site_export_payload.py   new
  test_site_export_writer.py    new
  test_site_export_git_sync.py  new
  test_export_site_data.py      new

README.md                       modified — new "Site data export" subsection under Running
```

---

### Task 1: `payload.py` scaffold + `get_projection_distributions`

**Files:**
- Create: `scripts/site_export/__init__.py`
- Create: `scripts/site_export/payload.py`
- Create: `tests/test_site_export_payload.py`

**Interfaces:**
- Produces: `payload.get_projection_distributions(db: Session, gw: int, season: str) -> dict[int, dict[str, float]]` — per-player `{"p10": float, "median": float, "mean": float, "p90": float}`, keyed by `player_id`. Empty dict if no samples exist for that (gw, season).

- [ ] **Step 1: Write the failing test**

Create `tests/test_site_export_payload.py`:

```python
"""scripts/site_export/payload.py"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, ProjectionSample
from scripts.site_export import payload as payload_module


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'export.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def _add_samples(
    session, player_id: int, gw: int, season: str, values: list[float],
    created_at: datetime | None = None,
) -> None:
    kwargs = {"created_at": created_at} if created_at is not None else {}
    session.add_all([
        ProjectionSample(player_id=player_id, gameweek=gw, season=season, scenario_id=i, xpts=v, **kwargs)
        for i, v in enumerate(values)
    ])
    session.commit()


def test_get_projection_distributions_returns_summary_per_player(session):
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[2.0, 4.0, 6.0, 8.0, 10.0])
    _add_samples(session, player_id=2, gw=3, season="2026-27", values=[1.0, 1.0, 1.0, 1.0, 1.0])

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert set(dist.keys()) == {1, 2}
    assert dist[1]["mean"] == 6.0
    assert dist[1]["median"] == 6.0
    assert dist[2]["mean"] == 1.0
    assert dist[2]["p10"] == 1.0
    assert dist[2]["p90"] == 1.0


def test_get_projection_distributions_ignores_other_gameweeks_and_seasons(session):
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[5.0, 5.0])
    _add_samples(session, player_id=1, gw=4, season="2026-27", values=[99.0])
    _add_samples(session, player_id=1, gw=3, season="2025-26", values=[99.0])

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert dist[1]["mean"] == 5.0


def test_get_projection_distributions_empty_when_no_samples(session):
    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")
    assert dist == {}


def test_get_projection_distributions_uses_only_latest_batch(session):
    """Guards against blending two pipeline runs' samples together: each
    persist_samples call shares one created_at across its whole batch
    (projection/assemble.py::_write_projection_samples), and old batches
    are never deleted, so a re-run for the same (gw, season) must not
    silently average with the stale batch."""
    _add_samples(
        session, player_id=1, gw=3, season="2026-27", values=[1.0, 1.0, 4.0],
        created_at=datetime(2026, 8, 1, 6, 0),
    )
    _add_samples(
        session, player_id=1, gw=3, season="2026-27", values=[7.0, 8.0, 9.0],
        created_at=datetime(2026, 8, 3, 6, 0),
    )

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert dist[1]["mean"] == 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'scripts.site_export'`

- [ ] **Step 3: Create the package and minimal implementation**

Create `scripts/site_export/__init__.py` (empty file).

Create `scripts/site_export/payload.py`:

```python
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


def get_projection_distributions(db: Session, gw: int, season: str) -> dict[int, dict[str, float]]:
    """Per-player {p10, median, mean, p90} xPts summary from projection_samples,
    aggregated across every MC scenario for one gameweek.

    Scoped to the latest persist_samples batch for this (gw, season):
    projection/assemble.py::_write_projection_samples computes one shared
    created_at per batch and never deletes prior batches, so an unscoped
    query would blend every historical run's samples together if the
    pipeline is ever re-run for the same upcoming gameweek (found during
    Task 1 review, 2026-08-09)."""
    query = text("""
        SELECT player_id, xpts
        FROM projection_samples
        WHERE gameweek = :gw AND season = :season
          AND created_at = (
              SELECT MAX(created_at) FROM projection_samples
              WHERE gameweek = :gw AND season = :season
          )
    """)
    df = pd.read_sql(query, db.bind, params={"gw": gw, "season": season})
    out: dict[int, dict[str, float]] = {}
    for player_id, values in df.groupby("player_id")["xpts"]:
        out[int(player_id)] = {
            "p10": float(values.quantile(0.10)),
            "median": float(values.quantile(0.50)),
            "mean": float(values.mean()),
            "p90": float(values.quantile(0.90)),
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/linus/Projects/FPL-26-27-bot
git add scripts/site_export/__init__.py scripts/site_export/payload.py tests/test_site_export_payload.py
git commit -m "feat(site-export): add projection distribution aggregation"
```

---

### Task 2: `_team_short_names` + `_label_for_gw`

**Files:**
- Modify: `scripts/site_export/payload.py`
- Modify: `tests/test_site_export_payload.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `payload._team_short_names(db: Session) -> dict[int, str]` (team id → short_name); `payload._label_for_gw(db: Session, season: str, gw: int) -> str` (e.g. `"GW3 — 3 Aug"`, falling back to `"GW{gw}"` when the gameweek row doesn't exist).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_site_export_payload.py` (add `from datetime import datetime` and `from data.models import Gameweek, Team` to the existing imports at the top, then append these tests at the end):

```python
def _add_team(session, team_id: int, short_name: str) -> None:
    session.add(Team(id=team_id, name=short_name, short_name=short_name))
    session.commit()


def test_team_short_names_maps_id_to_short_name(session):
    _add_team(session, 1, "ARS")
    _add_team(session, 2, "MCI")

    names = payload_module._team_short_names(session)

    assert names == {1: "ARS", 2: "MCI"}


def test_label_for_gw_uses_deadline_time(session):
    session.add(Gameweek(
        id=3, season="2026-27", name="Gameweek 3",
        deadline_time=datetime(2026, 8, 3, 10, 30),
    ))
    session.commit()

    label = payload_module._label_for_gw(session, "2026-27", 3)

    assert label == "GW3 — 3 Aug"


def test_label_for_gw_falls_back_when_gameweek_missing(session):
    label = payload_module._label_for_gw(session, "2026-27", 99)
    assert label == "GW99"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: FAIL — `AttributeError: module 'scripts.site_export.payload' has no attribute '_team_short_names'`

- [ ] **Step 3: Implement**

In `scripts/site_export/payload.py`, add `from data.models import Gameweek` to the imports, then append:

```python
def _team_short_names(db: Session) -> dict[int, str]:
    df = pd.read_sql(text("SELECT id, short_name FROM teams"), db.bind)
    return dict(zip(df["id"], df["short_name"]))


def _label_for_gw(db: Session, season: str, gw: int) -> str:
    row = db.query(Gameweek).filter(Gameweek.season == season, Gameweek.id == gw).first()
    if row and row.deadline_time:
        return f"GW{gw} — {row.deadline_time.day} {row.deadline_time.strftime('%b')}"
    return f"GW{gw}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/linus/Projects/FPL-26-27-bot
git add scripts/site_export/payload.py tests/test_site_export_payload.py
git commit -m "feat(site-export): add team-name and gameweek-label helpers"
```

---

### Task 3: `_xpts_entry` + `_build_squad_entries`

**Files:**
- Modify: `scripts/site_export/payload.py`
- Modify: `tests/test_site_export_payload.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `payload._xpts_entry(player_id: int, dist: dict[int, dict[str, float]], fallback_mean: float | None) -> dict[str, float] | None`; `payload._build_squad_entries(squad_df: pd.DataFrame, dist: dict[int, dict[str, float]]) -> list[dict]`. `squad_df` is the exact shape `dashboard.data.squad.get_current_squad` returns (columns: `player_id, web_name, position, team_short, now_cost, is_starting, is_captain, is_vice_captain, xpts`). Bench order: GK bench player always `1`, remaining bench players ordered `2, 3, ...` by `xpts` descending; `None` for starters.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_site_export_payload.py` (add `import math` and `import pandas as pd` to the top imports, then append):

```python
def test_xpts_entry_uses_distribution_when_available():
    dist = {1: {"p10": 1.0, "median": 2.0, "mean": 2.5, "p90": 4.0}}
    assert payload_module._xpts_entry(1, dist, fallback_mean=99.0) == dist[1]


def test_xpts_entry_falls_back_to_flat_mean_when_no_samples():
    entry = payload_module._xpts_entry(1, {}, fallback_mean=5.0)
    assert entry == {"p10": 5.0, "median": 5.0, "mean": 5.0, "p90": 5.0}


def test_xpts_entry_returns_none_when_nothing_available():
    assert payload_module._xpts_entry(1, {}, fallback_mean=None) is None
    assert payload_module._xpts_entry(1, {}, fallback_mean=math.nan) is None


def test_build_squad_entries_orders_bench_gk_first_then_by_xpts():
    squad_df = pd.DataFrame([
        {"player_id": 1, "web_name": "Starter", "position": "FWD", "team_short": "MCI",
         "now_cost": 10.0, "is_starting": True, "is_captain": True, "is_vice_captain": False,
         "xpts": 8.0},
        {"player_id": 2, "web_name": "BenchGK", "position": "GKP", "team_short": "ARS",
         "now_cost": 4.5, "is_starting": False, "is_captain": False, "is_vice_captain": False,
         "xpts": 0.5},
        {"player_id": 3, "web_name": "BenchLow", "position": "DEF", "team_short": "ARS",
         "now_cost": 4.5, "is_starting": False, "is_captain": False, "is_vice_captain": False,
         "xpts": 1.0},
        {"player_id": 4, "web_name": "BenchHigh", "position": "MID", "team_short": "ARS",
         "now_cost": 6.0, "is_starting": False, "is_captain": False, "is_vice_captain": False,
         "xpts": 3.0},
    ])

    entries = payload_module._build_squad_entries(squad_df, dist={})

    by_id = {e["player_id"]: e for e in entries}
    assert by_id[1]["bench_order"] is None
    # BenchGK has the LOWEST xpts here deliberately -- if this assertion
    # passed with the GK's xpts as the highest instead, a naive xPts-only
    # sort (ignoring position) would satisfy it too, without actually
    # proving the GK-first rule (found during Task 3 review, 2026-08-09).
    assert by_id[2]["bench_order"] == 1   # GK bench slot always first (despite lowest xPts)
    assert by_id[4]["bench_order"] == 2   # then outfield by xPts descending
    assert by_id[3]["bench_order"] == 3
    assert by_id[1]["xpts"] == {"p10": 8.0, "median": 8.0, "mean": 8.0, "p90": 8.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_xpts_entry'`

- [ ] **Step 3: Implement**

Append to `scripts/site_export/payload.py`:

```python
def _xpts_entry(
    player_id: int, dist: dict[int, dict[str, float]], fallback_mean: float | None
) -> dict[str, float] | None:
    if player_id in dist:
        return dist[player_id]
    if fallback_mean is not None and not pd.isna(fallback_mean):
        mean = float(fallback_mean)
        return {"p10": mean, "median": mean, "mean": mean, "p90": mean}
    return None


def _build_squad_entries(squad_df: pd.DataFrame, dist: dict[int, dict[str, float]]) -> list[dict]:
    bench = squad_df[~squad_df["is_starting"]]
    gk_bench = bench[bench["position"] == "GKP"]
    other_bench = bench[bench["position"] != "GKP"].sort_values("xpts", ascending=False)
    bench_order_by_player: dict[int, int] = {}
    for order, player_id in enumerate(
        [*gk_bench["player_id"], *other_bench["player_id"]], start=1
    ):
        bench_order_by_player[int(player_id)] = order

    entries = []
    for _, row in squad_df.iterrows():
        player_id = int(row["player_id"])
        entries.append({
            "player_id": player_id,
            "web_name": row["web_name"],
            "position": row["position"],
            "team_short": row["team_short"],
            "now_cost": float(row["now_cost"]),
            "is_starting": bool(row["is_starting"]),
            "is_captain": bool(row["is_captain"]),
            "is_vice_captain": bool(row["is_vice_captain"]),
            "bench_order": bench_order_by_player.get(player_id),
            "xpts": _xpts_entry(player_id, dist, row["xpts"]),
        })
    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/linus/Projects/FPL-26-27-bot
git add scripts/site_export/payload.py tests/test_site_export_payload.py
git commit -m "feat(site-export): build squad entries with bench ordering"
```

---

### Task 4: `_build_top15_entries` + `_build_history_entries`

**Files:**
- Modify: `scripts/site_export/payload.py`
- Modify: `tests/test_site_export_payload.py`

**Interfaces:**
- Consumes: `_xpts_entry` from Task 3.
- Produces: `payload._build_top15_entries(projections_df: pd.DataFrame, dist: dict, team_names: dict[int, str]) -> list[dict]` (first 15 rows of `projections_df`, which is expected pre-sorted by xPts descending — matches `projection.pipeline.get_latest_projections`'s own `ORDER BY pp.xpts DESC`). `payload._build_history_entries(history_df: pd.DataFrame) -> list[dict]` (maps `decision_type == "transfers"` and `"chip"` rows only; `"lineup"` rows are dropped from the history feed).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_site_export_payload.py`:

```python
def test_build_top15_entries_takes_first_15_and_maps_team_short():
    projections_df = pd.DataFrame([
        {"player_id": i, "web_name": f"P{i}", "position": "MID", "team_id": 1, "xpts_mean": 20.0 - i}
        for i in range(20)
    ])
    team_names = {1: "ARS"}

    entries = payload_module._build_top15_entries(projections_df, dist={}, team_names=team_names)

    assert len(entries) == 15
    assert entries[0]["player_id"] == 0
    assert entries[0]["team_short"] == "ARS"
    assert entries[0]["xpts"]["mean"] == 20.0


def test_build_history_entries_maps_transfers_and_chips_and_drops_lineup():
    history_df = pd.DataFrame([
        {"gameweek": 3, "decision_type": "transfers", "projected_gain": 1.4, "details": {
            "transfers_in": [{"player_id": 1, "web_name": "Haaland", "cost": 15.1}],
            "transfers_out": [{"player_id": 2, "web_name": "Wilson", "cost": 6.5}],
            "hits_taken": 0,
        }},
        {"gameweek": 3, "decision_type": "chip", "projected_gain": 0.0, "details": {
            "chip": "wildcard", "reason": "squad overhaul",
        }},
        {"gameweek": 3, "decision_type": "lineup", "projected_gain": 55.0, "details": {
            "squad_ids": [1, 2],
        }},
    ])

    entries = payload_module._build_history_entries(history_df)

    assert len(entries) == 2
    assert entries[0] == {
        "gameweek": 3, "type": "transfers",
        "transfers_in": ["Haaland"], "transfers_out": ["Wilson"],
        "hits_taken": 0, "net_xpts_gain": 1.4,
    }
    assert entries[1] == {
        "gameweek": 3, "type": "chip", "chip": "wildcard", "reason": "squad overhaul",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_build_top15_entries'`

- [ ] **Step 3: Implement**

Append to `scripts/site_export/payload.py`:

```python
def _build_top15_entries(
    projections_df: pd.DataFrame, dist: dict[int, dict[str, float]], team_names: dict[int, str]
) -> list[dict]:
    entries = []
    for _, row in projections_df.head(15).iterrows():
        player_id = int(row["player_id"])
        entries.append({
            "player_id": player_id,
            "web_name": row["web_name"],
            "position": row["position"],
            "team_short": team_names.get(int(row["team_id"]), ""),
            "xpts": _xpts_entry(player_id, dist, row["xpts_mean"]),
        })
    return entries


def _build_history_entries(history_df: pd.DataFrame) -> list[dict]:
    entries = []
    for _, row in history_df.iterrows():
        details = row["details"]
        if row["decision_type"] == "transfers":
            entries.append({
                "gameweek": int(row["gameweek"]),
                "type": "transfers",
                "transfers_in": [t["web_name"] for t in details.get("transfers_in", [])],
                "transfers_out": [t["web_name"] for t in details.get("transfers_out", [])],
                "hits_taken": details.get("hits_taken", 0),
                "net_xpts_gain": float(row["projected_gain"]),
            })
        elif row["decision_type"] == "chip":
            entries.append({
                "gameweek": int(row["gameweek"]),
                "type": "chip",
                "chip": details.get("chip"),
                "reason": details.get("reason", ""),
            })
    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/linus/Projects/FPL-26-27-bot
git add scripts/site_export/payload.py tests/test_site_export_payload.py
git commit -m "feat(site-export): build top-15 and history entries"
```

---

### Task 5: `build_run_payload` — full integration

**Files:**
- Modify: `scripts/site_export/payload.py`
- Modify: `tests/test_site_export_payload.py`

**Interfaces:**
- Consumes: `get_projection_distributions`, `_team_short_names`, `_label_for_gw`, `_build_squad_entries`, `_build_top15_entries`, `_build_history_entries` (all from Tasks 1–4); `dashboard.data.squad.get_current_squad(db, team_id)`; `dashboard.data.decisions.get_decision_history(db, limit_gws)`; `projection.pipeline._get_current_season()` and `projection.pipeline.get_latest_projections(gw)`.
- Produces: `payload.build_run_payload(db: Session, team_id: int) -> dict` — the full JSON-shaped payload (`schema_version`, `gameweek`, `label`, `generated_at`, `squad`, `top15`, `history`). Raises `RuntimeError` if no current squad can be resolved. This is what `scripts/export_site_data.py` (Task 8) calls directly.

- [ ] **Step 1: Write the failing tests**

Add these imports to the top of `tests/test_site_export_payload.py` (alongside the existing ones): `import json`, `import dashboard.data.squad as squad_module`, `from data.models import DecisionLog, Player`.

Append to `tests/test_site_export_payload.py`:

```python
def _seed_full_squad(session):
    session.add(Team(id=1, name="Man City", short_name="MCI"))
    session.add(Team(id=2, name="Arsenal", short_name="ARS"))
    session.add_all([
        Player(id=1, fpl_id=101, code=101, first_name="E", second_name="Haaland",
               web_name="Haaland", team_id=1, position="FWD", now_cost=15.1),
        Player(id=2, fpl_id=102, code=102, first_name="C", second_name="Wilson",
               web_name="Wilson", team_id=2, position="FWD", now_cost=6.5),
    ])
    session.add(DecisionLog(
        gameweek=3, decision_type="lineup",
        details=json.dumps({
            "squad_ids": [1, 2], "starting_ids": [1],
            "captain_id": 1, "vice_captain_id": 2,
        }),
        projected_gain=8.0, dry_run=True,
    ))
    session.add(DecisionLog(
        gameweek=3, decision_type="transfers",
        details=json.dumps({
            "transfers_in": [{"player_id": 1, "web_name": "Haaland", "cost": 15.1}],
            "transfers_out": [{"player_id": 3, "web_name": "Old", "cost": 6.0}],
            "hits_taken": 0,
        }),
        projected_gain=1.4, dry_run=True,
    ))
    session.add(Gameweek(
        id=3, season="2026-27", name="Gameweek 3", deadline_time=datetime(2026, 8, 3, 10, 30),
    ))
    session.add_all([
        ProjectionSample(player_id=1, gameweek=3, season="2026-27", scenario_id=i, xpts=v)
        for i, v in enumerate([6.0, 8.0, 10.0])
    ])
    session.commit()


def test_build_run_payload_assembles_full_schema(session, monkeypatch):
    _seed_full_squad(session)

    monkeypatch.setattr(squad_module, "_get_current_and_next_gw", lambda: (3, 3))
    monkeypatch.setattr(squad_module, "get_picks", lambda team_id, gw: {})
    monkeypatch.setattr(
        squad_module, "get_latest_projections",
        lambda gw: pd.DataFrame({"player_id": [1, 2], "xpts": [8.0, 2.0]}),
    )
    monkeypatch.setattr(payload_module, "_get_current_season", lambda: "2026-27")
    monkeypatch.setattr(
        payload_module, "get_latest_projections",
        lambda gw: pd.DataFrame([
            {"player_id": 1, "web_name": "Haaland", "position": "FWD", "team_id": 1,
             "xpts_mean": 8.0},
            {"player_id": 2, "web_name": "Wilson", "position": "FWD", "team_id": 2,
             "xpts_mean": 2.0},
        ]),
    )

    payload = payload_module.build_run_payload(session, team_id=12345)

    assert payload["schema_version"] == 1
    assert payload["gameweek"] == 3
    assert payload["label"] == "GW3 — 3 Aug"
    assert len(payload["squad"]) == 2
    assert len(payload["top15"]) == 2
    haaland_squad_entry = next(e for e in payload["squad"] if e["player_id"] == 1)
    assert haaland_squad_entry["xpts"]["mean"] == 8.0  # from real projection_samples rows
    assert len(payload["history"]) == 1
    assert payload["history"][0]["type"] == "transfers"


def test_build_run_payload_raises_when_no_squad_available(session):
    with pytest.raises(RuntimeError, match="No current squad"):
        payload_module.build_run_payload(session, team_id=12345)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'build_run_payload'`

- [ ] **Step 3: Implement**

In `scripts/site_export/payload.py`, replace the top import block with:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from dashboard.data.decisions import get_decision_history
from dashboard.data.squad import get_current_squad
from data.models import Gameweek
from projection.pipeline import _get_current_season, get_latest_projections

SCHEMA_VERSION = 1
```

Then append:

```python
def build_run_payload(db: Session, team_id: int) -> dict:
    squad_df = get_current_squad(db, team_id)
    if squad_df.empty:
        raise RuntimeError("No current squad found -- cannot export site data")

    gw = int(squad_df["gameweek"].iloc[0])
    season = _get_current_season()
    dist = get_projection_distributions(db, gw, season)

    projections_df = get_latest_projections(gw)
    team_names = _team_short_names(db)
    history_df = get_decision_history(db, limit_gws=20)

    return {
        "schema_version": SCHEMA_VERSION,
        "gameweek": gw,
        "label": _label_for_gw(db, season, gw),
        "generated_at": datetime.now(UTC).isoformat(),
        "squad": _build_squad_entries(squad_df, dist),
        "top15": _build_top15_entries(projections_df, dist, team_names),
        "history": _build_history_entries(history_df),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_payload.py -v`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/linus/Projects/FPL-26-27-bot
git add scripts/site_export/payload.py tests/test_site_export_payload.py
git commit -m "feat(site-export): assemble full export payload"
```

---

### Task 6: `writer.py` — write run file + maintain index

**Files:**
- Create: `scripts/site_export/writer.py`
- Create: `tests/test_site_export_writer.py`

**Interfaces:**
- Produces: `writer.write_run_file(out_dir: Path, gw: int, payload: dict) -> Path` (writes `out_dir/gw{gw}.json`, creates `out_dir` if needed); `writer.update_index(out_dir: Path, gw: int, label: str, generated_at: str) -> Path` (creates or updates `out_dir/index.json`, replacing any existing entry for that gw, kept sorted by gameweek descending).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_site_export_writer.py`:

```python
"""scripts/site_export/writer.py"""

from __future__ import annotations

import json

from scripts.site_export import writer


def test_write_run_file_creates_dir_and_file(tmp_path):
    out_dir = tmp_path / "simulations"
    payload = {"schema_version": 1, "gameweek": 3, "squad": []}

    path = writer.write_run_file(out_dir, gw=3, payload=payload)

    assert path == out_dir / "gw3.json"
    assert json.loads(path.read_text()) == payload


def test_update_index_creates_new_index_when_absent(tmp_path):
    out_dir = tmp_path / "simulations"

    path = writer.update_index(out_dir, gw=3, label="GW3 — 3 Aug", generated_at="2026-08-03T06:00:00Z")

    index = json.loads(path.read_text())
    assert index == {
        "schema_version": 1,
        "runs": [{"id": "gw3", "gameweek": 3, "label": "GW3 — 3 Aug", "generated_at": "2026-08-03T06:00:00Z"}],
    }


def test_update_index_replaces_existing_entry_for_same_gw(tmp_path):
    out_dir = tmp_path / "simulations"
    writer.update_index(out_dir, gw=3, label="GW3 — old label", generated_at="2026-08-03T06:00:00Z")

    path = writer.update_index(out_dir, gw=3, label="GW3 — 3 Aug", generated_at="2026-08-03T07:00:00Z")

    index = json.loads(path.read_text())
    assert len(index["runs"]) == 1
    assert index["runs"][0]["label"] == "GW3 — 3 Aug"


def test_update_index_keeps_runs_sorted_most_recent_gw_first(tmp_path):
    out_dir = tmp_path / "simulations"
    writer.update_index(out_dir, gw=2, label="GW2", generated_at="t2")
    writer.update_index(out_dir, gw=4, label="GW4", generated_at="t4")
    path = writer.update_index(out_dir, gw=3, label="GW3", generated_at="t3")

    index = json.loads(path.read_text())
    assert [r["gameweek"] for r in index["runs"]] == [4, 3, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.site_export.writer'`

- [ ] **Step 3: Implement**

Create `scripts/site_export/writer.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def write_run_file(out_dir: Path, gw: int, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / f"gw{gw}.json"
    run_path.write_text(json.dumps(payload, indent=2) + "\n")
    return run_path


def update_index(out_dir: Path, gw: int, label: str, generated_at: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"schema_version": 1, "runs": []}

    run_id = f"gw{gw}"
    index["runs"] = [r for r in index["runs"] if r["id"] != run_id]
    index["runs"].append({"id": run_id, "gameweek": gw, "label": label, "generated_at": generated_at})
    index["runs"].sort(key=lambda r: r["gameweek"], reverse=True)

    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_writer.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/linus/Projects/FPL-26-27-bot
git add scripts/site_export/writer.py tests/test_site_export_writer.py
git commit -m "feat(site-export): write run file and maintain index.json"
```

---

### Task 7: `git_sync.py` — scoped commit + push

**Files:**
- Create: `scripts/site_export/git_sync.py`
- Create: `tests/test_site_export_git_sync.py`

**Interfaces:**
- Produces: `git_sync.commit_and_push(repo_root: Path, data_dir: Path, message: str, push: bool = True) -> bool` — stages only `data_dir` (relative to `repo_root`), commits if there's anything staged, pushes if `push` is true. Returns `True` if a commit was made, `False` if there was nothing to commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_site_export_git_sync.py`:

```python
"""scripts/site_export/git_sync.py"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.site_export import git_sync


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("init\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


@pytest.fixture
def repo(tmp_path):
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    return repo_root


def test_commit_and_push_commits_new_data_without_pushing(repo):
    data_dir = repo / "data" / "simulations"
    data_dir.mkdir(parents=True)
    (data_dir / "gw3.json").write_text("{}")

    committed = git_sync.commit_and_push(repo, data_dir, "export: GW3 site data", push=False)

    assert committed is True
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"], cwd=repo, check=True, capture_output=True, text=True,
    )
    assert log.stdout.strip() == "export: GW3 site data"


def test_commit_and_push_returns_false_when_nothing_changed(repo):
    data_dir = repo / "data" / "simulations"
    data_dir.mkdir(parents=True)
    (data_dir / "gw3.json").write_text("{}")
    git_sync.commit_and_push(repo, data_dir, "export: GW3 site data", push=False)

    committed_again = git_sync.commit_and_push(repo, data_dir, "export: GW3 site data (rerun)", push=False)

    assert committed_again is False


def test_commit_and_push_pushes_to_remote_when_push_true(tmp_path, repo):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True)
    subprocess.run(["git", "push", "-q", "-u", "origin", "main"], cwd=repo, check=True)

    data_dir = repo / "data" / "simulations"
    data_dir.mkdir(parents=True)
    (data_dir / "gw3.json").write_text("{}")

    committed = git_sync.commit_and_push(repo, data_dir, "export: GW3 site data", push=True)

    assert committed is True
    remote_log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s", "main"], cwd=remote, check=True, capture_output=True, text=True,
    )
    assert remote_log.stdout.strip() == "export: GW3 site data"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_git_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.site_export.git_sync'`

- [ ] **Step 3: Implement**

Create `scripts/site_export/git_sync.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path


def commit_and_push(repo_root: Path, data_dir: Path, message: str, push: bool = True) -> bool:
    """Stage only data_dir, commit if there are staged changes, optionally push.
    Returns True if a commit was created, False if there was nothing to commit."""
    rel = data_dir.relative_to(repo_root)
    subprocess.run(["git", "add", str(rel)], cwd=repo_root, check=True)

    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_root)
    if staged.returncode == 0:
        return False

    subprocess.run(["git", "commit", "-m", message], cwd=repo_root, check=True)
    if push:
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_site_export_git_sync.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/linus/Projects/FPL-26-27-bot
git add scripts/site_export/git_sync.py tests/test_site_export_git_sync.py
git commit -m "feat(site-export): scoped git commit and push helper"
```

---

### Task 8: `scripts/export_site_data.py` CLI + README

**Files:**
- Create: `scripts/export_site_data.py`
- Create: `tests/test_export_site_data.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `scripts.site_export.payload.build_run_payload`, `scripts.site_export.writer.write_run_file`, `scripts.site_export.writer.update_index`, `scripts.site_export.git_sync.commit_and_push` (Tasks 5–7); `config.settings.settings.fpl_team_id`; `data.db.get_session`.
- Produces: `export_site_data.run(*, no_push: bool) -> None` (the testable core) and `export_site_data.main() -> None` (argparse wrapper, `--no-push` flag). This is the command the user runs by hand each gameweek.

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_site_data.py`:

```python
"""scripts/export_site_data.py"""

from __future__ import annotations

from unittest.mock import MagicMock

import scripts.export_site_data as cli


def test_run_wires_payload_write_index_and_commit_in_order(monkeypatch, tmp_path):
    calls = []
    fake_payload = {"gameweek": 3, "label": "GW3 — 3 Aug", "generated_at": "t"}

    def fake_write_run_file(out_dir, gw, payload):
        calls.append(("write", gw))
        path = tmp_path / f"gw{gw}.json"
        path.write_text("{}")
        return path

    monkeypatch.setattr(cli, "get_session", lambda: MagicMock())
    monkeypatch.setattr(
        cli, "build_run_payload",
        lambda db, team_id: (calls.append(("build", team_id)), fake_payload)[1],
    )
    monkeypatch.setattr(cli, "write_run_file", fake_write_run_file)
    monkeypatch.setattr(
        cli, "update_index",
        lambda out_dir, gw, label, generated_at: (calls.append(("index", gw)), tmp_path / "index.json")[1],
    )
    monkeypatch.setattr(
        cli, "commit_and_push",
        lambda repo_root, data_dir, message, push: (calls.append(("commit", push)), True)[1],
    )
    monkeypatch.setattr(cli.settings, "fpl_team_id", 99999)

    cli.run(no_push=True)

    assert calls == [("build", 99999), ("write", 3), ("index", 3), ("commit", False)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_export_site_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export_site_data'`

- [ ] **Step 3: Implement**

Create `scripts/export_site_data.py`:

```python
#!/usr/bin/env python
"""Export the current squad, top-15 projections, and decision history for
the portfolio site's $ fpl status panel. Run manually, after reviewing the
week's decision -- see README.md > Running > "Site data export"."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.settings import settings
from data.db import get_session
from scripts.site_export.git_sync import commit_and_push
from scripts.site_export.payload import build_run_payload
from scripts.site_export.writer import update_index, write_run_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "simulations"

logger = logging.getLogger(__name__)


def run(*, no_push: bool) -> None:
    db = get_session()
    try:
        payload = build_run_payload(db, settings.fpl_team_id)
    finally:
        db.close()

    gw = payload["gameweek"]
    run_path = write_run_file(DATA_DIR, gw, payload)
    logger.info("Wrote %s (%d bytes)", run_path, run_path.stat().st_size)

    index_path = update_index(DATA_DIR, gw, payload["label"], payload["generated_at"])
    logger.info("Updated %s", index_path)

    committed = commit_and_push(
        REPO_ROOT, DATA_DIR, f"export: GW{gw} site data", push=not no_push
    )
    if committed:
        logger.info("Committed%s", " (not pushed)" if no_push else " and pushed")
    else:
        logger.info("No changes to commit")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export current squad/projections/history for the portfolio site"
    )
    parser.add_argument(
        "--no-push", action="store_true", help="Write and commit locally without pushing"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(no_push=args.no_push)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python -m pytest tests/test_export_site_data.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Update README.md**

In `README.md`, find this block (in the "Running" section):

```markdown
**Backfill actual outcomes** (once a gameweek finishes, for the projected-vs-actual view in
the Decision History / Simulations dashboard pages):
```bash
uv run python scripts/backfill_decision_outcomes.py --season 2026-27
```

Backtest (walk-forward, retrains per GW):
```

Replace it with (adding the new subsection between the two existing ones):

```markdown
**Backfill actual outcomes** (once a gameweek finishes, for the projected-vs-actual view in
the Decision History / Simulations dashboard pages):
```bash
uv run python scripts/backfill_decision_outcomes.py --season 2026-27
```

**Site data export** (after reviewing/overruling a `run_agent.py --dry-run` decision, to publish
the current squad + top-15 xPts + transfer/chip history to the portfolio site's `$ fpl status`
panel — see `linus-j.github.io`'s own repo for the display side):
```bash
uv run python scripts/export_site_data.py            # writes, commits, and pushes
uv run python scripts/export_site_data.py --no-push   # writes + commits locally only, for review
```
Writes `data/simulations/gw{N}.json` and updates `data/simulations/index.json`. Not run
automatically by anything — a deliberate manual step, run once you're happy with the week's
decision (matches why the systemd timer below is disabled by default).

Backtest (walk-forward, retrains per GW):
```

- [ ] **Step 6: Commit**

```bash
cd /home/linus/Projects/FPL-26-27-bot
git add scripts/export_site_data.py tests/test_export_site_data.py README.md
git commit -m "feat(site-export): add export_site_data CLI entrypoint"
```

---

### Task 9: Manual smoke test against real local data

**Files:** none (verification only, no code changes)

**Interfaces:** none.

- [ ] **Step 1: Confirm local DB has a real decision to export**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python scripts/run_agent.py --dry-run`
Expected: exits 0 (or the benign pre-season `no_projections` exit 1 — if so, this smoke test can't proceed until a real squad/decision exists; skip to the next gameweek).

- [ ] **Step 2: Dry-run the export without pushing**

Run: `cd /home/linus/Projects/FPL-26-27-bot && uv run python scripts/export_site_data.py --no-push`
Expected: log lines `Wrote .../data/simulations/gw{N}.json (... bytes)`, `Updated .../data/simulations/index.json`, `Committed (not pushed)`.

- [ ] **Step 3: Inspect the written JSON**

Run: `cat data/simulations/gw*.json | uv run python -m json.tool | head -60`
Expected: valid JSON matching the schema in the design doc — `schema_version`, `gameweek`, `label`, `generated_at`, `squad` (15 entries), `top15` (15 entries), `history`. File size should be a few KB, not MB (`ls -la data/simulations/`).

- [ ] **Step 4: Confirm the commit landed locally and push for real**

Run: `cd /home/linus/Projects/FPL-26-27-bot && git log -1 --stat`
Expected: shows the `export: GW{N} site data` commit touching only `data/simulations/**`.

Run: `cd /home/linus/Projects/FPL-26-27-bot && git push`
Expected: pushes cleanly to `origin`. (From here on, `export_site_data.py` without `--no-push` does this automatically — this manual push is only needed because Step 2 used `--no-push` for inspection first.)
