# Cold-start fixture lookahead + manual transfer/rumour overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the GW1 cold-start squad builder a fixture-difficulty-weighted multi-gameweek lookahead (instead of pure single-GW xPts), and add a hand-edited, version-controlled mechanism to correct a player's club ahead of FPL's own API and to flag/discount rumoured departures — both fed through the one shared candidate-pool-loading seam so cold-start and in-season (including the January window) get identical treatment.

**Architecture:** Feature A reuses the already-tested `fixture_multiplier` and the optimiser's existing horizon-summing (`_multi_gw_xpts`/`_multi_gw_var`) unchanged; `project_cold_start` gains a `horizon` parameter and emits one row per `(player, gw)`, resolving each GW's opponent-defence strength from the fixtures table with a prior-season fallback keyed on the stable team `code`. Feature B adds a new `data/overrides.py` module reading a hand-edited YAML file, wired into the two candidate-pool loaders (`cold_start.load_current_players`, `decision_engine._load_players`) for team-id correction, and into the already-existing-but-unfed `optimiser/departure_risk.py::apply_departure_discount` for rumour discounting.

**Tech Stack:** Python 3.12, pandas, SQLAlchemy (SQLite), PyYAML, pytest.

**Reference spec:** `docs/superpowers/specs/2026-08-10-cold-start-lookahead-and-transfer-overrides-design.md` (approved).

## Global Constraints

- Python env: use `.venv/bin/python` for every command (system Python lacks project deps). Tests: `.venv/bin/python -m pytest <path> -v`. Lint: `.venv/bin/ruff check <path>`.
- `ruff` line-length = 100, target py312 (pyproject.toml).
- Never crash on missing/malformed override data — every new loader degrades to empty/no-op and logs a `warning` instead.
- `code` (not `id`/`team_id`) is the only stable cross-season/cross-transfer identifier — every join in this plan that needs to survive a transfer or a season boundary must key on `code`.
- Backward compatibility: `project_cold_start`'s default `horizon=1` must reproduce today's exact single-row-per-player output — every existing test in `tests/test_cold_start.py` must keep passing unmodified.
- Follow this repo's existing docstring convention (explain the *why*, especially past bugs/design decisions) rather than a terser default — match the style already in `cold_start.py`/`departure_risk.py`.

---

### Task 1: `data/overrides.py` — manual override loaders

**Files:**
- Create: `config/transfer_overrides.yaml`
- Create: `data/overrides.py`
- Modify: `pyproject.toml` (add `PyYAML` dependency)
- Test: `tests/test_overrides.py`

**Interfaces:**
- Produces: `load_team_overrides() -> dict[int, int]` (code → team_id)
- Produces: `apply_team_overrides(players: pd.DataFrame) -> pd.DataFrame` (copy with corrected `team_id`)
- Produces: `load_rumoured_overrides() -> dict[int, dict]` (player_id → `{"p_leave": float, "reason": str, "as_of": str}`)
- Produces: `load_p_leave_overrides() -> dict[int, float]` (player_id → p_leave, the shape `optimiser.departure_risk.apply_departure_discount` consumes)
- Produces: `log_rumoured_squad_members(squad_ids: list[int], players: pd.DataFrame) -> None`
- Consumes: `data.db.get_session` (existing), the `players` table's `code`/`id` columns (existing schema).

- [ ] **Step 1: Add the PyYAML dependency**

Edit `pyproject.toml` — add to the `dependencies` list (it's currently only a transitive dep, so `import yaml` in a new module needs it declared explicitly):

```toml
    # Config & validation
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "python-dotenv>=1.0.0",
    "PyYAML>=6.0.0",
```

Run: `.venv/bin/python -c "import yaml; print(yaml.__version__)"`
Expected: prints a version (e.g. `6.0.3`) — already installed transitively, this just makes the dependency explicit.

- [ ] **Step 2: Create the YAML scaffold**

Write `config/transfer_overrides.yaml`:

```yaml
# Manual transfer/rumour overrides (hand-edited, version-controlled).
# See docs/superpowers/specs/2026-08-10-cold-start-lookahead-and-transfer-overrides-design.md
#
# `code` is the player's/team's stable FPL `code` (NOT `id` -- `id` is
# reassigned per season and doesn't survive a transfer; `code` does).
#
# confirmed: a signing/move you know about that FPL's own team_id hasn't
# caught up to yet. Corrects team_id for squad-building (max-3-per-club,
# fixture lookahead) until FPL's bootstrap data itself updates.
#   - code: 123456
#     team_id: 1
#     reason: "Signed from Newcastle, not yet reflected in FPL team_id"
#     as_of: "2026-08-10"
#
# rumoured: a not-yet-confirmed departure. Discounts (does not exclude) the
# player's projected points by (1 - p_leave), and logs a warning if the
# optimiser still picks them into a squad.
#   - code: 234567
#     p_leave: 0.35
#     reason: "Strongly linked to a January move per <source>"
#     as_of: "2026-08-10"

confirmed: []

rumoured: []
```

- [ ] **Step 3: Write the failing tests**

Write `tests/test_overrides.py`:

```python
"""overrides.py — manual transfer/rumour corrections (plan: cold-start
fixture lookahead + transfer overrides, 2026-08-10). Every loader must
degrade safely (empty/no-op, warning log) on a missing file, empty file, or
unmatched code -- never crash the pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data import overrides as ov
from data.models import Base, Player


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'overrides.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(ov, "get_session", lambda: Local())
    return Local


@pytest.fixture
def overrides_file(tmp_path, monkeypatch):
    path = tmp_path / "transfer_overrides.yaml"
    monkeypatch.setattr(ov, "OVERRIDES_PATH", path)
    return path


def _write(path, data: dict):
    path.write_text(yaml.safe_dump(data))


def test_load_team_overrides_missing_file_is_empty(overrides_file):
    assert ov.load_team_overrides() == {}


def test_load_team_overrides_reads_confirmed_list(overrides_file):
    _write(overrides_file, {"confirmed": [
        {"code": 111, "team_id": 5, "reason": "x", "as_of": "2026-08-10"},
    ]})
    assert ov.load_team_overrides() == {111: 5}


def test_apply_team_overrides_replaces_matched_code(overrides_file):
    _write(overrides_file, {"confirmed": [{"code": 111, "team_id": 5}]})
    players = pd.DataFrame([
        {"id": 1, "code": 111, "team_id": 1, "web_name": "Moved"},
        {"id": 2, "code": 222, "team_id": 2, "web_name": "Unaffected"},
    ])
    out = ov.apply_team_overrides(players)
    assert out.loc[out["code"] == 111, "team_id"].iloc[0] == 5
    assert out.loc[out["code"] == 222, "team_id"].iloc[0] == 2
    # original untouched (returns a copy)
    assert players.loc[players["code"] == 111, "team_id"].iloc[0] == 1


def test_apply_team_overrides_noop_when_file_missing(overrides_file):
    players = pd.DataFrame([{"id": 1, "code": 111, "team_id": 1}])
    out = ov.apply_team_overrides(players)
    pd.testing.assert_frame_equal(out, players)


def test_apply_team_overrides_noop_when_no_code_column(overrides_file):
    _write(overrides_file, {"confirmed": [{"code": 111, "team_id": 5}]})
    players = pd.DataFrame([{"id": 1, "team_id": 1}])
    out = ov.apply_team_overrides(players)
    pd.testing.assert_frame_equal(out, players)


def test_load_p_leave_overrides_resolves_code_to_player_id(temp_session, overrides_file):
    s = temp_session()
    try:
        s.add(Player(id=42, fpl_id=42, code=999, first_name="A", second_name="B",
                     web_name="Rumoured", team_id=1, position="MID", now_cost=8.0))
        s.commit()
    finally:
        s.close()
    _write(overrides_file, {"rumoured": [
        {"code": 999, "p_leave": 0.4, "reason": "linked", "as_of": "2026-08-10"},
    ]})
    assert ov.load_p_leave_overrides() == {42: 0.4}


def test_load_p_leave_overrides_skips_unmatched_code_without_crashing(
    temp_session, overrides_file, caplog
):
    temp_session()  # ensure players table exists, empty
    _write(overrides_file, {"rumoured": [{"code": 999, "p_leave": 0.4}]})
    with caplog.at_level(logging.WARNING):
        result = ov.load_p_leave_overrides()
    assert result == {}
    assert "999" in caplog.text


def test_load_rumoured_overrides_missing_file_is_empty(overrides_file):
    assert ov.load_rumoured_overrides() == {}


def test_log_rumoured_squad_members_logs_matched_squad_member(
    temp_session, overrides_file, caplog
):
    s = temp_session()
    try:
        s.add(Player(id=42, fpl_id=42, code=999, first_name="A", second_name="B",
                     web_name="Rumoured", team_id=1, position="MID", now_cost=8.0))
        s.commit()
    finally:
        s.close()
    _write(overrides_file, {"rumoured": [
        {"code": 999, "p_leave": 0.4, "reason": "linked", "as_of": "2026-08-10"},
    ]})
    players = pd.DataFrame([{"id": 42, "web_name": "Rumoured"}, {"id": 7, "web_name": "Other"}])
    with caplog.at_level(logging.WARNING):
        ov.log_rumoured_squad_members([42, 7], players)
    assert "Rumoured" in caplog.text
    assert "0.40" in caplog.text


def test_log_rumoured_squad_members_noop_when_no_rumours(overrides_file, caplog):
    players = pd.DataFrame([{"id": 1, "web_name": "X"}])
    with caplog.at_level(logging.WARNING):
        ov.log_rumoured_squad_members([1], players)
    assert caplog.text == ""


def test_apply_team_overrides_result_respected_by_max_players_per_club_constraint(overrides_file):
    """Integration proof (spec's explicit testing requirement): the
    override must be visible to optimise_squad's own max-3-per-club
    constraint, not just a cosmetic column change nothing downstream reads.

    Team 1 already has 3 members (id1 GKP, id2 DEF, id3 MID) filling the
    cap. Player code=999 (id4, a MID) starts on team 2, where nothing
    competes with them -- overriding their code to team_id=1 forces the
    solver to choose between id3 (score 10) and id4 (score 6) for team 1's
    now-contested 3rd slot, since keeping both would push team 1 to 4
    members. If the override were a no-op, id4 would stay attributed to
    team 2 and would certainly be picked (nothing there outscores them);
    the assertion that id4 is EXCLUDED only holds if the override actually
    changed what the ILP sees."""
    from config.strategy import SQUAD
    from optimiser.squad import optimise_squad

    _write(overrides_file, {"confirmed": [{"code": 999, "team_id": 1}]})

    rows = [
        {"id": 1, "code": 1, "position": "GKP", "now_cost": 4.5, "team_id": 1,
         "status": "a", "web_name": "gk1"},
        {"id": 2, "code": 2, "position": "DEF", "now_cost": 4.5, "team_id": 1,
         "status": "a", "web_name": "def1"},
        {"id": 3, "code": 3, "position": "MID", "now_cost": 4.5, "team_id": 1,
         "status": "a", "web_name": "mid1"},
        {"id": 4, "code": 999, "position": "MID", "now_cost": 4.5, "team_id": 2,
         "status": "a", "web_name": "target"},
    ]
    # Fill out a legal 16-candidate pool (one MORE than squad_size needs per
    # position, in MID specifically -- see below) across teams 3-9, cheap
    # and unremarkable so the solver has no reason to prefer them on merit
    # alone, only when a hard constraint forces a choice.
    fillers = ["GKP"] * 1 + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 3
    next_id = 5
    for i, pos in enumerate(fillers):
        rows.append({
            "id": next_id, "code": next_id, "position": pos, "now_cost": 4.5,
            "team_id": 3 + (i % 7), "status": "a", "web_name": f"filler{next_id}",
        })
        next_id += 1
    # Pool per position: GKP=2 (exact fit, both forced), DEF=5 (exact fit,
    # all forced), FWD=3 (exact fit, all forced) -- MID=6 candidates for 5
    # slots (id3=10, id4=6, four fillers=2 each) is the ONLY position with
    # slack, which is what lets the solver drop id4 instead of being forced
    # into an infeasible 4-player team 1.

    players = pd.DataFrame(rows)
    players = ov.apply_team_overrides(players)
    assert players.loc[players["code"] == 999, "team_id"].iloc[0] == 1  # override took effect
    players["start_probability"] = 1.0

    proj_rows = [
        {"player_id": r.id, "gameweek": 1,
         "xpts": 10.0 if r.id in (1, 2, 3) else (6.0 if r.id == 4 else 2.0)}
        for r in players.itertuples()
    ]
    projections = pd.DataFrame(proj_rows)

    solution = optimise_squad(projections=projections, players=players, budget=100.0, horizon=1)

    team_counts = solution.squad["team_id"].value_counts()
    assert team_counts.get(1, 0) <= SQUAD.max_players_per_club
    assert 4 not in solution.squad["id"].tolist()  # target excluded by the (corrected) team cap
    assert 3 in solution.squad["id"].tolist()  # the higher-scoring incumbent wins the contested slot
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_overrides.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.overrides'`

- [ ] **Step 5: Implement `data/overrides.py`**

```python
"""overrides.py — manual transfer/rumour corrections (plan: cold-start
fixture lookahead + transfer overrides, 2026-08-10).

FPL's own team_id is trusted unconditionally with no correction mechanism
(see docs/superpowers/specs/2026-08-10-cold-start-lookahead-and-transfer-
overrides-design.md). config/transfer_overrides.yaml is a hand-edited,
version-controlled file the user updates when they know something FPL's
API hasn't caught up on yet -- a confirmed summer signing not yet
reflected in team_id, or a rumoured departure worth discounting. Every
loader here degrades safely to empty/no-op on a missing file, an empty
file, or an unmatched code -- a wrong automatic correction is a worse
failure mode than a missed one, so nothing here ever crashes the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import text

from data.db import get_session

logger = logging.getLogger(__name__)

OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "config" / "transfer_overrides.yaml"


def _load_yaml() -> dict:
    if not OVERRIDES_PATH.exists():
        return {}
    with OVERRIDES_PATH.open() as f:
        data = yaml.safe_load(f)
    return data or {}


def load_team_overrides() -> dict[int, int]:
    """code -> corrected team_id, from the `confirmed` list."""
    entries = _load_yaml().get("confirmed") or []
    return {int(e["code"]): int(e["team_id"]) for e in entries}


def apply_team_overrides(players: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of ``players`` with ``team_id`` replaced wherever
    ``players.code`` matches a `confirmed` override entry. A no-op (plain
    copy) when the file is missing/empty or ``players`` has no `code`
    column."""
    out = players.copy()
    overrides = load_team_overrides()
    if not overrides or "code" not in out.columns:
        return out
    mask = out["code"].isin(overrides.keys())
    out.loc[mask, "team_id"] = out.loc[mask, "code"].map(overrides)
    return out


def _code_to_player_id() -> dict[int, int]:
    db = get_session()
    try:
        rows = db.execute(text("SELECT code, id FROM players WHERE code IS NOT NULL")).fetchall()
        return {int(code): int(pid) for code, pid in rows}
    finally:
        db.close()


def load_rumoured_overrides() -> dict[int, dict]:
    """player_id -> {p_leave, reason, as_of}, from the `rumoured` list,
    resolved via the current players table's `code`. A `code` with no
    matching current player is skipped (logged at warning, never crashes --
    e.g. a rumoured entry left in the file after the player actually left)."""
    entries = _load_yaml().get("rumoured") or []
    if not entries:
        return {}
    code_to_pid = _code_to_player_id()
    result: dict[int, dict] = {}
    for entry in entries:
        code = int(entry["code"])
        pid = code_to_pid.get(code)
        if pid is None:
            logger.warning(
                "transfer_overrides.yaml: rumoured code %s has no matching "
                "current player, skipping",
                code,
            )
            continue
        result[pid] = {
            "p_leave": float(entry["p_leave"]),
            "reason": entry.get("reason", ""),
            "as_of": entry.get("as_of", ""),
        }
    return result


def load_p_leave_overrides() -> dict[int, float]:
    """player_id -> p_leave, the plain-float shape
    ``optimiser.departure_risk.apply_departure_discount`` consumes."""
    return {pid: entry["p_leave"] for pid, entry in load_rumoured_overrides().items()}


def log_rumoured_squad_members(squad_ids: list[int], players: pd.DataFrame) -> None:
    """Logs a warning naming the player + reason/as_of for every squad
    member present in the `rumoured` list. Deliberately log-only for this
    pass -- dashboard surfacing is a follow-up, not blocking."""
    details = load_rumoured_overrides()
    if not details:
        return
    name_by_id = (
        players.set_index("id")["web_name"].to_dict() if "web_name" in players.columns else {}
    )
    for pid in squad_ids:
        entry = details.get(pid)
        if entry is None:
            continue
        logger.warning(
            "Squad includes rumoured departure: %s (p_leave=%.2f) — %s (as_of %s)",
            name_by_id.get(pid, pid), entry["p_leave"], entry["reason"], entry["as_of"],
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_overrides.py -v`
Expected: all PASS

- [ ] **Step 7: Lint and commit**

Run: `.venv/bin/ruff check data/overrides.py tests/test_overrides.py`
Expected: no errors

```bash
git add pyproject.toml config/transfer_overrides.yaml data/overrides.py tests/test_overrides.py
git commit -m "feat: manual transfer/rumour override loaders (data/overrides.py)"
```

---

### Task 2: Wire team-id overrides into the two candidate-pool loaders

**Files:**
- Modify: `projection/cold_start.py` (`load_current_players`, lines 161-171)
- Modify: `agent/decision_engine.py` (`_load_players`, lines 29-40)
- Test: `tests/test_cold_start.py` (append)
- Test: `tests/test_decision_engine_load_players.py` (new)

**Interfaces:**
- Consumes: `data.overrides.apply_team_overrides(players: pd.DataFrame) -> pd.DataFrame` (Task 1)
- Produces: both loaders now return an already team-id-corrected `DataFrame` — every downstream consumer (cold-start, live squad building, transfers) sees the corrected club with no further change needed.

- [ ] **Step 1: Write the failing test for `cold_start.load_current_players`**

Append to `tests/test_cold_start.py` (after the existing imports, no new imports needed — `monkeypatch` and `pd` are already used in this file):

```python
def test_load_current_players_applies_team_overrides(temp_session, monkeypatch, tmp_path):
    import yaml

    from data import overrides as ov

    s = temp_session()
    try:
        s.add(Player(fpl_id=1, code=111, first_name="A", second_name="A", web_name="Moved",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    overrides_path = tmp_path / "transfer_overrides.yaml"
    overrides_path.write_text(yaml.safe_dump({"confirmed": [{"code": 111, "team_id": 99}]}))
    monkeypatch.setattr(ov, "OVERRIDES_PATH", overrides_path)

    players = cs.load_current_players()
    assert players.loc[players["web_name"] == "Moved", "team_id"].iloc[0] == 99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py::test_load_current_players_applies_team_overrides -v`
Expected: FAIL — `team_id` is still `1`, not `99` (override not applied yet).

- [ ] **Step 3: Wire `apply_team_overrides` into `load_current_players`**

In `projection/cold_start.py`, add the import near the top (with the other `projection`/`data` imports, after the existing `from data.db import get_session`):

```python
from data.overrides import apply_team_overrides
```

Then change `load_current_players` (currently lines 161-171):

```python
def load_current_players() -> pd.DataFrame:
    """Candidate universe for the initial squad: the current bootstrap
    players, with any manual team_id correction (Feature B, plan 2026-08-10)
    already applied -- a confirmed transfer FPL hasn't updated team_id for
    yet is visible to the max-3-per-club constraint and fixture lookahead
    from here on."""
    db = get_session()
    try:
        query = text("""
            SELECT id, code, web_name, position, now_cost, status, team_id
            FROM players
        """)
        players = pd.read_sql(query, db.bind)
    finally:
        db.close()
    return apply_team_overrides(players)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py::test_load_current_players_applies_team_overrides -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `decision_engine._load_players`**

Write `tests/test_decision_engine_load_players.py`:

```python
"""_load_players (agent/decision_engine.py) is one of the two live
candidate-pool loaders (the other is projection/cold_start.py::
load_current_players) that must apply the manual team_id override (Feature
B, plan 2026-08-10) -- this is the single shared seam serving both
cold-start and in-season transfer decisions."""

from __future__ import annotations

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent import decision_engine as de
from data import overrides as ov
from data.models import Base, Player


def test_load_players_applies_team_overrides(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'de.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(de, "get_session", lambda: Local())

    s = Local()
    try:
        s.add(Player(fpl_id=1, code=111, first_name="A", second_name="A", web_name="Moved",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    overrides_path = tmp_path / "transfer_overrides.yaml"
    overrides_path.write_text(yaml.safe_dump({"confirmed": [{"code": 111, "team_id": 99}]}))
    monkeypatch.setattr(ov, "OVERRIDES_PATH", overrides_path)

    players = de._load_players()
    assert players.loc[players["web_name"] == "Moved", "team_id"].iloc[0] == 99
    assert "code" in players.columns
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_decision_engine_load_players.py -v`
Expected: FAIL — `AssertionError` (`code` column missing / team_id still 1).

- [ ] **Step 7: Wire `apply_team_overrides` into `_load_players`**

In `agent/decision_engine.py`, add to the imports block (near the top, after `from data.db import get_session`):

```python
from data.overrides import apply_team_overrides
```

Then change `_load_players` (currently lines 29-40):

```python
def _load_players() -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT id, code, fpl_id, web_name, position, team_id, now_cost,
                   status, chance_of_playing_next_round, selected_by_percent,
                   form, ict_index, influence, creativity, threat
            FROM players
        """)
        players = pd.read_sql(query, db.bind)
    finally:
        db.close()
    return apply_team_overrides(players)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_decision_engine_load_players.py -v`
Expected: PASS

- [ ] **Step 9: Run the full existing decision-engine and cold-start suites to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py tests/test_decision_engine_sim_storage.py tests/test_simulation_engine.py tests/test_simulation_personas.py -v`
Expected: all PASS (adding a `code` column to `_load_players`'s SELECT must not break anything reading that DataFrame elsewhere).

- [ ] **Step 10: Lint and commit**

Run: `.venv/bin/ruff check projection/cold_start.py agent/decision_engine.py tests/test_decision_engine_load_players.py`
Expected: no errors

```bash
git add projection/cold_start.py agent/decision_engine.py tests/test_cold_start.py tests/test_decision_engine_load_players.py
git commit -m "feat: apply manual team_id overrides in both candidate-pool loaders"
```

---

### Task 3: Wire rumour discount + flagging into the decision loop

**Files:**
- Modify: `agent/decision_engine.py` (`_run_decision_cycle`)
- Modify: `projection/cold_start.py` (`build_initial_squad`, lines 369-419)
- Test: `tests/test_cold_start.py` (append)
- Test: `tests/test_decision_engine_sim_storage.py` or new focused test file (see Step 5)

**Interfaces:**
- Consumes: `data.overrides.load_p_leave_overrides() -> dict[int, float]`, `data.overrides.log_rumoured_squad_members(squad_ids, players) -> None` (Task 1); `optimiser.departure_risk.apply_departure_discount(projections, p_leave_by_player, rules=DEPARTURE_RISK) -> pd.DataFrame` (existing, unchanged).
- Produces: every `projections` DataFrame handed to `optimise_squad`/`optimise_starting_xi`/`evaluate_transfers` from here on has rumour-tier discounting applied; every built squad gets rumour members logged.

- [ ] **Step 1: Write the failing test for `build_initial_squad`'s discount wiring**

Append to `tests/test_cold_start.py`:

```python
def test_build_initial_squad_discounts_rumoured_player(temp_session, monkeypatch, tmp_path):
    import yaml

    from data import overrides as ov

    _seed_full_pool(temp_session)
    s = temp_session()
    try:
        rumoured_id = s.query(Player.id).filter_by(fpl_id=2).scalar()  # a non-leaver "p1"
        rumoured_code = s.query(Player.code).filter_by(fpl_id=2).scalar()
        injected = pd.read_sql(
            "SELECT id, code, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    overrides_path = tmp_path / "transfer_overrides.yaml"
    overrides_path.write_text(yaml.safe_dump(
        {"rumoured": [{"code": rumoured_code, "p_leave": 0.9, "reason": "x", "as_of": "2026-08-10"}]}
    ))
    monkeypatch.setattr(ov, "OVERRIDES_PATH", overrides_path)

    _, projections = cs.build_initial_squad("2026-27", players=injected)
    row = projections[projections["player_id"] == rumoured_id]
    assert (row["xpts"] == 0.0).all()  # p_leave=0.9 -> stay-probability multiplier 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py::test_build_initial_squad_discounts_rumoured_player -v`
Expected: FAIL — `xpts` is not `0.0` (discount not applied yet).

- [ ] **Step 3: Wire the discount + rumour logging into `build_initial_squad`**

In `projection/cold_start.py`, `build_initial_squad` currently (lines 369-419) ends with:

```python
    from config.strategy import SQUAD
    from optimiser.squad import optimise_squad

    budget = SQUAD.budget_total if budget is None else budget
    if players is None:
        players = load_current_players()
    players = apply_departure_gate(players)
    prior_season = prior_season_of(season)
    prior = load_prior_season_features(prior_season)
    raw_appearances = load_prior_season_appearances(prior_season)
    prior_league_lookup = load_prior_league_lookup(season)
    projections = project_cold_start(
        players, prior, raw_appearances=raw_appearances,
        prior_league_lookup=prior_league_lookup,
    )

    players = players.merge(
        projections[["player_id", "start_probability"]],
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    solution = optimise_squad(
        projections=projections, players=players, budget=budget, horizon=1, season=season,
        config=config,
    )
    return solution, projections
```

Replace with:

```python
    from config.strategy import SQUAD
    from data.overrides import load_p_leave_overrides, log_rumoured_squad_members
    from optimiser.departure_risk import apply_departure_discount
    from optimiser.squad import optimise_squad

    budget = SQUAD.budget_total if budget is None else budget
    if players is None:
        players = load_current_players()
    players = apply_departure_gate(players)
    prior_season = prior_season_of(season)
    prior = load_prior_season_features(prior_season)
    raw_appearances = load_prior_season_appearances(prior_season)
    prior_league_lookup = load_prior_league_lookup(season)
    projections = project_cold_start(
        players, prior, raw_appearances=raw_appearances,
        prior_league_lookup=prior_league_lookup,
    )
    # Feature B (plan 2026-08-10): the rumour-discount tier of the
    # already-existing departure-risk gate, fed with real data for the
    # first time -- previously always an empty dict (Phase 4's news layer
    # was never built), so this call was always a no-op before today.
    projections = apply_departure_discount(projections, load_p_leave_overrides())

    players = players.merge(
        projections[["player_id", "start_probability"]],
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    solution = optimise_squad(
        projections=projections, players=players, budget=budget, horizon=1, season=season,
        config=config,
    )
    log_rumoured_squad_members(solution.squad["id"].tolist(), players)
    return solution, projections
```

(Note: the `horizon=1` here is unchanged in this task — Task 7 changes it to `cfg.cold_start_lookahead_gws`. Keep this task's diff scoped to the discount/logging wiring only.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py::test_build_initial_squad_discounts_rumoured_player -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for the live decision-cycle wiring**

Write `tests/test_decision_engine_departure_discount.py`:

```python
"""_run_decision_cycle's live path must apply the rumour-discount tier
(Feature B, plan 2026-08-10) to whatever projections it reads, and must log
a warning for any rumoured player that still makes the final squad."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from agent import decision_engine as de


def test_run_decision_cycle_applies_p_leave_discount_to_projections(monkeypatch):
    captured = {}

    def _fake_apply_departure_discount(projections, p_leave_by_player, rules=None):
        captured["p_leave_by_player"] = p_leave_by_player
        captured["called"] = True
        return projections

    monkeypatch.setattr(de, "load_p_leave_overrides", lambda: {7: 0.5})
    monkeypatch.setattr(de, "apply_departure_discount", _fake_apply_departure_discount)
    monkeypatch.setattr(de, "get_latest_projections", lambda: pd.DataFrame([
        {"player_id": 7, "gameweek": 1, "xpts": 5.0, "start_probability": 0.9},
    ]))
    monkeypatch.setattr(de, "_get_current_and_next_gw", lambda: (1, 1))
    monkeypatch.setattr(de, "_load_squad_state", lambda *a, **k: ([], 100.0, 1))

    # Real squad-building is out of scope for this unit test -- short-circuit
    # once the discount call itself has been observed.
    class _Stop(Exception):
        pass

    def _boom(*a, **k):
        raise _Stop()

    monkeypatch.setattr(de, "_load_players", _boom)

    with pytest.raises(_Stop):
        de._run_decision_cycle(
            season="2026-27", dry_run=True, force_chip=None,
            config=de.OPTIMISER, chip_timing=de.CHIP_TIMING,
            team_id=None, sim_manager_id=None, refresh_projections=False,
        )

    assert captured.get("called") is True
    assert captured["p_leave_by_player"] == {7: 0.5}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_decision_engine_departure_discount.py -v`
Expected: FAIL — `apply_departure_discount`/`load_p_leave_overrides` are not yet imported into `agent/decision_engine.py`, so `de.load_p_leave_overrides`/`de.apply_departure_discount` raise `AttributeError` when monkeypatched (or the discount is never called).

- [ ] **Step 7: Wire the discount + rumour logging into `_run_decision_cycle`**

In `agent/decision_engine.py`, add to the imports (with the other `from data.overrides import ...` line added in Task 2 — extend it):

```python
from data.overrides import apply_team_overrides, load_p_leave_overrides, log_rumoured_squad_members
from optimiser.departure_risk import apply_departure_discount
```

In `_run_decision_cycle`, immediately after `projections = get_latest_projections()` (currently line 177), add:

```python
    projections = get_latest_projections()
    # Feature B (plan 2026-08-10): rumour-discount tier, real data for the
    # first time (previously always an empty dict).
    projections = apply_departure_discount(projections, load_p_leave_overrides())
```

Near the end of the live (non-cold-start) branch, after `squad_solution`/`dgw_coverage` are computed and before the `result = {...}` dict is built (currently right before line 352), add:

```python
    log_rumoured_squad_members(squad_solution.squad["id"].tolist(), players)

    result = {
```

(The cold-start branch needs no equivalent call here — `cold_start.build_initial_squad` already logs internally, from Task 3 Step 3.)

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_decision_engine_departure_discount.py -v`
Expected: PASS

- [ ] **Step 9: Run the full decision-engine and transfers suites to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py tests/test_transfers.py tests/test_departure_risk.py tests/test_decision_engine_sim_storage.py tests/test_simulation_engine.py tests/test_simulation_personas.py tests/test_p_xi_harness.py -v`
Expected: all PASS

- [ ] **Step 10: Lint and commit**

Run: `.venv/bin/ruff check agent/decision_engine.py projection/cold_start.py tests/test_decision_engine_departure_discount.py`
Expected: no errors

```bash
git add agent/decision_engine.py projection/cold_start.py tests/test_cold_start.py tests/test_decision_engine_departure_discount.py
git commit -m "feat: wire rumour-discount tier and squad rumour-flagging into the live decision loop"
```

---

### Task 4: `cold_start_lookahead_gws` config field

**Files:**
- Modify: `config/strategy.py` (`OptimiserConfig`, lines 318-401)
- Test: `tests/test_scoring_rules.py` (append) — check this is the right home for config-field tests first (Step 1).

**Interfaces:**
- Produces: `OPTIMISER.cold_start_lookahead_gws: int` (default `5`), and `OptimiserConfig(cold_start_lookahead_gws=...)` overridable per-call like every other field on this dataclass.

- [ ] **Step 1: Confirm the test home**

Run: `grep -n "class TestOptimiserConfig\|def test_optimiser_config\|OptimiserConfig(" tests/test_scoring_rules.py`

If `OptimiserConfig` field defaults are already tested there, add to that file. Otherwise create `tests/test_strategy_config.py` — use whichever the grep shows is the existing convention.

- [ ] **Step 2: Write the failing test**

Add (to whichever file Step 1 identified):

```python
def test_cold_start_lookahead_gws_default():
    from config.strategy import OptimiserConfig

    assert OptimiserConfig().cold_start_lookahead_gws == 5


def test_cold_start_lookahead_gws_overridable():
    from config.strategy import OptimiserConfig

    assert OptimiserConfig(cold_start_lookahead_gws=1).cold_start_lookahead_gws == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -k cold_start_lookahead_gws -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'cold_start_lookahead_gws'` (or `AttributeError`).

- [ ] **Step 4: Add the field**

In `config/strategy.py`, `OptimiserConfig` (lines 318-401), immediately after the `transfer_planning_horizon_gws` field (lines 320-321):

```python
    # Number of GWs to project ahead for transfer decisions
    transfer_planning_horizon_gws: int = 3

    # GWs to look ahead when building the GW1/pre-season initial squad
    # (fixture-difficulty-weighted, not just single-GW xPts) -- a distinct
    # knob from transfer_planning_horizon_gws since cold start is a one-shot
    # squad build with no in-season transfer plan to horizon-limit
    # (2026-08-10, plan/cold-start-lookahead-and-transfer-overrides -- the
    # user's own example: "It is why so many managers still have Haaland
    # despite the price since the fixtures are so good").
    cold_start_lookahead_gws: int = 5
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -k cold_start_lookahead_gws -v`
Expected: PASS

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/ruff check config/strategy.py`
Expected: no errors

```bash
git add config/strategy.py tests/test_scoring_rules.py
git commit -m "feat: add cold_start_lookahead_gws config field (default 5)"
```

(If Step 1 created a new test file instead, `git add` that path instead of `tests/test_scoring_rules.py`.)

---

### Task 5: `load_horizon_fixtures` — per-GW opponent/defence-strength resolution

**Files:**
- Modify: `projection/cold_start.py` (add new functions after `load_prior_league_lookup`, before `apply_departure_gate`)
- Test: `tests/test_cold_start.py` (append)

**Interfaces:**
- Consumes: `projection.fixture_adjust.fixture_multiplier` (existing, not called directly by this task — only by Task 6); `data.db.get_session` (existing); `data.models.Fixture`/`TeamSeasonStrength` schema (existing, read-only).
- Produces:
  - `load_current_defence_strength(season: str) -> dict[int, float]` (team_id → avg defence strength, only non-zero rows)
  - `load_team_codes(season: str) -> dict[int, int]` (team_id → code, only rows with a code)
  - `load_prior_defence_strength_by_code(prior_season: str) -> dict[int, float]` (code → avg defence strength, only non-zero rows)
  - `load_horizon_fixtures(players: pd.DataFrame, season: str, target_gw: int, horizon: int) -> pd.DataFrame` with columns `["player_id", "gameweek", "opp_defence_strength", "was_home"]` — `opp_defence_strength` is `float | None` (`None` when unresolvable, matching `fixture_multiplier`'s existing neutral-fallback contract).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cold_start.py` — add these imports at the top of the file alongside the existing ones:

```python
from data.models import Fixture, Team, TeamSeasonStrength
```

(the file already imports `Base, Player, PlayerGameweekStats, PriorLeagueStats` from `data.models` — extend that import line to include `Fixture, Team, TeamSeasonStrength` instead of adding a second import line).

```python
def _seed_two_team_fixture(Local, season="2026-27", gw=1, def_home=1000.0, def_away=1400.0):
    """Team 1 (weak defence, 1000) hosts Team 2 (strong defence, 1400) at
    ``gw`` for ``season``. Player p1 is on Team 1 (an easy home fixture vs a
    weak defence); p2 is on Team 2 (a hard away fixture vs a strong one)."""
    s = Local()
    try:
        s.add_all([
            Team(id=1, name="Weak", short_name="WEA"),
            Team(id=2, name="Strong", short_name="STR"),
        ])
        s.add(TeamSeasonStrength(
            season=season, team_id=1, code=101,
            strength_defence_home=def_home, strength_defence_away=def_home,
        ))
        s.add(TeamSeasonStrength(
            season=season, team_id=2, code=202,
            strength_defence_home=def_away, strength_defence_away=def_away,
        ))
        s.add(Fixture(fpl_id=1, season=season, gameweek=gw, team_h_id=1, team_a_id=2))
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Home",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.add(Player(fpl_id=2, code=2, first_name="B", second_name="B", web_name="Away",
                     team_id=2, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()


def test_load_horizon_fixtures_resolves_opponent_and_home_away(temp_session):
    _seed_two_team_fixture(temp_session)
    players = cs.load_current_players()
    out = cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=1)

    home_row = out[out["player_id"] == players.loc[players["web_name"] == "Home", "id"].iloc[0]].iloc[0]
    away_row = out[out["player_id"] == players.loc[players["web_name"] == "Away", "id"].iloc[0]].iloc[0]
    assert home_row["gameweek"] == 1
    assert bool(home_row["was_home"]) is True
    assert home_row["opp_defence_strength"] == pytest.approx(1400.0)
    assert bool(away_row["was_home"]) is False
    assert away_row["opp_defence_strength"] == pytest.approx(1000.0)


def test_load_horizon_fixtures_prior_season_fallback_when_current_is_zero(temp_session):
    # Current season (2026-27): both teams' defence strength unpublished (0,
    # the real pre-season state as of 2026-08-10). Prior season (2025-26)
    # has real values, joined on the stable `code`.
    s = temp_session()
    try:
        s.add_all([
            Team(id=1, name="Weak", short_name="WEA"),
            Team(id=2, name="Strong", short_name="STR"),
        ])
        s.add(TeamSeasonStrength(season="2026-27", team_id=1, code=101,
                                  strength_defence_home=0, strength_defence_away=0))
        s.add(TeamSeasonStrength(season="2026-27", team_id=2, code=202,
                                  strength_defence_home=0, strength_defence_away=0))
        s.add(TeamSeasonStrength(season="2025-26", team_id=1, code=101,
                                  strength_defence_home=1000, strength_defence_away=1000))
        s.add(TeamSeasonStrength(season="2025-26", team_id=2, code=202,
                                  strength_defence_home=1400, strength_defence_away=1400))
        s.add(Fixture(fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Home",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    out = cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=1)
    home_row = out[out["player_id"] == players.loc[players["web_name"] == "Home", "id"].iloc[0]].iloc[0]
    assert home_row["opp_defence_strength"] == pytest.approx(1400.0)  # from 2025-26, via code=202


def test_load_horizon_fixtures_degrades_to_none_when_no_strength_data_at_all(temp_session):
    s = temp_session()
    try:
        s.add_all([
            Team(id=1, name="Weak", short_name="WEA"),
            Team(id=2, name="Strong", short_name="STR"),
        ])
        s.add(Fixture(fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Home",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    out = cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=1)
    home_row = out[out["player_id"] == players.loc[players["web_name"] == "Home", "id"].iloc[0]].iloc[0]
    assert home_row["opp_defence_strength"] is None


def test_load_horizon_fixtures_spans_multiple_gws(temp_session):
    s = temp_session()
    try:
        s.add_all([
            Team(id=1, name="Weak", short_name="WEA"),
            Team(id=2, name="Strong", short_name="STR"),
            Team(id=3, name="Mid", short_name="MID"),
        ])
        s.add(TeamSeasonStrength(season="2026-27", team_id=1, code=101,
                                  strength_defence_home=1000, strength_defence_away=1000))
        s.add(TeamSeasonStrength(season="2026-27", team_id=2, code=202,
                                  strength_defence_home=1400, strength_defence_away=1400))
        s.add(TeamSeasonStrength(season="2026-27", team_id=3, code=303,
                                  strength_defence_home=1200, strength_defence_away=1200))
        s.add(Fixture(fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
        s.add(Fixture(fpl_id=2, season="2026-27", gameweek=2, team_h_id=3, team_a_id=1))
        s.add(Player(fpl_id=1, code=1, first_name="A", second_name="A", web_name="Home",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    out = cs.load_horizon_fixtures(players, "2026-27", target_gw=1, horizon=2)
    pid = players.loc[players["web_name"] == "Home", "id"].iloc[0]
    rows = out[out["player_id"] == pid].sort_values("gameweek")
    assert list(rows["gameweek"]) == [1, 2]
    assert rows.iloc[0]["opp_defence_strength"] == pytest.approx(1400.0)  # GW1 vs Strong
    assert rows.iloc[1]["opp_defence_strength"] == pytest.approx(1200.0)  # GW2 vs Mid


def test_load_horizon_fixtures_empty_players_or_gws_returns_empty(temp_session):
    empty = pd.DataFrame(columns=["id", "team_id"])
    out = cs.load_horizon_fixtures(empty, "2026-27", target_gw=1, horizon=1)
    assert out.empty
    assert list(out.columns) == ["player_id", "gameweek", "opp_defence_strength", "was_home"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -k load_horizon_fixtures -v`
Expected: FAIL with `AttributeError: module 'projection.cold_start' has no attribute 'load_horizon_fixtures'`

- [ ] **Step 3: Implement the four functions**

In `projection/cold_start.py`, insert after `load_prior_league_lookup` (currently ends line 202) and before `apply_departure_gate` (currently starts line 205):

```python
def load_current_defence_strength(season: str) -> dict[int, float]:
    """team_id -> average(strength_defence_home, strength_defence_away) for
    ``season``, treating an all-zero row (FPL hasn't published it yet -- the
    real 2026-27 pre-season state as of 2026-08-10) as ABSENT rather than a
    genuine 0 -- callers fall through to the prior-season fallback instead
    of being misled by a value on the wrong scale (using
    strength_overall_home/away, which IS populated this early but on an
    incompatible ~2-5 scale vs strength_defence's ~1000-1400, was
    considered and rejected -- see the design spec)."""
    db = get_session()
    try:
        query = text("""
            SELECT team_id, strength_defence_home, strength_defence_away
            FROM team_season_strength WHERE season = :season
        """)
        df = pd.read_sql(query, db.bind, params={"season": season})
    finally:
        db.close()
    result: dict[int, float] = {}
    for r in df.itertuples():
        avg = (r.strength_defence_home + r.strength_defence_away) / 2
        if avg > 0:
            result[int(r.team_id)] = avg
    return result


def load_team_codes(season: str) -> dict[int, int]:
    """team_id -> stable code for ``season`` (only rows where FPL has
    supplied one) -- used to resolve an opponent's PRIOR-season strength via
    the identity that survives promotion/relegation reshuffling team_id."""
    db = get_session()
    try:
        query = text("""
            SELECT team_id, code FROM team_season_strength
            WHERE season = :season AND code IS NOT NULL
        """)
        df = pd.read_sql(query, db.bind, params={"season": season})
    finally:
        db.close()
    return {int(r.team_id): int(r.code) for r in df.itertuples()}


def load_prior_defence_strength_by_code(prior_season: str) -> dict[int, float]:
    """code -> average defence strength from ``prior_season`` -- the
    fallback used when the CURRENT season's strength is still unpublished,
    so Feature A has real fixture-difficulty signal now rather than only
    once FPL catches up close to the GW1 deadline."""
    db = get_session()
    try:
        query = text("""
            SELECT code, strength_defence_home, strength_defence_away
            FROM team_season_strength WHERE season = :season AND code IS NOT NULL
        """)
        df = pd.read_sql(query, db.bind, params={"season": prior_season})
    finally:
        db.close()
    result: dict[int, float] = {}
    for r in df.itertuples():
        avg = (r.strength_defence_home + r.strength_defence_away) / 2
        if avg > 0:
            result[int(r.code)] = avg
    return result


def load_horizon_fixtures(
    players: pd.DataFrame, season: str, target_gw: int, horizon: int,
) -> pd.DataFrame:
    """(player_id, gameweek, opp_defence_strength, was_home) for each of the
    ``horizon`` GWs starting at ``target_gw``, resolved from ``players``'
    OWN team_id column -- post Feature-B override, since it is never
    re-derived by re-querying the players table from the DB. This is what
    lets a manual team_id correction (Feature B) actually change which
    fixtures a player is attributed to.

    opp_defence_strength resolution, per fixture: (1) current season's
    TeamSeasonStrength if non-zero, (2) prior-season TeamSeasonStrength for
    the same club, joined on the stable `code` (not team_id -- a per-season
    alphabetical index that shifts under promotion/relegation), (3) None if
    neither exists -- `fixture_multiplier` already treats None as neutral
    (1.0), so a promoted club with no 2025-26 row degrades safely rather
    than crashing or defaulting to a misleading value.
    """
    empty = pd.DataFrame(columns=["player_id", "gameweek", "opp_defence_strength", "was_home"])
    if players.empty or horizon <= 0:
        return empty

    team_ids = sorted({int(t) for t in players["team_id"].dropna().unique()})
    target_gws = list(range(target_gw, target_gw + horizon))
    if not team_ids or not target_gws:
        return empty

    db = get_session()
    try:
        team_placeholders = ",".join(f":team{i}" for i in range(len(team_ids)))
        gw_placeholders = ",".join(f":gw{i}" for i in range(len(target_gws)))
        params = {
            "season": season,
            **{f"team{i}": tid for i, tid in enumerate(team_ids)},
            **{f"gw{i}": gw for i, gw in enumerate(target_gws)},
        }
        query = text(f"""
            SELECT f.team_h_id, f.team_a_id, f.gameweek
            FROM fixtures f
            WHERE f.season = :season AND f.gameweek IN ({gw_placeholders})
              AND (f.team_h_id IN ({team_placeholders}) OR f.team_a_id IN ({team_placeholders}))
        """)
        raw = pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()
    if raw.empty:
        return empty

    current_strength = load_current_defence_strength(season)
    team_codes = load_team_codes(season)
    prior_strength_by_code = load_prior_defence_strength_by_code(prior_season_of(season))

    def _resolve(opp_team_id: int) -> float | None:
        if opp_team_id in current_strength:
            return current_strength[opp_team_id]
        code = team_codes.get(opp_team_id)
        if code is not None and code in prior_strength_by_code:
            return prior_strength_by_code[code]
        return None

    team_id_set = set(team_ids)
    fixture_rows: list[dict] = []
    for r in raw.itertuples():
        if r.team_h_id in team_id_set:
            fixture_rows.append({
                "team_id": r.team_h_id, "gameweek": r.gameweek,
                "opp_defence_strength": _resolve(r.team_a_id), "was_home": True,
            })
        if r.team_a_id in team_id_set:
            fixture_rows.append({
                "team_id": r.team_a_id, "gameweek": r.gameweek,
                "opp_defence_strength": _resolve(r.team_h_id), "was_home": False,
            })
    fixtures_by_team = pd.DataFrame(
        fixture_rows, columns=["team_id", "gameweek", "opp_defence_strength", "was_home"]
    )
    if fixtures_by_team.empty:
        return empty

    merged = players[["id", "team_id"]].merge(fixtures_by_team, on="team_id", how="inner")
    return merged.rename(columns={"id": "player_id"})[
        ["player_id", "gameweek", "opp_defence_strength", "was_home"]
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -k "load_horizon_fixtures or load_current_players_applies" -v`
Expected: all PASS

- [ ] **Step 5: Run the full cold-start suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -v`
Expected: all PASS (no existing test touches these new functions, so nothing should change)

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/ruff check projection/cold_start.py tests/test_cold_start.py`
Expected: no errors

```bash
git add projection/cold_start.py tests/test_cold_start.py
git commit -m "feat: load_horizon_fixtures — per-GW opponent defence strength with prior-season fallback"
```

---

### Task 6: `project_cold_start` gains `horizon`/`season`

**Files:**
- Modify: `projection/cold_start.py` (`project_cold_start`, lines 261-366)
- Test: `tests/test_cold_start.py` (append)

**Interfaces:**
- Consumes: `load_horizon_fixtures(players, season, target_gw, horizon)` (Task 5); `projection.fixture_adjust.fixture_multiplier(opp_defence_strength, was_home) -> float` (existing).
- Produces: `project_cold_start(players, prior_features, target_gw=1, raw_appearances=None, prior_league_lookup=None, horizon=1, season=None) -> pd.DataFrame` — when `horizon <= 1` or `season is None`, byte-for-byte identical output to before this task (one row per player at `target_gw`); when `horizon > 1` and `season` given, one row per `(player, gw)` for `gw` in `[target_gw, target_gw + horizon)`, with `xpts`/`xpts_var` scaled by that GW's `fixture_multiplier`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cold_start.py`:

```python
def test_project_cold_start_horizon_1_is_byte_identical_to_default(temp_session):
    _seed(temp_session)
    players = cs.apply_departure_gate(cs.load_current_players())
    prior = cs.load_prior_season_features("2025-26")

    default_proj = cs.project_cold_start(players, prior)
    explicit_proj = cs.project_cold_start(players, prior, horizon=1, season="2026-27")
    pd.testing.assert_frame_equal(
        default_proj.sort_values("player_id").reset_index(drop=True),
        explicit_proj.sort_values("player_id").reset_index(drop=True),
    )


def test_project_cold_start_horizon_emits_one_row_per_gw_with_distinct_xpts(temp_session):
    _seed_two_team_fixture(temp_session, gw=1)
    s = temp_session()
    try:
        s.add(Fixture(fpl_id=2, season="2026-27", gameweek=2, team_h_id=2, team_a_id=1))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    home_id = players.loc[players["web_name"] == "Home", "id"].iloc[0]

    proj = cs.project_cold_start(players, prior, target_gw=1, horizon=2, season="2026-27")
    home_rows = proj[proj["player_id"] == home_id].sort_values("gameweek")

    assert list(home_rows["gameweek"]) == [1, 2]
    gw1_xpts, gw2_xpts = home_rows["xpts"].tolist()
    # GW1: home vs weak Team 2... wait -- Home is Team 1, opponent Team 2 is
    # the STRONG defence in _seed_two_team_fixture (1400) at GW1 (home), and
    # Team 2 (still strong, now at home) hosts Team 1 (away) at GW2 -- both
    # legs are against the same strong opponent, but home/away differs, so
    # the multipliers (and therefore xpts) must differ between the two rows.
    assert gw1_xpts != gw2_xpts
    from projection.fixture_adjust import fixture_multiplier
    base_xpts = home_rows["xpts"].iloc[0] / fixture_multiplier(1400.0, True)
    assert gw2_xpts == pytest.approx(base_xpts * fixture_multiplier(1400.0, False))


def test_project_cold_start_horizon_var_scales_with_multiplier_squared(temp_session):
    _seed_variance_pool(temp_session)
    s = temp_session()
    try:
        s.add_all([Team(id=1, name="T1", short_name="T1_"), Team(id=2, name="T2", short_name="T2_")])
        s.add(TeamSeasonStrength(season="2026-27", team_id=1, code=1001,
                                  strength_defence_home=1000, strength_defence_away=1000))
        s.add(TeamSeasonStrength(season="2026-27", team_id=2, code=1002,
                                  strength_defence_home=1000, strength_defence_away=1000))
        s.add(Fixture(fpl_id=1, season="2026-27", gameweek=1, team_h_id=1, team_a_id=2))
        s.commit()
    finally:
        s.close()

    players = cs.load_current_players()
    prior = cs.load_prior_season_features("2025-26")
    raw = cs.load_prior_season_appearances("2025-26")
    base_proj = cs.project_cold_start(players, prior, raw_appearances=raw)
    horizon_proj = cs.project_cold_start(
        players, prior, raw_appearances=raw, target_gw=1, horizon=1, season="2026-27"
    )

    from projection.fixture_adjust import fixture_multiplier
    varied_id = players.loc[players["web_name"] == "Varied", "id"].iloc[0]
    base_row = base_proj[base_proj["player_id"] == varied_id].iloc[0]
    horizon_row = horizon_proj[horizon_proj["player_id"] == varied_id].iloc[0]
    mult = fixture_multiplier(1000.0, True)  # Varied is on team_id=1, home
    assert horizon_row["xpts_var"] == pytest.approx(base_row["xpts_var"] * mult ** 2)


def test_project_cold_start_horizon_with_no_fixture_data_repeats_base_row_neutrally(temp_session):
    _seed(temp_session)  # no Fixture/TeamSeasonStrength rows seeded at all
    players = cs.apply_departure_gate(cs.load_current_players())
    prior = cs.load_prior_season_features("2025-26")

    proj = cs.project_cold_start(players, prior, target_gw=1, horizon=3, season="2026-27")
    estab_id = players.loc[players["web_name"] == "Estab", "id"].iloc[0]
    rows = proj[proj["player_id"] == estab_id].sort_values("gameweek")
    assert list(rows["gameweek"]) == [1, 2, 3]
    assert (rows["xpts"] == pytest.approx(6.0)).all()  # same base value every GW, no crash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -k "project_cold_start_horizon" -v`
Expected: FAIL — `TypeError: project_cold_start() got an unexpected keyword argument 'horizon'`

- [ ] **Step 3: Add the import and extend `project_cold_start`**

In `projection/cold_start.py`, add to the imports (with the other `projection` imports):

```python
from projection.fixture_adjust import fixture_multiplier
```

Change the `project_cold_start` signature (currently lines 261-267):

```python
def project_cold_start(
    players: pd.DataFrame,
    prior_features: pd.DataFrame,
    target_gw: int = 1,
    raw_appearances: pd.DataFrame | None = None,
    prior_league_lookup: dict[int, dict] | None = None,
    horizon: int = 1,
    season: str | None = None,
) -> pd.DataFrame:
```

Extend the docstring (currently lines 268-288) by appending this paragraph at the end (keep everything already there unchanged):

```python
    ``horizon`` (default 1, preserving today's exact single-row-per-player
    behaviour for every existing caller): when > 1, emits one row per
    ``(player, gw)`` for ``gw`` in ``[target_gw, target_gw + horizon)``
    instead of one row per player, with xpts/xpts_var scaled by that GW's
    fixture_multiplier (plan 2026-08-10, cold-start fixture lookahead).
    Requires ``season`` to resolve fixtures/team strengths -- if
    ``horizon > 1`` but ``season`` is None, degrades to repeating the
    single-GW base row at every horizon GW with a neutral multiplier rather
    than crashing (mirrors ``load_horizon_fixtures`` returning empty when it
    has nothing to resolve).
    """
```

Change the function body's ending. Everything up to and including the existing `for r in merged.itertuples(): ... rows.append({...})` loop (lines 302-365) stays **exactly as-is** — only the final `return pd.DataFrame(rows)` line (line 366) changes:

```python
    base_df = pd.DataFrame(rows)
    if horizon <= 1 or season is None:
        return base_df

    fixtures = load_horizon_fixtures(players, season, target_gw, horizon)
    if fixtures.empty:
        # No resolvable fixture data (e.g. a synthetic/test season with no
        # fixtures rows at all) -- degrade to repeating the base projection
        # at every horizon GW with an implicit neutral multiplier, rather
        # than silently dropping the extra GWs the caller asked for.
        repeated = []
        for gw in range(target_gw, target_gw + horizon):
            gw_df = base_df.copy()
            gw_df["gameweek"] = gw
            repeated.append(gw_df)
        return pd.concat(repeated, ignore_index=True)

    base_by_player = base_df.set_index("player_id")
    horizon_rows: list[dict] = []
    for f in fixtures.itertuples():
        if f.player_id not in base_by_player.index:
            continue
        base = base_by_player.loc[f.player_id]
        mult = fixture_multiplier(f.opp_defence_strength, f.was_home)
        horizon_rows.append({
            "player_id": f.player_id,
            "gameweek": f.gameweek,
            "xpts": base["xpts"] * mult,
            "xpts_var": base["xpts_var"] * mult ** 2,
            "start_probability": base["start_probability"],
            "proj_source": base["proj_source"],
        })
    return pd.DataFrame(horizon_rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -k "project_cold_start_horizon" -v`
Expected: all PASS

- [ ] **Step 5: Run the full cold-start suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -v`
Expected: all PASS — every pre-existing test calls `project_cold_start` without `horizon`/`season`, so they exercise the unchanged `horizon <= 1` early-return path.

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/ruff check projection/cold_start.py tests/test_cold_start.py`
Expected: no errors

```bash
git add projection/cold_start.py tests/test_cold_start.py
git commit -m "feat: project_cold_start horizon parameter — per-GW fixture-scaled projections"
```

---

### Task 7: Thread the lookahead through `build_initial_squad` + end-to-end/regression tests

**Files:**
- Modify: `projection/cold_start.py` (`build_initial_squad`, current state after Task 3's edit)
- Test: `tests/test_cold_start.py` (append)

**Interfaces:**
- Consumes: `cfg.cold_start_lookahead_gws` (Task 4); `project_cold_start(..., horizon=..., season=...)` (Task 6); `optimiser.squad.optimise_squad(..., horizon=...)` (existing, unchanged — already sums whatever horizon of rows it's given).
- Produces: `build_initial_squad(season, budget=None, players=None, config=None)` now builds the squad from a real multi-GW fixture-weighted horizon by default (5 GWs), while a caller passing `config=OptimiserConfig(cold_start_lookahead_gws=1)` gets exactly today's pre-Feature-A single-GW behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cold_start.py`:

```python
def test_build_initial_squad_uses_horizon_sum_not_single_gw(temp_session):
    _seed_full_pool(temp_session)
    s = temp_session()
    try:
        # give every club a real GW1-5 fixture list against itself in a
        # round-robin so load_horizon_fixtures has something to resolve
        # (content doesn't matter here -- only that >1 distinct GW exists).
        club_ids = list(range(1, 9))
        fpl_id = 1
        for gw in range(1, 6):
            for i in range(0, len(club_ids), 2):
                s.add(Fixture(fpl_id=fpl_id, season="2026-27", gameweek=gw,
                              team_h_id=club_ids[i], team_a_id=club_ids[i + 1]))
                fpl_id += 1
        s.commit()
        injected = pd.read_sql(
            "SELECT id, code, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    solution_5gw, proj_5gw = cs.build_initial_squad("2026-27", players=injected)
    solution_1gw, proj_1gw = cs.build_initial_squad(
        "2026-27", players=injected,
        config=OptimiserConfig(cold_start_lookahead_gws=1),
    )

    assert proj_5gw["gameweek"].nunique() == 5
    assert proj_1gw["gameweek"].nunique() == 1
    assert len(solution_5gw.squad) == 15
    assert len(solution_1gw.squad) == 15


def test_build_initial_squad_regression_single_gw_config_matches_pre_feature_a_shape(
    temp_session,
):
    """Regression guard (spec's explicit requirement): a caller pinning
    cold_start_lookahead_gws=1 must get a projections frame shaped exactly
    like the pre-Feature-A single-GW output -- one row per player, all at
    target_gw=1."""
    _seed_full_pool(temp_session)
    s = temp_session()
    try:
        injected = pd.read_sql(
            "SELECT id, code, web_name, position, now_cost, status, team_id FROM players", s.bind
        )
    finally:
        s.close()

    _, projections = cs.build_initial_squad(
        "2026-27", players=injected, config=OptimiserConfig(cold_start_lookahead_gws=1),
    )
    assert (projections["gameweek"] == 1).all()
    assert projections["player_id"].nunique() == len(projections)  # exactly one row per player
```

Add the `OptimiserConfig` import at the top of `tests/test_cold_start.py` if not already present at module scope (Task 4's test additions may have only imported it inline inside a function — check with `grep -n "^from config.strategy import\|OptimiserConfig" tests/test_cold_start.py` first; add `from config.strategy import OptimiserConfig` near the top-level imports if it's missing there).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -k "build_initial_squad_uses_horizon_sum or build_initial_squad_regression" -v`
Expected: FAIL — `proj_5gw["gameweek"].nunique()` is `1`, not `5` (still hardcoded `horizon=1`).

- [ ] **Step 3: Thread `cfg.cold_start_lookahead_gws` through `build_initial_squad`**

In `projection/cold_start.py`, `build_initial_squad` (as left by Task 3) currently starts:

```python
    from config.strategy import SQUAD
    from data.overrides import load_p_leave_overrides, log_rumoured_squad_members
    from optimiser.departure_risk import apply_departure_discount
    from optimiser.squad import optimise_squad

    budget = SQUAD.budget_total if budget is None else budget
```

Change to resolve `cfg` up front and use it for both `project_cold_start`'s and `optimise_squad`'s horizon:

```python
    from config.strategy import OPTIMISER, SQUAD
    from data.overrides import load_p_leave_overrides, log_rumoured_squad_members
    from optimiser.departure_risk import apply_departure_discount
    from optimiser.squad import optimise_squad

    cfg = config or OPTIMISER
    budget = SQUAD.budget_total if budget is None else budget
```

Then change the `project_cold_start` call (added in Task 3):

```python
    projections = project_cold_start(
        players, prior, raw_appearances=raw_appearances,
        prior_league_lookup=prior_league_lookup,
        horizon=cfg.cold_start_lookahead_gws, season=season,
    )
```

Then fix the merge that follows — **critical**: `projections` now has `cfg.cold_start_lookahead_gws` rows per player instead of 1, so merging it into `players` unmodified would multiply every player's row by the horizon length (breaking the squad-size/position constraints downstream). Change:

```python
    players = players.merge(
        projections[["player_id", "start_probability"]],
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")
```

to:

```python
    players = players.merge(
        projections[["player_id", "start_probability"]].drop_duplicates("player_id"),
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")
```

Finally, change the `optimise_squad` call's `horizon=1` to `horizon=cfg.cold_start_lookahead_gws`:

```python
    solution = optimise_squad(
        projections=projections, players=players, budget=budget,
        horizon=cfg.cold_start_lookahead_gws, season=season, config=config,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -k "build_initial_squad_uses_horizon_sum or build_initial_squad_regression" -v`
Expected: all PASS

- [ ] **Step 5: Run the complete cold-start test file**

Run: `.venv/bin/python -m pytest tests/test_cold_start.py -v`
Expected: all PASS, including every pre-existing test (`test_build_initial_squad_uses_injected_players_not_live_bootstrap`, `test_build_initial_squad_passes_config_through_to_optimise_squad`, and the `test_build_initial_squad_discounts_rumoured_player` test from Task 3) — none of them assert an exact `total_xpts`/row-count that the new default `horizon=5` would break (confirmed during plan-writing by reading each assertion).

- [ ] **Step 6: Run the full project test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all PASS. Pay particular attention to any test that calls `cold_start.build_initial_squad`, `cold_start.project_cold_start`, `cold_start.load_current_players`, or `decision_engine._load_players`/`decision_engine._run_decision_cycle` outside `tests/test_cold_start.py` — grep first to be sure none were missed:

```bash
grep -rln "build_initial_squad\|project_cold_start\|load_current_players\|_run_decision_cycle" tests/
```

If any other test file calls these and asserts something horizon/override-sensitive (e.g. an exact `xpts` total or row count), read it and reconcile before declaring this step done.

- [ ] **Step 7: Full lint pass**

Run: `.venv/bin/ruff check .`
Expected: no errors (fix any that appear before proceeding).

- [ ] **Step 8: Commit**

```bash
git add projection/cold_start.py tests/test_cold_start.py
git commit -m "feat: cold-start build now uses the fixture-weighted multi-GW lookahead by default"
```

- [ ] **Step 9: Update project memory / close out**

This is the final task. Confirm with the user that:
- Feature A (fixture lookahead) and Feature B (manual overrides) are both implemented and tested per the approved spec.
- `config/transfer_overrides.yaml` currently has empty `confirmed`/`rumoured` lists — the user's own next step is to hand-edit it (e.g. Bruno Guimarães → Arsenal, if FPL still hasn't updated `team_id` by then).
- The "Future work" section of the spec (auto-populating the YAML from `injury_parser.py`/`press_conference.py`) remains explicitly out of scope and un-started.

No further skill invocation needed after this — this is the end of the plan.
