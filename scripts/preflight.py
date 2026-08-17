#!/usr/bin/env python
"""preflight.py — verify the DECISION SURFACE, not just the data.

Why this exists. On 2026-08-17, five defects were introduced by fixes made the
same day. Every one passed the test suite and the data quality gate, because
each was a change to STORED STATE that altered behaviour in a consumer nobody
was looking at:

  - persisting cold-start projections made ``projections.empty`` false, which
    sent the engine down the in-season branch and burned a Triple Captain;
  - the same change had 90 personas overwriting the real bot's projections;
  - correcting the live clean-sheet formula and not the training one flipped
    the sign between fit and inference;
  - storing a neutral 1200 in place of FPL's placeholder zeros disabled the
    cold start's prior-season fallback for 17 of 20 teams;
  - re-running a gameweek appended a lineup row that the scorer then counted
    as an independent observation.

Tests assert that code does what it was written to do. The gate asserts the
data is plausible. Neither notices that the ANSWER changed. This does: it
recomputes the decision surface, checks the invariants that must hold, and
diffs the result against a committed baseline. A change that alters the squad
is then loud and deliberate rather than silent.

    python scripts/preflight.py                 # verify
    python scripts/preflight.py --update-baseline   # accept a new answer

Exit code is 0 only when every check passes and nothing drifted unexpectedly.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)

BASELINE_PATH = Path(__file__).resolve().parents[1] / "config" / "preflight_baseline.json"

SEASON = "2026-27"
SQUAD_SIZE = 15
POSITION_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
BUDGET = 100.0


class Result:
    """Accumulates failures so one run reports everything, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            print(f"  ok    {label}")
        else:
            print(f"  FAIL  {label}{(' — ' + detail) if detail else ''}")
            self.failures.append(label)
        return ok

    def note(self, label: str, value: object) -> None:
        print(f"  ....  {label}: {value}")
        self.notes.append(f"{label}: {value}")


def _latest_decision(db) -> dict:
    from sqlalchemy import text

    row = db.execute(
        text(
            "SELECT details FROM decision_log WHERE decision_type = 'lineup' "
            "ORDER BY created_at DESC LIMIT 1"
        )
    ).fetchone()
    return json.loads(row[0]) if row else {}


def check_squad_is_legal(db, result: Result) -> dict:
    """FPL's own rules. A squad that breaks one cannot be entered at all, so
    this is the check whose failure costs the most and takes the least effort
    to run."""
    from sqlalchemy import text

    print("\n[squad legality]")
    d = _latest_decision(db)
    if not d:
        result.check(False, "a lineup decision exists")
        return {}

    players = {
        r[0]: r for r in db.execute(
            text("SELECT id, web_name, now_cost, position, team_id, status FROM players")
        ).fetchall()
    }
    squad = [players[i] for i in d["squad_ids"] if i in players]
    result.check(len(squad) == SQUAD_SIZE, f"squad is {SQUAD_SIZE} players", str(len(squad)))

    by_pos: dict[str, int] = {}
    for r in squad:
        by_pos[r[3]] = by_pos.get(r[3], 0) + 1
    result.check(by_pos == POSITION_QUOTA, "positions are 2/5/5/3", str(by_pos))

    cost = sum(r[2] for r in squad)
    result.check(cost <= BUDGET + 1e-9, f"cost within £{BUDGET}m", f"£{cost:.1f}m")

    by_club: dict[int, int] = {}
    for r in squad:
        by_club[r[4]] = by_club.get(r[4], 0) + 1
    worst = max(by_club.values()) if by_club else 0
    result.check(worst <= MAX_PER_CLUB, f"at most {MAX_PER_CLUB} per club", f"max {worst}")

    xi = [players[i] for i in d["starting_ids"] if i in players]
    xi_pos: dict[str, int] = {}
    for r in xi:
        xi_pos[r[3]] = xi_pos.get(r[3], 0) + 1
    result.check(len(xi) == 11, "XI is 11 players", str(len(xi)))
    result.check(xi_pos.get("GKP") == 1, "exactly 1 keeper starts")
    result.check(xi_pos.get("DEF", 0) >= 3, "at least 3 defenders start")
    result.check(xi_pos.get("FWD", 0) >= 1, "at least 1 forward starts")

    result.check(d["captain_id"] in d["starting_ids"], "captain is in the XI")
    result.check(d["vice_captain_id"] in d["starting_ids"], "vice-captain is in the XI")
    bench = [int(x) for x in d.get("bench_order", {})]
    result.check(
        sorted(d["squad_ids"]) == sorted(d["starting_ids"] + bench),
        "XI + bench reconciles to the squad",
    )
    unavailable = [r[1] for r in squad if r[5] not in ("a", "d")]
    result.check(not unavailable, "no unavailable player selected", ", ".join(unavailable))

    return {
        "squad": sorted(players[i][1] for i in d["squad_ids"] if i in players),
        "starting_xi": sorted(players[i][1] for i in d["starting_ids"] if i in players),
        "captain": players[d["captain_id"]][1],
        "vice_captain": players[d["vice_captain_id"]][1],
        "bench_order": [players[int(p)][1] for p, _ in
                        sorted(d.get("bench_order", {}).items(), key=lambda kv: kv[1])],
        "cost": round(cost, 1),
    }


def check_no_duplicate_live_decisions(db, result: Result) -> None:
    """A re-run appends rather than replaces. Every READ takes the latest, but
    the outcome scorer once took them all and counted seven re-runs of GW1 as
    seven observations."""
    from sqlalchemy import text

    print("\n[decision log]")
    for table, extra in (("decision_log", ""), ("sim_decision_log", ", sim_manager_id")):
        rows = db.execute(
            text(
                f"SELECT COUNT(*) FROM (SELECT gameweek{extra}, COUNT(*) n "
                f"FROM {table} WHERE decision_type = 'lineup' "
                f"GROUP BY gameweek{extra} HAVING n > 1)"
            )
        ).scalar() or 0
        result.check(rows == 0, f"{table}: one lineup per gameweek", f"{rows} duplicated")

    chips = db.execute(
        text("SELECT COUNT(*) FROM decision_log WHERE decision_type = 'chip'")
    ).scalar() or 0
    result.check(chips == 0, "no chip played before GW1", f"{chips} chip rows")


def check_no_leakage(db, result: Result) -> None:
    """Every point-in-time source must be stamped at or before the deadline."""
    from sqlalchemy import text

    print("\n[point-in-time integrity]")
    late_snapshots = db.execute(
        text(
            "SELECT COUNT(*) FROM player_state_snapshots ps "
            "JOIN gameweeks g ON g.id = 1 AND g.season = :s "
            "WHERE ps.season = :s AND ps.snapshot_ts > g.deadline_time"
        ),
        {"s": SEASON},
    ).scalar() or 0
    result.check(late_snapshots == 0, "no post-deadline player snapshots", str(late_snapshots))

    late_odds = db.execute(
        text(
            "SELECT COUNT(*) FROM fixture_odds fo JOIN fixtures f ON f.id = fo.fixture_id "
            "JOIN gameweeks g ON g.id = f.gameweek AND g.season = f.season "
            "WHERE f.season = :s AND f.gameweek = 1 AND fo.fetched_at > g.deadline_time"
        ),
        {"s": SEASON},
    ).scalar() or 0
    result.note("GW1 odds snapshots taken after the deadline (excluded on read)", late_odds)


def check_fallbacks_engage(result: Result) -> dict:
    """Fallbacks are invisible when they work and invisible when they don't.

    The prior-season strength fallback was silently disabled for a whole
    morning by a change that looked like a scale fix; nothing failed, the
    numbers just quietly got worse. Count the engagements so a regression
    shows up as a number, not a vibe.
    """
    from projection.cold_start import (
        load_current_defence_strength,
        load_prior_defence_strength_by_code,
        load_team_codes,
    )

    print("\n[fallbacks]")
    current = load_current_defence_strength(SEASON)
    prior = load_prior_defence_strength_by_code("2025-26")
    codes = load_team_codes(SEASON)
    resolved = sum(1 for c in codes.values() if c in prior)

    result.check(
        len(current) == 0 or len(current) == len(codes),
        "team strengths are all-published or all-placeholder, not half",
        f"{len(current)}/{len(codes)} usable",
    )
    if not current:
        result.check(
            resolved >= 15,
            "prior-season strength fallback resolves most teams",
            f"{resolved}/{len(codes)}",
        )
    result.note("teams on prior-season strength", f"{resolved}/{len(codes)}")
    return {"teams_on_prior_strength": resolved, "teams_with_live_strength": len(current)}


def check_model_features_are_usable(result: Result) -> dict:
    """Every feature the minutes model reads must vary in training, and any
    that does not must be pinned so the live season cannot become a hidden
    season indicator."""
    from projection.minutes_model import (
        FEATURE_COLS,
        _build_features,
        _degenerate_features,
        _load_training_data,
    )

    print("\n[model features]")
    X = _build_features(_load_training_data())[FEATURE_COLS].astype(float)
    degenerate = _degenerate_features(X)
    result.note("features constant in training (pinned at serve)", len(degenerate))
    result.check(
        len(degenerate) < len(FEATURE_COLS) / 2,
        "most features carry signal",
        f"{len(degenerate)}/{len(FEATURE_COLS)} constant",
    )
    return {"n_features": len(FEATURE_COLS), "degenerate": sorted(degenerate)}


def check_site_export_matches(db, result: Result) -> None:
    """The site is what the user actually reads. It has diverged from the
    decision log before."""
    print("\n[site export]")
    path = Path(__file__).resolve().parents[1] / "data" / "simulations" / "gw1.json"
    if not path.exists():
        result.check(False, "gw1.json exists")
        return
    site = json.loads(path.read_text())
    d = _latest_decision(db)
    site_ids = {e.get("player_id") or e.get("id") for e in site.get("squad", [])}
    result.check(site_ids == set(d.get("squad_ids", [])), "site squad matches decision_log")
    site_xi = {e.get("player_id") or e.get("id")
               for e in site.get("squad", []) if e.get("is_starting")}
    result.check(site_xi == set(d.get("starting_ids", [])), "site XI matches decision_log")


def compare_to_baseline(snapshot: dict, result: Result, update: bool) -> None:
    """The check that would have caught every self-inflicted defect today.

    A fix is allowed to change the answer -- but only on purpose. Drift is
    reported field by field so it can be read and accepted, not glossed over.
    """
    print("\n[baseline]")
    if update:
        BASELINE_PATH.parent.mkdir(exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        print(f"  ....  baseline updated → {BASELINE_PATH}")
        return

    if not BASELINE_PATH.exists():
        print("  ....  no baseline yet; run with --update-baseline to record one")
        return

    baseline = json.loads(BASELINE_PATH.read_text())
    drifted = [k for k in sorted(set(baseline) | set(snapshot))
               if baseline.get(k) != snapshot.get(k)]
    if not drifted:
        result.check(True, "decision surface unchanged since the baseline")
        return
    for k in drifted:
        print(f"  DRIFT {k}:\n          was: {baseline.get(k)}\n          now: {snapshot.get(k)}")
    result.failures.append(f"decision surface drifted: {', '.join(drifted)}")


def main() -> int:
    update = "--update-baseline" in sys.argv
    from data.db import get_session

    result = Result()
    db = get_session()
    try:
        squad = check_squad_is_legal(db, result)
        check_no_duplicate_live_decisions(db, result)
        check_no_leakage(db, result)
        check_site_export_matches(db, result)
    finally:
        db.close()

    fallbacks = check_fallbacks_engage(result)
    features = check_model_features_are_usable(result)

    snapshot = {**squad, **fallbacks, **features}
    compare_to_baseline(snapshot, result, update)

    print()
    if result.failures:
        print(f"PREFLIGHT FAILED — {len(result.failures)} problem(s):")
        for f in result.failures:
            print(f"  - {f}")
        return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
