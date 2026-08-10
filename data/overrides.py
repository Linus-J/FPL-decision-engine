"""overrides.py — manual transfer/rumour corrections (plan: cold-start
fixture lookahead + transfer overrides, 2026-08-10).

FPL's own team_id is trusted unconditionally with no correction mechanism
(see docs/superpowers/specs/2026-08-10-cold-start-lookahead-and-transfer-
overrides-design.md). config/transfer_overrides.yaml is a hand-edited,
version-controlled file the user updates when they know something FPL's
API hasn't caught up on yet -- a confirmed summer signing not yet
reflected in team_id, or a rumoured departure worth discounting. Every
loader here degrades safely to empty/no-op on a missing file, an empty
file, a malformed/unparseable file, a malformed entry, or an unmatched
code -- a wrong automatic correction is a worse failure mode than a
missed one, so nothing here ever crashes the pipeline.
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
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(
                "transfer_overrides.yaml: failed to parse, ignoring all overrides: %s", exc
            )
            return {}
    return data or {}


def load_team_overrides() -> dict[int, int]:
    """code -> corrected team_id, from the `confirmed` list. A malformed
    entry (missing/non-numeric `code`/`team_id`) is skipped and logged at
    warning rather than crashing the pipeline -- see module docstring."""
    entries = _load_yaml().get("confirmed") or []
    result: dict[int, int] = {}
    for e in entries:
        try:
            result[int(e["code"])] = int(e["team_id"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "transfer_overrides.yaml: malformed confirmed entry %r, skipping: %s", e, exc
            )
    return result


def apply_team_overrides(players: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of ``players`` with ``team_id`` replaced wherever
    ``players.code`` matches a `confirmed` override entry. Logs a warning
    for an override `code` with no matching player in ``players`` (a
    likely stale/typo'd entry -- the override itself is still harmless,
    since nothing in ``players`` gets touched, but a silent no-op here is
    easy to mistake for "the override worked"). A no-op (plain copy) when
    the file is missing/empty or ``players`` has no `code` column."""
    out = players.copy()
    overrides = load_team_overrides()
    if not overrides or "code" not in out.columns:
        return out
    present_codes = set(out["code"].dropna().astype(int))
    for code in sorted(set(overrides.keys()) - present_codes):
        logger.warning(
            "transfer_overrides.yaml: confirmed code %s has no matching current player",
            code,
        )
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
    matching current player, or a malformed entry (missing/non-numeric
    `code`/`p_leave`), is skipped (logged at warning, never crashes)."""
    entries = _load_yaml().get("rumoured") or []
    if not entries:
        return {}
    code_to_pid = _code_to_player_id()
    result: dict[int, dict] = {}
    for entry in entries:
        try:
            code = int(entry["code"])
            p_leave = float(entry["p_leave"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "transfer_overrides.yaml: malformed rumoured entry %r, skipping: %s", entry, exc
            )
            continue
        pid = code_to_pid.get(code)
        if pid is None:
            logger.warning(
                "transfer_overrides.yaml: rumoured code %s has no matching "
                "current player, skipping",
                code,
            )
            continue
        result[pid] = {
            "p_leave": p_leave,
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
