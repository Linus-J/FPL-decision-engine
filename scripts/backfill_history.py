#!/usr/bin/env python
import asyncio
import io
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import httpx
import pandas as pd
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session, init_db
from data.models import (
    Gameweek,
    PlayerGameweekStats,
    PlayerStateSnapshot,
    TeamSeasonStrength,
)

logger = logging.getLogger(__name__)

VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

SEASONS = [
    ("2021-22", "2021-22"),
    ("2022-23", "2022-23"),
    ("2023-24", "2023-24"),
    ("2024-25", "2024-25"),
    ("2025-26", "2025-26"),
]

GW_CSV_URL = "{base}/{season}/gws/merged_gw.csv"
FIXTURES_CSV_URL = "{base}/{season}/fixtures.csv"
PLAYERS_RAW_CSV_URL = "{base}/{season}/players_raw.csv"
TEAMS_CSV_URL = "{base}/{season}/teams.csv"

_STRENGTH_COLS = (
    "strength_overall_home", "strength_overall_away",
    "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
)

# FPL deadline is ~90 minutes before the first kickoff of the gameweek.
DEADLINE_LEAD = timedelta(minutes=90)


async def _fetch_csv(client: httpx.AsyncClient, url: str) -> pd.DataFrame | None:
    try:
        resp = await client.get(url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        logger.warning("Could not fetch %s: %s", url, exc)
        return None


# --- Per-season gameweek deadlines (T3a; fixes M1/M2 for the as-of boundary) ---

def compute_gw_deadlines(fixtures_df: pd.DataFrame) -> dict[int, datetime]:
    """Earliest kickoff per gameweek − 90 min, as naive UTC.

    Pure/testable. Rows with a missing event or kickoff_time are ignored
    (postponed/TBC fixtures). The result is the FPL-deadline proxy the
    leakage-free backtest reads against.
    """
    if not {"event", "kickoff_time"}.issubset(fixtures_df.columns):
        return {}
    df = fixtures_df.dropna(subset=["event", "kickoff_time"]).copy()
    if df.empty:
        return {}
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["kickoff_time"])
    deadlines: dict[int, datetime] = {}
    for gw, grp in df.groupby("event"):
        first_kickoff = grp["kickoff_time"].min()
        # store naive UTC to match the rest of the codebase (datetime.utcnow)
        deadline = first_kickoff.tz_convert("UTC").tz_localize(None).to_pydatetime()
        deadlines[int(gw)] = deadline - DEADLINE_LEAD
    return deadlines


def upsert_gameweek_deadlines(season: str, deadlines: dict[int, datetime]) -> int:
    db = get_session()
    written = 0
    try:
        for gw, deadline in deadlines.items():
            stmt = (
                insert(Gameweek)
                .values(
                    id=gw,
                    season=season,
                    name=f"Gameweek {gw}",
                    deadline_time=deadline,
                    finished=True,
                )
                .on_conflict_do_update(
                    index_elements=["id", "season"],
                    set_={"deadline_time": deadline},
                )
            )
            db.execute(stmt)
            written += 1
        db.commit()
    finally:
        db.close()
    return written


# --- Cross-season player mapping via stable code (T3a; fixes M3) ---

def element_code_map(players_raw_df: pd.DataFrame) -> dict[int, int]:
    """Per-season vaastav element id -> FPL cross-season `code`. Pure/testable."""
    if not {"id", "code"}.issubset(players_raw_df.columns):
        return {}
    out: dict[int, int] = {}
    for element, code in zip(players_raw_df["id"], players_raw_df["code"], strict=False):
        if pd.notna(element) and pd.notna(code):
            out[int(element)] = int(code)
    return out


def build_code_to_dbid_map() -> dict[int, int]:
    from sqlalchemy import text

    db = get_session()
    try:
        rows = db.execute(
            text("SELECT id, code FROM players WHERE code IS NOT NULL")
        ).fetchall()
        return {int(code): int(db_id) for db_id, code in rows}
    finally:
        db.close()


def resolve_player_id(
    element: int,
    elem_to_code: dict[int, int],
    code_to_dbid: dict[int, int],
) -> int | None:
    """vaastav element -> code -> players.id. None if the player is not in the
    current squad (e.g. left the league) — those rows are skipped, not misjoined."""
    code = elem_to_code.get(element)
    if code is None:
        return None
    return code_to_dbid.get(code)


# --- Per-season team strengths + fixture context (T3b) ---

def team_strength_rows(teams_df: pd.DataFrame, season: str) -> list[dict]:
    """Rows for TeamSeasonStrength from a season's teams.csv. Pure/testable."""
    if "id" not in teams_df.columns:
        return []
    rows: list[dict] = []
    for r in teams_df.itertuples():
        code = getattr(r, "code", None)
        row: dict = {
            "season": season,
            "team_id": int(r.id),
            "code": int(code) if pd.notna(code) else None,
        }
        for col in _STRENGTH_COLS:
            v = getattr(r, col, None)
            row[col] = int(v) if pd.notna(v) else 1200
        rows.append(row)
    return rows


def team_name_to_id(teams_df: pd.DataFrame) -> dict[str, int]:
    """name/short_name -> that season's team id, for resolving merged_gw `team`."""
    out: dict[str, int] = {}
    if "id" not in teams_df.columns:
        return out
    for r in teams_df.itertuples():
        tid = int(r.id)
        for attr in ("name", "short_name"):
            v = getattr(r, attr, None)
            if pd.notna(v):
                out[str(v)] = tid
    return out


def _resolve_team_id(raw: object, name_to_id: dict[str, int]) -> int | None:
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(raw)  # already a numeric team id
    except (ValueError, TypeError):
        return name_to_id.get(str(raw))  # a team name/short_name


def write_team_strengths(rows: list[dict]) -> int:
    db = get_session()
    try:
        for row in rows:
            stmt = (
                insert(TeamSeasonStrength)
                .values(**row)
                .on_conflict_do_update(
                    index_elements=["season", "team_id"],
                    set_={col: row[col] for col in _STRENGTH_COLS},
                )
            )
            db.execute(stmt)
        db.commit()
        return len(rows)
    finally:
        db.close()


# --- Point-in-time snapshot backfill, reconciled to live bootstrap (T3; C1) ---

SNAPSHOT_EPSILON = timedelta(minutes=1)   # snapshot_ts = deadline − ε (must be < deadline)
FORM_WINDOW_GWS = 5                        # bootstrap `form` proxy: mean points over prior GWs
SQUAD_SIZE = 15                            # Σ(selected)/15 ≈ total managers that GW


def compute_snapshot_rows(
    df: pd.DataFrame,
    deadlines: dict[int, datetime],
    epsilon: timedelta = SNAPSHOT_EPSILON,
) -> list[dict]:
    """merged_gw.csv → point-in-time snapshot rows matching live-bootstrap semantics.

    Pure/testable. The live path (write_player_snapshots) writes bootstrap
    values, which are *season-cumulative* for ICT/influence/creativity/threat
    and a *percent* for ownership. merged_gw carries *per-GW* values and a raw
    *count*, so for a snapshot informing GW g (taken at its deadline) we emit:
      - ict/influence/creativity/threat = cumulative sum over GWs < g (matches
        what bootstrap holds once GWs 1..g-1 are played);
      - now_cost / transfers = the GW-g row (state at the g deadline);
      - selected_by_percent = selected / (Σ selected that GW / 15) × 100;
      - form = mean total_points over the prior FORM_WINDOW_GWS (proxy).
    status / chance_of_playing / news are NOT recoverable from merged_gw and
    default (documented residual skew — Phase 2 may move form/ICT to rate
    features derived identically on both paths).
    """
    df = df.rename(columns={"round": "GW"}) if "GW" not in df.columns else df
    required = {
        "element", "GW", "ict_index", "influence", "creativity", "threat", "value", "selected",
    }
    if not required.issubset(df.columns):
        return []

    d = df.copy()
    for opt in ("transfers_in", "transfers_out", "total_points"):
        if opt not in d.columns:
            d[opt] = 0
    numeric = [
        "element", "GW", "ict_index", "influence", "creativity", "threat",
        "value", "selected", "total_points", "transfers_in", "transfers_out",
    ]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["element", "GW"])
    d["element"] = d["element"].astype(int)
    d["GW"] = d["GW"].astype(int)

    # One row per (element, GW): sum flows (DGW = two fixtures), first for stocks.
    agg = (
        d.groupby(["element", "GW"], as_index=False)
        .agg(
            ict_index=("ict_index", "sum"),
            influence=("influence", "sum"),
            creativity=("creativity", "sum"),
            threat=("threat", "sum"),
            total_points=("total_points", "sum"),
            transfers_in=("transfers_in", "sum"),
            transfers_out=("transfers_out", "sum"),
            value=("value", "first"),
            selected=("selected", "first"),
        )
        .sort_values(["element", "GW"])
    )
    grp = agg.groupby("element")
    # cumulative THROUGH the previous GW (exclude current): cumsum − current.
    for col in ("ict_index", "influence", "creativity", "threat"):
        agg[f"cum_{col}"] = grp[col].cumsum() - agg[col]
    # form ≈ mean points over the prior window (as-of: shift 1 excludes current).
    agg["form"] = (
        grp["total_points"]
        .transform(lambda s: s.shift(1).rolling(FORM_WINDOW_GWS, min_periods=1).mean())
        .fillna(0.0)
    )
    # ownership %: Σ(selected) over all players that GW ≈ SQUAD_SIZE × managers.
    gw_total = agg.groupby("GW")["selected"].transform("sum")
    agg["sel_pct"] = (
        (agg["selected"] * SQUAD_SIZE / gw_total.replace(0, pd.NA) * 100.0)
        .fillna(0.0)
        .clip(0.0, 100.0)
    )

    rows: list[dict] = []
    for r in agg.itertuples():
        deadline = deadlines.get(int(r.GW))
        if deadline is None:
            continue
        rows.append(
            {
                "element": int(r.element),
                "gameweek_context": int(r.GW),
                "snapshot_ts": deadline - epsilon,
                "now_cost": float(r.value) / 10.0 if pd.notna(r.value) else 0.0,
                "selected_by_percent": float(r.sel_pct),
                "form": float(r.form),
                "ict_index": float(r.cum_ict_index),
                "influence": float(r.cum_influence),
                "creativity": float(r.cum_creativity),
                "threat": float(r.cum_threat),
                "transfers_in_event": int(r.transfers_in) if pd.notna(r.transfers_in) else 0,
                "transfers_out_event": int(r.transfers_out) if pd.notna(r.transfers_out) else 0,
            }
        )
    return rows


def write_snapshot_rows(
    season: str,
    rows: list[dict],
    elem_to_code: dict[int, int],
    code_to_dbid: dict[int, int],
) -> tuple[int, int]:
    """Insert reconciled snapshot rows, append-only. Returns (written, skipped)."""
    db = get_session()
    written = skipped = 0
    try:
        for row in rows:
            player_id = resolve_player_id(row["element"], elem_to_code, code_to_dbid)
            if not player_id:
                skipped += 1
                continue
            values = {k: v for k, v in row.items() if k != "element"}
            stmt = (
                insert(PlayerStateSnapshot)
                .values(player_id=player_id, season=season, **values)
                .on_conflict_do_nothing(index_elements=["player_id", "snapshot_ts"])
            )
            result = db.execute(stmt)
            written += result.rowcount or 0
        db.commit()
    finally:
        db.close()
    return written, skipped


def _ingest_dataframe(
    df: pd.DataFrame,
    season: str,
    elem_to_code: dict[int, int],
    code_to_dbid: dict[int, int],
    name_to_id: dict[str, int] | None = None,
) -> tuple[int, int]:
    name_to_id = name_to_id or {}
    df = df.rename(columns={"round": "GW"}) if "GW" not in df.columns else df

    required = {"element", "GW", "minutes", "total_points"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("Season %s CSV missing columns: %s — skipping", season, missing)
        return 0, 0

    db = get_session()
    inserted = skipped = 0
    try:
        for _, row in df.iterrows():
            element = int(row.get("element", 0) or 0)
            player_id = resolve_player_id(element, elem_to_code, code_to_dbid)
            if not player_id:
                skipped += 1
                continue

            gw = int(row.get("GW", 0) or 0)
            if not gw:
                skipped += 1
                continue

            opp = row.get("opponent_team")
            opponent_team_id = int(opp) if pd.notna(opp) else None
            was_home_raw = row.get("was_home")
            was_home = bool(was_home_raw) if pd.notna(was_home_raw) else None

            stmt = (
                insert(PlayerGameweekStats)
                .values(
                    player_id=player_id,
                    gameweek=gw,
                    season=season,
                    team_id_season=_resolve_team_id(row.get("team"), name_to_id),
                    opponent_team_id=opponent_team_id,
                    was_home=was_home,
                    total_points=int(row.get("total_points", 0) or 0),
                    minutes=int(row.get("minutes", 0) or 0),
                    goals_scored=int(row.get("goals_scored", 0) or 0),
                    assists=int(row.get("assists", 0) or 0),
                    clean_sheets=int(row.get("clean_sheets", 0) or 0),
                    goals_conceded=int(row.get("goals_conceded", 0) or 0),
                    saves=int(row.get("saves", 0) or 0),
                    yellow_cards=int(row.get("yellow_cards", 0) or 0),
                    red_cards=int(row.get("red_cards", 0) or 0),
                    bonus=int(row.get("bonus", 0) or 0),
                    bps=int(row.get("bps", 0) or 0),
                    selected=int(row.get("selected", 0) or 0),
                    transfers_in=int(row.get("transfers_in", 0) or 0),
                    transfers_out=int(row.get("transfers_out", 0) or 0),
                    value=float(row.get("value", 0) or 0) / 10.0,
                )
                .on_conflict_do_nothing(
                    index_elements=["player_id", "gameweek", "season", "opponent_team_id"]
                )
            )
            db.execute(stmt)
            inserted += 1

        db.commit()
    finally:
        db.close()

    return inserted, skipped


async def backfill() -> None:
    init_db()
    code_to_dbid = build_code_to_dbid_map()
    logger.info("Code→DB-id map: %d players with a stable code", len(code_to_dbid))

    async with httpx.AsyncClient() as client:
        for vaastav_season, db_season in SEASONS:
            # 1. Per-season gameweek deadlines (the as-of boundary for T3/T4).
            fixtures = await _fetch_csv(
                client, FIXTURES_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            )
            deadlines: dict[int, datetime] = {}
            if fixtures is not None:
                deadlines = compute_gw_deadlines(fixtures)
                n = upsert_gameweek_deadlines(db_season, deadlines)
                logger.info("Season %s: %d gameweek deadlines written", db_season, n)
            else:
                logger.warning(
                    "Season %s: no fixtures.csv — deadlines/snapshots skipped", db_season
                )

            # 2. Element→code crosswalk (fixes the season-unstable id join, M3).
            players_raw = await _fetch_csv(
                client, PLAYERS_RAW_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            )
            elem_to_code = element_code_map(players_raw) if players_raw is not None else {}
            logger.info("Season %s: %d element→code entries", db_season, len(elem_to_code))

            # 2b. Per-season team strengths + name→id map (T3b, for FDR).
            teams_df = await _fetch_csv(
                client, TEAMS_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            )
            name_to_id: dict[str, int] = {}
            if teams_df is not None:
                n_str = write_team_strengths(team_strength_rows(teams_df, db_season))
                name_to_id = team_name_to_id(teams_df)
                logger.info("Season %s: %d team strengths written", db_season, n_str)

            # 3. Per-GW player stats, mapped via code (not the reassigned element id).
            df = await _fetch_csv(
                client, GW_CSV_URL.format(base=VAASTAV_BASE, season=vaastav_season)
            )
            if df is None:
                logger.warning("Skipping season %s — no merged_gw.csv", vaastav_season)
                continue

            total_rows = len(df)
            inserted, skipped = _ingest_dataframe(
                df, db_season, elem_to_code, code_to_dbid, name_to_id
            )
            match_rate = inserted / total_rows * 100 if total_rows else 0
            logger.info(
                "Season %s: %d rows → %d inserted (%.0f%% match rate), %d unmatched",
                db_season, total_rows, inserted, match_rate, skipped,
            )

            # 4. Point-in-time snapshots reconciled to live-bootstrap semantics (C1).
            if deadlines:
                snap_rows = compute_snapshot_rows(df, deadlines)
                snap_written, snap_skipped = write_snapshot_rows(
                    db_season, snap_rows, elem_to_code, code_to_dbid
                )
                logger.info(
                    "Season %s: %d snapshots written (%d unmatched)",
                    db_season, snap_written, snap_skipped,
                )

    logger.info("Historical backfill complete")


def main() -> None:
    asyncio.run(backfill())


if __name__ == "__main__":
    main()
