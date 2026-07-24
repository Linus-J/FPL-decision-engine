"""fbref.py — OPTIONAL FBref event adapter for player_match_events (T5b).

This is the one writer of ``player_match_events`` that needs the network. It is
deliberately kept OUT of the core dependency set: ``soccerdata`` drags in a full
SeleniumBase/Chrome stack (FBref sits behind Cloudflare and is scraped via a
real browser), which cannot run in a headless CI/backfill environment. Install
it only where a browser is available. `uv run` re-syncs the venv from pyproject
(where soccerdata is intentionally absent), so layer it on per-run with `--with`::

    uv run --with soccerdata python scripts/scrape_fbref.py 2025-26

The pure column-mapping (``map_summary_row`` / ``map_keeper_row``) has no
soccerdata dependency and is fully unit-tested; only ``ingest_fbref_season``
touches the network. That live path is UNTESTED against a real browser here —
``FBREF_SUMMARY_MAP`` keys are FBref's match-summary columns flattened as
"<Section> <Leaf>" and may need a one-line correction on the first live run
(they are centralised here precisely so that correction is trivial).

Coverage reality (the sanity harness's tolerance): FBref match tables carry a
subset of FPL's Opta BPS metrics. Available → goals, assists, tackles,
interceptions, blocks, take-ons, passing accuracy, cards, penalties, shots,
saves. Not available → clearances, recoveries, crosses, key passes, big
chances, errors-leading-to-goal/shot, own goals. Those default to 0.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.models import Player, PlayerMatchEvents

logger = logging.getLogger(__name__)

FBREF_LEAGUE = "ENG-Premier League"

# season string (our format) -> soccerdata season string
SEASON_MAP = {
    "2021-22": "2021-2022",
    "2022-23": "2022-2023",
    "2023-24": "2023-2024",
    "2024-25": "2024-2025",
    "2025-26": "2025-2026",
    "2026-27": "2026-2027",
}

# our field -> flattened FBref match-summary column ("<Section> <Leaf>").
# Fields we cannot source from the summary table are simply absent (default 0).
FBREF_SUMMARY_MAP: dict[str, str] = {
    "minutes": "min",
    "goals": "Performance Gls",
    "assists": "Performance Ast",
    "yellow_cards": "Performance CrdY",
    "red_cards": "Performance CrdR",
    "tackles": "Performance Tkl",         # summary Tkl (won-tackles unavailable here)
    "interceptions": "Performance Int",
    "blocks": "Performance Blocks",
    "dribbles": "Take-Ons Succ",          # successful take-ons
    "passes": "Passes Att",
    "pass_completion_pct": "Passes Cmp%",
}
# fields needing a small derivation from two summary columns
_SUMMARY_SHOTS = "Performance Sh"
_SUMMARY_SOT = "Performance SoT"
_SUMMARY_PK = "Performance PK"
_SUMMARY_PKATT = "Performance PKatt"

FBREF_KEEPER_MAP: dict[str, str] = {
    "saves": "Shot Stopping Saves",
    "penalties_saved": "Penalty Kicks PKsv",
}


def _num(raw: Mapping, key: str, default: float = 0.0) -> float:
    v = raw.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def map_summary_row(raw: Mapping) -> dict:
    """FBref match-summary row (flattened dict) → PlayerMatchEvents field dict.

    Pure and network-free. Unknown metrics are omitted so the ORM defaults
    (0) apply. Derives shots-off-target and missed penalties from paired cols.
    """
    out: dict[str, float] = {}
    for field, col in FBREF_SUMMARY_MAP.items():
        if col in raw and raw[col] is not None:
            out[field] = _num(raw, col)
    shots = _num(raw, _SUMMARY_SHOTS)
    sot = _num(raw, _SUMMARY_SOT)
    if shots or sot:
        out["shots_off_target"] = max(int(shots) - int(sot), 0)
    pk = _num(raw, _SUMMARY_PK)
    pkatt = _num(raw, _SUMMARY_PKATT)
    if pkatt:
        out["penalties_missed"] = max(int(pkatt) - int(pk), 0)
    # ints for everything except the percentage
    return {
        k: (round(v, 1) if k == "pass_completion_pct" else int(v))
        for k, v in out.items()
    }


def map_keeper_row(raw: Mapping) -> dict:
    """FBref match-keeper row (flattened dict) → PlayerMatchEvents field dict."""
    out: dict[str, int] = {}
    for field, col in FBREF_KEEPER_MAP.items():
        if col in raw and raw[col] is not None:
            out[field] = int(_num(raw, col))
    return out


def normalize_position(fbref_pos: str | None) -> str:
    """FBref position token (e.g. 'GK', 'CB', 'DF,MF', 'FW') → FPL group."""
    if not fbref_pos:
        return "MID"
    token = str(fbref_pos).split(",")[0].strip().upper()
    if token in ("GK",):
        return "GK"
    if token in ("DF", "CB", "LB", "RB", "WB", "DEF"):
        return "DEF"
    if token in ("FW", "ST", "CF", "FWD"):
        return "FWD"
    return "MID"


def _flatten_columns(df) -> object:  # pragma: no cover - needs pandas + live df
    """Join a soccerdata MultiIndex column into '<Section> <Leaf>' strings.
    Single-level columns pass through unchanged."""
    import pandas as pd

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            " ".join(str(p) for p in tup if str(p) and "Unnamed" not in str(p)).strip()
            for tup in df.columns
        ]
    return df


def _build_name_map() -> dict[str, int]:
    """Player display-name → players.id, mirroring the understat matcher."""
    db = get_session()
    try:
        name_map: dict[str, int] = {}
        for p in db.query(Player).all():
            name_map[f"{p.first_name} {p.second_name}".lower()] = p.id
            name_map[p.web_name.lower()] = p.id
        return name_map
    finally:
        db.close()


def _match_player(name: str, name_map: dict[str, int]) -> int | None:
    key = name.strip().lower()
    if key in name_map:
        return name_map[key]
    for cand, pid in name_map.items():
        if key in cand or cand in key:
            return pid
    return None


def ingest_fbref_season(  # pragma: no cover - live network + browser only
    season: str,
    *,
    no_cache: bool = False,
) -> tuple[int, int]:
    """Scrape one PL season's per-match player events into player_match_events.

    Requires ``soccerdata`` + a browser (see module docstring). Returns
    ``(rows_written, players_unmatched)``. UNTESTED against a live browser in
    this environment — the pure mappers above carry the tested logic.
    """
    try:
        import soccerdata as sd
    except ImportError as exc:  # keep the failure actionable
        raise ImportError(
            "fbref ingest needs soccerdata (+ a browser), intentionally not a core "
            "dependency. Run the scrape with soccerdata layered on for that run:\n"
            "    DB_PATH=fpl_bot_v2.db uv run --with soccerdata "
            "python scripts/scrape_fbref.py 2025-26\n"
            "(`uv run` re-syncs the venv from pyproject, so a bare `uv pip install "
            "soccerdata` doesn't survive to the run — `--with` is the robust form)."
        ) from exc

    sd_season = SEASON_MAP.get(season)
    if not sd_season:
        raise ValueError(f"No FBref season mapping for {season!r}")

    fbref = sd.FBref(leagues=FBREF_LEAGUE, seasons=sd_season, no_cache=no_cache)
    schedule = fbref.read_schedule()
    summary = _flatten_columns(fbref.read_player_match_stats(stat_type="summary"))
    keepers = _flatten_columns(fbref.read_player_match_stats(stat_type="keepers"))

    # game_id -> gameweek from the schedule (FBref 'week'/'round' column).
    gw_of = _schedule_gameweeks(schedule)
    name_map = _build_name_map()

    keeper_by_key = _index_keeper_rows(keepers)
    rows, unmatched = _assemble_rows(season, summary, keeper_by_key, gw_of, name_map)

    written = _write_events(rows)
    logger.info("FBref %s: %d event rows written, %d unmatched", season, written, unmatched)
    return written, unmatched


def _schedule_gameweeks(schedule) -> dict[str, int | None]:  # pragma: no cover
    out: dict[str, int | None] = {}
    week_col = next((c for c in ("week", "round", "Wk") if c in schedule.columns), None)
    for game_id, row in zip(schedule["game_id"], schedule.to_dict("records"), strict=False):
        wk = row.get(week_col) if week_col else None
        try:
            out[game_id] = int(wk) if wk is not None else None
        except (TypeError, ValueError):
            out[game_id] = None
    return out


def _index_keeper_rows(keepers) -> dict[tuple, dict]:  # pragma: no cover
    idx: dict[tuple, dict] = {}
    for rec in keepers.reset_index().to_dict("records"):
        idx[(rec.get("game_id"), str(rec.get("player", "")).lower())] = rec
    return idx


def _assemble_rows(  # pragma: no cover - exercised by the live path
    season: str,
    summary,
    keeper_by_key: dict[tuple, dict],
    gw_of: dict[str, int | None],
    name_map: dict[str, int],
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    unmatched = 0
    for rec in summary.reset_index().to_dict("records"):
        name = str(rec.get("player", ""))
        player_id = _match_player(name, name_map)
        if not player_id:
            unmatched += 1
            continue
        game_id = rec.get("game_id")
        values = {
            "player_id": player_id,
            "season": season,
            "gameweek": gw_of.get(game_id),
            "game_id": game_id,
            "position": normalize_position(rec.get("pos")),
            "source": "fbref",
            **map_summary_row(rec),
        }
        keeper = keeper_by_key.get((game_id, name.lower()))
        if keeper:
            values.update(map_keeper_row(keeper))
        rows.append(values)
    return rows, unmatched


def _write_events(rows: list[dict]) -> int:  # pragma: no cover - live DB write
    db = get_session()
    written = 0
    try:
        for values in rows:
            stmt = (
                insert(PlayerMatchEvents)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=["player_id", "season", "game_id"]
                )
            )
            written += db.execute(stmt).rowcount or 0
        db.commit()
    finally:
        db.close()
    return written
