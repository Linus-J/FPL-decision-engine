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

Coverage reality (the sanity harness's tolerance; verified against real 25/26
cached match pages 2026-07-25 — the "may need a one-line correction" above
already applied once: the live column is "Performance TklW" i.e. tackles WON,
not "Tkl"). Available → goals, assists, tackles(won), interceptions,
take-ons, passing accuracy, cards, penalties, shots, saves. Not available →
blocks, clearances, recoveries, crosses, key passes, big chances,
errors-leading-to-goal/shot, own goals — the modern FBref summary table simply
has no such columns (confirmed by inspecting the raw flattened headers, not
assumed). Those default to 0.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Mapping

from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.models import Player, PlayerMatchEvents, PlayerXGStats

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
    "tackles": "Performance TklW",        # summary only carries tackles WON
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


# per-MATCH xG columns from the summary table → player_xg_stats fields (P3/P4).
FBREF_XG_MAP: dict[str, str] = {
    "xg": "Expected xG",
    "npxg": "Expected npxG",
    "xa": "Expected xAG",          # FBref xAG ≈ expected assists
    "shots": "Performance Sh",
}


def map_xg_row(raw: Mapping) -> dict:
    """FBref match-summary row → per-match xG fields (floats; shots int)."""
    out: dict[str, float] = {}
    for field, col in FBREF_XG_MAP.items():
        if col in raw and raw[col] is not None:
            out[field] = _num(raw, col)
    return {
        "xg": round(out.get("xg", 0.0), 4),
        "xa": round(out.get("xa", 0.0), 4),
        "npxg": round(out.get("npxg", 0.0), 4),
        "shots": int(out.get("shots", 0)),
    }


def aggregate_xg_rows(
    per_match: list[tuple[int, int, dict]],
) -> dict[tuple[int, int], dict]:
    """Sum per-match xG into per (player_id, gameweek) totals — a DGW player's
    two matches in one GW combine (player_xg_stats is keyed per GW). Input is
    ``(player_id, gameweek, xg_fields)`` triples."""
    agg: dict[tuple[int, int], dict] = {}
    for player_id, gw, fields in per_match:
        key = (player_id, gw)
        cur = agg.setdefault(
            key, {"xg": 0.0, "xa": 0.0, "npxg": 0.0, "shots": 0, "key_passes": 0}
        )
        cur["xg"] += fields.get("xg", 0.0)
        cur["xa"] += fields.get("xa", 0.0)
        cur["npxg"] += fields.get("npxg", 0.0)
        cur["shots"] += fields.get("shots", 0)
        cur["key_passes"] += fields.get("key_passes", 0)
    for cur in agg.values():
        cur["xg"] = round(cur["xg"], 4)
        cur["xa"] = round(cur["xa"], 4)
        cur["npxg"] = round(cur["npxg"], 4)
        cur["xgi"] = round(cur["xg"] + cur["xa"], 4)
    return agg


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


# Non-decomposing Latin letters NFKD can't strip to ASCII (they're independent
# base characters, not a letter + combining accent, so NFKD+ascii-encode would
# otherwise silently DROP them rather than transliterate) -- Turkish, Nordic,
# Polish, Czech-ish characters that show up in real PL rosters (e.g. Ferdi
# "Kadıoğlu" -- confirmed live: NFKD alone turns this into "Kadoglu", losing
# the dotless-i entirely, vs the correct "Kadioglu" both external sources use).
_NON_DECOMPOSING_TRANSLIT = str.maketrans({
    "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G", "ş": "s", "Ş": "S",
    "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
})


def _normalize_name(name: str) -> str:
    """Lowercased, diacritic-stripped name for cross-source matching. NFKD
    handles most Latin accents (é→e, ñ→n, ç→c, ...); ``_NON_DECOMPOSING_TRANSLIT``
    covers the ones NFKD can't."""
    translated = name.translate(_NON_DECOMPOSING_TRANSLIT)
    decomposed = unicodedata.normalize("NFKD", translated)
    return decomposed.encode("ascii", "ignore").decode().strip().lower()


def _build_name_map() -> dict[str, int]:
    """Player display-name → players.id, mirroring the understat matcher."""
    db = get_session()
    try:
        name_map: dict[str, int] = {}
        for p in db.query(Player).all():
            name_map[_normalize_name(f"{p.first_name} {p.second_name}")] = p.id
            name_map[_normalize_name(p.web_name)] = p.id
        return name_map
    finally:
        db.close()


def _match_player(name: str, name_map: dict[str, int]) -> int | None:
    """Real bugs found 2026-07-28 (walk-forward gate investigation):

    1. Plain substring containment misses any player whose STORED full legal
       name carries a middle name or a second surname the external source
       drops -- e.g. FBref/Understat's "Bruno Fernandes" vs our stored
       "Bruno Borges Fernandes", or "Nico Gonzalez" vs stored "Gonzalez
       Iglesias" (Iberian dual-surname convention) -- neither is a
       CONTIGUOUS substring of the other, so both directions of the old
       check failed and 21+ significant players (some of them this bot's
       own most-favoured captains) silently got ZERO event/xG data for the
       entire season. The token-subset fallback below catches this: a
       commonly-used football name is almost always a SUBSET of the full
       legal name's tokens, regardless of word order or what's dropped in
       between.

    2. A FAR WORSE, actively-corrupting bug in the OLD substring check
       itself (pre-dating today, not introduced by fix #1): a short,
       generic SINGLE-TOKEN ``web_name`` (e.g. "Gabriel", Arsenal's Gabriel
       Magalhães) is trivially "in" any longer external name that happens
       to start with the same common first name -- "Gabriel Martinelli"
       and "Gabriel Jesus" were both silently merging their real xG/xA into
       Arsenal's Gabriel MAGALHÃES, a centre-back, whose season xG totals
       included single-match readings above 1.5 (striker-level, impossible
       for a CB) -- confirmed live. Fixed by requiring at least 2 tokens on
       the CANDIDATE side of the substring check; single-token web_names
       can still match via the EXACT-match branch above (a bare "Gabriel"
       query correctly resolves to him), just never via fuzzy containment
       against a longer, different person's name.
    """
    key = _normalize_name(name)
    if key in name_map:
        return name_map[key]

    for cand, pid in name_map.items():
        if len(cand.split()) < 2:
            continue
        if key in cand or cand in key:
            return pid

    key_tokens = set(key.split())
    if len(key_tokens) >= 2:
        for cand, pid in name_map.items():
            cand_tokens = set(cand.split())
            if len(cand_tokens) >= 2 and key_tokens <= cand_tokens:
                return pid
    return None


def ingest_fbref_season(  # pragma: no cover - live network + browser only
    season: str,
    *,
    no_cache: bool = False,
    path_to_browser: str | None = None,
    headless: bool = True,
) -> tuple[int, int]:
    """Scrape one PL season's per-match player events into player_match_events.

    Requires ``soccerdata`` + a Chromium/Chrome browser (FBref's reader drives
    SeleniumBase in undetected-chromedriver mode — Chrome-only, no Firefox).
    ``path_to_browser`` overrides auto-detection (e.g. ``/usr/bin/chromium``);
    ``headless=False`` runs headed, which often clears Cloudflare when headless
    is blocked. Returns ``(rows_written, players_unmatched)``. UNTESTED against a
    live browser here — the pure mappers above carry the tested logic.
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

    fbref_kwargs: dict = {"leagues": FBREF_LEAGUE, "seasons": sd_season,
                          "no_cache": no_cache, "headless": headless}
    if path_to_browser:
        fbref_kwargs["path_to_browser"] = path_to_browser
    fbref = sd.FBref(**fbref_kwargs)
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


def ingest_fbref_xg_season(  # pragma: no cover - live network + browser (cache-backed)
    season: str,
    *,
    no_cache: bool = False,
    path_to_browser: str | None = None,
    headless: bool = True,
) -> tuple[int, int]:
    """Per-match npxG/xG/xAG/shots from FBref summary → player_xg_stats (P3/P4).

    Reuses the SAME summary pages the event scrape cached, so re-running on an
    already-scraped season is a cache hit (no browser). Real point-in-time
    per-GW xG (DGW matches summed per gameweek). Returns (rows_written, unmatched).
    """
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError(
            "fbref xg ingest needs soccerdata (+ Chromium for uncached seasons). "
            "See scripts/scrape_fbref.py."
        ) from exc

    sd_season = SEASON_MAP.get(season)
    if not sd_season:
        raise ValueError(f"No FBref season mapping for {season!r}")

    kwargs: dict = {"leagues": FBREF_LEAGUE, "seasons": sd_season,
                    "no_cache": no_cache, "headless": headless}
    if path_to_browser:
        kwargs["path_to_browser"] = path_to_browser
    fbref = sd.FBref(**kwargs)
    schedule = fbref.read_schedule()
    summary = _flatten_columns(fbref.read_player_match_stats(stat_type="summary"))
    gw_of = _schedule_gameweeks(schedule)
    name_map = _build_name_map()

    per_match: list[tuple[int, int, dict]] = []
    unmatched = 0
    for rec in summary.reset_index().to_dict("records"):
        player_id = _match_player(str(rec.get("player", "")), name_map)
        gw = gw_of.get(rec.get("game_id"))
        if not player_id or gw is None:
            unmatched += 1
            continue
        per_match.append((player_id, int(gw), map_xg_row(rec)))

    written = _write_xg_rows(season, aggregate_xg_rows(per_match))
    logger.info("FBref xg %s: %d player-GW rows written, %d unmatched", season, written, unmatched)
    return written, unmatched


def _write_xg_rows(  # pragma: no cover - live DB write
    season: str, agg: dict[tuple[int, int], dict]
) -> int:
    db = get_session()
    written = 0
    try:
        for (player_id, gw), fields in agg.items():
            stmt = (
                insert(PlayerXGStats)
                .values(player_id=player_id, gameweek=gw, season=season, **fields)
                .on_conflict_do_update(
                    index_elements=["player_id", "gameweek", "season"],
                    set_=fields,
                )
            )
            written += db.execute(stmt).rowcount or 0
        db.commit()
    finally:
        db.close()
    return written
