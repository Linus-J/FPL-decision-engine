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
not "Tkl"). Corrected 2026-08-18 against soccerdata's own parsed frame, which is the only
thing that matters here -- the previous list was wrong in both directions.

Available → goals, assists, tackles(WON), interceptions, take-ons, cards,
penalties (taken/won/missed), shots, saves, AND -- previously read as
unavailable and simply discarded -- fouls, offsides, crosses, own goals.

Not available → blocks, clearances, recoveries (WhoScored's event stream
supplies these), key passes, big chances, errors-leading-to-goal/shot.

**Passing accuracy is NOT available**, despite this docstring having claimed it
was and FBREF_SUMMARY_MAP having mapped it. soccerdata's
``read_player_match_stats`` accepts only ``['summary', 'keepers']``; there is
no match-level passing table to fetch. That matters more than it looks: pass
completion is worth up to +6 BPS, the largest positive component available to
an outfielder, and its absence is why defenders are under-credited for bonus
once the tackle count is counted correctly (see data/ingestors/whoscored.py).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.models import Player, PlayerMatchEvents, PlayerXGStats

logger = logging.getLogger(__name__)

FBREF_LEAGUE = "ENG-Premier League"

# season string (our format) -> soccerdata season string
# soccerdata's cache root, honouring SOCCERDATA_DIR when the user has set it.
# Imported lazily-ish at module scope because it is only a path constant.
try:
    from soccerdata._config import DATA_DIR as SD_DATA_DIR
except Exception:  # noqa: BLE001 -- soccerdata is optional at import time
    SD_DATA_DIR = Path.home() / "soccerdata" / "data"

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
    "dribbles": "Take-Ons Succ",          # successful take-ons
    # 2026-08-18: five BPS inputs that were in the downloaded page all along
    # and simply never read -- the same shape as the npg90 finding in the
    # 2026-08-16 audit. Verified against the real cached match pages, whose
    # flattened summary header is exactly:
    #   Player # Nation Pos Age Min | Performance Gls Ast PK PKatt Sh SoT
    #   CrdY CrdR Fls Fld Off Crs TklW Int OG PKwon PKcon
    "fouls": "Performance Fls",              # BPS -1 each
    "offsides": "Performance Off",           # BPS -1 each
    "open_play_crosses": "Performance Crs",  # BPS +1 each
    "own_goals": "Performance OG",           # BPS -6
    "penalties_conceded": "Performance PKcon",  # BPS -3
}
# REMOVED 2026-08-18, and this is why every one of them read 0 on all 11,182
# rows: `map_summary_row` matches column names EXACTLY, and none of these
# columns exists in FBref's match-summary table --
#   "passes": "Passes Att", "pass_completion_pct": "Passes Cmp%",
#   "blocks": "Performance Blocks"
# A mapping that matches nothing is silently omitted and the ORM default (0)
# applies, so a field the codebase believed it collected was indistinguishable
# from a genuine zero. `blocks` does get populated -- by WhoScored's
# BlockedPass events, not from here. Passing volume and completion are simply
# not in this table; they live in FBref's separate "passing" stat type, which
# this ingest does not fetch.
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
    """FBref match-summary row → per-match xG fields (floats; shots int).

    Only the fields FBref ACTUALLY published are returned. An absent column
    must not become a zero here (2026-09-02): FBref's Expected columns are
    published per competition-season, and 2026-27's match reports carry 30
    stat columns with no xG/npxG/xAG among them. Defaulting those to 0.0 and
    upserting them wrote a confident zero over Understat's real value --
    ``_write_xg_rows`` sets exactly the keys it is given, so a missing key
    now leaves the stored value alone.

    The damage was invisible because ``run_weekly.py`` happens to run FBref
    before Understat, so Understat overwrote the zeros seconds later. Running
    the FBref scrape by hand AFTER a weekly run reversed the order and cut
    2026-27 from 304 rows carrying real xG to 19 -- and
    ``projection/assemble.py`` LEFT JOINs this table and COALESCEs to 0, so a
    zeroed xG is indistinguishable from a genuine one.
    """
    out: dict[str, float] = {}
    for field, col in FBREF_XG_MAP.items():
        if col in raw and raw[col] is not None:
            out[field] = _num(raw, col)
    fields: dict[str, float | int] = {}
    for field in ("xg", "xa", "npxg"):
        if field in out:
            fields[field] = round(out[field], 4)
    if "shots" in out:
        fields["shots"] = int(out["shots"])
    return fields


def aggregate_xg_rows(
    per_match: list[tuple[int, int, dict]],
) -> dict[tuple[int, int], dict]:
    """Sum per-match xG into per (player_id, gameweek) totals — a DGW player's
    two matches in one GW combine (player_xg_stats is keyed per GW). Input is
    ``(player_id, gameweek, xg_fields)`` triples."""
    agg: dict[tuple[int, int], dict] = {}
    seen: dict[tuple[int, int], set[str]] = {}
    for player_id, gw, fields in per_match:
        key = (player_id, gw)
        cur = agg.setdefault(
            key, {"xg": 0.0, "xa": 0.0, "npxg": 0.0, "shots": 0, "key_passes": 0}
        )
        present = seen.setdefault(key, set())
        for field, zero in (
            ("xg", 0.0), ("xa", 0.0), ("npxg", 0.0), ("shots", 0), ("key_passes", 0)
        ):
            if field in fields:
                cur[field] += fields.get(field, zero)
                present.add(field)
    # A field no source row supplied is DROPPED, not written as 0 -- see
    # map_xg_row. Callers upsert exactly these keys, so dropping one leaves
    # whatever another source (Understat) already stored intact.
    out: dict[tuple[int, int], dict] = {}
    for key, cur in agg.items():
        present = seen.get(key, set())
        row: dict = {}
        for field in ("xg", "xa", "npxg"):
            if field in present:
                row[field] = round(cur[field], 4)
        for field in ("shots", "key_passes"):
            if field in present:
                row[field] = cur[field]
        # xgi is xg + xa; it is only meaningful when BOTH were supplied.
        if "xg" in present and "xa" in present:
            row["xgi"] = round(cur["xg"] + cur["xa"], 4)
        if row:
            out[key] = row
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


# Non-decomposing Latin letters NFKD can't strip to ASCII (they're independent
# base characters, not a letter + combining accent, so NFKD+ascii-encode would
# otherwise silently DROP them rather than transliterate) -- Turkish, Nordic,
# Polish, Czech-ish characters that show up in real PL rosters (e.g. Ferdi
# "Kadıoğlu" -- confirmed live: NFKD alone turns this into "Kadoglu", losing
# the dotless-i entirely, vs the correct "Kadioglu" both external sources use).
_NON_DECOMPOSING_TRANSLIT = str.maketrans({
    "ı": "i", "İ": "I", "ğ": "g", "Ğ": "G", "ş": "s", "Ş": "S",
    "ø": "o", "Ø": "O", "ł": "l", "Ł": "L",
    # đ/Đ (Serbo-Croatian d-with-stroke) latinizes to "dj", not bare "d" --
    # dropping the "j" broke "Đorđe" -> "Djordje" matching (real case: FBref's
    # "Djordje Petrovic" vs our stored "Đorđe Petrović").
    "đ": "dj", "Đ": "Dj",
    # ß/æ/œ don't decompose under NFKD either, so the ascii pass DROPPED them
    # rather than transliterating: "Groß" normalised to "gro", a truncated
    # stem that cannot match an English source's "Gross" and could collide
    # with an unrelated name. Same class of bug as đ above (2026-08-16).
    "ß": "ss", "ẞ": "Ss",
    "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe",
})


def _normalize_name(name: str) -> str:
    """Lowercased, diacritic-stripped, hyphen-flattened name for cross-source
    matching. NFKD handles most Latin accents (é→e, ñ→n, ç→c, ...);
    ``_NON_DECOMPOSING_TRANSLIT`` covers the ones NFKD can't. Hyphens are
    flattened to spaces because sources disagree on hyphenation for compound
    names (Understat's "Ben Doak" vs our stored "Ben Gannon-Doak"; "Rayan Ait
    Nouri" vs our stored "Rayan Aït-Nouri") -- flattening lets the token-based
    fallbacks below see each part as its own token either way."""
    translated = name.translate(_NON_DECOMPOSING_TRANSLIT)
    decomposed = unicodedata.normalize("NFKD", translated)
    ascii_only = decomposed.encode("ascii", "ignore").decode()
    return " ".join(ascii_only.replace("-", " ").lower().split())


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


# Nickname / transliteration variants no generic rule can safely cover --
# each verified against a real stored player record during the 2026-07-28
# data-completeness audit (23 unmatched Understat players). Maps an
# external-source spelling to the exact normalized form already in our
# name_map, so it resolves through the ordinary exact/substring/subset
# checks below rather than being special-cased itself. Deliberately NOT a
# fuzzy/edit-distance matcher -- that would reintroduce the same collision
# risk fix #2 below removes (a low-confidence nickname guess merging two
# different players' stats is worse than leaving one unmatched).
_KNOWN_ALIASES: dict[str, str] = {
    "ben white": "benjamin white",
    "matthew cash": "matty cash",
    "joseph gomez": "joe gomez",
    "joshua king": "josh king",
    "oliver scarles": "ollie scarles",
    "alejandro jimenez": "alex jimenez",
    "treymaurice nyoni": "trey nyoni",
    "yeremi pino": "yeremy pino",
    "yehor yarmolyuk": "yehor yarmoliuk",
    "naif aguerd": "nayef aguerd",
    "abduqodir khusanov": "abdukodir khusanov",
    "lucas paqueta": "lucas tolentino coelho de lima",
    "chimuanya ugochukwu": "lesley ugochukwu",
    "max kilman": "maximilian kilman",
    "dan ballard": "daniel ballard",
    "fernando lopez": "fer lopez gonzalez",
    # 2026-08-16, from the allaboutfpl set-piece depth chart: both are
    # single-letter misspellings in the source, each verified against
    # the stored record before being added here.
    "woltemate": "woltemade",
    "devenney": "devenny",
}


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
       between. A second, reversed direction (candidate's tokens are a
       subset of the QUERY's) catches the mirror case -- the external
       source adds an extra given name ours doesn't have, e.g. Understat's
       "Hamed Junior Traore" vs our stored "Hamed Traore".

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
       against a longer, different person's name. The token-subset fallback
       (both directions) only ever returns a match when it is UNIQUE across
       the whole name_map, for the same reason -- an ambiguous subset match
       is treated as no match rather than a guess.
    """
    key = _normalize_name(name)
    key = _KNOWN_ALIASES.get(key, key)
    if key in name_map:
        return name_map[key]

    for cand, pid in name_map.items():
        if len(cand.split()) < 2:
            continue
        if key in cand or cand in key:
            return pid

    key_tokens = set(key.split())
    if len(key_tokens) < 2:
        return None

    matches: set[int] = set()
    for cand, pid in name_map.items():
        cand_tokens = set(cand.split())
        if len(cand_tokens) < 2:
            continue
        if key_tokens <= cand_tokens or cand_tokens <= key_tokens:
            matches.add(pid)
    if len(matches) == 1:
        return matches.pop()
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

    refresh_stale_seasons_cache(sd_season)
    purge_wrong_season_caches(sd_season)

    fbref_kwargs: dict = {"leagues": FBREF_LEAGUE, "seasons": sd_season,
                          "no_cache": no_cache, "headless": headless}
    if path_to_browser:
        fbref_kwargs["path_to_browser"] = path_to_browser
    fbref = sd.FBref(**fbref_kwargs)
    schedule = fbref.read_schedule()
    _validate_schedule_season(schedule, season)
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


def cached_seasons(seasons_html: Path) -> list[str]:
    """Season labels (e.g. '2026-2027') listed in a cached FBref seasons page.

    Empty when the file is missing or unparseable -- callers treat that as "no
    information", never as "the season is absent".
    """
    if not seasons_html.exists():
        return []
    try:
        from lxml import html as lxml_html

        tree = lxml_html.parse(str(seasons_html))
        return [
            str(y).strip()
            for y in tree.xpath(
                "//table[@id='seasons']"
                "//th[@data-stat='year_id' or @data-stat='year']/a/text()"
            )
        ]
    except Exception as exc:  # noqa: BLE001 -- unreadable cache == no information
        logger.debug("Could not read cached FBref seasons page: %s", exc)
        return []


def refresh_stale_seasons_cache(sd_season: str) -> bool:
    """Drop the cached FBref seasons index if it predates ``sd_season``.

    soccerdata resolves a season by parsing every label FBref lists into a
    two-digit code and de-duplicating, keeping the first
    (``soccerdata/fbref.py``: ``drop_duplicates(..., keep="first")``). Both
    '2026-2027' and '1926-1927' parse to '2627'. While the newer season is
    listed it wins, because FBref lists newest first -- but if the cached copy
    of that index PREDATES the new season being added, the only '2627' in it is
    1926-1927, and the lookup silently resolves to a season a century old.

    That is what happened on 2026-08-25: a seasons page cached 2026-07-24, one
    gameweek into 26/27, had 1926-1927 at row 92 and no 2026-2027 at all.
    ``_validate_schedule_season`` caught the result and refused, which was
    right, but its message blamed FBref for not listing the season. FBref lists
    it; our cache was a month old.

    Only this one small index file is removed -- the match-report caches, which
    are the expensive ones, are untouched. Returns True if a refresh was
    triggered.
    """
    seasons_html = Path(SD_DATA_DIR) / "FBref" / f"seasons_{FBREF_LEAGUE}.html"
    listed = cached_seasons(seasons_html)
    if not listed or sd_season in listed:
        return False
    logger.warning(
        "Cached FBref seasons index does not list %s (it has %d seasons, newest "
        "%s) -- removing %s so it is re-fetched. Stale here resolves the season "
        "code to a century-old season rather than failing.",
        sd_season, len(listed), listed[0] if listed else "?", seasons_html.name,
    )
    seasons_html.unlink(missing_ok=True)
    return True


# Signals that a cached page is an interstitial rather than content. Used only
# to explain the removal in the log -- the decision itself is made on whether
# the expected table marker is present, which is definitive.
_BLOCK_PAGE_SIGNALS = (
    "just a moment",
    "consent banner",
    "checking your browser",
    "cf-browser-verification",
    "enable javascript and cookies",
)


def season_code(sd_season: str) -> str:
    """'2025-2026' -> '2526', the code soccerdata uses in cache filenames."""
    try:
        from soccerdata._common import SeasonCode

        return SeasonCode.MULTI_YEAR.parse(sd_season)
    except Exception:  # noqa: BLE001 -- fall back to the same truncation rule
        digits = sd_season.replace("-", "")
        return digits[2:4] + digits[6:8] if len(digits) == 8 else sd_season


def purge_unusable_stats_cache(
    sd_season: str, stat_types: tuple[str, ...], league: str = FBREF_LEAGUE
) -> list[str]:
    """Delete cached FBref season-stat pages that do not contain their table.

    soccerdata caches whatever HTML came back, including Cloudflare
    interstitials, and reads that cache forever after. A poisoned entry then
    fails deep inside the library with ``ValueError: not enough values to
    unpack (expected 1, got 0)`` -- soccerdata's xpath for
    ``div_stats_<stat_type>`` matching nothing -- which says nothing about the
    real cause and cannot be fixed by re-running.

    Found 2026-08-25: ``players_ENG-Premier League_2526_shooting.html``, cached
    2026-08-16, was 132KB of consent banner titled "Close this consent banner"
    with no stats table at all. Every set-piece scrape since had been reading
    it.

    The marker is the same string soccerdata itself looks for, so a page that
    passes here is one it can parse. Returns the stat types purged.
    """
    removed: list[str] = []
    code = season_code(sd_season)
    for stat_type in stat_types:
        path = Path(SD_DATA_DIR) / "FBref" / f"players_{league}_{code}_{stat_type}.html"
        if not path.exists():
            continue
        try:
            text_ = path.read_text(errors="replace")
        except OSError as exc:
            logger.debug("Could not read %s: %s", path.name, exc)
            continue
        if f"div_stats_{stat_type}" in text_:
            continue
        lowered = text_[:200_000].lower()
        blocked = next((m for m in _BLOCK_PAGE_SIGNALS if m in lowered), None)
        logger.warning(
            "Cached FBref %s page for %s has no div_stats_%s table (%d bytes%s) "
            "-- removing it so it is re-fetched. A cached block page fails deep "
            "inside soccerdata with an unpack error that no amount of re-running "
            "will clear.",
            stat_type, sd_season, stat_type, len(text_),
            f", looks like a block page: {blocked!r}" if blocked else "",
        )
        path.unlink(missing_ok=True)
        removed.append(stat_type)
    return removed


def _block_page_signal(text_: str) -> str | None:
    """Which interstitial signature this page matches, if any."""
    lowered = text_[:200_000].lower()
    return next((m for m in _BLOCK_PAGE_SIGNALS if m in lowered), None)


def modal_year(text_: str) -> int | None:
    """The most frequent YYYY among ISO dates in the page, or None."""
    years = [int(m[:4]) for m in re.findall(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", text_)]
    if not years:
        return None
    return Counter(years).most_common(1)[0][0]


def purge_wrong_season_caches(sd_season: str, league: str = FBREF_LEAGUE) -> list[str]:
    """Delete season-keyed FBref caches holding the WRONG season, or a block page.

    Fixing the seasons index (``refresh_stale_seasons_cache``) only fixes which
    URL soccerdata resolves. Anything already downloaded under the ambiguous
    two-digit code stays on disk and is served from cache regardless -- so the
    ingest keeps reading the old season with no further network request, and no
    amount of re-running changes it.

    That is exactly what happened on 2026-08-25: after the seasons index was
    repaired and correctly listed 2026-2027, ``schedule_ENG-Premier
    League_2627.html`` -- written 13:35 that day by the run that resolved to
    1926-27 -- still held 1926/1927 fixtures, and ``teams_ENG-Premier
    League_2627.html`` was a consent wall.

    A schedule is judged by its MODAL date year: the page carries incidental
    recent dates in its chrome, so presence of a 2026 date proves nothing,
    while the year most of its fixtures fall in is decisive. Pages with no
    dates at all are judged only on the block-page signatures.
    """
    purged: list[str] = []
    code = season_code(sd_season)
    start_year = int(sd_season[:4])
    valid_years = {start_year, start_year + 1}
    base = Path(SD_DATA_DIR) / "FBref"

    for kind in ("schedule", "teams"):
        path = base / f"{kind}_{league}_{code}.html"
        if not path.exists():
            continue
        try:
            text_ = path.read_text(errors="replace")
        except OSError as exc:
            logger.debug("Could not read %s: %s", path.name, exc)
            continue

        blocked = _block_page_signal(text_)
        year = modal_year(text_)
        wrong_season = year is not None and year not in valid_years
        if not blocked and not wrong_season:
            continue

        reason = (
            f"its fixtures are mostly from {year}, not {sorted(valid_years)}"
            if wrong_season else f"it looks like a block page: {blocked!r}"
        )
        logger.warning(
            "Cached FBref %s for %s is unusable -- %s. Removing %s so it is "
            "re-fetched; until it is, soccerdata serves it from cache and the "
            "ingest never reaches the network.",
            kind, sd_season, reason, path.name,
        )
        path.unlink(missing_ok=True)
        purged.append(kind)
    return purged


def _validate_schedule_season(schedule, season: str) -> None:  # pragma: no cover - live path
    """Guard against soccerdata's season-code collision: '2026-2027' and
    '1926-1927' both truncate to the same internal code '2627', since the
    parser keeps only the last two digits of each year. If FBref's site
    doesn't list the requested season yet (e.g. it hasn't started), the
    lookup can silently resolve to a decades-old season sharing that code
    instead of raising. Check the schedule's actual match dates fall in the
    requested season's real year range and fail loudly if not."""
    if schedule is None or schedule.empty:
        return
    date_col = next((c for c in ("date", "Date") if c in schedule.columns), None)
    if date_col is None:
        return
    import pandas as pd

    expected_start_year = int(season[:4])
    valid_years = {expected_start_year, expected_start_year + 1}
    actual_years = set(pd.to_datetime(schedule[date_col]).dt.year.dropna().astype(int))
    if actual_years and not actual_years & valid_years:
        raise ValueError(
            f"FBref schedule for season {season!r} has match dates in "
            f"{sorted(actual_years)}, expected {sorted(valid_years)}. "
            "soccerdata's season-code collision has resolved the wrong season: "
            f"'{season[:4]}-{int(season[:4]) + 1}' and "
            f"'{season[:2]}26-{season[:2]}27'-style pairs a century apart share "
            "one two-digit code, and it kept the older one. Refusing to ingest "
            "it.\n"
            "Most likely cause is a STALE cached seasons index -- FBref does "
            "list the current season, but a cached copy taken before it was "
            "added contains only the century-old match. "
            "refresh_stale_seasons_cache() clears that file automatically "
            "before each ingest; if you are seeing this anyway, delete "
            f"{Path(SD_DATA_DIR) / 'FBref' / f'seasons_{FBREF_LEAGUE}.html'} "
            "and re-run."
        )


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


# Key columns -- never part of the update payload below.
#
# `source` is excluded too, deliberately (2026-08-18). FBref runs FIRST and
# WhoScored then patches the same row and stamps 'fbref+whoscored'; letting a
# later FBref re-parse write its own 'fbref' back would erase that, which is
# the provenance defect §4 existed to fix, reintroduced from the other side.
# The value is still set on INSERT, where it is correct by construction.
_EVENT_KEY_COLS = ("player_id", "season", "game_id")
_EVENT_NO_OVERWRITE_COLS = ("source",)


def _write_events(rows: list[dict]) -> int:  # pragma: no cover - live DB write
    """Upsert match events, UPDATING rows that already exist.

    This was ``on_conflict_do_nothing`` until 2026-08-18, which made the whole
    ingest write-once: re-running it could never correct anything. That is not
    a theoretical concern -- it is why fixing the summary-column mapping the
    same day backfilled almost nothing. FBref's table carries fouls, offsides
    and crosses for ~5,000 player-rows a season and the DB had 169, because
    every existing row silently declined the update.

    Only the fields FBref actually SUPPLIED are updated. ``map_summary_row``
    omits what it cannot find, so columns FBref has no opinion on -- notably
    clearances, blocks and recoveries, which come from WhoScored's event
    stream -- are absent from ``values`` and therefore left alone. Without that
    restriction a re-ingest would zero WhoScored's contribution.
    """
    db = get_session()
    written = 0
    try:
        for values in rows:
            updatable = {
                k: v for k, v in values.items()
                if k not in _EVENT_KEY_COLS and k not in _EVENT_NO_OVERWRITE_COLS
            }
            stmt = (
                insert(PlayerMatchEvents)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=list(_EVENT_KEY_COLS),
                    set_=updatable,
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
    _validate_schedule_season(schedule, season)
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
            if not fields:
                # Nothing this source actually measured -- writing the row
                # would only stamp column defaults over another source's data.
                continue
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
