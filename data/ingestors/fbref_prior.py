"""fbref_prior.py — prior-league SEASON stats → prior_league_stats (P11).

Unlike the match-level PL scrape (fbref.py), the cold-start prior only needs
season-aggregate per-90 rates from a player's *prior* league, so this reads
``read_player_season_stats`` — one light request per (league, stat_type), not a
380-match grind. Top-5 leagues are registered in soccerdata out of the box;
ENG-Championship needs a ``~/soccerdata/config/league_dict.json`` entry (see
scripts/scrape_prior_league.py).

Pure ``compute_per90`` is unit-tested; the live read (``ingest_prior_league_season``)
is browser-only and ``pragma: no cover``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.models import PriorLeagueStats

logger = logging.getLogger(__name__)

# our league label -> soccerdata league id (top-5 are OOTB; Championship needs
# a custom league_dict.json entry registering "ENG-Championship").
PRIOR_LEAGUES = {
    "ENG-Championship": "ENG-Championship",
    "ESP-La Liga": "ESP-La Liga",
    "ITA-Serie A": "ITA-Serie A",
    "GER-Bundesliga": "GER-Bundesliga",
    "FRA-Ligue 1": "FRA-Ligue 1",
}

# flattened FBref season-standard columns ("<Section> <Leaf>"), first present wins
_MIN = ("Playing Time Min", "Min")
_MP = ("Playing Time MP", "MP")
_GLS = ("Performance Gls", "Gls")
_AST = ("Performance Ast", "Ast")
_NPXG = ("Expected npxG", "npxG")
_XAG = ("Expected xAG", "xAG", "Expected xA", "xA")


def _first(row: Mapping, keys: tuple[str, ...], default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return default


def compute_per90(
    minutes: float, goals: float, assists: float, npxg: float, xa: float
) -> dict[str, float]:
    """Season totals → per-90 rates. Pure/testable. Zero minutes → zero rates
    (a player with no minutes carries no signal, not a divide-by-zero)."""
    if minutes <= 0:
        return {"goals90": 0.0, "assists90": 0.0, "npxg90": 0.0, "xa90": 0.0}
    factor = 90.0 / minutes
    return {
        "goals90": round(goals * factor, 4),
        "assists90": round(assists * factor, 4),
        "npxg90": round(npxg * factor, 4),
        "xa90": round(xa * factor, 4),
    }


def row_to_prior_stats(row: Mapping, league: str, season: str) -> dict | None:
    """One flattened FBref season-stats row → a prior_league_stats value dict.
    Returns None for rows with no minutes (nothing to learn from)."""
    minutes = _first(row, _MIN)
    if minutes <= 0:
        return None
    per90 = compute_per90(
        minutes, _first(row, _GLS), _first(row, _AST), _first(row, _NPXG), _first(row, _XAG)
    )
    return {
        "player_name": str(row.get("player", "")).strip(),
        "team": str(row.get("team", "")).strip(),
        "league": league,
        "season": season,
        "position": str(row.get("pos", "")).strip()[:8],
        "minutes": int(minutes),
        "matches": int(_first(row, _MP)),
        **per90,
    }


def ingest_prior_league_season(  # pragma: no cover - live network + browser
    league: str,
    season: str,
    *,
    no_cache: bool = False,
    path_to_browser: str | None = None,
    headless: bool = True,
) -> int:
    """Scrape one league-season of FBref season stats → prior_league_stats.
    Returns rows written. Needs soccerdata + Chromium (headed recommended)."""
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError(
            "prior-league ingest needs soccerdata (+ Chromium). Run via:\n"
            "  FBREF_HEADED=1 DB_PATH=fpl_bot_v2.db uv run --with soccerdata "
            "python scripts/scrape_prior_league.py 'ESP-La Liga' 2025-2026"
        ) from exc

    from data.ingestors.fbref import _flatten_columns  # reuse the MultiIndex flattener

    kwargs: dict = {"leagues": league, "seasons": season, "no_cache": no_cache,
                    "headless": headless}
    if path_to_browser:
        kwargs["path_to_browser"] = path_to_browser
    fbref = sd.FBref(**kwargs)
    df = _flatten_columns(fbref.read_player_season_stats(stat_type="standard"))

    rows = []
    for rec in df.reset_index().to_dict("records"):
        vals = row_to_prior_stats(rec, league, season)
        if vals:
            rows.append(vals)

    db = get_session()
    written = 0
    try:
        for vals in rows:
            stmt = (
                insert(PriorLeagueStats)
                .values(**vals)
                .on_conflict_do_update(
                    index_elements=["player_name", "team", "league", "season"],
                    set_={k: vals[k] for k in (
                        "position", "minutes", "matches",
                        "goals90", "assists90", "npxg90", "xa90",
                    )},
                )
            )
            written += db.execute(stmt).rowcount or 0
        db.commit()
    finally:
        db.close()
    logger.info("Prior-league %s %s: %d player rows written", league, season, written)
    return written
