"""understat_prior.py — expected-goals metrics for the prior-league tier.

``prior_league_stats.npxg90``/``xa90`` were zero on all 15,323 rows because the
FBref season-standard table carries no ``Expected`` column group for any of the
scraped leagues. That was confirmed on 2026-08-17 against a FRESHLY fetched
page (``--no-cache``): its ``data-stat`` attributes contain no xG field at all,
so it is a genuine source limitation, not a parsing or caching bug.

Understat has the data, covers four of the five prior leagues, and needs no
browser -- it is already this project's Premier League xG source, so it is a
dependency we run every week rather than a new one. ENG-Championship is the
exception: Understat does not cover it, so those rows keep falling back to
``npg90`` (realized non-penalty goals).

This module only ever UPDATES ``npxg90``/``xa90`` on rows the FBref scrape
already created, and only for players already matched to an FPL ``code`` --
which is exactly the set ``cold_start.load_prior_league_lookup`` reads. It
never inserts, never touches identity, and never widens the matched set.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from data.db import get_session

logger = logging.getLogger(__name__)

# Our league label -> whether Understat serves it. "ITA-Serie A" needs an
# {"Understat": "Serie A"} entry in ~/soccerdata/config/league_dict.json; the
# other three are registered in soccerdata out of the box.
UNDERSTAT_PRIOR_LEAGUES = (
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
)

# A matched player's Understat and FBref minutes should agree closely -- they
# are the same competition and season. A large gap means the NAME matched the
# wrong player, which would attribute one footballer's xG to another. The
# sources disagree by a few percent on substitute timings, so this is loose
# enough not to reject genuine matches and tight enough to catch a swap.
MINUTES_AGREEMENT_TOLERANCE = 0.25
# Below this, the relative tolerance is meaningless (10 vs 25 minutes is a 150%
# gap between two players who barely featured). Such rows carry no signal
# anyway -- MIN_HOLDOUT_MINUTES excludes them from the cold start.
MIN_MINUTES_FOR_CHECK = 180


def per90(minutes: float, np_xg: float, xa: float) -> tuple[float, float]:
    """(npxg90, xa90). Zero minutes -> zero rates, not a divide-by-zero."""
    if minutes <= 0:
        return 0.0, 0.0
    factor = 90.0 / minutes
    return round(np_xg * factor, 4), round(xa * factor, 4)


def minutes_agree(fbref_minutes: float, understat_minutes: float) -> bool:
    """Do the two sources agree this is the same player's season?

    The guard exists because matching Understat names onto FPL codes is a new
    entity-resolution surface, and this project has already been bitten by
    silent mis-attribution (a set-piece taker's duty landing on a different
    player who shared a first name). Minutes are an independent signal: if two
    sources disagree wildly about how long someone played, the name match is
    not trustworthy enough to write xG from.
    """
    if fbref_minutes < MIN_MINUTES_FOR_CHECK:
        return True
    if understat_minutes <= 0:
        return False
    return abs(understat_minutes - fbref_minutes) / fbref_minutes <= (
        MINUTES_AGREEMENT_TOLERANCE
    )


def _player_name_to_code() -> dict[str, int]:
    """The same map ``fbref_prior.backfill_prior_league_codes`` builds, via the
    same hardened matcher -- not a second, parallel matching implementation."""
    from data.ingestors.fbref import _normalize_name
    from data.models import Player

    db = get_session()
    try:
        name_map: dict[str, int] = {}
        for p in db.query(Player).filter(Player.code.isnot(None)).all():
            name_map[_normalize_name(f"{p.first_name} {p.second_name}")] = p.code
            name_map[_normalize_name(p.web_name)] = p.code
        return name_map
    finally:
        db.close()


def ingest_prior_league_expected(  # pragma: no cover - live network
    league: str, season: str, *, no_cache: bool = False
) -> dict[str, int]:
    """Fill npxg90/xa90 on existing prior_league_stats rows from Understat.

    Returns a counts dict -- ``updated`` is the only number that means the data
    arrived. ``unmatched`` and ``minutes_mismatch`` are reported rather than
    swallowed, because a scrape that writes nothing must not look like one that
    worked (the lesson from ``fbref_prior.report_missing_metrics``).
    """
    import soccerdata as sd

    from data.ingestors.fbref import SEASON_MAP, _match_player

    df = sd.Understat(
        leagues=league, seasons=season, no_cache=no_cache
    ).read_player_season_stats().reset_index()

    stored_season = SEASON_MAP.get(season, season)
    name_map = _player_name_to_code()
    counts = {"rows": len(df), "matched": 0, "unmatched": 0,
              "minutes_mismatch": 0, "updated": 0}

    db = get_session()
    try:
        for r in df.to_dict("records"):
            code = _match_player(str(r.get("player", "")), name_map)
            if code is None:
                counts["unmatched"] += 1
                continue
            counts["matched"] += 1

            existing = db.execute(
                text(
                    "SELECT minutes FROM prior_league_stats "
                    "WHERE code = :code AND league = :league AND season = :season"
                ),
                {"code": code, "league": league, "season": stored_season},
            ).fetchall()
            if not existing:
                continue
            if not all(minutes_agree(float(m[0] or 0), float(r.get("minutes") or 0))
                       for m in existing):
                counts["minutes_mismatch"] += 1
                continue

            npxg90, xa90 = per90(
                float(r.get("minutes") or 0),
                float(r.get("np_xg") or 0.0),
                float(r.get("xa") or 0.0),
            )
            counts["updated"] += db.execute(
                text(
                    "UPDATE prior_league_stats SET npxg90 = :npxg90, xa90 = :xa90 "
                    "WHERE code = :code AND league = :league AND season = :season"
                ),
                {"npxg90": npxg90, "xa90": xa90, "code": code,
                 "league": league, "season": stored_season},
            ).rowcount or 0
        db.commit()
    finally:
        db.close()

    logger.info(
        "Understat %s %s: %d rows -> %d matched, %d updated "
        "(%d unmatched names, %d rejected on minutes disagreement)",
        league, season, counts["rows"], counts["matched"], counts["updated"],
        counts["unmatched"], counts["minutes_mismatch"],
    )
    if not counts["updated"]:
        logger.warning(
            "Understat %s %s updated NOTHING -- npxg90/xa90 remain whatever "
            "they were. Do not read this run as a repopulation.", league, season,
        )
    return counts
