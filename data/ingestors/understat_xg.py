"""understat_xg.py — per-match xG/xA/key-passes from Understat → player_xg_stats.

The FREE solution to the xG gap: soccerdata's Understat reader returns real
per-player-per-match ``xg``/``xa``/``key_passes``/``shots`` (TLS-client, no
browser, no API key), which is exactly what P3 (goals) and P4 (assists) want.
This replaces the shots-only interim: P3's weight becomes real xG, P4's becomes
real xA/key-passes.

Player rows are matched to FPL ids by name; each match is assigned to a
gameweek via the per-season deadlines (T3a). Pure parsers are unit-tested; only
``ingest_understat_xg_season`` needs soccerdata + network.

``npxg`` is REAL non-penalty xG (2026-08-16). The player-match feed carries
only total ``xg``, so npxg used to be stored equal to it -- which meant
anything treating the pair as a decomposition (non-penalty xG plus penalty
duty) silently double-counted a taker's penalties. The shot-event feed does
have the split, one row per shot with a ``situation``, so npxg is summed from
a player's non-penalty shots in that match. ``xa`` is Understat
expected-assists.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from data.ingestors.fbref import (
    _build_name_map,
    _match_player,
    _write_xg_rows,
    aggregate_xg_rows,
)
from scripts.backfill_odds import assign_gameweek

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "ENG-Premier League"
SEASON_MAP = {
    "2021-22": "2021", "2022-23": "2022", "2023-24": "2023",
    "2024-25": "2024", "2025-26": "2025", "2026-27": "2026",
}


def parse_game_date(game: str) -> datetime | None:
    """Understat ``game`` label starts 'YYYY-MM-DD ...' → date (fallback when a
    kickoff time isn't joined from the schedule)."""
    if not game:
        return None
    token = str(game).strip().split(" ")[0]
    try:
        return datetime.strptime(token, "%Y-%m-%d")
    except ValueError:
        return None


# Understat prices every penalty identically. soccerdata's shot-event reader
# has no "Penalty" label in its `situation` mapping, so Understat's penalty
# situation arrives as NULL -- verified against 2025-26, where ALL 92
# null-situation shots carry this xG and no other shot has a null situation.
_PENALTY_XG_LOW = 0.74
_PENALTY_XG_HIGH = 0.79


def is_penalty_shot(situation: object, xg: float) -> bool:
    """Is this shot-event row a penalty? (pure)

    Two ways in, deliberately. An explicit ``Penalty`` situation is trusted
    outright, so this keeps working if soccerdata starts labelling them.
    Otherwise a NULL situation counts only when the xG also matches
    Understat's fixed penalty price -- a null that appears for some other
    reason then falls through as open play rather than silently stripping
    real xG out of a player's non-penalty total.
    """
    label = str(situation).strip().lower() if situation is not None else ""
    if label == "penalty":
        return True
    is_null = situation is None or label in ("", "nan", "none")
    return is_null and _PENALTY_XG_LOW <= float(xg or 0.0) <= _PENALTY_XG_HIGH


def aggregate_npxg(shot_rows: list[dict]) -> dict[tuple[object, object], float]:
    """(understat player_id, game_id) -> non-penalty xG, from shot events.

    Players with no shots in a match simply have no key; the caller treats
    that as 0.0, which is correct -- no shots means no non-penalty xG.
    """
    out: dict[tuple[object, object], float] = {}
    for row in shot_rows:
        if is_penalty_shot(row.get("situation"), row.get("xg", 0.0)):
            continue
        key = (row.get("player_id"), row.get("game_id"))
        out[key] = out.get(key, 0.0) + float(row.get("xg", 0.0) or 0.0)
    return out


def understat_row_to_xg(row: dict, npxg: float | None = None) -> dict:
    """One Understat player-match row → player_xg_stats field dict (pure).

    ``npxg`` comes from the shot events (see ``aggregate_npxg``). Passing
    ``None`` means the shot feed was unavailable, and npxg falls back to
    total xg — the pre-2026-08-16 behaviour, which over-counts penalty
    takers. That fallback is reported by the caller and caught by
    ``data/quality_checks.py::check_column_is_not_a_copy``, so it cannot go
    unnoticed the way it did before.
    """
    xg = round(float(row.get("xg", 0.0) or 0.0), 4)
    return {
        "xg": xg,
        # never above total xg: a shot-event sum that exceeds the match feed's
        # own total means the two disagree, and the total is authoritative.
        "npxg": xg if npxg is None else round(min(float(npxg), xg), 4),
        "xa": round(float(row.get("xa", 0.0) or 0.0), 4),
        "shots": int(row.get("shots", 0) or 0),
        "key_passes": int(row.get("key_passes", 0) or 0),
    }


def _load_deadlines(season: str) -> dict[int, datetime]:
    from data.db import get_session
    from data.models import Gameweek
    db = get_session()
    try:
        rows = db.query(Gameweek.id, Gameweek.deadline_time).filter(
            Gameweek.season == season
        ).all()
        return {gw: dl for gw, dl in rows if dl is not None}
    finally:
        db.close()


class UnderstatScheduleUnreadable(RuntimeError):
    """Understat's schedule could not be read at all.

    Deliberately NOT the same state as "this season is not published yet".
    Conflating the two is what hid a live gap: on 2026-08-28 two runs
    reported 2026-27 unpublished -- and said so reassuringly -- while the
    season was live, the match pages existed, and ``player_xg_stats`` held
    2 non-zero xg rows out of 309. The attacking signal was switched off for
    the live season and the log said everything was fine.
    """


# The failure observed on 2026-08-28 was transient: identical calls read 380
# fixtures cleanly about ninety minutes later, from a cold cache. A couple of
# retries cost seconds and cover exactly that.
_SCHEDULE_READ_ATTEMPTS = 3
_SCHEDULE_RETRY_SECONDS = 5.0


def _season_is_published(
    us,
    *,
    attempts: int = _SCHEDULE_READ_ATTEMPTS,
    sleep_seconds: float = _SCHEDULE_RETRY_SECONDS,
) -> bool:
    """Has Understat published this season yet?

    A published season returns a schedule indexed by (league, season, game).
    An unpublished one returns a degenerate frame -- no columns, a single
    unnamed index level -- which every downstream reader then crashes on. The
    check is on the index SHAPE rather than on row count, because the
    degenerate frame is not empty: it has three rows holding the level names
    themselves.

    Raises ``UnderstatScheduleUnreadable`` if the schedule cannot be read
    after ``attempts`` tries. That propagates out of the ingest and exits the
    script non-zero, so ``run_weekly.py``'s warn-and-continue wrapper names
    the failing step instead of the run reporting success having ingested
    nothing.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            schedule = us.read_schedule()
        except Exception as exc:  # noqa: BLE001 -- retried, then re-raised typed
            last_exc = exc
            logger.warning(
                "Understat schedule read failed (attempt %d/%d): %s",
                attempt, attempts, exc,
            )
            if attempt < attempts:
                time.sleep(sleep_seconds)
            continue
        return schedule.index.nlevels >= 3 and not schedule.empty
    raise UnderstatScheduleUnreadable(
        f"Understat schedule unreadable after {attempts} attempts: {last_exc}"
    ) from last_exc


def ingest_understat_xg_season(  # pragma: no cover - live network (no browser)
    season: str,
    *,
    no_cache: bool = False,
) -> tuple[int, int]:
    """Scrape one PL season of Understat per-match xG → player_xg_stats.
    Returns (rows_written, unmatched). Browserless (TLS client)."""
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError(
            "understat xg ingest needs soccerdata: "
            "`uv run --with soccerdata python scripts/scrape_understat_xg.py 2025-26`"
        ) from exc

    yr = SEASON_MAP.get(season)
    if not yr:
        raise ValueError(f"No Understat season mapping for {season!r}")

    us = sd.Understat(leagues=UNDERSTAT_LEAGUE, seasons=yr, no_cache=no_cache)

    # Understat publishes a season only once it has data to publish, and it
    # lags the Premier League's own start by some weeks. Until then the
    # schedule comes back DEGENERATE rather than empty -- a frame whose single
    # unnamed index level holds the literal strings 'league', 'season', 'game'
    # -- and soccerdata's read_player_match_stats raises
    # "too many values to unpack (expected 3)" trying to destructure it
    # (confirmed for 2026-27 on 2026-08-25, one gameweek into the season).
    #
    # That is not a failure worth propagating: this ingest is wired into
    # run_weekly.py, which runs every week from GW1 onward, and a traceback
    # there reads as a broken pipeline rather than "the source is not live
    # yet". Return an explicit no-op instead, and say why.
    if not _season_is_published(us):
        logger.warning(
            "Understat has no data for %s yet -- skipping the xG refresh. "
            "This resolves itself once Understat publishes the season; it is "
            "a source lag, not a failure. Prior seasons are unaffected.",
            season,
        )
        return 0, 0

    pm = us.read_player_match_stats().reset_index()
    # Shot events are what make npxg real -- the player-match feed has only
    # total xg. Degrading to npxg == xg is visible (logged here, and flagged
    # weekly by the copied-column quality check) rather than silent.
    try:
        shots = us.read_shot_events().reset_index()
        npxg_by_key = aggregate_npxg(shots.to_dict("records"))
        logger.info("Understat shot events: npxg resolved for %d player-matches",
                    len(npxg_by_key))
    except Exception as exc:
        npxg_by_key = {}
        logger.warning(
            "Understat shot events unavailable (%s) — npxg falls back to total "
            "xg, which OVER-COUNTS penalty takers and invalidates any "
            "non-penalty decomposition downstream", exc,
        )
    schedule = us.read_schedule().reset_index()
    kickoff_of = dict(zip(schedule["game_id"], schedule["date"], strict=False))

    deadlines = _load_deadlines(season)
    name_map = _build_name_map()

    per_match: list[tuple[int, int, dict]] = []
    unmatched = 0
    for rec in pm.to_dict("records"):
        player_id = _match_player(str(rec.get("player", "")), name_map)
        kickoff = kickoff_of.get(rec.get("game_id"))
        if kickoff is None:
            kickoff = parse_game_date(rec.get("game", ""))
        gw = assign_gameweek(kickoff, deadlines) if kickoff is not None else None
        if not player_id or gw is None:
            unmatched += 1
            continue
        npxg = npxg_by_key.get((rec.get("player_id"), rec.get("game_id")))
        if not npxg_by_key:
            npxg = None  # no shot feed at all: keep the documented fallback
        elif npxg is None:
            npxg = 0.0   # had shot data, this player took none in this match
        per_match.append((player_id, int(gw), understat_row_to_xg(rec, npxg)))

    written = _write_xg_rows(season, aggregate_xg_rows(per_match))
    logger.info(
        "Understat xg %s: %d player-GW rows written, %d unmatched",
        season, written, unmatched,
    )
    return written, unmatched
