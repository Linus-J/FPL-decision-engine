"""setpiece.py — penalty and set-piece takers (P3.7,
plan/decision-engine-recovery-plan.md).

``player_setpiece_roles`` has existed, and been empty, for the whole project.
``projection/features.py`` already LEFT JOINs it and ``COALESCE``s every field
to zero, so ``is_penalty_taker``, ``penalty_xg_per_game`` and
``is_set_piece_taker`` have silently been 0 for every player, always -- the
plumbing was built and the data never followed, and the COALESCE made its
absence invisible. Penalties are worth several points a season to a taker and
are among the most predictable things in the game, so this was the largest
missing source of real signal in the model.

**Source and method.** FBref's season-level shooting table carries penalties
attempted (``PKatt``) and scored (``PK``); its passing tables carry corner
kicks and key passes. Taker duty is inferred from ATTEMPT SHARE within a
squad rather than from any published depth chart, because attempts are what
actually happened -- a player credited with most of his team's penalties took
them, whatever the pre-season billing said.

**The honest limitation**, stated up front: this measures LAST season's
duties. Penalty responsibility moves with transfers, managers and the odd
public falling-out, and a new signing who took every penalty at his old club
has no PL record at all. Treat it as a strong prior that early-season
evidence should override, not as ground truth -- the same posture
``cold_start.py`` takes toward prior-season form.

Follows ``fbref.py``'s structure deliberately: the mapping logic below is
pure and unit-tested; only ``ingest_setpiece_roles`` needs soccerdata and a
browser, and it is excluded from coverage like its sibling.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import datetime

from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.models import PlayerSetPieceRole

logger = logging.getLogger(__name__)

# A player needs at least this share of his team's penalty attempts to be
# called the taker. Below it, attempts are more likely a stand-in during an
# absence than a duty.
MIN_PENALTY_SHARE = 0.5
# ...and at least this many attempts, so one penalty in a season of ten
# doesn't crown a taker on a single event.
MIN_PENALTY_ATTEMPTS = 2

# Same idea for dead balls, looser: corner duty is more often shared, and a
# secondary taker still carries real assist value.
MIN_SETPIECE_SHARE = 0.3
MIN_SETPIECE_EVENTS = 10

# Long-run conversion rate of a penalty. Used to turn "expected penalties per
# game" into expected goal value; the widely-cited figure is ~0.76-0.79 and
# the exact choice matters far less than having the term at all.
PENALTY_CONVERSION = 0.79


def _num(raw: Mapping, *keys: str, default: float = 0.0) -> float:
    """First present, non-null key wins. FBref's column names vary between
    the flattened season and match tables ('Standard PKatt' vs 'PKatt'), and
    a missing column must degrade to 0 rather than raise."""
    for key in keys:
        if key in raw and raw[key] is not None:
            try:
                value = float(raw[key])
            except (TypeError, ValueError):
                continue
            if value == value:  # not NaN
                return value
    return default


def derive_setpiece_roles(rows: Iterable[Mapping]) -> list[dict]:
    """Per-player set-piece roles from season-level FBref rows.

    Each input row needs ``player``, ``team`` and whatever of
    ``penalty_attempts`` / ``corners`` / ``key_passes`` / ``matches`` is
    available. Shares are computed WITHIN a team, which is the only level at
    which "is he the taker" is a meaningful question.

    Returns one dict per player carrying a role, keyed by name for the
    caller to resolve against ``players`` -- name matching lives in
    ``fbref.py`` and is not duplicated here.
    """
    by_team: dict[str, list[dict]] = {}
    for raw in rows:
        player = (raw.get("player") or "").strip()
        team = (raw.get("team") or "").strip()
        if not player or not team:
            continue
        by_team.setdefault(team, []).append({
            "player": player,
            "team": team,
            "penalty_attempts": _num(raw, "penalty_attempts", "PKatt", "Standard PKatt"),
            "corners": _num(raw, "corners", "CK", "Pass Types CK"),
            "key_passes": _num(raw, "key_passes", "KP"),
            "matches": _num(raw, "matches", "MP", "Playing Time MP", default=0.0),
        })

    roles: list[dict] = []
    for team, players in by_team.items():
        team_pens = sum(p["penalty_attempts"] for p in players)
        team_corners = sum(p["corners"] for p in players)

        for p in players:
            pen_share = p["penalty_attempts"] / team_pens if team_pens else 0.0
            corner_share = p["corners"] / team_corners if team_corners else 0.0
            is_penalty_taker = (
                p["penalty_attempts"] >= MIN_PENALTY_ATTEMPTS
                and pen_share >= MIN_PENALTY_SHARE
            )
            is_setpiece_taker = (
                p["corners"] >= MIN_SETPIECE_EVENTS
                and corner_share >= MIN_SETPIECE_SHARE
            )
            matches = p["matches"] or 0.0
            # Expected penalty GOAL value per game: attempts per game scaled
            # by the long-run conversion rate. Zero for a non-taker, so a
            # backup who took one while the taker was injured doesn't carry
            # a phantom expectation into next season.
            penalty_xg_per_game = (
                (p["penalty_attempts"] / matches) * PENALTY_CONVERSION
                if is_penalty_taker and matches > 0
                else 0.0
            )
            key_passes_per_game = p["key_passes"] / matches if matches > 0 else 0.0

            if not (is_penalty_taker or is_setpiece_taker or key_passes_per_game):
                continue
            roles.append({
                "player": p["player"],
                "team": team,
                "is_penalty_taker": is_penalty_taker,
                "penalty_xg_per_game": round(penalty_xg_per_game, 4),
                "is_set_piece_taker": is_setpiece_taker,
                "key_passes_per_game": round(key_passes_per_game, 4),
            })
    return roles


def write_setpiece_roles(season: str, roles: Iterable[Mapping]) -> int:
    """Upsert resolved roles (each needing a ``player_id``) for ``season``.

    Idempotent on (player_id, season) -- re-scraping mid-season updates the
    duty rather than accumulating rows, which matters because penalty
    responsibility genuinely changes during a season."""
    rows = [r for r in roles if r.get("player_id")]
    if not rows:
        return 0

    db = get_session()
    try:
        for role in rows:
            stmt = (
                insert(PlayerSetPieceRole)
                .values(
                    player_id=int(role["player_id"]),
                    season=season,
                    is_penalty_taker=bool(role.get("is_penalty_taker", False)),
                    penalty_xg_per_game=float(role.get("penalty_xg_per_game", 0.0)),
                    is_set_piece_taker=bool(role.get("is_set_piece_taker", False)),
                    key_passes_per_game=float(role.get("key_passes_per_game", 0.0)),
                    updated_at=datetime.utcnow(),
                )
                .on_conflict_do_update(
                    index_elements=["player_id", "season"],
                    set_={
                        "is_penalty_taker": bool(role.get("is_penalty_taker", False)),
                        "penalty_xg_per_game": float(role.get("penalty_xg_per_game", 0.0)),
                        "is_set_piece_taker": bool(role.get("is_set_piece_taker", False)),
                        "key_passes_per_game": float(role.get("key_passes_per_game", 0.0)),
                        "updated_at": datetime.utcnow(),
                    },
                )
            )
            db.execute(stmt)
        db.commit()
        return len(rows)
    finally:
        db.close()


def ingest_setpiece_roles(  # pragma: no cover - live network + browser only
    season: str,
    *,
    source_season: str | None = None,
    no_cache: bool = False,
    path_to_browser: str | None = None,
    headless: bool = True,
) -> tuple[int, int]:
    """Scrape one season's set-piece duties into ``player_setpiece_roles``.

    ``source_season`` is where the EVIDENCE comes from and defaults to the
    season before ``season`` -- at pre-season there is no current-season
    record to read, so duties are carried across the boundary exactly as
    ``cold_start.py`` carries form. Pass ``source_season == season`` once the
    season is under way to refresh from live evidence.

    Returns ``(rows_written, players_unmatched)``. Needs soccerdata and a
    Chromium/Chrome browser, same constraint as ``fbref.py`` -- FBref sits
    behind Cloudflare and headless is frequently blocked (see
    ``scripts/run_weekly.py``, which forces headed for this reason).
    """
    try:
        import soccerdata as sd
    except ImportError as exc:
        raise ImportError(
            "setpiece ingest needs soccerdata (+ a browser), intentionally not a "
            "core dependency. Layer it on for the run:\n"
            "    DB_PATH=fpl_bot_v2.db uv run --with soccerdata "
            "python scripts/scrape_setpieces.py 2026-27"
        ) from exc

    from data.ingestors.fbref import (
        FBREF_LEAGUE,
        SEASON_MAP,
        _build_name_map,
        _flatten_columns,
        _match_player,
    )
    from projection.cold_start import prior_season_of

    evidence_season = source_season or prior_season_of(season)
    sd_season = SEASON_MAP.get(evidence_season)
    if not sd_season:
        raise ValueError(f"No FBref season mapping for {evidence_season!r}")

    fbref_kwargs: dict = {
        "leagues": FBREF_LEAGUE, "seasons": sd_season,
        "no_cache": no_cache, "headless": headless,
    }
    if path_to_browser:
        fbref_kwargs["path_to_browser"] = path_to_browser
    fbref = sd.FBref(**fbref_kwargs)

    shooting = _flatten_columns(fbref.read_player_season_stats(stat_type="shooting"))
    passing = _flatten_columns(fbref.read_player_season_stats(stat_type="passing"))
    pass_types = _flatten_columns(fbref.read_player_season_stats(stat_type="passing_types"))

    merged = _merge_season_tables(shooting, passing, pass_types)
    roles = derive_setpiece_roles(merged)

    name_map = _build_name_map()
    unmatched = 0
    for role in roles:
        player_id = _match_player(role["player"], name_map)
        if player_id is None:
            unmatched += 1
            continue
        role["player_id"] = player_id

    written = write_setpiece_roles(season, roles)
    logger.info(
        "Set-piece roles %s (evidence: %s): %d written, %d unmatched, "
        "%d penalty takers",
        season, evidence_season, written, unmatched,
        sum(1 for r in roles if r.get("player_id") and r["is_penalty_taker"]),
    )
    return written, unmatched


def _merge_season_tables(shooting, passing, pass_types) -> list[dict]:  # pragma: no cover
    """Flatten soccerdata's three season tables into the plain mappings
    ``derive_setpiece_roles`` consumes. Indexed by (league, season, team,
    player) in soccerdata, so the index carries the identity."""
    records: dict[tuple[str, str], dict] = {}

    def _absorb(df, mapping: dict[str, tuple[str, ...]]) -> None:
        if df is None or getattr(df, "empty", True):
            return
        frame = df.reset_index()
        for raw in frame.to_dict("records"):
            player = str(raw.get("player") or "").strip()
            team = str(raw.get("team") or "").strip()
            if not player or not team:
                continue
            record = records.setdefault((team, player), {"player": player, "team": team})
            for target, sources in mapping.items():
                value = _num(raw, *sources, default=float("nan"))
                if value == value:
                    record[target] = value

    _absorb(shooting, {
        "penalty_attempts": ("Standard PKatt", "PKatt"),
        "matches": ("Playing Time MP", "MP", "matches"),
    })
    _absorb(passing, {
        "key_passes": ("KP", "Pass Types KP"),
        "matches": ("Playing Time MP", "MP", "matches"),
    })
    _absorb(pass_types, {"corners": ("Pass Types CK", "CK")})
    return list(records.values())
