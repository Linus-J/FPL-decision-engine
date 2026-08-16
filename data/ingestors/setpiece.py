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


# Columns a role dict may set. A source only writes the ones it actually
# has an opinion about -- a published taker list says nothing about key
# passes, and writing a 0.0 there would clobber a real FBref-derived value.
_ROLE_COLUMNS = (
    "is_penalty_taker",
    "penalty_xg_per_game",
    "is_set_piece_taker",
    "key_passes_per_game",
    "penalty_order",
    "freekick_order",
    "corner_order",
    "source",
)


def write_setpiece_roles(season: str, roles: Iterable[Mapping]) -> int:
    """Upsert resolved roles (each needing a ``player_id``) for ``season``.

    Idempotent on (player_id, season) -- re-running mid-season updates the
    duty rather than accumulating rows, which matters because penalty
    responsibility genuinely changes during a season.

    The update is PARTIAL: only keys present in the role dict are written.
    Two sources feed this table with different knowledge (a published depth
    chart knows order but not key passes; FBref knows the reverse), and a
    blanket write would have each silently erase the other's contribution.
    """
    rows = [r for r in roles if r.get("player_id")]
    if not rows:
        return 0

    db = get_session()
    try:
        for role in rows:
            fields = {k: role[k] for k in _ROLE_COLUMNS if k in role}
            stmt = (
                insert(PlayerSetPieceRole)
                .values(
                    player_id=int(role["player_id"]),
                    season=season,
                    updated_at=datetime.utcnow(),
                    **fields,
                )
                .on_conflict_do_update(
                    index_elements=["player_id", "season"],
                    set_={**fields, "updated_at": datetime.utcnow()},
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


# ---------------------------------------------------------------------------
# Published depth charts (2026-08-16)
#
# The FBref route above infers duty from last season's attempt share. Two
# problems with that at the start of a season, both real: FBref has not
# populated the new season yet, and attempt share cannot survive a summer
# transfer window -- Isak, Semenyo and Gyokeres all changed clubs, and their
# old clubs' penalty duty went with them.
#
# A published pre-season taker list beats both. It also carries ORDER, which
# attempt share only approximates and a boolean cannot express at all: the
# first-choice penalty taker is worth several times the third-choice one.
# ---------------------------------------------------------------------------

# Team names as published, mapped onto the names FPL uses in `teams`.
_TEAM_ALIASES = {
    "man united": "Man Utd",
    "manchester united": "Man Utd",
    "manchester city": "Man City",
    "newcastle united": "Newcastle",
    "leeds united": "Leeds",
    "nottingham forest": "Nott'm Forest",
    "tottenham": "Spurs",
    "tottenham hotspur": "Spurs",
    "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
    "west ham united": "West Ham",
}

# A published list marks some names with an asterisk (a doubt: not yet
# registered, injured, or a rumoured departure). Recorded normally but
# reported, because silently trusting a flagged name is how a phantom
# penalty taker gets into projections.
_UNCERTAIN_MARKER = "*"

# League-average penalties awarded to a given team in a given match. ~90-100
# penalties across 380 PL matches is ~0.12 per team per game. Combined with
# PENALTY_CONVERSION this turns depth-chart position into expected penalty
# goal value -- a prior, to be superseded by observed attempts once the
# season provides them.
TEAM_PENALTIES_PER_GAME = 0.12

# Share of a team's penalties taken by the 1st, 2nd, 3rd... choice. The
# primary taker takes nearly all of them; the rest only appear when he is
# absent or has already been substituted.
_PENALTY_ORDER_SHARE = (0.85, 0.12, 0.03)


def penalty_xg_for_order(order: int | None) -> float:
    """Expected penalty GOAL value per game for a taker at this depth-chart
    position. ``None`` (not on penalties) is 0.0, which is what the
    consuming query already COALESCEs a missing row to."""
    if not order or order < 1:
        return 0.0
    share = (
        _PENALTY_ORDER_SHARE[order - 1]
        if order <= len(_PENALTY_ORDER_SHARE)
        else 0.0
    )
    return round(TEAM_PENALTIES_PER_GAME * share * PENALTY_CONVERSION, 5)


def parse_setpiece_table(raw: str) -> list[dict]:
    """Parse a ``Team | Penalties | Free Kicks | Corners`` table.

    Each cell is a comma-separated list in DEPTH-CHART ORDER. Returns one
    dict per (team, player) with 1-based ``penalty_order`` / ``freekick_order``
    / ``corner_order``, ``None`` where the player is not on that duty.

    Pure: no DB, no network. Name resolution happens in the caller, scoped to
    the team -- which matters, because a bare surname like "Wilson" is
    ambiguous league-wide and unique within a squad.
    """
    roles: dict[tuple[str, str], dict] = {}
    duty_columns = ("penalty_order", "freekick_order", "corner_order")

    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        team = cells[0]
        if not team or team.lower() == "team":
            continue  # header
        team = _TEAM_ALIASES.get(team.lower(), team)

        for column, duty in zip(cells[1:4], duty_columns, strict=False):
            for order, name in enumerate(
                (n.strip() for n in column.split(",")), start=1
            ):
                if not name:
                    continue
                uncertain = _UNCERTAIN_MARKER in name
                name = name.replace(_UNCERTAIN_MARKER, "").strip()
                if not name:
                    continue
                role = roles.setdefault(
                    (team, name),
                    {
                        "team": team, "player": name, "uncertain": False,
                        "penalty_order": None, "freekick_order": None,
                        "corner_order": None,
                    },
                )
                role[duty] = order
                role["uncertain"] = role["uncertain"] or uncertain

    return list(roles.values())


def roles_from_depth_chart(parsed: list[dict]) -> list[dict]:
    """Turn parsed depth-chart rows into ``player_setpiece_roles`` fields.

    ``key_passes_per_game`` is deliberately ABSENT rather than 0.0: a
    published taker list has no opinion on it, and writing a zero would
    clobber a real value from the FBref path. ``write_setpiece_roles`` only
    updates the keys it is given.
    """
    out = []
    for row in parsed:
        out.append({
            "player": row["player"],
            "team": row["team"],
            "uncertain": row["uncertain"],
            "penalty_order": row["penalty_order"],
            "freekick_order": row["freekick_order"],
            "corner_order": row["corner_order"],
            "is_penalty_taker": row["penalty_order"] == 1,
            "penalty_xg_per_game": penalty_xg_for_order(row["penalty_order"]),
            "is_set_piece_taker": (
                row["freekick_order"] is not None or row["corner_order"] is not None
            ),
        })
    return out


def ingest_depth_chart(season: str, raw: str, source: str) -> tuple[int, list[str]]:
    """Load a published taker list into ``player_setpiece_roles``.

    Returns ``(rows_written, unresolved_names)``. Names are matched WITHIN
    the stated team, so a surname that is ambiguous league-wide resolves
    cleanly -- and a name that matches nobody in that squad is reported
    rather than guessed at, since a wrong match would hand one player's
    penalty duty to another.
    """
    from data.ingestors.fbref import _match_player, _normalize_name

    roles = roles_from_depth_chart(parse_setpiece_table(raw))
    if not roles:
        return 0, []

    db = get_session()
    try:
        from data.models import Player, Team

        team_ids = {t.name.lower(): t.id for t in db.query(Team).all()}
        by_team: dict[int, dict[str, int]] = {}
        for p in db.query(Player).all():
            squad = by_team.setdefault(p.team_id, {})
            squad[_normalize_name(f"{p.first_name} {p.second_name}")] = p.id
            squad[_normalize_name(p.web_name)] = p.id
    finally:
        db.close()

    unresolved: list[str] = []
    uncertain: list[str] = []
    for role in roles:
        team_id = team_ids.get(role["team"].lower())
        if team_id is None:
            unresolved.append(f"{role['player']} (unknown team {role['team']!r})")
            continue
        player_id = _match_player(role["player"], by_team.get(team_id, {}))
        if player_id is None:
            unresolved.append(f"{role['player']} ({role['team']})")
            continue
        role["player_id"] = player_id
        role["source"] = source
        if role["uncertain"]:
            uncertain.append(f"{role['player']} ({role['team']})")

    written = write_setpiece_roles(season, roles)
    if uncertain:
        logger.warning(
            "%d taker(s) flagged uncertain in the source and recorded anyway "
            "— verify before the deadline: %s",
            len(uncertain), ", ".join(sorted(uncertain)),
        )
    if unresolved:
        logger.warning(
            "%d name(s) did not resolve to a player in that squad and were "
            "SKIPPED: %s", len(unresolved), ", ".join(sorted(unresolved)),
        )
    logger.info(
        "Depth chart %s: %d roles written (%d penalty takers)",
        season, written,
        sum(1 for r in roles if r.get("player_id") and r["is_penalty_taker"]),
    )
    return written, unresolved
