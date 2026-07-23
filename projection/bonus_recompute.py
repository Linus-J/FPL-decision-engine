"""bonus_recompute.py — re-derive historical bonus under the 26/27 BPS rules.

Thin, network-free orchestration over ``projection.bps_sim``: read
``player_match_events`` (populated by any event provider — the FBref adapter is
one), group by match, run the simulator, and persist ``recomputed_bonus``
(plan §3.4 / T5b). Also provides the old-rules *sanity harness*: recompute
under the rules that were actually in force (``BPS_WEIGHTS_2526``) and measure
agreement with FPL's awarded bonus before trusting the 26/27 numbers.

Keep this module import-light (no ``soccerdata``/network): it operates on the
DB and plain mappings so it is fully unit-testable on synthetic events.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from config.strategy import BPS_WEIGHTS, BPS_WEIGHTS_2526, BPSWeights
from data.models import PlayerMatchEvents, RecomputedBonus
from projection.bps_sim import award_bonus, compute_player_bps

logger = logging.getLogger(__name__)

# Columns bps_sim reads. Named identically on PlayerMatchEvents, so an ORM row
# maps to a sim event mapping by attribute name (position is carried too).
EVENT_FIELDS: tuple[str, ...] = (
    "position", "minutes", "goals", "winning_goals", "assists", "clean_sheet",
    "big_chances_created", "key_passes", "open_play_crosses", "dribbles",
    "saves", "saves_in_box", "big_chances_saved", "penalties_saved",
    "tackles", "clearances", "blocks", "interceptions", "recoveries",
    "passes", "pass_completion_pct",
    "being_tackled", "penalties_conceded", "penalties_missed", "yellow_cards",
    "red_cards", "own_goals", "big_chances_missed", "errors_leading_to_goal",
    "errors_leading_to_shot", "fouls", "offsides", "shots_off_target",
)


def event_to_mapping(row: PlayerMatchEvents) -> dict:
    """Project an ORM event row onto the mapping the simulator consumes."""
    return {f: getattr(row, f) for f in EVENT_FIELDS}


def recompute_fixture(
    events_by_player: Mapping[int, Mapping],
    weights: BPSWeights = BPS_WEIGHTS,
) -> dict[int, tuple[int, int]]:
    """One match → ``{player_id: (bps, bonus)}`` under ``weights``.

    Bonus is the 3/2/1 award over this match's BPS totals (FPL tie rules); it
    is only meaningful per fixture, which is why the caller groups by match
    before calling this.
    """
    bps = {pid: compute_player_bps(ev, weights) for pid, ev in events_by_player.items()}
    bonus = award_bonus(bps)
    return {pid: (bps[pid], bonus[pid]) for pid in events_by_player}


def recompute_season(
    session: Session,
    season: str,
    weights: BPSWeights = BPS_WEIGHTS,
) -> tuple[int, int]:
    """Recompute bonus for every match with events in ``season``; upsert
    ``recomputed_bonus``. Returns ``(matches, rows_written)``.

    Idempotent: re-running overwrites the recompute for a match (weights may
    have changed) via ``on_conflict_do_update``.
    """
    rows = (
        session.query(PlayerMatchEvents)
        .filter(PlayerMatchEvents.season == season)
        .all()
    )
    by_game: dict[str, dict[int, Mapping]] = defaultdict(dict)
    gw_of: dict[str, int | None] = {}
    for r in rows:
        by_game[r.game_id][r.player_id] = event_to_mapping(r)
        gw_of[r.game_id] = r.gameweek

    written = 0
    for game_id, events_by_player in by_game.items():
        result = recompute_fixture(events_by_player, weights)
        for player_id, (bps, bonus) in result.items():
            stmt = (
                insert(RecomputedBonus)
                .values(
                    player_id=player_id,
                    season=season,
                    gameweek=gw_of[game_id],
                    game_id=game_id,
                    bps_2627=bps,
                    bonus_2627=bonus,
                )
                .on_conflict_do_update(
                    index_elements=["player_id", "season", "game_id"],
                    set_={"bps_2627": bps, "bonus_2627": bonus},
                )
            )
            session.execute(stmt)
            written += 1
    session.commit()
    logger.info(
        "Recomputed bonus for %s: %d matches, %d player-rows written",
        season, len(by_game), written,
    )
    return len(by_game), written


def recomputed_bonus_coverage(session: Session, season: str) -> float:
    """Fraction of matches that have events which also have a recomputed-bonus
    row. The plan's ≥95% gate is against *finished fixtures with event data*;
    this is the mechanical (recompute-side) coverage — the ingest-side share of
    finished fixtures that actually carry events is measured on the live run.
    """
    with_events = {
        gid for (gid,) in session.query(PlayerMatchEvents.game_id)
        .filter(PlayerMatchEvents.season == season).distinct()
    }
    if not with_events:
        return 0.0
    recomputed = {
        gid for (gid,) in session.query(RecomputedBonus.game_id)
        .filter(RecomputedBonus.season == season).distinct()
    }
    return len(with_events & recomputed) / len(with_events)


def oldrules_reproduction(
    events_by_game: Mapping[str, Mapping[int, Mapping]],
    actual_bonus_by_game: Mapping[str, Mapping[int, int]],
    weights: BPSWeights = BPS_WEIGHTS_2526,
) -> dict[str, float]:
    """Sanity harness: recompute bonus under the *old* rules and compare to
    FPL's awarded bonus, per match.

    The recompute cannot be exact — FPL's BPS uses Opta metrics an event feed
    may lack (big chances, error-leading-to-shot, exact recoveries) — so this
    reports agreement, not equality:

    - ``slot_exact_rate``: share of player-slots where recomputed bonus == actual
    - ``recipient_jaccard``: mean Jaccard of {players with bonus>0} per match
    - ``n_matches``

    A high score validates that the *plumbing* (events → sim → award) is sound
    before switching to the 26/27 weights; residual error is the metric gap.
    """
    n_matches = 0
    slot_hits = slot_total = 0
    jaccard_sum = 0.0
    for game_id, events_by_player in events_by_game.items():
        actual = actual_bonus_by_game.get(game_id, {})
        recomputed = recompute_fixture(events_by_player, weights)
        n_matches += 1
        for pid in events_by_player:
            slot_total += 1
            if recomputed[pid][1] == actual.get(pid, 0):
                slot_hits += 1
        rec_set = {pid for pid, (_, b) in recomputed.items() if b > 0}
        act_set = {pid for pid, b in actual.items() if b > 0}
        union = rec_set | act_set
        jaccard_sum += (len(rec_set & act_set) / len(union)) if union else 1.0
    return {
        "n_matches": float(n_matches),
        "slot_exact_rate": slot_hits / slot_total if slot_total else 0.0,
        "recipient_jaccard": jaccard_sum / n_matches if n_matches else 0.0,
    }


def load_events_grouped(
    session: Session, season: str
) -> tuple[dict[str, dict[int, dict]], dict[str, int | None]]:
    """DB helper for the sanity harness: ``player_match_events`` for a season as
    ``{game_id: {player_id: event_mapping}}`` plus each match's gameweek."""
    rows = (
        session.query(PlayerMatchEvents)
        .filter(PlayerMatchEvents.season == season)
        .all()
    )
    by_game: dict[str, dict[int, dict]] = defaultdict(dict)
    gw_of: dict[str, int | None] = {}
    for r in rows:
        by_game[r.game_id][r.player_id] = event_to_mapping(r)
        gw_of[r.game_id] = r.gameweek
    return dict(by_game), gw_of


def synthesize_game_id(season: str, gameweek: int, home: str, away: str) -> str:
    """Deterministic match id when a provider gives no stable one (used by the
    FBref adapter and by DGW-aware callers). Order-stable: home before away."""
    return f"{season}_gw{int(gameweek):02d}_{home}_{away}"


def game_ids_for_players(
    events: Sequence[PlayerMatchEvents],
) -> dict[int, list[str]]:
    """Map player → the matches they have events in (DGW players get 2+)."""
    out: dict[int, list[str]] = defaultdict(list)
    for e in events:
        out[e.player_id].append(e.game_id)
    return dict(out)
