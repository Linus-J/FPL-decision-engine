"""bps_sim.py — deterministic 26/27 Bonus Points System simulator.

Not a regressor: given a player's match events, sum the per-action BPS
contributions from ``BPS_WEIGHTS`` (config.strategy, the 26/27 source of
truth), rank the players in a fixture, and award 3/2/1 bonus with FPL's
tie rules. DefCon is scored separately (compute_defcon_points) — it awards
real points, not BPS, and shares no term with the BPS calc (plan §4.7 / T5).

Event rows are plain mappings (dict or DataFrame row) so this works over
FBref-derived events once those land. Missing keys default to 0.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from config.strategy import BPS_WEIGHTS, DEFCON, BPSWeights, DefConRules

# Positions that can earn a clean-sheet BPS bonus.
_GK_DEF = {"GK", "GKP", "DEF"}
_BONUS_BY_POSITION = {1: 3, 2: 2, 3: 1}


def _val(ev: Mapping, key: str, default: float = 0) -> float:
    v = ev.get(key, default)
    return default if v is None else v


def compute_player_bps(ev: Mapping, w: BPSWeights = BPS_WEIGHTS) -> int:
    """Total 26/27 BPS for one player in one fixture from their event row.

    The BPS ``cbi`` metric is clearances+blocks+interceptions (tackles are
    scored separately as successful_tackle); DefCon adds tackles/recoveries
    on top — see compute_defcon_points. No term is shared between the two.
    """
    pos = ev.get("position", "MID")
    minutes = _val(ev, "minutes")
    bps = 0.0

    # --- Appearance ---
    if minutes >= 60:
        bps += w.play_over_60
    elif minutes > 0:
        bps += w.play_1_to_60

    # --- Goals (by position) + winning goal ---
    goal_weight = {
        "GK": w.goal_gk, "GKP": w.goal_gk, "DEF": w.goal_def,
        "MID": w.goal_mid, "FWD": w.goal_fwd,
    }.get(pos, w.goal_mid)
    bps += _val(ev, "goals") * goal_weight
    bps += _val(ev, "winning_goals") * w.winning_goal

    # --- Attacking contribution ---
    bps += _val(ev, "assists") * w.assist
    bps += _val(ev, "big_chances_created") * w.big_chance_created
    bps += _val(ev, "key_passes") * w.key_pass
    bps += _val(ev, "open_play_crosses") * w.open_play_cross
    bps += _val(ev, "dribbles") * w.successful_dribble

    # --- Clean sheet (GK/DEF, 60+ mins) ---
    if pos in _GK_DEF and minutes >= 60 and _val(ev, "clean_sheet"):
        bps += w.clean_sheet_gk_def

    # --- Goalkeeping ---
    bps += _val(ev, "saves") * w.save
    bps += _val(ev, "saves_in_box") * w.save_inside_box_extra
    bps += _val(ev, "big_chances_saved") * w.big_chance_saved
    bps += _val(ev, "penalties_saved") * w.penalty_saved

    # --- Defensive actions ---
    bps += _val(ev, "tackles") * w.successful_tackle
    cbi = _val(ev, "clearances") + _val(ev, "blocks") + _val(ev, "interceptions")
    bps += int(cbi) // w.cbi_per_point
    bps += int(_val(ev, "recoveries")) // w.recoveries_per_point

    # --- Passing accuracy (needs enough passes) ---
    if _val(ev, "passes") >= w.pass_completion_min_passes:
        pct = _val(ev, "pass_completion_pct")
        if pct >= 90:
            bps += w.pass_completion_90_plus
        elif pct >= 80:
            bps += w.pass_completion_80_89
        elif pct >= 70:
            bps += w.pass_completion_70_79

    # --- Negative ---
    bps += _val(ev, "being_tackled") * w.being_tackled
    bps += _val(ev, "penalties_conceded") * w.conceding_penalty
    bps += _val(ev, "penalties_missed") * w.missing_penalty
    bps += _val(ev, "yellow_cards") * w.yellow_card
    bps += _val(ev, "red_cards") * w.red_card
    bps += _val(ev, "own_goals") * w.own_goal
    bps += _val(ev, "big_chances_missed") * w.missing_big_chance
    bps += _val(ev, "errors_leading_to_goal") * w.error_leading_to_goal
    bps += _val(ev, "errors_leading_to_shot") * w.error_leading_to_shot
    bps += _val(ev, "fouls") * w.conceding_foul
    bps += _val(ev, "offsides") * w.caught_offside
    bps += _val(ev, "shots_off_target") * w.shot_off_target

    return int(bps)


def award_bonus(bps_by_player: Mapping[int, int]) -> dict[int, int]:
    """3/2/1 bonus for the top-3 BPS in a fixture, with FPL tie rules.

    Ties occupy consecutive rank positions and each tied player gets the bonus
    of the group's starting position: 2 tied for 1st → both 3, next gets 1;
    2 tied for 2nd → top 3, both 2; 3 tied for 1st → all 3.
    """
    if not bps_by_player:
        return {}
    groups: dict[int, list[int]] = defaultdict(list)
    for pid, bps in bps_by_player.items():
        groups[bps].append(pid)

    result = {pid: 0 for pid in bps_by_player}
    position = 1
    for value in sorted(groups, reverse=True):
        members = groups[value]
        bonus = _BONUS_BY_POSITION.get(position, 0)
        for pid in members:
            result[pid] = bonus
        position += len(members)
        if position > 3:
            break
    return result


def compute_fixture_bonus(
    events_by_player: Mapping[int, Mapping],
    w: BPSWeights = BPS_WEIGHTS,
) -> dict[int, int]:
    """Convenience: event rows per player → awarded bonus per player."""
    bps = {pid: compute_player_bps(ev, w) for pid, ev in events_by_player.items()}
    return award_bonus(bps)


def compute_defcon_points(ev: Mapping, rules: DefConRules = DEFCON) -> int:
    """Defensive Contribution points (real points, not BPS). DEF need CBIT
    (clearances+blocks+interceptions+tackles) ≥ threshold; MID/FWD need CBIRT
    (adds recoveries) ≥ a higher threshold. GK do not earn DefCon."""
    pos = ev.get("position", "MID")
    cbit = (
        _val(ev, "clearances") + _val(ev, "blocks")
        + _val(ev, "interceptions") + _val(ev, "tackles")
    )
    if pos == "DEF":
        return rules.points if cbit >= rules.def_threshold else 0
    if pos in ("MID", "FWD"):
        cbirt = cbit + _val(ev, "recoveries")
        return rules.points if cbirt >= rules.mid_fwd_threshold else 0
    return 0
