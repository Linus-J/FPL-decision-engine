"""
T0 acceptance gate — verify ScoringRules reconstructs official FPL points.

Self-contained (no live-DB dependency): each case is a synthetic player-GW
line whose expected `total_points` (excluding bonus) is computed by hand from
the official 25/26 scoring table. This locks the config against regressions
such as the clean-sheet bug (cs_gk/cs_def wrongly 6 instead of 4).

Bonus is deliberately excluded here — bonus is a match-relative ranking of the
BPS totals (see BPSWeights / bps_sim), not a per-player deterministic sum.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.strategy import BPS_WEIGHTS, DEFCON, SCORING


@dataclass(frozen=True)
class PlayerGWLine:
    """Minimal component line mirroring player_gw_stats columns."""

    position: str  # "GK" | "DEF" | "MID" | "FWD"
    minutes: int
    goals: int = 0
    assists: int = 0
    clean_sheet: bool = False
    goals_conceded: int = 0
    saves: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0


def base_points(line: PlayerGWLine, s=SCORING) -> int:
    """Reconstruct non-bonus FPL points from components using ScoringRules."""
    pts = 0

    # Minutes
    if line.minutes >= 60:
        pts += s.points_full_appearance
    elif line.minutes > 0:
        pts += s.points_sub_appearance

    # Goals
    goal_pts = {
        "GK": s.points_goal_gk,
        "DEF": s.points_goal_def,
        "MID": s.points_goal_mid,
        "FWD": s.points_goal_fwd,
    }[line.position]
    pts += line.goals * goal_pts

    # Assists
    pts += line.assists * s.points_assist

    # Clean sheets (only if 60+ mins per FPL rules)
    if line.clean_sheet and line.minutes >= 60:
        cs_pts = {
            "GK": s.points_cs_gk,
            "DEF": s.points_cs_def,
            "MID": s.points_cs_mid,
            "FWD": s.points_cs_fwd,
        }[line.position]
        pts += cs_pts

    # Goals conceded (GK/DEF only)
    if line.position in ("GK", "DEF"):
        conceded_penalties = line.goals_conceded // s.goals_conceded_per_penalty
        pts += conceded_penalties * s.points_goals_conceded_penalty

    # Saves (GK)
    pts += (line.saves // s.saves_per_bonus_point) * s.points_save_bonus

    # Penalties
    pts += line.penalties_saved * s.points_penalty_save
    pts += line.penalties_missed * s.points_penalty_miss

    # Discipline
    pts += line.yellow_cards * s.points_yellow_card
    pts += line.red_cards * s.points_red_card

    return pts


def test_clean_sheet_defender() -> None:
    """CS defender, 90 mins: 2 (appearance) + 4 (CS) = 6."""
    line = PlayerGWLine(position="DEF", minutes=90, clean_sheet=True, goals_conceded=0)
    assert base_points(line) == 6


def test_returning_midfielder() -> None:
    """Mid, 90 mins, 1 goal + 1 assist, team kept CS: 2 + 5 + 3 + 1 = 11."""
    line = PlayerGWLine(
        position="MID", minutes=90, goals=1, assists=1, clean_sheet=True
    )
    assert base_points(line) == 11


def test_hauling_forward() -> None:
    """FWD, 90 mins, 2 goals + 1 assist (no CS pts for FWD): 2 + 8 + 3 = 13."""
    line = PlayerGWLine(position="FWD", minutes=90, goals=2, assists=1, clean_sheet=True)
    assert base_points(line) == 13


def test_keeper_saves_and_concede() -> None:
    """GK, 90 mins, 5 saves, 2 conceded, no CS: 2 + 1 (3-save) - 1 (2-conceded) = 2."""
    line = PlayerGWLine(
        position="GK", minutes=90, saves=5, goals_conceded=2, clean_sheet=False
    )
    assert base_points(line) == 2


def test_sub_appearance_no_clean_sheet_bonus() -> None:
    """DEF playing <60 mins gets no CS points even with a team clean sheet."""
    line = PlayerGWLine(position="DEF", minutes=45, clean_sheet=True)
    assert base_points(line) == SCORING.points_sub_appearance


def test_penalty_save_and_card() -> None:
    """GK, 90 mins, pen saved + yellow: 2 + 5 - 1 = 6 (standard penalty-save = 5)."""
    line = PlayerGWLine(
        position="GK", minutes=90, penalties_saved=1, yellow_cards=1, clean_sheet=False
    )
    assert base_points(line) == 6


# --- Config-truth guards (Appendix A) -------------------------------------

def test_clean_sheet_values_fixed() -> None:
    """The 26/27 audit fix: GK/DEF clean sheet is 4, not the old buggy 6."""
    assert SCORING.points_cs_gk == 4
    assert SCORING.points_cs_def == 4


def test_defcon_rules() -> None:
    assert DEFCON.def_threshold == 10
    assert DEFCON.mid_fwd_threshold == 12
    assert DEFCON.points == 2
    assert DEFCON.cap_per_match == 2


def test_bps_2627_deltas() -> None:
    """Lock the 26/27 BPS changes so a future edit can't silently revert them."""
    # "being tackled" penalty removed
    assert BPS_WEIGHTS.being_tackled == 0
    # CBI now 1 point per 3 (was per 2)
    assert BPS_WEIGHTS.cbi_per_point == 3
    # save is flat +2 for any save, +1 extra inside box
    assert BPS_WEIGHTS.save == 2
    assert BPS_WEIGHTS.save_inside_box_extra == 1
    # big-chance-saved is new
    assert BPS_WEIGHTS.big_chance_saved == 1
    # penalty-save BPS dropped 8 -> 7
    assert BPS_WEIGHTS.penalty_saved == 7
