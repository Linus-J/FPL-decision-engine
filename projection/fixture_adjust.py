"""fixture_adjust.py — per-GW fixture-difficulty multiplier (Phase-2 P0).

The v1 backtest broadcast one xPts across the whole horizon (defect D3). This
gives each future GW a multiplier from that GW's actual opponent, so horizon
projections differ by fixture. It is a deliberately simple, bounded scaffold:
the real fixture-conditioning comes from the component models (P3/P5 anchor on
odds-implied team goals). Using the *future* opponent is not leakage — the
schedule is known before the deadline; only the model *inputs* must be as-of.
"""

from __future__ import annotations

import math

from projection.team_goals import NEUTRAL_LAMBDA_AWAY, NEUTRAL_LAMBDA_HOME

LEAGUE_AVG_STRENGTH = 1200.0   # TeamSeasonStrength defaults/centre (Phase-1 T3b)
_HOME_BOOST = 0.05             # modest home advantage on attacking output
_MIN_MULT, _MAX_MULT = 0.70, 1.40   # clamp so one fixture can't dominate
# The odds-anchored multiplier below works on expected GOALS, which vary far
# more than the strength ratio above -- the observed GW1 spread is 0.46 to
# 1.72 -- so it gets its own, wider clamp. Still bounded: a freak price
# should not be able to treble a projection on its own.
_MIN_ATTACK_MULT, _MAX_ATTACK_MULT = 0.35, 2.00


def fixture_multiplier(
    opp_defence_strength: float | None,
    was_home: bool | None,
    league_avg: float = LEAGUE_AVG_STRENGTH,
    sensitivity: float = 0.5,
) -> float:
    """Attacking-output multiplier for a fixture vs a given opponent defence.

    Weaker opponent defence (``opp_defence_strength`` below average) → >1;
    stronger → <1. ``was_home`` adds a small home boost. Missing opponent data
    → neutral home/away-only adjustment (never a hard 0). Clamped to
    [0.70, 1.40] so a single fixture can't swamp the projection.
    """
    if opp_defence_strength and opp_defence_strength > 0:
        # ratio > 1 when opponent is weaker than average → easier fixture
        ratio = league_avg / opp_defence_strength
        mult = ratio ** sensitivity
    else:
        mult = 1.0
    if was_home:
        mult *= 1.0 + _HOME_BOOST
    elif was_home is False:
        mult *= 1.0 - _HOME_BOOST
    return max(_MIN_MULT, min(_MAX_MULT, round(mult, 4)))


# Points a player banks purely for turning out (60+ minutes). This part of a
# per-appearance average does NOT vary with the opponent, so the fixture
# multiplier must not be applied to it -- scaling a player's whole average by
# 1.7 credits them with 3.4 appearance points.
APPEARANCE_POINTS = 2.0

# How a position's above-appearance points split between the attacking channel
# (own expected goals) and the clean-sheet channel (opponent's). A keeper's
# return is almost entirely about whether the opponent scores; a forward's is
# almost entirely about whether his own side does. Saves rise when the opponent
# threatens more, which partly offsets a keeper's clean-sheet loss -- hence 0.05
# rather than 0.00 on the attacking side for GKP.
_CHANNEL_WEIGHTS = {
    "GKP": (0.05, 0.95),
    "DEF": (0.40, 0.60),
    "MID": (0.85, 0.15),
    "FWD": (0.95, 0.05),
}


def fixture_points_multiplier(
    lam_for: float | None,
    lam_against: float | None,
    is_home: bool | None,
    per_appearance_points: float,
    position: str,
) -> float:
    """Multiplier on a player's per-APPEARANCE points for one fixture, from
    that fixture's two expected-goal figures.

    ``fixture_multiplier`` above is, in its own words, "a deliberately simple,
    bounded scaffold: the real fixture-conditioning comes from the component
    models (P3/P5 anchor on odds-implied team goals)". For the in-season engine
    that is true. The cold start never reaches those models, so the scaffold
    silently became the permanent answer for the season's single biggest
    decision -- and it is far too flat to be that. Measured on GW1 2026-27: the
    scaffold spans 0.89 to 1.10 across all twenty sides, the bookmakers span
    0.46 to 1.72.

    **Two channels, not one.** A fixture is good for a forward when HIS team is
    expected to score; it is good for a goalkeeper when the OPPONENT is not.
    Scaling everyone by their own team's expected goals boosts the defenders of
    teams that score freely and concede freely, which is precisely backwards --
    it briefly had the engine captaining a defender off an attacking multiplier.

    So the above-appearance points are split by position between an attacking
    channel (own expected goals) and a clean-sheet channel (P(opponent fails to
    score), the same ``exp(-lambda)`` the in-season engine uses):

    ==========  ========  ============
    position    attack    clean sheet
    ==========  ========  ============
    GKP           0.05        0.95
    DEF           0.40        0.60
    MID           0.85        0.15
    FWD           0.95        0.05
    ==========  ========  ============

    Only the ABOVE-APPEARANCE part is scaled at all. A striker averaging 6.8 is
    roughly 2 for showing up plus 4.8 earned; a strong fixture should treble the
    4.8, not the 6.8. Without that floor a hard fixture drives a player toward
    zero points for playing ninety minutes.
    """
    base_for = NEUTRAL_LAMBDA_HOME if is_home else NEUTRAL_LAMBDA_AWAY
    base_against = NEUTRAL_LAMBDA_AWAY if is_home else NEUTRAL_LAMBDA_HOME

    if lam_for is None or lam_for <= 0:
        attack_mult = 1.0
    else:
        attack_mult = lam_for / base_for
    attack_mult = max(_MIN_ATTACK_MULT, min(_MAX_ATTACK_MULT, attack_mult))

    if lam_against is None or lam_against <= 0:
        cs_mult = 1.0
    else:
        cs_mult = math.exp(-lam_against) / math.exp(-base_against)
    cs_mult = max(_MIN_ATTACK_MULT, min(_MAX_ATTACK_MULT, cs_mult))

    w_attack, w_cs = _CHANNEL_WEIGHTS.get(position, _CHANNEL_WEIGHTS["MID"])
    blended = w_attack * attack_mult + w_cs * cs_mult

    if per_appearance_points <= 0:
        return 1.0
    earned = max(0.0, per_appearance_points - APPEARANCE_POINTS)
    floor = per_appearance_points - earned
    return (floor + earned * blended) / per_appearance_points
