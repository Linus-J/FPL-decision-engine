"""fixture_adjust.py — per-GW fixture-difficulty multiplier (Phase-2 P0).

The v1 backtest broadcast one xPts across the whole horizon (defect D3). This
gives each future GW a multiplier from that GW's actual opponent, so horizon
projections differ by fixture. It is a deliberately simple, bounded scaffold:
the real fixture-conditioning comes from the component models (P3/P5 anchor on
odds-implied team goals). Using the *future* opponent is not leakage — the
schedule is known before the deadline; only the model *inputs* must be as-of.
"""

from __future__ import annotations

LEAGUE_AVG_STRENGTH = 1200.0   # TeamSeasonStrength defaults/centre (Phase-1 T3b)
_HOME_BOOST = 0.05             # modest home advantage on attacking output
_MIN_MULT, _MAX_MULT = 0.70, 1.40   # clamp so one fixture can't dominate


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
