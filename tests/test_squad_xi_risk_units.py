"""optimise_starting_xi fed `mu` a VARIANCE instead of a semi-deviation.

Found live on 2026-08-25 running the real GW2 decision. `mu` is calibrated
against quantities in POINTS; `xpts_var` is in points-SQUARED. The XI
optimiser built its per-gameweek frame with only `xpts`/`xpts_var`, dropping
the `upside`/`downside` columns, so `risk_adjusted_score` received
upside=downside=None and fell back to variance.

Because variance scales with the mean, a negative `mu` then penalised the best
players hardest and inverted the XI. On the live GW2 frame at mu=-0.25:
B.Fernandes went from 7.71 xPts to an effective -1.48 (13th of 15) and Gabriel
from 7.07 to -5.13 (last), while Gibbs-White's 4.35 came top on 1.42. The
armband followed onto the goalkeeper.

`captaincy.py` already carries a comment describing this same units bug being
fixed there; `optimise_squad` has passed the columns since 2026-08-18. This
function was the one that was missed.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from config.strategy import OPTIMISER
from optimiser.squad import optimise_starting_xi

_RISK_AVERSE = dataclasses.replace(OPTIMISER, mu_baseline=-0.25, risk_level=0.0)


def _squad() -> pd.DataFrame:
    """Two GKP / five DEF / five MID / three FWD, the legal 15."""
    rows = []
    plan = [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]
    pid = 1
    for pos, count in plan:
        for _ in range(count):
            rows.append({"id": pid, "position": pos, "now_cost": 5.0})
            pid += 1
    return pd.DataFrame(rows)


def _projections(star_id: int) -> pd.DataFrame:
    """One clear best player, whose variance is high BECAUSE his mean is."""
    rows = []
    for pid in range(1, 16):
        star = pid == star_id
        xpts = 8.0 if star else 3.0
        rows.append({
            "player_id": pid, "gameweek": 2, "xpts": xpts,
            # Variance scales with the mean -- the property that makes the
            # units bug invert the ranking rather than merely perturb it.
            "xpts_var": xpts * 5.0,
            # Semi-deviation is on the same scale as points, and stays a
            # modest fraction of the mean.
            "upside": xpts * 0.45, "downside": xpts * 0.45,
        })
    return pd.DataFrame(rows)


def test_risk_averse_xi_still_captains_the_best_player():
    """The regression. At mu=-0.25 the best player must keep the armband: his
    semi-deviation is proportionally the same as everyone else's, so nothing
    should demote him."""
    star = 11  # a MID, so he is always XI-eligible
    solution = optimise_starting_xi(
        _squad(), _projections(star), 2, season="2026-27", config=_RISK_AVERSE,
    )

    assert solution.captain_id == star


def test_risk_averse_xi_starts_the_best_player():
    star = 11
    solution = optimise_starting_xi(
        _squad(), _projections(star), 2, season="2026-27", config=_RISK_AVERSE,
    )

    assert star in set(solution.starting_xi["id"])


def test_absent_semideviation_columns_make_mu_inert_not_variance_driven():
    """Projections carrying no semi-deviations (the in-season assemble path)
    must leave `mu` with nothing to bite on, rather than silently switching it
    to a quantity on a different scale. Same choice optimise_squad documents.
    """
    star = 11
    proj = _projections(star).drop(columns=["upside", "downside"])

    solution = optimise_starting_xi(
        _squad(), proj, 2, season="2026-27", config=_RISK_AVERSE,
    )

    assert solution.captain_id == star, "variance fallback would demote the best player"
