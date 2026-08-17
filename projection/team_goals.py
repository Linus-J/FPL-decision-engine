"""team_goals.py — expected team goals (λ_home, λ_away) from match odds (P3/P5).

The fixture anchor for the attacking + clean-sheet components: given a fixture's
de-vigged 1X2 (home/draw/away win) and Over-2.5 probabilities (from the T6 odds
tables), recover the two Poisson scoring rates under an independent double-Poisson
match model. Two market signals — supremacy (1X2) and total (O/U 2.5) — pin the
two rates. P5 then reads clean-sheet as P(opponent scores 0) = exp(-λ_opp); P3
distributes λ across a team's players by their npxG share.

Pure + deterministic (scipy). No DB or network.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

_MAX_GOALS = 10        # truncation for the score grid (P(>10) is negligible)
_LAMBDA_LO, _LAMBDA_HI = 0.05, 6.0
_OVER_WEIGHT = 1.0     # weight of the O/U 2.5 term vs the 1X2 terms


def _grid(lam: float) -> np.ndarray:
    return poisson.pmf(np.arange(_MAX_GOALS + 1), lam)


def outcome_probs(lam_home: float, lam_away: float) -> tuple[float, float, float, float]:
    """(P_home_win, P_draw, P_away_win, P_over_2.5) under independent Poissons."""
    ph, pa = _grid(lam_home), _grid(lam_away)
    joint = np.outer(ph, pa)                 # joint[i, j] = P(home=i, away=j)
    home = float(np.tril(joint, -1).sum())   # i > j
    draw = float(np.trace(joint))            # i == j
    away = float(np.triu(joint, 1).sum())    # i < j
    idx = np.add.outer(np.arange(_MAX_GOALS + 1), np.arange(_MAX_GOALS + 1))
    over = float(joint[idx >= 3].sum())      # total goals >= 3
    return home, draw, away, over


def team_goals_from_odds(
    home_win: float,
    draw: float,
    away_win: float,
    over25: float | None = None,
) -> tuple[float, float]:
    """Recover (λ_home, λ_away) that best reproduce the given de-vigged odds.

    Least-squares fit of the double-Poisson outcome probabilities to the four
    market probabilities. ``over25`` is optional — without it the total is pinned
    only by the 1X2 shape (less precise). Falls back to a neutral (1.35, 1.15)
    if the odds are degenerate/missing.
    """
    if not (home_win > 0 and away_win > 0) or (home_win + draw + away_win) <= 0:
        return 1.35, 1.15
    # renormalise the 1X2 in case of rounding
    tot = home_win + draw + away_win
    h, d, a = home_win / tot, draw / tot, away_win / tot

    def loss(params: np.ndarray) -> float:
        lam_h, lam_a = params
        mh, md, ma, mo = outcome_probs(lam_h, lam_a)
        err = (mh - h) ** 2 + (md - d) ** 2 + (ma - a) ** 2
        if over25 is not None and over25 > 0:
            err += _OVER_WEIGHT * (mo - over25) ** 2
        return err

    # seed from a rough supremacy/total guess for fast, stable convergence
    seed_total = 2.6
    seed_sup = (h - a) * 2.0
    x0 = np.clip([(seed_total + seed_sup) / 2, (seed_total - seed_sup) / 2],
                 _LAMBDA_LO, _LAMBDA_HI)
    res = minimize(
        loss, x0, method="L-BFGS-B",
        bounds=[(_LAMBDA_LO, _LAMBDA_HI), (_LAMBDA_LO, _LAMBDA_HI)],
    )
    lam_h, lam_a = res.x
    return float(round(lam_h, 4)), float(round(lam_a, 4))


def clean_sheet_prob(lam_opponent: float) -> float:
    """P(opponent scores 0) = Poisson(0; λ) — the odds-anchored clean-sheet
    probability for P5 (retires the capped 1X2 heuristic)."""
    return float(np.exp(-max(0.0, lam_opponent)))


def clean_sheet_probs_from_odds(
    home_win: float, draw: float, away_win: float, over25: float | None = None
) -> tuple[float, float]:
    """(P(home keeps a clean sheet), P(away keeps a clean sheet)) from 1X2 + O/U.

    THE canonical derivation. It lives here — not in an ingestor — because two
    separate call sites need it and they must agree:

    - ``data.ingestors.odds_api`` writes ``fixture_odds`` (the LIVE path), and
    - ``scripts.backfill_odds`` writes ``historical_fixture_odds`` (the TRAINING
      path).

    ``projection.features`` reads both into the same ``my_cs_prob``/
    ``opp_cs_prob`` columns and hands them to the minutes model, so any
    difference between the two is train/serve skew: the model would learn a
    coefficient on one scale and apply it on another.

    That is not hypothetical — it is exactly what happened. Both sites used a
    capped 1X2 heuristic that was inverted (``home_cs = draw + away_win * 0.3``
    is P(the *home* team fails to score)); fixing only the live one on
    2026-08-16 left the model training on inverted clean sheets and predicting
    on correct ones, which is worse than a consistent error because the sign
    flips between fit and inference. The old backfill helper documented itself
    as "mirrors the live odds_api heuristic so historical and live features are
    on the same scale" — a comment cannot enforce that, so a shared function
    does instead.

    A clean sheet belongs to the DEFENCE: ``home_cs`` is the AWAY side failing
    to score, hence ``exp(-λ_away)``.
    """
    lam_home, lam_away = team_goals_from_odds(home_win, draw, away_win, over25)
    return (
        round(clean_sheet_prob(lam_away), 3),
        round(clean_sheet_prob(lam_home), 3),
    )
