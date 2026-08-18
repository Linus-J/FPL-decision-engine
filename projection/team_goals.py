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

# --- Strength-based fallback (2026-08-18, engine review §2) ------------------
#
# CALIBRATED, not a guess. Fitted by least squares on log(lambda) over the 1,900
# fixtures in the five backfilled seasons where historical_fixture_odds and real
# published team_season_strength BOTH exist, against the odds-implied lambda
# that `team_goals_from_odds` returns for the same fixture:
#
#     lambda = base * (attack_rel ** attack_exp) * (defence_rel ** defence_exp)
#
# where *_rel is the team's strength divided by that season's league average, so
# the numbers are relative within their own season and the absolute scale FPL
# happens to use does not matter.
#
#                    base    attack_exp  defence_exp   R^2     MAE    flat MAE
#   lambda_home    1.5102      2.6642      -2.0883    0.583   0.278    0.454
#   lambda_away    1.2232      2.6055      -2.2115    0.572   0.225    0.371
#
# A 39% reduction in mean absolute error against the flat fallback on both
# sides. Defence exponents are negative because FPL's scale runs the other way:
# a HIGHER strength_defence means a BETTER defence.
#
# Note what the fitted bases also say: the odds-implied league averages are
# 1.60 and 1.30, not the 1.35 and 1.15 this module previously fell back to. The
# old constants under-projected goals for every fixture without odds, on top of
# giving them all the same ones.
_STRENGTH_BASE_HOME, _STRENGTH_ATT_HOME, _STRENGTH_DEF_HOME = 1.5102, 2.6642, -2.0883
_STRENGTH_BASE_AWAY, _STRENGTH_ATT_AWAY, _STRENGTH_DEF_AWAY = 1.2232, 2.6055, -2.2115

# The league-average fixture, used when even strengths are unavailable. These
# are the fitted bases above, i.e. what the model returns when both teams are
# exactly average — so the two fallbacks agree at the centre instead of
# disagreeing by 0.25 goals a game.
NEUTRAL_LAMBDA_HOME = _STRENGTH_BASE_HOME
NEUTRAL_LAMBDA_AWAY = _STRENGTH_BASE_AWAY


def team_goals_from_strength(
    home_attack_rel: float | None,
    home_defence_rel: float | None,
    away_attack_rel: float | None,
    away_defence_rel: float | None,
) -> tuple[float, float]:
    """(λ_home, λ_away) from team strengths RELATIVE to the league average.

    The fallback for fixtures the bookmakers have not priced yet. Odds remain
    strictly better and are always preferred where they exist — this exists
    because they cover only the next week or two, while the transfer planner
    looks three gameweeks ahead and the wildcard evaluation five.

    Before this, every unpriced fixture got the same flat pair, so across the
    whole planning horizon no fixture was distinguishable from any other: a
    defender's clean-sheet probability did not depend on who they played.

    Any missing input falls back to the neutral (league-average) fixture for
    that side rather than guessing, so a partially-known matchup degrades one
    term at a time instead of all at once.
    """
    def _rel(v: float | None) -> float:
        return v if v is not None and v > 0 else 1.0

    lam_home = (
        _STRENGTH_BASE_HOME
        * _rel(home_attack_rel) ** _STRENGTH_ATT_HOME
        * _rel(away_defence_rel) ** _STRENGTH_DEF_HOME
    )
    lam_away = (
        _STRENGTH_BASE_AWAY
        * _rel(away_attack_rel) ** _STRENGTH_ATT_AWAY
        * _rel(home_defence_rel) ** _STRENGTH_DEF_AWAY
    )
    return (
        float(np.clip(lam_home, _LAMBDA_LO, _LAMBDA_HI)),
        float(np.clip(lam_away, _LAMBDA_LO, _LAMBDA_HI)),
    )


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
    only by the 1X2 shape (less precise). Falls back to the league-average
    fixture if the odds are degenerate/missing.

    That neutral pair was ``(1.35, 1.15)`` until 2026-08-18. Fitting the
    strength model against 1,900 real priced fixtures put the odds-implied
    league means at ``(1.51, 1.22)`` — so the old constants were about a
    quarter of a goal per game low, biasing every unpriced fixture downwards
    on top of making them all identical.
    """
    if not (home_win > 0 and away_win > 0) or (home_win + draw + away_win) <= 0:
        return NEUTRAL_LAMBDA_HOME, NEUTRAL_LAMBDA_AWAY
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
