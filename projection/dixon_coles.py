"""dixon_coles.py — team strength fitted to results, for the fixtures bookmakers
have not priced (2026-08-18).

Bookmakers price about one round ahead. Everything past that runs on a fallback,
and on the live GW1 frame that fallback carried 78% of the decision — so it is
worth more than the power law it replaces, which regressed onto FPL's own
published strength ratings and reached R^2 ~ 0.58 against odds-implied lambda.

Two things are wrong with regressing onto those ratings. They are a coarse,
hand-maintained integer scale, four numbers per team, with no uncertainty and no
recency. And FPL publishes attack/defence as 0 until a season is underway, so at
GW1 — exactly when the horizon reaches furthest past the last priced fixture —
every team resolves through a prior-season fallback and the three promoted sides
have no rating at all.

This fits team parameters directly to match RESULTS instead, which the database
holds for five seasons, and which need no cooperation from FPL's own ratings:

    log lambda_home = intercept_home + attack[home] + concede[away]
    log lambda_away = intercept_away + attack[away] + concede[home]

plus Dixon and Coles' (1997) correction for the dependence between low scores,
which independent Poissons get wrong — 0-0, 1-0, 0-1 and 1-1 are respectively
more, less, less and more common than independence implies.

Recency is handled by exponentially decaying each match's weight, so a result
from three seasons ago informs the fit without dominating it.

Promoted teams are the case the old model simply had no answer for. A side with
no history in the window gets the average parameters of teams in THEIR first
season after promotion, measured over this same data, rather than being treated
as league-average — which is what a shrinkage-to-zero prior would wrongly imply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sqlalchemy import text

from data.db import get_session

logger = logging.getLogger(__name__)

# Weight halves roughly every 40 gameweeks -- a bit over one season, so the
# current campaign dominates while several years still inform the fit. Dixon and
# Coles fitted an equivalent decay on days; gameweeks are the unit this database
# actually stores.
DEFAULT_HALF_LIFE_GWS = 40.0

# Pulls attack/concede toward zero. Does identifiability duty (the intercepts
# and the team terms are otherwise free to shift against each other) and doubles
# as shrinkage, so a team with few matches lands nearer the league average
# instead of chasing a small sample.
RIDGE = 0.02

_LAMBDA_LO, _LAMBDA_HI = 0.05, 6.0


@dataclass
class DixonColesFit:
    attack: dict[int, float]
    concede: dict[int, float]
    intercept_home: float
    intercept_away: float
    rho: float
    promoted_attack: float = 0.0
    promoted_concede: float = 0.0
    n_matches: int = 0
    teams_fitted: list[int] = field(default_factory=list)

    def lambdas(self, home_code: int | None, away_code: int | None) -> tuple[float, float]:
        """(lambda_home, lambda_away) for a fixture, by cross-season team code.

        An unknown code — a promoted side with no history in the window — takes
        the promoted-team prior rather than league average.
        """
        ah = self.attack.get(home_code, self.promoted_attack)
        aa = self.attack.get(away_code, self.promoted_attack)
        ch = self.concede.get(home_code, self.promoted_concede)
        ca = self.concede.get(away_code, self.promoted_concede)
        lam_home = np.exp(self.intercept_home + ah + ca)
        lam_away = np.exp(self.intercept_away + aa + ch)
        return (
            float(np.clip(lam_home, _LAMBDA_LO, _LAMBDA_HI)),
            float(np.clip(lam_away, _LAMBDA_LO, _LAMBDA_HI)),
        )


def load_match_results() -> pd.DataFrame:
    """Historical scorelines, reconstructed from per-player gameweek stats.

    There is no results table: ``fixtures`` holds only the current season. But
    FPL records ``goals_conceded`` per player, counting goals let in while that
    player was on the pitch, so the MAXIMUM over a team's appearances in a match
    is the full-match total — which is the OPPONENT's score. Taking it from both
    sides recovers the scoreline exactly, including own goals, which summing
    each team's ``goals_scored`` would misattribute.

    Verified against five seasons: 1,891 matches at 1.59 home / 1.32 away goals
    and a 44.1/24.1/31.9 home/draw/away split, which are ordinary Premier League
    numbers, and league averages that agree with the odds-implied 1.60/1.30.

    Teams are keyed by cross-season ``code``, not by the per-season team id FPL
    reassigns each year.
    """
    db = get_session()
    try:
        sides = pd.read_sql(text("""
            SELECT s.season, s.gameweek, s.team_id_season AS team, s.opponent_team_id AS opp,
                   s.was_home, MAX(s.goals_conceded) AS conceded
            FROM player_gw_stats s
            WHERE s.minutes > 0
            GROUP BY s.season, s.gameweek, s.team_id_season, s.opponent_team_id, s.was_home
        """), db.bind)
        codes = pd.read_sql(
            text("SELECT season, team_id, code FROM team_season_strength"), db.bind
        )
    finally:
        db.close()

    if sides.empty:
        return pd.DataFrame()

    home = sides[sides["was_home"] == 1].rename(columns={"conceded": "away_goals"})
    away = sides[sides["was_home"] == 0].rename(columns={"conceded": "home_goals"})
    matches = home.merge(
        away[["season", "gameweek", "team", "opp", "home_goals"]],
        left_on=["season", "gameweek", "team", "opp"],
        right_on=["season", "gameweek", "opp", "team"],
        suffixes=("", "_r"),
    )
    matches = matches.rename(columns={"team": "home_team", "opp": "away_team"})
    matches = matches[
        ["season", "gameweek", "home_team", "away_team", "home_goals", "away_goals"]
    ]

    code_map = {(r.season, r.team_id): r.code for r in codes.itertuples()}
    matches["home_code"] = [
        code_map.get((s, t)) for s, t in zip(matches["season"], matches["home_team"], strict=True)
    ]
    matches["away_code"] = [
        code_map.get((s, t)) for s, t in zip(matches["season"], matches["away_team"], strict=True)
    ]
    before = len(matches)
    matches = matches.dropna(subset=["home_code", "away_code"])
    if len(matches) < before:
        logger.info(
            "Dixon-Coles: dropped %d matches with no cross-season team code",
            before - len(matches),
        )
    matches["home_code"] = matches["home_code"].astype(int)
    matches["away_code"] = matches["away_code"].astype(int)
    return matches.reset_index(drop=True)


def _match_age_gws(matches: pd.DataFrame) -> np.ndarray:
    """Gameweeks between each match and the most recent one in the frame."""
    seasons = sorted(matches["season"].unique())
    season_index = {s: i for i, s in enumerate(seasons)}
    absolute = (
        matches["season"].map(season_index) * 38 + matches["gameweek"]
    ).to_numpy(dtype=float)
    return absolute.max() - absolute


def _tau(home_goals, away_goals, lam_home, lam_away, rho):
    """Dixon-Coles' low-score dependence correction.

    Independent Poissons misprice exactly four scorelines; everything else is
    left alone. Clipped away from zero because the log of it enters the
    likelihood and the optimiser will happily walk rho somewhere invalid.
    """
    out = np.ones_like(lam_home)
    both_zero = (home_goals == 0) & (away_goals == 0)
    home_one = (home_goals == 1) & (away_goals == 0)
    away_one = (home_goals == 0) & (away_goals == 1)
    both_one = (home_goals == 1) & (away_goals == 1)
    out[both_zero] = 1.0 - lam_home[both_zero] * lam_away[both_zero] * rho
    out[home_one] = 1.0 + lam_away[home_one] * rho
    out[away_one] = 1.0 + lam_home[away_one] * rho
    out[both_one] = 1.0 - rho
    return np.clip(out, 1e-9, None)


def fit_dixon_coles(
    matches: pd.DataFrame,
    half_life_gws: float = DEFAULT_HALF_LIFE_GWS,
    ridge: float = RIDGE,
) -> DixonColesFit:
    """Weighted maximum likelihood over the supplied scorelines."""
    if matches.empty:
        raise ValueError("fit_dixon_coles: no matches supplied")

    codes = sorted(set(matches["home_code"]) | set(matches["away_code"]))
    index = {code: i for i, code in enumerate(codes)}
    n = len(codes)

    hi = matches["home_code"].map(index).to_numpy()
    ai = matches["away_code"].map(index).to_numpy()
    hg = matches["home_goals"].to_numpy(dtype=float)
    ag = matches["away_goals"].to_numpy(dtype=float)
    weights = 0.5 ** (_match_age_gws(matches) / half_life_gws)

    def negative_log_likelihood(params):
        attack = params[:n]
        concede = params[n:2 * n]
        intercept_home, intercept_away, rho = params[2 * n], params[2 * n + 1], params[2 * n + 2]
        lam_home = np.exp(intercept_home + attack[hi] + concede[ai])
        lam_away = np.exp(intercept_away + attack[ai] + concede[hi])
        lam_home = np.clip(lam_home, 1e-6, 20.0)
        lam_away = np.clip(lam_away, 1e-6, 20.0)
        log_lik = (
            np.log(_tau(hg, ag, lam_home, lam_away, rho))
            + hg * np.log(lam_home) - lam_home
            + ag * np.log(lam_away) - lam_away
        )
        penalty = ridge * (np.sum(attack ** 2) + np.sum(concede ** 2))
        return -float(np.sum(weights * log_lik)) + penalty

    start = np.concatenate([
        np.zeros(2 * n),
        [np.log(max(hg.mean(), 0.1)), np.log(max(ag.mean(), 0.1)), 0.0],
    ])
    bounds = [(-2.0, 2.0)] * (2 * n) + [(-2.0, 2.0), (-2.0, 2.0), (-0.4, 0.4)]
    result = minimize(negative_log_likelihood, start, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        logger.warning("Dixon-Coles fit did not converge cleanly: %s", result.message)

    params = result.x
    fit = DixonColesFit(
        attack={code: float(params[index[code]]) for code in codes},
        concede={code: float(params[n + index[code]]) for code in codes},
        intercept_home=float(params[2 * n]),
        intercept_away=float(params[2 * n + 1]),
        rho=float(params[2 * n + 2]),
        n_matches=len(matches),
        teams_fitted=list(codes),
    )
    pa, pc = _promoted_prior(matches, fit)
    fit.promoted_attack, fit.promoted_concede = pa, pc
    return fit


def _promoted_prior(matches: pd.DataFrame, fit: DixonColesFit) -> tuple[float, float]:
    """Average fitted parameters of teams in their FIRST season in the window.

    A newly promoted side is not an average Premier League team, and shrinking
    it to zero — which is what a plain ridge prior does on its own — says
    exactly that. Teams appearing for the first time partway through the window
    are the closest observable analogue, so their fitted attack and concede
    values become the prior for a side with no history at all.

    Falls back to a mild penalty if the window is too short to contain any
    newcomers, rather than silently returning league average.
    """
    first_season = {}
    for season, group in matches.groupby("season"):
        for code in set(group["home_code"]) | set(group["away_code"]):
            first_season.setdefault(code, season)
    seasons = sorted(matches["season"].unique())
    if len(seasons) < 2:
        return 0.0, 0.15
    newcomers = [c for c, s in first_season.items() if s != seasons[0]]
    if not newcomers:
        return 0.0, 0.15
    return (
        float(np.mean([fit.attack[c] for c in newcomers])),
        float(np.mean([fit.concede[c] for c in newcomers])),
    )


def team_codes(season: str) -> dict[int, int]:
    """This season's team id -> cross-season code."""
    db = get_session()
    try:
        rows = pd.read_sql(
            text("SELECT team_id, code FROM team_season_strength WHERE season = :s"),
            db.bind, params={"s": season},
        )
    finally:
        db.close()
    return {int(r.team_id): int(r.code) for r in rows.itertuples()}


def has_published_strength(season: str) -> bool:
    """Whether FPL has published real attack/defence ratings for this season.

    They are all zero until a season is underway, which is precisely when the
    planning horizon reaches furthest past the last priced fixture. Deciding
    which fallback to use turns on this: measured walk-forward against
    odds-implied lambda, the published-strength power law is the better model
    when it has THIS season's ratings (MAE 0.249 vs 0.262) and clearly the
    worse one when it is running on last season's, which is what it actually
    does at GW1 (0.319 vs 0.262 — Dixon-Coles is 18% better there).
    """
    db = get_session()
    try:
        row = db.execute(text("""
            SELECT COUNT(*) FROM team_season_strength
            WHERE season = :s AND strength_attack_home > 0 AND strength_defence_home > 0
        """), {"s": season}).scalar()
    finally:
        db.close()
    return bool(row and row > 0)


def fit_from_database(half_life_gws: float = DEFAULT_HALF_LIFE_GWS) -> DixonColesFit | None:
    """Convenience wrapper. ``None`` (not an exception) when there is no usable
    history, so callers can fall back to the published-strength model."""
    matches = load_match_results()
    if matches.empty:
        logger.warning("Dixon-Coles: no reconstructable match results; falling back")
        return None
    return fit_dixon_coles(matches, half_life_gws=half_life_gws)
