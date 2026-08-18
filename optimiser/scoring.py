"""scoring.py — P3-3 objective v1: E[pts] + λ·differential_value + μ·variance
(v2-build-plan §5).

The plan's literal differential_value formula (``your_pts - EO*field_pts``,
summed over players) is a NO-OP for the mean objective: ``EO_i * xpts_i`` is a
constant with respect to your own selection (both EO and xpts are exogenous
inputs), so maximising ``Σ(your_i - EO_i)*xpts_i`` picks EXACTLY the same team
as maximising ``Σ your_i*xpts_i`` alone — confirmed algebraically before
building this, not assumed. The plan's own §5 wording clarifies the real
intent regardless: "benching a 60%-EO player is a short position" — EO is
about RANK-OUTCOME VARIANCE (how much you swing relative to the field), not
the mean pick.

**v1 (this module, ships now):** a linear ILP-compatible approximation.
``differential_multiplier`` reweights each player's score by ownership —
this DOES change relative rankings between similar-xpts players (unlike the
literal formula above), pushing toward template safety or contrarian
differentials depending on ``risk_level``. ``variance`` is linear own-variance
only (``μ * xpts_var``) — TEAMMATE COVARIANCE is quadratic in a 0/1 selection
vector (``w'Σw``) and cannot be expressed in the current linear MILP
(`pulp`/CBC) without linearisation tricks or a QP solver. The plan itself
frames true covariance-aware risk as "Objective v2 (upgrade): scenario-based
stochastic programming" — not a v1 requirement. P3-1 already persists the
raw MC scenario samples a v2 implementation would need.

Pure + deterministic. No DB/network.

**Risk-aware cold start (plan/risk-aware-cold-start-v1.md, 2026-07-31):**
``risk_mode`` (a 3-way ``safe``/``balanced``/``aggressive`` switch) is
replaced by a continuous ``risk_level`` float in [-1.0, 1.0]. The old switch
made "balanced" mean *exactly zero* variance-awareness by construction —
not a genuine medium setting, just the dead centre of a sign flip. ``lambda``
(differential weight) keeps that sign-based shape (chasing differentials is
a taste axis, not a risk axis); ``mu`` (variance weight) now has a non-zero
baseline that ``risk_level`` shifts up or down, so ``risk_level=0`` carries
real, moderate variance-awareness instead of none.
"""

from __future__ import annotations

import pandas as pd

from config.strategy import OPTIMISER, OptimiserConfig


def lambda_mu_for_risk_level(
    risk_level: float,
    lambda_magnitude: float,
    mu_baseline: float,
    mu_range: float,
) -> tuple[float, float]:
    """(λ, μ) from a continuous risk posture in [-1.0, 1.0].
    ``lambda = risk_level * lambda_magnitude`` (zero at risk_level=0, same
    sign-flip shape as before) — a pure "taste" axis for template safety
    (-1) vs contrarian differentials (+1).
    ``mu = mu_baseline + risk_level * mu_range`` — ``mu_baseline`` (untuned,
    > 0) is the genuine "medium" variance-awareness; ``risk_level=-1`` can
    go net risk-averse (``mu_baseline - mu_range``, possibly negative —
    actively preferring low-variance picks over equal-mean volatile ones);
    ``+1`` leans further into upside variance."""
    lam = risk_level * lambda_magnitude
    mu = mu_baseline + risk_level * mu_range
    return lam, mu


def differential_multiplier(eo_pct: float, lam: float) -> float:
    """1 + λ*(1 - eo/100). λ>0 upweights low-owned (differential) players up
    to (1+λ)x at eo=0%; λ<0 downweights them, keeping high-owned (template)
    players at full value (multiplier=1 at eo=100% regardless of λ's sign).
    λ=0 -> always 1 (no EO effect, today's existing behaviour). Clamped at 0
    as a defensive floor (only reachable if λ is misconfigured below -1)."""
    return max(0.0, 1.0 + lam * (1.0 - eo_pct / 100.0))


def risk_adjusted_score(
    xpts: float,
    xpts_var: float,
    eo_pct: float,
    lam: float,
    mu: float,
    upside: float | None = None,
    downside: float | None = None,
) -> float:
    """Per-player-per-GW ILP objective coefficient, replacing raw xpts.
    Linear in the selection variable, so it drops into the existing
    squad.py/transfers.py MILP formulations with no solver change.

    ``upside`` (2026-08-18): the player's upper semi-deviation --
    ``sqrt(E[max(0, x - mean)^2])`` over his own per-appearance scores -- in
    POINTS. Used in place of ``xpts_var`` for the risk term whenever it is
    available, because variance is the wrong quantity for a risk appetite and
    measurably so.

    Measured on the real 2025-26 season. Haaland returns 13+ points in 22.9% of
    his appearances against Gabriel's 6.2%, and their upper semi-deviations are
    3.63 and 2.83. But the modelled ``xpts_var`` runs the other way -- 36.8 for
    Haaland, 50.5 for Gabriel -- because unconditional variance is dominated by
    the level of the mean and by availability, not by how big the good weeks
    are. A positive ``mu`` on ``xpts_var`` therefore rewarded the STEADIER
    player, which is the opposite of what a risk-seeking persona is asking for.

    Note the units. ``mu * upside`` is points x points; ``mu * xpts_var`` was
    points x points-squared. The scale of ``mu`` differs accordingly, which is
    why ``OptimiserConfig.mu_range`` moved with this change.
    """
    # Which SIDE of the distribution the appetite is about (2026-08-18).
    # Chasing risk means wanting big good weeks; avoiding it means wanting few
    # bad ones, and those are different players. Penalising upside instead --
    # which is what a single risk term does -- makes a risk-averse persona
    # select against GOOD players rather than against blank-prone ones, so its
    # squads were contrasting but useless.
    chosen = upside if mu >= 0 else downside
    # NaN is "not measured", not "zero" -- and left alone it propagates through
    # the whole ILP objective and silently produces a meaningless solution.
    if chosen is None or chosen != chosen:
        risk_term = xpts_var
    else:
        risk_term = chosen
    return xpts * differential_multiplier(eo_pct, lam) + mu * risk_term


def add_effective_score(
    projections: pd.DataFrame,
    ownership: pd.DataFrame | None = None,
    config: OptimiserConfig = OPTIMISER,
    risk_level: float | None = None,
) -> pd.DataFrame:
    """Adds an ``effective_score`` column to a copy of ``projections``
    (needs ``player_id``, ``xpts``; uses ``xpts_var`` if present, else 0 —
    the live path always has it via P10's assemble.py, the old monolithic
    path never did). ``ownership``: optional ``(player_id,
    top10k_selected_pct)`` frame (P3-2) — a player missing from it gets
    eo_pct=0 (maximally "differential" — a defensible default when EO is
    simply unavailable, e.g. pre-GW1, rather than crashing or assuming
    template-level ownership). With ``ownership=None`` entirely (the
    current live reality — EO sampling can't produce real data until GW1),
    every player gets eo_pct=0, so ``effective_score`` reduces to
    ``xpts*(1+λ) + μ*xpts_var`` for everyone uniformly — a CONSTANT
    rescaling that doesn't change relative ranking at all, regardless of
    EO availability.
    """
    out = projections.copy()
    if "xpts_var" not in out.columns:
        out["xpts_var"] = 0.0

    lam, mu = lambda_mu_for_risk_level(
        risk_level if risk_level is not None else config.risk_level,
        config.max_ownership_differential,
        config.mu_baseline,
        config.mu_range,
    )

    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        eo_pct = out["player_id"].map(eo_map).fillna(0.0)
    else:
        eo_pct = pd.Series(0.0, index=out.index)

    def _col(name):
        if name in out.columns:
            return out[name]
        return pd.Series([None] * len(out), index=out.index)

    out["effective_score"] = [
        risk_adjusted_score(xpts, var, eo, lam, mu, up, down)
        for xpts, var, eo, up, down in zip(
            out["xpts"], out["xpts_var"], eo_pct,
            _col("upside"), _col("downside"), strict=True,
        )
    ]
    return out
