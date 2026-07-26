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
differentials depending on ``risk_mode``. ``variance`` is linear own-variance
only (``μ * xpts_var``) — TEAMMATE COVARIANCE is quadratic in a 0/1 selection
vector (``w'Σw``) and cannot be expressed in the current linear MILP
(`pulp`/CBC) without linearisation tricks or a QP solver. The plan itself
frames true covariance-aware risk as "Objective v2 (upgrade): scenario-based
stochastic programming" — not a v1 requirement. P3-1 already persists the
raw MC scenario samples a v2 implementation would need.

Pure + deterministic. No DB/network.
"""

from __future__ import annotations

import pandas as pd

from config.strategy import OPTIMISER, OptimiserConfig

_RISK_MODE_SIGN = {"safe": -1.0, "balanced": 0.0, "aggressive": 1.0}


def lambda_mu_for_risk_mode(
    risk_mode: str,
    lambda_magnitude: float,
    mu_magnitude: float,
) -> tuple[float, float]:
    """(λ, μ) from a risk posture. "safe" penalises both differentials and
    variance (negative sign — prefer template, low-variance picks);
    "aggressive" rewards both (chase differentials and upside variance);
    "balanced" is a no-op (0, 0) — pure E[pts], today's existing behaviour.
    Unknown risk_mode strings default to balanced (0, 0) rather than
    raising — a config typo should degrade to the old behaviour, not crash
    the optimiser."""
    sign = _RISK_MODE_SIGN.get(risk_mode, 0.0)
    return sign * lambda_magnitude, sign * mu_magnitude


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
) -> float:
    """Per-player-per-GW ILP objective coefficient, replacing raw xpts.
    Linear in the selection variable, so it drops into the existing
    squad.py/transfers.py MILP formulations with no solver change."""
    return xpts * differential_multiplier(eo_pct, lam) + mu * xpts_var


def add_effective_score(
    projections: pd.DataFrame,
    ownership: pd.DataFrame | None = None,
    config: OptimiserConfig = OPTIMISER,
    risk_mode: str | None = None,
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
    rescaling that doesn't change relative ranking at all, i.e. this
    degrades to the pre-P3-3 behaviour exactly when EO isn't available,
    not silently to something else.
    """
    out = projections.copy()
    if "xpts_var" not in out.columns:
        out["xpts_var"] = 0.0

    lam, mu = lambda_mu_for_risk_mode(
        risk_mode if risk_mode is not None else config.risk_mode,
        config.max_ownership_differential,
        config.variance_weight,
    )

    if ownership is not None and not ownership.empty:
        eo_map = ownership.set_index("player_id")["top10k_selected_pct"]
        eo_pct = out["player_id"].map(eo_map).fillna(0.0)
    else:
        eo_pct = pd.Series(0.0, index=out.index)

    out["effective_score"] = [
        risk_adjusted_score(xpts, var, eo, lam, mu)
        for xpts, var, eo in zip(out["xpts"], out["xpts_var"], eo_pct, strict=True)
    ]
    return out
