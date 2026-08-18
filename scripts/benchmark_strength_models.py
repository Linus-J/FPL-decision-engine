#!/usr/bin/env python
"""benchmark_strength_models.py — is the Dixon-Coles fit actually better?

The unpriced-gameweek fallback carried 78% of the live GW1 decision, so it is
worth knowing which fallback is best rather than assuming. Both candidates are
scored against the same target on the same fixtures: the odds-implied lambda
that `team_goals_from_odds` returns, which is the quantity the fallback exists
to approximate and the closest thing to ground truth available.

Held out by SEASON and WALK-FORWARD: each season is predicted using only the
seasons BEFORE it. A random split would let the model see other matches
involving the same teams in the same season, which is most of what it needs to
know. Training on later seasons is subtler and just as wrong — it predicts the
past from the future, which is not a position the engine is ever in, and it
interacts badly with recency weighting (the most heavily weighted training
matches end up the most temporally distant from the target). A first pass here
did exactly that and made Dixon-Coles look 15% worse than it is.

Two comparisons for the published-strength model, because they answer different
questions:

  - `strength (same season)` uses the held-out season's own published ratings.
    That is what the model does mid-season, and it is a generous benchmark: the
    fitted exponents in team_goals.py were themselves calibrated on all five
    seasons, so this row is partly in-sample.
  - `strength (prior season)` uses the previous season's ratings. That is what
    the model ACTUALLY does at GW1, when FPL publishes attack and defence as
    zero and every team falls through — which is the case the fallback carries
    the most weight in, and the case worth deciding on.

    uv run python scripts/benchmark_strength_models.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from data.db import get_session  # noqa: E402
from projection.dixon_coles import fit_dixon_coles, load_match_results  # noqa: E402
from projection.team_goals import team_goals_from_odds, team_goals_from_strength  # noqa: E402


def _odds_targets() -> pd.DataFrame:
    """Odds-implied lambda per historical fixture — the benchmark target."""
    db = get_session()
    try:
        odds = pd.read_sql(text("""
            SELECT season, gameweek, home_team_id, away_team_id,
                   home_win_prob, draw_prob, away_win_prob, over25_prob
            FROM historical_fixture_odds
        """), db.bind)
        strength = pd.read_sql(text("""
            SELECT season, team_id, code,
                   strength_attack_home, strength_attack_away,
                   strength_defence_home, strength_defence_away
            FROM team_season_strength
        """), db.bind)
    finally:
        db.close()

    rows = []
    for r in odds.itertuples():
        try:
            lam_h, lam_a = team_goals_from_odds(
                r.home_win_prob, r.draw_prob, r.away_win_prob, r.over25_prob
            )
        except Exception:
            continue
        rows.append({
            "season": r.season, "gameweek": r.gameweek,
            "home_team_id": r.home_team_id, "away_team_id": r.away_team_id,
            "lam_home": lam_h, "lam_away": lam_a,
        })
    return pd.DataFrame(rows), strength


def _strength_rel(strength: pd.DataFrame, season: str) -> dict:
    sub = strength[strength["season"] == season]
    if sub.empty:
        return {}
    att = pd.concat([sub["strength_attack_home"], sub["strength_attack_away"]]).mean()
    dfn = pd.concat([sub["strength_defence_home"], sub["strength_defence_away"]]).mean()
    if not att or not dfn:
        return {}
    return {
        int(r.team_id): {
            "attack": (r.strength_attack_home + r.strength_attack_away) / 2 / att,
            "defence": (r.strength_defence_home + r.strength_defence_away) / 2 / dfn,
            "code": int(r.code),
        }
        for r in sub.itertuples()
    }


def main() -> int:
    targets, strength = _odds_targets()
    if targets.empty:
        print("no historical odds available — nothing to benchmark against")
        return 1
    matches = load_match_results()
    if matches.empty:
        print("no reconstructable match results — cannot fit Dixon-Coles")
        return 1

    seasons = sorted(targets["season"].unique())
    print(f"fixtures with odds: {len(targets)} across {len(seasons)} seasons\n")

    rows = []
    for i, held_out in enumerate(seasons):
        if i == 0:
            continue  # nothing earlier to train on
        prior_seasons = seasons[:i]
        train = matches[matches["season"].isin(prior_seasons)]
        if train.empty:
            continue
        fit = fit_dixon_coles(train)
        rel = _strength_rel(strength, held_out)
        rel_prior = _strength_rel(strength, prior_seasons[-1])
        # The prior season's ratings are keyed by that season's team ids; the
        # cross-season code is what carries identity across the boundary.
        prior_by_code = {v["code"]: v for v in rel_prior.values()}
        test = targets[targets["season"] == held_out]

        err = {k: [] for k in ("dc_h", "dc_a", "st_h", "st_a", "pr_h", "pr_a", "fl_h", "fl_a")}
        for r in test.itertuples():
            home = rel.get(int(r.home_team_id))
            away = rel.get(int(r.away_team_id))
            if not home or not away:
                continue
            dh, da = fit.lambdas(home["code"], away["code"])
            sh, sa = team_goals_from_strength(
                home["attack"], home["defence"], away["attack"], away["defence"]
            )
            ph_src = prior_by_code.get(home["code"])
            pa_src = prior_by_code.get(away["code"])
            ph, pa = team_goals_from_strength(
                ph_src["attack"] if ph_src else None,
                ph_src["defence"] if ph_src else None,
                pa_src["attack"] if pa_src else None,
                pa_src["defence"] if pa_src else None,
            )
            err["dc_h"].append(abs(dh - r.lam_home))
            err["dc_a"].append(abs(da - r.lam_away))
            err["st_h"].append(abs(sh - r.lam_home))
            err["st_a"].append(abs(sa - r.lam_away))
            err["pr_h"].append(abs(ph - r.lam_home))
            err["pr_a"].append(abs(pa - r.lam_away))
            err["fl_h"].append(abs(1.5102 - r.lam_home))
            err["fl_a"].append(abs(1.2232 - r.lam_away))

        if not err["dc_h"]:
            continue
        rows.append({
            "held-out season": held_out, "fixtures": len(err["dc_h"]),
            "DC home": np.mean(err["dc_h"]), "DC away": np.mean(err["dc_a"]),
            "strength home": np.mean(err["st_h"]), "strength away": np.mean(err["st_a"]),
            "prior-str home": np.mean(err["pr_h"]), "prior-str away": np.mean(err["pr_a"]),
            "flat home": np.mean(err["fl_h"]), "flat away": np.mean(err["fl_a"]),
        })

    if not rows:
        print("no season had both odds and published strengths — cannot compare")
        return 1

    table = pd.DataFrame(rows)
    print("Mean absolute error against odds-implied lambda (lower is better):\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nWeighted over all held-out fixtures:")
    w = table["fixtures"]
    means = {}
    for label in ("DC", "strength", "prior-str", "flat"):
        h = float(np.average(table[f"{label} home"], weights=w))
        a = float(np.average(table[f"{label} away"], weights=w))
        means[label] = (h + a) / 2
        print(f"  {label:>10}: home {h:.3f}  away {a:.3f}  mean {means[label]:.3f}")

    print("\nThe comparison that matters is against `prior-str`, which is what the")
    print("engine actually runs at GW1 — published ratings are zero until a season")
    print("is underway, so every team falls through to the previous year's.")
    dc, prior, same = means["DC"], means["prior-str"], means["strength"]
    verdict = "BETTER" if dc < prior else "NOT better"
    print(f"\n  Dixon-Coles vs GW1 reality  : {verdict} "
          f"({dc:.3f} vs {prior:.3f}, {100 * (prior - dc) / prior:+.1f}%)")
    print(f"  Dixon-Coles vs mid-season   : {'better' if dc < same else 'not better'} "
          f"({dc:.3f} vs {same:.3f}) — and that row is partly in-sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
