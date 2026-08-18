# How this engine compares to the rest of the field

2026-08-18. A design-level comparison against the main public FPL decision
engines and the academic literature, prompted by "how much do we differ, and
what are the pros and cons".

## The field

There are three broadly distinct traditions, and they are not competing on the
same axis.

**AIrsenal** (Alan Turing Institute) is the closest thing to a peer: a full
open-source engine that builds its own predictions. Team strength comes from a
Bayesian hierarchical model fitted to historical results; player involvement
comes from a Dirichlet model over (goal, assist, neither) shares with
empirical-Bayes prior alphas, sampled with numpyro/NUTS. Squad selection is a
genetic algorithm; transfer planning is a brute-force search over zero, one or
two transfers across roughly three weeks.

**The optimiser community** — Sertalp Çay's FPL-Optimization-Tools and the
FPLReview solvers — split the problem in two. FPLReview sells the projections;
the tooling consumes them and does very sophisticated multi-period mixed-integer
programming on top. This tradition has by far the best *optimisation* layer and
deliberately does not build its own *prediction* layer.

**The academic literature** is, bluntly, behind both. The most recent
data-driven FPL selection paper uses ARIMA and weighted averages for point
forecasts, a single-week horizon, gives the bench a fixed residual budget and
excludes it from the objective, and models no substitution probability at all.
Its robust-optimisation variant underperforms its own deterministic baseline.

## Where we sit

The short version: **our prediction layer is ahead of the field; our
optimisation layer is behind the optimiser community.** We have been building
the harder half and under-investing in the easier one.

### Genuine strengths

**1. Full predictive distributions, not point estimates.** This is the biggest
difference and it is not close. AIrsenal and the MIP tooling both ultimately
optimise against expected points per player. We run 150 Monte Carlo scenarios
per fixture in which team goals are drawn *once per fixture-scenario* and every
dependent quantity conditions on that same draw — so clean sheets, concessions
and saves for a given team are genuinely correlated within a scenario, as they
are in reality. That is what makes scenario-based captaincy possible: doubling a
player changes the team's joint variance, which a per-player point estimate
cannot express.

**2. Odds-derived expected goals.** We de-vig bookmaker 1X2 and over/under 2.5
prices into a double-Poisson λ. AIrsenal infers the same quantity from
historical results via MCMC. For the next match specifically, the market is
very hard to beat, and this is probably a real edge over a hierarchical fit.
The catch is in the weaknesses below.

**3. Explicit BPS and bonus simulation.** We simulate reduced-BPS event rows
per fixture-scenario across both teams and rank them for the bonus points.
Most engines treat bonus as a per-player historical average, which cannot
represent "he only gets bonus when his team keeps a clean sheet".

**4. Component-level scoring including DefCon.** The 26/27 defensive-contribution
rules are modelled explicitly rather than absorbed into a historical average.

**5. Optimiser's-curse correction.** Per-player empirical-Bayes shrinkage by
estimation standard error, price-banded so an unknown regresses toward players
at his own price rather than toward the league. I have not found this anywhere
in the public FPL tooling, and the effect is real — the top-50 players by
projection carried a consistent +1.2 to +1.3 point bias before it.

**6. Validation infrastructure.** Backtest, walk-forward gate, a preflight
decision-surface baseline that fails when the answer changes, 819 tests across
85 files, and a ~100-persona shadow simulation cohort. Most hobby engines have
essentially none of this, and it is the reason defects in this project get
found rather than silently absorbed.

**7. A risk axis at all.** One-sided upside/downside semi-deviation selected by
appetite. Almost nobody models risk in FPL classic; the one academic attempt
uses a box uncertainty set and underperforms its own baseline.

### Genuine weaknesses

**1. No gameweek decay. This is the clearest actionable gap.** Çay's
`solve_multi_period_fpl` defaults to `decay_base = 0.84`; FPLReview recommends
0.80–0.95. We sum our five-gameweek horizon with **equal weight**. That is
indefensible on our own evidence: the decision explainer measures that 22% of
the squad's projected points come from gameweeks with real odds and 78% from
the strength model, with 17 of 20 teams still on prior-season fallback. We are
weighting our least reliable numbers exactly as heavily as our most reliable
ones. Decay is the standard correction and we simply do not have it.

**2. Squad-level covariance is unmodelled.** The objective sums per-player
variances as if independent, so it cannot see a keeper and centre-back sharing
one clean sheet, and it happily fills both club slots to the three-player cap.
The daily-fantasy literature solves this with mixed-integer *quadratic*
programming; the machinery transfers even though the sign does not (DFS wants
correlation for top-heavy payouts, we want less of it). We already persist the
joint draws this needs.

**3. The strength-model fallback is our weakest link, and it carries most of
the weight.** Bookmakers price about one round ahead. Everything beyond that
runs on a regression fitted to team strength with most teams on prior-season
values. AIrsenal's Bayesian team model is a better answer for exactly this
regime. Our odds advantage is real but narrow, and the fallback is doing 78% of
the work.

**4. No price-change or team-value modelling.** `use_price_change_signals` was
removed as dead code. Serious FPL play treats team value as a compounding
resource — buying risers early funds better squads later. We ingest
`transfers_in_event` and do nothing with it.

**5. A thinner optimisation layer than the specialists.** Çay's tooling offers
configurable multi-week horizons, chip scenario planning, "no-flip"
constraints against churn, and solution pools of near-optimal alternatives. We
plan three gameweeks ahead and return one answer. The explainer's margin
column partly compensates by showing how close each pick was, but generating
genuine alternative squads would be better.

**6. No rank-aware objective in live use.** The ownership/EO term exists but is
dormant at `risk_level = 0`. Competitive FPL is a rank game and the template
matters. This one is a **deliberate choice** — the stated aim is expected-points
maximisation — so it is listed as a difference rather than a defect.

**7. Everything depends on our own minutes model.** Start probabilities now
drive projections, the candidate-pool filter, and (since today) the bench slot
weights. FPLReview's xMins is a heavily-tuned commercial product; ours is
in-house and untested against theirs. Errors here propagate further than
anywhere else in the system.

## What I would do next, in order

1. **Add gameweek decay** (~0.85). Cheapest change with the clearest external
   support, and our own provenance numbers argue for it independently.
2. **Improve the unpriced-gameweek strength model.** It carries 78% of the
   decision. A hierarchical model fitted across seasons would beat the current
   regression-with-fallback.
3. **Squad covariance via sample-based re-ranking.** Generate the top-N ILP
   solutions with no-good cuts, score them on true joint variance from the
   persisted draws. No new solver required.
4. **Solution pools in the explainer**, so alternatives are visible rather than
   inferred from margins.
5. **Team-value modelling**, if the season is going to be played out fully.

Items 1 and 3 are the ones that change decisions. Item 2 is the one that
changes how much the decisions can be trusted.

## Sources

- [AIrsenal — Alan Turing Institute](https://www.turing.ac.uk/news/airsenal)
- [AIrsenal source](https://github.com/alan-turing-institute/AIrsenal)
- [FPL-Optimization-Tools (Sertalp Çay)](https://github.com/sertalpbilal/FPL-Optimization-Tools)
- [FPLReview solver settings](https://docs.fplreview.com/the-model/solvers/settings/)
- [FPLReview solver guide](https://fplreview.com/solving-for-points-instructional-guide/)
- [A data-driven framework for team selection in Fantasy Premier League](https://arxiv.org/html/2505.02170v2)
- [Competing in daily fantasy sports using generative models — Mlčoch et al., ITOR 2024](https://onlinelibrary.wiley.com/doi/10.1111/itor.13344)
