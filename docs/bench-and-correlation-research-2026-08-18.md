# Bench value and squad correlation: what other engines do

2026-08-18. Prompted by two observations on the live GW1 squad: three Arsenal
players (plus three Man City), and a bench of four players who were never
going to score. The question was whether either is a defect or a defensible
choice, and what established practice is.

## 1. The bench

### What we were doing

A single flat `bench_value_weight = 0.15` applied to every bench player. The
result: four £4.0–4.5m enablers averaging ~2.0 xPts a gameweek, all four with
a re-solve margin of exactly 0.00 — the optimiser was not choosing them, it
was satisfying a constraint.

### Why a flat weight is wrong

FPL performs **automatic substitutions**: if a starter does not play, the
first eligible bench player is promoted. A bench is therefore an ordered
queue, not a set, and the slots have very different odds of being reached.

Measured from this engine's own start probabilities on the live GW1 XI
(mean P(start) = 0.93, min 0.79):

| bench slot | P(reached) |
| --- | --- |
| 1 | **0.53** |
| 2 | 0.15 |
| 3 | 0.03 |

A flat 0.15 is wrong in both directions at once: it underpays slot 1 by more
than three times and overpays slot 3 by five times.

### What others do

**AIrsenal** (Alan Turing Institute) uses fixed, hand-tuned, position-ordered
weights:

```python
DEFAULT_SUB_WEIGHTS = {"GK": 0.03, "Outfield": (0.65, 0.3, 0.1)}
```

Same shape and roughly the same magnitudes as our data-derived numbers,
arrived at independently. That is decent corroboration that the ordering is
real and the values are not wildly off.

**FPLReview's solver** goes further: it derives the bench contribution from
expected minutes and availability per player rather than using constants, and
exposes a scalar `bench_weight` where 1.0 keeps that derived value and 0.0
ignores the bench entirely. It also has `xmin_lb`, a hard minutes floor for
the candidate pool — our `min_start_probability = 0.4` is the same idea.

**The academic literature is behind the practitioners here.** The most recent
data-driven FPL selection paper gives the bench a fixed residual budget and
excludes bench players from the objective altogether — it models no
substitution probability at all, treating missing player-gameweek rows as
unavailability.

### What we changed

Replaced the flat weight with per-slot weights `(0.53, 0.15, 0.03)` plus a
separate `bench_gk_weight = 0.03`, implemented as bench-slot assignment
variables in the ILP. `bench_value_weight` survives as a **multiplier** on
those (FPLReview's convention): 1.0 keeps them, 0.0 ignores the bench.

Applied to **both** `optimiser/squad.py` and `optimiser/transfers.py`. That
second one matters: the squad build now pays real money for a substitute who
plays, and had the weekly transfer ILP kept valuing bench players at a flat
rate it would have sold him the next week and charged a transfer for it.

### Measured effect on the live GW1 squad

| scheme | XI+captain xPts (5 GW) | first substitute |
| --- | --- | --- |
| flat 0.15 (old) | 329.23 | Targett, £4.0m, 1.76/GW |
| **slot-weighted (new)** | **323.93** | **Tarkowski, £6.0m, 4.79/GW** |
| AIrsenal's constants | 323.93 | Tarkowski (identical squad) |
| bench ignored (0.0) | 329.23 | pure fodder |

The trade: give up 5.30 xPts of XI strength over five gameweeks (1.06/GW) to
hold a first substitute worth 4.79/GW instead of 1.76/GW, in the 53% of
gameweeks he is needed. Expected value ≈ 0.53 × 3.03 − 1.06 ≈ **+0.55 xPts
per gameweek**, or roughly +21 over a season.

That the AIrsenal constants produce a byte-identical squad is reassuring —
the result is not balanced on the exact weights.

### What is still missing

The weights are static, so they do not tighten as a squad becomes more
nailed-on. Deriving them per solve from the chosen XI's minutes is better and
is what FPLReview does; it is circular (the weights depend on the XI being
chosen) and wants a fixed-point pass. Worth doing, not three days before a
deadline.

## 2. Correlation and club concentration

### The problem

Two clubs sit at the three-player cap. That is a corner solution: the
optimiser took every player the rules allowed and would have taken more.
Meanwhile `_multi_gw_var` sums per-player variances **as if independent**, so
nothing in the objective sees that Raya and Gabriel share a single Arsenal
clean sheet — one event, paid twice, modelled as two uncorrelated bets.

### What others do

Nobody in FPL classic handles this well. The recent FPL paper's only
club-level constraint is the standard three-per-club quota, and its attempt at
robustness (a box uncertainty set on expected points) "largely tracks or
underperforms" the deterministic version.

The real work is in **daily fantasy sports**, where the term is *stacking*.
Mlčoch et al. (2024) sample player point distributions from generative models
and optimise mean, variance **and covariance** directly in a mixed-integer
*quadratic* program.

**Their conclusion does not transfer, and it is important to say why.** DFS
tournaments have top-heavy payouts, so correlation is deliberately *sought* —
you need a high-variance lineup to win a jackpot. FPL classic is rank-based
over 38 gameweeks against a maximise-expected-points objective, which is much
closer to mean-variance. For us correlation is a **cost**, not a feature. The
DFS machinery is right; the sign is not.

### Why we can do this, and why not yet

We already persist joint Monte Carlo draws (`projection_samples`, with
disjoint per-fixture scenario ranges), and `optimiser/captaincy.py` already
uses them to compute true team-total variance. The inputs Mlčoch et al. sample
are the inputs we already have.

The blocker is that squad selection is a **linear** ILP and PuLP/CBC cannot
take a quadratic objective. Three options:

1. **Sample-based re-ranking.** Generate the top-N squads from the existing
   ILP with no-good cuts, then score each on true sample-based variance using
   the persisted draws. Reuses everything, needs no new solver. Cheapest path
   and the one I would take first.
2. **Linearised proxy.** Penalise club concentration directly (a soft cost on
   the third player from a club). Crude, but it targets the observed symptom.
3. **A real MIQP solver.** Correct, and the largest change.

Not before the deadline. Option 1 after GW1.

## Sources

- [AIrsenal — Alan Turing Institute](https://www.turing.ac.uk/news/airsenal)
- [AIrsenal source (`optimization_utils.py`, `DEFAULT_SUB_WEIGHTS`)](https://github.com/alan-turing-institute/AIrsenal)
- [FPLReview solver settings — bench weights and `xmin_lb`](https://docs.fplreview.com/the-model/solvers/settings/)
- [FPL-Optimization-Tools (Sertalp Çay)](https://github.com/sertalpbilal/FPL-Optimization-Tools)
- [A data-driven framework for team selection in Fantasy Premier League](https://arxiv.org/html/2505.02170v2)
- [Competing in daily fantasy sports using generative models — Mlčoch et al., ITOR 2024](https://onlinelibrary.wiley.com/doi/10.1111/itor.13344)
- [Sharpstack: Cholesky correlations for building better lineups](https://cdn.prod.website-files.com/5f1af76ed86d6771ad48324b/607a4434a565aa7763bd1312_AndyAsh-Sharpstack-RPpaper.pdf)
