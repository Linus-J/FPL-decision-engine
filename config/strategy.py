"""
strategy.py — all parameters that are likely to change season-to-season.

When FPL announces 26/27 rule changes, this is the ONLY file you should need
to edit for scoring, chip, and structural rule changes.
"""

from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# SCORING RULES
# These mirror the official FPL points system. Update if Opta/FPL change
# the scoring structure (e.g. 25/26 introduced no points changes, but
# prior seasons changed assist/CS rules).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringRules:
    # --- Minutes played ---
    points_sub_appearance: int = 1       # played but < 60 mins
    points_full_appearance: int = 2      # 60+ mins

    # --- Goals scored ---
    points_goal_gk: int = 10
    points_goal_def: int = 6
    points_goal_mid: int = 5
    points_goal_fwd: int = 4

    # --- Assists ---
    points_assist: int = 3

    # --- Clean sheets ---
    points_cs_gk: int = 4                 # official FPL standard (was wrongly 6)
    points_cs_def: int = 4                # official FPL standard (was wrongly 6)
    points_cs_mid: int = 1               # mid CS bonus (was 0 pre-2019)
    points_cs_fwd: int = 0

    # --- Goals conceded (GK + DEF only) ---
    goals_conceded_per_penalty: int = 2  # every N goals conceded = -1 pt
    points_goals_conceded_penalty: int = -1

    # --- Saves (GK) ---
    saves_per_bonus_point: int = 3       # every N saves = +1 pt
    points_save_bonus: int = 1

    # --- Penalties ---
    points_penalty_save: int = 5
    points_penalty_miss: int = -2

    # --- Discipline ---
    points_yellow_card: int = -1
    points_red_card: int = -3

    # --- Bonus points (final award — see BPSWeights for the metric table) ---
    points_bps_first: int = 3
    points_bps_second: int = 2
    points_bps_third: int = 1


# ---------------------------------------------------------------------------
# DEFENSIVE CONTRIBUTION (DefCon) — introduced 25/26, unchanged in 26/27.
# Awards real FPL points (not BPS). Capped per match. See plan Appendix A.2.
# DEF counts CBIT (clearances+blocks+interceptions+tackles); MID/FWD count
# CBIRT (adds recoveries) and need a higher threshold.
# NOTE: the full-back DefCon threshold was flagged "under review" for 26/27;
# treat 10 as current until FPL confirms otherwise.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DefConRules:
    def_threshold: int = 10              # DEF: CBIT count for the award
    mid_fwd_threshold: int = 12          # MID/FWD: CBIRT count for the award
    points: int = 2                      # points granted when threshold met
    cap_per_match: int = 2               # max DefCon points per player per match


# ---------------------------------------------------------------------------
# BONUS POINTS SYSTEM (BPS) — 26/27 metric weights. Source of truth for the
# BPS simulator (plan §4.7 / Appendix A.3). These are the per-action BPS
# contributions; the top-3 BPS totals in a match receive 3/2/1 bonus points.
# 26/27 deltas vs 25/26 are called out inline.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BPSWeights:
    # --- Appearance ---
    play_1_to_60: int = 3
    play_over_60: int = 6

    # --- Goals (by position) ---
    goal_gk: int = 12
    goal_def: int = 12
    goal_mid: int = 18
    goal_fwd: int = 24
    winning_goal: int = 3

    # --- Attacking contribution ---
    assist: int = 9
    big_chance_created: int = 3
    key_pass: int = 1
    open_play_cross: int = 1
    successful_dribble: int = 1

    # --- Clean sheet / goalkeeping ---
    clean_sheet_gk_def: int = 12
    save: int = 2                        # 26/27: flat +2 for ANY save (out-of-box metric removed)
    save_inside_box_extra: int = 1       # additional +1 when the save is from inside the box
    big_chance_saved: int = 1            # 26/27: NEW
    # 26/27: penalty-save BPS dropped 8->7; a penalty is a big chance, so
    # big_chance_saved (+1) stacks to net 8. Exact stacking of `save`/inside-box
    # on penalties is UNVERIFIED-STACKING — validate vs early-season observed BPS.
    penalty_saved: int = 7

    # --- Defensive actions ---
    successful_tackle: int = 2
    cbi_per_point: int = 3               # 26/27: +1 per 3 CBI (was per 2) to de-overlap with DefCon
    recoveries_per_point: int = 3        # +1 per 3 recoveries

    # --- Passing accuracy (requires 30+ passes) ---
    pass_completion_min_passes: int = 30
    pass_completion_70_79: int = 2
    pass_completion_80_89: int = 4
    pass_completion_90_plus: int = 6

    # --- Negative ---
    # 26/27: "being tackled" penalty REMOVED (was -1) so dribblers aren't punished.
    being_tackled: int = 0
    conceding_penalty: int = -3
    missing_penalty: int = -6
    yellow_card: int = -3
    red_card: int = -9
    own_goal: int = -6
    missing_big_chance: int = -3
    error_leading_to_goal: int = -3
    error_leading_to_shot: int = -1
    conceding_foul: int = -1
    caught_offside: int = -1
    shot_off_target: int = -1


# 25/26 ("old-rules") BPS weights: the four numeric changes 26/27 made vs the
# prior season, reconstructed from the inline deltas above. This is the source
# of truth for the T5b sanity harness (recompute historical bonus under the
# rules that were actually in force, then compare to FPL's awarded bonus).
# Built below the singletons via dataclasses.replace — see BPS_WEIGHTS_2526.


# ---------------------------------------------------------------------------
# CHIP RULES
# Chips available and how many of each per season.
# Real bug found 2026-07-28 (user's own squad-trace review: "only one
# wildcard chip was played when we should have 2 of each"). This module
# previously claimed 25/26 gave 2x Wildcard but only 1x each of Free Hit/
# Bench Boost/Triple Captain for the whole season -- WRONG, confirmed via
# the Premier League's own 2025/26 changes announcement: 2025/26 is a major
# rules change giving 1 of EACH of the 4 chips per half of the season (8
# chips total), with NO carryover -- an unused first-half chip is lost at
# the GW19 deadline, not banked for the second half. Update here if 26/27
# changes this again.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipRules:
    wildcards_total: int = 2             # total WCs per season (1 per half)
    wildcard_first_half_deadline_gw: int = 19  # GW after which WC1/FH1/BB1/TC1 expire
    free_hits_total: int = 2             # 1 per half, no carryover
    bench_boosts_total: int = 2          # 1 per half, no carryover
    triple_captains_total: int = 2       # 1 per half, no carryover

    # Can two chips be played in the same GW? (FPL rule: no)
    allow_chip_stacking: bool = False


# ---------------------------------------------------------------------------
# FREE TRANSFER RULES
# 25/26: 1 free transfer per GW, bank up to 2. 26/27 may change this.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferRules:
    free_transfers_per_gw: int = 1
    max_banked_free_transfers: int = 5
    hit_cost_points: int = -4
    max_hits_per_gw: int = 2
    ft_terminal_value: float = 2.0

    # 2026-07-29 (user's own squad-trace review, real numbers): Bruno
    # Fernandes was sold at GW10 despite 4 straight nailed-on, solidly-
    # scoring gameweeks (90 min, 3/8/4/5 pts), replaced by Gakpo, who
    # proved less reliable (including his own real injury/rotation gap) --
    # a premium, proven performer churned for a WITHIN-FREE-TRANSFER swap
    # that the hit-cost mechanism has zero power to discourage (it only
    # taxes transfers BEYOND the free allowance). This is the P3-7
    # optimiser's-curse pattern surviving in a different shape: even
    # curse-shrunk projections still carry noise, and a proven track
    # record has value beyond what one week's projection captures. A
    # flat, always-on cost per transfer made (independent of hits) forces
    # every swap -- free or not -- to clear a real bar, not just a
    # marginal, noise-sized edge. Untuned starting value pending
    # backtesting, same convention as other heuristic constants this
    # session; 0.0 disables it exactly.
    transfer_switching_cost: float = 1.5


# ---------------------------------------------------------------------------
# SQUAD STRUCTURE
# Standard FPL squad constraints. Update if FPL changes squad size or
# position quotas (unlikely but parameterised for safety).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SquadRules:
    squad_size: int = 15
    budget_total: float = 100.0          # £m at season start
    max_players_per_club: int = 3

    # Squad composition
    gk_count: int = 2
    def_count: int = 5
    mid_count: int = 5
    fwd_count: int = 3

    # Starting XI valid ranges per position
    starting_gk: int = 1
    starting_def_min: int = 3
    starting_def_max: int = 5
    starting_mid_min: int = 2
    starting_mid_max: int = 5
    starting_fwd_min: int = 1
    starting_fwd_max: int = 3


# ---------------------------------------------------------------------------
# DOUBLE GAMEWEEK (DGW) STRATEGY
# Controls how aggressively the bot builds for and targets DGWs.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DGWStrategy:
    # How many GWs ahead to scan for confirmed DGWs
    lookahead_gws: int = 6

    # Minimum DGW starters to target when a DGW is within lookahead window
    target_dgw_starters: int = 9

    # How many extra hits we're willing to take to build DGW coverage
    max_extra_hits_for_dgw: int = 2

    # Point premium applied to DGW players' xPts (they play twice)
    # Set to ~1.85 rather than 2.0 to discount rotation/injury risk in DGW
    dgw_xpts_multiplier: float = 1.85

    # BGW (blank GW) discount — players with blanks get xPts multiplied by this
    bgw_xpts_multiplier: float = 0.0     # blanked player = 0 projected pts

    # When DGW is this many GWs away, start preferring DGW-eligible players
    # in transfers even if they're not the immediate best pick
    dgw_preparation_window_gws: int = 3


# ---------------------------------------------------------------------------
# CHIP TIMING THRESHOLDS
# Thresholds that trigger chip usage. Tune these after backtesting.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipTimingThresholds:
    wildcard_pts_gain_threshold: float = 25.0
    wildcard_eval_horizon_gws: int = 5
    wildcard_min_managed_gws: int = 6

    free_hit_single_gw_gain_threshold: float = 12.0

    bench_boost_min_bench_xpts: float = 20.0

    # 2026-07-30 (user's own review: "how can it ever be worth not playing
    # [TC]? It is only negative if the player gets < 0 points"). Rebased
    # from a GAP (best captain xPts minus the second-best) to the captain's
    # own ABSOLUTE projected points — TC doesn't change who you captain,
    # only the multiplier on whoever you'd already pick, so its real value
    # is one extra copy of THEIR points, not how far ahead they are of the
    # alternative. Low floor: clears for almost any healthy, nailed-on
    # captain pick; only declines on a genuinely weak/uncertain one
    # (rotation risk, tough fixture). Untuned starting value pending
    # backtesting, same convention as this session's other heuristic
    # constants.
    triple_captain_min_gain: float = 4.0

    # Multiplies (raises) the TC bar above when a DGW is visible within the
    # caller's dgw_gws lookahead but hasn't arrived yet — the real
    # remaining tradeoff once TC's own EV is ~always positive is scarcity
    # (only 1 use per half, no carryover): is this week worth spending it
    # on, versus a likely-bigger double-fixture haul coming up this half.
    triple_captain_dgw_wait_multiplier: float = 2.5

    # P3-5: minimum P(gain >= 0) over real persisted MC scenarios (P3-1)
    # required, IN ADDITION to the point-estimate thresholds above, before a
    # chip is recommended. Only applied when real samples exist for the
    # gameweek (live serving); the backtest walk-forward never persists
    # samples, so it always falls back to the point-estimate-only gates
    # above unchanged. Initial defaults, not yet backtested — tune alongside
    # the thresholds above.
    wildcard_min_payoff_probability: float = 0.6
    free_hit_min_payoff_probability: float = 0.6
    bench_boost_min_payoff_probability: float = 0.6
    triple_captain_min_payoff_probability: float = 0.6

    # 2026-07-30 (user's own squad-trace review: chips going completely
    # unused all season "is just not acceptable"). Every chip's per-half
    # threshold above was an all-or-nothing bar with no time-pressure —
    # a chip that never quite cleared it simply evaporated, unused, at the
    # half/season boundary. Within the final `panic_window_gws` gameweeks of
    # its half, every chip's threshold is linearly shrunk toward
    # `panic_threshold_shrink` (fraction of the normal bar) so a marginal,
    # still-real opportunity is far more likely to clear it before expiring.
    # As a genuine last resort — the user's own words: "at worst the default
    # behaviour is to panic and use the triple captain on the last day" —
    # `recommend_chip` force-plays Triple Captain on the half's very last
    # gameweek if it's still unused and nothing else fired, since doubling
    # that week's best captain candidate is close to always worth something,
    # unlike Free Hit/Bench Boost/Wildcard, which need a real structural
    # opportunity (a DGW/BGW/rebuild) to be worth anything at all.
    panic_window_gws: int = 3
    panic_threshold_shrink: float = 0.3


# ---------------------------------------------------------------------------
# OPTIMISER BEHAVIOUR
# Controls the projection horizon and risk profile.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimiserConfig:
    # Number of GWs to project ahead for transfer decisions
    transfer_planning_horizon_gws: int = 3

    # Rolling window for recent form (xG, xA, minutes)
    form_window_gws: int = 5

    # Minimum P(starts) threshold — players below this are excluded
    min_start_probability: float = 0.4

    # Extra points gain required above the raw hit cost to justify taking a hit.
    # With hit_cost=-4 and buffer=2, we only hit when net gain >= 6 over horizon.
    hit_min_gain_buffer: float = 2.0

    # Ownership-differential weight magnitude (P3-3, plan §5's λ) — was a
    # dead field (read nowhere) until optimiser/scoring.py wired it in.
    # Sign comes from risk_level, not from this value directly: negative
    # risk_level -> penalises differentials (prefer template); positive ->
    # rewards them; risk_level=0 -> 0 (no EO effect at all). This is the
    # MAGNITUDE only.
    max_ownership_differential: float = 0.5

    # Whether to factor in price change predictions
    use_price_change_signals: bool = True

    # Continuous risk posture in [-1.0, 1.0]: -1.0 = safe, 0.0 = medium,
    # +1.0 = aggressive (plan/risk-aware-cold-start-v1.md, 2026-07-31,
    # superseding the old 3-way "safe"/"balanced"/"aggressive" string
    # switch). Sets lambda linearly (see max_ownership_differential above,
    # zero at risk_level=0) and shifts mu around mu_baseline (see below) —
    # deliberately NOT the same shape, since "medium" should carry real
    # variance-awareness rather than none.
    risk_level: float = 0.0

    # mu = mu_baseline + risk_level * mu_range (optimiser/scoring.py::
    # lambda_mu_for_risk_level). mu_baseline is the genuine "medium"
    # variance-awareness (own-variance only, xpts_var per player; teammate
    # COVARIANCE is quadratic in a 0/1 selection and needs the v2
    # scenario-based objective, not this linear MILP) — without it,
    # risk_level=0 would silently mean "ignore variance entirely", which is
    # not what "medium risk" should mean. mu_range is the spread risk_level
    # scales across; risk_level=-1 can go net negative (actively prefer
    # low-variance picks at equal mean). mu_range is still an untuned
    # starting value, pending backtesting.
    #
    # mu_baseline calibrated 2026-07-31 (scripts/calibrate_risk_constants.py):
    # swept [-0.05, 0.0, 0.05, 0.1, 0.15, 0.2] against the naive-XI exit-gate
    # over GW6-20/2025-26 -- 0.0 won (57.67 avg actual pts/GW vs the prior
    # 0.05 default's 57.27). Reduced window for speed, not a final gate
    # validation; re-run GW6-38 to firm this up.
    mu_baseline: float = 0.0
    mu_range: float = 0.08

    # Optimiser's-curse shrinkage (2026-07-28 data-completeness audit,
    # superseding the narrower P3-6 transfer_variance_penalty): shrinks
    # xpts toward its (gameweek, position) group mean before ANY selection
    # step sees it (see projection.assemble.apply_curse_shrinkage) — an
    # always-on bias correction, independent of OPTIMISER.risk_level (which
    # stays a pure preference dial). Confirmed empirically: the raw model is
    # ~unbiased across the whole player pool, but the top-50 players by
    # projected xpts each week showed a consistent +1.2-1.3 pt/player bias
    # -- exactly the pool squad-building/captaincy/transfers all select
    # from. False disables it exactly (byte-identical to pre-this-fix
    # behaviour) for comparison/debugging.
    curse_shrinkage_enabled: bool = True

    # 2026-07-30 (user's own review): optimise_squad's objective only summed
    # scores for STARTING players (`scores[i] * (starting[i] + captain[i])`)
    # -- a bench player contributed exactly zero to the objective regardless
    # of quality, so once the best starting XI was picked, the solver had no
    # incentive to do anything but fill the remaining budget with the
    # cheapest feasible fodder (the user's own examples: a Leeds/Burnley
    # enabler GK/DEF, a bench so weak it's only fine if minutes predictions
    # were 100% accurate -- they aren't; there's always some chance of a
    # last-minute injury, a freak bench decision, or the minutes model just
    # being wrong). A fractional weight on bench players' own scores gives
    # the solver a real reason to prefer a decent backup over the cheapest
    # possible one, without letting bench quality compete with the starting
    # XI for budget on equal terms. Untuned starting value pending
    # backtesting, same convention as this session's other heuristic
    # constants.
    bench_value_weight: float = 0.15


# ---------------------------------------------------------------------------
# DEPARTURE RISK (v2-build-plan §6.5) — squad-construction gate, LIVE from the
# initial-15 build (not shadow-mode like the rest of the news layer): the
# asymmetric cost of picking a rumoured/confirmed leaver into the initial 15
# is high enough to act on immediately, not wait for A/B validation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepartureRiskRules:
    # p_leave >= this -> hard-exclude from the candidate pool entirely (never
    # picked, and force-sold immediately if already owned). FPL status='u' /
    # an element dropped from bootstrap is ground truth => p_leave=1.0,
    # already well above this threshold.
    hard_exclude_p_leave: float = 0.7
    # p_leave below this -> no effect (too uncertain a rumour to act on).
    rumour_floor_p_leave: float = 0.2
    # Treat this GW range as a mini-preseason re-plan trigger (incoming
    # signings enter cold-start, outgoing players get the departure gate
    # applied proactively) — approximate, season-tunable.
    january_window_start_gw: int = 20
    january_window_end_gw: int = 24


# ---------------------------------------------------------------------------
# SINGLETON INSTANCES — import these everywhere
# ---------------------------------------------------------------------------

SCORING = ScoringRules()
DEFCON = DefConRules()
BPS_WEIGHTS = BPSWeights()

# The four numeric BPS changes 26/27 made vs 25/26 (see inline deltas above).
# Used only by the T5b sanity harness to recompute historical bonus under the
# rules in force at the time before trusting the 26/27 recompute.
BPS_WEIGHTS_2526 = replace(
    BPS_WEIGHTS,
    being_tackled=-1,    # 26/27: removed (0)
    cbi_per_point=2,     # 26/27: 1 per 3 CBI (was per 2)
    penalty_saved=8,     # 26/27: 7
    big_chance_saved=0,  # 26/27: +1 (new metric)
)
CHIPS = ChipRules()
TRANSFERS = TransferRules()
SQUAD = SquadRules()
DGW = DGWStrategy()
CHIP_TIMING = ChipTimingThresholds()
OPTIMISER = OptimiserConfig()
DEPARTURE_RISK = DepartureRiskRules()
