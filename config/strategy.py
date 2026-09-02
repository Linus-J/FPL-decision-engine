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

    # points_bps_first/second/third removed 2026-08-01 -- confirmed dead,
    # no test coverage. The real 3/2/1 bonus award is hardcoded directly in
    # projection/bonus.py, which never read these.


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
    wildcard_first_half_deadline_gw: int = 19  # GW after which WC1/FH1/BB1/TC1 expire
    # wildcards_total/free_hits_total/bench_boosts_total/triple_captains_total/
    # allow_chip_stacking removed 2026-08-01 -- confirmed dead, no test
    # coverage. The real "1 per half, no carryover" cap is hardcoded directly
    # in optimiser/chips.py::_chip_uses_remaining, which never read these.


# ---------------------------------------------------------------------------
# FREE TRANSFER RULES
# 25/26: 1 free transfer per GW, bank up to 2. 26/27 may change this.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferRules:
    free_transfers_per_gw: int = 1
    max_banked_free_transfers: int = 5
    hit_cost_points: int = -4
    # NOT an FPL rule (flagged 2026-08-18, engine review §18). FPL allows
    # unlimited hits; this is a self-imposed guard against the optimiser
    # talking itself into a huge one-week rebuild off noisy projections, in the
    # same family as transfer_switching_cost. Left in place because it is
    # almost never binding in practice, but it IS a hard constraint on the ILP,
    # so if a plan ever looks oddly timid at 2 hits this is the first thing to
    # check. It does not apply on a wildcard, where hits are zeroed outright.
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

    # target_dgw_starters/max_extra_hits_for_dgw/dgw_xpts_multiplier/
    # bgw_xpts_multiplier/dgw_preparation_window_gws removed 2026-08-01 --
    # confirmed dead (P12 already flagged the two multipliers; the other
    # three turned out to be the same pattern). DGW/BGW-aware transfer
    # preference was never actually implemented beyond what falls out of
    # projection/assemble.py's per-fixture summing (P12).


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

    # --- horizon-total chip comparison (2026-09-02) -----------------------
    # recommend_chip gates each chip on a gain measured against DOING NOTHING
    # with the current squad, decided before evaluate_transfers ever runs. So
    # Free Hit and Wildcard are never compared against the transfer plan they
    # displace, and the free transfers a Free Hit preserves are never priced.
    # Those are opposing errors that partly cancel, which is why this has not
    # shown up as a visibly bad decision.
    #
    # LIVE from the start (2026-09-02). The cohort sweep below is the real
    # experiment and runs whatever this is set to, so leaving the bot on the
    # known-wrong baseline only delayed any benefit. Set False and every
    # existing path is byte-identical again -- that is the reverse gear, and
    # the legacy cohort personas exercise it every week.
    chip_comparison_enabled: bool = True
    # The common horizon every option is scored over. MUST be registered in
    # assert_horizons_consistent below.
    chip_comparison_horizon_gws: int = 5
    # Margins, NOT thresholds: how far a chip must beat the best non-chip plan
    # by, rather than how far it must beat doing nothing. Separate constants
    # because reusing the thresholds above would silently redefine a
    # calibrated number. They default to those values so the first run starts
    # from today's strictness -- but note 12.0 was calibrated as a gain over
    # DOING NOTHING, and as a margin over the best alternative it is a
    # stricter and so far unjustified bar. That is what the cohort axis is
    # sweeping.
    free_hit_comparison_margin: float = 12.0
    wildcard_comparison_margin: float = 25.0

    bench_boost_min_bench_xpts: float = 20.0

    # 2026-07-30 (user's own review: "how can it ever be worth not playing
    # [TC]? It is only negative if the player gets < 0 points"). Rebased
    # from a GAP (best captain xPts minus the second-best) to the captain's
    # own ABSOLUTE projected points — TC doesn't change who you captain,
    # only the multiplier on whoever you'd already pick, so its real value
    # is one extra copy of THEIR points, not how far ahead they are of the
    # Raised 4.0 -> 7.5 on 2026-08-25, on the first live evidence.
    #
    # At 4.0 this was a floor almost any nailed-on captain cleared, so the chip
    # fired at the FIRST opportunity of each half -- GW2 -- which is close to
    # the worst time to spend one of only two season uses. The old comment
    # called it "untuned pending backtesting", and it was.
    #
    # The bar is now "clearly better than this squad's own typical best week"
    # rather than "is a real captain". Measured over the GW2-6 projections: the
    # squad's best captain each week ran 6.56, 6.77, 6.87, 6.87 and 7.71, so a
    # typical peak is about 6.9 and 7.5 sits roughly 10% above it. GW2's 7.71
    # clears; the other four do not.
    #
    # Two things this number is NOT. It is not derived from a backtest -- one
    # gameweek of live projections is the whole sample. And it is an ABSOLUTE
    # bar on a scale the engine is currently known to inflate: GW1 came in at
    # bias +28.6 pts/GW across 90 personas, so if that holds, every projected
    # captain figure here is high and this bar is effectively tighter than it
    # looks. The principled version is relative -- "top decile of best-captain
    # weeks across the half" -- which needs a distribution the season has not
    # produced yet. Revisit once four or five gameweeks have been scored.
    #
    # Raising it cannot waste the chip: _panic_shrink drops every threshold to
    # panic_threshold_shrink (30%) over the final panic_window_gws (3)
    # gameweeks of a half, and recommend_chip force-plays an unused Triple
    # Captain on the half's last gameweek regardless.
    triple_captain_min_gain: float = 7.5

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
    # How many GWs the projection pipeline actually BUILDS and persists.
    # Must be >= every downstream consumer's own horizon, since each of them
    # slices this frame rather than requesting its own (P3.5, 2026-08-16).
    # Previously the pipeline built transfer_planning_horizon_gws (3) while
    # CHIP_TIMING.wildcard_eval_horizon_gws asked for 5, so the wildcard's
    # 25-point threshold was compared against a 3-gameweek gain -- and the
    # simulation personas cannot sweep a longer planning horizon than this.
    # Checked by assert_horizons_consistent() below.
    projection_horizon_gws: int = 5

    # Number of GWs to project ahead for transfer decisions
    transfer_planning_horizon_gws: int = 3

    # What the vice-captaincy is worth in the objective (2026-08-18).
    #
    # The armband passes to the vice ONLY when the captain does not feature, so
    # the vice's expected contribution is P(captain blanks) x his own score.
    # Until today `vice` appeared in the ILP's constraints -- exactly one, must
    # be a starter, must not be the captain -- and NOWHERE in its objective, so
    # every legal choice was equally optimal and the solver returned whichever
    # it happened to branch on. On the live GW1 frame that was Raya, at 5.53
    # xPts, while Gabriel sat in the same XI on 7.43: the insurance was being
    # spent at random, on a goalkeeper.
    #
    # 0.15 is P(a captain-calibre player does not feature), measured on this
    # engine's own start probabilities: the top ten by GW1 xPts average 0.827,
    # giving 0.173. Rounded down slightly because that frame carries pre-season
    # uncertainty and a nailed-on captain mid-season blanks less often.
    #
    # Static, with the same caveat as the bench slot weights: it does not
    # tighten as a squad becomes more predictable. Small enough that it breaks
    # ties without letting vice-captaincy bid for squad places.
    vice_captain_weight: float = 0.15

    # Per-gameweek discount applied to the OBJECTIVE over a multi-gameweek
    # horizon (2026-08-18). The i-th gameweek ahead is weighted `decay ** i`,
    # so at 0.85 a five-gameweek horizon runs 1.00 / 0.85 / 0.72 / 0.61 / 0.52.
    #
    # Summing a horizon with equal weight — what this engine did until today —
    # claims a projection five weeks out is worth as much as one for the match
    # about to kick off. On the live GW1 frame that was measurably false: 22%
    # of the squad's projected points came from gameweeks bookmakers had
    # priced, 78% from the strength model with 17 of 20 teams still on
    # prior-season fallback.
    #
    # Every serious FPL optimiser discounts. Sertalp Çay's
    # `solve_multi_period_fpl` defaults to `decay_base = 0.84`; FPLReview's
    # solvers recommend 0.80-0.95, lower for short-term aggression and higher
    # for long-term planning. 0.85 sits in the middle of both and is the
    # field's rough consensus.
    #
    # NOT calibrated on this project's own backtest — inherited from the
    # field, same convention as the other heuristic constants here, and worth
    # sweeping over the walk-forward gate once real 26/27 gameweeks exist.
    # Set to 1.0 to restore the old equal-weight behaviour exactly.
    gameweek_decay: float = 0.85

    # GWs to look ahead when building the GW1/pre-season initial squad
    # (fixture-difficulty-weighted, not just single-GW xPts) -- a distinct
    # knob from transfer_planning_horizon_gws since cold start is a one-shot
    # squad build with no in-season transfer plan to horizon-limit
    # (2026-08-10, plan/cold-start-lookahead-and-transfer-overrides -- the
    # user's own example: "It is why so many managers still have Haaland
    # despite the price since the fixtures are so good").
    cold_start_lookahead_gws: int = 5

    # form_window_gws removed 2026-08-01 -- confirmed dead, no test coverage.
    # Rolling windows are hardcoded as [3, 5] directly in
    # projection/minutes_model.py, which never read this.

    # Minimum P(starts) threshold — players below this are excluded
    min_start_probability: float = 0.4

    # hit_min_gain_buffer removed 2026-08-01 -- confirmed dead (found while
    # config-threading the simulation engine's persona knobs, never read by
    # optimiser/transfers.py), no test coverage.

    # Ownership-differential weight magnitude (P3-3, plan §5's λ) — was a
    # dead field (read nowhere) until optimiser/scoring.py wired it in.
    # Sign comes from risk_level, not from this value directly: negative
    # risk_level -> penalises differentials (prefer template); positive ->
    # rewards them; risk_level=0 -> 0 (no EO effect at all). This is the
    # MAGNITUDE only.
    max_ownership_differential: float = 0.5

    # use_price_change_signals removed 2026-08-16 (P3.3) -- read NOWHERE, so
    # it advertised a feature that did not exist: there is no price-change
    # modelling in this project at all. The useful half of that idea (what a
    # player actually SELLS for) is now real -- see optimiser/transfers.py::
    # selling_price. If prediction is built later, player_state_snapshots
    # already holds the per-gameweek now_cost history to train on.

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
    #
    # CONSEQUENCE of a ZERO baseline, spelled out 2026-08-18 (engine review
    # §16) and kept here because it is the argument that eventually moved the
    # value off 0.0: at mu=0, with the default risk_level of 0, mu is 0 and
    # lambda is 0 -- so four documented capabilities were dormant, not live:
    #   * optimiser/scoring.py's variance term (score reduces to plain xpts),
    #   * its ownership/differential weighting (multiplier is 1 for everyone),
    #   * optimiser/captaincy.py's scenario-based covariance-aware captaincy,
    #     which short-circuits to mean argmax without touching the DB,
    #   * optimiser/joint_risk.py's covariance-aware SQUAD selection (added
    #     2026-08-20), which short-circuits to the mean-optimal squad without
    #     generating a pool at all.
    #
    # As of 2026-08-25 mu is -0.25, so the variance term and the joint squad
    # selection are LIVE. Lambda is still zero for the real bot (it scales off
    # risk_level, which remains 0), so the ownership weighting stays dormant
    # and covariance-aware captaincy still short-circuits -- those two are
    # exercised only by the persona cohort.
    #
    # The joint objective was calibrated on 2026-08-20 and mu was HELD at 0.0
    # through GW1, then set to -0.25 on 2026-08-25 (see the decision at the
    # foot of this comment). The evidence, recorded honestly because it is
    # weak: scripts/calibrate_risk_constants.py --harness rebuild,
    # results/mu_joint_calibration*.csv. On 2025-26 GW6-38, mu=-0.25 looked
    # strong: +3.91 actual pts/GW over mu=0, paired on identical pools and
    # draws, t=+2.16, 20 wins to 11. It did NOT survive out of sample. On
    # 2024-25 the same mu is worth +0.06 pts/GW (t=+0.03), mu=-0.5 and -1.0
    # are NEGATIVE, and avg_clubs_at_cap moves the WRONG WAY -- negative mu
    # increased concentration there while it decreased it in 2025-26, so even
    # the mechanism failed to replicate. Pooled over both seasons: +1.98
    # pts/GW, t=+1.39. Settling this properly needs a THIRD season, which
    # 2026-27 is now generating.
    #
    # WHY -0.25 ANYWAY (user decision, 2026-08-21, applied 2026-08-25 once the
    # GW1 deadline had passed). Two separate arguments, neither of which is
    # "the t-statistic is convincing":
    #
    #   1. Timing. At GW1 `projection_samples` is empty, so the joint
    #      re-ranker cannot fire at all, while projection/cold_start.py:1063
    #      DOES emit upside/downside -- so a non-zero mu at GW1 would have
    #      driven the per-player term, which the sweep never measured, on a
    #      different scale. From GW2 that inverts: samples exist, the
    #      re-ranker engages, and mu drives exactly what WAS measured. This
    #      is the gameweek the value becomes meaningful, not merely allowed.
    #   2. The alternative is not neutral. At 0.0 the variance term, the
    #      ownership weighting, the covariance-aware captaincy and the joint
    #      squad selection are ALL dormant (see the consequence note above) --
    #      so "wait for more evidence" means running the season on the one
    #      configuration that generates no evidence about any of them.
    #
    # The point estimate is positive in both seasons tested and the pooled
    # effect is +1.98 pts/GW; the honest reading is "probably small and
    # positive, possibly zero", not "probably harmful". Treat this as a live
    # experiment with a cost ceiling, and revisit it against 2026-27's own
    # outcomes once enough gameweeks have been scored -- the persona cohort
    # sweeps risk_level around this baseline, so the season measures it.
    mu_baseline: float = 0.0
    # How many distinct squads the joint re-ranker considers
    # (optimiser/joint_risk.py). Measured at 0.24s per MILP solve on the live
    # GW1 frame, so 200 costs ~48s per gameweek -- affordable for a calibration
    # sweep, since the pool is built once at mu=0 and reused for every
    # candidate mu. This is LIVE cost as of 2026-08-25 (mu=-0.25): the
    # short-circuit in optimise_squad_joint no longer fires, so every real
    # decision pays for the pool. It stays inert only if mu returns to 0.
    joint_rerank_pool_size: int = 200
    # Widened 2026-08-18, together with the switch from variance to upside
    # semi-deviation as the risk term (optimiser/scoring.risk_adjusted_score).
    #
    # Two things were wrong with 0.08. The units changed -- mu now multiplies a
    # quantity in POINTS rather than points-squared -- and the old axis could
    # not move a decision anyway: at risk_level=1.0 it bought 0.08 x 0.8 = 0.06
    # points of preference between Haaland and Gabriel, against a 1.0-point gap
    # in expected return. A persona sweep whose extremes produce near-identical
    # squads cannot teach anything about risk, which is the whole reason the
    # cohort exists.
    #
    # 1.25 is the level at which the most aggressive persona will actually
    # trade a point of expected return for a point of upside -- i.e. genuinely
    # buys the explosive player over the steady one. That is meant to be an
    # EXTREME of the sweep, not a recommendation.
    #
    # This value does not touch the real bot: mu = mu_baseline + risk_level *
    # mu_range, and risk_level defaults to 0, so the live engine sees exactly
    # mu_baseline (-0.25 since 2026-08-25) whatever mu_range is. It sets the
    # width of the cohort's risk axis, nothing else.
    mu_range: float = 1.25

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
    # 2026-08-18: this became a MULTIPLIER on `bench_slot_weights` below
    # rather than a flat per-player weight of its own. 1.0 keeps the derived
    # slot weights as they are, 0.0 makes the solver ignore the bench
    # entirely and spend everything on the XI. Same convention as FPLReview's
    # solver, where the bench weight scales a probability-derived default
    # rather than replacing it.
    #
    # Kept under its old name because it is a persisted column on the
    # simulation manager table and one of the persona sweep axes; a rename
    # would silently reinterpret stored rows. Historic simulation rows hold
    # values on the OLD scale (0.15 meant "15% of every bench player"), so
    # they are not comparable with runs after this date.
    bench_value_weight: float = 1.0

    # Per-SLOT bench weights, replacing the flat one above (2026-08-18).
    #
    # A bench is an ordered queue, not a set. FPL's automatic substitutions
    # promote the first eligible bench player when a starter does not appear,
    # so bench slot 1 is worth whatever P(at least one starter blanks) is, and
    # slot 3 is worth almost nothing. Weighting all four equally at 0.15 is
    # wrong in both directions at once: it underpays the slot that actually
    # gets used and overpays the two that do not.
    #
    # Derived from this engine's own start probabilities on the live GW1 XI
    # (mean P(start) 0.93): P(>=1 outfield starter misses) = 0.53,
    # P(>=2) = 0.15, P(>=3) = 0.03. The Alan Turing Institute's AIrsenal
    # reaches the same shape with hand-tuned constants
    # (DEFAULT_SUB_WEIGHTS = {"GK": 0.03, "Outfield": (0.65, 0.3, 0.1)}),
    # which is independent corroboration of the ordering and rough magnitude.
    #
    # These are static, so they do not tighten as the squad becomes more
    # nailed-on. Deriving them per-solve from the chosen XI's minutes is what
    # FPLReview's solver does and is the better answer; it is circular (the
    # weights depend on the XI being chosen) and wants a fixed-point pass,
    # which is not worth doing three days before a deadline.
    bench_slot_weights: tuple[float, float, float] = (0.53, 0.15, 0.03)

    # The reserve keeper plays only if the first-choice keeper does not, and
    # unlike outfield slots there is no queue to inherit from: one keeper,
    # one chance. AIrsenal uses 0.03; this engine's own GK start probability
    # implies much the same.
    bench_gk_weight: float = 0.03


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
# PRIOR LEAGUE RULES (P11) — cross-league translation factors for cold-start
# prior tier. Controls how much non-PL league data translates to PL projections.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorLeagueRules:
    """P11: cross-league translation factors + variance for the cold-start
    prior-league prior tier (plan/p11-prior-league-cold-start.md). Defaults
    are the plan's literature-style guess (Championship discounted, top-5
    treated as roughly PL-equivalent) -- replace with
    scripts/calibrate_prior_league_factors.py's real output once the
    historical hold-out has actually been scraped (needs a browser)."""

    # Top-5 factors were 1.0 -- i.e. a claim that a Ligue 1 goal is worth
    # exactly a Premier League goal. Almost nobody believes that, and the
    # direction of the error is known: it over-projects foreign signings, who
    # are precisely the cold-start players entering the GW1 squad.
    #
    # Revised 2026-08-18 from a transfer-based league-strength study (Elhabr,
    # regressing the change in players' z-scored VAEP/90 as they move between
    # leagues). Its coefficients, rescaled so the Premier League is 1.00:
    #
    #     La Liga 0.89   Ligue 1 0.81   Serie A 0.76   Bundesliga 0.67
    #
    # Deliberately NOT used raw. That study measures z-scored VAEP, a
    # whole-game action-value metric, while this factor multiplies npxG90/xA90
    # -- attacking output. Applying its numbers literally would claim a
    # precision the metric mismatch does not support, so they are compressed
    # toward 1.0: the ordering is trusted, the magnitude is halved.
    #
    # Still a prior, not a measurement. A direct calibration was attempted and
    # rejected for survivorship bias (it returned a La Liga factor of 2.11x,
    # because only successful imports are observed). The honest position is
    # that some discount is much more likely right than none.
    translation_factor_championship: float = 0.65
    translation_factor_la_liga: float = 0.95
    translation_factor_serie_a: float = 0.88
    translation_factor_bundesliga: float = 0.84
    translation_factor_ligue_1: float = 0.90

    # Deliberately unremarkable variance guess (mirrors cold_start.py's own
    # _FALLBACK_VAR reasoning) until a real hold-out replaces it.
    translation_variance_championship: float = 6.0
    translation_variance_la_liga: float = 6.0
    translation_variance_serie_a: float = 6.0
    translation_variance_bundesliga: float = 6.0
    translation_variance_ligue_1: float = 6.0

    def translation_factor(self, league: str) -> float:
        return {
            "ENG-Championship": self.translation_factor_championship,
            "ESP-La Liga": self.translation_factor_la_liga,
            "ITA-Serie A": self.translation_factor_serie_a,
            "GER-Bundesliga": self.translation_factor_bundesliga,
            "FRA-Ligue 1": self.translation_factor_ligue_1,
        }[league]

    def translation_variance(self, league: str) -> float:
        return {
            "ENG-Championship": self.translation_variance_championship,
            "ESP-La Liga": self.translation_variance_la_liga,
            "ITA-Serie A": self.translation_variance_serie_a,
            "GER-Bundesliga": self.translation_variance_bundesliga,
            "FRA-Ligue 1": self.translation_variance_ligue_1,
        }[league]


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
PRIOR_LEAGUE = PriorLeagueRules()


def assert_horizons_consistent(
    optimiser: OptimiserConfig | None = None,
    chip_timing: ChipTimingThresholds | None = None,
) -> None:
    """Every consumer SLICES the persisted projection frame rather than
    building its own, so a consumer asking for more gameweeks than the
    pipeline persists silently gets fewer -- and then compares the result
    against a threshold calibrated for the longer window. That is exactly
    how the wildcard came to evaluate a 3-gameweek gain against a
    5-gameweek bar. Raises rather than warns: a silently short horizon is
    indistinguishable from a genuine decision not to act.
    """
    cfg = optimiser or OPTIMISER
    timing = chip_timing or CHIP_TIMING
    consumers = {
        "transfer_planning_horizon_gws": cfg.transfer_planning_horizon_gws,
        "wildcard_eval_horizon_gws": timing.wildcard_eval_horizon_gws,
        "chip_comparison_horizon_gws": timing.chip_comparison_horizon_gws,
    }
    too_long = {name: gws for name, gws in consumers.items() if gws > cfg.projection_horizon_gws}
    if too_long:
        raise ValueError(
            f"projection_horizon_gws={cfg.projection_horizon_gws} is shorter than "
            f"{too_long} — those consumers would silently see fewer gameweeks "
            f"than their thresholds assume"
        )


assert_horizons_consistent()
