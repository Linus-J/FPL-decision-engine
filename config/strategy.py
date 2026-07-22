"""
strategy.py — all parameters that are likely to change season-to-season.

When FPL announces 26/27 rule changes, this is the ONLY file you should need
to edit for scoring, chip, and structural rule changes.
"""

from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# CHIP RULES
# Chips available and how many of each per season.
# 25/26 season had: 1x Wildcard per half, 1x Free Hit, 1x Bench Boost,
# 1x Triple Captain. Update here if 26/27 adds/removes/changes chips.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipRules:
    wildcards_total: int = 2             # total WCs per season
    wildcard_first_half_deadline_gw: int = 19  # GW after which WC1 expires
    free_hits_total: int = 1
    bench_boosts_total: int = 1
    triple_captains_total: int = 1

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

    triple_captain_min_gain: float = 6.0


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

    # Ownership differential cap — how far to deviate from template
    # (higher = more contrarian, lower = safer template-following)
    max_ownership_differential: float = 0.5  # 50pp below overall ownership

    # Whether to factor in price change predictions
    use_price_change_signals: bool = True

    # Risk mode: "safe" | "balanced" | "aggressive"
    # Affects how much variance is tolerated in picks
    risk_mode: str = "balanced"


# ---------------------------------------------------------------------------
# SINGLETON INSTANCES — import these everywhere
# ---------------------------------------------------------------------------

SCORING = ScoringRules()
DEFCON = DefConRules()
BPS_WEIGHTS = BPSWeights()
CHIPS = ChipRules()
TRANSFERS = TransferRules()
SQUAD = SquadRules()
DGW = DGWStrategy()
CHIP_TIMING = ChipTimingThresholds()
OPTIMISER = OptimiserConfig()
