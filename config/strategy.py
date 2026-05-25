"""
strategy.py — all parameters that are likely to change season-to-season.

When FPL announces 26/27 rule changes, this is the ONLY file you should need
to edit for scoring, chip, and structural rule changes.
"""

from dataclasses import dataclass, field


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
    points_cs_gk: int = 6
    points_cs_def: int = 6
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

    # --- Bonus points ---
    points_bps_first: int = 3
    points_bps_second: int = 2
    points_bps_third: int = 1


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
    max_banked_free_transfers: int = 2   # cap on accumulated FTs
    hit_cost_points: int = -4            # points deducted per extra transfer
    max_hits_per_gw: int = 1            # bot will never exceed this in normal play


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
    # Wildcard: trigger if optimal squad beats current squad by this many
    # pts over a WILDCARD_EVAL_HORIZON_GWS rolling window
    wildcard_pts_gain_threshold: float = 15.0
    wildcard_eval_horizon_gws: int = 5

    # Free Hit: use when free hit squad beats current squad by this many pts
    # in a single GW (useful for blank GWs where your squad has many missing)
    free_hit_single_gw_gain_threshold: float = 12.0

    # Bench Boost: use when bench is projected to score this many pts in a DGW
    bench_boost_min_bench_xpts: float = 20.0

    # Triple Captain: use when best TC candidate scores this many more pts
    # than the standard captain (net gain after accounting for 2x vs 3x)
    triple_captain_min_gain: float = 6.0


# ---------------------------------------------------------------------------
# OPTIMISER BEHAVIOUR
# Controls the projection horizon and risk profile.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimiserConfig:
    # Number of GWs to project ahead for transfer decisions
    transfer_planning_horizon_gws: int = 5

    # Rolling window for recent form (xG, xA, minutes)
    form_window_gws: int = 5

    # Minimum P(starts) threshold — players below this are excluded
    min_start_probability: float = 0.4

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
CHIPS = ChipRules()
TRANSFERS = TransferRules()
SQUAD = SquadRules()
DGW = DGWStrategy()
CHIP_TIMING = ChipTimingThresholds()
OPTIMISER = OptimiserConfig()
