from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    short_name: Mapped[str] = mapped_column(String(3), nullable=False)
    strength_overall_home: Mapped[int] = mapped_column(Integer, default=0)
    strength_overall_away: Mapped[int] = mapped_column(Integer, default=0)
    strength_attack_home: Mapped[int] = mapped_column(Integer, default=0)
    strength_attack_away: Mapped[int] = mapped_column(Integer, default=0)
    strength_defence_home: Mapped[int] = mapped_column(Integer, default=0)
    strength_defence_away: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    players: Mapped[list["Player"]] = relationship("Player", back_populates="team")
    home_fixtures: Mapped[list["Fixture"]] = relationship(
        "Fixture", foreign_keys="Fixture.team_h_id", back_populates="team_h"
    )
    away_fixtures: Mapped[list["Fixture"]] = relationship(
        "Fixture", foreign_keys="Fixture.team_a_id", back_populates="team_a"
    )


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # fpl_id (FPL's per-season `element` id) is NOT unique across seasons — FPL
    # reassigns it yearly, so a departed player and a current one can share one
    # (M3). `code` is the stable cross-season identity and the upsert key.
    fpl_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    code: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    second_name: Mapped[str] = mapped_column(String, nullable=False)
    web_name: Mapped[str] = mapped_column(String, nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    position: Mapped[str] = mapped_column(String(3), nullable=False)  # GKP DEF MID FWD

    now_cost: Mapped[float] = mapped_column(Float, nullable=False)  # in £m (divided by 10)
    cost_change_start: Mapped[float] = mapped_column(Float, default=0.0)

    status: Mapped[str] = mapped_column(String(1), default="a")  # a=available, d=doubtful, i=injured, u=unavailable, s=suspended, n=not in squad
    news: Mapped[str] = mapped_column(String, default="")
    news_added: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    selected_by_percent: Mapped[float] = mapped_column(Float, default=0.0)
    form: Mapped[float] = mapped_column(Float, default=0.0)
    total_points: Mapped[int] = mapped_column(Integer, default=0)
    minutes: Mapped[int] = mapped_column(Integer, default=0)
    goals_scored: Mapped[int] = mapped_column(Integer, default=0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    clean_sheets: Mapped[int] = mapped_column(Integer, default=0)
    goals_conceded: Mapped[int] = mapped_column(Integer, default=0)
    saves: Mapped[int] = mapped_column(Integer, default=0)
    yellow_cards: Mapped[int] = mapped_column(Integer, default=0)
    red_cards: Mapped[int] = mapped_column(Integer, default=0)
    bonus: Mapped[int] = mapped_column(Integer, default=0)
    bps: Mapped[int] = mapped_column(Integer, default=0)
    ict_index: Mapped[float] = mapped_column(Float, default=0.0)
    influence: Mapped[float] = mapped_column(Float, default=0.0)
    creativity: Mapped[float] = mapped_column(Float, default=0.0)
    threat: Mapped[float] = mapped_column(Float, default=0.0)

    chance_of_playing_next_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chance_of_playing_this_round: Mapped[int | None] = mapped_column(Integer, nullable=True)

    transfers_in_event: Mapped[int] = mapped_column(Integer, default=0)
    transfers_out_event: Mapped[int] = mapped_column(Integer, default=0)
    injury_severity: Mapped[int] = mapped_column(Integer, default=0)

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    team: Mapped["Team"] = relationship("Team", back_populates="players")
    gw_stats: Mapped[list["PlayerGameweekStats"]] = relationship("PlayerGameweekStats", back_populates="player")
    xg_stats: Mapped[list["PlayerXGStats"]] = relationship("PlayerXGStats", back_populates="player")
    projections: Mapped[list["PlayerProjection"]] = relationship("PlayerProjection", back_populates="player")


class Fixture(Base):
    __tablename__ = "fixtures"
    # season disambiguates fixtures across seasons; FPL fpl_id repeats yearly
    # (Phase-1 finding M2). Feature JOINs key on (season, gameweek).
    __table_args__ = (
        UniqueConstraint("season", "fpl_id", name="uq_fixture_season_fpl"),
        Index("ix_fixture_season_gameweek", "season", "gameweek"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fpl_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[str] = mapped_column(String(7), nullable=False, default="2026-27")
    gameweek: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team_h_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    team_a_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    team_h_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team_a_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)
    kickoff_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_dgw: Mapped[bool] = mapped_column(Boolean, default=False)

    team_h: Mapped["Team"] = relationship("Team", foreign_keys=[team_h_id], back_populates="home_fixtures")
    team_a: Mapped["Team"] = relationship("Team", foreign_keys=[team_a_id], back_populates="away_fixtures")
    odds: Mapped[list["FixtureOdds"]] = relationship("FixtureOdds", back_populates="fixture")


class Gameweek(Base):
    __tablename__ = "gameweeks"

    # Composite key (season, id): id is the GW number, which repeats every
    # season. Without season, deadline(gw) would resolve to the wrong season's
    # calendar in a historical backtest (Phase-1 finding M1).
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[str] = mapped_column(String(7), primary_key=True, default="2026-27")
    name: Mapped[str] = mapped_column(String, nullable=False)
    deadline_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    is_next: Mapped[bool] = mapped_column(Boolean, default=False)
    average_entry_score: Mapped[int] = mapped_column(Integer, default=0)
    highest_score: Mapped[int] = mapped_column(Integer, default=0)
    is_dgw: Mapped[bool] = mapped_column(Boolean, default=False)
    is_bgw: Mapped[bool] = mapped_column(Boolean, default=False)


class PlayerGameweekStats(Base):
    __tablename__ = "player_gw_stats"
    # season in the key: (player_id, gameweek) repeats every season, so without
    # it multi-season backfill collides and drops rows (Phase-1 finding M5).
    __table_args__ = (
        UniqueConstraint("player_id", "gameweek", "season", name="uq_player_gw_season"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[str] = mapped_column(String(7), nullable=False)  # e.g. "2025-26"

    total_points: Mapped[int] = mapped_column(Integer, default=0)
    minutes: Mapped[int] = mapped_column(Integer, default=0)
    goals_scored: Mapped[int] = mapped_column(Integer, default=0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    clean_sheets: Mapped[int] = mapped_column(Integer, default=0)
    goals_conceded: Mapped[int] = mapped_column(Integer, default=0)
    saves: Mapped[int] = mapped_column(Integer, default=0)
    yellow_cards: Mapped[int] = mapped_column(Integer, default=0)
    red_cards: Mapped[int] = mapped_column(Integer, default=0)
    bonus: Mapped[int] = mapped_column(Integer, default=0)
    bps: Mapped[int] = mapped_column(Integer, default=0)
    selected: Mapped[int] = mapped_column(Integer, default=0)
    transfers_in: Mapped[int] = mapped_column(Integer, default=0)
    transfers_out: Mapped[int] = mapped_column(Integer, default=0)
    value: Mapped[float] = mapped_column(Float, default=0.0)

    # Per-GW fixture context (that season's team ids) — the leakage-safe source
    # of FDR without a cross-season fixtures/teams JOIN (Phase-1 T3b).
    team_id_season: Mapped[int | None] = mapped_column(Integer, nullable=True)
    opponent_team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    was_home: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    player: Mapped["Player"] = relationship("Player", back_populates="gw_stats")


class TeamSeasonStrength(Base):
    """Per-season FPL team strengths (teams change every season). Keyed by
    (season, team_id) where team_id is that season's FPL id — read alongside
    PlayerGameweekStats.team_id_season / opponent_team_id for point-in-time FDR
    (Phase-1 T3b). Avoids making the FK'd `teams` table season-aware."""

    __tablename__ = "team_season_strength"
    __table_args__ = (
        UniqueConstraint("season", "team_id", name="uq_team_season_strength"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strength_overall_home: Mapped[int] = mapped_column(Integer, default=1200)
    strength_overall_away: Mapped[int] = mapped_column(Integer, default=1200)
    strength_attack_home: Mapped[int] = mapped_column(Integer, default=1200)
    strength_attack_away: Mapped[int] = mapped_column(Integer, default=1200)
    strength_defence_home: Mapped[int] = mapped_column(Integer, default=1200)
    strength_defence_away: Mapped[int] = mapped_column(Integer, default=1200)


class PlayerStateSnapshot(Base):
    """Point-in-time snapshot of a player's dynamic FPL attributes.

    Append-only: one row per capture (daily + pre-deadline). Never UPDATE.
    Feature reads take the latest row with ``snapshot_ts < deadline(gw)``, which
    is what kills the v1 backtest leakage (plan §1/§3.1). The mutable ``players``
    table stays for current-state convenience, but this is the source of truth
    for any leakage-free training/backtest read.
    """

    __tablename__ = "player_state_snapshots"
    __table_args__ = (
        UniqueConstraint("player_id", "snapshot_ts", name="uq_player_snapshot"),
        Index("ix_player_state_snapshot_ts", "snapshot_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    snapshot_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    gameweek_context: Mapped[int | None] = mapped_column(Integer, nullable=True)

    now_cost: Mapped[float] = mapped_column(Float, default=0.0)  # £m
    status: Mapped[str] = mapped_column(String(1), default="a")
    chance_of_playing_this_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chance_of_playing_next_round: Mapped[int | None] = mapped_column(Integer, nullable=True)

    selected_by_percent: Mapped[float] = mapped_column(Float, default=0.0)
    form: Mapped[float] = mapped_column(Float, default=0.0)
    ict_index: Mapped[float] = mapped_column(Float, default=0.0)
    influence: Mapped[float] = mapped_column(Float, default=0.0)
    creativity: Mapped[float] = mapped_column(Float, default=0.0)
    threat: Mapped[float] = mapped_column(Float, default=0.0)

    news: Mapped[str] = mapped_column(String, default="")
    news_added: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    transfers_in_event: Mapped[int] = mapped_column(Integer, default=0)
    transfers_out_event: Mapped[int] = mapped_column(Integer, default=0)


class PlayerXGStats(Base):
    __tablename__ = "player_xg_stats"
    __table_args__ = (UniqueConstraint("player_id", "gameweek", "season", name="uq_player_xg_gw"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[str] = mapped_column(String(7), nullable=False)

    xg: Mapped[float] = mapped_column(Float, default=0.0)
    xa: Mapped[float] = mapped_column(Float, default=0.0)
    xgi: Mapped[float] = mapped_column(Float, default=0.0)
    npxg: Mapped[float] = mapped_column(Float, default=0.0)
    shots: Mapped[int] = mapped_column(Integer, default=0)
    key_passes: Mapped[int] = mapped_column(Integer, default=0)

    player: Mapped["Player"] = relationship("Player", back_populates="xg_stats")


class PlayerMatchEvents(Base):
    """Per-player-per-match raw event counts (Opta/FBref-derived) — the input
    to the 26/27 BPS simulator (plan §3.3-3.4 / T5b).

    Grain is one match, NOT one gameweek: bonus is awarded per fixture, so a
    DGW player has two rows (one per ``game_id``). Source-agnostic — the FBref
    adapter (``data/ingestors/fbref.py``) is one writer, but the columns are
    exactly the metrics ``projection.bps_sim`` reads, so recompute needs no
    JOINs. Metrics a provider cannot supply (e.g. FBref lacks Opta big-chances)
    are left 0; that gap is the documented tolerance of the sanity harness.
    """

    __tablename__ = "player_match_events"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "game_id", name="uq_match_events"),
        Index("ix_match_events_season_gw", "season", "gameweek"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    gameweek: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Stable per-match identifier (FBref match id, or synthesised season+gw+teams).
    game_id: Mapped[str] = mapped_column(String, nullable=False)
    position: Mapped[str] = mapped_column(String(3), default="MID")
    source: Mapped[str] = mapped_column(String, default="fbref")

    minutes: Mapped[int] = mapped_column(Integer, default=0)
    goals: Mapped[int] = mapped_column(Integer, default=0)
    winning_goals: Mapped[int] = mapped_column(Integer, default=0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    clean_sheet: Mapped[int] = mapped_column(Integer, default=0)

    # Attacking contribution
    big_chances_created: Mapped[int] = mapped_column(Integer, default=0)
    key_passes: Mapped[int] = mapped_column(Integer, default=0)
    open_play_crosses: Mapped[int] = mapped_column(Integer, default=0)
    dribbles: Mapped[int] = mapped_column(Integer, default=0)

    # Goalkeeping
    saves: Mapped[int] = mapped_column(Integer, default=0)
    saves_in_box: Mapped[int] = mapped_column(Integer, default=0)
    big_chances_saved: Mapped[int] = mapped_column(Integer, default=0)
    penalties_saved: Mapped[int] = mapped_column(Integer, default=0)

    # Defensive actions
    tackles: Mapped[int] = mapped_column(Integer, default=0)
    clearances: Mapped[int] = mapped_column(Integer, default=0)
    blocks: Mapped[int] = mapped_column(Integer, default=0)
    interceptions: Mapped[int] = mapped_column(Integer, default=0)
    recoveries: Mapped[int] = mapped_column(Integer, default=0)

    # Passing accuracy
    passes: Mapped[int] = mapped_column(Integer, default=0)
    pass_completion_pct: Mapped[float] = mapped_column(Float, default=0.0)

    # Negative
    being_tackled: Mapped[int] = mapped_column(Integer, default=0)
    penalties_conceded: Mapped[int] = mapped_column(Integer, default=0)
    penalties_missed: Mapped[int] = mapped_column(Integer, default=0)
    yellow_cards: Mapped[int] = mapped_column(Integer, default=0)
    red_cards: Mapped[int] = mapped_column(Integer, default=0)
    own_goals: Mapped[int] = mapped_column(Integer, default=0)
    big_chances_missed: Mapped[int] = mapped_column(Integer, default=0)
    errors_leading_to_goal: Mapped[int] = mapped_column(Integer, default=0)
    errors_leading_to_shot: Mapped[int] = mapped_column(Integer, default=0)
    fouls: Mapped[int] = mapped_column(Integer, default=0)
    offsides: Mapped[int] = mapped_column(Integer, default=0)
    shots_off_target: Mapped[int] = mapped_column(Integer, default=0)


class RecomputedBonus(Base):
    """Historical bonus re-derived under the 26/27 BPS rules (plan §3.4 / T5b).

    One row per (player, match). ``bps_2627``/``bonus_2627`` are the simulator's
    output over ``player_match_events`` using the current ``BPS_WEIGHTS``; Phase 2
    reads these instead of the as-played ``PlayerGameweekStats.bonus`` so that
    training targets reflect the rules the bot will actually play under.
    """

    __tablename__ = "recomputed_bonus"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "game_id", name="uq_recomputed_bonus"),
        Index("ix_recomputed_bonus_season_gw", "season", "gameweek"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    gameweek: Mapped[int | None] = mapped_column(Integer, nullable=True)
    game_id: Mapped[str] = mapped_column(String, nullable=False)
    bps_2627: Mapped[int] = mapped_column(Integer, default=0)
    bonus_2627: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FixtureOdds(Base):
    """Live odds for a (current-season) fixture. APPEND-ONLY: one row per fetch,
    keyed ``(fixture_id, fetched_at)`` — never UPDATE (Phase-1 finding L4). The
    leakage-free read (``features.load_live_odds_asof``) takes the latest row
    with ``fetched_at <= deadline(season, gw)``. Historical closing odds live in
    ``HistoricalFixtureOdds`` instead (past seasons have no ``fixtures`` rows)."""

    __tablename__ = "fixture_odds"
    __table_args__ = (
        UniqueConstraint("fixture_id", "fetched_at", name="uq_fixture_odds_asof"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(Integer, ForeignKey("fixtures.id"), nullable=False)
    home_win_prob: Mapped[float] = mapped_column(Float, default=0.0)
    draw_prob: Mapped[float] = mapped_column(Float, default=0.0)
    away_win_prob: Mapped[float] = mapped_column(Float, default=0.0)
    over25_prob: Mapped[float] = mapped_column(Float, default=0.0)
    btts_prob: Mapped[float] = mapped_column(Float, default=0.0)
    home_cs_prob: Mapped[float] = mapped_column(Float, default=0.0)
    away_cs_prob: Mapped[float] = mapped_column(Float, default=0.0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    fixture: Mapped["Fixture"] = relationship("Fixture", back_populates="odds")


class HistoricalFixtureOdds(Base):
    """Closing pre-match odds for a past-season fixture (plan T6).

    Season-keyed rather than FK'd to ``fixtures`` — past seasons have no fixture
    rows (same reason T3b keyed FDR off the stat row). Read via the point-in-time
    context on ``player_gw_stats`` (``team_id_season``/``opponent_team_id``/
    ``was_home``). ``fetched_at`` is stamped ``deadline(season, gw) − ε`` (NOT
    kickoff − ε, per finding C2) so the as-of ``< deadline`` filter keeps it."""

    __tablename__ = "historical_fixture_odds"
    __table_args__ = (
        UniqueConstraint(
            "season", "gameweek", "home_team_id", "away_team_id",
            name="uq_hist_odds",
        ),
        Index("ix_hist_odds_season_gw", "season", "gameweek"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    # That season's FPL team ids (match player_gw_stats.team_id_season).
    home_team_id: Mapped[int] = mapped_column(Integer, nullable=False)
    away_team_id: Mapped[int] = mapped_column(Integer, nullable=False)

    home_win_prob: Mapped[float] = mapped_column(Float, default=0.0)
    draw_prob: Mapped[float] = mapped_column(Float, default=0.0)
    away_win_prob: Mapped[float] = mapped_column(Float, default=0.0)
    over25_prob: Mapped[float] = mapped_column(Float, default=0.0)
    btts_prob: Mapped[float] = mapped_column(Float, default=0.0)
    home_cs_prob: Mapped[float] = mapped_column(Float, default=0.0)
    away_cs_prob: Mapped[float] = mapped_column(Float, default=0.0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PlayerProjection(Base):
    __tablename__ = "player_projections"
    __table_args__ = (UniqueConstraint("player_id", "gameweek", "created_at", name="uq_projection"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    # xpts kept as the back-compatible alias of xpts_mean during the Phase-2
    # migration; xpts_mean/xpts_var are the distributional contract (P0), summed
    # from the component Monte-Carlo draws in projection_samples (P10).
    xpts: Mapped[float] = mapped_column(Float, default=0.0)
    xpts_mean: Mapped[float] = mapped_column(Float, default=0.0)
    xpts_var: Mapped[float] = mapped_column(Float, default=0.0)
    start_probability: Mapped[float] = mapped_column(Float, default=0.0)
    cs_probability: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    player: Mapped["Player"] = relationship("Player", back_populates="projections")


class ProjectionSample(Base):
    """One Monte-Carlo draw of a player's GW xPts (Phase-2 P0 output contract;
    populated by the joint sampler in P10). ``scenario_id`` is shared across all
    players drawn in the SAME scenario, so teammate covariance is recoverable
    (P-COV) — the reason samples are stored rather than just mean/var."""

    __tablename__ = "projection_samples"
    __table_args__ = (
        UniqueConstraint(
            "player_id", "gameweek", "season", "scenario_id", "created_at",
            name="uq_projection_sample",
        ),
        Index("ix_projection_sample_run", "created_at", "scenario_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[str] = mapped_column(String(7), default="")
    scenario_id: Mapped[int] = mapped_column(Integer, nullable=False)
    xpts: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OwnershipSnapshot(Base):
    """Effective ownership (P3-2, v2-build-plan §3.2) — feeds the rank-aware
    objective's differential-value term (§5): ``your_pts - EO*field_pts``,
    where the "field" that actually matters for rank is the top-10k, not the
    whole 11M+ player base a casual manager competes against.

    ``overall_selected_pct`` is real (FPL bootstrap-static's
    ``selected_by_percent``, already ingested elsewhere — duplicated here so
    one snapshot row carries the full picture for a given ``snapshot_ts``).
    ``top10k_selected_pct``/``captaincy_pct_top10k`` come from sampling the
    "Overall" classic league (id 314) standings + a sample of those
    managers' picks for the gameweek — real but a SAMPLE, not the true
    population value (the plan's own §3.2 wording: "aggregating a top-10k
    mini-league sample"). ``captaincy_pct_overall`` has no free population-
    wide source (would need sampling all ~11M managers) — left nullable,
    NOT populated by the current ingestor; a documented gap, not silently 0.

    UNVERIFIED AGAINST LIVE DATA as of authoring (2026-07-26): the 2026-27
    season has zero played gameweeks, so the "Overall" league has zero
    ranked entries and the sampling endpoints return empty — this schema
    and its ingestor are built against the well-documented, stable shape of
    FPL's public (undocumented but widely relied-upon) API, not verified
    against a real populated response. Re-verify at GW1.
    """

    __tablename__ = "ownership_snapshots"
    __table_args__ = (
        UniqueConstraint("player_id", "snapshot_ts", name="uq_ownership_snapshot"),
        Index("ix_ownership_snapshot_ts", "snapshot_ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    snapshot_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    overall_selected_pct: Mapped[float] = mapped_column(Float, default=0.0)
    top10k_selected_pct: Mapped[float] = mapped_column(Float, default=0.0)
    captaincy_pct_overall: Mapped[float | None] = mapped_column(Float, nullable=True)
    captaincy_pct_top10k: Mapped[float] = mapped_column(Float, default=0.0)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)


class PlayerSetPieceRole(Base):
    __tablename__ = "player_setpiece_roles"
    __table_args__ = (UniqueConstraint("player_id", "season", name="uq_setpiece_player_season"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    is_penalty_taker: Mapped[bool] = mapped_column(Boolean, default=False)
    penalty_xg_per_game: Mapped[float] = mapped_column(Float, default=0.0)
    is_set_piece_taker: Mapped[bool] = mapped_column(Boolean, default=False)
    key_passes_per_game: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PlayerPressSignal(Base):
    __tablename__ = "player_press_signals"
    __table_args__ = (UniqueConstraint("player_id", "scraped_date", name="uq_press_player_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    scraped_date: Mapped[str] = mapped_column(String(10), nullable=False)
    sentiment: Mapped[float] = mapped_column(Float, default=0.0)
    raw_quote: Mapped[str] = mapped_column(Text, default="")
    source_url: Mapped[str] = mapped_column(String, default="")


class PriorLeagueStats(Base):
    """Prior-season per-90 attacking rates from a non-PL league (FBref season
    stats) — the raw input to P11's cold-start prior for players new to the PL
    (foreign signings + promoted-team players). Keyed by (player_name, team,
    league, season): these players have no PL `code` until P11's identity step
    matches them to an FPL entry (`code` is filled in then). Rates are stored
    already normalised per-90; league-strength translation to PL-equivalent is
    applied downstream (P11), not here."""

    __tablename__ = "prior_league_stats"
    __table_args__ = (
        UniqueConstraint("player_name", "team", "league", "season", name="uq_prior_league"),
        Index("ix_prior_league_season", "league", "season"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_name: Mapped[str] = mapped_column(String, nullable=False)
    team: Mapped[str] = mapped_column(String, default="")
    league: Mapped[str] = mapped_column(String, nullable=False)   # e.g. "ENG-Championship"
    season: Mapped[str] = mapped_column(String(7), nullable=False)
    position: Mapped[str] = mapped_column(String(8), default="")
    # matched FPL identity, filled by P11 (null until then)
    code: Mapped[int | None] = mapped_column(Integer, nullable=True)

    minutes: Mapped[int] = mapped_column(Integer, default=0)
    matches: Mapped[int] = mapped_column(Integer, default=0)
    goals90: Mapped[float] = mapped_column(Float, default=0.0)
    assists90: Mapped[float] = mapped_column(Float, default=0.0)
    npxg90: Mapped[float] = mapped_column(Float, default=0.0)
    xa90: Mapped[float] = mapped_column(Float, default=0.0)
    sca90: Mapped[float] = mapped_column(Float, default=0.0)


class DecisionLog(Base):
    __tablename__ = "decision_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    decision_type: Mapped[str] = mapped_column(String, nullable=False)
    details: Mapped[str] = mapped_column(String, nullable=False)
    projected_gain: Mapped[float] = mapped_column(Float, default=0.0)
    actual_outcome: Mapped[float | None] = mapped_column(Float, nullable=True)
    dry_run: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
