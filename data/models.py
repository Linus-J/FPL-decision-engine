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
    fpl_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    # FPL's cross-season-stable player code (fpl_id/element is reassigned each
    # season). Join key for multi-season backfill (Phase-1 finding M3).
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
    odds: Mapped["FixtureOdds | None"] = relationship("FixtureOdds", back_populates="fixture", uselist=False)


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
    __table_args__ = (UniqueConstraint("player_id", "gameweek", name="uq_player_gw"),)

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

    player: Mapped["Player"] = relationship("Player", back_populates="gw_stats")


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


class FixtureOdds(Base):
    __tablename__ = "fixture_odds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(Integer, ForeignKey("fixtures.id"), unique=True, nullable=False)
    home_win_prob: Mapped[float] = mapped_column(Float, default=0.0)
    draw_prob: Mapped[float] = mapped_column(Float, default=0.0)
    away_win_prob: Mapped[float] = mapped_column(Float, default=0.0)
    btts_prob: Mapped[float] = mapped_column(Float, default=0.0)
    home_cs_prob: Mapped[float] = mapped_column(Float, default=0.0)
    away_cs_prob: Mapped[float] = mapped_column(Float, default=0.0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    fixture: Mapped["Fixture"] = relationship("Fixture", back_populates="odds")


class PlayerProjection(Base):
    __tablename__ = "player_projections"
    __table_args__ = (UniqueConstraint("player_id", "gameweek", "created_at", name="uq_projection"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"), nullable=False)
    gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    xpts: Mapped[float] = mapped_column(Float, default=0.0)
    start_probability: Mapped[float] = mapped_column(Float, default=0.0)
    cs_probability: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    player: Mapped["Player"] = relationship("Player", back_populates="projections")


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
