import numpy as np
import pandas as pd
from sqlalchemy import text

from data.db import get_session

# What identifies ONE fixture for one player. ``opponent_team_id`` is the part
# that is easy to forget and expensive to omit: in a DOUBLE gameweek a player
# has two rows under the same (player, gameweek, season), against different
# opponents, at different venues. Keying the per-fixture features on the first
# three columns alone silently collapses the pair -- one fixture's opponent
# strength, home/away flag and clean-sheet odds get copied onto both, so half
# of every double gameweek is fed the wrong fixture.
#
# Measured on 2026-08-17 before this was keyed properly: 2,716 rows carrying
# the wrong opponent defence strength, 1,481 the wrong home/away flag, and
# 2,751 the wrong clean-sheet probabilities. Double gameweeks are precisely
# where chip and transfer planning is decided, so these are the rows the model
# can least afford to be wrong about.
_FIXTURE_KEYS = ["player_id", "gameweek", "season", "opponent_team_id"]

# Real published team strengths sit around 900-1400. FPL serves PLACEHOLDERS
# until a season starts -- attack/defence 0, overall on its 1-5 tier -- and
# those are stored as-is, because 0 is this project's established "not
# published" signal and cold_start.load_current_defence_strength depends on it
# to fall back to prior-season strengths.
#
# So the honouring of that convention has to happen HERE, at the read. Without
# it, 2026-27 rows would carry opp_defence_strength=0 and own_overall_strength=3
# against 975-1390 in every training season, and add_fdr_features' fillna could
# not help: NaN is missing, 0 is present. Below the floor is treated as NULL so
# the existing median/1200 fallback takes over.
_PLAUSIBLE_STRENGTH_FLOOR = 100


def _usable(expr: str) -> str:
    """SQL: the strength expression, or NULL when it is a placeholder.

    Must wrap EVERY read of a raw strength column, including the ones inside
    the lookahead subqueries below. That was missed when this was introduced
    (fixed 2026-08-18, engine review §1): the six direct columns were guarded
    and the two ``next_3gw_avg_*`` aggregates were not, so they averaged raw
    placeholders. ``AVG(0,0,0)`` is 0, not NULL, so their ``COALESCE(..., 1200)``
    never fired and both features read 0 for an entire pre-publication season.
    Wrapping the aggregate's INPUT is what makes the COALESCE reachable: AVG
    skips NULLs, and returns NULL when they are all NULL.

    See ``test_team_fdr.py::test_lookahead_strengths_are_not_zero_when_fpl_
    serves_placeholders`` — the bug is invisible in any season with real
    strengths, so that test is the guard, not this comment.
    """
    return f"CASE WHEN ({expr}) >= {_PLAUSIBLE_STRENGTH_FLOOR} THEN ({expr}) END"


def _merge_on_fixture(
    df: pd.DataFrame, right: pd.DataFrame, value_cols: list[str]
) -> pd.DataFrame:
    """Left-merge per-fixture features, keyed so a double gameweek's two
    fixtures stay distinct. Both frames come from ``player_gw_stats``, so the
    key is always present; a KeyError here means a caller has dropped it and
    would otherwise get the silent collapse described above."""
    missing = [k for k in _FIXTURE_KEYS if k not in df.columns or k not in right.columns]
    if missing:
        raise KeyError(
            f"per-fixture merge is missing {missing}; without it a double "
            f"gameweek's two fixtures collapse into one and half of them are "
            f"fed the wrong opponent"
        )
    right = right.loc[:, [*_FIXTURE_KEYS, *value_cols]].drop_duplicates(
        subset=_FIXTURE_KEYS
    )
    return df.merge(right, on=_FIXTURE_KEYS, how="left")


def load_fixture_difficulty(season: str | None = None) -> pd.DataFrame:
    """Per player-GW FDR from the point-in-time fixture context stored on the
    stat row (team_id_season / opponent_team_id / was_home) and the per-season
    TeamSeasonStrength table (Phase-1 T3b). This is season-accurate — a player's
    club and opponents in a past season, not their current club. Keyed on
    player_gw_stats rows (historical), same as before; missing strengths default
    to 1200 via add_fdr_features."""
    db = get_session()
    try:
        season_filter = "AND s.season = :season" if season else ""
        params = {"season": season} if season else {}
        query = text(f"""
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                s.opponent_team_id,
                s.team_id_season AS team_id,
                CASE WHEN s.was_home THEN 1 ELSE 0 END AS is_home,
                COALESCE(g.is_bgw, 0) AS is_bgw,
                {_usable("CASE WHEN s.was_home THEN tss_opp.strength_defence_away "
                         "ELSE tss_opp.strength_defence_home END")} AS opp_defence_strength,
                {_usable("CASE WHEN s.was_home THEN tss_opp.strength_attack_away "
                         "ELSE tss_opp.strength_attack_home END")} AS opp_attack_strength,
                {_usable("CASE WHEN s.was_home THEN tss_own.strength_attack_home "
                         "ELSE tss_own.strength_attack_away END")} AS own_attack_strength,
                {_usable("CASE WHEN s.was_home THEN tss_own.strength_defence_home "
                         "ELSE tss_own.strength_defence_away END")} AS own_defence_strength,
                {_usable("CASE WHEN s.was_home THEN tss_own.strength_overall_home "
                         "ELSE tss_own.strength_overall_away END")} AS own_overall_strength,
                COALESCE((
                    SELECT AVG({_usable("CASE WHEN s2.was_home THEN t2.strength_defence_away "
                                        "ELSE t2.strength_defence_home END")})
                    FROM player_gw_stats s2
                    JOIN team_season_strength t2
                        ON t2.season = s2.season AND t2.team_id = s2.opponent_team_id
                    WHERE s2.player_id = s.player_id AND s2.season = s.season
                        AND s2.gameweek > s.gameweek AND s2.gameweek <= s.gameweek + 3
                ), 1200) AS next_3gw_avg_opp_defence,
                COALESCE((
                    SELECT AVG({_usable("CASE WHEN s2.was_home THEN t2.strength_attack_away "
                                        "ELSE t2.strength_attack_home END")})
                    FROM player_gw_stats s2
                    JOIN team_season_strength t2
                        ON t2.season = s2.season AND t2.team_id = s2.opponent_team_id
                    WHERE s2.player_id = s.player_id AND s2.season = s.season
                        AND s2.gameweek > s.gameweek AND s2.gameweek <= s.gameweek + 3
                ), 1200) AS next_3gw_avg_opp_attack
            FROM player_gw_stats s
            LEFT JOIN team_season_strength tss_own
                ON tss_own.season = s.season AND tss_own.team_id = s.team_id_season
            LEFT JOIN team_season_strength tss_opp
                ON tss_opp.season = s.season AND tss_opp.team_id = s.opponent_team_id
            LEFT JOIN gameweeks g ON g.id = s.gameweek AND g.season = s.season
            WHERE 1 = 1 {season_filter}
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def add_fdr_features(df: pd.DataFrame, fdr: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        c for c in (
            "is_home",
            "is_bgw",
            "opp_defence_strength",
            "opp_attack_strength",
            "own_attack_strength",
            "own_defence_strength",
            "own_overall_strength",
            "next_3gw_avg_opp_defence",
            "next_3gw_avg_opp_attack",
        ) if c in fdr.columns
    ]
    merged = _merge_on_fixture(df, fdr, value_cols)

    flag_cols = {"is_home", "is_bgw"}
    strength_cols = {
        "opp_defence_strength", "opp_attack_strength", "own_attack_strength",
        "own_defence_strength", "own_overall_strength",
        "next_3gw_avg_opp_defence", "next_3gw_avg_opp_attack",
    }
    for col in flag_cols | strength_cols:
        default = 0.0 if col in flag_cols else 1200.0
        if col not in merged.columns:
            merged[col] = default
        else:
            fill = 0.0 if col in flag_cols else merged[col].median()
            # median() is NaN when every row is NaN (e.g. no historical
            # fixtures backfilled yet) — fall back to the constant default.
            if pd.isna(fill):
                fill = default
            merged[col] = merged[col].fillna(fill)

    merged["attack_vs_defence"] = (
        merged["own_attack_strength"] / merged["opp_defence_strength"].clip(lower=1)
    )
    merged["defence_vs_attack"] = (
        merged["own_defence_strength"] / merged["opp_attack_strength"].clip(lower=1)
    )

    return merged


def load_player_enrichment(season: str | None = None) -> pd.DataFrame:
    """Per (player_id, gameweek, season) enrichment, point-in-time.

    The dynamic fields (transfers/ownership → price_momentum, transfer_velocity)
    are read from the snapshot as-of the gameweek deadline, NOT the mutable
    players.* row broadcast onto every historical row (Phase-1 leak L3). The
    set-piece role fields are per-(player, season) and safe. injury_severity is
    not point-in-time-recoverable historically → 0.
    """
    db = get_session()
    try:
        season_filter = "AND s.season = :season" if season else ""
        params = {"season": season} if season else {}
        query = text(f"""
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                COALESCE(sp.is_penalty_taker, 0) AS is_penalty_taker,
                COALESCE(sp.penalty_xg_per_game, 0.0) AS penalty_xg_per_game,
                COALESCE(sp.is_set_piece_taker, 0) AS is_set_piece_taker,
                COALESCE(sp.key_passes_per_game, 0.0) AS key_passes_per_game,
                0 AS injury_severity,
                COALESCE(ps.transfers_in_event, 0) AS transfers_in_event,
                COALESCE(ps.transfers_out_event, 0) AS transfers_out_event,
                COALESCE(ps.selected_by_percent, 0.0) AS selected_by_percent_enrich
            FROM player_gw_stats s
            JOIN gameweeks g ON g.id = s.gameweek AND g.season = s.season
            LEFT JOIN player_state_snapshots ps ON ps.id = (
                SELECT ps2.id FROM player_state_snapshots ps2
                WHERE ps2.player_id = s.player_id
                    AND ps2.season = s.season
                    AND ps2.snapshot_ts < g.deadline_time
                ORDER BY ps2.snapshot_ts DESC LIMIT 1
            )
            LEFT JOIN player_setpiece_roles sp
                ON sp.player_id = s.player_id AND sp.season = s.season
            WHERE 1 = 1 {season_filter}
        """)
        df = pd.read_sql(query, db.bind, params=params)
        df["price_momentum"] = (
            (df["transfers_in_event"] - df["transfers_out_event"])
            / (df["selected_by_percent_enrich"].clip(lower=0.1) * 1000)
        ).clip(-1.0, 1.0)
        df["transfer_velocity"] = (
            df["transfers_in_event"] / (df["selected_by_percent_enrich"].clip(lower=0.1) * 1000)
        ).clip(0.0, 1.0)
        return df
    finally:
        db.close()


def _merge_asof_within_season(
    df: pd.DataFrame, enr: pd.DataFrame, available: list[str]
) -> pd.DataFrame:
    """Exact (player, gameweek, season) match where one exists, otherwise that
    player's most recent EARLIER gameweek in the same season.

    Why the fallback exists (2026-09-11). ``load_player_enrichment`` selects
    ``FROM player_gw_stats``, so it has no row for a gameweek that has not been
    played yet. The row the minutes model actually serves from is exactly that
    row -- ``_append_asof_rows`` puts the as-of row at ``last_played + 1`` so
    the rolling features summarise every completed gameweek. A plain equality
    merge therefore missed for every served player and the ``fillna(0.0)``
    below turned the miss into a confident zero.

    Measured on the live 2026-27 database at GW4: all 615 served players had
    ``is_set_piece_taker = 0`` and ``price_momentum = 0``, where the training
    rows had 21% set-piece takers and signed, non-zero momentum. Summed served
    P(60+) was 155.2 against 210.3 realised starters per gameweek, and the
    league-wide maximum was 0.811 -- the model could not say that ANY player
    was likely to start. With both blocks carried forward the sum is 218.4 and
    the maximum 0.972. The model itself was never the problem: scored on
    historical rows, where the merge hits, it is well calibrated
    (out-of-sample 2025-26 mean prediction 0.273 against a 0.263 base rate,
    maximum 0.975).

    The two blocks are not equally exact. The set-piece fields come from
    ``player_setpiece_roles``, which is keyed per (player, season) -- carrying
    them forward reads the correct value, and the gameweek in the merge key
    was spurious for them all along. ``price_momentum``/``transfer_velocity``
    are genuinely point-in-time, so the carried value is a gameweek stale;
    last week's transfer momentum is a far better estimate of this week's than
    zero is, but it is an estimate. Fresher snapshots do exist (the live table
    is written daily), and reading them as-of the target deadline would be the
    exact fix -- it is not done here because the carried value already closes
    the whole measured gap.

    ``direction="backward"`` is what keeps this honest: the fallback can only
    ever see gameweeks at or before the one being asked about, so it cannot
    leak an outcome into a backtest. Historical rows have their own exact
    match and are unaffected; this changes the served row only.
    """
    if enr.empty:
        # merge_asof cannot type-check its keys against an empty frame, and
        # there is nothing to carry forward anyway. The caller's fillna gives
        # the same answer a miss on every row would have.
        return df.merge(enr, on=["player_id", "gameweek", "season"], how="left")

    order_key = "_enrich_order"
    left = df.copy()
    left[order_key] = np.arange(len(left))
    right = enr.copy()
    # merge_asof is strict about key dtypes on both the ``on`` and ``by``
    # columns, and the two frames reach here from different readers.
    for frame in (left, right):
        frame["gameweek"] = frame["gameweek"].astype("int64")
        frame["player_id"] = frame["player_id"].astype("int64")
        frame["season"] = frame["season"].astype(str)
    merged = pd.merge_asof(
        left.sort_values("gameweek"),
        right.sort_values("gameweek"),
        on="gameweek",
        by=["player_id", "season"],
        direction="backward",
        allow_exact_matches=True,
    )
    return (
        merged.sort_values(order_key)
        .drop(columns=order_key)
        .reset_index(drop=True)
    )


def add_enrichment_features(df: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    dynamic = [
        "is_penalty_taker", "penalty_xg_per_game", "is_set_piece_taker",
        "key_passes_per_game", "injury_severity",
        "price_momentum", "transfer_velocity",
    ]
    keys = ["player_id", "gameweek", "season"]
    if all(k in enrichment.columns for k in keys) and {"gameweek", "season"}.issubset(df.columns):
        available = [c for c in dynamic if c in enrichment.columns]
        enr = enrichment[keys + available].drop_duplicates(subset=keys)
        merged = _merge_asof_within_season(df, enr, available)
    else:
        # Fallback (single-row predict without gameweek/season): take each
        # player's latest enrichment row and broadcast on player_id.
        available = [c for c in dynamic if c in enrichment.columns]
        enr = enrichment.sort_values("gameweek").drop_duplicates(
            subset=["player_id"], keep="last"
        ) if "gameweek" in enrichment.columns else enrichment
        merged = df.merge(enr[["player_id", *available]], on="player_id", how="left")
    for col in available:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
    return merged


def load_fixture_odds(season: str | None = None) -> pd.DataFrame:
    """Per player-GW odds features (my/opp clean-sheet + over-2.5) for BOTH the
    historical training path and the live in-season path.

    Sourced from ``historical_fixture_odds`` via the point-in-time context on
    the stat row (``team_id_season``/``opponent_team_id``/``was_home``) — NOT
    the old ``fixtures``-table join, which used the player's *current* club and
    latest odds against historical rows (Phase-1 finding L4). The odds row is
    matched on ``(season, gameweek)`` and the season-correct home/away team pair,
    and only counts when ``fetched_at < deadline(season, gw)`` — proving the
    closing odds were stamped at the deadline, not kickoff (finding C2). Missing
    odds default to 0.2 CS / 0.5 over-2.5 via ``add_odds_features``.

    **Live seasons fall back to ``fixture_odds``.** ``historical_fixture_odds``
    is a football-data.co.uk backfill, so it only ever covers *finished*
    seasons — it ran to 2025-26 while the bot was about to play 2026-27. This
    function is the only odds reader ``minutes_model._build_features`` calls, on
    the training AND the prediction pass alike, so without the fallback every
    2026-27 row would take the ``COALESCE`` defaults: all three odds features
    pinned to a constant for the entire live season, in a model fitted on five
    seasons where they varied. The features would look present and be inert.

    The live leg applies the same as-of discipline as
    ``load_live_odds_asof``: the latest snapshot per fixture stamped at or
    before that gameweek's deadline, never after. Historical wins where both
    exist — it is the settled closing line.
    """
    db = get_session()
    try:
        season_filter = "AND s.season = :season" if season else ""
        params = {"season": season} if season else {}
        query = text(f"""
            WITH live AS (
                SELECT fo.fixture_id, fo.home_cs_prob, fo.away_cs_prob,
                       fo.over25_prob, f.season, f.gameweek,
                       f.team_h_id, f.team_a_id
                FROM fixture_odds fo
                JOIN fixtures f ON f.id = fo.fixture_id
                LEFT JOIN gameweeks lg
                    ON lg.id = f.gameweek AND lg.season = f.season
                WHERE (lg.deadline_time IS NULL OR fo.fetched_at <= lg.deadline_time)
                  AND fo.fetched_at = (
                      SELECT MAX(fo2.fetched_at) FROM fixture_odds fo2
                      WHERE fo2.fixture_id = fo.fixture_id
                        AND (lg.deadline_time IS NULL
                             OR fo2.fetched_at <= lg.deadline_time)
                  )
            )
            SELECT
                s.player_id,
                s.gameweek,
                s.season,
                s.opponent_team_id,
                CASE WHEN s.was_home
                     THEN COALESCE(o.home_cs_prob, l.home_cs_prob, 0.2)
                     ELSE COALESCE(o.away_cs_prob, l.away_cs_prob, 0.2) END AS my_cs_prob,
                CASE WHEN s.was_home
                     THEN COALESCE(o.away_cs_prob, l.away_cs_prob, 0.2)
                     ELSE COALESCE(o.home_cs_prob, l.home_cs_prob, 0.2) END AS opp_cs_prob,
                COALESCE(o.over25_prob, l.over25_prob, 0.5) AS over25_prob
            FROM player_gw_stats s
            LEFT JOIN gameweeks g ON g.id = s.gameweek AND g.season = s.season
            LEFT JOIN historical_fixture_odds o
                ON o.season = s.season AND o.gameweek = s.gameweek
                AND (
                    (s.was_home = 1 AND o.home_team_id = s.team_id_season
                        AND o.away_team_id = s.opponent_team_id)
                    OR (s.was_home = 0 AND o.away_team_id = s.team_id_season
                        AND o.home_team_id = s.opponent_team_id)
                )
                AND (g.deadline_time IS NULL OR o.fetched_at < g.deadline_time)
            LEFT JOIN live l
                ON l.season = s.season AND l.gameweek = s.gameweek
                AND (
                    (s.was_home = 1 AND l.team_h_id = s.team_id_season
                        AND l.team_a_id = s.opponent_team_id)
                    OR (s.was_home = 0 AND l.team_a_id = s.team_id_season
                        AND l.team_h_id = s.opponent_team_id)
                )
            WHERE 1 = 1 {season_filter}
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def load_live_odds_asof(season: str, gameweek: int) -> pd.DataFrame:
    """As-of read of the append-only ``fixture_odds`` for the LIVE projection
    path: the latest snapshot per fixture with ``fetched_at <= deadline`` of the
    target GW (Phase-1 findings L4/C2). Multiple fetches accumulate per fixture;
    this excludes any stamped at/after the deadline. Keyed on ``(season, gw)``.
    Returns one row per fixture with home/away CS + BTTS probabilities."""
    db = get_session()
    try:
        query = text("""
            SELECT
                f.id AS fixture_id,
                f.team_h_id,
                f.team_a_id,
                fo.home_cs_prob,
                fo.away_cs_prob,
                fo.btts_prob
            FROM fixtures f
            JOIN gameweeks g ON g.id = :gw AND g.season = :season
            JOIN fixture_odds fo ON fo.fixture_id = f.id
                AND fo.fetched_at <= g.deadline_time
                AND fo.fetched_at = (
                    SELECT MAX(fo2.fetched_at) FROM fixture_odds fo2
                    WHERE fo2.fixture_id = f.id AND fo2.fetched_at <= g.deadline_time
                )
            WHERE f.season = :season AND f.gameweek = :gw
        """)
        return pd.read_sql(query, db.bind, params={"season": season, "gw": gameweek})
    finally:
        db.close()


def add_odds_features(df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    merged = _merge_on_fixture(df, odds, list(ODDS_FEATURE_COLS))
    merged["my_cs_prob"] = merged["my_cs_prob"].fillna(0.2)
    merged["opp_cs_prob"] = merged["opp_cs_prob"].fillna(0.2)
    merged["over25_prob"] = merged["over25_prob"].fillna(0.5)
    return merged


def load_live_fixture_difficulty(
    fixture_context: pd.DataFrame, season: str
) -> pd.DataFrame:
    """Per-fixture FDR for the LIVE serving horizon (item 5 / A9 follow-up).

    ``load_fixture_difficulty`` above is keyed on ``player_gw_stats``, which
    has no row for a fixture that has not been played yet — so it cannot
    serve GW5-GW8 of a live horizon. This reads ``team_season_strength``
    directly against ``fixture_context`` (``player_id``, ``gameweek``,
    ``team_id_season``, ``opponent_team_id``, ``was_home`` — the same shape
    ``_build_live_fixture_context`` and the backtest's own later stat rows
    both produce), so it works for any fixture already on the calendar.

    Pass the WHOLE horizon in one call, not one gameweek at a time:
    ``next_3gw_avg_opp_defence``/``_attack`` read a player's LATER rows in
    THIS SAME frame, and a gameweek missing from it cannot feed another
    gameweek's lookahead. Missing entirely (no strengths published, or the
    player is past the frame's last gameweek) is left ``NaN`` — the existing
    ``add_fdr_features`` fallback (median, or 1200 if the whole column is
    NaN) takes over unchanged, so a horizon this cannot fully cover degrades
    exactly like a played fixture whose strengths are unpublished.

    Returns the same 9 base columns ``load_fixture_difficulty`` returns
    (NOT the two ratio columns) — ``add_fdr_features`` computes
    ``attack_vs_defence``/``defence_vs_attack`` itself after merging."""
    value_cols = [
        "is_home", "is_bgw", "opp_defence_strength", "opp_attack_strength",
        "own_attack_strength", "own_defence_strength", "own_overall_strength",
        "next_3gw_avg_opp_defence", "next_3gw_avg_opp_attack",
    ]
    columns = [*_FIXTURE_KEYS, *value_cols]
    if fixture_context.empty:
        return pd.DataFrame(columns=columns)

    db = get_session()
    try:
        strength = pd.read_sql(
            text(
                "SELECT team_id, strength_attack_home, strength_attack_away, "
                "strength_defence_home, strength_defence_away, "
                "strength_overall_home, strength_overall_away "
                "FROM team_season_strength WHERE season = :season"
            ),
            db.bind, params={"season": season},
        )
        bgw = pd.read_sql(
            text("SELECT id AS gameweek, is_bgw FROM gameweeks WHERE season = :season"),
            db.bind, params={"season": season},
        )
    finally:
        db.close()

    out = fixture_context[
        ["player_id", "gameweek", "team_id_season", "opponent_team_id", "was_home"]
    ].copy()
    out["season"] = season
    out["was_home"] = out["was_home"].astype(bool)
    out["is_home"] = out["was_home"].astype(int)

    own = strength.rename(columns={"team_id": "team_id_season"})
    opp = strength.rename(columns={"team_id": "opponent_team_id"})
    out = out.merge(own, on="team_id_season", how="left")
    out = out.merge(opp, on="opponent_team_id", how="left", suffixes=("", "_opp"))
    out = out.merge(bgw, on="gameweek", how="left")
    out["is_bgw"] = out["is_bgw"].fillna(0).astype(int)

    out["own_attack_strength"] = out["strength_attack_home"].where(
        out["was_home"], out["strength_attack_away"]
    )
    out["own_defence_strength"] = out["strength_defence_home"].where(
        out["was_home"], out["strength_defence_away"]
    )
    out["own_overall_strength"] = out["strength_overall_home"].where(
        out["was_home"], out["strength_overall_away"]
    )
    out["opp_defence_strength"] = out["strength_defence_away_opp"].where(
        out["was_home"], out["strength_defence_home_opp"]
    )
    out["opp_attack_strength"] = out["strength_attack_away_opp"].where(
        out["was_home"], out["strength_attack_home_opp"]
    )

    # Forward lookahead: THIS player's own later rows in the same frame,
    # strictly after this gameweek and within 3 of it -- mirrors
    # ``load_fixture_difficulty``'s ``s2.gameweek > s.gameweek AND
    # s2.gameweek <= s.gameweek + 3`` self-join exactly, just done in pandas
    # because there is no table to self-join against here.
    lookahead = out[["player_id", "gameweek", "opp_defence_strength", "opp_attack_strength"]]
    fwd_defence = []
    fwd_attack = []
    for pid, gw in out[["player_id", "gameweek"]].itertuples(index=False):
        future = lookahead[
            (lookahead["player_id"] == pid)
            & (lookahead["gameweek"] > gw)
            & (lookahead["gameweek"] <= gw + 3)
        ]
        fwd_defence.append(future["opp_defence_strength"].mean())
        fwd_attack.append(future["opp_attack_strength"].mean())
    out["next_3gw_avg_opp_defence"] = fwd_defence
    out["next_3gw_avg_opp_attack"] = fwd_attack

    return out[columns]


def load_live_fixture_odds(
    fixture_context: pd.DataFrame, match_odds: pd.DataFrame, season: str
) -> pd.DataFrame:
    """Per-fixture clean-sheet odds for the LIVE serving horizon, from the
    same raw de-vigged 1X2 + O/U2.5 frame ``assemble_gw_projections`` already
    takes as ``match_odds`` (``load_match_odds`` / ``_load_live_match_odds`` —
    identical shape on the live and backtest paths). Converts through the
    same ``team_goals_from_odds`` fit the P10 xPts engine uses, so a clean
    sheet here means exactly what it means there: ``exp(-lambda)`` of the
    opposing side's Poisson goal rate.

    A fixture beyond the priced window (odds do not exist for it) is left out
    of the returned frame entirely; ``add_odds_features``'s existing
    fillna(0.2 / 0.2 / 0.5) takes over unchanged, same as it already does for
    the whole horizon today."""
    columns = [*_FIXTURE_KEYS, *ODDS_FEATURE_COLS]
    if fixture_context.empty or match_odds.empty:
        return pd.DataFrame(columns=columns)

    from projection.team_goals import team_goals_from_odds

    odds = match_odds.copy()
    lambdas = odds.apply(
        lambda r: team_goals_from_odds(
            float(r["home_win_prob"]), float(r["draw_prob"]),
            float(r["away_win_prob"]), float(r.get("over25_prob")),
        ),
        axis=1, result_type="expand",
    )
    odds["lambda_home"], odds["lambda_away"] = lambdas[0], lambdas[1]
    odds["home_cs_prob"] = np.exp(-odds["lambda_away"])
    odds["away_cs_prob"] = np.exp(-odds["lambda_home"])

    out = fixture_context[
        ["player_id", "gameweek", "team_id_season", "opponent_team_id", "was_home"]
    ].copy()
    out["was_home"] = out["was_home"].astype(bool)
    home_side = odds.rename(columns={
        "home_team_id": "team_id_season", "away_team_id": "opponent_team_id",
    })
    away_side = odds.rename(columns={
        "away_team_id": "team_id_season", "home_team_id": "opponent_team_id",
    })
    merged = pd.concat([
        out[out["was_home"]].merge(
            home_side[["gameweek", "team_id_season", "opponent_team_id",
                       "home_cs_prob", "away_cs_prob", "over25_prob"]],
            on=["gameweek", "team_id_season", "opponent_team_id"], how="inner",
        ).rename(columns={"home_cs_prob": "my_cs_prob", "away_cs_prob": "opp_cs_prob"}),
        out[~out["was_home"]].merge(
            away_side[["gameweek", "team_id_season", "opponent_team_id",
                       "home_cs_prob", "away_cs_prob", "over25_prob"]],
            on=["gameweek", "team_id_season", "opponent_team_id"], how="inner",
        ).rename(columns={"away_cs_prob": "my_cs_prob", "home_cs_prob": "opp_cs_prob"}),
    ], ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["season"] = season
    return merged[columns]


def load_midweek_flags(target_gw: int) -> dict[int, float]:
    from data.ingestors.midweek import get_team_midweek_flags
    flags = get_team_midweek_flags(target_gw)
    return {team_id: float(v) for team_id, v in flags.items()}


FDR_FEATURE_COLS = [
    "is_home",
    "is_bgw",
    "opp_defence_strength",
    "opp_attack_strength",
    "own_attack_strength",
    "own_defence_strength",
    "own_overall_strength",
    "next_3gw_avg_opp_defence",
    "next_3gw_avg_opp_attack",
    "attack_vs_defence",
    "defence_vs_attack",
]

# ``btts_prob`` was here until 2026-08-17 and was never a real feature. The
# training table stored the over-2.5 probability under that name -- a documented
# proxy, and byte-for-byte identical on all 1900 rows -- while the live API
# rejects the BTTS market outright (HTTP 422), leaving the column NULL and the
# served value pinned at the 0.5 default. So the model fitted a coefficient
# against a variable spanning 0.341-0.81 and applied it to a constant. Using
# ``over25_prob`` directly keeps the identical signal, is populated on both
# sides, and stops the column claiming to be something it never was.
ODDS_FEATURE_COLS = [
    "my_cs_prob",
    "opp_cs_prob",
    "over25_prob",
]

ENRICHMENT_FEATURE_COLS = [
    "is_penalty_taker",
    "penalty_xg_per_game",
    "is_set_piece_taker",
    "key_passes_per_game",
    "injury_severity",
    "price_momentum",
    "transfer_velocity",
]


# ---------------------------------------------------------------------------
# Rate-feature contract (Phase-2 P2, defect D4)
#
# Model features must be RATES, not season-cumulative VOLUME, and must be
# computed identically on the train (backtest) and serve (live pipeline) paths.
# The rolling `avg_*_{n}gw` features already satisfy this: per-GW quantities
# averaged over a shift(1) window (as-of, leakage-free), from data present on
# both paths.
#
# The columns below are BANNED as model features:
#   - ict_index/influence/creativity/threat: stored only as season-CUMULATIVE
#     (snapshot) — volume, not rate, and read from different sources on the two
#     paths (snapshot as-of vs the mutable players row) → train/serve skew.
#   - form: the T3 prior-window proxy, not reconcilable across paths; the
#     rolling avg_pts_*gw features carry recent-form signal as a clean rate.
#
# A true ICT-rate is deferred: it needs either per-GW ICT or cumulative
# exposure (minutes/appearances) stored alongside the cumulative ICT, neither
# of which the spine currently carries. See plan/phase-2-xpts-engine.md P2.
# ---------------------------------------------------------------------------

CUMULATIVE_BANNED_FEATURES = frozenset(
    {"ict_index", "influence", "creativity", "threat", "form"}
)


def assert_rate_only(feature_cols: list[str]) -> None:
    """Guard: fail fast if a banned season-cumulative/proxy column is used as a
    model feature (D4). Called at import by each component model."""
    offenders = CUMULATIVE_BANNED_FEATURES.intersection(feature_cols)
    if offenders:
        raise ValueError(
            f"Cumulative/proxy features are banned as model inputs (D4): "
            f"{sorted(offenders)}. Use the rolling avg_*_{{n}}gw rates instead."
        )
