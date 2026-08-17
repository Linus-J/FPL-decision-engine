"""Regression tests for data/quality_checks.py, each tied to a real bug
found during the 2026-07-28 data-completeness audit."""

from __future__ import annotations

from data.quality_checks import (
    check_name_match_coverage,
    check_no_single_teammate_monopoly,
    check_stat_column_not_dead,
    check_team_id_matches_live,
)


def test_name_match_coverage_passes_above_floor():
    assert check_name_match_coverage("understat", 500, 537) == []


def test_name_match_coverage_flags_below_floor():
    issues = check_name_match_coverage("understat", 500, 537, min_coverage=0.95)
    assert len(issues) == 1
    assert issues[0].check == "name_match_coverage"


def test_name_match_coverage_zero_total_is_not_flagged():
    assert check_name_match_coverage("understat", 0, 0) == []


def test_stat_column_not_dead_flags_the_real_fbref_bug():
    # Real case: FBref's "Expected xG" mapping pointed at a column that
    # doesn't exist -- only 14/524 players ever got a nonzero xg.
    issues = check_stat_column_not_dead("xg", nonzero_count=14, eligible_count=524)
    assert len(issues) == 1
    assert issues[0].severity == "error"


def test_stat_column_not_dead_passes_when_populated():
    assert check_stat_column_not_dead("xg", nonzero_count=480, eligible_count=524) == []


def test_team_id_matches_live_flags_genuine_staleness():
    # Real case: Penders/Anselmino found stale relative to live during the
    # audit -- our team_id=7 ("Coventry City"), live says 6 ("Chelsea").
    player_team_ids = {179268: ("Penders", 7)}
    live_player_team = {179268: 6}
    issues = check_team_id_matches_live(player_team_ids, live_player_team)
    assert len(issues) == 1
    assert "Penders" in issues[0].message


def test_team_id_matches_live_ignores_departed_players():
    # A player whose code no longer appears in the live feed at all (left
    # the league) must NOT be flagged -- their frozen team_id is expected.
    player_team_ids = {118748: ("M.Salah", 12)}
    live_player_team: dict[int, int] = {}
    assert check_team_id_matches_live(player_team_ids, live_player_team) == []


def test_no_single_teammate_monopoly_flags_the_gabriel_case():
    # Real case: Gabriel Magalhaes captured ~100% of Arsenal's attacking
    # weight because every real attacker's weight measured exactly 0.0.
    team_weights = {5: 0.98, 18: 0.01, 29: 0.01}
    issues = check_no_single_teammate_monopoly(team_weights)
    assert len(issues) == 1


def test_no_single_teammate_monopoly_allows_genuine_single_signal():
    # Only one teammate has any signal at all -- not a monopoly, just early
    # in the season for everyone else.
    team_weights = {5: 0.4, 18: 0.0, 29: 0.0}
    assert check_no_single_teammate_monopoly(team_weights) == []


def test_no_single_teammate_monopoly_allows_balanced_team():
    team_weights = {5: 0.4, 18: 0.35, 29: 0.25}
    assert check_no_single_teammate_monopoly(team_weights) == []


# --- Checks added 2026-08-16: verifying the DATA, not the code -------------
#
# The rest of the suite proves the code does what it was written to do.
# These ask whether the data it runs on is actually there and whether the
# numbers coming out are plausible — the failure mode where every unit test
# passes and the answer is still wrong.


def test_source_coverage_weights_by_minutes_not_player_count():
    """A source missing thirty fringe players is noise; one missing three
    everpresents is a hole in every projection they appear in. Counting
    names treats those as identical, which is why this exists alongside
    check_name_match_coverage."""
    from data.quality_checks import check_source_coverage

    # 95% of minutes covered, even if many individual players are missing
    assert check_source_coverage("understat", 9500.0, 10000.0) == []
    # 60% of minutes covered is an error however few players it is
    issues = check_source_coverage("understat", 6000.0, 10000.0)
    assert len(issues) == 1
    assert issues[0].severity == "error"


def test_source_coverage_warns_before_it_errors():
    from data.quality_checks import check_source_coverage

    warn = check_source_coverage("events", 8000.0, 10000.0)
    assert warn and warn[0].severity == "warning"


def test_source_coverage_ignores_an_empty_season():
    """A season with no minutes played yet is pre-season, not a data gap."""
    from data.quality_checks import check_source_coverage

    assert check_source_coverage("understat", 0.0, 0.0) == []


def test_projection_sanity_catches_a_scale_error():
    """The exact regression P0.1 fixed: points-per-APPEARANCE used where
    points-per-gameweek was meant. Every value is individually reasonable
    and the arithmetic is faithful — the pool mean is what gives it away."""
    from data.quality_checks import check_projection_sanity

    per_appearance = [5.5] * 100      # plausible per appearance, absurd as a pool mean
    issues = check_projection_sanity("xpts", per_appearance, low=0.5, high=4.0)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "outside the plausible range" in issues[0].message


def test_projection_sanity_catches_a_collapse_to_zero():
    from data.quality_checks import check_projection_sanity

    assert check_projection_sanity("xpts", [0.0] * 100, low=0.5, high=4.0)


def test_projection_sanity_passes_a_realistic_pool():
    """Most of the player pool are fringe, so a realistic per-gameweek mean
    across everyone is low — around 1-2 points, not a starter's 5."""
    from data.quality_checks import check_projection_sanity

    pool = [0.2] * 60 + [1.5] * 25 + [4.0] * 15
    assert check_projection_sanity("xpts", pool, low=0.5, high=4.0) == []


def test_projection_sanity_treats_no_values_as_an_error():
    """An empty projection frame means the pipeline produced nothing, which
    is a failure rather than a benign zero."""
    from data.quality_checks import check_projection_sanity

    issues = check_projection_sanity("xpts", [], low=0.5, high=4.0)
    assert len(issues) == 1 and issues[0].severity == "error"


def test_referential_integrity_flags_orphans():
    """An orphaned row is SILENT — it simply never joins, so that player's
    data vanishes from projections rather than raising."""
    from data.quality_checks import check_referential_integrity

    assert check_referential_integrity("player_xg_stats", 0, 1000) == []
    issues = check_referential_integrity("player_xg_stats", 12, 1000)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "silently never join" in issues[0].message


def test_copied_column_is_flagged_even_though_it_looks_healthy():
    """The failure mode check_stat_column_not_dead cannot see: npxg held
    real, plausible, non-zero values for all 11,306 rows — they were the xg
    values verbatim. Nothing was empty or out of range, so nothing flagged
    it until a decomposition built on the pair turned out to double-count."""
    from data.quality_checks import check_column_is_not_a_copy

    # identical on every row
    issues = check_column_is_not_a_copy("npxg vs xg", 0, 11306)
    assert len(issues) == 1
    assert "double-count" in issues[0].message

    # genuinely different data is fine — even when it differs RARELY. npxg
    # differs from xg on 0.79% of player-matches because that is how often a
    # penalty is taken; an earlier 1% threshold flagged that correct data.
    assert check_column_is_not_a_copy("npxg vs xg", 89, 11306) == []
    assert check_column_is_not_a_copy("npxg vs xg", 4000, 11306) == []


def test_copied_column_check_accepts_a_caller_supplied_expectation():
    """A caller that DOES know the expected rate can be stricter; the generic
    default cannot guess it."""
    from data.quality_checks import check_column_is_not_a_copy

    assert check_column_is_not_a_copy(
        "a vs b", 89, 11306, min_distinct_fraction=0.05
    )


def test_copied_column_check_ignores_an_empty_table():
    from data.quality_checks import check_column_is_not_a_copy

    assert check_column_is_not_a_copy("npxg vs xg", 0, 0) == []


# --- odds feature liveness (2026-08-17) -------------------------------------
def _patch_odds_gate(monkeypatch, frame, *, played=True):
    """Point the gate's two imports at fixtures. It imports inside the
    function, so patch the source modules rather than the gate's namespace."""
    import projection.features as feats
    import projection.pipeline as pipe

    monkeypatch.setattr(pipe, "season_has_played_history", lambda season: played)
    monkeypatch.setattr(feats, "load_fixture_odds", lambda season=None: frame)


def test_odds_liveness_fires_when_a_feature_is_stuck_on_its_default(monkeypatch):
    """The whole point of the check. A season whose live odds never arrive
    still yields a full, plausible-looking frame -- every row on the COALESCE
    default -- so only a constancy test can tell the difference."""
    import pandas as pd

    from scripts.data_quality_gate import run_odds_feature_liveness_check

    dead = pd.DataFrame({
        "my_cs_prob": [0.2] * 50,
        "opp_cs_prob": [0.2] * 50,
        "over25_prob": [0.5] * 50,
    })
    _patch_odds_gate(monkeypatch, dead)

    issues = run_odds_feature_liveness_check("2026-27")
    assert len(issues) == 3, "all three defaulted columns should be flagged"
    assert all(i.severity == "error" for i in issues)
    assert "inert for the whole season" in issues[0].message


def test_odds_liveness_passes_on_real_variation(monkeypatch):
    import pandas as pd

    from scripts.data_quality_gate import run_odds_feature_liveness_check

    live = pd.DataFrame({
        "my_cs_prob": [0.41, 0.12, 0.33],
        "opp_cs_prob": [0.12, 0.41, 0.28],
        "over25_prob": [0.63, 0.63, 0.51],
    })
    _patch_odds_gate(monkeypatch, live)
    assert run_odds_feature_liveness_check("2026-27") == []


def test_odds_liveness_is_silent_pre_season(monkeypatch):
    """Pre-season there are no player-gameweek rows and the cold start does
    not read these features; flagging then would cry wolf every week."""
    import pandas as pd

    from scripts.data_quality_gate import run_odds_feature_liveness_check

    dead = pd.DataFrame({
        "my_cs_prob": [0.2] * 5,
        "opp_cs_prob": [0.2] * 5,
        "over25_prob": [0.5] * 5,
    })
    _patch_odds_gate(monkeypatch, dead, played=False)
    assert run_odds_feature_liveness_check("2026-27") == []


# --- degenerate model features (2026-08-17) ---------------------------------
def test_constant_training_features_are_detected():
    """Six of eight enrichment features are identically zero across all five
    backfilled seasons, because their sources only exist for the season being
    played."""
    import pandas as pd

    from projection.minutes_model import _degenerate_features

    X = pd.DataFrame({
        "real": [1.0, 2.0, 3.0],
        "always_zero": [0.0, 0.0, 0.0],
        "always_one": [1.0, 1.0, 1.0],
    })
    assert _degenerate_features(X) == {"always_zero": 0.0, "always_one": 1.0}


def test_a_constant_feature_cannot_become_a_hidden_season_indicator():
    """The actual risk. A column that is zero for five seasons and non-zero
    only for the season being played is a perfect 'this row is 2026-27' flag;
    the model would attribute current-season effects to a variable carrying
    none. Pinning it to the training constant makes that impossible."""
    import pandas as pd

    from projection.minutes_model import _pin_degenerate

    serve = pd.DataFrame({
        "real": [5.0, 6.0],
        "press_sentiment": [-1.0, 1.0],   # populated live, absent in history
    })
    pinned = _pin_degenerate(serve, {"press_sentiment": 0.0})

    assert list(pinned["press_sentiment"]) == [0.0, 0.0]
    assert list(pinned["real"]) == [5.0, 6.0], "real features must be untouched"
    assert list(serve["press_sentiment"]) == [-1.0, 1.0], "must not mutate caller"


def test_pinning_is_a_no_op_when_nothing_is_degenerate():
    import pandas as pd

    from projection.minutes_model import _pin_degenerate

    X = pd.DataFrame({"a": [1.0, 2.0]})
    assert _pin_degenerate(X, {}) is X
