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
