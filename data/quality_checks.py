"""quality_checks.py — reusable, pure data-integrity checks (2026-07-28 audit).

Each check takes already-fetched data (no DB/network calls of its own) and
returns a list of ``QualityIssue`` -- fast to unit-test, and composable by
any ingestion script without duplicating query logic. Encodes the bug
classes actually found during the 2026-07-28 data-completeness audit: a
source column silently defaulting to zero because the real column doesn't
exist (FBref's dead ``Expected xG`` mapping), a name-matcher missing enough
of an external source to matter (season-wide Understat xG gap), a
name-matcher collision letting one player's stats absorb another's
(Gabriel Magalhães/Martinelli/Jesus), and a player's ``team_id`` going
stale relative to the live FPL feed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityIssue:
    check: str
    severity: str  # "warning" | "error"
    message: str


def check_name_match_coverage(
    source: str, matched: int, total: int, *, min_coverage: float = 0.90,
) -> list[QualityIssue]:
    """Flag a name-matching pass against an external source that resolves
    fewer than ``min_coverage`` of names. Run after every Understat/
    WhoScored ingest -- would have caught the season-wide xG gap (only
    14/524 players had nonzero xG) the moment it happened instead of it
    surfacing later as a captaincy-monopoly symptom."""
    if total == 0:
        return []
    coverage = matched / total
    if coverage < min_coverage:
        return [
            QualityIssue(
                check="name_match_coverage",
                severity="error" if coverage < 0.75 else "warning",
                message=(
                    f"{source}: only {matched}/{total} ({coverage:.1%}) names "
                    f"matched to a player -- below the {min_coverage:.0%} floor."
                ),
            )
        ]
    return []


def check_stat_column_not_dead(
    column: str,
    nonzero_count: int,
    eligible_count: int,
    *,
    min_nonzero_fraction: float = 0.05,
) -> list[QualityIssue]:
    """Flag a stat column that's supposed to carry real per-player data but
    comes back zero for almost every eligible row -- the FBref dead-column
    bug class (a mapping pointed at ``"Expected xG"``, a column that simply
    doesn't exist in the real per-match table, silently defaulting every
    row to 0.0 instead of raising)."""
    if eligible_count == 0:
        return []
    fraction = nonzero_count / eligible_count
    if fraction < min_nonzero_fraction:
        return [
            QualityIssue(
                check="stat_column_not_dead",
                severity="error",
                message=(
                    f"{column}: only {nonzero_count}/{eligible_count} "
                    f"({fraction:.1%}) eligible rows are nonzero -- likely a "
                    f"dead/misnamed source column silently defaulting to 0."
                ),
            )
        ]
    return []


def check_team_id_matches_live(
    player_team_ids: dict[int, tuple[str, int]],
    live_player_team: dict[int, int],
) -> list[QualityIssue]:
    """Flag a player who IS present in today's live FPL feed but whose
    stored ``team_id`` disagrees with it right now.

    ``player_team_ids`` maps a player's stable FPL ``code`` to
    ``(web_name, our_team_id)``; ``live_player_team`` maps that same code to
    the CURRENT live team_id, for players still in the live feed. Players
    who have left the league entirely (code absent from
    ``live_player_team``) are correctly excluded rather than flagged --
    their frozen team_id is expected and harmless (see the 2026-07-28 audit:
    FPL reassigns all 20 numeric team ids every season, so a departed
    player's team_id will legitimately no longer match a later live pull).
    """
    issues = []
    for code, (web_name, our_team_id) in player_team_ids.items():
        live_team_id = live_player_team.get(code)
        if live_team_id is not None and live_team_id != our_team_id:
            issues.append(
                QualityIssue(
                    check="team_id_matches_live",
                    severity="warning",
                    message=(
                        f"{web_name} (code={code}): our team_id={our_team_id} "
                        f"disagrees with live team_id={live_team_id} -- "
                        f"re-run upsert_teams/upsert_players."
                    ),
                )
            )
    return issues


def check_no_single_teammate_monopoly(
    team_weights: dict[int, float], *, monopoly_threshold: float = 0.95
) -> list[QualityIssue]:
    """Flag when one player captures almost the entire team's weight share
    while at least one teammate ALSO has nonzero share -- the
    ``split_multinomial`` degenerate-weight bug class that turned missing
    xG data into a captaincy monopoly (first N.Gonzalez, then Gabriel
    Magalhães). A single nonzero player among an otherwise-all-zero team is
    a different, expected case (everyone else genuinely has no signal yet)
    and is intentionally not flagged here."""
    total = sum(team_weights.values())
    nonzero = [w for w in team_weights.values() if w > 0]
    if total <= 0 or len(nonzero) < 2:
        return []
    top_share = max(team_weights.values()) / total
    if top_share >= monopoly_threshold:
        return [
            QualityIssue(
                check="no_single_teammate_monopoly",
                severity="warning",
                message=(
                    f"one player holds {top_share:.1%} of this team's weight "
                    f"share despite {len(nonzero)} teammates having nonzero "
                    f"weight -- check for a name-matching collision."
                ),
            )
        ]
    return []


def check_source_coverage(
    source: str,
    covered_minutes: float,
    total_minutes: float,
    *,
    min_coverage: float = 0.90,
) -> list[QualityIssue]:
    """Flag an external source that fails to reach the players who actually
    PLAY (2026-08-16).

    ``check_name_match_coverage`` counts names, which treats a squad
    also-ran and a nailed-on starter as equally important. They are not: a
    source missing thirty fringe players is noise, and one missing three
    everpresents is a hole in every projection they appear in. Weighting by
    minutes asks the question that matters — how much of the football we are
    modelling does this source actually see.
    """
    if total_minutes <= 0:
        return []
    coverage = covered_minutes / total_minutes
    if coverage < min_coverage:
        return [
            QualityIssue(
                check="source_coverage",
                severity="error" if coverage < 0.75 else "warning",
                message=(
                    f"{source}: covers only {coverage:.1%} of played minutes "
                    f"— below the {min_coverage:.0%} floor. Players outside it "
                    f"project on defaults, not data."
                ),
            )
        ]
    return []


def check_projection_sanity(
    label: str, values: list[float], *, low: float, high: float
) -> list[QualityIssue]:
    """Flag a projection distribution that has drifted somewhere implausible.

    Unit tests prove the arithmetic is what was written; they cannot tell
    you the answer is sane. A mean projected score of 3 or 300 points is
    wrong regardless of how faithfully the code computed it, and every such
    failure this project has actually had (points-per-appearance mistaken
    for points-per-gameweek, a hit costing 4x the candidate-pool size, a
    horizon silently collapsing to one gameweek) showed up first as a number
    outside its plausible range.
    """
    if not values:
        return [
            QualityIssue(
                check="projection_sanity",
                severity="error",
                message=f"{label}: no values at all — the pipeline produced nothing.",
            )
        ]
    mean = sum(values) / len(values)
    if not low <= mean <= high:
        return [
            QualityIssue(
                check="projection_sanity",
                severity="error",
                message=(
                    f"{label}: mean {mean:.2f} is outside the plausible range "
                    f"[{low}, {high}] — the arithmetic may be right and the "
                    f"answer still wrong."
                ),
            )
        ]
    return []


def check_referential_integrity(
    label: str, orphan_count: int, total: int
) -> list[QualityIssue]:
    """Flag rows referencing a player that does not exist.

    SQLite enforces foreign keys only when ``PRAGMA foreign_keys=ON`` is set
    on the connection, which this project does — but rows written before
    that, or against a different database file, can still be orphaned. An
    orphan is silent: it simply never joins, so the player's data vanishes
    from projections rather than erroring.
    """
    if orphan_count == 0:
        return []
    return [
        QualityIssue(
            check="referential_integrity",
            severity="error",
            message=(
                f"{label}: {orphan_count}/{total} rows reference a player_id "
                f"that is not in `players` — those rows silently never join."
            ),
        )
    ]


def check_column_is_not_a_copy(
    label: str, distinct_count: int, total: int, *, min_distinct_fraction: float = 0.0
) -> list[QualityIssue]:
    """Flag a column that is supposed to carry DIFFERENT information from
    another but is byte-identical to it (2026-08-16).

    Distinct from ``check_stat_column_not_dead``: a copied column is not
    empty and not obviously wrong. ``player_xg_stats.npxg`` held real,
    plausible, non-zero values for every row — they were simply the ``xg``
    values verbatim, because the per-gameweek Understat feed has no penalty
    split. Anything treating the pair as a decomposition (non-penalty xG plus
    penalties) then silently double-counts.

    ``min_distinct_fraction`` defaults to 0.0, i.e. flag only when the two
    columns differ on NO rows at all. That is the signal a verbatim copy
    actually gives; HOW OFTEN two genuinely-different columns should differ
    is domain knowledge a generic check cannot assume. An earlier 1% default
    proved the point by flagging correct data: npxg differs from xg on 0.79%
    of player-matches, because that is simply how often a penalty is taken.
    Callers that do know the expected rate can pass a stricter bound.
    """
    if total == 0:
        return []
    fraction = distinct_count / total
    if distinct_count == 0 or fraction < min_distinct_fraction:
        return [
            QualityIssue(
                check="column_is_not_a_copy",
                severity="warning",
                message=(
                    f"{label}: differs on only {distinct_count}/{total} rows "
                    f"({fraction:.2%}) — effectively a copy, so anything treating "
                    f"the pair as a decomposition will double-count."
                ),
            )
        ]
    return []


def check_setpiece_duty_consistency(
    mismatches: list[str], published_total: int
) -> list[QualityIssue]:
    """Flag ``player_setpiece_roles`` rows whose flags contradict the depth
    chart they were loaded from.

    Two sources write this table with different knowledge and
    ``write_setpiece_roles`` merges them PARTIALLY, which is what makes the
    table useful and also what makes a bad write invisible: an ingest that
    overwrites one column leaves the rest intact, so the row still looks
    populated and plausible. That is how, on 2026-09-02, Understat's weekly
    pass wrote 15 of 20 published first-choice penalty takers back to
    ``is_penalty_taker = 0, penalty_xg_per_game = 0.0`` while leaving
    ``penalty_order = 1`` sitting next to it. ``load_penalty_duty`` filters on
    ``penalty_order IS NOT NULL`` and then reads the value, so it counted
    those players as on duty and worth nothing -- and nothing raised.

    The invariant is local to a single row and needs no external truth:
    ``penalty_order = 1`` and ``is_penalty_taker = 0`` cannot both be right
    about the same player, whichever source wrote which. An error, not a
    warning -- goal_weight is a first-order projection input, and a wrong
    penalty prior moves captaincy.
    """
    if not mismatches:
        return []
    shown = ", ".join(mismatches[:10])
    more = f" (+{len(mismatches) - 10} more)" if len(mismatches) > 10 else ""
    return [
        QualityIssue(
            check="setpiece_duty_consistency",
            severity="error",
            message=(
                f"{len(mismatches)}/{published_total} published depth-chart "
                f"rows contradict their own flags: {shown}{more}. An ingest has "
                f"overwritten FPL's published taker orders -- re-run "
                f"run_full_ingest and check which source failed to defer to "
                f"ingest_fpl_setpiece_roles."
            ),
        )
    ]
