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
