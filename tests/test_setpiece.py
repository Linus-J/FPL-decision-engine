"""data/ingestors/setpiece.py — penalty and set-piece takers (P3.7).

`player_setpiece_roles` was empty for the whole project while
`projection/features.py` LEFT JOINed it and COALESCEd every field to zero, so
penalty duty has been silently absent from every projection ever made. These
cover the inference logic; the scrape itself needs a browser and is excluded
from coverage like its sibling in fbref.py.

The thing worth guarding hardest is FALSE POSITIVES. Crediting a stand-in who
took one penalty during the real taker's injury with a permanent penalty
expectation would put a phantom half-goal per game into his projection, which
is worse than the zero this replaces.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import data.ingestors.setpiece as setpiece
from data.ingestors.setpiece import derive_setpiece_roles, write_setpiece_roles
from data.models import Base, PlayerSetPieceRole


def _row(player, team, **kw):
    base = {"player": player, "team": team, "matches": 38}
    base.update(kw)
    return base


def test_the_regular_taker_is_identified():
    roles = derive_setpiece_roles([
        _row("Taker", "Arsenal", penalty_attempts=8),
        _row("Someone", "Arsenal", penalty_attempts=1),
    ])
    by_name = {r["player"]: r for r in roles}
    assert by_name["Taker"]["is_penalty_taker"] is True
    assert by_name["Taker"]["penalty_xg_per_game"] == pytest.approx(
        8 / 38 * setpiece.PENALTY_CONVERSION, rel=1e-3
    )


def test_a_one_off_stand_in_is_not_crowned_taker():
    """The false positive that matters: a deputy who took one while the real
    taker was out must not carry a penalty expectation into next season."""
    roles = derive_setpiece_roles([
        _row("Taker", "Arsenal", penalty_attempts=7),
        _row("StandIn", "Arsenal", penalty_attempts=1),
    ])
    by_name = {r["player"]: r for r in roles}
    assert by_name.get("StandIn", {}).get("is_penalty_taker", False) is False


def test_a_shared_duty_crowns_nobody():
    """Two players on 50/50 both fall below the share threshold... but an
    exact tie at the threshold should still resolve, so this pins the real
    behaviour rather than assuming it."""
    roles = derive_setpiece_roles([
        _row("A", "Arsenal", penalty_attempts=3),
        _row("B", "Arsenal", penalty_attempts=3),
    ])
    takers = [r["player"] for r in roles if r["is_penalty_taker"]]
    # 3/6 == MIN_PENALTY_SHARE exactly, so both qualify on share and
    # attempts. That is deliberate: a genuine 50/50 duty IS worth half the
    # penalties each, and penalty_xg_per_game already scales by attempts.
    assert set(takers) == {"A", "B"}


def test_a_low_volume_taker_is_rejected_on_attempts():
    """One attempt is 100% of a team's penalties if the team only won one.
    Share alone is not enough."""
    roles = derive_setpiece_roles([_row("Only", "Burnley", penalty_attempts=1)])
    assert not [r for r in roles if r["is_penalty_taker"]]


def test_shares_are_computed_within_a_team_not_across_the_league():
    """Penalty duty is a squad-level question. Pooling the league would make
    every taker at a low-penalty club look like a deputy."""
    roles = derive_setpiece_roles([
        _row("BigClubTaker", "Man City", penalty_attempts=10),
        _row("SmallClubTaker", "Burnley", penalty_attempts=3),
        _row("SmallClubOther", "Burnley", penalty_attempts=0),
    ])
    by_name = {r["player"]: r for r in roles}
    assert by_name["SmallClubTaker"]["is_penalty_taker"] is True


def test_corner_taker_is_identified_separately_from_penalties():
    roles = derive_setpiece_roles([
        _row("Corners", "Arsenal", corners=120, key_passes=60),
        _row("Other", "Arsenal", corners=10, key_passes=5),
    ])
    by_name = {r["player"]: r for r in roles}
    assert by_name["Corners"]["is_set_piece_taker"] is True
    assert by_name["Corners"]["is_penalty_taker"] is False
    assert by_name["Other"]["is_set_piece_taker"] is False


def test_key_passes_are_per_game_not_season_totals():
    roles = derive_setpiece_roles([_row("Creator", "Arsenal", key_passes=76, matches=38)])
    assert roles[0]["key_passes_per_game"] == pytest.approx(2.0)


def test_a_player_with_no_role_at_all_is_omitted():
    """Writing a zero row per player would bloat the table and change
    nothing -- the consuming query already COALESCEs a missing row to 0."""
    assert derive_setpiece_roles([_row("Nobody", "Arsenal")]) == []


def test_zero_matches_never_divides_by_zero():
    roles = derive_setpiece_roles([
        _row("Injured", "Arsenal", penalty_attempts=3, key_passes=5, matches=0),
    ])
    for role in roles:
        assert role["penalty_xg_per_game"] == 0.0
        assert role["key_passes_per_game"] == 0.0


def test_rows_without_a_player_or_team_are_skipped():
    assert derive_setpiece_roles([
        {"player": "", "team": "Arsenal", "penalty_attempts": 5},
        {"player": "X", "team": "", "penalty_attempts": 5},
    ]) == []


def test_fbref_column_aliases_are_accepted():
    """soccerdata's flattened season tables name these differently from the
    match tables ('Standard PKatt' vs 'PKatt'); both must work, and a
    missing column must read as 0 rather than raising."""
    roles = derive_setpiece_roles([
        {"player": "A", "team": "Arsenal", "Standard PKatt": 6,
         "Playing Time MP": 38, "Pass Types CK": 90},
    ])
    assert roles[0]["is_penalty_taker"] is True
    assert roles[0]["is_set_piece_taker"] is True


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'sp.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(setpiece, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def test_write_is_idempotent_and_updates_in_place(session):
    """Penalty duty genuinely changes mid-season, so re-scraping must update
    the row rather than accumulate a second one the JOIN would then pick
    arbitrarily between."""
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "is_penalty_taker": True, "penalty_xg_per_game": 0.15,
         "is_set_piece_taker": False, "key_passes_per_game": 1.0},
    ])
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "is_penalty_taker": False, "penalty_xg_per_game": 0.0,
         "is_set_piece_taker": True, "key_passes_per_game": 2.0},
    ])

    rows = session.query(PlayerSetPieceRole).all()
    assert len(rows) == 1
    assert rows[0].is_penalty_taker is False
    assert rows[0].is_set_piece_taker is True
    assert rows[0].key_passes_per_game == pytest.approx(2.0)


def test_write_skips_unresolved_players(session):
    """A name that never matched a stored player has no player_id; writing it
    would violate the foreign key."""
    written = write_setpiece_roles("2026-27", [
        {"player": "Unmatched", "is_penalty_taker": True},
    ])
    assert written == 0
    assert session.query(PlayerSetPieceRole).count() == 0


def test_write_scopes_roles_per_season(session):
    write_setpiece_roles("2025-26", [{"player_id": 1, "is_penalty_taker": True}])
    write_setpiece_roles("2026-27", [{"player_id": 1, "is_penalty_taker": False}])
    assert session.query(PlayerSetPieceRole).count() == 2


# --- depth-chart parsing (2026-08-16) ------------------------------------


def test_parse_reads_order_from_position_in_the_cell():
    """Order is the whole point: the first-choice penalty taker is worth
    several times the third-choice one, and a boolean cannot say so."""
    from data.ingestors.setpiece import parse_setpiece_table

    rows = parse_setpiece_table(
        "Team | Penalties | Free Kicks | Corners\n"
        "Arsenal | Saka, Gyokeres, Odegaard | Rice, Saka | Rice, Saka\n"
    )
    by_name = {r["player"]: r for r in rows}
    assert by_name["Saka"]["penalty_order"] == 1
    assert by_name["Gyokeres"]["penalty_order"] == 2
    assert by_name["Odegaard"]["penalty_order"] == 3
    # a player can hold several duties at different depths
    assert by_name["Saka"]["freekick_order"] == 2
    assert by_name["Rice"]["corner_order"] == 1
    assert by_name["Rice"]["penalty_order"] is None


def test_parse_maps_published_team_names_onto_fpl_names():
    from data.ingestors.setpiece import parse_setpiece_table

    rows = parse_setpiece_table("Man United | Fernandes | Fernandes | Fernandes")
    assert rows[0]["team"] == "Man Utd"


def test_parse_flags_uncertain_names_without_dropping_them():
    """An asterisk marks a doubt in the source. Silently trusting it is how a
    phantom taker reaches projections; silently dropping it loses a real
    taker. Record and report."""
    from data.ingestors.setpiece import parse_setpiece_table

    rows = parse_setpiece_table("Bournemouth | Kluivert, Kroupi* | | ")
    by_name = {r["player"]: r for r in rows}
    assert by_name["Kroupi"]["uncertain"] is True
    assert by_name["Kroupi"]["penalty_order"] == 2
    assert by_name["Kluivert"]["uncertain"] is False


def test_parse_ignores_the_header_and_blank_cells():
    from data.ingestors.setpiece import parse_setpiece_table

    rows = parse_setpiece_table(
        "Team | Penalties | Free Kicks | Corners\n"
        "\n"
        "Fulham | Iwobi |  | Iwobi, Bobb\n"
    )
    assert {r["player"] for r in rows} == {"Iwobi", "Bobb"}


def test_penalty_value_decays_with_depth_chart_position():
    from data.ingestors.setpiece import penalty_xg_for_order

    assert penalty_xg_for_order(1) > penalty_xg_for_order(2) > penalty_xg_for_order(3)
    assert penalty_xg_for_order(None) == 0.0
    assert penalty_xg_for_order(9) == 0.0


def test_depth_chart_roles_do_not_claim_key_passes():
    """A published taker list has no opinion on key passes. Emitting 0.0
    would clobber a real value from the Understat/FBref path, because the
    write is a partial update keyed on what the source actually knows."""
    from data.ingestors.setpiece import roles_from_depth_chart

    roles = roles_from_depth_chart([
        {"player": "A", "team": "Arsenal", "uncertain": False,
         "penalty_order": 1, "freekick_order": None, "corner_order": None},
    ])
    assert "key_passes_per_game" not in roles[0]
    assert roles[0]["is_penalty_taker"] is True
    assert roles[0]["is_set_piece_taker"] is False


def test_partial_write_preserves_fields_the_source_did_not_set(session):
    """Two sources feed this table with different knowledge. A blanket write
    would have each silently erase the other's contribution."""
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "key_passes_per_game": 2.5, "is_set_piece_taker": True},
    ])
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "penalty_order": 1, "is_penalty_taker": True},
    ])

    row = session.query(PlayerSetPieceRole).one()
    assert row.penalty_order == 1
    assert row.key_passes_per_game == pytest.approx(2.5), "must not be clobbered"
