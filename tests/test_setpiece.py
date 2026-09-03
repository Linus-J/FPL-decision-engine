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
from data.models import Base, Player, PlayerSetPieceRole, Team


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


def test_penalty_value_decays_with_depth_chart_position():
    from data.ingestors.setpiece import penalty_xg_for_order

    assert penalty_xg_for_order(1) > penalty_xg_for_order(2) > penalty_xg_for_order(3)
    assert penalty_xg_for_order(None) == 0.0
    assert penalty_xg_for_order(9) == 0.0


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


# --- source precedence (2026-08-16) --------------------------------------


def test_published_duty_is_reported_for_precedence(session):
    """run_weekly runs the FBref scrape WEEKLY, and its inference is from
    last season. Without precedence it would overwrite a published
    current-season duty with a stale one."""
    from data.ingestors.setpiece import players_with_published_duty

    write_setpiece_roles("2026-27", [
        {"player_id": 1, "penalty_order": 1, "is_penalty_taker": True,
         "penalty_xg_per_game": 0.08},
        {"player_id": 2, "key_passes_per_game": 1.0},  # no published duty
    ])
    assert players_with_published_duty("2026-27") == {1}
    assert players_with_published_duty("2025-26") == set()


def test_a_stale_scrape_cannot_zero_a_published_taker(session):
    """The worst case, and the reason precedence exists: a summer signing.
    The depth chart says he takes them; FBref has no PL record of him doing
    so. Writes are partial, so a naive scrape would leave penalty_order=1
    while zeroing penalty_xg_per_game — and load_penalty_duty reads the
    VALUE while filtering on the ORDER, so his duty would silently vanish."""
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "penalty_order": 1, "is_penalty_taker": True,
         "penalty_xg_per_game": 0.08, "source": "depth-chart"},
    ])

    # What the scrape writes for a player with no prior PL penalties, AFTER
    # ingest_setpiece_roles has applied precedence — it drops exactly the two
    # penalty fields for anyone carrying a published duty.
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "key_passes_per_game": 1.7},
    ])

    row = session.query(PlayerSetPieceRole).one()
    assert row.penalty_order == 1
    assert row.is_penalty_taker is True
    assert row.penalty_xg_per_game == pytest.approx(0.08), "published duty survived"
    assert row.key_passes_per_game == pytest.approx(1.7), "scrape still contributed"


# --- name-variant and ambiguity handling (2026-08-16) --------------------
#
# Both of these caused real data loss against the live depth chart, found by
# auditing the database rather than by any test: the file produced 102 roles
# and the table held 100.


def test_requested_stat_types_are_ones_soccerdata_actually_serves():
    """The 2026-08-25 break: this module asked for 'passing' and
    'passing_types', which soccerdata 1.9.1 does not serve for PLAYER SEASON
    stats, so the scrape died on TypeError after doing all the browser work.

    Pinned against soccerdata's own list rather than a hardcoded copy, so a
    library upgrade that adds or removes a table fails here -- cheaply, in
    CI -- instead of at the end of a headed scrape.
    """
    import inspect
    import re

    import soccerdata as sd

    src = inspect.getsource(sd.FBref.read_player_season_stats)
    block = re.search(r"player_stats\s*=\s*\[(.*?)\]", src, re.S)
    assert block, "could not locate soccerdata's player_stats list"
    served = set(re.findall(r"[\"']([a-z_]+)[\"']", block.group(1)))

    from data.ingestors.setpiece import REQUIRED_SEASON_STAT_TYPES

    missing = set(REQUIRED_SEASON_STAT_TYPES) - served
    assert not missing, f"soccerdata no longer serves {sorted(missing)}; served: {sorted(served)}"


def test_absent_corner_data_states_no_opinion_rather_than_False():
    """soccerdata 1.9.1 serves no player-season passing tables, so corners and
    key passes arrive ABSENT. They must be omitted from the role, not written
    as False/0.0.

    Found live 2026-08-25: the scrape wrote is_set_piece_taker=0 over nine
    players from the published depth chart, including B.Fernandes, who is on
    both corners and free kicks there. write_setpiece_roles updates partially,
    so an omitted key preserves the published value while a present one
    destroys it.
    """
    rows = [
        {"player": "A", "team": "T", "penalty_attempts": 6, "matches": 30},
        {"player": "B", "team": "T", "penalty_attempts": 0, "matches": 30},
    ]

    roles = {r["player"]: r for r in derive_setpiece_roles(rows)}

    assert roles["A"]["is_penalty_taker"] is True
    assert "is_set_piece_taker" not in roles["A"], "must not claim an opinion it lacks"
    assert "key_passes_per_game" not in roles["A"]
    assert "B" not in roles, "a player with no duty at all should not get a row"


def test_measured_zero_corners_is_still_an_opinion():
    """A source that DOES carry corners and reports zero is saying he does not
    take them. That is real information and must be written, unlike absence."""
    rows = [
        {"player": "A", "team": "T", "penalty_attempts": 6, "corners": 0, "matches": 30},
        {"player": "C", "team": "T", "penalty_attempts": 0, "corners": 40, "matches": 30},
    ]

    roles = {r["player"]: r for r in derive_setpiece_roles(rows)}

    assert roles["A"]["is_set_piece_taker"] is False
    assert roles["C"]["is_set_piece_taker"] is True


def test_role_with_no_fields_is_not_written(session):
    """A deferred player whose every derivable field was stripped has nothing
    to say. Writing the row anyway only bumps updated_at and inflates the
    reported count with writes that changed nothing."""
    session.add(Team(id=1, name="T", short_name="T"))
    session.add(Player(id=1, fpl_id=1, code=1, first_name="A", second_name="B",
                       web_name="AB", team_id=1, position="MID", now_cost=5.0))
    session.commit()

    n = write_setpiece_roles("2026-27", [{"player_id": 1, "player": "AB", "team": "T"}])

    assert n == 0
    assert session.query(PlayerSetPieceRole).count() == 0


# --- FPL's own published orders (2026-09-03) -----------------------------

def _el(code, *, pen=None, fk=None, corner=None):
    return {
        "code": code,
        "penalties_order": pen,
        "direct_freekicks_order": fk,
        "corners_and_indirect_freekicks_order": corner,
    }


def test_orders_become_roles_with_the_value_the_order_implies():
    from data.ingestors.setpiece import roles_from_fpl_elements

    roles = roles_from_fpl_elements([_el(100, pen=1, fk=2, corner=2)], {100: 7})

    assert roles == [{
        "player_id": 7, "penalty_order": 1, "freekick_order": 2,
        "corner_order": 2, "is_penalty_taker": True,
        "penalty_xg_per_game": 0.08058, "is_set_piece_taker": True,
        "source": "fpl",
    }]


def test_only_the_first_choice_is_the_penalty_taker():
    """Order is the whole point. A second-choice taker carries real value but
    is not "the" taker, and the boolean is what features.py reads."""
    from data.ingestors.setpiece import roles_from_fpl_elements

    roles = roles_from_fpl_elements([_el(100, pen=2)], {100: 7})

    assert roles[0]["is_penalty_taker"] is False
    assert roles[0]["penalty_xg_per_game"] == pytest.approx(0.01138)


def test_players_with_no_duty_at_all_are_skipped():
    """Most of a 650-player pool takes nothing. A row asserting False carries
    no more information than no row, and writing one for everybody would make
    players_with_published_roles meaningless."""
    from data.ingestors.setpiece import roles_from_fpl_elements

    assert roles_from_fpl_elements([_el(100), _el(101, corner=3)], {100: 7, 101: 8}) == [
        {"player_id": 8, "penalty_order": None, "freekick_order": None,
         "corner_order": 3, "is_penalty_taker": False,
         "penalty_xg_per_game": 0.0, "is_set_piece_taker": True, "source": "fpl"},
    ]


def test_a_set_piece_only_taker_is_still_a_set_piece_taker():
    from data.ingestors.setpiece import roles_from_fpl_elements

    roles = roles_from_fpl_elements([_el(100, fk=1)], {100: 7})

    assert roles[0]["is_set_piece_taker"] is True
    assert roles[0]["is_penalty_taker"] is False


def test_an_unknown_code_is_skipped_rather_than_guessed_at():
    """The join is on `code` precisely so there is nothing to guess. A code
    with no player is a squad we have not ingested yet, not a near-match."""
    from data.ingestors.setpiece import roles_from_fpl_elements

    assert roles_from_fpl_elements([_el(999, pen=1)], {100: 7}) == []


def test_fpl_roles_claim_nothing_about_key_passes():
    """FPL has no opinion on key passes, and write_setpiece_roles updates
    partially — so omitting the key preserves what Understat wrote."""
    from data.ingestors.setpiece import roles_from_fpl_elements

    assert "key_passes_per_game" not in roles_from_fpl_elements(
        [_el(100, pen=1)], {100: 7}
    )[0]


def test_a_player_fpl_no_longer_lists_is_retired():
    """Only possible because the feed is COMPLETE: absence from it is a
    positive statement of no duty. Enzo, Ndiaye and Marmoush all held orders
    at clubs they had left because the old text file was reloaded, never
    diffed."""
    from data.ingestors.setpiece import retirement_roles

    assert retirement_roles({7, 8}, {7}) == [{
        "player_id": 8, "penalty_order": None, "freekick_order": None,
        "corner_order": None, "is_penalty_taker": False,
        "penalty_xg_per_game": 0.0, "is_set_piece_taker": False,
        "source": "fpl",
    }]


def test_retirement_clears_the_flags_as_well_as_the_orders():
    """load_penalty_duty filters on penalty_order and reads the VALUE, so a
    half-cleared row is exactly the shape that hid the Understat clobber."""
    from data.ingestors.setpiece import retirement_roles

    (role,) = retirement_roles({8}, set())
    assert role["penalty_order"] is None
    assert role["is_penalty_taker"] is False
    assert role["penalty_xg_per_game"] == 0.0


def test_a_still_listed_player_is_not_retired():
    from data.ingestors.setpiece import retirement_roles

    assert retirement_roles({7}, {7}) == []


def test_retirement_actually_clears_the_row(session):
    from data.ingestors.setpiece import (
        players_with_published_roles,
        retirement_roles,
    )

    write_setpiece_roles("2026-27", [
        {"player_id": 8, "penalty_order": 1, "is_penalty_taker": True,
         "penalty_xg_per_game": 0.08058},
    ])
    assert players_with_published_roles("2026-27") == {8}

    write_setpiece_roles("2026-27", retirement_roles({8}, set()))

    assert players_with_published_roles("2026-27") == set()
    row = session.query(PlayerSetPieceRole).one()
    assert row.penalty_order is None and row.penalty_xg_per_game == 0.0
