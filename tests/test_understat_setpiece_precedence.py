"""data/ingestors/understat.py must not overwrite a published depth chart.

The FBref set-piece scrape was given source precedence in 0ef8f75. The
Understat ingest -- which runs on EVERY weekly run via scripts/run_agent.py,
where the FBref one is comparatively occasional -- never was, and quietly
undid the depth chart every week.

Found live on 2026-09-03 while auditing the GW3 decision: two games into
2026-27, Szoboszlai (Liverpool's SECOND-choice penalty taker, who had taken
one) carried 0.3806 penalty xG per game against the 0.01138 his order
implies, while Haaland, Saka, Palmer, Isak, Mateta and ten other published
FIRST-choice takers had been written back to 0.0. projection/assemble.py's
load_penalty_duty reads that column straight into goal_weight, and the GW3
run captained Szoboszlai over Haaland.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import data.ingestors.setpiece as setpiece
from data.ingestors.setpiece import (
    penalty_xg_for_order,
    players_with_published_roles,
    write_setpiece_roles,
)
from data.ingestors.understat import setpiece_role_from_understat
from data.models import Base, PlayerSetPieceRole


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'sp.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(setpiece, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


# --- the role dict itself ------------------------------------------------

def test_a_published_player_gets_no_penalty_opinion():
    """The withheld keys are the mechanism: write_setpiece_roles updates only
    what it is given, so an absent key keeps the depth chart's value."""
    role = setpiece_role_from_understat(
        7, xg=0.93, npxg=0.17, key_passes_per_game=2.5, games=2, is_published=True
    )

    assert "is_penalty_taker" not in role
    assert "penalty_xg_per_game" not in role
    assert "is_set_piece_taker" not in role


def test_a_published_player_still_contributes_key_passes():
    """A taker list has no opinion on key passes, so deference must not
    degrade into contributing nothing at all."""
    role = setpiece_role_from_understat(
        7, xg=0.93, npxg=0.17, key_passes_per_game=2.5, games=2, is_published=True
    )

    assert role == {"player_id": 7, "key_passes_per_game": 2.5}


def test_an_unpublished_player_is_still_inferred_from_realised_penalties():
    """Deference is not a disabling. A player absent from the taker list has
    no published duty to protect, and realised penalty xG is the only
    evidence there is for him."""
    role = setpiece_role_from_understat(
        7, xg=1.53, npxg=0.01, key_passes_per_game=0.2, games=2, is_published=False
    )

    assert role["is_penalty_taker"] is True
    assert role["penalty_xg_per_game"] == pytest.approx(0.76)
    assert role["is_set_piece_taker"] is False


def test_realised_penalties_never_crown_a_published_second_choice(session):
    """The exact live defect. Szoboszlai is order 2 and took one penalty in
    two games; unprotected, that reads as 0.3806 per game -- 33x the 0.01138
    his order implies, and worth ~1.85 xPts a game to a midfielder."""
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "penalty_order": 2, "is_penalty_taker": False,
         "penalty_xg_per_game": penalty_xg_for_order(2), "source": "allaboutfpl"},
    ])

    role = setpiece_role_from_understat(
        1, xg=0.9323, npxg=0.1712, key_passes_per_game=2.5, games=2,
        is_published=1 in players_with_published_roles("2026-27"),
    )
    write_setpiece_roles("2026-27", [role])

    row = session.query(PlayerSetPieceRole).one()
    assert row.penalty_order == 2
    assert row.is_penalty_taker is False
    assert row.penalty_xg_per_game == pytest.approx(0.01138)
    assert row.key_passes_per_game == pytest.approx(2.5), "still contributed"


def test_a_first_choice_taker_who_has_not_taken_one_yet_keeps_his_duty(session):
    """Haaland's half of the same bug. Two games in, the first-choice taker
    may simply not have had a penalty awarded -- which is not evidence that
    he is off them."""
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "penalty_order": 1, "is_penalty_taker": True,
         "penalty_xg_per_game": penalty_xg_for_order(1), "source": "allaboutfpl"},
    ])

    role = setpiece_role_from_understat(
        1, xg=1.4407, npxg=1.4407, key_passes_per_game=0.5, games=2,
        is_published=1 in players_with_published_roles("2026-27"),
    )
    write_setpiece_roles("2026-27", [role])

    row = session.query(PlayerSetPieceRole).one()
    assert row.is_penalty_taker is True
    assert row.penalty_xg_per_game == pytest.approx(0.08058)


def test_key_passes_never_overwrite_published_set_piece_duty(session):
    """is_set_piece_taker was derived from key passes per game, which is a
    creativity proxy rather than a duty. It had written 43 published
    corner/free-kick takers back to False."""
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "penalty_order": 1, "freekick_order": 2,
         "corner_order": 2, "is_set_piece_taker": True, "source": "allaboutfpl"},
    ])

    role = setpiece_role_from_understat(
        1, xg=0.5, npxg=0.5, key_passes_per_game=0.4, games=2,
        is_published=1 in players_with_published_roles("2026-27"),
    )
    write_setpiece_roles("2026-27", [role])

    assert session.query(PlayerSetPieceRole).one().is_set_piece_taker is True


# --- who counts as published --------------------------------------------

def test_a_penalty_only_taker_counts_as_published(session):
    """Haaland holds no free-kick or corner duty, and the taker list saying so
    IS a published claim. Keying deference off the set-piece orders alone
    would let the key-pass proxy invent duty for exactly this population."""
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "penalty_order": 1, "is_set_piece_taker": False},
    ])

    assert players_with_published_roles("2026-27") == {1}


def test_a_set_piece_only_taker_counts_as_published(session):
    """The converse: a corner taker who is not on penalties. players_with_
    published_duty (penalties only) would miss him, which is why this wider
    sibling exists."""
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "corner_order": 1, "is_set_piece_taker": True},
    ])

    assert players_with_published_roles("2026-27") == {1}


def test_a_player_with_no_published_row_is_not_protected(session):
    write_setpiece_roles("2026-27", [
        {"player_id": 1, "key_passes_per_game": 1.0},
    ])

    assert players_with_published_roles("2026-27") == set()


def test_published_roles_are_scoped_per_season(session):
    write_setpiece_roles("2025-26", [{"player_id": 1, "penalty_order": 1}])

    assert players_with_published_roles("2026-27") == set()
