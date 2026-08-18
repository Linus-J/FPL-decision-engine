"""FBref match-summary column mapping (data/ingestors/fbref.py).

2026-08-18. Three entries in ``FBREF_SUMMARY_MAP`` named columns that do not
exist in FBref's match-summary table -- ``"Passes Att"``, ``"Passes Cmp%"`` and
``"Performance Blocks"``. ``map_summary_row`` matches names EXACTLY and
silently omits anything it cannot find, so the ORM default (0) applied and
``passes``, ``pass_completion_pct`` and ``blocks`` read zero on all 11,182
rows. A field the codebase believed it collected was indistinguishable from a
genuine zero -- this project's recurring failure shape.

Meanwhile five columns that ARE in the table went unread: fouls, offsides,
crosses, own goals and conceded penalties, all of them BPS inputs. Same shape
as the ``npg90`` finding in the 2026-08-16 audit: the data was in the
downloaded page the whole time.

The header below is copied verbatim from a real cached match page, so this
test fails if either the mapping or FBref's layout drifts.
"""

from __future__ import annotations

from data.ingestors.fbref import FBREF_SUMMARY_MAP, map_summary_row

# Exactly what pandas produces from FBref's summary table once the MultiIndex
# is flattened -- verified against /home/linus/soccerdata/data/FBref/match_*.html.
REAL_SUMMARY_COLUMNS = {
    "Player", "#", "Nation", "Pos", "Age", "Min",
    "Performance Gls", "Performance Ast", "Performance PK", "Performance PKatt",
    "Performance Sh", "Performance SoT", "Performance CrdY", "Performance CrdR",
    "Performance Fls", "Performance Fld", "Performance Off", "Performance Crs",
    "Performance TklW", "Performance Int", "Performance OG",
    "Performance PKwon", "Performance PKcon",
}

# Verified against soccerdata's own parsed frame for 2025-26 (11,492 rows):
#   Performance Fls   5,331 non-zero rows
#   Performance Crs   5,044
#   Performance Off     999
#   Performance OG       39
#   Performance PKcon     0 non-NULL -- the column exists but FBref does
#                           not populate it, so it maps to a real absence
#                           rather than a wrong name.


def test_every_mapped_column_actually_exists_in_the_table():
    """The bug, stated directly: a mapping that matches nothing is silently
    dropped, and the resulting zero is indistinguishable from a real one."""
    # soccerdata hands over a MultiIndex the ingest flattens as
    # "<Section> <Leaf>"; the leading identity columns come through lowercase
    # and un-sectioned ("min"), which is why that one is exempt here.
    exempt = {"min"}
    missing = {
        field: col for field, col in FBREF_SUMMARY_MAP.items()
        if col not in REAL_SUMMARY_COLUMNS
        and col not in exempt
        and not col.startswith("Take-Ons")
    }
    assert not missing, (
        f"these map to columns FBref's summary table does not have, so they "
        f"will silently read 0 forever: {missing}"
    )


def test_the_bps_inputs_the_summary_table_carries_are_read():
    """Fouls, offsides and crosses are BPS-scoring events sitting in a page the
    scraper already downloads. They were dropped on the floor."""
    for field in ("fouls", "offsides", "open_play_crosses", "own_goals",
                  "penalties_conceded"):
        assert field in FBREF_SUMMARY_MAP, f"{field} is available and unmapped"


def test_map_summary_row_reads_a_real_row():
    raw = {
        "Player": "A Player", "Min": 90,
        "Performance Gls": 1, "Performance Ast": 1,
        "Performance Sh": 4, "Performance SoT": 2,
        "Performance CrdY": 1, "Performance CrdR": 0,
        "Performance Fls": 3, "Performance Off": 2, "Performance Crs": 5,
        "Performance TklW": 2, "Performance Int": 1,
        "Performance OG": 1, "Performance PKcon": 1,
        "Performance PK": 0, "Performance PKatt": 1,
    }
    out = map_summary_row(raw)
    assert out["goals"] == 1
    assert out["fouls"] == 3
    assert out["offsides"] == 2
    assert out["open_play_crosses"] == 5
    assert out["own_goals"] == 1
    assert out["penalties_conceded"] == 1
    assert out["tackles"] == 2                 # TklW: tackles WON, not attempted
    assert out["shots_off_target"] == 2        # derived, 4 - 2
    assert out["penalties_missed"] == 1        # derived, PKatt - PK


def test_passing_volume_is_not_claimed_from_this_table():
    """`passes`/`pass_completion_pct` live in FBref's separate "passing" stat
    type, which this ingest does not fetch. Mapping them here produced a
    confident zero instead of an honest absence, and pass-completion is the
    single largest positive BPS component available to an outfielder -- so a
    future reader must not quietly re-add them."""
    assert "passes" not in FBREF_SUMMARY_MAP
    assert "pass_completion_pct" not in FBREF_SUMMARY_MAP
