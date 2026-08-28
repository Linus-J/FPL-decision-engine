"""A weekly step must not report success having written nothing (2026-08-28).

Twice in one day a warn-and-continue step exited 0 having ingested nothing,
and the run reported success:

- ``scrape_understat_xg.py`` treated an unreadable schedule as "the season
  isn't published yet", skipped the refresh and exited 0. ``player_xg_stats``
  sat at 2 non-zero xg rows out of 309 for weeks.
- Before that, the same shape hid the missing Understat step entirely --
  ``run_weekly.py``'s own comments record 11,495 rows for 2025-26 with 87
  non-zero xg, "left in place because nothing ever ran it".

``_run_or_warn`` only ever checked the exit code, and every one of these
steps exits 0 on an empty result. A post-condition asks the different
question: did the data this step is responsible for actually end up there?

Post-conditions rather than before/after deltas, deliberately. These steps
are idempotent upserts, so a re-run legitimately changes no row counts --
warning on "nothing changed" would fire every week on a healthy run, which
is the cry-wolf failure the gate's own comments warn about.
"""

from __future__ import annotations

from scripts.run_weekly import _postcondition_warning


def test_zero_rows_is_warned_and_names_the_step_and_the_data():
    warning = _postcondition_warning(
        "scripts/scrape_understat_xg.py", "2026-27 xG rows carrying real xg", 0
    )
    assert warning is not None
    assert "scripts/scrape_understat_xg.py" in warning
    assert "2026-27 xG rows carrying real xg" in warning


def test_rows_present_is_silent():
    assert _postcondition_warning("step", "some rows", 147) is None


def test_an_uncheckable_postcondition_is_warned_not_silently_passed():
    """A check that cannot run must never read as a pass -- that is the same
    failure class it exists to catch."""
    warning = _postcondition_warning("step", "some rows", None)
    assert warning is not None
    assert "could not be checked" in warning.lower()
