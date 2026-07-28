"""Regression test for the legacy Understat ingestor (data/ingestors/
understat.py), which is wired into scripts/run_agent.py's live production
pipeline. Real bug found 2026-07-28 (data-completeness audit): this module
used to carry its own local name matcher with the exact unguarded-substring
collision the Gabriel Magalhães fix (fbref.py) addressed, but unlike
understat_xg.py this one runs on every live agent cycle, not just an
occasional manual backfill. It now reuses fbref.py's hardened matcher --
this test just confirms the wiring, not the matcher logic itself (already
covered by test_fbref_name_matching.py)."""

from __future__ import annotations

from data.ingestors import understat
from data.ingestors.fbref import _match_player as fbref_match_player


def test_legacy_understat_reuses_the_hardened_matcher():
    assert understat._match_player is fbref_match_player


def test_legacy_understat_does_not_hijack_a_short_name_collision():
    name_map = {
        "gabriel dos santos magalhaes": 5,
        "gabriel": 5,
        "gabriel martinelli silva": 18,
        "martinelli": 18,
    }
    assert understat._match_player("Gabriel Martinelli", name_map) == 18
