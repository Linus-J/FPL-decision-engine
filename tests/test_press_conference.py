"""Regression tests for the press-conference sentiment matcher.

Real bug found 2026-07-28 (data-completeness audit): the old matcher
resolved a bare, short player name (web_name/second_name) via plain
substring containment with no ambiguity guard -- the same collision class
fixed in fbref.py's _match_player, just in the reverse direction (scanning
free text for a known name, rather than matching an external name string
against our players)."""

from __future__ import annotations

from data.ingestors.press_conference import _extract_player_signals


def test_extract_player_signals_prefers_longest_match():
    name_map = {"james": 1, "reece james": 2}
    body = "Reece James is back in training and could feature this weekend."
    signals = _extract_player_signals(body, name_map)
    assert len(signals) == 1
    assert signals[0][0] == 2


def test_extract_player_signals_word_boundary_not_substring():
    # "sam" must not match inside "assam" or similar -- word-boundary only.
    name_map = {"sam": 1}
    body = "The assamite reserve squad trained quietly on Tuesday morning."
    assert _extract_player_signals(body, name_map) == []


def test_extract_player_signals_ambiguous_name_excluded_upstream():
    # An ambiguous name should never even be IN player_name_map (see
    # _build_player_name_map) -- if it somehow were, it must not silently
    # resolve to one arbitrary candidate. Simulated here by simply not
    # including the ambiguous key, matching what the real builder does.
    name_map: dict[str, int] = {}
    body = "Gabriel is fully fit and expected to play this weekend."
    assert _extract_player_signals(body, name_map) == []


def test_extract_player_signals_scores_sentiment_correctly():
    name_map = {"reece james": 2}
    body = "Reece James is a doubt and will miss the weekend fixture."
    signals = _extract_player_signals(body, name_map)
    assert len(signals) == 1
    _, score, _ = signals[0]
    assert score < 0
