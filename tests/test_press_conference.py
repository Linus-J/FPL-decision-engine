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


# --- real misclassifications found in the live table (2026-08-17) -----------
def test_unavailable_is_not_read_as_available():
    """Substring containment scored a player who was UNAVAILABLE as +1.0,
    because "available" is inside "unavailable". Observed live on Odegaard."""
    from data.ingestors.press_conference import _score_sentence

    assert _score_sentence("the captain was also unavailable for spells") < 0


def test_match_commentary_is_not_read_as_a_long_term_absence():
    """"out for" matched ordinary commentary -- a ball hooked out for a throw,
    a shot deflected out for a corner -- and scored it as an absence. Both were
    in the live table, on Sarmiento and Whittaker."""
    from data.ingestors.press_conference import _score_sentence

    assert _score_sentence("boro eventually hook it out for a throw") == 0.0
    assert _score_sentence("a shot that is deflected out for a corner") == 0.0


def test_a_genuine_long_term_absence_still_scores_negative():
    """The narrowing must not cost the signal it exists to catch."""
    from data.ingestors.press_conference import _score_sentence

    for s in (
        "the frenchman out for an extended period",
        "he is out for the season with a knee injury",
        "he will be out for several weeks",
        "he is out injured",
    ):
        assert _score_sentence(s) < 0, s


def test_clear_positives_still_score_positive():
    from data.ingestors.press_conference import _score_sentence

    assert _score_sentence("he is fully fit and available") > 0
    assert _score_sentence("he is back in training and in contention") > 0


def test_plural_keywords_survive_word_boundary_matching():
    """Word-bounding the match fixed the "unavailable" false positive but
    broke plurals that substring containment used to catch -- "fitness doubts"
    scored 0.0 and lost a genuine availability signal. Caught by re-scoring
    the live table rather than by the suite."""
    from data.ingestors.press_conference import _score_sentence

    assert _score_sentence("were not considered because of fitness doubts") < 0
    assert _score_sentence("there is a doubt over his fitness") < 0
    assert _score_sentence("he is awaiting scans on the injury") < 0
