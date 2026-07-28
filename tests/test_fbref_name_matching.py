"""Real bug found 2026-07-28 (walk-forward gate investigation): the shared
name matcher (data/ingestors/fbref.py, used by fbref.py/understat_xg.py/
whoscored.py) missed 21+ significant players for an ENTIRE season --
including this bot's own most-favoured captains -- because their stored
full legal name carries a middle name or second surname the external
source (FBref/Understat) drops, which breaks contiguous-substring
containment in both directions, and because Turkish/Nordic/Polish
characters (ı, ğ, ø, ł...) don't decompose under NFKD the way accented
Latin letters do.
"""

from __future__ import annotations

from data.ingestors.fbref import _match_player, _normalize_name


def test_normalize_name_strips_standard_accents():
    assert _normalize_name("González") == "gonzalez"
    assert _normalize_name("Paquetá") == "paqueta"
    assert _normalize_name("Ekitiké") == "ekitike"


def test_normalize_name_handles_non_decomposing_characters():
    # NFKD alone turns this into "kadoglu" (drops the dotless-i entirely) --
    # confirmed live before this fix.
    assert _normalize_name("Kadıoğlu") == "kadioglu"


def test_match_player_exact_match():
    name_map = {"bruno fernandes": 1}
    assert _match_player("Bruno Fernandes", name_map) == 1


def test_match_player_substring_fallback_still_works():
    name_map = {"martin dubravka": 5, "d.dubravka": 5}
    assert _match_player("Dubravka", name_map) == 5


def test_match_player_extra_middle_name_uses_token_subset_fallback():
    # Real case: stored "first_name second_name" = "Bruno Borges Fernandes",
    # external source reports "Bruno Fernandes" -- neither substring
    # direction matches, only token-subset does.
    name_map = {"bruno borges fernandes": 553, "b.fernandes": 553}
    assert _match_player("Bruno Fernandes", name_map) == 553


def test_match_player_iberian_dual_surname_uses_token_subset_fallback():
    # Real case: stored second_name = "Gonzalez Iglesias" (dual surname),
    # external source drops the maternal surname.
    name_map = {"nico gonzalez iglesias": 521, "n.gonzalez": 521}
    assert _match_player("Nico Gonzalez", name_map) == 521


def test_match_player_long_legal_name_uses_token_subset_fallback():
    # Real case: Palhinha's stored full legal name has 8 tokens; "Palhinha"
    # (his common surname) isn't the LAST token, ruling out a naive
    # first/last-token heuristic -- only a full token-subset check catches it.
    name_map = {
        "joao maria lobo alves palhares costa palhinha goncalves": 745,
        "j.palhinha": 745,
    }
    assert _match_player("Joao Palhinha", name_map) == 745


def test_match_player_no_match_returns_none():
    name_map = {"bruno borges fernandes": 553}
    assert _match_player("Completely Different Player", name_map) is None


def test_match_player_short_web_name_does_not_hijack_a_different_player():
    # Real bug found live: web_name "Gabriel" (Arsenal's Gabriel Magalhaes,
    # a CB) matched as a substring of "Gabriel Martinelli" and "Gabriel
    # Jesus" -- two DIFFERENT real players -- merging their real xG/xA into
    # a centre-back's totals (single-match readings above 1.5, confirmed
    # live). A single-token candidate must never win via fuzzy containment.
    name_map = {
        "gabriel dos santos magalhaes": 5, "gabriel": 5,
        "gabriel martinelli silva": 18, "martinelli": 18,
        "gabriel fernando de jesus": 29, "g.jesus": 29,
    }
    assert _match_player("Gabriel Martinelli", name_map) == 18
    assert _match_player("Gabriel Jesus", name_map) == 29
    # exact single-word match still correctly resolves the real Gabriel
    assert _match_player("Gabriel", name_map) == 5


def test_match_player_token_subset_requires_all_tokens_present():
    # "Nico Silva" shares one token ("nico") with the candidate but not
    # both -- the subset fallback must not match on partial overlap.
    name_map = {"nico gonzalez iglesias": 521}
    assert _match_player("Nico Silva", name_map) is None
