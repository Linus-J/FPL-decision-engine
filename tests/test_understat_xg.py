"""Understat per-match xG ingest — pure parsers + key-passes aggregation."""

from __future__ import annotations

from datetime import datetime

import pytest

from data.ingestors import understat_xg
from data.ingestors.fbref import aggregate_xg_rows
from data.ingestors.understat_xg import parse_game_date, understat_row_to_xg


def test_parse_game_date():
    assert parse_game_date("2025-08-15 Liverpool-Bournemouth") == datetime(2025, 8, 15)
    assert parse_game_date("") is None
    assert parse_game_date("not a date") is None


def test_understat_row_to_xg():
    out = understat_row_to_xg({"xg": 0.42, "xa": 0.18, "shots": 3, "key_passes": 2})
    assert out == {"xg": 0.42, "npxg": 0.42, "xa": 0.18, "shots": 3, "key_passes": 2}
    # npxg mirrors xg (no penalty split); missing fields → 0
    assert understat_row_to_xg({"xg": 0.5}) == {
        "xg": 0.5, "npxg": 0.5, "xa": 0.0, "shots": 0, "key_passes": 0
    }


def test_aggregate_sums_key_passes_dgw():
    per_match = [
        (1, 5, {"xg": 0.4, "xa": 0.1, "npxg": 0.4, "shots": 2, "key_passes": 1}),
        (1, 5, {"xg": 0.2, "xa": 0.3, "npxg": 0.2, "shots": 1, "key_passes": 2}),
    ]
    agg = aggregate_xg_rows(per_match)
    assert agg[(1, 5)]["key_passes"] == 3
    assert agg[(1, 5)]["xg"] == 0.6
    assert agg[(1, 5)]["xgi"] == 1.0   # xg 0.6 + xa 0.4


# --- real non-penalty xG (2026-08-16) ------------------------------------
#
# npxg used to be stored equal to xg because the player-match feed has no
# penalty split. Anything treating the pair as a decomposition (non-penalty
# xG + penalty duty) therefore double-counted a taker's penalties — which is
# exactly what projection/assemble.py briefly did. The shot-event feed does
# carry the split.


def test_explicit_penalty_situation_is_recognised():
    """Trusted outright, so this keeps working if soccerdata starts labelling
    penalties instead of nulling them."""
    from data.ingestors.understat_xg import is_penalty_shot

    assert is_penalty_shot("Penalty", 0.7612) is True
    assert is_penalty_shot("penalty", 0.5) is True


def test_null_situation_at_the_penalty_price_is_a_penalty():
    """soccerdata's situation mapping has no Penalty label, so Understat's
    penalties arrive NULL. Verified against 2025-26: all 92 null-situation
    shots carry this xG, and no other shot has a null situation."""
    from data.ingestors.understat_xg import is_penalty_shot

    assert is_penalty_shot(None, 0.7612) is True
    assert is_penalty_shot(float("nan"), 0.7611) is True
    assert is_penalty_shot("", 0.7612) is True


def test_a_null_situation_at_an_ordinary_xg_is_not_a_penalty():
    """The guard that matters. If a null appears for some other reason, the
    shot falls through as open play rather than silently stripping real xG
    out of a player's non-penalty total."""
    from data.ingestors.understat_xg import is_penalty_shot

    assert is_penalty_shot(None, 0.05) is False
    assert is_penalty_shot(None, 0.95) is False


def test_ordinary_shots_are_never_penalties():
    from data.ingestors.understat_xg import is_penalty_shot

    assert is_penalty_shot("Open Play", 0.7612) is False
    assert is_penalty_shot("From Corner", 0.3) is False


def test_aggregate_npxg_sums_only_non_penalty_shots():
    from data.ingestors.understat_xg import aggregate_npxg

    rows = [
        {"player_id": 1, "game_id": 9, "situation": "Open Play", "xg": 0.20},
        {"player_id": 1, "game_id": 9, "situation": None, "xg": 0.7612},  # penalty
        {"player_id": 1, "game_id": 9, "situation": "From Corner", "xg": 0.10},
        {"player_id": 2, "game_id": 9, "situation": "Open Play", "xg": 0.40},
    ]
    out = aggregate_npxg(rows)
    assert out[(1, 9)] == pytest.approx(0.30), "the penalty is excluded"
    assert out[(2, 9)] == pytest.approx(0.40)


def test_npxg_is_used_when_supplied():
    from data.ingestors.understat_xg import understat_row_to_xg

    row = understat_row_to_xg({"xg": 1.0, "xa": 0.1, "shots": 3, "key_passes": 1}, npxg=0.25)
    assert row["xg"] == pytest.approx(1.0)
    assert row["npxg"] == pytest.approx(0.25)


def test_npxg_never_exceeds_total_xg():
    """If the shot-event sum disagrees with the match feed's own total, the
    total is authoritative — npxg above xg would make the penalty component
    negative downstream."""
    from data.ingestors.understat_xg import understat_row_to_xg

    row = understat_row_to_xg({"xg": 0.5, "xa": 0.0, "shots": 1, "key_passes": 0}, npxg=0.9)
    assert row["npxg"] == pytest.approx(0.5)


def test_missing_shot_feed_falls_back_to_total_xg():
    """Documented degradation, not silent: the caller logs it and the weekly
    copied-column check flags the result."""
    from data.ingestors.understat_xg import understat_row_to_xg

    row = understat_row_to_xg({"xg": 0.8, "xa": 0.0, "shots": 2, "key_passes": 0}, npxg=None)
    assert row["npxg"] == pytest.approx(0.8)


def test_a_player_who_took_no_shots_gets_zero_not_total_xg():
    """Absent from the shot aggregation means no shots, which is genuinely
    0.0 non-penalty xG — not 'unknown, use the total'."""
    from data.ingestors.understat_xg import understat_row_to_xg

    row = understat_row_to_xg({"xg": 0.0, "xa": 0.3, "shots": 0, "key_passes": 4}, npxg=0.0)
    assert row["npxg"] == pytest.approx(0.0)


def test_unpublished_season_is_a_no_op_not_a_crash(monkeypatch, caplog):
    """Understat lags the season start. Until it publishes, read_schedule
    returns a DEGENERATE frame -- no columns, one unnamed index level holding
    the literal strings 'league', 'season', 'game' -- and soccerdata's own
    reader raises "too many values to unpack (expected 3)" on it (confirmed
    for 2026-27 on 2026-08-25). run_weekly.py calls this every week from GW1,
    so it must read as "source not live yet", not as a broken pipeline.
    """
    import pandas as pd

    degenerate = pd.DataFrame(index=pd.Index(["league", "season", "game"]))

    class _Us:
        def read_schedule(self):
            return degenerate

        def read_player_match_stats(self):  # pragma: no cover - must not run
            raise AssertionError("must not read stats for an unpublished season")

    assert understat_xg._season_is_published(_Us()) is False


def test_published_season_passes_the_check():
    import pandas as pd

    idx = pd.MultiIndex.from_tuples(
        [("ENG-Premier League", "2425", "game one")],
        names=["league", "season", "game"],
    )

    class _Us:
        def read_schedule(self):
            return pd.DataFrame({"date": ["2024-08-16"]}, index=idx)

    assert understat_xg._season_is_published(_Us()) is True


def test_unreadable_schedule_raises_rather_than_claiming_unpublished():
    """Reversal of a deliberate earlier choice (2026-08-28).

    This used to swallow every exception and return False, so the caller
    reported "it is a source lag, not a failure. Prior seasons are
    unaffected." -- a definitive claim built on an unknown. It was wrong on
    the GW2 deadline: two runs reported 2026-27 unpublished while the season
    was live and player_xg_stats sat at 2 non-zero xg rows out of 309, with
    nothing in the log suggesting anything was wrong. A read that fails and a
    season that genuinely has not started are different states and must not
    share a message.
    """
    class _Us:
        def read_schedule(self):
            raise RuntimeError("network is down")

    with pytest.raises(understat_xg.UnderstatScheduleUnreadable):
        understat_xg._season_is_published(_Us(), sleep_seconds=0.0)


def test_schedule_read_is_retried_before_giving_up():
    """The observed failure was transient -- it read fine an hour later."""
    import pandas as pd

    idx = pd.MultiIndex.from_tuples(
        [("ENG-Premier League", "2627", "game one")],
        names=["league", "season", "game"],
    )

    class _Us:
        def __init__(self):
            self.calls = 0

        def read_schedule(self):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("transient TLS failure")
            return pd.DataFrame({"date": ["2026-08-21"]}, index=idx)

    us = _Us()
    assert understat_xg._season_is_published(us, sleep_seconds=0.0) is True
    assert us.calls == 2


def test_retries_are_bounded():
    class _Us:
        def __init__(self):
            self.calls = 0

        def read_schedule(self):
            self.calls += 1
            raise RuntimeError("still down")

    us = _Us()
    with pytest.raises(understat_xg.UnderstatScheduleUnreadable):
        understat_xg._season_is_published(us, attempts=3, sleep_seconds=0.0)
    assert us.calls == 3
