"""sync_played_team — record the team FPL says was played when it differs from
the bot's last logged decision.

Real case, 2026-27 GW6: the manager made the bot's transfers but started Joao
Pedro over Thomas, and a later rerun had logged a squad (Rudoni, Barry) that
was never owned. ``_load_squad_state`` takes the newest lineup row as next
week's squad, so GW7 would have started from players the entry did not have.
"""

from __future__ import annotations

from scripts.sync_played_team import played_team_rows

# element -> internal id is the identity here; the mapping is the caller's job.
POSITIONS = {1: "GKP", 2: "GKP", 3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 7: "DEF",
             8: "MID", 9: "MID", 10: "MID", 11: "MID", 12: "MID",
             13: "FWD", 14: "FWD", 15: "FWD", 16: "FWD"}
XI = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]
BENCH = [2, 6, 7, 12]   # GK first, then outfield priority


def _picks(order=XI + BENCH, captain=10, vice=13, chip=None, bank=10, cost=0):
    return {
        "active_chip": chip,
        "entry_history": {"bank": bank, "event_transfers_cost": cost},
        "picks": [
            {"element": e, "position": i + 1,
             "is_captain": e == captain, "is_vice_captain": e == vice}
            for i, e in enumerate(order)
        ],
    }


def _bot_lineup(**over):
    details = {
        "squad_ids": sorted(XI + BENCH), "starting_ids": sorted(XI),
        "captain_id": 10, "vice_captain_id": 13,
        "positions": {str(k): v for k, v in POSITIONS.items() if k in XI + BENCH},
        "bench_order": {"2": 0, "6": 1, "7": 2, "12": 3},
        "chip_considered": None, "bank": 1.0, "budget": 100.0, "free_transfers": 1,
        "purchase_prices": {str(p): 5.0 for p in XI + BENCH},
    }
    details.update(over)
    return details


PRIOR_PRICES = {p: 5.0 for p in XI + BENCH}


def _rows(picks, transfers=(), bot=None, prior=PRIOR_PRICES):
    return played_team_rows(
        picks=picks, transfers=list(transfers), positions=POSITIONS,
        prior_purchase_prices=prior,
        bot_lineup=_bot_lineup() if bot is None else bot,
    )


def test_nothing_is_written_when_fpl_matches_the_bot():
    assert _rows(_picks()) is None


def test_a_reordered_bench_is_recorded():
    played = _picks(order=XI + [2, 12, 6, 7])
    _, lineup = _rows(played)
    assert lineup["bench_order"] == {"2": 0, "12": 1, "6": 2, "7": 3}


def test_a_swapped_starter_is_recorded():
    """The GW6 case: a bench player started over a bot starter."""
    xi = [p for p in XI if p != 3] + [6]
    _, lineup = _rows(_picks(order=xi + [2, 3, 7, 12]))
    assert sorted(lineup["starting_ids"]) == sorted(xi)
    assert lineup["bench_order"] == {"2": 0, "3": 1, "7": 2, "12": 3}


def test_captaincy_is_recorded():
    _, lineup = _rows(_picks(captain=8, vice=9))
    assert (lineup["captain_id"], lineup["vice_captain_id"]) == (8, 9)


def test_transfers_the_bot_did_not_log_set_the_squad_and_purchase_prices():
    """Next week's sell prices come from these, so they must be what FPL
    charged, not the bot's quoted price."""
    order = [p if p != 15 else 16 for p in XI] + BENCH
    transfers = [{"element_in": 16, "element_in_cost": 61, "element_out": 15,
                  "element_out_cost": 50}]
    t, lineup = _rows(_picks(order=order, bank=1, cost=0), transfers=transfers)
    assert 16 in lineup["squad_ids"] and 15 not in lineup["squad_ids"]
    assert lineup["purchase_prices"]["16"] == 6.1
    assert "15" not in lineup["purchase_prices"]
    assert lineup["bank"] == 0.1
    assert t["transfers_in"] == [{"player_id": 16, "cost": 6.1, "position": "FWD"}]
    assert t["transfers_out"] == [{"player_id": 15, "cost": 5.0, "position": "FWD"}]


def test_hits_come_from_fpl():
    order = [p if p != 15 else 16 for p in XI] + BENCH
    transfers = [{"element_in": 16, "element_in_cost": 50, "element_out": 15,
                  "element_out_cost": 50}]
    t, lineup = _rows(_picks(order=order, cost=4), transfers=transfers)
    assert t["hits_taken"] == lineup["hits_taken"] == 1


def test_a_played_chip_is_recorded_as_the_chip():
    _, lineup = _rows(_picks(chip="bboost"))
    assert lineup["chip_considered"] == "bboost"


def test_a_free_hit_leaves_purchase_prices_on_the_owned_squad():
    """The Free Hit squad is handed back at the deadline. _load_squad_state
    skips Free Hit rows for squad_ids but reads purchase_prices from the
    newest row, so those must stay the OWNED squad's."""
    order = [p if p != 15 else 16 for p in XI] + BENCH
    transfers = [{"element_in": 16, "element_in_cost": 61, "element_out": 15,
                  "element_out_cost": 50}]
    _, lineup = _rows(_picks(order=order, chip="freehit"), transfers=transfers)
    assert lineup["chip_considered"] == "freehit"
    assert lineup["purchase_prices"] == {str(p): 5.0 for p in PRIOR_PRICES}


def test_unrelated_bot_fields_are_carried_and_the_row_is_marked():
    _, lineup = _rows(_picks(captain=8))
    assert lineup["budget"] == 100.0 and lineup["free_transfers"] == 1
    assert lineup["source"] == "fpl_actual"
