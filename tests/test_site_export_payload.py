"""scripts/site_export/payload.py"""

from __future__ import annotations

import json
import math
from datetime import datetime

import dashboard.data.squad as squad_module
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import Base, DecisionLog, Gameweek, Player, ProjectionSample, Team
from scripts.site_export import payload as payload_module


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'export.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


def _add_samples(session, player_id: int, gw: int, season: str, values: list[float], created_at: datetime | None = None) -> None:
    session.add_all([
        ProjectionSample(player_id=player_id, gameweek=gw, season=season, scenario_id=i, xpts=v, created_at=created_at)
        for i, v in enumerate(values)
    ])
    session.commit()


def test_get_projection_distributions_returns_summary_per_player(session):
    batch_ts = datetime(2026, 8, 1, 6, 0)
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[2.0, 4.0, 6.0, 8.0, 10.0], created_at=batch_ts)
    _add_samples(session, player_id=2, gw=3, season="2026-27", values=[1.0, 1.0, 1.0, 1.0, 1.0], created_at=batch_ts)

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert set(dist.keys()) == {1, 2}
    assert dist[1]["mean"] == 6.0
    assert dist[1]["median"] == 6.0
    assert dist[2]["mean"] == 1.0
    assert dist[2]["p10"] == 1.0
    assert dist[2]["p90"] == 1.0


def test_get_projection_distributions_ignores_other_gameweeks_and_seasons(session):
    batch_ts = datetime(2026, 8, 1, 6, 0)
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[5.0, 5.0], created_at=batch_ts)
    _add_samples(session, player_id=1, gw=4, season="2026-27", values=[99.0], created_at=batch_ts)
    _add_samples(session, player_id=1, gw=3, season="2025-26", values=[99.0], created_at=batch_ts)

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert dist[1]["mean"] == 5.0


def test_get_projection_distributions_empty_when_no_samples(session):
    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")
    assert dist == {}


def test_get_projection_distributions_uses_only_latest_batch(session):
    batch1_ts = datetime(2026, 8, 1, 6, 0)
    batch2_ts = datetime(2026, 8, 3, 6, 0)
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[1.0, 1.0, 4.0], created_at=batch1_ts)
    _add_samples(session, player_id=1, gw=3, season="2026-27", values=[7.0, 8.0, 9.0], created_at=batch2_ts)

    dist = payload_module.get_projection_distributions(session, gw=3, season="2026-27")

    assert dist[1]["mean"] == 8.0


def _add_team(session, team_id: int, short_name: str) -> None:
    session.add(Team(id=team_id, name=short_name, short_name=short_name))
    session.commit()


def test_team_short_names_maps_id_to_short_name(session):
    _add_team(session, 1, "ARS")
    _add_team(session, 2, "MCI")

    names = payload_module._team_short_names(session)

    assert names == {1: "ARS", 2: "MCI"}


def test_label_for_gw_uses_deadline_time(session):
    session.add(Gameweek(
        id=3, season="2026-27", name="Gameweek 3",
        deadline_time=datetime(2026, 8, 3, 10, 30),
    ))
    session.commit()

    label = payload_module._label_for_gw(session, "2026-27", 3)

    assert label == "GW3 — 3 Aug 2026"


def test_label_for_gw_falls_back_when_gameweek_missing(session):
    label = payload_module._label_for_gw(session, "2026-27", 99)
    assert label == "GW99"


def test_xpts_entry_uses_distribution_when_available():
    dist = {1: {"p10": 1.0, "median": 2.0, "mean": 2.5, "p90": 4.0}}
    assert payload_module._xpts_entry(1, dist, fallback_mean=99.0) == dist[1]


def test_xpts_entry_falls_back_to_flat_mean_when_no_samples():
    entry = payload_module._xpts_entry(1, {}, fallback_mean=5.0)
    assert entry == {"p10": 5.0, "median": 5.0, "mean": 5.0, "p90": 5.0}


def test_xpts_entry_returns_none_when_nothing_available():
    assert payload_module._xpts_entry(1, {}, fallback_mean=None) is None
    assert payload_module._xpts_entry(1, {}, fallback_mean=math.nan) is None


def test_build_squad_entries_orders_bench_gk_first_then_by_xpts():
    squad_df = pd.DataFrame([
        {"player_id": 1, "web_name": "Starter", "position": "FWD", "team_short": "MCI",
         "now_cost": 10.0, "is_starting": True, "is_captain": True, "is_vice_captain": False,
         "xpts": 8.0},
        {"player_id": 2, "web_name": "BenchGK", "position": "GKP", "team_short": "ARS",
         "now_cost": 4.5, "is_starting": False, "is_captain": False, "is_vice_captain": False,
         "xpts": 0.5},
        {"player_id": 3, "web_name": "BenchLow", "position": "DEF", "team_short": "ARS",
         "now_cost": 4.5, "is_starting": False, "is_captain": False, "is_vice_captain": False,
         "xpts": 1.0},
        {"player_id": 4, "web_name": "BenchHigh", "position": "MID", "team_short": "ARS",
         "now_cost": 6.0, "is_starting": False, "is_captain": False, "is_vice_captain": False,
         "xpts": 3.0},
    ])

    entries = payload_module._build_squad_entries(squad_df, dist={})

    by_id = {e["player_id"]: e for e in entries}
    assert by_id[1]["bench_order"] is None
    assert by_id[2]["bench_order"] == 1   # GK bench slot always first (despite lowest xPts)
    assert by_id[4]["bench_order"] == 2   # then outfield by xPts descending
    assert by_id[3]["bench_order"] == 3
    assert by_id[1]["xpts"] == {"p10": 8.0, "median": 8.0, "mean": 8.0, "p90": 8.0}


def test_build_top15_entries_takes_first_15_and_maps_team_short():
    projections_df = pd.DataFrame([
        {"player_id": i, "web_name": f"P{i}", "position": "MID", "team_id": 1, "xpts_mean": 20.0 - i}
        for i in range(20)
    ])
    team_names = {1: "ARS"}

    entries = payload_module._build_top15_entries(projections_df, dist={}, team_names=team_names)

    assert len(entries) == 15
    assert entries[0]["player_id"] == 0
    assert entries[0]["team_short"] == "ARS"
    assert entries[0]["xpts"]["mean"] == 20.0


def test_build_history_entries_maps_transfers_and_chips_and_drops_lineup():
    history_df = pd.DataFrame([
        {"gameweek": 3, "decision_type": "transfers", "projected_gain": 1.4, "details": {
            "transfers_in": [{"player_id": 1, "web_name": "Haaland", "cost": 15.1}],
            "transfers_out": [{"player_id": 2, "web_name": "Wilson", "cost": 6.5}],
            "hits_taken": 0,
        }},
        {"gameweek": 3, "decision_type": "chip", "projected_gain": 0.0, "details": {
            "chip": "wildcard", "reason": "squad overhaul",
        }},
        {"gameweek": 3, "decision_type": "lineup", "projected_gain": 55.0, "details": {
            "squad_ids": [1, 2],
        }},
    ])

    entries = payload_module._build_history_entries(history_df)

    assert len(entries) == 2
    assert entries[0] == {
        "gameweek": 3, "type": "transfers",
        "transfers_in": ["Haaland"], "transfers_out": ["Wilson"],
        "hits_taken": 0, "net_xpts_gain": 1.4,
    }
    assert entries[1] == {
        "gameweek": 3, "type": "chip", "chip": "wildcard", "reason": "squad overhaul",
    }


def _seed_full_squad(session):
    session.add(Team(id=1, name="Man City", short_name="MCI"))
    session.add(Team(id=2, name="Arsenal", short_name="ARS"))
    session.add_all([
        Player(id=1, fpl_id=101, code=101, first_name="E", second_name="Haaland",
               web_name="Haaland", team_id=1, position="FWD", now_cost=15.1),
        Player(id=2, fpl_id=102, code=102, first_name="C", second_name="Wilson",
               web_name="Wilson", team_id=2, position="FWD", now_cost=6.5),
    ])
    session.add(DecisionLog(
        gameweek=3, decision_type="lineup",
        details=json.dumps({
            "squad_ids": [1, 2], "starting_ids": [1],
            "captain_id": 1, "vice_captain_id": 2,
        }),
        projected_gain=8.0, dry_run=True,
    ))
    session.add(DecisionLog(
        gameweek=3, decision_type="transfers",
        details=json.dumps({
            "transfers_in": [{"player_id": 1, "web_name": "Haaland", "cost": 15.1}],
            "transfers_out": [{"player_id": 3, "web_name": "Old", "cost": 6.0}],
            "hits_taken": 0,
        }),
        projected_gain=1.4, dry_run=True,
    ))
    session.add(Gameweek(
        id=3, season="2026-27", name="Gameweek 3", deadline_time=datetime(2026, 8, 3, 10, 30),
    ))
    session.add_all([
        ProjectionSample(
            player_id=1, gameweek=3, season="2026-27", scenario_id=i, xpts=v,
            created_at=datetime(2026, 8, 3, 6, 0),
        )
        for i, v in enumerate([6.0, 8.0, 10.0])
    ])
    session.commit()


def test_build_run_payload_assembles_full_schema(session, monkeypatch):
    _seed_full_squad(session)

    monkeypatch.setattr(squad_module, "_get_current_and_next_gw", lambda: (3, 3))
    monkeypatch.setattr(squad_module, "get_picks", lambda team_id, gw: {})
    monkeypatch.setattr(
        squad_module, "get_latest_projections",
        lambda gw: pd.DataFrame({"player_id": [1, 2], "xpts": [8.0, 2.0]}),
    )
    monkeypatch.setattr(payload_module, "_get_current_season", lambda: "2026-27")
    monkeypatch.setattr(
        payload_module, "get_latest_projections",
        lambda gw: pd.DataFrame([
            {"player_id": 1, "web_name": "Haaland", "position": "FWD", "team_id": 1,
             "xpts_mean": 8.0},
            {"player_id": 2, "web_name": "Wilson", "position": "FWD", "team_id": 2,
             "xpts_mean": 2.0},
        ]),
    )

    payload = payload_module.build_run_payload(session, team_id=12345)

    assert payload["schema_version"] == 1
    assert payload["gameweek"] == 3
    assert payload["label"] == "GW3 — 3 Aug 2026"
    assert len(payload["squad"]) == 2
    assert len(payload["top15"]) == 2
    haaland_squad_entry = next(e for e in payload["squad"] if e["player_id"] == 1)
    assert haaland_squad_entry["xpts"]["mean"] == 8.0  # from real projection_samples rows
    assert len(payload["history"]) == 1
    assert payload["history"][0]["type"] == "transfers"
    json.dumps(payload)  # must be JSON-serializable end to end


def test_build_run_payload_raises_when_no_squad_available(session):
    with pytest.raises(RuntimeError, match="No current squad"):
        payload_module.build_run_payload(session, team_id=12345)


# --- projected-points spread (2026-08-16) --------------------------------
#
# p10/median/mean/p90 used to collapse to a single value whenever
# projection_samples had no draws — which is every pre-season gameweek, since
# the cold start produces none. The site showed a zero-width bar for every
# player, implying a certainty that does not exist.


def test_real_monte_carlo_quantiles_take_precedence():
    from scripts.site_export.payload import _xpts_entry

    real = {"p10": 1.0, "median": 4.0, "mean": 4.5, "p90": 9.0}
    out = _xpts_entry(7, {7: real}, fallback_mean=99.0, fallback_var=99.0)
    assert out == real


def test_spread_is_derived_from_variance_when_samples_are_missing():
    from scripts.site_export.payload import _xpts_entry

    out = _xpts_entry(7, {}, fallback_mean=5.0, fallback_var=4.0)  # sd = 2.0
    assert out["mean"] == pytest.approx(5.0)
    assert out["median"] == pytest.approx(5.0)
    assert out["p10"] < out["mean"] < out["p90"], "the bar must have real width"
    # symmetric about the mean by construction (normal approximation)
    assert (out["p90"] - out["mean"]) == pytest.approx(out["mean"] - out["p10"])


def test_a_higher_variance_gives_a_wider_spread():
    from scripts.site_export.payload import _xpts_entry

    tight = _xpts_entry(7, {}, fallback_mean=5.0, fallback_var=1.0)
    loose = _xpts_entry(7, {}, fallback_mean=5.0, fallback_var=9.0)
    assert (loose["p90"] - loose["p10"]) > (tight["p90"] - tight["p10"])


def test_p10_is_floored_at_zero():
    """A negative score needs a card or own goal, which is far rarer than a
    symmetric normal implies down there."""
    from scripts.site_export.payload import _xpts_entry

    out = _xpts_entry(7, {}, fallback_mean=0.5, fallback_var=9.0)
    assert out["p10"] == 0.0


def test_zero_variance_still_collapses_to_a_point():
    """Genuinely no spread information is not the same as inventing one."""
    from scripts.site_export.payload import _xpts_entry

    out = _xpts_entry(7, {}, fallback_mean=3.0, fallback_var=0.0)
    assert out == {"p10": 3.0, "median": 3.0, "mean": 3.0, "p90": 3.0}


def test_no_projection_at_all_returns_none():
    from scripts.site_export.payload import _xpts_entry

    assert _xpts_entry(7, {}, fallback_mean=None, fallback_var=1.0) is None
