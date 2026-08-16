"""simulation/analysis.py — the season read-out for the shadow cohort (P2.5).

The cohort is the project's only validation instrument now that the backtest
has been demoted, so these guard the properties the post-season conclusions
will rest on: paired comparison against the baseline, correct handling of
re-decided gameweeks, and a calibration series that measures the decision
path rather than the projection layer.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import simulation.analysis as analysis
from data.models import Base, SimDecisionLog, SimManager


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'analysis.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(analysis, "get_session", lambda: Local())
    s = Local()
    yield s
    s.close()


def _persona(session, label: str, swept_axis: str, **params) -> int:
    defaults = {
        "risk_level": 0.0, "max_ownership_differential": 0.5,
        "chip_aggressiveness": 1.0, "transfer_switching_cost": 1.5,
        "ft_terminal_value": 2.0, "bench_value_weight": 0.15,
        "transfer_planning_horizon_gws": 3, "mu_baseline": 0.0,
    }
    defaults.update(params)
    row = SimManager(season="2026-27", label=label, swept_axis=swept_axis, **defaults)
    session.add(row)
    session.commit()
    return row.id


def _lineup(session, manager_id, gw, predicted, actual, hits=0):
    session.add(SimDecisionLog(
        sim_manager_id=manager_id, gameweek=gw, decision_type="lineup",
        details=json.dumps({"squad_ids": [1], "hits_taken": hits}),
        projected_gain=predicted, actual_outcome=actual,
    ))
    session.commit()


def test_empty_cohort_returns_empty_frames_not_errors(session):
    assert analysis.persona_season_summary("2026-27").empty
    assert analysis.axis_effect("2026-27").empty
    assert analysis.calibration("2026-27").empty


def test_unscored_gameweeks_are_excluded(session):
    """A gameweek that has not finished has no actual_outcome yet; it must
    not be counted as a zero, which would drag every mean down."""
    pid = _persona(session, "baseline", "baseline")
    _lineup(session, pid, 1, predicted=60.0, actual=55)
    _lineup(session, pid, 2, predicted=62.0, actual=None)

    summary = analysis.persona_season_summary("2026-27")
    assert summary.loc[0, "gws_scored"] == 1
    assert summary.loc[0, "total_actual"] == 55


def test_a_re_decided_gameweek_counts_once(session):
    """Re-running a gameweek appends another lineup row. Only the last is
    the decision that stood -- counting both would double that week."""
    pid = _persona(session, "baseline", "baseline")
    _lineup(session, pid, 1, predicted=60.0, actual=50)
    _lineup(session, pid, 1, predicted=61.0, actual=55)

    summary = analysis.persona_season_summary("2026-27")
    assert summary.loc[0, "gws_scored"] == 1
    assert summary.loc[0, "total_actual"] == 55


def test_delta_vs_baseline_is_paired_per_gameweek(session):
    """The whole point of the cohort: personas share a season, so the
    per-gameweek difference against the control is the low-variance signal,
    not the absolute total."""
    base = _persona(session, "baseline", "baseline")
    swept = _persona(session, "sc=4.0", "transfer_switching_cost",
                     transfer_switching_cost=4.0)
    for gw, (b, s) in enumerate([(50, 55), (60, 62), (40, 38)], start=1):
        _lineup(session, base, gw, predicted=55.0, actual=b)
        _lineup(session, swept, gw, predicted=55.0, actual=s)

    summary = analysis.persona_season_summary("2026-27").set_index("label")
    assert summary.loc["sc=4.0", "delta_vs_baseline"] == (55 - 50) + (62 - 60) + (38 - 40)
    assert summary.loc["sc=4.0", "gws_better_than_baseline"] == 2
    assert summary.loc["baseline", "delta_vs_baseline"] == 0


def test_summary_is_ranked_best_first(session):
    _persona(session, "baseline", "baseline")
    worse = _persona(session, "worse", "mu_baseline", mu_baseline=0.2)
    better = _persona(session, "better", "mu_baseline", mu_baseline=-0.1)
    _lineup(session, worse, 1, predicted=60.0, actual=30)
    _lineup(session, better, 1, predicted=60.0, actual=70)

    summary = analysis.persona_season_summary("2026-27")
    assert list(summary["label"]) == ["better", "worse"]
    assert list(summary["rank"]) == [1, 2]


def test_axis_effect_reports_the_swept_value_not_the_persona_id(session):
    """The post-season question is "which VALUE of this parameter did best",
    so the value has to travel with the result."""
    _persona(session, "baseline", "baseline")
    low = _persona(session, "low", "transfer_switching_cost", transfer_switching_cost=0.0)
    high = _persona(session, "high", "transfer_switching_cost", transfer_switching_cost=4.0)
    _lineup(session, low, 1, predicted=60.0, actual=40)
    _lineup(session, high, 1, predicted=60.0, actual=65)

    effect = analysis.axis_effect("2026-27")
    assert set(effect["swept_axis"]) == {"transfer_switching_cost"}
    assert list(effect["value"]) == [0.0, 4.0]
    assert list(effect["total_actual"]) == [40, 65]


def test_axis_effect_excludes_the_baseline_control(session):
    """The baseline has no swept value, so it is a reference rather than a
    row in any axis's curve."""
    base = _persona(session, "baseline", "baseline")
    swept = _persona(session, "s", "bench_value_weight", bench_value_weight=0.5)
    _lineup(session, base, 1, predicted=60.0, actual=50)
    _lineup(session, swept, 1, predicted=60.0, actual=55)

    assert "baseline" not in set(analysis.axis_effect("2026-27")["swept_axis"])


def test_calibration_measures_signed_bias_across_the_cohort(session):
    """The live instrument for the +7.98 pts/GW over-prediction the backtest
    reported. Mean points are noisy; a mean SIGNED error over ~90 personas
    per gameweek is not."""
    a = _persona(session, "a", "baseline")
    b = _persona(session, "b", "mu_baseline", mu_baseline=0.1)
    _lineup(session, a, 1, predicted=60.0, actual=50)
    _lineup(session, b, 1, predicted=64.0, actual=54)
    _lineup(session, a, 2, predicted=50.0, actual=52)
    _lineup(session, b, 2, predicted=50.0, actual=52)

    cal = analysis.calibration("2026-27").set_index("gameweek")
    assert cal.loc[1, "personas"] == 2
    assert cal.loc[1, "bias"] == pytest.approx(10.0)   # over-predicted
    assert cal.loc[2, "bias"] == pytest.approx(-2.0)   # under-predicted


def test_hits_are_summed_from_the_lineup_details(session):
    pid = _persona(session, "baseline", "baseline")
    _lineup(session, pid, 1, predicted=60.0, actual=50, hits=1)
    _lineup(session, pid, 2, predicted=60.0, actual=50, hits=2)

    summary = analysis.persona_season_summary("2026-27")
    assert summary.loc[0, "hits_taken"] == 3


def test_load_lineup_history_is_scoped_to_the_season(session):
    pid = _persona(session, "baseline", "baseline")
    _lineup(session, pid, 1, predicted=60.0, actual=50)
    other = SimManager(
        season="2027-28", label="x", swept_axis="baseline", risk_level=0.0,
        max_ownership_differential=0.5, chip_aggressiveness=1.0,
        transfer_switching_cost=1.5, ft_terminal_value=2.0,
        bench_value_weight=0.15, transfer_planning_horizon_gws=3, mu_baseline=0.0,
    )
    session.add(other)
    session.commit()
    _lineup(session, other.id, 1, predicted=99.0, actual=99)

    history = analysis.load_lineup_history("2026-27")
    assert list(history["sim_manager_id"]) == [pid]


def test_summary_columns_survive_a_cohort_with_no_baseline(session):
    """Defensive: if the control is ever missing, the paired columns are
    simply absent rather than the whole read-out failing."""
    swept = _persona(session, "s", "mu_baseline", mu_baseline=0.1)
    _lineup(session, swept, 1, predicted=60.0, actual=50)

    summary = analysis.persona_season_summary("2026-27")
    assert not summary.empty
    assert summary.loc[0, "total_actual"] == 50


def test_pandas_frames_are_returned_not_none(session):
    """Callers (the dashboard) index these directly."""
    pid = _persona(session, "baseline", "baseline")
    _lineup(session, pid, 1, predicted=60.0, actual=50)
    for frame in (
        analysis.load_lineup_history("2026-27"),
        analysis.persona_season_summary("2026-27"),
        analysis.calibration("2026-27"),
    ):
        assert isinstance(frame, pd.DataFrame)
