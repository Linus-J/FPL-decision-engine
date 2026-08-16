"""simulation/personas.py — the P2.4 one-factor-at-a-time cohort.

Replaces the tests for the original random sweep. The properties that matter
are different now: the cohort is a designed experiment, not a sample, so what
needs guarding is that each persona isolates exactly one axis and that the
baseline really is the real bot's configuration.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.strategy import OPTIMISER, TRANSFERS
from data.models import Base, SimManager
from simulation.personas import (
    SIM_COUNT,
    SWEPT_AXES,
    generate_personas,
    load_or_create_personas,
)


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'personas.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    s = Local()
    yield s
    s.close()


_PARAM_FIELDS = (
    "risk_level",
    "max_ownership_differential",
    "chip_aggressiveness",
    "transfer_switching_cost",
    "ft_terminal_value",
    "bench_value_weight",
    "transfer_planning_horizon_gws",
    "mu_baseline",
)


def test_generate_personas_is_deterministic():
    assert generate_personas("2026-27") == generate_personas("2026-27")


def test_cohort_size_matches_the_declared_design():
    personas = generate_personas("2026-27")
    assert len(personas) == SIM_COUNT
    assert SIM_COUNT == 1 + sum(len(v) for v in SWEPT_AXES.values())


def test_labels_are_unique_so_no_persona_is_a_duplicate():
    """A duplicated persona is wasted cohort -- it costs a full season's
    compute and contributes no new information."""
    labels = [p["label"] for p in generate_personas("2026-27")]
    assert len(set(labels)) == len(labels)


def test_baseline_persona_is_exactly_the_real_bot_configuration():
    """The control. Its decisions should match the real bot's, which makes
    it both the reference every swept persona is read against and a live
    check that the simulation path has not drifted from the real one."""
    baseline = generate_personas("2026-27")[0]
    assert baseline["swept_axis"] == "baseline"
    assert baseline["risk_level"] == OPTIMISER.risk_level
    assert baseline["bench_value_weight"] == OPTIMISER.bench_value_weight
    assert baseline["mu_baseline"] == OPTIMISER.mu_baseline
    assert baseline["transfer_planning_horizon_gws"] == OPTIMISER.transfer_planning_horizon_gws
    assert baseline["transfer_switching_cost"] == TRANSFERS.transfer_switching_cost
    assert baseline["ft_terminal_value"] == TRANSFERS.ft_terminal_value


def test_every_swept_persona_differs_from_the_baseline_in_exactly_one_axis():
    """The defining property of a one-factor-at-a-time design. If a persona
    varied two things at once its result could not be attributed to either,
    and a single noisy season has no power to disentangle them."""
    personas = generate_personas("2026-27")
    baseline = personas[0]
    for persona in personas[1:]:
        differing = {
            field for field in _PARAM_FIELDS if persona[field] != baseline[field]
        }
        assert differing <= {persona["swept_axis"]}, (
            f"{persona['label']} varies {differing}, not just its own axis"
        )


def test_each_axis_actually_moves_off_the_default_somewhere():
    """A sweep whose range happens to sit entirely on the default value
    would look fine but measure nothing."""
    personas = generate_personas("2026-27")
    baseline = personas[0]
    for axis in SWEPT_AXES:
        values = {p[axis] for p in personas if p["swept_axis"] == axis}
        assert values - {baseline[axis]}, f"{axis} never leaves its default"


def test_planning_horizon_never_exceeds_what_the_pipeline_persists():
    """Personas read the persisted projection frame rather than building
    their own, so a horizon beyond it silently does nothing -- the persona
    would look like a distinct experiment and quietly be a duplicate."""
    personas = generate_personas("2026-27")
    horizons = [
        p["transfer_planning_horizon_gws"]
        for p in personas
        if p["swept_axis"] == "transfer_planning_horizon_gws"
    ]
    assert horizons, "the horizon axis must actually be swept"
    assert max(horizons) <= OPTIMISER.projection_horizon_gws


def test_load_or_create_personas_persists_once(session):
    first = load_or_create_personas(session, "2026-27")
    assert len(first) == SIM_COUNT
    assert session.query(SimManager).filter_by(season="2026-27").count() == SIM_COUNT

    second = load_or_create_personas(session, "2026-27")
    assert [p.id for p in second] == [p.id for p in first]
    assert session.query(SimManager).filter_by(season="2026-27").count() == SIM_COUNT


def test_load_or_create_personas_never_regenerates_mid_season(session):
    """Persona identity and configuration must stay fixed for a whole
    season, or its decision history stops being interpretable."""
    first = load_or_create_personas(session, "2026-27")
    second = load_or_create_personas(session, "2026-27", seed=999)
    assert [p.id for p in second] == [p.id for p in first]


def test_load_or_create_personas_scoped_per_season(session):
    load_or_create_personas(session, "2026-27")
    load_or_create_personas(session, "2027-28")
    assert session.query(SimManager).filter_by(season="2026-27").count() == SIM_COUNT
    assert session.query(SimManager).filter_by(season="2027-28").count() == SIM_COUNT


def test_every_swept_axis_is_readable_by_the_season_analysis():
    """The cross-module invariant that would only bite at the END of a season.

    simulation/analysis.py finds a persona's setting by indexing the joined
    lineup frame BY THE AXIS NAME, and that frame comes from _LINEUP_QUERY.
    An axis that is a SimManager column but is NOT selected by that query
    would raise months later, when the data is finally read — so tie the two
    together here.
    """
    from simulation.analysis import _LINEUP_QUERY

    for axis in SWEPT_AXES:
        assert f"m.{axis}" in _LINEUP_QUERY, (
            f"{axis} is swept but not selected by the analysis query"
        )


def test_swept_axes_are_real_simmanager_columns():
    """Asserted at import too; pinned here so the reason survives."""
    from data.models import SimManager

    columns = {c.name for c in SimManager.__table__.columns}
    assert set(SWEPT_AXES) <= columns
