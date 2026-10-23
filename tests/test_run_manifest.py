"""Run manifests: the configuration is recorded in full, secrets are not, and
the id of the run in progress reaches the rows written during it."""

import json

import config.settings as settings_module
import config.strategy as strategy
from data.db import get_session
from data.models import RunManifest
from data.run_manifest import (
    current_manifest_id,
    manifest_context,
    record_run,
    strategy_config,
)


def test_strategy_config_holds_the_optimiser_values():
    config = strategy_config()
    assert config["OPTIMISER"]["risk_level"] == strategy.OPTIMISER.risk_level
    assert config["OPTIMISER"]["mu_baseline"] == strategy.OPTIMISER.mu_baseline


def test_strategy_config_excludes_settings_and_secrets():
    # Settings hold the team id and API credentials. The manifest records the
    # strategy that determined a decision, and nothing else.
    flat = json.dumps(strategy_config())
    for name, value in settings_module.settings.model_dump().items():
        assert f'"{name}"' not in flat
        if isinstance(value, str) and len(value) >= 8:
            assert value not in flat


def test_manifest_context_sets_and_restores_the_id():
    assert current_manifest_id() is None
    with manifest_context(42):
        assert current_manifest_id() == 42
        with manifest_context(7):
            assert current_manifest_id() == 7
        assert current_manifest_id() == 42
    assert current_manifest_id() is None


def test_record_run_persists_config_and_hash():
    manifest_id = record_run(
        "backtest", season="2026-27", seed=42, notes="unit test",
        config={"OPTIMISER": {"risk_level": 0.0}},
    )
    db = get_session()
    try:
        row = db.get(RunManifest, manifest_id)
        assert row.kind == "backtest"
        assert row.seed == 42
        assert json.loads(row.config_json) == {"OPTIMISER": {"risk_level": 0.0}}
        assert len(row.config_sha256) == 64
    finally:
        db.close()


def test_identical_config_gives_identical_hash():
    a = record_run("backtest", season="2026-27", config={"A": 1, "B": 2})
    b = record_run("backtest", season="2026-27", config={"B": 2, "A": 1})
    db = get_session()
    try:
        assert db.get(RunManifest, a).config_sha256 == db.get(RunManifest, b).config_sha256
    finally:
        db.close()


def test_changed_config_gives_different_hash():
    a = record_run("backtest", season="2026-27", config={"A": 1})
    b = record_run("backtest", season="2026-27", config={"A": 2})
    db = get_session()
    try:
        assert db.get(RunManifest, a).config_sha256 != db.get(RunManifest, b).config_sha256
    finally:
        db.close()


def test_decision_rows_carry_the_run_manifest_id():
    import agent.decision_engine as decision_engine
    from data.models import DecisionLog

    manifest_id = record_run("decision", season="2026-27", notes="threading test")
    with manifest_context(manifest_id):
        decision_engine._record_decision(
            None, gameweek=999, decision_type="manifest_threading_test", details={}
        )
    db = get_session()
    try:
        row = (
            db.query(DecisionLog)
            .filter(DecisionLog.decision_type == "manifest_threading_test")
            .one()
        )
        assert row.manifest_id == manifest_id
    finally:
        db.close()


def test_decision_rows_outside_a_run_have_no_manifest():
    import agent.decision_engine as decision_engine
    from data.models import DecisionLog

    decision_engine._record_decision(
        None, gameweek=998, decision_type="unmanifested_test", details={}
    )
    db = get_session()
    try:
        row = db.query(DecisionLog).filter(DecisionLog.decision_type == "unmanifested_test").one()
        assert row.manifest_id is None
    finally:
        db.close()
