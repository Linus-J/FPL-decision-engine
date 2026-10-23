"""The raw input archive: captures are kept exactly as received, and altering
one is detectable."""

import json
from datetime import datetime

import pytest

from data.archive import MANIFEST_NAME, verify_input_archive, write_input_archive

BOOTSTRAP = {"teams": [{"id": 1, "name": "Arsenal"}], "elements": [{"id": 10, "status": "a"}]}
FIXTURES = [{"id": 5, "event": 6, "finished": False}]


def test_capture_round_trips_raw_payloads(tmp_path):
    captured = datetime(2026, 10, 10, 1, 26, 12, 275723)
    out = write_input_archive(
        "2026-27",
        captured,
        {"bootstrap": BOOTSTRAP, "fixtures": FIXTURES},
        gameweek_context=6,
        root=tmp_path,
    )
    assert out == tmp_path / "2026-27" / "20261010T012612275723Z"
    assert json.loads((out / "bootstrap.json").read_text()) == BOOTSTRAP
    assert json.loads((out / "fixtures.json").read_text()) == FIXTURES


def test_manifest_records_context_and_hashes(tmp_path):
    out = write_input_archive(
        "2026-27", datetime(2026, 10, 10), {"bootstrap": BOOTSTRAP},
        gameweek_context=6, root=tmp_path,
    )
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["season"] == "2026-27"
    assert manifest["gameweek_context"] == 6
    assert set(manifest["files"]) == {"bootstrap"}
    assert len(manifest["files"]["bootstrap"]["sha256"]) == 64


def test_fresh_capture_verifies(tmp_path):
    out = write_input_archive(
        "2026-27", datetime(2026, 10, 10), {"bootstrap": BOOTSTRAP}, 6, root=tmp_path
    )
    assert verify_input_archive(out)


def test_altered_file_fails_verification(tmp_path):
    out = write_input_archive(
        "2026-27", datetime(2026, 10, 10), {"bootstrap": BOOTSTRAP}, 6, root=tmp_path
    )
    (out / "bootstrap.json").write_text('{"teams": []}')
    assert not verify_input_archive(out)


def test_missing_file_fails_verification(tmp_path):
    out = write_input_archive(
        "2026-27", datetime(2026, 10, 10), {"bootstrap": BOOTSTRAP}, 6, root=tmp_path
    )
    (out / "bootstrap.json").unlink()
    assert not verify_input_archive(out)


def test_same_timestamp_is_refused_not_overwritten(tmp_path):
    ts = datetime(2026, 10, 10, 12, 0, 0)
    write_input_archive("2026-27", ts, {"bootstrap": BOOTSTRAP}, 6, root=tmp_path)
    with pytest.raises(FileExistsError):
        write_input_archive("2026-27", ts, {"bootstrap": {"teams": []}}, 6, root=tmp_path)


def test_empty_capture_is_refused(tmp_path):
    with pytest.raises(ValueError):
        write_input_archive("2026-27", datetime(2026, 10, 10), {}, 6, root=tmp_path)
