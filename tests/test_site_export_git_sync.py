"""scripts/site_export/git_sync.py"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.site_export import git_sync


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("init\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


@pytest.fixture
def repo(tmp_path):
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    return repo_root


def test_commit_and_push_commits_new_data_without_pushing(repo):
    data_dir = repo / "data" / "simulations"
    data_dir.mkdir(parents=True)
    (data_dir / "gw3.json").write_text("{}")

    committed = git_sync.commit_and_push(repo, data_dir, "export: GW3 site data", push=False)

    assert committed is True
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"], cwd=repo, check=True, capture_output=True, text=True,
    )
    assert log.stdout.strip() == "export: GW3 site data"


def test_commit_and_push_returns_false_when_nothing_changed(repo):
    data_dir = repo / "data" / "simulations"
    data_dir.mkdir(parents=True)
    (data_dir / "gw3.json").write_text("{}")
    git_sync.commit_and_push(repo, data_dir, "export: GW3 site data", push=False)

    committed_again = git_sync.commit_and_push(
        repo, data_dir, "export: GW3 site data (rerun)", push=False
    )

    assert committed_again is False


def test_commit_and_push_pushes_to_remote_when_push_true(tmp_path, repo):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True)
    subprocess.run(["git", "push", "-q", "-u", "origin", "main"], cwd=repo, check=True)

    data_dir = repo / "data" / "simulations"
    data_dir.mkdir(parents=True)
    (data_dir / "gw3.json").write_text("{}")

    committed = git_sync.commit_and_push(repo, data_dir, "export: GW3 site data", push=True)

    assert committed is True
    remote_log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s", "main"],
        cwd=remote, check=True, capture_output=True, text=True,
    )
    assert remote_log.stdout.strip() == "export: GW3 site data"


def test_commit_and_push_does_not_commit_unrelated_staged_changes(repo):
    """Regression test: unrelated staged changes must not be swept into the commit."""
    data_dir = repo / "data" / "simulations"
    data_dir.mkdir(parents=True)
    (data_dir / "gw3.json").write_text("{}")

    unrelated = repo / "unrelated.txt"
    unrelated.write_text("unrelated content\n")

    subprocess.run(["git", "add", "unrelated.txt"], cwd=repo, check=True)

    committed = git_sync.commit_and_push(repo, data_dir, "export: GW3 site data", push=False)

    assert committed is True

    diff_names = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    committed_files = diff_names.stdout.strip().split("\n")
    assert committed_files == ["data/simulations/gw3.json"], (
        f"Expected only data/simulations/gw3.json, got {committed_files}"
    )

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True,
    )
    assert "unrelated.txt" in status.stdout, (
        "unrelated.txt should still be staged but not committed"
    )
