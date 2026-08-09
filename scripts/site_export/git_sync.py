from __future__ import annotations

import subprocess
from pathlib import Path


def commit_and_push(repo_root: Path, data_dir: Path, message: str, push: bool = True) -> bool:
    """Stage only data_dir, commit if there are staged changes, optionally push.
    Returns True if a commit was created, False if there was nothing to commit."""
    rel = data_dir.relative_to(repo_root)
    subprocess.run(["git", "add", str(rel)], cwd=repo_root, check=True)

    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", str(rel)], cwd=repo_root)
    if staged.returncode == 0:
        return False

    subprocess.run(["git", "commit", "-m", message, "--", str(rel)], cwd=repo_root, check=True)
    if push:
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
    return True
