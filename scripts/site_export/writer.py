from __future__ import annotations

import json
from pathlib import Path


def write_run_file(out_dir: Path, gw: int, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / f"gw{gw}.json"
    run_path.write_text(json.dumps(payload, indent=2) + "\n")
    return run_path


def update_index(out_dir: Path, gw: int, label: str, generated_at: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"schema_version": 1, "runs": []}

    run_id = f"gw{gw}"
    index["runs"] = [r for r in index["runs"] if r["id"] != run_id]
    index["runs"].append({"id": run_id, "gameweek": gw, "label": label, "generated_at": generated_at})
    index["runs"].sort(key=lambda r: r["gameweek"], reverse=True)

    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index_path
