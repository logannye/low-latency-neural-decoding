"""One-shot V9 fresh assessment using a committed, source-audited selection."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_experiments.v3.study import digest, relative, write_json
from neural_experiments.v6.data import _study_lock
from neural_experiments.v9 import data, metrics, model
from neural_experiments.v9.prepare_fresh import validate_selection

ROOT = Path(__file__).resolve().parents[2]


def scientific_sources():
    versions = {
        "v2": ("data",),
        "v3": ("data", "model", "study"),
        "v4": ("data", "model"),
        "v5": ("data",),
        "v6": ("data",),
        "v7": ("data", "fresh_data"),
        "v8": ("data",),
        "v9": (
            "data",
            "fresh_data",
            "model",
            "development",
            "assessment",
            "metrics",
            "prepare_fresh",
        ),
    }
    return tuple(
        f"neural_experiments/{version}/{name}.py"
        for version, names in versions.items()
        for name in names
    )


def source_hashes():
    return {p: digest(ROOT / p) for p in scientific_sources()}


def run(index_path, output, selection_path, selection_sha):
    index_path, output, selection_path = map(Path, (index_path, output, selection_path))
    if digest(selection_path) != selection_sha:
        raise ValueError("exact source selection required")
    _study_lock(selection_path, selection_sha)
    selection = validate_selection(selection_path, selection_sha)
    plan_path = Path(__file__).with_name("study-plan.json")
    _study_lock(plan_path, selection["plan_sha256"])
    if selection.get("source_audit_status") != "PASS":
        raise ValueError("source audit must pass before fresh assessment")
    hashes = source_hashes()
    if hashes != selection["scientific_code_sha256"]:
        raise ValueError("scientific source changed after source lock")
    index = data.load_index(index_path, expected_role="fresh", verify_hashes=True)
    configurations = selection["configurations"]
    primary = selection["primary_method"]
    if primary != metrics.PRIMARY or set(configurations) != set(metrics.METHODS):
        raise ValueError("locked primary and controls required")
    output.mkdir(parents=True, exist_ok=False)
    for path in scientific_sources():
        destination = output / "source_snapshot" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    manifest = {
        "schema_version": 1,
        "study": "V9",
        "phase": "assessment",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "plan": {"path": relative(plan_path), "sha256": digest(plan_path)},
        "index": {"path": relative(index_path), "sha256": digest(index_path)},
        "source_selection": {"path": relative(selection_path), "sha256": selection_sha},
        "code_sha256": hashes,
        "configurations": configurations,
        "primary_method": primary,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threadpool_limit": 1,
        },
        "rows": [],
        "sessions": [],
    }
    write_json(output / "manifest.json", manifest)
    with threadpool_limits(limits=1):
        for item in index["sessions"]:
            case = data.load_case(index_path, item, labels=True, context=True)
            artifact_dir = output / "sessions" / item["session_id"]
            result = model.evaluate_session(case, configurations, artifact_dir=artifact_dir)
            manifest["rows"].extend(result["rows"])
            manifest["sessions"].append(
                {
                    "session_id": item["session_id"],
                    "artifact_dir": relative(artifact_dir),
                    "fit_receipts": result["fit_receipts"],
                    "model_refs": result["model_refs"],
                }
            )
            write_json(output / "manifest.json", manifest)
            print(json.dumps({"session": item["session_id"], "status": "scored"}), flush=True)
    manifest["summary"] = metrics.summarize(manifest["rows"], primary)
    if source_hashes() != hashes:
        raise ValueError("scientific source changed during assessment")
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    args = parser.parse_args()
    result = run(args.index, args.output, args.selection, args.selection_sha256)
    print(json.dumps(result["summary"]["gates"]))
