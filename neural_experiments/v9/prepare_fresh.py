"""Open fresh arrays only after committed, independently audited source selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from neural_experiments.v6.data import _study_lock

ROOT = Path(__file__).resolve().parents[2]


def validate_selection(path, expected_sha256):
    from neural_experiments.v9.assessment import source_hashes

    path = Path(path)
    _study_lock(path, expected_sha256)
    selection = json.loads(path.read_text())
    if (
        selection.get("schema_version") != 1
        or selection.get("study") != "V9"
        or selection.get("source_audit_status") != "PASS"
    ):
        raise ValueError("V9 source audit must pass before opening fresh arrays")
    relative = Path(selection["source_audit_path"])
    audit_path = (ROOT / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not audit_path.is_relative_to(ROOT):
        raise ValueError("source audit path must remain contained in the repository")
    _study_lock(audit_path, selection["source_audit_sha256"])
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "PASS"
        or audit.get("phase") != "development"
        or audit.get("manifest_sha256") != selection["source_manifest_sha256"]
    ):
        raise ValueError("development source audit does not authenticate the selection manifest")
    original = {
        key: value
        for key, value in selection.items()
        if key
        not in {
            "source_manifest_sha256",
            "source_audit_sha256",
            "source_audit_path",
        }
    }
    original["source_audit_status"] = "PENDING"
    payload = json.dumps(original, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(payload.encode()).hexdigest() != audit.get("selection_sha256"):
        raise ValueError("source audit does not bind this exact selected recipe payload")
    identities = selection.get("scientific_code_sha256")
    if not isinstance(identities, dict) or any(
        re.fullmatch(r"neural_experiments/v[0-9]+/[a-z_]+\.py", name) is None
        or not isinstance(sha, str)
        or re.fullmatch("[0-9a-f]{64}", sha) is None
        for name, sha in identities.items()
    ):
        raise ValueError("invalid scientific source identity map")
    plan = Path(__file__).with_name("study-plan.json")
    _study_lock(plan, selection["plan_sha256"])
    if source_hashes() != selection["scientific_code_sha256"]:
        raise ValueError("scientific source changed after source selection")
    for name, sha in selection["scientific_code_sha256"].items():
        _study_lock(ROOT / name, sha)
    return selection


def run(cache, selection_path, selection_sha256):
    from neural_experiments.v9.fresh_data import prepare

    selection = validate_selection(selection_path, selection_sha256)
    return prepare(
        cache,
        ROOT / "data/neural_v8/development/index.json",
        ROOT / "data/neural_v8/fresh/index.json",
        study_plan=Path(__file__).with_name("study-plan.json"),
        study_plan_sha256=selection["plan_sha256"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    args = parser.parse_args()
    run(args.cache, args.selection, args.selection_sha256)
