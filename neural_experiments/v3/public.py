"""Fail-closed aggregate-only exports from independently audited private runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from neural_experiments.v3.study import digest, resolve, write_json


def read(path):
    return json.loads(Path(path).read_text())


def check_receipt(receipt, **expected):
    if receipt.get("status") != "passed" or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("passed audit is not bound to these exact artifacts")


def safe_write(path, value):
    text = json.dumps(value, allow_nan=False)
    if re.search(r"/Users/|/home/|X-Amz-(?:Signature|Credential)|BEGIN .*PRIVATE KEY", text):
        raise ValueError("private path or credential in proposed publication")
    write_json(path, value)


def publish_selection(selection_path, audit_path, output):
    selection, receipt = read(selection_path), read(audit_path)
    check_receipt(receipt, selection_sha256=digest(selection_path))
    result = {
        "status": "complete",
        "scope": (
            "Source-only selection using exposed development windows; no fresh assessment scores."
        ),
        "selection_sha256": digest(selection_path),
        "audit_sha256": digest(audit_path),
        "independent_audit": receipt,
        "plan": selection["plan"],
        "source_files": selection["source_files"],
        "development_index_sha256": selection["development_index"]["sha256"],
        "created_at": selection["created_at"],
        "assessment_labels_opened": selection["assessment_labels_opened"],
        "candidates": selection["candidates"],
        "choices": selection["choices"],
        "development_unavailable": selection["development_unavailable"],
        "development_cases": [
            {k: v for k, v in row.items() if k != "trace_path"}
            for row in selection["development_cases"]
        ],
        "elapsed_seconds": selection["elapsed_seconds"],
    }
    safe_write(output, result)


def publish_study(run, audit_path, output):
    run = Path(run)
    manifest, receipt = read(run / "manifest.json"), read(audit_path)
    check_receipt(receipt, manifest_sha256=digest(run / "manifest.json"))
    if manifest["status"] != "complete":
        raise ValueError("incomplete assessment")
    summary_path = resolve(manifest["summary"]["path"])
    if digest(summary_path) != manifest["summary"]["sha256"]:
        raise ValueError("changed summary")
    index_path = resolve(manifest["assessment_index"]["path"])
    if digest(index_path) != manifest["assessment_index"]["sha256"]:
        raise ValueError("changed assessment cohort")
    index = read(index_path)
    models = []
    for reference in manifest["models"]:
        identity_path = resolve(reference["identity_path"])
        if digest(identity_path) != reference["sha256"]:
            raise ValueError("changed model identity")
        identity = read(identity_path)
        choice = identity["source_choice"]
        models.append(
            {
                **{
                    k: identity[k]
                    for k in (
                        "target_participant",
                        "session_id",
                        "method",
                        "budget_seconds",
                        "config",
                        "model_sha256",
                        "assessment_day_minus_calibration_day",
                        "fit_export_inference_seconds",
                    )
                },
                "identity_sha256": reference["sha256"],
                "candidate_id": None if choice is None else choice["candidate_id"],
                "calibration": {
                    k: v for k, v in identity["calibration"].items() if k != "source_index_path"
                },
                "traces": [
                    {k: v for k, v in trace.items() if k != "path"} for trace in identity["traces"]
                ],
            }
        )
    cohort = {
        "provider": index["provider"],
        "participants": index["participants"],
        "selection": index["selection"],
        "participant_dispositions": index["participant_dispositions"],
        "sessions": [
            {
                **{
                    k: s[k]
                    for k in (
                        "participant_id",
                        "session_id",
                        "post_implant_day",
                        "n_bins",
                        "n_channels",
                        "feature_sha256",
                        "target_sha256",
                    )
                },
                "n_blocks": len(s["segments"]),
                "regular_session_rank": s["source"]["regular_session_rank"],
                "raw_sha256": s["source"]["member_sha256"],
                "archive_file_id": s["source"]["archive_file_id"],
            }
            for s in index["sessions"]
        ],
    }
    result = {
        "status": "complete",
        "scope": (
            "New sessions from previously studied people; supervised daily calibration, "
            "not zero-shot, untouched-person, or clinical confirmation."
        ),
        "manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(summary_path),
        "audit_sha256": digest(audit_path),
        "independent_audit": receipt,
        "plan": manifest["plan"],
        "source_files": manifest["source_files"],
        "selection_sha256": manifest["selection"]["sha256"],
        "selection_audit_sha256": manifest["selection_audit"]["sha256"],
        "development_index_sha256": manifest["development_index"]["sha256"],
        "assessment_index_sha256": manifest["assessment_index"]["sha256"],
        "assessment_started_at": manifest["assessment_started_at"],
        "assessment_completed_at": manifest["assessment_completed_at"],
        "assessment_elapsed_seconds": manifest["assessment_elapsed_seconds"],
        "environment": manifest["environment"],
        "cohort": cohort,
        "summary": read(summary_path),
        "models": models,
        "unavailable": manifest["unavailable"],
    }
    safe_write(output, result)


def publish_runtime(run, audit_path, output):
    run = Path(run)
    manifest, receipt = read(run / "manifest.json"), read(audit_path)
    check_receipt(receipt, manifest_sha256=digest(run / "manifest.json"))
    if manifest["status"] != "complete":
        raise ValueError("incomplete runtime")
    result = {k: v for k, v in manifest.items() if k not in ("model_path", "feature_path")}
    result["blocks"] = [
        {k: v for k, v in b.items() if k != "artifact_path"} for b in manifest["blocks"]
    ]
    result.update(
        manifest_sha256=digest(run / "manifest.json"),
        audit_sha256=digest(audit_path),
        independent_audit=receipt,
    )
    safe_write(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("selection", "study", "runtime"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"selection": publish_selection, "study": publish_study, "runtime": publish_runtime}[args.kind](
        args.input, args.audit, args.output
    )


if __name__ == "__main__":
    main()
