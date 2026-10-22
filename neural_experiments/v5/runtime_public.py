"""Publish allowlisted V5 runtime aggregates bound to exact passed audits.

No feature arrays, model coefficients, predictions, raw timestamps or private
artifact paths are exported. Both producer and independent timing values remain.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from neural_experiments.v4 import runtime_public as shared
from neural_experiments.v5 import runtime_audit as audit

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = {
    "runtime": "neural_experiments/v5/runtime.py",
    "frozen_measurement": "neural_experiments/v4/runtime.py",
    "frozen_model": "neural_experiments/v4/model.py",
    "feature_loader": "neural_experiments/v4/data.py",
    "shared_feature_loader": "neural_experiments/v2/data.py",
}
BOUND_ARTIFACTS = (
    "plan",
    "model",
    "feature",
    "assessment",
    "assessment_result",
    "assessment_audit",
    "index",
    "selection",
    "assessment_prediction",
)
REPRESENTATIVE_FIELDS = (
    "participant",
    "session_id",
    "archive_rank",
    "method",
    "action",
    "candidate_id",
    "label_seconds",
)
REPLAY_FIELDS = (
    "assessed_trace_max_absolute_error",
    "measured_trace_max_absolute_error",
    "rtol",
    "atol",
    "sample_count_per_full_trace",
)


def select(value, fields):
    return {key: value[key] for key in fields}


def export(run, audit_path, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    manifest, summary = audit.read(run / "manifest.json"), audit.read(run / "summary.json")
    receipt = audit.read(audit_path)
    expected = {
        "manifest_sha256": audit.digest(run / "manifest.json"),
        "summary_sha256": audit.digest(run / "summary.json"),
        "auditor_source_sha256": audit.digest(audit.__file__),
        "timestamp_reference_sha256": audit.digest(ROOT / "neural_experiments/v2/audit_runtime.py"),
        "numeric_inspection_reference_sha256": audit.digest(
            ROOT / "neural_experiments/v4/runtime_audit.py"
        ),
        **{name + "_sha256": manifest[name + "_sha256"] for name in BOUND_ARTIFACTS},
        "source_sha256": manifest["source"],
    }
    audit.require(receipt.get("status") == "passed", "passed runtime audit required")
    audit.require(
        all(receipt.get(key) == value for key, value in expected.items()),
        "runtime audit identity differs",
    )
    audit.require(
        manifest["status"] == "complete" and manifest["kind"] == "v5_adaptive_calibration",
        "runtime incomplete or wrong kind",
    )
    audit.require(
        summary.pop("manifest_sha256") == expected["manifest_sha256"] and summary == manifest,
        "runtime summary identity",
    )
    audit.require(manifest["plan_sha256"] == audit.PLAN_SHA256, "unrecognized runtime plan")
    paths = {}
    for name in BOUND_ARTIFACTS:
        path = audit.contained(root, manifest[name + "_path"])
        audit.require(audit.digest(path) == manifest[name + "_sha256"], "changed " + name)
        paths[name] = path
    # Revalidate metadata binding, including the separately PASS quality receipt.
    # This opens JSON only and never reads target/prediction arrays.
    audit.assessment_identity(manifest, paths, root)
    audit.require(
        set(manifest["source"]) == set(COMPONENTS.values())
        and set(manifest["source_snapshots"]) == set(COMPONENTS.values()),
        "source inventory",
    )
    for name, sha in manifest["source"].items():
        path = audit.contained(root, manifest["source_snapshots"][name])
        audit.require(
            path.is_relative_to(run / "source") and audit.digest(path) == sha,
            "changed source snapshot",
        )
    audit.require(
        receipt["representative"] == manifest["representative"], "audited representative differs"
    )
    audit.require(
        receipt["blocks_checked"] == 5 and len(receipt["blocks"]) == len(manifest["blocks"]) == 5,
        "audited block count",
    )
    for block, checked in zip(manifest["blocks"], receipt["blocks"], strict=True):
        path = audit.contained(root, block["artifact_path"])
        audit.require(
            path.is_relative_to(run) and audit.digest(path) == block["artifact_sha256"],
            "changed timing block",
        )
        audit.require(
            (block["mode"], block["repeat"]) == (checked["mode"], checked["repeat"]),
            "audited block identity",
        )
        audit.require(
            all(
                math.isclose(block[k], checked[k], abs_tol=1e-12, rel_tol=0) for k in shared.METRICS
            ),
            "audited timing differs",
        )
    replay = select(receipt["prediction_replay"], REPLAY_FIELDS)
    audit.require(
        replay["rtol"] == replay["atol"] == 1e-10
        and replay["sample_count_per_full_trace"] == manifest["protocol"]["workload_bins"],
        "audited prediction scope",
    )
    audit.require(
        all(math.isfinite(replay[k]) and 0 <= replay[k] <= 2e-10 for k in REPLAY_FIELDS[:2]),
        "audited prediction errors",
    )
    sources = {key: manifest["source"][path] for key, path in COMPONENTS.items()}
    result = {
        **select(manifest, shared.MANIFEST_FIELDS),
        **{name + "_sha256": expected[name + "_sha256"] for name in BOUND_ARTIFACTS},
        "manifest_sha256": expected["manifest_sha256"],
        "summary_sha256": expected["summary_sha256"],
        "runtime_audit_sha256": audit.digest(audit_path),
        "exporter_source_sha256": audit.digest(__file__),
        "exporter_field_reference_sha256": audit.digest(shared.__file__),
        "source_sha256": sources,
        "representative": select(manifest["representative"], REPRESENTATIVE_FIELDS),
        "config": select(
            manifest["config"], ("family", "tau_seconds", "baseline_tau_seconds", "adapt_gamma")
        ),
        "full_trace_parity": select(
            manifest["full_trace_parity"],
            ("numpy_incremental_vs_batch", "batch_vs_assessed_prediction"),
        ),
        "protocol": select(manifest["protocol"], shared.PROTOCOL_FIELDS),
        "filter_response": select(manifest["filter_response"], shared.FILTER_FIELDS),
        "environment": {
            **select(manifest["environment"], ("python", "platform", "numpy", "scipy", "threads")),
            "threadpools": [
                {
                    k: pool[k]
                    for k in ("user_api", "internal_api", "num_threads", "version")
                    if k in pool
                }
                for pool in manifest["environment"]["threadpools"]
            ],
        },
        "blocks": [select(block, shared.BLOCK_FIELDS) for block in manifest["blocks"]],
        "independent_audit": {
            **select(
                receipt,
                (
                    "status",
                    "auditor_source_sha256",
                    "timestamp_reference_sha256",
                    "numeric_inspection_reference_sha256",
                    "manifest_sha256",
                    "summary_sha256",
                    "blocks_checked",
                    "scope",
                ),
            ),
            **{name + "_sha256": expected[name + "_sha256"] for name in BOUND_ARTIFACTS},
            "source_sha256": sources,
            "representative": select(receipt["representative"], REPRESENTATIVE_FIELDS),
            "prediction_replay": replay,
            "blocks": [
                select(block, ("mode", "repeat", *shared.METRICS)) for block in receipt["blocks"]
            ],
        },
        "data_policy": (
            "Aggregate timing, resource observations, representation and identity hashes only; "
            "no paths or sample arrays."
        ),
    }
    encoded = json.dumps(result, allow_nan=False)
    audit.require(
        not any(value in encoded for value in ("/Users/", "/home/", "X-Amz-", "PRIVATE KEY")),
        "private path or credential in publication",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "neural_experiments/results/v5-runtime.json"
    )
    args = parser.parse_args()
    audit.require(not args.output.exists(), "preserve earlier public receipts")
    result = export(args.run, args.audit, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
