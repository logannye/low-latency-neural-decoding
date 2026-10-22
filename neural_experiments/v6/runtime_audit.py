"""Independent V6 clock arithmetic, representative binding and numerical replay.

Uses frozen V5 independent numerical replay, without timing-producer imports or
target arrays. The assessed prediction trace is bound to the passed quality audit
and independently recomputed from features and numeric coefficients. Calibration
quality is outside this audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit_runtime import digest, read, reconstruct, require
from neural_experiments.v4.runtime_audit import contained, inspect_features, inspect_model
from neural_experiments.v5.runtime_audit import close_predictions, independent_prediction

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA256 = "f8c075f99c51b6da1bfe07a57861746fff158ec2fd518ec21ecf45f10b170b7a"
SOURCE_FILES = {
    "neural_experiments/v6/runtime.py",
    "neural_experiments/v5/runtime.py",
    "neural_experiments/v4/runtime.py",
    "neural_experiments/v4/model.py",
    "neural_experiments/v4/data.py",
    "neural_experiments/v2/data.py",
}


def assessment_identity(manifest, paths, root):
    assessment = read(paths["assessment"])
    result = read(paths["assessment_result"])
    receipt = read(paths["assessment_audit"])
    index, selection = read(paths["index"]), read(paths["selection"])
    require(assessment == result, "assessment result differs from manifest")
    require(
        assessment.get("status") == "assessment_complete"
        and assessment.get("stage") == "assessment",
        "assessment incomplete",
    )
    require(
        receipt.get("status") == "PASS" and receipt.get("stage") == "assessment",
        "assessment audit failed",
    )
    for name, key in (
        ("assessment", "manifest_sha256"),
        ("assessment_result", "assessment_sha256"),
        ("selection", "selection_sha256"),
        ("plan", "plan_sha256"),
        ("index", "index_sha256"),
        ("source_audit", "source_audit_sha256"),
    ):
        require(receipt.get(key) == digest(paths[name]), "assessment audit identity: " + name)
    declared = assessment["index"]
    require(
        contained(root, declared["path"]) == paths["index"]
        and declared["sha256"] == digest(paths["index"]),
        "assessment index identity",
    )
    require(
        contained(root, assessment["selection_path"]) == paths["selection"]
        and assessment["selection_sha256"] == digest(paths["selection"]),
        "assessment selection identity",
    )
    require(
        index.get("role") == "new_sessions" and index["study_lock"]["sha256"] == PLAN_SHA256,
        "fresh index role or plan",
    )
    require(
        index["acquisition_plan_sha256"] == read(paths["plan"])["acquisition_plan_sha256"],
        "acquisition plan",
    )
    require(selection["plan"]["sha256"] == PLAN_SHA256, "source selection plan")
    source_receipt = read(paths["source_audit"])
    require(
        contained(root, assessment["source_audit_path"]) == paths["source_audit"]
        and assessment["source_audit_sha256"] == digest(paths["source_audit"]),
        "source audit identity",
    )
    require(
        source_receipt.get("status") == "PASS"
        and source_receipt.get("stage") == "source"
        and source_receipt.get("selection_sha256") == digest(paths["selection"])
        and source_receipt.get("plan_sha256") == PLAN_SHA256,
        "source audit failed or differs",
    )
    chosen = [
        s
        for s in index["sessions"]
        if s["participant_id"] == "T2" and s["source"]["regular_session_rank"] == 16
    ]
    require(len(chosen) == 1, "predetermined rank16 workload missing or duplicated")
    item = chosen[0]
    rows = [
        r
        for r in assessment["cases"]
        if r["domain"] == "new_sessions"
        and r["session_id"] == item["session_id"]
        and r["method"] == "confirmed"
    ]
    require(
        len(rows) == 1 and rows[0]["participant"] == "T2",
        "predetermined confirmed export missing or duplicated",
    )
    row = rows[0]
    decision = row["decision"]
    require(
        row["action"] in ("accept", "abstain")
        and decision["action"] == row["action"]
        and decision["training_seconds"] == row["training_label_seconds"]
        and decision["requested_label_seconds"] == row["label_seconds"],
        "decision action or label accounting differs",
    )
    require(
        (row["action"] == "accept" and decision["model_sha256"] == row["model_sha256"])
        or (row["action"] == "abstain" and decision["model_sha256"] is None),
        "deployed artifact differs from confirmation decision",
    )
    expected = {
        "participant": "T2",
        "session_id": item["session_id"],
        "archive_rank": 16,
        "method": "confirmed",
        "action": row["action"],
        "candidate_id": row["decision"]["candidate_id"],
        "label_seconds": row["label_seconds"],
        "training_label_seconds": row["training_label_seconds"],
        "available_label_seconds": decision["available_label_seconds"],
        "decision_time_seconds": decision["decision_time_seconds"],
    }
    require(manifest["representative"] == expected, "representative metadata differs")
    require(1500 <= item["n_bins"] <= 30000, "authorized feature extent")
    for name, key in (("model", "model"), ("assessment_prediction", "trace")):
        require(
            contained(root, row[key + "_path"]) == paths[name]
            and row[key + "_sha256"] == digest(paths[name]),
            "assessed " + key + " identity",
        )
    require(
        contained(paths["index"].parent, item["feature_path"]) == paths["feature"]
        and item["feature_sha256"] == digest(paths["feature"]),
        "representative feature identity",
    )
    for name in (
        "neural_experiments/v4/model.py",
        "neural_experiments/v4/data.py",
        "neural_experiments/v2/data.py",
    ):
        require(
            manifest["source"][name] == selection["code_sha256"][name],
            "frozen shared implementation differs",
        )
    return item


def audit(run, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    manifest, summary = read(run / "manifest.json"), read(run / "summary.json")
    require(manifest["status"] == "complete", "runtime incomplete")
    require(
        summary.pop("manifest_sha256") == digest(run / "manifest.json") and summary == manifest,
        "summary identity",
    )
    require(manifest["kind"] == "v6_exact_model_confirmation", "runtime kind")
    paths = {}
    for name in (
        "plan",
        "feature",
        "model",
        "assessment",
        "assessment_result",
        "assessment_audit",
        "source_audit",
        "index",
        "selection",
        "assessment_prediction",
    ):
        paths[name] = contained(root, manifest[name + "_path"])
        require(digest(paths[name]) == manifest[name + "_sha256"], "changed " + name)
    require(manifest["plan_sha256"] == PLAN_SHA256, "unrecognized V6 plan")
    require(
        manifest["representative_rule"] == read(paths["plan"])["runtime"], "representative rule"
    )
    require(
        set(manifest["source"]) == SOURCE_FILES
        and set(manifest["source_snapshots"]) == SOURCE_FILES,
        "source inventory",
    )
    for name, sha in manifest["source"].items():
        snapshot = contained(root, manifest["source_snapshots"][name])
        require(
            snapshot.is_relative_to(run / "source") and digest(snapshot) == sha,
            "changed executable snapshot",
        )
    require(
        manifest["runtime_source_sha256"] == manifest["source"]["neural_experiments/v6/runtime.py"],
        "runtime source identity",
    )
    item = assessment_identity(manifest, paths, root)
    config, mean, scale, weights, channels = inspect_model(paths["model"])
    workload = inspect_features(paths["feature"], channels)
    require(workload == item["n_bins"], "feature index length")
    require(config == manifest["config"], "runtime model config")
    for key, value in {
        "parameter_count": weights.size,
        "numeric_weight_bytes": weights.nbytes,
        "numeric_transform_bytes": mean.nbytes + scale.nbytes,
        "numeric_model_array_bytes": mean.nbytes + scale.nbytes + weights.nbytes,
    }.items():
        require(manifest[key] == value, key)
    protocol = {
        "period_ms": 20,
        "warmup_bins": 500,
        "fast_repeats": 3,
        "paced_blocks_per_implementation": 2,
        "paced_seconds": 60,
        "implementation": "numpy_incremental",
        "feature_prefix_cap_bins": 30000,
        "parity_atol": 1e-10,
        "parity_rtol": 1e-10,
        "workload_bins": workload,
    }
    require(all(manifest["protocol"].get(k) == v for k, v in protocol.items()), "runtime protocol")
    require(
        manifest["environment"]["threads"] == 1
        and all(p["num_threads"] == 1 for p in manifest["environment"]["threadpools"]),
        "thread declaration",
    )
    require(
        [(b["mode"], b["repeat"]) for b in manifest["blocks"]]
        == [("fast", i) for i in range(3)] + [("paced", i) for i in range(2)],
        "block inventory",
    )
    parity = manifest["full_trace_parity"]
    require(
        set(parity) == {"numpy_incremental_vs_batch", "batch_vs_assessed_prediction"}
        and all(np.isfinite(v) and 0 <= v <= 2e-10 for v in parity.values()),
        "parity declaration",
    )
    expected = independent_prediction(paths["feature"], config, mean, scale, weights)
    with np.load(paths["assessment_prediction"], allow_pickle=False) as z:
        assessed = z["prediction"]
    assessed_error = close_predictions(assessed, expected, "assessed full prediction")
    if manifest["representative"]["action"] == "abstain":
        require(
            np.count_nonzero(weights) == 0 and np.count_nonzero(expected) == 0,
            "abstained model is not exactly zero",
        )
    blocks, errors = [], []
    for block in manifest["blocks"]:
        path = contained(root, block["artifact_path"])
        require(
            path.is_relative_to(run) and digest(path) == block["artifact_sha256"],
            "changed timestamp artifact",
        )
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == {"timestamps_ns", "predictions"}, "timing array inventory")
            stamps, pred = z["timestamps_ns"], z["predictions"]
        n = 3000 if block["mode"] == "paced" else workload
        require(len(stamps) == n, "timing workload length")
        errors.append(
            close_predictions(pred, expected[np.arange(n) % workload], "measured prediction")
        )
        require(
            block["persistent_numeric_state_bytes"] == 2 * mean.nbytes
            and block["implementation"] == "numpy_incremental",
            "numeric state or implementation",
        )
        derived = reconstruct(stamps, paced=block["mode"] == "paced", period_ns=20_000_000)
        for key, value in derived.items():
            require(np.isclose(value, block[key], atol=1e-12, rtol=0), "timing mismatch: " + key)
        for key in (
            "initialization_ms",
            "process_cpu_seconds",
            "block_wall_seconds",
            "process_peak_rss_bytes",
        ):
            require(
                np.isfinite(block[key]) and block[key] >= 0, "invalid resource observation: " + key
            )
        require(
            block["block_wall_seconds"] > 0
            and np.isclose(
                block["process_cpu_fraction_of_block_wall"],
                block["process_cpu_seconds"] / block["block_wall_seconds"],
                atol=1e-12,
                rtol=0,
            ),
            "CPU fraction arithmetic",
        )
        blocks.append({"mode": block["mode"], "repeat": block["repeat"], **derived})
    return {
        "status": "passed",
        "auditor_source_sha256": digest(__file__),
        "prediction_reference_sha256": digest(ROOT / "neural_experiments/v5/runtime_audit.py"),
        "timestamp_reference_sha256": digest(ROOT / "neural_experiments/v2/audit_runtime.py"),
        "numeric_inspection_reference_sha256": digest(
            ROOT / "neural_experiments/v4/runtime_audit.py"
        ),
        "manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
        **{key + "_sha256": digest(value) for key, value in paths.items()},
        "source_sha256": manifest["source"],
        "representative": manifest["representative"],
        "blocks_checked": len(blocks),
        "blocks": blocks,
        "prediction_replay": {
            "assessed_trace_max_absolute_error": assessed_error,
            "measured_trace_max_absolute_error": max(errors),
            "rtol": 1e-10,
            "atol": 1e-10,
            "sample_count_per_full_trace": workload,
        },
        "scope": (
            "Independent selection, bound audited export, feature-only numerical recurrence, "
            "prediction trace parity, integer timestamp arithmetic and schedule validation. "
            "CPU/RSS remain producer observations. An abstained zero model demonstrates "
            "execution of the frozen core, not useful decoding. No target arrays opened; "
            "calibration quality, fit cost and label wait are separate."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve earlier audit receipts")
    result = audit(args.run, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "blocks": result["blocks_checked"]}))


if __name__ == "__main__":
    main()
