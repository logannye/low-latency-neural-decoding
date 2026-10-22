"""Replay the predetermined exact V5 adaptive export, including an abstained model.

Uses frozen V4 numerical/measurement helpers without changing their constants.
No outcome-based representative selection and no target-array reads.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from neural_experiments.v4.runtime import (
    WARMUP_BINS,
    digest,
    filter_response,
    load_workload,
    measure,
    timestamp_summary,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA256 = "87c417f915259e7f77b7999217d3a2461884fffd175c676188e20c346472390e"
SOURCE_FILES = (
    "neural_experiments/v5/runtime.py",
    "neural_experiments/v4/runtime.py",
    "neural_experiments/v4/model.py",
    "neural_experiments/v4/data.py",
    "neural_experiments/v2/data.py",
)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def checked_path(root, name, expected):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or digest(path) != expected:
        raise ValueError("bound artifact path or hash differs")
    return path


def bind_representative(assessment_path, index_path, plan_path, audit_path, root=ROOT):
    """Select by prespecified metadata only; never inspect a score to choose."""
    root = Path(root).resolve()
    paths = {
        "assessment": Path(assessment_path),
        "index": Path(index_path),
        "plan": Path(plan_path),
        "assessment_audit": Path(audit_path),
    }
    payloads = {key: json.loads(path.read_text()) for key, path in paths.items()}
    assessment, index, plan, audit = (payloads[k] for k in paths)
    if digest(plan_path) != PLAN_SHA256:
        raise ValueError("runtime requires exact locked V5 plan")
    if (
        assessment.get("status") != "assessment_complete"
        or assessment.get("stage") != "assessment"
        or audit.get("status") != "PASS"
        or audit.get("stage") != "assessment"
        or audit.get("manifest_sha256") != digest(assessment_path)
    ):
        raise ValueError("completed assessment and exact independent audit required")
    result_path = Path(assessment_path).with_name("assessment.json")
    if (
        audit.get("assessment_sha256") != digest(result_path)
        or json.loads(result_path.read_text()) != assessment
        or audit.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("assessment result/audit identity differs")
    paths["assessment_result"] = result_path
    declared = assessment["indices"]["new_sessions"]
    if (
        checked_path(root, declared["path"], declared["sha256"]) != Path(index_path).resolve()
        or index.get("role") != "new_sessions"
        or index["study_lock"]["sha256"] != PLAN_SHA256
        or index["acquisition_plan_sha256"] != plan["acquisition_plan_sha256"]
    ):
        raise ValueError("assessment new-session index binding differs")
    items = [
        s
        for s in index["sessions"]
        if s["participant_id"] == "T2" and s["source"]["regular_session_rank"] == 13
    ]
    if len(items) != 1:
        raise ValueError("exact predetermined T2 archive-rank13 workload unavailable")
    item = items[0]
    rows = [
        r
        for r in assessment["cases"]
        if r["domain"] == "new_sessions"
        and r["session_id"] == item["session_id"]
        and r["method"] == "adaptive"
    ]
    if len(rows) != 1 or rows[0]["participant"] != "T2":
        raise ValueError("unique predetermined adaptive model required")
    row = rows[0]
    if not 1500 <= item["n_bins"] <= 30000:
        raise ValueError("predetermined feature extent outside V5 workload")
    model_path = checked_path(root, row["model_path"], row["model_sha256"])
    trace_path = checked_path(root, row["trace_path"], row["trace_sha256"])
    feature_path = checked_path(
        Path(index_path).resolve().parent, item["feature_path"], item["feature_sha256"]
    )
    selection_path = checked_path(
        root, assessment["selection_path"], assessment["selection_sha256"]
    )
    selection = json.loads(selection_path.read_text())
    if selection["plan"]["sha256"] != PLAN_SHA256 or audit.get("selection_sha256") != digest(
        selection_path
    ):
        raise ValueError("selected policy uses a different plan")
    for name in (
        "neural_experiments/v4/model.py",
        "neural_experiments/v4/data.py",
        "neural_experiments/v2/data.py",
    ):
        checked_path(root, name, selection["code_sha256"][name])

    def rel(path):
        return str(Path(path).resolve().relative_to(root))

    binding = {
        key + suffix: (rel(path) if suffix == "_path" else digest(path))
        for key, path in paths.items()
        for suffix in ("_path", "_sha256")
    }
    binding.update(
        assessment_prediction_path=rel(trace_path),
        assessment_prediction_sha256=row["trace_sha256"],
        selection_path=rel(selection_path),
        selection_sha256=digest(selection_path),
        representative={
            "participant": "T2",
            "session_id": item["session_id"],
            "archive_rank": 13,
            "method": "adaptive",
            "action": row["action"],
            "candidate_id": row["candidate_id"],
            "label_seconds": row["label_seconds"],
        },
    )
    return binding, model_path, feature_path, trace_path


def run(assessment_path, index_path, plan_path, output, *, assessment_audit_path):
    binding, model_path, feature_path, trace_path = bind_representative(
        assessment_path, index_path, plan_path, assessment_audit_path, root=ROOT
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(plan_path).read_text())
    manifest = {
        "schema_version": 1,
        "kind": "v5_adaptive_calibration",
        **binding,
        "status": "running",
        "representative_rule": plan["runtime"],
        "model_path": relative(model_path),
        "model_sha256": digest(model_path),
        "feature_path": relative(feature_path),
        "feature_sha256": digest(feature_path),
        "plan_path": relative(plan_path),
        "plan_sha256": digest(plan_path),
        "runtime_source_sha256": digest(__file__),
        "protocol": {
            "period_ms": 20,
            "warmup_bins": WARMUP_BINS,
            "fast_repeats": 3,
            "paced_blocks_per_implementation": 2,
            "paced_seconds": 60,
            "implementation": "numpy_incremental",
            "feature_prefix_cap_bins": 30000,
            "parity_atol": 1e-10,
            "parity_rtol": 1e-10,
            "service_boundary": (
                "raw validation, transforms, filter, normalization, linear head, "
                "projection, state, block reset"
            ),
            "exclude": "feature file loading, acquisition, upstream feature extraction, transport",
            "scheduler": "ordinary sleep until fixed nominal release; no busy waiting",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threads": 1,
        },
        "source": {},
        "source_snapshots": {},
        "blocks": [],
    }
    for name in SOURCE_FILES:
        path = ROOT / name
        name = relative(path)
        snapshot = output / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
        manifest["source"][name] = digest(path)
        manifest["source_snapshots"][name] = relative(snapshot)
    write_json(output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            decoder_model, features, boundaries, expected, error, load_ms = load_workload(
                model_path, feature_path
            )
            with np.load(trace_path, allow_pickle=False) as saved:
                assessed = saved["prediction"]
            if assessed.shape != expected.shape or assessed.dtype != np.float64:
                raise ValueError("assessed full prediction trace geometry differs")
            np.testing.assert_allclose(expected, assessed, rtol=1e-10, atol=1e-10)
            assessment_error = float(np.max(np.abs(expected - assessed)))
            manifest["environment"]["threadpools"] = [
                {
                    k: pool[k]
                    for k in ("user_api", "internal_api", "num_threads", "version")
                    if k in pool
                }
                for pool in threadpool_info()
            ]
            manifest["protocol"]["workload_bins"] = len(features["x_tx3"])
            manifest["full_trace_parity"] = {
                "numpy_incremental_vs_batch": error,
                "batch_vs_assessed_prediction": assessment_error,
            }
            manifest["config"] = decoder_model.config
            manifest["filter_response"] = filter_response(decoder_model.config)
            manifest["model_load_ms"] = load_ms
            manifest["parameter_count"] = decoder_model.weights.size
            manifest["numeric_weight_bytes"] = decoder_model.weights.nbytes
            manifest["numeric_transform_bytes"] = (
                decoder_model.mean.nbytes + decoder_model.scale.nbytes
            )
            manifest["numeric_model_array_bytes"] = (
                manifest["numeric_weight_bytes"] + manifest["numeric_transform_bytes"]
            )
            write_json(output / "manifest.json", manifest)
            schedule = [("fast", repeat) for repeat in range(3)] + [
                ("paced", repeat) for repeat in range(2)
            ]
            for mode, repeat in schedule:
                count = len(features["x_tx3"]) if mode == "fast" else 3000
                stamps, prediction, details = measure(
                    decoder_model,
                    features,
                    boundaries,
                    paced=mode == "paced",
                    count=count,
                )
                np.testing.assert_allclose(
                    prediction, expected[np.arange(count) % len(expected)], rtol=1e-10, atol=1e-10
                )
                name = f"{mode}-{repeat}.npz"
                np.savez(output / name, timestamps_ns=stamps, predictions=prediction)
                manifest["blocks"].append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "implementation": "numpy_incremental",
                        "artifact_path": relative(output / name),
                        "artifact_sha256": digest(output / name),
                        **timestamp_summary(stamps),
                        **details,
                    }
                )
                write_json(output / "manifest.json", manifest)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "repeat": repeat,
                            "service_p99_ms": manifest["blocks"][-1]["service_p99_ms"],
                        }
                    ),
                    flush=True,
                )
        manifest["status"] = "complete"
        manifest["scope"] = (
            "feature-ready single-model replay; finite observed deadline test, "
            "not hard real-time or neural acquisition latency. Calibration label wait and "
            "policy/fit costs are separate. A zero-output export still runs the full core."
        )
        write_json(output / "manifest.json", manifest)
        write_json(
            output / "summary.json",
            {**manifest, "manifest_sha256": digest(output / "manifest.json")},
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(output / "manifest.json", manifest)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--assessment-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.assessment,
        args.index,
        args.plan,
        args.output,
        assessment_audit_path=args.assessment_audit,
    )


if __name__ == "__main__":
    main()
