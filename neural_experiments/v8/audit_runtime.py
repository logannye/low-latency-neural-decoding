"""Independent V8 source-runtime raw-clock and feature-only numerical audit.

No producer or timing implementation is imported. Runtime replays first60s with
a model calibrated through150s and therefore supplies no predictive efficacy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v8.audit import (
    ROOT,
    artifact,
    compare_structure,
    contained,
    numeric,
    read,
    representation_reference,
    sha256,
    verify,
)

COUNT = 3000
PERIOD = 20_000_000


def clock_reference(clocks):
    t = np.asarray(clocks)
    if (
        t.dtype != np.int64
        or t.ndim != 2
        or t.shape[1] != 4
        or not len(t)
        or np.any(t < 0)
        or np.any(t[:, 1] < t[:, 0])
        or np.any(t[:, 2] < t[:, 1])
        or np.any(t[:, 3] - t[:, 0] != PERIOD)
        or np.any(np.diff(t[:, 0]) != PERIOD)
        or np.any(t[1:, 1] < t[:-1, 2])
    ):
        raise ValueError("Invalid ordered serial20ms raw clocks")
    summary = {"n": len(t), "deadline_misses": int(np.count_nonzero(t[:, 2] > t[:, 3]))}
    for label, duration in (
        ("service", t[:, 2] - t[:, 1]),
        ("queue", t[:, 1] - t[:, 0]),
        ("release_to_completion", t[:, 2] - t[:, 0]),
    ):
        values = duration.astype(float) / 1e6
        for percentile in (50, 95, 99):
            summary[f"{label}_p{percentile}_ms"] = float(
                np.percentile(values, percentile, method="linear")
            )
        summary[f"{label}_max_ms"] = float(np.max(values))
    return summary


def block_reference(row, arrays, expected, state_bytes):
    if set(arrays) != {"clocks", "prediction"}:
        raise ValueError("Unknown raw runtime fields")
    clocks, prediction = arrays["clocks"], arrays["prediction"]
    if (
        clocks.shape != (COUNT, 4)
        or prediction.dtype != np.float64
        or prediction.shape != expected.shape
        or not np.isfinite(prediction).all()
        or not np.allclose(prediction, expected, atol=1e-10, rtol=1e-10)
    ):
        raise ValueError("Wrong raw clock count or decoder prediction replay")
    rebuilt = clock_reference(clocks)
    for key in ("wall_start_ns", "wall_end_ns", "cpu_start_ns", "cpu_end_ns"):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError("Integer nonnegative process/block clocks required")
    if (
        not row["wall_start_ns"] <= clocks[0, 0] <= clocks[-1, 2] <= row["wall_end_ns"]
        or row["cpu_end_ns"] < row["cpu_start_ns"]
    ):
        raise ValueError("Invalid measured block/process clock containment")
    rebuilt.update(
        block_wall_seconds=(row["wall_end_ns"] - row["wall_start_ns"]) / 1e9,
        process_cpu_seconds=(row["cpu_end_ns"] - row["cpu_start_ns"]) / 1e9,
        numeric_filter_state_bytes=state_bytes,
    )
    for key, value in rebuilt.items():
        compare_structure(row[key], value, "raw runtime " + key)
    error = float(np.max(np.abs(prediction - expected)))
    # Producer's independently computed batch reference can differ by floating
    # roundoff from our recurrence. Compare declaration at the same parity tolerance.
    if (
        not np.isfinite(row["prediction_max_absolute_error"])
        or not 0 <= row["prediction_max_absolute_error"] <= 1e-10
        or abs(row["prediction_max_absolute_error"] - error) > 1e-10
    ):
        raise ValueError("Unsupported prediction-parity declaration")
    return {**rebuilt, "independent_prediction_max_absolute_error": error}


def audit_run(run, output):
    run = Path(run).resolve()
    mpath = run / "manifest.json"
    m = read(mpath)
    source = artifact(m["source_manifest_path"])
    index_path = artifact(m["index_path"])
    passed_path = artifact(m["source_audit_path"])
    for path, key in (
        (source, "source_manifest_sha256"),
        (index_path, "index_sha256"),
        (passed_path, "source_audit_sha256"),
    ):
        verify(path, m[key])
    original, index, passed = read(source), read(index_path), read(passed_path)
    if (
        m["status"] != "complete"
        or m["kind"] != "source_runtime"
        or m["study"] != "V8"
        or passed["status"] != "PASS"
        or passed["phase"] != "development"
        or passed["manifest_sha256"] != sha256(source)
        or passed["index_sha256"] != sha256(index_path)
        or passed["plan_sha256"] != original["plan"]["sha256"]
        or original["status"] != "complete"
        or original["phase"] != "development"
        or index["role"] != "development"
    ):
        raise ValueError("Passed exact source-quality binding required")
    plan_path = ROOT / "neural_experiments/v8/runtime-plan.json"
    verify(plan_path, m["plan_sha256"])
    verify(run / "runtime-plan.json", m["plan_sha256"])
    plan = read(plan_path)
    for key, value in (
        ("count", COUNT),
        ("repeats", 2),
        ("period_ns", PERIOD),
        ("warmup_steps", 500),
        ("threadpool_limit", 1),
    ):
        if plan[key] != value:
            raise ValueError("Runtime protocol changed")
    sources = {
        **original["code_sha256"],
        "neural_experiments/v8/runtime.py": sha256(ROOT / "neural_experiments/v8/runtime.py"),
    }
    compare_structure(m["source_sha256"], sources, "frozen runtime source")
    for path, pin in sources.items():
        verify(artifact(path), pin)
        verify(contained(run / "source_snapshot", path), pin)
    t2 = sorted(
        [r for r in index["sessions"] if r["participant_id"] == "T2"],
        key=lambda r: r["source"]["archive_ordinal"],
    )
    if len(t2) != 21 or len({r["source"]["archive_ordinal"] for r in t2}) != 21:
        raise ValueError("Wrong representative source inventory")
    item = t2[15]
    primary = original["selection"]["primary_method"]
    rows = [
        r
        for r in original["factorial_rows"]
        if r["session_id"] == item["session_id"] and r["method"] == primary
    ]
    if len(rows) != 1 or item["n_bins"] < COUNT:
        raise ValueError("Representative must be exact archive-rank16 primary")
    row = rows[0]
    compare_structure(
        m["representative"],
        {
            "participant": "T2",
            "session_id": item["session_id"],
            "post_implant_day": item["post_implant_day"],
            "archive_rank": 16,
            "method": primary,
            "config": row["config"],
        },
        "representative",
    )
    model_path = artifact(row["model"]["model_path"])
    feature_path = contained(index_path.parent, item["feature_path"])
    for kind, path, pin in (
        ("model", model_path, row["model"]["model_sha256"]),
        ("features", feature_path, item["feature_sha256"]),
    ):
        verify(path, pin)
        if artifact(m["artifacts"][kind]["path"]) != path or m["artifacts"][kind]["sha256"] != pin:
            raise ValueError("Runtime artifact differs from representative")
    model = numeric(model_path)
    config = json.loads(model["config"].item())
    if config != row["config"]:
        raise ValueError("Numeric export configuration changed")
    features = numeric(feature_path)
    c = item["n_channels"]
    feature_data = {key: features[key][:COUNT] for key in ("x_tx3", "x_tx45", "x_sbp")}
    segments = [
        (s["start"], min(s["end"], COUNT))
        for s in item["segments"]
        if s["start"] < COUNT and s["end"] > s["start"]
    ]
    for s in item["segments"]:
        a, b = s["start"], min(s["end"], COUNT)
        if a < b and not np.all(features["sequence_id"][a:b] == s["sequence_id"]):
            raise ValueError("Feature reset metadata differs from measured replay")
    raw = representation_reference({"features": feature_data, "segments": segments}, config)
    z = np.clip((raw - model["mean"]) / model["scale"], -8, 8) / np.sqrt(
        len(config["taus_seconds"])
    )
    expected = z @ model["weights"][:-1] + model["weights"][-1]
    expected /= np.maximum(1.0, np.linalg.norm(expected, axis=1))[:, None]
    compare_structure(
        m["memory"],
        {
            "model_numeric_bytes": sum(model[k].nbytes for k in ("mean", "scale", "weights")),
            "model_artifact_bytes": model_path.stat().st_size,
            "feature_channels": c,
            "filtered_dimensions": len(model["mean"]),
            "workload_rows": COUNT,
        },
        "numeric component sizes",
    )
    if [r["repeat"] for r in m["blocks"]] != [0, 1]:
        raise ValueError("Both planned runs must be retained")
    blocks = []
    for row in m["blocks"]:
        path = artifact(row["artifact_path"])
        if not path.is_relative_to(run):
            raise ValueError("Raw timing artifact outside run")
        verify(path, row["artifact_sha256"])
        rebuilt = block_reference(
            row, numeric(path), expected, (2 * c + 1) * len(config["taus_seconds"]) * 8
        )
        blocks.append({"repeat": row["repeat"], **rebuilt})
    result = {
        "schema_version": 1,
        "status": "PASS",
        "manifest_sha256": sha256(mpath),
        "auditor_sha256": sha256(__file__),
        "numeric_reference_sha256": sha256(ROOT / "neural_experiments/v8/audit.py"),
        "source_manifest_sha256": sha256(source),
        "source_audit_sha256": sha256(passed_path),
        "plan_sha256": m["plan_sha256"],
        "index_sha256": sha256(index_path),
        "model_sha256": sha256(model_path),
        "feature_sha256": sha256(feature_path),
        "blocks_checked": len(blocks),
        "blocks": blocks,
        "memory": m["memory"],
        "scope": (
            "Raw timestamp arithmetic and full feature-only numerical replay. Source model "
            "trained after the measured input prefix; no efficacy or causal prediction claim. "
            "Numeric memory excludes Python overhead and scratch/input arrays."
        ),
    }
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve existing runtime audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run, args.output)))
