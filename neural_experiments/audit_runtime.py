"""Independent raw-timestamp audit; imports no production timing or metric code.

The primary run also checks measured traces against independently reconstructed
whole-file convolutions. Compression runs receive timing/hash auditing; their
approximate arithmetic belongs to their separately recorded quality experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"changed evidence: {path}")


def independently_summarize(stamps, period_ns):
    """Derive the complete reporting contract directly from timestamp differences."""
    stamps = np.asarray(stamps)
    if period_ns <= 0:
        raise ValueError("period must be positive")
    if stamps.dtype != np.int64 or stamps.ndim != 2 or stamps.shape[1] != 5 or not len(stamps):
        raise ValueError("invalid timing matrix")
    release, start, ready, complete, deadline = stamps.T
    if (
        np.any(release > start)
        or np.any(start > ready)
        or np.any(ready > complete)
        or np.any(start[1:] < complete[:-1])
        or np.any(np.diff(release) < 0)
    ):
        raise ValueError("invalid chronological or within-step event order")
    if not np.array_equal(deadline - release, np.full(len(stamps), period_ns)):
        raise ValueError("deadline interval mismatch")

    def quantiles(ns):
        ms = ns / 1_000_000.0
        return {
            "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)),
            "max_ms": float(ms.max()),
        }

    return {
        "bins": len(stamps),
        "step_service": quantiles(complete - start),
        "prediction_service": quantiles(ready - start),
        "queue_delay": quantiles(start - release),
        "release_to_prediction": quantiles(ready - release),
        "release_to_completion": quantiles(complete - release),
        "missed_deadlines": int(np.count_nonzero(complete > deadline)),
        "max_backlog_bins": int(np.floor_divide(start - release, period_ns).max()),
        "nominal_neural_seconds": len(stamps) * period_ns / 1e9,
        "first_start_to_last_completion_seconds": (complete[-1] - start[0]) / 1e9,
    }


def compare_summary(actual, saved, prefix=""):
    for key, value in actual.items():
        if key not in saved:
            raise ValueError(f"missing summary field {prefix}{key}")
        if isinstance(value, dict):
            compare_summary(value, saved[key], f"{prefix}{key}.")
        elif isinstance(value, (int, np.integer)):
            if value != saved[key]:
                raise ValueError(f"summary differs at {prefix}{key}")
        elif not np.isclose(value, saved[key], atol=1e-12, rtol=1e-12):
            raise ValueError(f"summary differs at {prefix}{key}")


def reference_trace(counts, weights):
    """Independent whole-file evaluation, without production model or stream code."""
    values = (counts.astype(np.float32) - weights["input_mean"]) / weights["input_scale"]
    hidden = torch.from_numpy(values.T.copy())[None]
    with torch.inference_mode():
        for layer, dilation in enumerate((1, 2, 4), 1):
            hidden = F.relu(
                F.conv1d(
                    F.pad(hidden, (2 * dilation, 0)),
                    torch.from_numpy(weights[f"conv{layer}_weight"]),
                    torch.from_numpy(weights[f"conv{layer}_bias"]),
                    dilation=dilation,
                )
            )
        output = F.linear(
            hidden.transpose(1, 2),
            torch.from_numpy(weights["head_weight"]),
            torch.from_numpy(weights["head_bias"]),
        )[0].numpy()
    return output * weights["target_scale"] + weights["target_mean"]


def audit(run, index_path, model_path, *, onnx_path=None, quality_run=None):
    run, index_path, model_path = map(Path, (run, index_path, model_path))
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("only complete runs can pass independent audit")
    verify(run / "summary.json", manifest["summary_sha256"])
    verify(index_path, manifest["index_sha256"])
    verify(model_path, manifest["model_sha256"])
    for relative, expected in manifest["source"]["files"].items():
        verify(run / "source" / relative, expected)
    verify(run / "source.diff", manifest["source"]["diff_sha256"])
    compression = "comparison" in manifest
    if compression:
        if quality_run is None:
            raise ValueError("compression timing audit requires its quality run")
        quality_run = Path(quality_run)
        verify(quality_run / "summary.json", manifest["quality_summary_sha256"])
        for precision, expected in manifest["graph_sha256"].items():
            filename = "float32.onnx" if precision == "float32" else "dynamic-int8.onnx"
            verify(quality_run / "seed-0/S4" / filename, expected)
        period_ns = manifest["period_ns"]
    else:
        if onnx_path is None:
            raise ValueError("primary timing audit requires the exported ONNX graph")
        verify(onnx_path, manifest["onnx_sha256"])
        config = manifest["configuration"]
        config_snapshot = run / "source/neural_experiments/configs/runtime-v1.json"
        verify(config_snapshot, manifest["config_sha256"])
        if read_json(config_snapshot) != config:
            raise ValueError("recorded config differs from its retained source bytes")
        if manifest["model_sha256"] != config["primary_model_sha256"]:
            raise ValueError("prespecified runtime model identity changed")
        if manifest.get("parity_status") != "passed":
            raise ValueError("arithmetic gate did not pass")
        period_ns = config["period_ns"]
    index = read_json(index_path)
    if (
        index.get("schema_version") != 1
        or index.get("role") != "development"
        or any(
            item["day"] not in {"S0", "S1", "S2", "S3", "S4"}
            or item["provider_split"] != "held-in-calib"
            for item in index["files"]
        )
    ):
        raise ValueError("refusing sealed or non-development sample arrays")
    files = {item["file_id"]: item for item in index["files"]}
    if len(files) != 11 or len(index["files"]) != 11:
        raise ValueError("expected eleven development files")
    if {item["day"] for item in files.values()} != {"S0", "S1", "S2", "S3", "S4"}:
        raise ValueError("expected all five development days")
    ordered_files = sorted(files, key=lambda key: (int(files[key]["day"][1:]), key))
    reference = {}
    with np.load(model_path, allow_pickle=False) as bundle:
        weights = {name: bundle[name] for name in bundle.files}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for file_id, item in files.items():
            path = index_path.parent / item["features"]
            if not path.resolve().is_relative_to(index_path.parent.resolve()):
                raise ValueError("source feature path escapes prepared cache")
            verify(path, item["features_sha256"])
            if not compression:
                with np.load(path, allow_pickle=False) as bundle:
                    if not np.array_equal(bundle["sequence_id"], np.arange(item["n_bins"])):
                        raise ValueError("source sequence mismatch")
                    reference[file_id] = reference_trace(bundle["counts"], weights)
        summary = read_json(run / "summary.json")
        reports, seen, max_difference = [], [], 0.0
        for row in summary["runs"]:
            name = row["precision"] if compression else row["implementation"]
            folder_name = (
                f"{row['mode']}-{row['repeat']}-{name}"
                if compression
                else f"{row['mode']}-repeat{row['repeat']}-{row['position']}-{name}"
            )
            folder = run / folder_name
            if read_json(folder / "summary.json") != row:
                raise ValueError("per-block and overall summaries disagree")
            expected_files = ordered_files if row["mode"] == "fast" else ordered_files[:1]
            if [record["file_id"] for record in row["files"]] != expected_files:
                raise ValueError("measured file inventory or chronological order changed")
            measured = []
            for record in row["files"]:
                file_id = record["file_id"]
                if file_id not in files:
                    raise ValueError("unexpected measured file")
                path = folder / f"{file_id}.npz"
                verify(path, record["trace_sha256"])
                with np.load(path, allow_pickle=False) as bundle:
                    stamps, predictions, ids = (
                        bundle["stamps"],
                        bundle["predictions"],
                        bundle["sequence_id"],
                    )
                    if not compression and not np.array_equal(
                        bundle["stamp_fields"],
                        [
                            "release_ns",
                            "start_ns",
                            "prediction_ready_ns",
                            "complete_ns",
                            "deadline_ns",
                        ],
                    ):
                        raise ValueError("raw timestamp column labels changed")
                expected_bins = (
                    files[file_id]["n_bins"]
                    if row["mode"] == "fast"
                    else 3001
                    if compression
                    else math.ceil(config["paced_seconds_per_block"] * 1e9 / period_ns) + 1
                )
                if record["bins"] != expected_bins:
                    raise ValueError("measured trace length differs from frozen workload")
                if (
                    not np.array_equal(ids, np.arange(record["bins"]))
                    or len(stamps) != record["bins"]
                ):
                    raise ValueError("measured bins are missing, duplicated or out of order")
                if predictions.shape != (len(stamps), 7) or not np.isfinite(predictions).all():
                    raise ValueError("invalid measured prediction trace")
                if row["mode"] == "paced":
                    expected_release = stamps[0, 0] + np.arange(len(stamps)) * period_ns
                    if not np.array_equal(stamps[:, 0], expected_release):
                        raise ValueError("paced release schedule changed")
                    if (stamps[-1, 0] - stamps[0, 0]) < 60_000_000_000:
                        raise ValueError("paced block spans less than sixty seconds")
                elif not np.array_equal(stamps[:, 0], stamps[:, 1]):
                    raise ValueError("fast replay must release inputs at processing start")
                if not compression:
                    expected = reference[file_id][: len(stamps)]
                    np.testing.assert_allclose(
                        predictions,
                        expected,
                        rtol=config["parity_rtol"],
                        atol=config["parity_atol"],
                    )
                    max_difference = max(
                        max_difference, float(np.max(np.abs(predictions - expected)))
                    )
                measured.append(stamps)
            recomputed = independently_summarize(np.concatenate(measured), period_ns)
            compare_summary(recomputed, row["summary"] if compression else row)
            seen.append((row["mode"], row["repeat"], name))
            reports.append(
                {
                    "mode": row["mode"],
                    "repeat": row["repeat"],
                    "implementation": name,
                    "metrics": recomputed,
                }
            )
        if compression:
            expected = [
                (mode, repeat, name)
                for mode in ("fast", "paced")
                for repeat, order in enumerate(
                    (("float32", "dynamic_int8"), ("dynamic_int8", "float32"))
                )
                for name in order
            ]
        else:
            expected = [
                (mode, repeat, name)
                for mode in ("fast", "paced")
                for repeat, order in enumerate(config[f"{mode}_orders"])
                for name in order
            ]
        if seen != expected:
            raise ValueError("run inventory or counterbalanced implementation order changed")
        return {
            "status": "passed",
            "audited_utc": datetime.now(UTC).isoformat(),
            "auditor_sha256": digest(__file__),
            "run_manifest_sha256": digest(run / "manifest.json"),
            "measured_source_head": manifest["source"]["head"],
            "runs": reports,
            "max_primary_trace_absolute_difference": None if compression else max_difference,
            "scope": (
                "raw timing/hash/summary audit; primary run also independent whole-file arithmetic"
            ),
            "limitations": (
                "RSS/CPU/constructor observations cannot be regenerated from per-bin stamps"
            ),
        }
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--quality-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.run, args.index, args.model, onnx_path=args.onnx, quality_run=args.quality_run
    )
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "runs": len(result["runs"])}), flush=True)


if __name__ == "__main__":
    main()
