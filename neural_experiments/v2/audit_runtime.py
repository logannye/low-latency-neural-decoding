"""Independent V2 timestamp/provenance audit; does not execute a decoder.

Prediction shapes/finiteness and recorded parity declarations are checked.
This auditor does not independently establish prediction correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(values, probability):
    """Linear order-statistic interpolation, independently implemented."""
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def reconstruct(stamps, *, paced, period_ns):
    require(
        stamps.dtype == np.int64 and stamps.ndim == 2 and stamps.shape[1] == 4,
        "invalid timestamp shape or dtype",
    )
    require(len(stamps) > 0, "empty timestamps")
    release, start, completion, deadline = stamps.T
    require(np.all(start >= release), "negative queue duration")
    require(np.all(completion >= start), "completion precedes step start")
    require(
        all(np.all(np.diff(column) >= 0) for column in (release, start, completion)),
        "nonmonotonic release/start/completion",
    )
    require(np.all(start[1:] >= completion[:-1]), "overlapping sequential steps")
    require(np.all(deadline - release == period_ns), "deadline is not release plus20ms")
    if paced:
        require(np.all(np.diff(release) == period_ns), "paced releases are not20ms apart")
    service = [(int(c) - int(s)) / 1_000_000 for s, c in zip(start, completion, strict=True)]
    queue = [(int(s) - int(r)) / 1_000_000 for r, s in zip(release, start, strict=True)]
    total = [(int(c) - int(r)) / 1_000_000 for r, c in zip(release, completion, strict=True)]
    return {
        "n": len(stamps),
        "service_p50_ms": percentile(service, 0.5),
        "service_p99_ms": percentile(service, 0.99),
        "service_max_ms": max(service),
        "queue_p99_ms": percentile(queue, 0.99),
        "release_to_completion_p99_ms": percentile(total, 0.99),
        "deadline_misses": sum(int(c) > int(d) for c, d in zip(completion, deadline, strict=True)),
        "negative_queue_count": 0,
    }


def audit(run, *, root=None):
    run = Path(run).resolve()
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    manifest = read(run / "manifest.json")
    summary = read(run / "summary.json")
    require(manifest["status"] == "complete", "runtime run incomplete")
    require(
        summary.pop("manifest_sha256") == digest(run / "manifest.json"),
        "summary manifest hash mismatch",
    )
    require(summary == manifest, "summary differs from completed manifest")
    kind = manifest["kind"]
    require(kind in {"h1", "transfer"}, "unknown decoder kind")
    for prefix in ("model", "feature"):
        require(
            digest(root / manifest[f"{prefix}_path"]) == manifest[f"{prefix}_sha256"],
            f"changed {prefix} artifact",
        )
    runtime_source = "neural_experiments/v2/runtime.py"
    required_sources = {runtime_source, "neural_experiments/v2/transfer.py"}
    required_sources |= (
        {"neural_experiments/v2/accuracy.py", "src/neural_decoder/data.py"}
        if kind == "h1"
        else {"neural_experiments/v2/data.py"}
    )
    require(required_sources <= set(manifest["source"]), "missing executable source identity")
    require(
        manifest["source"][runtime_source] == manifest["runtime_source_sha256"],
        "runtime source identity mismatch",
    )
    for original, expected in manifest["source"].items():
        snapshot = (root / manifest["source_snapshots"][original]).resolve()
        require(snapshot.is_relative_to(run / "source"), "source snapshot outside retained run")
        require(digest(snapshot) == expected, "changed source snapshot")
    with np.load(root / manifest["model_path"], allow_pickle=False) as model:
        names = [name for name in model.files if name.startswith("state__")]
        if kind == "transfer":
            names = [
                name for name in model.files if name.startswith(("adapter.", "core.", "head."))
            ]
        require(bool(names), "missing numeric model weights")
        parameter_count = sum(model[name].size for name in names)
        weight_bytes = sum(model[name].nbytes for name in names)
    require(parameter_count == manifest["parameter_count"], "parameter count differs")
    require(weight_bytes == manifest["numeric_weight_bytes"], "numeric weight bytes differ")
    protocol = manifest["protocol"]
    require(
        protocol["period_ms"] == 20 and protocol["warmup_bins"] == 500,
        "runtime period/warmup differs from protocol",
    )
    require(
        protocol["parity_atol"] == (1e-7 if kind == "h1" else 3e-5)
        and protocol["parity_rtol"] == 3e-5,
        "parity tolerance differs",
    )
    names = ["numpy_incremental", "torch_window"]
    require(
        set(manifest["full_trace_parity"]) == set(names), "missing full-trace parity declaration"
    )
    require(
        all(np.isfinite(value) and value >= 0 for value in manifest["full_trace_parity"].values()),
        "invalid declared parity error",
    )
    repeats = protocol["fast_repeats"]
    require(type(repeats) is int and repeats >= 1, "invalid fast repeat count")
    require(
        protocol["paced_blocks_per_implementation"] == 2 and protocol["paced_seconds"] >= 60,
        "paced replay does not meet two60-second blocks",
    )
    workload = protocol["workload_bins"]
    require(
        0 < workload <= protocol["feature_prefix_cap_bins"] == 30000,
        "invalid feature prefix workload",
    )
    with np.load(root / manifest["feature_path"], allow_pickle=False) as features:
        observed_bins = len(features["counts" if kind == "h1" else "x"])
    require(workload == min(observed_bins, 30000), "declared workload differs from feature prefix")
    expected_order = []
    for mode, count in (("fast", repeats), ("paced", 2)):
        for repeat in range(count):
            order = names if repeat % 2 == 0 else names[::-1]
            expected_order.extend((mode, repeat, name) for name in order)
    observed_order = [(b["mode"], b["repeat"], b["implementation"]) for b in manifest["blocks"]]
    require(observed_order == expected_order, "block counts or alternating execution order differ")
    blocks = []
    for block in manifest["blocks"]:
        path = root / block["artifact_path"]
        require(digest(path) == block["artifact_sha256"], "changed raw timing artifact")
        with np.load(path, allow_pickle=False) as data:
            require(
                set(data.files) == {"timestamps", "predictions"}, "unexpected raw timing arrays"
            )
            stamps, predictions = data["timestamps"], data["predictions"]
        expected_n = (
            round(protocol["paced_seconds"] / 0.02) if block["mode"] == "paced" else workload
        )
        require(len(stamps) == expected_n, "timestamp count differs from declared workload")
        require(
            predictions.shape == (expected_n, 7 if kind == "h1" else 2)
            and predictions.dtype == np.float32
            and np.isfinite(predictions).all(),
            "invalid retained prediction shape/dtype/finiteness",
        )
        derived = reconstruct(stamps, paced=block["mode"] == "paced", period_ns=20_000_000)
        for key, value in derived.items():
            require(
                np.isclose(value, block[key], atol=1e-12, rtol=0),
                f"independent timing reconstruction differs: {key}",
            )
        blocks.append(
            {
                "mode": block["mode"],
                "repeat": block["repeat"],
                "implementation": block["implementation"],
                **derived,
            }
        )
    return {
        "status": "pass",
        "auditor_source_sha256": digest(Path(__file__)),
        "manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
        "model_sha256": manifest["model_sha256"],
        "feature_sha256": manifest["feature_sha256"],
        "source_snapshots_checked": len(manifest["source"]),
        "blocks_checked": len(blocks),
        "blocks": blocks,
        "prediction_correctness_independently_verified": False,
        "prediction_check_scope": (
            "shape, finiteness, hashes and parity declarations only; decoder not executed"
        ),
        "clock_check_scope": (
            "sequential release/start/completion, nonnegative queue, "
            "20 ms paced releases and deadlines"
        ),
        "memory_check_scope": (
            "numeric parameter bytes only; process/Python/scratch memory not audited"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "choose a fresh audit output; preserve earlier receipts")
    receipt = audit(arguments.run, root=arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))
