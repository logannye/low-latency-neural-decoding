"""Independent saved-timestamp reconstruction for the V3 runtime protocol.

Reuses the previously independent integer-clock/order-statistic reference, not
the runtime producer. Decoder correctness is covered by a separate parity check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit_runtime import digest, read, reconstruct, require

ROOT = Path(__file__).resolve().parents[2]


def audit(run, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    manifest = read(run / "manifest.json")
    summary = read(run / "summary.json")
    require(manifest["status"] == "complete", "runtime incomplete")
    require(summary.pop("manifest_sha256") == digest(run / "manifest.json"), "manifest hash")
    require(summary == manifest, "summary differs")
    require(manifest["kind"] == "v3_regularized_ridge", "unknown runtime kind")
    for name in ("plan", "feature", "model"):
        require(
            digest(root / manifest[f"{name}_path"]) == manifest[f"{name}_sha256"], f"changed {name}"
        )
    needed = {f"neural_experiments/v3/{name}.py" for name in ("runtime", "model", "data")}
    needed.add("neural_experiments/v2/data.py")
    require(set(manifest["source"]) == needed, "executable source inventory")
    for name, expected in manifest["source"].items():
        snapshot = (root / manifest["source_snapshots"][name]).resolve()
        require(snapshot.is_relative_to(run / "source"), "source outside retained snapshot")
        require(digest(snapshot) == expected, "changed executable snapshot")
    require(
        manifest["runtime_source_sha256"] == manifest["source"]["neural_experiments/v3/runtime.py"],
        "runtime source hash",
    )
    with np.load(root / manifest["model_path"], allow_pickle=False) as m:
        count = m["coefficients"].size + m["intercept"].size
        weights = m["coefficients"].nbytes + m["intercept"].nbytes
        transforms = m["mean"].nbytes + m["scale"].nbytes
        state = 2 * m["mean"].nbytes
        config = json.loads(m["config"].item())
    require(count == manifest["parameter_count"], "parameter count")
    require(weights == manifest["numeric_weight_bytes"], "weight bytes")
    require(transforms == manifest["numeric_transform_bytes"], "transform bytes")
    require(weights + transforms == manifest["numeric_model_array_bytes"], "model bytes")
    require(config == manifest["config"], "runtime/model config")
    with np.load(root / manifest["feature_path"], allow_pickle=False) as f:
        workload = min(len(f["x_tx"]), 30000)
    p = manifest["protocol"]
    expected = {
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
    require(all(p.get(k) == v for k, v in expected.items()), "runtime protocol differs")
    order = [("fast", i) for i in range(3)] + [("paced", i) for i in range(2)]
    require(
        [(b["mode"], b["repeat"]) for b in manifest["blocks"]] == order, "runtime block inventory"
    )
    require(
        set(manifest["full_trace_parity"]) == {"numpy_incremental_vs_batch"}, "parity declaration"
    )
    error = manifest["full_trace_parity"]["numpy_incremental_vs_batch"]
    require(np.isfinite(error) and 0 <= error <= 2e-10, "declared parity error")
    blocks = []
    for block in manifest["blocks"]:
        path = root / block["artifact_path"]
        require(digest(path) == block["artifact_sha256"], "changed timestamp artifact")
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == {"timestamps_ns", "predictions"}, "unexpected timing fields")
            stamps, pred = z["timestamps_ns"], z["predictions"]
        n = 3000 if block["mode"] == "paced" else workload
        require(
            pred.shape == (n, 2) and pred.dtype == np.float64 and np.isfinite(pred).all(),
            "prediction shape/finiteness",
        )
        require(len(stamps) == n, "timing workload length")
        require(block["persistent_numeric_state_bytes"] == state, "numeric filter-state bytes")
        require(block["implementation"] == "numpy_incremental", "unexpected implementation")
        derived = reconstruct(stamps, paced=block["mode"] == "paced", period_ns=20_000_000)
        for key, value in derived.items():
            require(np.isclose(value, block[key], atol=1e-12, rtol=0), f"timing mismatch: {key}")
        blocks.append({"mode": block["mode"], "repeat": block["repeat"], **derived})
    return {
        "status": "passed",
        "auditor_source_sha256": digest(__file__),
        "timestamp_reference_sha256": digest(ROOT / "neural_experiments/v2/audit_runtime.py"),
        "manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
        "model_sha256": manifest["model_sha256"],
        "feature_sha256": manifest["feature_sha256"],
        "blocks_checked": len(blocks),
        "blocks": blocks,
        "scope": (
            "Independent timestamp, schedule, counts, source/array identity, and numeric state "
            "checks; decoder predictions not independently replayed by this timestamp auditor."
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
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "blocks": result["blocks_checked"]}))


if __name__ == "__main__":
    main()
