"""Independent V4 runtime audit from retained monotonic timestamps and identities.

Uses the prior independent integer-clock/order-statistic reference, never the
runtime producer or its summary function. This is not prediction replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit_runtime import digest, read, reconstruct, require

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA256 = "f00419baf098ab5e42d6757714bd43cc5107579edf2a7c915c24a4f6ff7d8aba"


def contained(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root), "artifact outside declared root")
    return path


def inspect_model(path):
    """Validate the numeric export without importing the deployed decoder."""
    with np.load(path, allow_pickle=False) as z:
        require(
            set(z.files) == {"format_version", "config", "mean", "scale", "weights"},
            "model field inventory",
        )
        version, encoded = z["format_version"], z["config"]
        require(
            version.shape == () and version.dtype == np.int64 and version.item() == 4,
            "model format version",
        )
        require(encoded.shape == () and encoded.dtype.kind == "U", "model config encoding")
        config = json.loads(encoded.item())
        require(
            isinstance(config, dict)
            and set(config) == {"family", "tau_seconds", "baseline_tau_seconds", "adapt_gamma"},
            "model config fields",
        )
        require(config["family"] in ("tx3", "tx45", "tx3_sbp", "tx45_sbp"), "model family")
        for key, choices in (
            ("tau_seconds", (0.2, 0.4)),
            ("baseline_tau_seconds", (None, 5.0)),
            ("adapt_gamma", (1.0, 10.0)),
        ):
            require(not isinstance(config[key], bool) and config[key] in choices, key)
        mean, scale, weights = z["mean"], z["scale"], z["weights"]
    require(
        mean.ndim == 1
        and 1 <= len(mean) <= 1024
        and scale.shape == mean.shape
        and weights.shape == (len(mean) + 1, 2)
        and all(a.dtype == np.float64 and np.isfinite(a).all() for a in (mean, scale, weights))
        and np.all(scale > 0),
        "model numeric geometry",
    )
    factor = 2 if config["family"].endswith("_sbp") else 1
    require(len(mean) % factor == 0 and 1 <= len(mean) // factor <= 512, "model channels")
    return config, mean, scale, weights, len(mean) // factor


def inspect_features(path, channels):
    """Check only the feature container; no target file is opened."""
    with np.load(path, allow_pickle=False) as z:
        require(
            set(z.files)
            == {"x_tx3", "x_tx45", "x_sbp", "timestamps", "sequence_id", "channel_ids"},
            "feature field inventory",
        )
        n = len(z["x_tx3"])
        require(n > 0 and channels in (96, 192), "feature dimensions")
        for key in ("x_tx3", "x_tx45", "x_sbp"):
            value = z[key]
            require(
                value.dtype == np.float32
                and value.shape == (n, channels)
                and np.isfinite(value).all()
                and np.all(value >= 0),
                "feature neural geometry",
            )
            if key != "x_sbp":
                require(np.all(value == np.floor(value)), "noninteger threshold counts")
        timestamps, sequence, channel_ids = z["timestamps"], z["sequence_id"], z["channel_ids"]
    require(
        timestamps.shape == (n,)
        and timestamps.dtype == np.float64
        and np.isfinite(timestamps).all()
        and np.all(np.diff(timestamps) > 0),
        "feature time geometry",
    )
    require(
        sequence.shape == (n,)
        and sequence.dtype.kind in "iu"
        and np.all(sequence >= 0)
        and np.all(sequence[1:] >= sequence[:-1]),
        "feature sequence geometry",
    )
    same = sequence[1:] == sequence[:-1]
    require(
        np.allclose(np.diff(timestamps)[same], 0.02, atol=1e-9, rtol=0),
        "feature within-sequence cadence",
    )
    require(
        channel_ids.shape == (channels,)
        and channel_ids.dtype.kind in "iu"
        and np.all(channel_ids[1:] > channel_ids[:-1]),
        "feature channel identity",
    )
    return min(n, 30000)


def audit(run, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    manifest, summary = read(run / "manifest.json"), read(run / "summary.json")
    require(manifest["status"] == "complete", "runtime incomplete")
    require(summary.pop("manifest_sha256") == digest(run / "manifest.json"), "manifest hash")
    require(summary == manifest, "summary differs")
    require(manifest["kind"] == "v4_history_adapted_ridge", "unknown runtime kind")
    paths = {}
    for name in ("plan", "feature", "model"):
        paths[name] = contained(root, manifest[f"{name}_path"])
        require(digest(paths[name]) == manifest[f"{name}_sha256"], f"changed {name}")
    require(manifest["plan_sha256"] == PLAN_SHA256, "unrecognized runtime plan")
    require(
        manifest["representative_rule"] == read(paths["plan"])["runtime"], "representative rule"
    )
    needed = {f"neural_experiments/v4/{name}.py" for name in ("runtime", "model", "data")}
    needed.add("neural_experiments/v2/data.py")
    require(set(manifest["source"]) == needed, "executable source inventory")
    require(set(manifest["source_snapshots"]) == needed, "source snapshot inventory")
    for name, expected in manifest["source"].items():
        snapshot = contained(root, manifest["source_snapshots"][name])
        require(snapshot.is_relative_to(run / "source"), "source outside retained snapshot")
        require(digest(snapshot) == expected, "changed executable snapshot")
    require(
        manifest["runtime_source_sha256"] == manifest["source"]["neural_experiments/v4/runtime.py"],
        "runtime source hash",
    )
    config, mean, scale, weights, channels = inspect_model(paths["model"])
    require(weights.size == manifest["parameter_count"], "parameter count")
    require(weights.nbytes == manifest["numeric_weight_bytes"], "weight bytes")
    transforms = mean.nbytes + scale.nbytes
    require(transforms == manifest["numeric_transform_bytes"], "transform bytes")
    require(weights.nbytes + transforms == manifest["numeric_model_array_bytes"], "model bytes")
    require(config == manifest["config"], "runtime/model config")
    workload = inspect_features(paths["feature"], channels)
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
    require(all(manifest["protocol"].get(k) == v for k, v in expected.items()), "runtime protocol")
    require(manifest["environment"]["threads"] == 1, "declared thread count")
    require(
        all(p["num_threads"] == 1 for p in manifest["environment"]["threadpools"]),
        "declared threadpool count",
    )
    order = [("fast", i) for i in range(3)] + [("paced", i) for i in range(2)]
    require([(b["mode"], b["repeat"]) for b in manifest["blocks"]] == order, "block inventory")
    require(
        set(manifest["full_trace_parity"]) == {"numpy_incremental_vs_batch"}, "parity declaration"
    )
    error = manifest["full_trace_parity"]["numpy_incremental_vs_batch"]
    require(np.isfinite(error) and 0 <= error <= 2e-10, "declared parity error")
    blocks = []
    for block in manifest["blocks"]:
        path = contained(root, block["artifact_path"])
        require(path.is_relative_to(run), "timing artifact outside run")
        require(digest(path) == block["artifact_sha256"], "changed timestamp artifact")
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == {"timestamps_ns", "predictions"}, "unexpected timing fields")
            stamps, pred = z["timestamps_ns"], z["predictions"]
        n = 3000 if block["mode"] == "paced" else workload
        require(
            pred.shape == (n, 2)
            and pred.dtype == np.float64
            and np.isfinite(pred).all()
            and np.all(np.hypot(pred[:, 0], pred[:, 1]) <= 1 + 1e-12),
            "prediction geometry or unit disk",
        )
        require(len(stamps) == n, "timing workload length")
        require(block["persistent_numeric_state_bytes"] == 2 * mean.nbytes, "filter state bytes")
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
        "plan_sha256": PLAN_SHA256,
        "model_sha256": manifest["model_sha256"],
        "feature_sha256": manifest["feature_sha256"],
        "source_sha256": manifest["source"],
        "blocks_checked": len(blocks),
        "blocks": blocks,
        "scope": (
            "Independent timestamp arithmetic, schedule, counts, source/model/feature identity "
            "and numeric state checks. Prediction parity and representative selection are "
            "recorded declarations, not independently replayed/selected by this auditor; "
            "CPU/RSS measurements are producer observations. No labels opened."
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
