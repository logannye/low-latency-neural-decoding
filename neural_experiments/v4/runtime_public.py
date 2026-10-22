"""Publish only allowlisted V4 runtime aggregates bound to a passed clock audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from neural_experiments.v4 import runtime_audit

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = {
    "runtime": "neural_experiments/v4/runtime.py",
    "model": "neural_experiments/v4/model.py",
    "feature_loader": "neural_experiments/v4/data.py",
    "shared_feature_loader": "neural_experiments/v2/data.py",
}
METRICS = (
    "n",
    "service_p50_ms",
    "service_p99_ms",
    "service_max_ms",
    "queue_p99_ms",
    "release_to_completion_p99_ms",
    "deadline_misses",
    "negative_queue_count",
)
BLOCK_FIELDS = (
    "mode",
    "repeat",
    "implementation",
    "artifact_sha256",
    *METRICS,
    "initialization_ms",
    "initialization_scope",
    "persistent_numeric_state_bytes",
    "state_measurement",
    "process_cpu_seconds",
    "block_wall_seconds",
    "process_cpu_fraction_of_block_wall",
    "process_peak_rss_bytes",
    "rss_scope",
)
MANIFEST_FIELDS = (
    "schema_version",
    "kind",
    "status",
    "representative_rule",
    "model_sha256",
    "feature_sha256",
    "plan_sha256",
    "runtime_source_sha256",
    "config",
    "full_trace_parity",
    "model_load_ms",
    "parameter_count",
    "numeric_weight_bytes",
    "numeric_transform_bytes",
    "numeric_model_array_bytes",
    "scope",
)
PROTOCOL_FIELDS = (
    "period_ms",
    "warmup_bins",
    "fast_repeats",
    "paced_blocks_per_implementation",
    "paced_seconds",
    "implementation",
    "feature_prefix_cap_bins",
    "parity_atol",
    "parity_rtol",
    "service_boundary",
    "exclude",
    "scheduler",
    "workload_bins",
)
FILTER_FIELDS = (
    "fast_tau_seconds",
    "slow_baseline_tau_seconds",
    "fast_decay_per_bin",
    "fast_steady_state_mean_sample_age_seconds",
    "fast_95_percent_step_response_bins",
    "temporal_support",
    "startup",
    "interpretation",
)
AUDIT_FIELDS = (
    "status",
    "auditor_source_sha256",
    "timestamp_reference_sha256",
    "manifest_sha256",
    "summary_sha256",
    "plan_sha256",
    "model_sha256",
    "feature_sha256",
    "blocks_checked",
    "scope",
)


def selected(value, keys):
    return {key: value[key] for key in keys}


def export(run, audit_path, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    manifest = runtime_audit.read(run / "manifest.json")
    summary = runtime_audit.read(run / "summary.json")
    receipt = runtime_audit.read(audit_path)
    digest, require = runtime_audit.digest, runtime_audit.require
    expected = {
        "manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
        "auditor_source_sha256": digest(runtime_audit.__file__),
        "timestamp_reference_sha256": digest(ROOT / "neural_experiments/v2/audit_runtime.py"),
        **{f"{name}_sha256": manifest[f"{name}_sha256"] for name in ("plan", "model", "feature")},
        "source_sha256": manifest["source"],
    }
    require(receipt.get("status") == "passed", "passed independent audit required")
    require(all(receipt.get(k) == v for k, v in expected.items()), "audit identity differs")
    require(manifest["status"] == "complete", "runtime incomplete")
    require(summary.pop("manifest_sha256") == expected["manifest_sha256"], "summary identity")
    require(summary == manifest, "summary differs from manifest")
    require(set(manifest["source"]) == set(COMPONENTS.values()), "source inventory")
    require(manifest["plan_sha256"] == runtime_audit.PLAN_SHA256, "unrecognized runtime plan")
    for name in ("plan", "model", "feature"):
        path = runtime_audit.contained(root, manifest[f"{name}_path"])
        require(digest(path) == manifest[f"{name}_sha256"], f"changed {name}")
    for name, sha in manifest["source"].items():
        path = runtime_audit.contained(root, manifest["source_snapshots"][name])
        require(path.is_relative_to(run / "source") and digest(path) == sha, "changed source")
    for block in manifest["blocks"]:
        path = runtime_audit.contained(root, block["artifact_path"])
        require(
            path.is_relative_to(run) and digest(path) == block["artifact_sha256"], "changed block"
        )
    require(receipt["blocks_checked"] == 5 and len(receipt["blocks"]) == 5, "audited block count")
    require(len(manifest["blocks"]) == 5, "runtime block count")
    for block, audited in zip(manifest["blocks"], receipt["blocks"], strict=True):
        require(
            (block["mode"], block["repeat"]) == (audited["mode"], audited["repeat"]),
            "audited block identity",
        )
        require(
            all(math.isclose(block[k], audited[k], abs_tol=1e-12, rel_tol=0) for k in METRICS),
            "audited block metrics",
        )
    audit_blocks = [selected(b, ("mode", "repeat", *METRICS)) for b in receipt["blocks"]]
    sources = {name: manifest["source"][path] for name, path in COMPONENTS.items()}
    result = {
        **selected(manifest, MANIFEST_FIELDS),
        "manifest_sha256": expected["manifest_sha256"],
        "summary_sha256": expected["summary_sha256"],
        "audit_sha256": digest(audit_path),
        "exporter_source_sha256": digest(__file__),
        "source_sha256": sources,
        "protocol": selected(manifest["protocol"], PROTOCOL_FIELDS),
        "filter_response": selected(manifest["filter_response"], FILTER_FIELDS),
        "environment": {
            **selected(
                manifest["environment"], ("python", "platform", "numpy", "scipy", "threads")
            ),
            "threadpools": [
                {k: p[k] for k in ("user_api", "internal_api", "num_threads", "version") if k in p}
                for p in manifest["environment"]["threadpools"]
            ],
        },
        "blocks": [selected(b, BLOCK_FIELDS) for b in manifest["blocks"]],
        "independent_audit": {
            **selected(receipt, AUDIT_FIELDS),
            "source_sha256": sources,
            "blocks": audit_blocks,
        },
        "data_policy": "Aggregate timing, resources and hashes only; no paths or sample arrays.",
    }
    encoded = json.dumps(result, allow_nan=False)
    require(
        all(s not in encoded for s in ("/Users/", "/home/", "X-Amz-", "PRIVATE KEY")),
        "private path or credential in publication",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_audit.require(not args.output.exists(), "preserve earlier public receipts")
    result = export(args.run, args.audit, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
