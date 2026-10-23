"""Publish audited V7B runtime aggregates without local paths or sample arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

BLOCK_NUMBERS = (
    "n",
    "service_p50_ms",
    "service_p99_ms",
    "service_max_ms",
    "queue_p99_ms",
    "release_to_completion_p99_ms",
    "deadline_misses",
    "negative_queue_count",
    "initialization_ms",
    "persistent_numeric_state_bytes",
    "process_cpu_seconds",
    "block_wall_seconds",
    "process_cpu_fraction_of_block_wall",
    "process_peak_rss_bytes",
)
REFIT_NUMBERS = (
    "n",
    "p50_ms",
    "p95_ms",
    "max_ms",
    "exceeded_two_second_allowance",
    "cached_representation_bytes",
)
AUDIT_HASHES = (
    "auditor_source_sha256",
    "timestamp_reference_sha256",
    "prediction_reference_sha256",
    "manifest_sha256",
    "source_manifest_sha256",
    "source_audit_sha256",
    "model_sha256",
    "feature_sha256",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checksum(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("SHA256 identity required")
    return value


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative aggregate required")
    return value


def numbers(record, fields):
    return {field: number(record[field]) for field in fields}


def environment_text(value):
    if not isinstance(value, str) or any(c in value for c in ("/", "\\", "\n", "\r")):
        raise ValueError("scalar environment descriptor required")
    return value


def aggregate(manifest, audit, *, manifest_sha256, audit_sha256, annotation_sha256):
    """Require an exact PASS receipt and build every nested field explicitly."""
    if (
        manifest.get("status") != "complete"
        or manifest.get("kind") != "V7B_source_runtime"
        or audit.get("status") != "PASS"
        or audit.get("manifest_sha256") != manifest_sha256
        or audit.get("blocks_checked") != 5
        or audit.get("refits_checked") != 10
    ):
        raise ValueError("completed runtime and exact passed audit required")
    for field, expected in (
        ("source_manifest_sha256", manifest["source_manifest_sha256"]),
        ("source_audit_sha256", manifest["source_audit_sha256"]),
        ("model_sha256", manifest["artifacts"]["model"]["sha256"]),
        ("feature_sha256", manifest["artifacts"]["features"]["sha256"]),
    ):
        if audit.get(field) != expected:
            raise ValueError("runtime audit provenance mismatch: " + field)
    if audit["refit_summary"] != manifest["refit_summary"]:
        raise ValueError("audited refit summary differs")
    expected_blocks = [("fast", i) for i in range(3)] + [("paced", i) for i in range(2)]
    if [(b["mode"], b["repeat"]) for b in manifest["blocks"]] != expected_blocks or [
        (b["mode"], b["repeat"]) for b in audit["blocks"]
    ] != expected_blocks:
        raise ValueError("complete ordered runtime blocks required")
    blocks = []
    for measured, verified in zip(manifest["blocks"], audit["blocks"], strict=True):
        for key in ("n", "deadline_misses", "negative_queue_count"):
            if measured[key] != verified[key]:
                raise ValueError("audited block counts differ")
        for key in (
            "service_p50_ms",
            "service_p99_ms",
            "service_max_ms",
            "queue_p99_ms",
            "release_to_completion_p99_ms",
        ):
            if not math.isclose(number(measured[key]), number(verified[key]), abs_tol=1e-9):
                raise ValueError("audited block timing differs")
        blocks.append(
            {
                "mode": measured["mode"],
                "repeat": measured["repeat"],
                **numbers(measured, BLOCK_NUMBERS),
            }
        )
    environment = {
        k: environment_text(manifest["environment"][k])
        for k in ("python", "platform", "numpy", "scipy")
    }
    environment["threadpools"] = [
        {
            k: number(p[k]) if k == "num_threads" else environment_text(p[k])
            for k in ("user_api", "internal_api", "num_threads", "version")
            if k in p
        }
        for p in manifest["environment"]["threadpools"]
    ]
    representative = manifest["representative"]
    if (
        representative["participant"],
        representative["post_implant_day"],
        representative["method"],
        representative["fit_stop_seconds"],
        representative["deployment_seconds"],
    ) != ("T2", 250, "periodic240", 540, 542):
        raise ValueError("declared source representative required")
    return {
        "schema_version": 1,
        "evidence_role": "source-only numerical and compute measurement; no fresh efficacy",
        "private_manifest_sha256": checksum(manifest_sha256),
        "plan_sha256": checksum(manifest["plan_sha256"]),
        "benchmark_source_sha256": checksum(
            manifest["source_sha256"]["neural_experiments/v7/runtime_continual.py"]
        ),
        "representative": {
            "participant": "T2",
            "post_implant_day": 250,
            "method": "periodic240",
            "label_seconds": 240,
            "fit_stop_seconds": 540,
            "deployment_seconds": 542,
            "training_rows": number(representative["training_rows"]),
            "selection": "greatest numeric exposed T2 day and last complete periodic fit",
        },
        "protocol": numbers(
            manifest["protocol"],
            (
                "fast_repeats",
                "paced_repeats",
                "paced_seconds_each",
                "refit_repeats",
                "period_ms",
                "warmup_bins",
                "threads",
                "parity_atol",
                "parity_rtol",
                "refit_allowance_seconds",
            ),
        ),
        "environment": environment,
        "blocks": blocks,
        "refit_summary": numbers(manifest["refit_summary"], REFIT_NUMBERS),
        "model_load_ms": number(manifest["model_load_ms"]),
        "numeric_model_array_bytes": number(manifest["numeric_model_array_bytes"]),
        "parity": numbers(
            manifest,
            ("numpy_incremental_vs_batch_max_error", "post_deployment_vs_source_trace_max_error"),
        ),
        "audit": {
            "status": "PASS",
            "receipt_sha256": checksum(audit_sha256),
            **{k: checksum(audit[k]) for k in AUDIT_HASHES},
            "blocks_checked": 5,
            "refits_checked": 10,
            **numbers(
                audit,
                (
                    "refit_max_numeric_error",
                    "post_deployment_source_trace_max_error",
                    "timed_prediction_max_error",
                ),
            ),
        },
        "annotation": {
            "report_sha256": checksum(annotation_sha256),
            "evidence_role": "post-run operational annotation, outside numerical audit",
            "ambient_load": (
                "approximately 0.79 seconds of synthetic tests and a background download "
                "overlapped this attempt; individual timing effects were not isolated"
            ),
            "attempt_selection": "original attempt preserved; no favorable rerun selected",
        },
        "scope": {
            "prediction": (
                "feature-ready validation, transforms, causal filtering, normalization, head "
                "and projection; scheduler included only in release-to-completion"
            ),
            "refit": (
                "paid-mask selection, normalization, ridge and export/reload with "
                "representations and labels already cached"
            ),
            "excluded": (
                "input acquisition, upstream feature extraction, transport, input loading "
                "and refit representation caching"
            ),
            "clocks": (
                "prediction statistics independently reconstructed from raw clocks; refit "
                "elapsed observations, CPU and RSS have checked arithmetic but no retained "
                "raw refit clocks"
            ),
            "memory": (
                "numeric state/model sizes are component sizes; process RSS includes loaded "
                "data and parity workspaces"
            ),
            "causality": (
                "model uses labels through 540s; replaying earlier inputs is compute/parity "
                "testing, never causal accuracy evidence"
            ),
            "limits": (
                "finite source workload only; no hard realtime or acquisition-latency "
                "guarantee; an empty threadpool inventory does not independently verify "
                "library threading"
            ),
        },
    }


def publish(manifest_path, audit_path, annotation_path, output):
    result = aggregate(
        json.loads(Path(manifest_path).read_text()),
        json.loads(Path(audit_path).read_text()),
        manifest_sha256=digest(manifest_path),
        audit_sha256=digest(audit_path),
        annotation_sha256=digest(annotation_path),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "audit", "annotation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    publish(args.manifest, args.audit, args.annotation, args.output)


if __name__ == "__main__":
    main()
