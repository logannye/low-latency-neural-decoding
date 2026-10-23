"""Independent V7B source-only runtime identity, replay and clock arithmetic.

Prediction blocks retain raw clocks. Refit receipts retain elapsed durations,
not raw clocks: only their arithmetic, quantiles and exported numerical result
can be independently checked. This does not turn timing into efficacy evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit_runtime import percentile, reconstruct, require
from neural_experiments.v4.runtime_audit import inspect_features, inspect_model
from neural_experiments.v5.runtime_audit import close_predictions, independent_prediction
from neural_experiments.v7.audit import (
    ROOT,
    compare_structure,
    contained,
    numeric,
    read,
    sha256,
    verify,
)
from neural_experiments.v7.audit_continual import CONFIG, PLAN_SHA

SOURCES = {
    "neural_experiments/v7/runtime_continual.py",
    "neural_experiments/v7/model.py",
    "neural_experiments/v4/runtime.py",
    "neural_experiments/v4/model.py",
    "neural_experiments/v4/data.py",
    "neural_experiments/v2/data.py",
}


def binding(manifest, root):
    paths = {}
    for name in ("source_manifest", "source_audit", "index"):
        paths[name] = contained(root, manifest[name + "_path"])
        verify(paths[name], manifest[name + "_sha256"])
    source, passed, index = [read(paths[k]) for k in ("source_manifest", "source_audit", "index")]
    require(
        source["status"] == "complete" and source["phase"] == "source" and source["study"] == "V7B",
        "source role/status",
    )
    require(passed["status"] == "PASS" and passed["stage"] == "source", "source audit status")
    for key, expected in (
        ("manifest_sha256", sha256(paths["source_manifest"])),
        ("index_sha256", sha256(paths["index"])),
        ("plan_sha256", PLAN_SHA),
        ("code_sha256", source["code_sha256"]),
    ):
        require(passed[key] == expected, "source audit binding: " + key)
    require(manifest["plan_sha256"] == source["plan"]["sha256"] == PLAN_SHA, "study plan")
    verify(contained(root, source["plan"]["path"]), PLAN_SHA)
    require(
        contained(root, source["index"]["path"]) == paths["index"]
        and source["index"]["sha256"] == sha256(paths["index"]),
        "source index",
    )
    require(index["role"] == "development", "source-only workload")
    item = max(
        (s for s in index["sessions"] if s["participant_id"] == "T2"),
        key=lambda s: (s["post_implant_day"], s["session_id"]),
    )
    require(item["post_implant_day"] == 250 and item["n_bins"] == 30000, "fixed latest source day")
    fits = [
        e
        for e in source["events"]
        if e["session_id"] == item["session_id"]
        and e["method"] == "periodic240"
        and e["threshold"] == 0.1
        and e["event"] == "model_fitted"
    ]
    require(
        sorted(e["event_bin"] for e in fits) == [7600, 12100, 18100, 24100, 27100],
        "fixed five periodic fits",
    )
    event = max(fits, key=lambda e: e["event_bin"])
    require(event["fit"]["fit_stop_bin"] == 27000, "last fit cutoff")
    rows = [
        r
        for r in source["rows"]
        if r["session_id"] == item["session_id"]
        and r["method"] == "periodic240"
        and r["threshold"] == 0.1
    ]
    require(len(rows) == 1, "unique source trace")
    expected = {
        "participant": "T2",
        "session_id": item["session_id"],
        "post_implant_day": 250,
        "method": "periodic240",
        "threshold_context": 0.1,
        "fit_stop_seconds": 540,
        "deployment_seconds": 542,
        "training_rows": event["fit"]["training_rows"],
        "selection_rule": (
            "greatest numeric exposed T2 day; last complete periodic fit; no outcome selection"
        ),
    }
    compare_structure(manifest["representative"], expected, "fixed source representative")
    declarations = {
        "model": (event, "model", root),
        "trace": (rows[0], "trace", root),
        "features": (item, "feature", paths["index"].parent),
        "targets": (item, "target", paths["index"].parent),
    }
    require(set(manifest["artifacts"]) == set(declarations), "runtime artifact inventory")
    for name, (record, prefix, base) in declarations.items():
        reference = manifest["artifacts"][name]
        path = contained(root, reference["path"])
        require(
            path == contained(base, record[prefix + "_path"])
            and reference["sha256"] == record[prefix + "_sha256"],
            "representative artifact: " + name,
        )
        verify(path, reference["sha256"])
        paths[name] = path
    for name, digest in source["code_sha256"].items():
        verify(contained(root, name), digest)
    for name in SOURCES & source["code_sha256"].keys():
        require(
            manifest["source_sha256"][name] == source["code_sha256"][name],
            "frozen numerical dependency",
        )
    return paths, item, event


def refits_reference(rows, original, training_rows, root, run):
    require([r["repeat"] for r in rows] == list(range(10)), "ten ordered refits required")
    errors = []
    for row in rows:
        for name in ("fit_ns", "export_reload_ns", "total_ns"):
            require(type(row[name]) is int and row[name] >= 0, "invalid recorded refit duration")
        require(
            row["total_ns"] == row["fit_ns"] + row["export_reload_ns"], "refit duration arithmetic"
        )
        require(row["training_rows"] == training_rows, "refit eligible labels")
        path = contained(root, row["model_path"])
        require(path.is_relative_to(run), "refit outside retained run")
        verify(path, row["model_sha256"])
        exported = numeric(path)
        require(set(exported) == set(original), "refit model fields")
        errors_for_model = []
        for name, value in original.items():
            require(
                exported[name].shape == value.shape and exported[name].dtype == value.dtype,
                "refit model geometry",
            )
            if name in ("mean", "scale", "weights"):
                require(
                    np.isfinite(exported[name]).all()
                    and np.allclose(exported[name], value, atol=1e-10, rtol=1e-10),
                    "refit numerical mismatch",
                )
                errors_for_model.append(float(np.max(np.abs(exported[name] - value))))
            else:
                require(np.array_equal(exported[name], value), "refit configuration mismatch")
        error = max(errors_for_model)
        require(
            np.isclose(error, row["numeric_max_absolute_error"], atol=1e-15, rtol=0),
            "refit parity declaration",
        )
        errors.append(error)
    values = [r["total_ns"] / 1e6 for r in rows]
    summary = {
        "n": 10,
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
        "exceeded_two_second_allowance": sum(v > 2000 for v in values),
    }
    return summary, max(errors)


def audit_run(run, root=ROOT):
    run, root = Path(run).resolve(), Path(root).resolve()
    path = run / "manifest.json"
    manifest = read(path)
    require(
        manifest["schema_version"] == 1
        and manifest["status"] == "complete"
        and manifest["kind"] == "V7B_source_runtime",
        "completed V7B runtime required",
    )
    require(
        set(manifest["source_sha256"]) == SOURCES and set(manifest["source_snapshots"]) == SOURCES,
        "runtime source inventory",
    )
    for name, digest in manifest["source_sha256"].items():
        snapshot = contained(root, manifest["source_snapshots"][name])
        require(snapshot.is_relative_to(run / "source_snapshot"), "retained runtime snapshot")
        verify(snapshot, digest)
        verify(contained(root, name), digest)
    paths, item, event = binding(manifest, root)
    config, mean, scale, weights, channels = inspect_model(paths["model"])
    require(
        config == CONFIG and inspect_features(paths["features"], channels) == 30000,
        "model/workload schema",
    )
    require(
        manifest["numeric_model_array_bytes"] == mean.nbytes + scale.nbytes + weights.nbytes,
        "numeric model size",
    )
    expected_protocol = {
        "fast_repeats": 3,
        "paced_repeats": 2,
        "paced_seconds_each": 60,
        "refit_repeats": 10,
        "period_ms": 20,
        "warmup_bins": 500,
        "threads": 1,
        "parity_atol": 1e-10,
        "parity_rtol": 1e-10,
        "refit_allowance_seconds": 2,
        "timing_scope": (
            "feature-ready steps; cached causal features and loaded labels for refit, "
            "including mask, normalizer, ridge, export/reload"
        ),
    }
    compare_structure(manifest["protocol"], expected_protocol, "timing protocol")
    require(
        all(p["num_threads"] == 1 for p in manifest["environment"]["threadpools"]),
        "runtime thread count",
    )
    for name in (
        "numpy_incremental_vs_batch_max_error",
        "post_deployment_vs_source_trace_max_error",
    ):
        require(
            np.isfinite(manifest[name]) and 0 <= manifest[name] <= 2e-10,
            "declared prediction parity",
        )
    expected = independent_prediction(paths["features"], config, mean, scale, weights)
    assessed = numeric(paths["trace"])["prediction"]
    source_error = close_predictions(
        assessed[event["event_bin"] :],
        expected[event["event_bin"] :],
        "post-deployment causal trace",
    )
    require(
        [(r["mode"], r["repeat"]) for r in manifest["blocks"]]
        == [("fast", i) for i in range(3)] + [("paced", i) for i in range(2)],
        "timing block inventory",
    )
    blocks, errors = [], []
    for block in manifest["blocks"]:
        artifact = contained(root, block["path"])
        require(artifact.is_relative_to(run), "timing artifact outside run")
        verify(artifact, block["sha256"])
        arrays = numeric(artifact)
        require(set(arrays) == {"timestamps_ns", "predictions"}, "timestamp/prediction fields")
        n = 3000 if block["mode"] == "paced" else item["n_bins"]
        require(len(arrays["timestamps_ns"]) == n, "timing workload length")
        errors.append(close_predictions(arrays["predictions"], expected[:n], "timed prediction"))
        computed = reconstruct(
            arrays["timestamps_ns"], paced=block["mode"] == "paced", period_ns=20000000
        )
        compare_structure({k: block[k] for k in computed}, computed, "raw-clock arithmetic")
        require(
            block["persistent_numeric_state_bytes"] == 2 * mean.nbytes, "streaming numeric state"
        )
        for key in (
            "initialization_ms",
            "process_cpu_seconds",
            "block_wall_seconds",
            "process_peak_rss_bytes",
        ):
            require(np.isfinite(block[key]) and block[key] >= 0, "invalid resource observation")
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
        blocks.append({"mode": block["mode"], "repeat": block["repeat"], **computed})
    summary, refit_error = refits_reference(
        manifest["refits"], numeric(paths["model"]), event["fit"]["training_rows"], root, run
    )
    summary["cached_representation_bytes"] = item["n_bins"] * len(mean) * 8
    compare_structure(manifest["refit_summary"], summary, "refit summary arithmetic")
    return {
        "status": "PASS",
        "auditor_source_sha256": sha256(__file__),
        "timestamp_reference_sha256": sha256(root / "neural_experiments/v2/audit_runtime.py"),
        "prediction_reference_sha256": sha256(root / "neural_experiments/v5/runtime_audit.py"),
        "manifest_sha256": sha256(path),
        "source_manifest_sha256": sha256(paths["source_manifest"]),
        "source_audit_sha256": sha256(paths["source_audit"]),
        "model_sha256": sha256(paths["model"]),
        "feature_sha256": sha256(paths["features"]),
        "blocks_checked": 5,
        "refits_checked": 10,
        "blocks": blocks,
        "refit_summary": summary,
        "refit_max_numeric_error": refit_error,
        "post_deployment_source_trace_max_error": source_error,
        "timed_prediction_max_error": max(errors),
        "scope": (
            "Independent source representative/model binding, feature-only recurrence and "
            "five raw-clock blocks. Ten refit exports match audited coefficients; elapsed "
            "refit durations and CPU/RSS are producer observations, with arithmetic checked "
            "but no raw refit clocks. Source-only compute proof; no fresh efficacy or "
            "hard realtime claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve existing audit receipts")
    result = audit_run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
