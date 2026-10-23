"""Independent centered-moment, bounded-state and raw-clock engineering audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit_runtime import percentile, require
from neural_experiments.v7.audit import (
    CONFIG,
    ROOT,
    burned_mask,
    compare_structure,
    contained,
    mask_hash,
    numeric,
    read,
    reference_representation,
    segment_ranges,
    sha256,
    verify,
)
from neural_experiments.v7.audit_continual import PLAN_SHA, fit_reference
from neural_experiments.v7.audit_runtime_continual import binding


def clock_summary(stamps, first, last):
    require(
        stamps.dtype == np.int64 and stamps.ndim == 2 and len(stamps) > 0,
        "int64 raw clocks required",
    )
    values = [(int(row[last]) - int(row[first])) / 1e6 for row in stamps]
    require(min(values) >= 0, "negative clock duration")
    return {
        "n": len(values),
        "p50_ms": percentile(values, 0.5),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
        "total_ms": sum(values),
    }


def moments_reference(raw, y, use, stop, mean, scale):
    x = np.clip((raw[:stop][use] - mean) / scale, -8, 8)
    labels = np.asarray(y[:stop][use], np.float64)
    if len(x):
        xm, ym = x.mean(0), labels.mean(0)
        xx, yy = x - xm, labels - ym
    else:
        xm, ym = np.zeros(raw.shape[1]), np.zeros(2)
        xx, yy = x, labels
    return {
        "normalizer_mean": mean,
        "normalizer_scale": scale,
        "x_mean": xm,
        "y_mean": ym,
        "centered_gram": xx.T @ xx,
        "centered_rhs": xx.T @ yy,
        "count": np.array(len(x), np.int64),
        "last_bin": np.array(stop - 1, np.int64),
    }


def validate_moments(observed, expected):
    require(set(observed) == set(expected), "fixed-size moment state fields")
    errors = []
    for name, value in expected.items():
        actual = observed[name]
        require(
            actual.shape == value.shape and actual.dtype == value.dtype, "moment state geometry"
        )
        require(
            np.isfinite(actual).all() and np.allclose(actual, value, rtol=1e-10, atol=1e-8),
            "moment value mismatch: " + name,
        )
        if name in ("count", "last_bin"):
            require(np.array_equal(actual, value), "moment integer state")
        errors.append(float(np.max(np.abs(actual - value))))
    return max(errors)


def update_geometry_reference(geometry, updates, solves, paid, eligible, stops):
    require(
        geometry.dtype == np.int64 and geometry.ndim == 2 and geometry.shape[1] == 4,
        "update geometry must contain start/stop/before/after",
    )
    require(updates.shape == (len(geometry), 4) and solves.shape == (5, 6), "raw clock inventory")
    require(updates.dtype == solves.dtype == np.int64, "int64 clocks")
    start, count = 1500, 0
    for i, (a, b, before, after) in enumerate(geometry):
        require(
            a == start and 1 <= b - a <= 64 and before == count,
            "bounded contiguous chronological updates",
        )
        require(not any(a < stop < b for stop in stops), "chunk crosses solve boundary")
        count += int(eligible[a:b].sum())
        require(after == count, "paid eligible update count")
        require(
            updates[i, 0] <= updates[i, 1] and updates[i, 2] <= updates[i, 3],
            "negative update clocks",
        )
        if i:
            require(
                updates[i, 0] >= updates[i - 1, 1] and updates[i, 2] >= updates[i - 1, 3],
                "overlapping updates",
            )
        start = int(b)
    require(
        start == stops[-1] and all(stop in geometry[:, 1] for stop in stops),
        "complete update/solve coverage",
    )
    for i, stop in enumerate(stops):
        require(
            np.all(np.diff(solves[i, :3]) >= 0) and np.all(np.diff(solves[i, 3:]) >= 0),
            "negative solve clocks",
        )
        j = int(np.flatnonzero(geometry[:, 1] == stop)[0])
        require(
            solves[i, 0] >= updates[j, 1] and solves[i, 3] >= updates[j, 3],
            "solve precedes final update",
        )
        if j + 1 < len(geometry):
            require(
                updates[j + 1, 0] >= solves[i, 2] and updates[j + 1, 2] >= solves[i, 5],
                "next update overlaps solve/export",
            )
    require(np.all(~eligible | paid), "training labels outside purchased intervals")


def audit_run(run, root=ROOT):
    root, run = Path(root).resolve(), Path(run).resolve()
    path = run / "manifest.json"
    manifest = read(path)
    require(
        manifest["schema_version"] == 1
        and manifest["status"] == "complete"
        and manifest["kind"] == "V7B_source_incremental_ridge",
        "completed engineering replay required",
    )
    sources = {
        "neural_experiments/v7/incremental.py",
        "neural_experiments/v7/incremental_runtime.py",
    }
    require(set(manifest["source_sha256"]) == sources, "engineering source inventory")
    for name, digest in manifest["source_sha256"].items():
        verify(contained(root, name), digest)
        verify(contained(run / "source_snapshot", name), digest)
    protocol = {
        "max_chunk_rows": 64,
        "parity_atol": 1e-10,
        "parity_rtol": 1e-10,
        "feature_penalty": 1,
        "free_intercept": True,
        "label_policy_changed": False,
        "normalization_changed": False,
        "solve_count": 5,
        "replay_count": 1,
    }
    compare_structure(manifest["protocol"], protocol, "engineering protocol")
    compare_structure(
        manifest["representative"],
        {
            "participant": "T2",
            "post_implant_day": 250,
            "method": "periodic240",
            "threshold_context": 0.1,
        },
        "engineering representative",
    )
    source = read(contained(root, manifest["source_manifest_path"]))
    index_path = contained(root, manifest["index_path"])
    index = read(index_path)
    item = max(
        (s for s in index["sessions"] if s["participant_id"] == "T2"),
        key=lambda s: (s["post_implant_day"], s["session_id"]),
    )
    events = sorted(
        (
            e
            for e in source["events"]
            if e["session_id"] == item["session_id"]
            and e["method"] == "periodic240"
            and e["threshold"] == 0.1
            and e["event"] == "model_fitted"
        ),
        key=lambda e: e["event_bin"],
    )
    require(len(events) == 5, "all five source exports required")
    row = next(
        r
        for r in source["rows"]
        if r["session_id"] == item["session_id"]
        and r["method"] == "periodic240"
        and r["threshold"] == 0.1
    )
    # Reuse the separate identity checker; the extension records a shorter identity.
    bound = {
        **manifest,
        "plan_sha256": PLAN_SHA,
        "source_sha256": source["code_sha256"],
        "representative": {
            **manifest["representative"],
            "session_id": item["session_id"],
            "fit_stop_seconds": 540,
            "deployment_seconds": 542,
            "training_rows": events[-1]["fit"]["training_rows"],
            "selection_rule": (
                "greatest numeric exposed T2 day; last complete periodic fit; no outcome selection"
            ),
        },
        "artifacts": {},
    }
    for name, record, prefix, base in (
        ("model", events[-1], "model", root),
        ("trace", row, "trace", root),
        ("features", item, "feature", index_path.parent),
        ("targets", item, "target", index_path.parent),
    ):
        resolved = contained(base, record[prefix + "_path"])
        bound["artifacts"][name] = {
            "path": str(resolved.relative_to(root)),
            "sha256": record[prefix + "_sha256"],
        }
    paths, _, _ = binding(bound, root)
    features, labels = numeric(paths["features"]), numeric(paths["targets"])
    segments = segment_ranges(item["segments"], item["n_bins"])
    raw = reference_representation(features, segments, CONFIG)
    original = numeric(paths["model"])
    mean, scale = original["mean"], original["scale"]
    paid = np.zeros(item["n_bins"], bool)
    for a, b in events[-1]["fit"]["purchased_intervals"]:
        paid[a:b] = True
    eligible = paid & burned_mask(labels["eval_mask"], segments)
    require(len(manifest["fits"]) == 5, "five saved moment/model states")
    moments_error, coefficient_error, residuals = [], [], []
    d = raw.shape[1]
    moment_bytes = 8 * (d + d * d + 2 * d + 2 + 2)
    normalizer_bytes = 16 * d
    for ordinal, (fit, event) in enumerate(zip(manifest["fits"], events, strict=True)):
        stop = event["fit"]["fit_stop_bin"]
        require(
            fit["ordinal"] == ordinal
            and fit["fit_stop_bin"] == stop
            and fit["deployment_bin"] == event["event_bin"],
            "fit chronology",
        )
        require(fit["source_model_sha256"] == event["model_sha256"], "original export binding")
        require(
            fit["training_rows"] == int(eligible[:stop].sum()) == event["fit"]["training_rows"],
            "eligible label count",
        )
        require(
            fit["selected_mask_sha256"] == mask_hash(eligible[:stop]), "eligible paid-prefix digest"
        )
        model_path = contained(root, fit["model_path"])
        moments_path = contained(root, fit["moments_path"])
        require(
            model_path.is_relative_to(run) and moments_path.is_relative_to(run),
            "retained engineering artifacts",
        )
        verify(model_path, fit["model_sha256"])
        verify(moments_path, fit["moments_sha256"])
        moments_error.append(
            validate_moments(
                numeric(moments_path),
                moments_reference(raw, labels["y"], eligible[:stop], stop, mean, scale),
            )
        )
        arrays = numeric(model_path)
        source_path = contained(root, event["model_path"])
        verify(source_path, event["model_sha256"])
        expected = numeric(source_path)
        require(
            np.allclose(arrays["weights"], expected["weights"], rtol=1e-10, atol=1e-10),
            "incremental coefficient parity",
        )
        error = float(np.max(np.abs(arrays["weights"] - expected["weights"])))
        require(
            np.isclose(error, fit["coefficient_max_error"], atol=1e-15, rtol=0),
            "coefficient error declaration",
        )
        coefficient_error.append(error)
        base = {
            k: v
            for k, v in event["fit"].items()
            if k not in {"deployment_bin", "purchased_intervals", "purchased_mask_sha256"}
        }
        residuals.append(
            fit_reference(
                arrays,
                base,
                raw,
                labels,
                segments,
                item["n_channels"],
                stop,
                event["fit"]["purchased_intervals"],
            )
        )
        require(
            fit["moment_bytes"] == moment_bytes and fit["normalizer_bytes"] == normalizer_bytes,
            "fixed state memory",
        )
    clocks_path = contained(root, manifest["clocks"]["path"])
    verify(clocks_path, manifest["clocks"]["sha256"])
    clocks = numeric(clocks_path)
    require(
        set(clocks) == {"update_geometry", "update_timestamps_ns", "solve_timestamps_ns"},
        "raw clock fields",
    )
    updates, solves = clocks["update_timestamps_ns"], clocks["solve_timestamps_ns"]
    update_geometry_reference(
        clocks["update_geometry"],
        updates,
        solves,
        paid,
        eligible,
        [e["fit"]["fit_stop_bin"] for e in events],
    )
    timing = {
        name: clock_summary(stamps, a, b)
        for name, stamps, a, b in (
            ("update_wall", updates, 0, 1),
            ("update_cpu", updates, 2, 3),
            ("solve_wall", solves, 0, 1),
            ("solve_cpu", solves, 3, 4),
            ("solve_export_reload_wall", solves, 0, 2),
            ("solve_export_reload_cpu", solves, 3, 5),
        )
    }
    compare_structure(manifest["timing"], timing, "raw engineering clock arithmetic")
    memory = {
        "moment_bytes": moment_bytes,
        "normalizer_bytes": normalizer_bytes,
        "persistent_calibration_bytes": moment_bytes + normalizer_bytes,
        "raw_and_target_chunk_buffers_bytes": 64 * (d + 2) * 8,
        "normalization_startup_buffer_bytes": 1500 * d * 8,
        "causal_filter_state_bytes": 2 * d * 8,
        "batch_full_prefix_representation_bytes": item["n_bins"] * d * 8,
        "feature_dimension": d,
        "scope": (
            "numeric calibration state and explicit buffers only; validation loaded source "
            "inputs, masks and references are excluded; arithmetic scratch and Python "
            "overhead excluded"
        ),
    }
    compare_structure(manifest["memory"], memory, "numeric component memory")
    geometry = clocks["update_geometry"]
    label_bearing = updates[geometry[:, 3] > geometry[:, 2]]
    purchased_timing = {
        "chunks_adding_labels": len(label_bearing),
        "wall": clock_summary(label_bearing, 0, 1) if len(label_bearing) else None,
        "cpu": clock_summary(label_bearing, 2, 3) if len(label_bearing) else None,
    }
    return {
        "status": "PASS",
        "auditor_source_sha256": sha256(__file__),
        "manifest_sha256": sha256(path),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "source_audit_sha256": manifest["source_audit_sha256"],
        "fits_checked": 5,
        "update_chunks_checked": len(updates),
        "max_moment_absolute_error": max(moments_error),
        "max_coefficient_absolute_error": max(coefficient_error),
        "max_normal_equation_relative_residual": max(residuals),
        "memory": memory,
        "timing": timing,
        "label_bearing_update_timing": purchased_timing,
        "scope": (
            "Independent batch centered-moment reconstruction, paid-label coverage, five "
            "original-export/ridge optimum checks, numeric memory and raw-clock arithmetic. "
            "Source-only engineering evidence; no efficacy or fixed realtime guarantee."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve audit receipts")
    result = audit_run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
