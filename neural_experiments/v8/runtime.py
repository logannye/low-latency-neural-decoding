"""Bounded portable-decoder timing, separate from frozen V8 efficacy producers.

Replaying a model fitted through150s on the first60s is a numerical/compute check,
never an accuracy result. No targets or context sidecars are opened here.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from neural_experiments.v3.study import digest, relative, write_json
from neural_experiments.v6.data import _study_lock
from neural_experiments.v8 import data, model

ROOT = Path(__file__).resolve().parents[2]
COUNT = 3000
REPEATS = 2
PERIOD_NS = 20_000_000
WARMUP_STEPS = 500


def checked(root, path, expected):
    root, path = Path(root).resolve(), Path(path)
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or digest(resolved) != expected:
        raise ValueError("runtime artifact path or identity mismatch")
    return resolved


def bind(source_path, index_path, audit_path, *, root=ROOT):
    """Archive-rank16 source model; outcome-independent representative selection."""
    source, index, audit = [
        json.loads(Path(p).read_text()) for p in (source_path, index_path, audit_path)
    ]
    if (
        source.get("status") != "complete"
        or source.get("study") != "V8"
        or source.get("phase") != "development"
        or index.get("role") != "development"
        or source["index"]["sha256"] != digest(index_path)
        or audit.get("status") != "PASS"
        or audit.get("phase") != "development"
        or audit.get("manifest_sha256") != digest(source_path)
        or audit.get("index_sha256") != digest(index_path)
        or audit.get("plan_sha256") != source["plan"]["sha256"]
    ):
        raise ValueError("completed independently audited exact V8 development run required")
    for path, checksum in source["code_sha256"].items():
        checked(root, path, checksum)
    items = sorted(
        [item for item in index["sessions"] if item["participant_id"] == "T2"],
        key=lambda item: item["source"]["archive_ordinal"],
    )
    if len(items) != 21 or len({v["source"]["archive_ordinal"] for v in items}) != 21:
        raise ValueError("exact21-member ordered T2 source inventory required")
    item, primary = items[15], source["selection"]["primary_method"]
    if item["n_bins"] < COUNT:
        raise ValueError("declared source representative has insufficient timing rows")
    rows = [
        r
        for r in source["factorial_rows"]
        if r["session_id"] == item["session_id"] and r["method"] == primary
    ]
    if len(rows) != 1:
        raise ValueError("unique representative primary exported model required")
    row = rows[0]
    paths = {
        "model": checked(root, row["model"]["model_path"], row["model"]["model_sha256"]),
        "features": checked(Path(index_path).parent, item["feature_path"], item["feature_sha256"]),
    }
    if row["config"] != source["selection"]["configurations"][primary]:
        raise ValueError("representative differs from selected primary configuration")
    return item, row, paths


def public_summary(manifest, audit, *, manifest_sha256, audit_sha256):
    """Explicit aggregate allowlist; raw clocks, paths and arrays cannot escape."""
    if (
        manifest.get("status") != "complete"
        or manifest.get("kind") != "source_runtime"
        or audit.get("status") != "PASS"
        or audit.get("manifest_sha256") != manifest_sha256
        or audit.get("blocks_checked") != REPEATS
        or len(manifest.get("blocks", [])) != REPEATS
    ):
        raise ValueError("exact completed independently audited runtime required")

    def number(value):
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative scalar aggregate required")
        return value

    def checksum(value):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError("exact SHA256 required")
        return value

    fields = [
        "n",
        "deadline_misses",
        "block_wall_seconds",
        "process_cpu_seconds",
        "numeric_filter_state_bytes",
        "prediction_max_absolute_error",
    ]
    fields += [
        f"{scope}_{stat}_ms"
        for scope in ("service", "queue", "release_to_completion")
        for stat in ("p50", "p95", "p99", "max")
    ]
    if [b["repeat"] for b in manifest["blocks"]] != [0, 1]:
        raise ValueError("both declared timing runs required")
    if any(b["n"] != COUNT for b in manifest["blocks"]):
        raise ValueError("complete3000-step blocks required")
    representative = manifest["representative"]
    if representative["participant"] != "T2" or representative["archive_rank"] != 16:
        raise ValueError("declared source representative required")
    model.validate_config(representative["config"])
    environment = {}
    for key in ("python", "numpy", "platform"):
        value = manifest["environment"][key]
        if not isinstance(value, str) or any(c in value for c in ("/", "\\", "\n", "\r")):
            raise ValueError("scalar environment descriptor required")
        environment[key] = value
    return {
        "schema_version": 1,
        "evidence_role": "source numerical and compute replay; no accuracy evidence",
        "manifest_sha256": checksum(manifest_sha256),
        "audit_sha256": checksum(audit_sha256),
        "source_manifest_sha256": checksum(manifest["source_manifest_sha256"]),
        "source_audit_sha256": checksum(manifest["source_audit_sha256"]),
        "plan_sha256": checksum(manifest["plan_sha256"]),
        "model_sha256": checksum(manifest["artifacts"]["model"]["sha256"]),
        "feature_sha256": checksum(manifest["artifacts"]["features"]["sha256"]),
        "representative": {
            "participant": "T2",
            "archive_rank": 16,
            "post_implant_day": number(representative["post_implant_day"]),
            "config": {key: representative["config"][key] for key in sorted(model.CONFIG_KEYS)},
        },
        "protocol": {
            "repeats": REPEATS,
            "steps_each": COUNT,
            "period_ms": PERIOD_NS / 1e6,
            "warmup_steps": WARMUP_STEPS,
            "threadpool_limit": 1,
        },
        "environment": environment,
        "memory": {
            key: number(manifest["memory"][key])
            for key in (
                "model_numeric_bytes",
                "model_artifact_bytes",
                "feature_channels",
                "filtered_dimensions",
                "workload_rows",
            )
        },
        "blocks": [
            {"repeat": b["repeat"], **{key: number(b[key]) for key in fields}}
            for b in manifest["blocks"]
        ],
        "limitations": (
            "Model fitted through150s replayed onfirst60s; no causal accuracy. Feature-ready "
            "steps only; excludes acquisition, upstream extraction, transport, loading and "
            "fitting. Two finite runs on one source session; no hard realtime guarantee. "
            "Numeric component sizes exclude Python overhead, scratch arrays and input cache."
        ),
    }


def timestamp_summary(stamps):
    stamps = np.asarray(stamps)
    if (
        stamps.dtype != np.int64
        or stamps.ndim != 2
        or stamps.shape[1] != 4
        or not len(stamps)
        or np.any(stamps[:, 1] < stamps[:, 0])
        or np.any(stamps[:, 2] < stamps[:, 1])
        or np.any(stamps[:, 3] - stamps[:, 0] != PERIOD_NS)
        or (len(stamps) > 1 and np.any(np.diff(stamps[:, 0]) != PERIOD_NS))
    ):
        raise ValueError("ordered valid raw fixed-period monotonic clocks required")
    values = {
        "service": stamps[:, 2] - stamps[:, 1],
        "release_to_completion": stamps[:, 2] - stamps[:, 0],
        "queue": stamps[:, 1] - stamps[:, 0],
    }
    result = {"n": len(stamps), "deadline_misses": int((stamps[:, 2] > stamps[:, 3]).sum())}
    for name, value in values.items():
        for percentile in (50, 95, 99):
            result[f"{name}_p{percentile}_ms"] = float(np.quantile(value / 1e6, percentile / 100))
        result[f"{name}_max_ms"] = float(value.max() / 1e6)
    return result


def measure(decoder, features, segments, *, count=COUNT, warmup=WARMUP_STEPS, clock=time):
    """Clock-injectable paced replay; no input samples are dropped after a miss."""
    n = len(features["x_tx3"])
    if type(count) is not int or not 1 <= count <= n or type(warmup) is not int or warmup < 0:
        raise ValueError("positive bounded row count and nonnegative warmup required")
    core_segments = [(a, min(b, count)) for a, b in segments if a < count]
    if not core_segments or core_segments[0][0] != 0 or core_segments[-1][1] != count:
        raise ValueError("segments must cover the replay prefix")
    boundaries = {a for a, _ in core_segments}
    stream = model.StreamingDecoder(decoder)
    for i in range(warmup):
        row = i % count
        if row in boundaries:
            stream.reset()
        stream.step({key: value[row] for key, value in features.items()})
    stream.reset()
    stamps, predictions = np.empty((count, 4), np.int64), np.empty((count, 2), np.float64)
    wall_start, cpu_start = clock.perf_counter_ns(), clock.process_time_ns()
    origin = clock.perf_counter_ns() + PERIOD_NS
    for row in range(count):
        release = origin + row * PERIOD_NS
        remaining = release - clock.perf_counter_ns()
        while remaining > 0:
            clock.sleep(remaining / 1e9)
            remaining = release - clock.perf_counter_ns()
        started = clock.perf_counter_ns()
        if row in boundaries:
            stream.reset()
        prediction = stream.step({key: value[row] for key, value in features.items()})
        completed = clock.perf_counter_ns()
        stamps[row] = release, started, completed, release + PERIOD_NS
        predictions[row] = prediction
    cpu_end, wall_end = clock.process_time_ns(), clock.perf_counter_ns()
    return (
        stamps,
        predictions,
        {
            "wall_start_ns": wall_start,
            "wall_end_ns": wall_end,
            "cpu_start_ns": cpu_start,
            "cpu_end_ns": cpu_end,
            "block_wall_seconds": (wall_end - wall_start) / 1e9,
            "process_cpu_seconds": (cpu_end - cpu_start) / 1e9,
            "numeric_filter_state_bytes": stream.state.nbytes + stream.decay.nbytes,
        },
    )


def run(source_path, index_path, audit_path, output, *, plan_sha256):
    plan_path = Path(__file__).with_name("runtime-plan.json")
    _study_lock(plan_path, plan_sha256)
    # Both timing implementation and plan must be committed before measurement.
    _study_lock(Path(__file__), digest(__file__))
    plan = json.loads(plan_path.read_text())
    if any(
        plan[key] != value
        for key, value in (
            ("count", COUNT),
            ("repeats", REPEATS),
            ("period_ns", PERIOD_NS),
            ("warmup_steps", WARMUP_STEPS),
            ("threadpool_limit", 1),
        )
    ):
        raise ValueError("runtime source and committed protocol disagree")
    item, row, paths = bind(source_path, index_path, audit_path)
    source = json.loads(Path(source_path).read_text())
    source_hashes = {**source["code_sha256"], relative(__file__): digest(__file__)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for path in source_hashes:
        destination = output / "source_snapshot" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    shutil.copyfile(plan_path, output / "runtime-plan.json")
    manifest = {
        "schema_version": 1,
        "study": "V8",
        "kind": "source_runtime",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "plan_sha256": plan_sha256,
        "source_manifest_sha256": digest(source_path),
        "source_audit_sha256": digest(audit_path),
        "index_sha256": digest(index_path),
        "source_sha256": source_hashes,
        "source_manifest_path": relative(source_path),
        "source_audit_path": relative(audit_path),
        "index_path": relative(index_path),
        "representative": {
            "participant": "T2",
            "session_id": item["session_id"],
            "post_implant_day": item["post_implant_day"],
            "archive_rank": 16,
            "method": row["method"],
            "config": row["config"],
        },
        "artifacts": {
            key: {"path": relative(path), "sha256": digest(path)} for key, path in paths.items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threadpool_limit": 1,
        },
        "blocks": [],
        "scope": (
            "Source-only numerical/compute testing; model uses labels later than replay inputs."
        ),
    }
    write_json(output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            fitted = model.Decoder.load(paths["model"])
            if fitted.config != row["config"]:
                raise ValueError("loaded model configuration changed")
            loaded = data.load_features(paths["features"])
            features = {key: loaded[key][:COUNT] for key in ("x_tx3", "x_tx45", "x_sbp")}
            segments = [
                (s["start"], min(s["end"], COUNT)) for s in item["segments"] if s["start"] < COUNT
            ]
            expected = fitted.predict(features, segments)
            manifest["environment"]["threadpools"] = [
                {
                    key: pool[key]
                    for key in ("user_api", "internal_api", "num_threads", "version")
                    if key in pool
                }
                for pool in threadpool_info()
            ]
            manifest["memory"] = {
                "model_numeric_bytes": sum(
                    getattr(fitted, key).nbytes for key in ("mean", "scale", "weights")
                ),
                "model_artifact_bytes": paths["model"].stat().st_size,
                "feature_channels": fitted.channels,
                "filtered_dimensions": len(fitted.mean),
                "workload_rows": COUNT,
            }
            for repeat in range(REPEATS):
                stamps, predictions, clocks = measure(fitted, features, segments)
                np.testing.assert_allclose(predictions, expected, rtol=1e-10, atol=1e-10)
                path = output / f"paced-{repeat}.npz"
                np.savez_compressed(path, clocks=stamps, prediction=predictions)
                manifest["blocks"].append(
                    {
                        "repeat": repeat,
                        **timestamp_summary(stamps),
                        **clocks,
                        "prediction_max_absolute_error": float(
                            np.max(np.abs(predictions - expected))
                        ),
                        "artifact_path": relative(path),
                        "artifact_sha256": digest(path),
                    }
                )
                write_json(output / "manifest.json", manifest)
                print(json.dumps({"repeat": repeat, "status": "complete"}), flush=True)
        if any(digest(ROOT / path) != checksum for path, checksum in source_hashes.items()):
            raise ValueError("source changed during runtime measurement")
        if digest(plan_path) != plan_sha256:
            raise ValueError("runtime plan changed during measurement")
        manifest.update(status="complete", finished_at=datetime.now(UTC).isoformat())
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(
            status="failed",
            failure_type=type(error).__name__,
            finished_at=datetime.now(UTC).isoformat(),
        )
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "index", "audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    result = run(args.source, args.index, args.audit, args.output, plan_sha256=args.plan_sha256)
    print(json.dumps({"status": result["status"], "blocks": len(result["blocks"])}))
