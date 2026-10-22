"""Feature-ready replay of the exact exported V3 ridge model.

Finite filter state does not imply finite temporal support. Filter response
times and computation/scheduling latency are recorded as distinct quantities.
The runner opens feature files only; no targets or score masks enter inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from neural_experiments.v3 import data, model

ROOT = Path(__file__).resolve().parents[2]
PERIOD_NS = 20_000_000
WARMUP_BINS = 500


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def timestamp_summary(stamps):
    service = (stamps[:, 2] - stamps[:, 1]) / 1e6
    queue = (stamps[:, 1] - stamps[:, 0]) / 1e6
    total = (stamps[:, 2] - stamps[:, 0]) / 1e6
    return {
        "n": len(stamps),
        "service_p50_ms": float(np.quantile(service, 0.5)),
        "service_p99_ms": float(np.quantile(service, 0.99)),
        "service_max_ms": float(service.max()),
        "queue_p99_ms": float(np.quantile(queue, 0.99)),
        "release_to_completion_p99_ms": float(np.quantile(total, 0.99)),
        "deadline_misses": int(np.sum(stamps[:, 2] > stamps[:, 3])),
        "negative_queue_count": int(np.sum(stamps[:, 1] < stamps[:, 0])),
    }


def filter_response(config):
    tau = config["tau_seconds"]
    decay = float(np.exp(-model.DT / tau))
    return {
        "fast_tau_seconds": tau,
        "slow_baseline_tau_seconds": config["baseline_tau_seconds"],
        "fast_decay_per_bin": decay,
        "fast_steady_state_mean_sample_age_seconds": model.DT * decay / (1 - decay),
        "fast_95_percent_step_response_bins": int(np.ceil(np.log(0.05) / np.log(decay))),
        "temporal_support": "infinite exponentially decaying past, bounded sufficient state",
        "startup": "fast and slow states initialize to first transformed observed sample per block",
        "interpretation": (
            "Fast-filter response descriptors are not compute latency or total group delay; "
            "optional slow-baseline subtraction has a distinct frequency response"
        ),
    }


def process_peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def measure(decoder_model, tx, sbp, boundaries, *, paced, count, warmup=WARMUP_BINS, clock=time):
    """One complete-step block; injectable clock supports deterministic unit tests."""
    if not len(tx) or count < 1:
        raise ValueError("nonempty workload and positive count required")
    started = clock.perf_counter_ns()
    decoder = model.StreamingDecoder(decoder_model)
    initialization_ns = clock.perf_counter_ns() - started
    for i in range(warmup):
        row = i % len(tx)
        if row in boundaries:
            decoder.reset()
        decoder.step(tx[row], sbp[row])
    decoder.reset()
    stamps = np.empty((count, 4), dtype=np.int64)
    predictions = np.empty((count, 2), dtype=np.float64)
    origin = clock.perf_counter_ns() + (PERIOD_NS if paced else 0)
    cpu_start = clock.process_time_ns()
    wall_start = clock.perf_counter_ns()
    for i in range(count):
        release = origin + i * PERIOD_NS if paced else clock.perf_counter_ns()
        if paced:
            remaining = release - clock.perf_counter_ns()
            if remaining > 0:
                clock.sleep(remaining / 1e9)
        start = clock.perf_counter_ns()
        row = i % len(tx)
        if row in boundaries:
            decoder.reset()
        prediction = decoder.step(tx[row], sbp[row])
        completion = clock.perf_counter_ns()
        stamps[i] = release, start, completion, release + PERIOD_NS
        predictions[i] = prediction
    wall_ns = clock.perf_counter_ns() - wall_start
    cpu_ns = clock.process_time_ns() - cpu_start
    return (
        stamps,
        predictions,
        {
            "initialization_ms": initialization_ns / 1e6,
            "initialization_scope": "stream constructor after model loading, before warmup",
            "persistent_numeric_state_bytes": decoder.state_nbytes,
            "state_measurement": "fast/slow arrays; excludes model, Python, scratch and RSS",
            "process_cpu_seconds": cpu_ns / 1e9,
            "block_wall_seconds": wall_ns / 1e9,
            "process_cpu_fraction_of_block_wall": cpu_ns / wall_ns,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
            "rss_scope": "process high-water mark including loaded inputs and parity workspaces",
        },
    )


def load_workload(model_path, feature_path):
    started = time.perf_counter_ns()
    decoder_model = model.Decoder.load(model_path)
    load_ms = (time.perf_counter_ns() - started) / 1e6
    if decoder_model.config["kind"] != "regularized":
        raise ValueError("runtime requires selected regularized V3 export")
    features = data.load_features(feature_path)
    tx, sbp = features["x_tx"][:30000], features["x_sbp"][:30000]
    if len(tx) == 0:
        raise ValueError("empty timing workload")
    sequence = features["sequence_id"][: len(tx)]
    bounds = np.r_[0, np.flatnonzero(np.diff(sequence)) + 1, len(tx)]
    segments = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    expected = decoder_model.predict(tx, sbp, segments)
    decoder = model.StreamingDecoder(decoder_model)
    actual = np.empty_like(expected)
    boundaries = {a for a, _ in segments}
    for i in range(len(tx)):
        if i in boundaries:
            decoder.reset()
        actual[i] = decoder.step(tx[i], sbp[i])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    return (
        decoder_model,
        tx,
        sbp,
        boundaries,
        expected,
        float(np.max(np.abs(actual - expected))),
        load_ms,
    )


def run(model_path, feature_path, plan_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(plan_path).read_text())
    cfg = plan["runtime"]
    if (cfg["fast_repeats"], cfg["paced_blocks"], cfg["paced_seconds"], cfg["period_ms"]) != (
        3,
        2,
        60,
        20,
    ):
        raise ValueError("runtime plan differs from locked three-fast/two-60-second protocol")
    manifest = {
        "schema_version": 1,
        "kind": "v3_regularized_ridge",
        "status": "running",
        "model_path": relative(model_path),
        "model_sha256": digest(model_path),
        "feature_path": relative(feature_path),
        "feature_sha256": digest(feature_path),
        "plan_path": relative(plan_path),
        "plan_sha256": digest(plan_path),
        "runtime_source_sha256": digest(__file__),
        "protocol": {
            "period_ms": 20,
            "warmup_bins": WARMUP_BINS,
            "fast_repeats": 3,
            "paced_blocks_per_implementation": 2,
            "paced_seconds": 60,
            "implementation": "numpy_incremental",
            "feature_prefix_cap_bins": 30000,
            "parity_atol": 1e-10,
            "parity_rtol": 1e-10,
            "service_boundary": (
                "raw validation, transforms, filter, normalization, linear head, "
                "projection, state, block reset"
            ),
            "exclude": "feature file loading, acquisition, upstream feature extraction, transport",
            "scheduler": "ordinary sleep until fixed nominal release; no busy waiting",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threads": 1,
        },
        "source": {},
        "source_snapshots": {},
        "blocks": [],
    }
    for path in (
        Path(__file__),
        Path(model.__file__),
        Path(data.__file__),
        ROOT / "neural_experiments/v2/data.py",
    ):
        name = relative(path)
        snapshot = output / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
        manifest["source"][name] = digest(path)
        manifest["source_snapshots"][name] = relative(snapshot)
    write_json(output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            decoder_model, tx, sbp, boundaries, expected, error, load_ms = load_workload(
                model_path, feature_path
            )
            manifest["environment"]["threadpools"] = [
                {
                    k: pool[k]
                    for k in ("user_api", "internal_api", "num_threads", "version")
                    if k in pool
                }
                for pool in threadpool_info()
            ]
            manifest["protocol"]["workload_bins"] = len(tx)
            manifest["full_trace_parity"] = {"numpy_incremental_vs_batch": error}
            manifest["config"] = decoder_model.config
            manifest["filter_response"] = filter_response(decoder_model.config)
            manifest["model_load_ms"] = load_ms
            manifest["parameter_count"] = (
                decoder_model.coefficients.size + decoder_model.intercept.size
            )
            manifest["numeric_weight_bytes"] = (
                decoder_model.coefficients.nbytes + decoder_model.intercept.nbytes
            )
            manifest["numeric_transform_bytes"] = (
                decoder_model.mean.nbytes + decoder_model.scale.nbytes
            )
            manifest["numeric_model_array_bytes"] = (
                manifest["numeric_weight_bytes"] + manifest["numeric_transform_bytes"]
            )
            write_json(output / "manifest.json", manifest)
            schedule = [("fast", repeat) for repeat in range(3)] + [
                ("paced", repeat) for repeat in range(2)
            ]
            for mode, repeat in schedule:
                count = len(tx) if mode == "fast" else 3000
                stamps, prediction, details = measure(
                    decoder_model,
                    tx,
                    sbp,
                    boundaries,
                    paced=mode == "paced",
                    count=count,
                )
                np.testing.assert_allclose(
                    prediction, expected[np.arange(count) % len(expected)], rtol=1e-10, atol=1e-10
                )
                name = f"{mode}-{repeat}.npz"
                np.savez(output / name, timestamps_ns=stamps, predictions=prediction)
                manifest["blocks"].append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "implementation": "numpy_incremental",
                        "artifact_path": relative(output / name),
                        "artifact_sha256": digest(output / name),
                        **timestamp_summary(stamps),
                        **details,
                    }
                )
                write_json(output / "manifest.json", manifest)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "repeat": repeat,
                            "service_p99_ms": manifest["blocks"][-1]["service_p99_ms"],
                        }
                    ),
                    flush=True,
                )
        manifest["status"] = "complete"
        manifest["scope"] = (
            "feature-ready single-model replay; finite observed deadline test, "
            "not hard real-time or neural acquisition latency"
        )
        write_json(output / "manifest.json", manifest)
        write_json(
            output / "summary.json",
            {**manifest, "manifest_sha256": digest(output / "manifest.json")},
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(output / "manifest.json", manifest)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.features, args.plan, args.output)


if __name__ == "__main__":
    main()
