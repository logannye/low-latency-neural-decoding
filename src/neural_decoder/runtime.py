"""Counts-ready replay, arithmetic gates and retained batch-one timing evidence.

Only ``benchmark`` performs the prescribed final timing campaign. It refuses an
uncommitted implementation/protocol. Profiling is instrumented diagnostic work,
and validation is arithmetic/quality evidence, neither a latency measurement.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.metadata
import io
import json
import math
import os
import pstats
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from .artifacts import environment, sha256, source_identity, write_json
from .metrics import FileEvaluation, score_files
from .optimized import ONNXTCNStream, export_step_onnx, model_fingerprint
from .temporal import TCNStream, load_tcn
from .train import load_development_files

IMPLEMENTATIONS = ("torch_window", "numpy_incremental", "onnx_incremental")
STAMP_FIELDS = ("release_ns", "start_ns", "prediction_ready_ns", "complete_ns", "deadline_ns")


def make_decoder(frozen, implementation, onnx_path):
    if implementation == "onnx_incremental":
        return ONNXTCNStream(onnx_path, expected_model_identity=model_fingerprint(frozen))
    if implementation == "torch_window":
        return TCNStream(frozen, "window")
    if implementation == "numpy_incremental":
        return TCNStream(frozen, "incremental")
    raise ValueError("unknown runtime implementation")


def timing_summary(stamps, *, period_ns):
    """Recompute all latency/queue/deadline fields from retained monotonic stamps."""
    stamps = np.asarray(stamps)
    if stamps.ndim != 2 or stamps.shape[1] != 5 or not len(stamps):
        raise ValueError("expected a nonempty [bins,5] timing array")
    if stamps.dtype.kind not in "iu" or period_ns < 1:
        raise ValueError("timestamps and interval must be integer nanoseconds")
    release, start, ready, complete, deadline = stamps.T
    if np.any(start < release) or np.any(ready < start) or np.any(complete < ready):
        raise ValueError("invalid timing event order")
    if np.any(start[1:] < complete[:-1]) or np.any(release[1:] < release[:-1]):
        raise ValueError("timing events violate sequential replay order")
    if not np.array_equal(deadline, release + period_ns):
        raise ValueError("deadline is not the declared next-bin boundary")

    def distribution(values):
        values = values.astype(np.float64) / 1e6
        return {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(values.max()),
        }

    queue = start - release
    return {
        "bins": len(stamps),
        "step_service": distribution(complete - start),
        "prediction_service": distribution(ready - start),
        "queue_delay": distribution(queue),
        "release_to_prediction": distribution(ready - release),
        "release_to_completion": distribution(complete - release),
        "missed_deadlines": int(np.count_nonzero(complete > deadline)),
        "max_backlog_bins": int((queue // period_ns).max()),
        "nominal_neural_seconds": float(len(stamps) * period_ns / 1e9),
        "first_start_to_last_completion_seconds": float((complete[-1] - start[0]) / 1e9),
    }


def replay_timed(
    decoder, counts, *, paced, period_ns=20_000_000, clock=time.perf_counter_ns, sleep=time.sleep
):
    """Time complete steps; logging/copying the returned outputs is outside service time.

    Frozen decoders have no post-prediction adaptation: prediction ready and step
    completion are the same event. Paced release is the scheduled availability
    time; sleep/host scheduling lateness appears in queue and end-to-end metrics.
    No input bin is skipped or replaced when the process falls behind.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 176 or not len(counts):
        raise ValueError("nonempty [bins,176] trace required")
    if not np.isfinite(counts).all() or (counts < 0).any() or period_ns < 1:
        raise ValueError("invalid count trace or bin period")
    stamps = np.empty((len(counts), 5), dtype=np.int64)
    predictions = np.empty((len(counts), 7), dtype=np.float32)
    origin = clock()
    for sequence_id, row in enumerate(counts):
        if paced:
            release = origin + sequence_id * period_ns
            while (remaining := release - clock()) > 0:
                sleep(remaining / 1e9)
            start = clock()
        else:
            start = clock()
            release = start
        prediction = decoder.step(row)
        complete = clock()
        stamps[sequence_id] = (release, start, complete, complete, release + period_ns)
        predictions[sequence_id] = prediction
    return predictions, stamps


def validate_runtime(frozen, onnx_path, files, destination, config):
    """Validate every full development file and quality before accepting timings."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model = frozen.to_torch()
    references, scores, errors = [], {}, {}
    factories = {name: make_decoder(frozen, name, onnx_path) for name in IMPLEMENTATIONS}
    records = {name: [] for name in IMPLEMENTATIONS}
    for item, counts, targets, mask in files:
        values = counts.astype(np.float32) - frozen.arrays["input_mean"]
        values /= frozen.arrays["input_scale"]
        with torch.inference_mode():
            reference = model(torch.from_numpy(values)[None])[0].numpy()
        reference = reference * frozen.arrays["target_scale"] + frozen.arrays["target_mean"]
        file_id, day = item["file_id"], item["day"]
        np.save(destination / f"{file_id}.reference.npy", reference, allow_pickle=False)
        references.append(FileEvaluation(file_id, day, reference, targets, mask))
        for name, decoder in factories.items():
            decoder.reset()
            actual = decoder.predict_chunk(counts)
            np.save(destination / f"{file_id}.{name}.npy", actual, allow_pickle=False)
            np.testing.assert_allclose(
                actual, reference, rtol=config["parity_rtol"], atol=config["parity_atol"]
            )
            errors.setdefault(name, {})[file_id] = float(np.max(np.abs(actual - reference)))
            records[name].append(FileEvaluation(file_id, day, actual, targets, mask))
    reference_score = score_files(references)
    for name in IMPLEMENTATIONS:
        scores[name] = score_files(records[name])
        deltas = [
            abs(
                scores[name]["per_day"][day]["r2_variance_weighted"] - value["r2_variance_weighted"]
            )
            for day, value in reference_score["per_day"].items()
        ]
        if max(deltas) > config["quality_numerical_tolerance"]:
            raise ValueError(f"{name} quality differs beyond numerical tolerance")
    result = {
        "status": "passed",
        "max_absolute_errors": errors,
        "scores": scores,
        "reference_scores": reference_score,
        "interpretation": "arithmetic comparison on development inputs, not new efficacy evidence",
    }
    write_json(destination / "parity.json", result)
    return result


def profile_baselines(frozen, counts, destination):
    """Bounded instrumented hotspot diagnosis only; never report these as latency."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in IMPLEMENTATIONS[:2]:
        decoder = make_decoder(frozen, name, None)
        profiler = cProfile.Profile()
        profiler.runcall(decoder.predict_chunk, counts[:256])
        profiler.dump_stats(str(destination / f"{name}.prof"))
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(30)
        (destination / f"{name}.txt").write_text(buffer.getvalue())
    write_json(
        destination / "scope.json",
        {
            "kind": "instrumented_hotspot_diagnostic",
            "bins": min(256, len(counts)),
            "not_a_latency_benchmark": True,
            "hypothesis": (
                "cache layer activations and replace repeated Python/kernel dispatch "
                "with one optimized graph"
            ),
        },
    )


def _require_committed(root):
    checked = [
        "src/neural_decoder",
        "tests/neural_decoder",
        "pyproject.toml",
        "neural_experiments/configs/runtime-v1.json",
        "neural_experiments/RUNTIME_PROTOCOL.md",
    ]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *checked], cwd=root, text=True
    )
    if status.strip():
        raise ValueError("commit the runtime implementation/tests/protocol before final timing")


def kernel_probe(frozen, counts, destination, *, iterations=10_000):
    """First-layer Python-invoked GEMV probe, never a complete-step substitute."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    a = frozen.arrays
    history = (counts[:3].astype(np.float32) - a["input_mean"]) / a["input_scale"]
    if history.shape != (3, 176) or iterations < 1:
        raise ValueError("kernel probe requires three input bins and positive iterations")
    flat_weight = np.ascontiguousarray(a["conv1_weight"].transpose(0, 2, 1).reshape(32, -1))
    flat_history = history.ravel()

    def split():
        value = a["conv1_bias"].copy()
        for tap in range(3):
            value += a["conv1_weight"][:, :, tap] @ history[tap]
        return np.maximum(value, 0)

    def flattened():
        return np.maximum(flat_weight @ flat_history + a["conv1_bias"], 0)

    np.testing.assert_allclose(split(), flattened(), rtol=3e-5, atol=3e-6)
    calls = {"three_strided_gemv": split, "one_contiguous_gemv": flattened}
    results = []
    for repeat, order in enumerate((tuple(calls), tuple(reversed(calls)))):
        for name in order:
            function = calls[name]
            for _ in range(500):
                function()
            samples = np.empty(iterations, dtype=np.int64)
            for i in range(iterations):
                start = time.perf_counter_ns()
                function()
                samples[i] = time.perf_counter_ns() - start
            path = destination / f"repeat{repeat}-{name}.npy"
            np.save(path, samples, allow_pickle=False)
            results.append(
                {
                    "name": name,
                    "repeat": repeat,
                    "p50_ns": float(np.percentile(samples, 50)),
                    "p99_ns": float(np.percentile(samples, 99)),
                    "max_ns": int(samples.max()),
                    "raw_sha256": sha256(path),
                }
            )
    write_json(
        destination / "summary.json",
        {
            "results": results,
            "scope": (
                "first-layer Python-invoked matrix-vector products; excludes other decoder costs"
            ),
        },
    )
    return results


def _validate_config(config):
    if config["schema_version"] != 1 or config["threads"] != 1:
        raise ValueError("runtime protocol requires schema1 and one thread")
    if config["warmup_bins"] != 500 or config["period_ns"] != 20_000_000:
        raise ValueError("prespecified warmup or interval changed")
    if config["paced_seconds_per_block"] < 60:
        raise ValueError("paced blocks must span at least60 seconds")
    for mode in ("fast_orders", "paced_orders"):
        if len(config[mode]) < 2:
            raise ValueError("counterbalanced repeated orders required")
        for order in config[mode]:
            if len(order) != 3 or set(order) != set(IMPLEMENTATIONS):
                raise ValueError("each block must contain every implementation once")
    if len(config["fast_orders"]) != 3 or any(
        len({order[position] for order in config["fast_orders"]}) != 3 for position in range(3)
    ):
        raise ValueError("fast implementation order must use a complete Latin rotation")
    if (
        len(config["paced_orders"]) != 2
        or config["paced_orders"][0] != config["paced_orders"][1][::-1]
    ):
        raise ValueError("paced implementation orders must be a reversed pair")
    if config["kernel_iterations"] != 10_000:
        raise ValueError("prespecified kernel iterations changed")


def run_runtime(model_path, onnx_path, index_path, config_path, output, *, root):
    """Run after source freeze; retain failed/partial runs and never overwrite them."""
    config = json.loads(Path(config_path).read_text())
    _validate_config(config)
    if sha256(model_path) != config["primary_model_sha256"]:
        raise ValueError("runtime model differs from the prespecified seed0 artifact")
    _require_committed(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "configuration": config,
        "environment": environment(),
        "source": source_identity(root, output),
        "model_sha256": sha256(model_path),
        "onnx_sha256": sha256(onnx_path),
        "index_sha256": sha256(index_path),
        "config_sha256": sha256(config_path),
        "boundary": "counts ready to prediction and all frozen-decoder state work complete",
    }
    manifest["runtime_dependencies"] = {
        name: importlib.metadata.version(name) for name in ("onnx", "onnxruntime", "psutil")
    }
    write_json(output / "manifest.json", manifest)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        frozen = load_tcn(model_path)
        index, files = load_development_files(index_path)
        if len(files) != 11:
            raise ValueError("runtime workload must contain all11 development files")
        write_json(output / "input_index.json", index)
        if sum(len(record[1]) for record in files) < 20_000:
            raise ValueError("final fast replay requires at least20,000 real bins")
        warmup = files[0][1][: config["warmup_bins"]]
        if len(warmup) != 500:
            raise ValueError("first development file is too short for prespecified warmup")
        with threadpool_limits(limits=1):
            manifest["threadpools"] = threadpool_info()
            parity = validate_runtime(frozen, onnx_path, files, output / "parity", config)
            manifest["parity_status"] = parity["status"]
            process = psutil.Process()
            results = []
            for mode in ("fast", "paced"):
                for repeat, order in enumerate(config[f"{mode}_orders"]):
                    for position, name in enumerate(order):
                        folder = output / f"{mode}-repeat{repeat}-{position}-{name}"
                        folder.mkdir()
                        init_start = time.perf_counter_ns()
                        decoder = make_decoder(frozen, name, onnx_path)
                        init_ns = time.perf_counter_ns() - init_start
                        reset_start = time.perf_counter_ns()
                        decoder.reset()
                        cold_reset_ns = time.perf_counter_ns() - reset_start
                        decoder.predict_chunk(warmup)
                        if mode == "paced":
                            bins = (
                                math.ceil(
                                    config["paced_seconds_per_block"] * 1e9 / config["period_ns"]
                                )
                                + 1
                            )
                            if bins > len(files[0][1]):
                                raise ValueError("paced trace exceeds first development file")
                            selected = [(files[0][0], files[0][1][:bins])]
                        else:
                            selected = [(item, counts) for item, counts, _, _ in files]
                        rss_start = process.memory_info().rss
                        system_load_start = os.getloadavg()
                        cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
                        blocks, ids, reset_samples = [], [], []
                        for file_index, (item, counts) in enumerate(selected):
                            reset_start = time.perf_counter_ns()
                            decoder.reset()
                            reset_samples.append(time.perf_counter_ns() - reset_start)
                            prediction, stamps = replay_timed(
                                decoder,
                                counts,
                                paced=mode == "paced",
                                period_ns=config["period_ns"],
                            )
                            # The measured trace must also preserve predictions.
                            reference = np.load(
                                output / "parity" / f"{item['file_id']}.reference.npy",
                                allow_pickle=False,
                            )
                            np.testing.assert_allclose(
                                prediction,
                                reference[: len(counts)],
                                rtol=config["parity_rtol"],
                                atol=config["parity_atol"],
                            )
                            np.savez(
                                folder / f"{item['file_id']}.npz",
                                predictions=prediction,
                                stamps=stamps,
                                sequence_id=np.arange(len(counts)),
                                stamp_fields=np.asarray(STAMP_FIELDS),
                            )
                            blocks.append(stamps)
                            ids.append(
                                {
                                    "file_index": file_index,
                                    "file_id": item["file_id"],
                                    "bins": len(counts),
                                    "trace_sha256": sha256(folder / f"{item['file_id']}.npz"),
                                }
                            )
                        cpu_ns = time.process_time_ns() - cpu_start
                        wall_ns = time.perf_counter_ns() - wall_start
                        summary = timing_summary(
                            np.concatenate(blocks), period_ns=config["period_ns"]
                        )
                        summary.update(
                            {
                                "implementation": name,
                                "mode": mode,
                                "repeat": repeat,
                                "position": position,
                                "constructor_ns": init_ns,
                                "cold_reset_ns": cold_reset_ns,
                                "file_reset_ns": reset_samples,
                                "files": ids,
                                "cpu_seconds_including_harness": cpu_ns / 1e9,
                                "wall_seconds_including_harness": wall_ns / 1e9,
                                "one_core_cpu_percent_including_harness": 100 * cpu_ns / wall_ns,
                                "process_rss_before_bytes": rss_start,
                                "process_rss_after_bytes": process.memory_info().rss,
                                "system_load_average_start": system_load_start,
                                "system_load_average_end": os.getloadavg(),
                                "logical_cpu_count": psutil.cpu_count(),
                                "total_system_memory_bytes": psutil.virtual_memory().total,
                                "process_os_threads": process.num_threads(),
                                "state_array_bytes": decoder.state_nbytes,
                                "model_parameter_bytes": frozen.parameter_count * 4,
                                "frozen_array_bytes": sum(v.nbytes for v in frozen.arrays.values()),
                                "onnx_file_bytes": Path(onnx_path).stat().st_size,
                            }
                        )
                        write_json(folder / "summary.json", summary)
                        results.append(summary)
                        write_json(output / "summary.partial.json", results)
                        print(
                            json.dumps(
                                {
                                    "mode": mode,
                                    "repeat": repeat,
                                    "implementation": name,
                                    "p99_ms": summary["step_service"]["p99_ms"],
                                }
                            ),
                            flush=True,
                        )
                        del decoder
            kernel_probe(
                frozen, files[0][1], output / "kernel", iterations=config["kernel_iterations"]
            )
            write_json(
                output / "summary.json",
                {
                    "runs": results,
                    "interpretation": (
                        "replay service time, not neural acquisition or live closed-loop latency"
                    ),
                },
            )
            manifest.update(
                {"status": "complete", "summary_sha256": sha256(output / "summary.json")}
            )
    except BaseException as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        torch.set_num_threads(old_threads)
        write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "profile", "validate", "benchmark"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("neural_experiments/configs/runtime-v1.json")
    )
    args = parser.parse_args()
    if args.action == "export":
        if args.onnx is None:
            parser.error("export requires --onnx")
        print(json.dumps(export_step_onnx(load_tcn(args.model), args.onnx)), flush=True)
        return
    if args.index is None or args.output is None:
        parser.error("profile, validate and benchmark require --index and --output")
    if args.action == "benchmark":
        if args.onnx is None:
            parser.error("benchmark requires --onnx")
        run_runtime(
            args.model,
            args.onnx,
            args.index,
            args.config,
            args.output,
            root=Path(__file__).resolve().parents[2],
        )
        return
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, files = load_development_files(args.index)
        frozen = load_tcn(args.model)
        with threadpool_limits(limits=1):
            if args.action == "profile":
                profile_baselines(frozen, files[0][1], args.output)
            else:
                if args.onnx is None:
                    parser.error("validate requires --onnx")
                config = json.loads(args.config.read_text())
                validate_runtime(frozen, args.onnx, files, args.output, config)
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    main()
