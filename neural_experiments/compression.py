"""Separate dynamic-int8 quality/runtime experiment; never same-weight parity.

One prespecified per-channel QInt8 MatMul conversion, no threshold search. Execute
only after the required primary experiment is frozen and evaluated. Code and graph
identities, all seeds/day differences and full traces remain in fresh local runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import psutil
from onnxruntime.quantization import QuantType, quantize_dynamic
from threadpoolctl import threadpool_limits

from neural_decoder.artifacts import environment, sha256, source_identity, write_json
from neural_decoder.metrics import FileEvaluation, score_files
from neural_decoder.optimized import ONNXTCNStream, export_step_onnx, model_fingerprint
from neural_decoder.runtime import replay_timed, timing_summary
from neural_decoder.temporal import load_tcn
from neural_decoder.train import FOLDS, load_development_files


def convert(frozen, folder):
    original, compressed = folder / "float32.onnx", folder / "dynamic-int8.onnx"
    export_step_onnx(frozen, original)
    # ORT's dynamic quantizer expects constant weights on the right. Rewrite
    # W @ v as v @ W.T before conversion; this keeps the float32 function intact.
    canonical = onnx.load(original)
    weights = {value.name: value for value in canonical.graph.initializer}
    for node in canonical.graph.node:
        if node.op_type != "MatMul":
            continue
        weight_name, vector_name = node.input
        tensor = weights[weight_name]
        array = onnx.numpy_helper.to_array(tensor)
        tensor.CopyFrom(onnx.numpy_helper.from_array(array.T.copy(), weight_name))
        node.input[:] = [vector_name, weight_name]
    canonical = onnx.shape_inference.infer_shapes(canonical)
    onnx.checker.check_model(canonical, full_check=True)
    quantization_input = folder / "quantization-input.onnx"
    onnx.save(canonical, quantization_input)
    quantize_dynamic(
        quantization_input,
        compressed,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    graph = onnx.load(compressed)
    properties = {p.key: p.value for p in graph.metadata_props}
    properties["precision"] = "dynamic_int8"
    properties["compression"] = "per_channel_QInt8_weights_dynamic_activation_quantization"
    onnx.helper.set_model_props(graph, properties)
    onnx.save(graph, compressed)
    operations = [node.op_type for node in graph.graph.node]
    if operations.count("MatMulInteger") != 4:
        raise ValueError("Expected all four TCN matrix operators to be quantized")
    return original, compressed, {op: operations.count(op) for op in sorted(set(operations))}


def run_quality(development_run, index_path, output, root):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _, files = load_development_files(index_path)
    base = json.loads((Path(development_run) / "manifest.json").read_text())
    if base["status"] != "complete" or base["configuration"]["seeds"] != [0, 1, 2]:
        raise ValueError("Requires completed three-seed development study")
    if base.get("pilot") or sha256(index_path) != base["index_sha256"]:
        raise ValueError("Compression inputs differ from the completed development inputs")
    manifest = {
        "status": "running",
        "source": source_identity(root, output),
        "environment": environment(),
        "index_sha256": sha256(index_path),
        "development_manifest_sha256": sha256(Path(development_run) / "manifest.json"),
        "script_sha256": sha256(__file__),
        "setting": "per-channel QInt8 MatMul, dynamic activations, no reduced range",
        "regime": "frozen inputs with explicitly approximate numerical execution",
        "runtime_versions": {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
    }
    write_json(output / "manifest.json", manifest)
    rows, artifacts = [], {}
    try:
        with threadpool_limits(limits=1):
            for seed in (0, 1, 2):
                for _, day in FOLDS:
                    source = Path(development_run) / f"seed-{seed}" / day
                    identity = json.loads((source / "identity.json").read_text())
                    if sha256(source / "model.npz") != identity["model_sha256"]:
                        raise ValueError("Development model identity changed")
                    frozen = load_tcn(source / "model.npz")
                    folder = output / f"seed-{seed}" / day
                    folder.mkdir(parents=True)
                    original, compressed, operations = convert(frozen, folder)
                    refs, quantized, max_error = [], [], 0.0
                    canonical_error = 0.0
                    for item, x, y, mask in files:
                        if item["day"] != day:
                            continue
                        decoder = ONNXTCNStream(
                            compressed,
                            expected_model_identity=model_fingerprint(frozen),
                            precision="dynamic_int8",
                        )
                        actual = decoder.predict_chunk(x)
                        reference_path = source / f"{item['file_id']}.npy"
                        if sha256(reference_path) != identity["prediction_sha256"][item["file_id"]]:
                            raise ValueError("Saved float32 prediction changed")
                        reference = np.load(reference_path, allow_pickle=False)
                        original_trace = ONNXTCNStream(
                            original, expected_model_identity=model_fingerprint(frozen)
                        ).predict_chunk(x)
                        canonical_trace = ONNXTCNStream(
                            folder / "quantization-input.onnx",
                            expected_model_identity=model_fingerprint(frozen),
                        ).predict_chunk(x)
                        # Attribute approximation to quantization only after verifying the rewrite.
                        np.testing.assert_allclose(original_trace, reference, rtol=3e-5, atol=1e-7)
                        np.testing.assert_allclose(
                            canonical_trace, original_trace, rtol=3e-5, atol=1e-7
                        )
                        canonical_error = max(
                            canonical_error, float(np.max(np.abs(canonical_trace - original_trace)))
                        )
                        max_error = max(max_error, float(np.max(np.abs(actual - reference))))
                        path = folder / f"{item['file_id']}.npy"
                        np.save(path, actual, allow_pickle=False)
                        artifacts[str(path.relative_to(output))] = sha256(path)
                        refs.append(FileEvaluation(item["file_id"], day, reference, y, mask))
                        quantized.append(FileEvaluation(item["file_id"], day, actual, y, mask))
                    original_scores, scores = score_files(refs), score_files(quantized)
                    rows.append(
                        {
                            "seed": seed,
                            "day": day,
                            "float32": original_scores,
                            "dynamic_int8": scores,
                            "delta_r2": scores["r2_mean"] - original_scores["r2_mean"],
                            "max_absolute_trace_difference": max_error,
                            "model_sha256": sha256(source / "model.npz"),
                            "float32_graph_sha256": sha256(original),
                            "int8_graph_sha256": sha256(compressed),
                            "canonical_float32_graph_sha256": sha256(
                                folder / "quantization-input.onnx"
                            ),
                            "canonical_float32_max_trace_difference": canonical_error,
                            "float32_graph_bytes": original.stat().st_size,
                            "int8_graph_bytes": compressed.stat().st_size,
                            "operations": operations,
                        }
                    )
                    print(
                        json.dumps({"seed": seed, "day": day, "delta_r2": rows[-1]["delta_r2"]}),
                        flush=True,
                    )
        worst = min(row["delta_r2"] for row in rows)
        summary = {
            "rows": rows,
            "mean_delta_r2": float(np.mean([r["delta_r2"] for r in rows])),
            "worst_seed_day_delta": worst,
            "quality_gate": worst >= -0.005,
            "gate": "no seed/day loses more than .005 absolute R2",
            "prediction_sha256": artifacts,
            "sealed_days_used_for_selection": [],
        }
        write_json(output / "summary.json", summary)
        manifest.update(status="complete", summary_sha256=sha256(output / "summary.json"))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "manifest.json", manifest)


def benchmark(quality_run, development_run, index_path, output, root):
    """Same preselected seed0/S4 graph, thread1, 500 warmup, sequential counterbalance."""
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src/neural_decoder",
            "neural_experiments/compression.py",
        ],
        cwd=root,
        text=True,
    )
    if status.strip():
        raise ValueError("Commit compression/runtime code before final measurement")
    quality_run, output = Path(quality_run), Path(output)
    manifest = json.loads((quality_run / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or sha256(quality_run / "summary.json") != manifest["summary_sha256"]
    ):
        raise ValueError("Requires completed quality experiment")
    summary = json.loads((quality_run / "summary.json").read_text())
    row = next(r for r in summary["rows"] if r["seed"] == 0 and r["day"] == "S4")
    frozen = load_tcn(Path(development_run) / "seed-0/S4/model.npz")
    if sha256(Path(development_run) / "seed-0/S4/model.npz") != row["model_sha256"]:
        raise ValueError("Preselected model bytes changed")
    if sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("Compression benchmark inputs differ from its quality experiment")
    graphs = {
        "float32": quality_run / "seed-0/S4/float32.onnx",
        "dynamic_int8": quality_run / "seed-0/S4/dynamic-int8.onnx",
    }
    for precision, key in [
        ("float32", "float32_graph_sha256"),
        ("dynamic_int8", "int8_graph_sha256"),
    ]:
        if sha256(graphs[precision]) != row[key]:
            raise ValueError("Graph identity changed")
    output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "status": "running",
        "source": source_identity(root, output),
        "environment": environment(),
        "script_sha256": sha256(__file__),
        "quality_summary_sha256": sha256(quality_run / "summary.json"),
        "index_sha256": sha256(index_path),
        "warmup_bins": 500,
        "period_ns": 20_000_000,
        "comparison": "float32 vs dynamic_int8 explicit state CPU ORT; changed arithmetic",
        "runtime_versions": {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
        "model_sha256": row["model_sha256"],
        "graph_sha256": {name: sha256(path) for name, path in graphs.items()},
    }
    write_json(output / "manifest.json", evidence)
    _, files = load_development_files(index_path)
    runs = []
    process = psutil.Process()
    try:
        with threadpool_limits(limits=1):
            for mode in ("fast", "paced"):
                for repeat, order in enumerate(
                    (("float32", "dynamic_int8"), ("dynamic_int8", "float32"))
                ):
                    for precision in order:
                        initialized = time.perf_counter_ns()
                        decoder = ONNXTCNStream(
                            graphs[precision],
                            expected_model_identity=model_fingerprint(frozen),
                            precision=precision,
                        )
                        constructor_ns = time.perf_counter_ns() - initialized
                        resetting = time.perf_counter_ns()
                        decoder.reset()
                        cold_reset_ns = time.perf_counter_ns() - resetting
                        decoder.predict_chunk(files[0][1][:500])
                        selected = (
                            files
                            if mode == "fast"
                            else [(files[0][0], files[0][1][:3001], None, None)]
                        )
                        blocks = []
                        folder = output / f"{mode}-{repeat}-{precision}"
                        folder.mkdir()
                        started = time.perf_counter()
                        cpu_start = time.process_time_ns()
                        rss_start = process.memory_info().rss
                        traces, resets = [], []
                        for item, x, _, _ in selected:
                            resetting = time.perf_counter_ns()
                            decoder.reset()
                            resets.append(time.perf_counter_ns() - resetting)
                            predictions, stamps = replay_timed(decoder, x, paced=mode == "paced")
                            path = folder / f"{item['file_id']}.npz"
                            np.savez_compressed(
                                path,
                                sequence_id=np.arange(len(x)),
                                stamps=stamps,
                                predictions=predictions,
                            )
                            traces.append(
                                {
                                    "file_id": item["file_id"],
                                    "bins": len(x),
                                    "trace_sha256": sha256(path),
                                }
                            )
                            # Compare measured outputs with a separate reset replay.
                            decoder.reset()
                            np.testing.assert_array_equal(predictions, decoder.predict_chunk(x))
                            blocks.append(stamps)
                        entry = {
                            "mode": mode,
                            "repeat": repeat,
                            "precision": precision,
                            "summary": timing_summary(np.concatenate(blocks), period_ns=20_000_000),
                            "wall_seconds": time.perf_counter() - started,
                            "constructor_ns": constructor_ns,
                            "cold_reset_ns": cold_reset_ns,
                            "file_reset_ns": resets,
                            "files": traces,
                            "state_array_bytes": decoder.state_nbytes,
                            "graph_file_bytes": graphs[precision].stat().st_size,
                            "process_rss_before_bytes": rss_start,
                            "process_rss_after_bytes": process.memory_info().rss,
                            "cpu_seconds_including_validation_and_harness": (
                                time.process_time_ns() - cpu_start
                            )
                            / 1e9,
                            "cpu_scope": (
                                "includes second parity replay and file persistence; "
                                "not decoder-only CPU"
                            ),
                        }
                        entry["one_core_cpu_percent_including_validation_and_harness"] = (
                            100
                            * entry["cpu_seconds_including_validation_and_harness"]
                            / entry["wall_seconds"]
                        )
                        runs.append(entry)
                        write_json(folder / "summary.json", entry)
                        print(
                            json.dumps(
                                {
                                    "mode": mode,
                                    "precision": precision,
                                    "p99_ms": entry["summary"]["step_service"]["p99_ms"],
                                }
                            ),
                            flush=True,
                        )
        write_json(output / "summary.json", {"runs": runs, "quality_gate": summary["quality_gate"]})
        evidence.update(status="complete", summary_sha256=sha256(output / "summary.json"))
    except BaseException as error:
        evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "manifest.json", evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["quality", "benchmark"])
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "quality":
        run_quality(args.development_run, args.index, args.output, root)
    else:
        benchmark(args.quality_run, args.development_run, args.index, args.output, root)


if __name__ == "__main__":
    main()
