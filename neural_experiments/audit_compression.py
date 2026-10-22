"""Independently audit compression identities, graph rewrite, traces and sklearn scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from audit_runtime import digest, read_json, reference_trace, verify
from sklearn.metrics import r2_score


def fingerprint(weights):
    result = hashlib.sha256()
    for name, value in sorted(weights.items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(value.shape).encode())
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def verify_transpose(original, canonical):
    """Only four constant-left matrix products and their exact weights may change."""
    original_nodes = {node.name: node for node in original.graph.node}
    canonical_nodes = {node.name: node for node in canonical.graph.node}
    if original_nodes.keys() != canonical_nodes.keys():
        raise ValueError("canonical graph node inventory changed")
    original_weights = {p.name: onnx.numpy_helper.to_array(p) for p in original.graph.initializer}
    canonical_weights = {p.name: onnx.numpy_helper.to_array(p) for p in canonical.graph.initializer}
    if original_weights.keys() != canonical_weights.keys():
        raise ValueError("canonical initializer inventory changed")
    transposed = set()
    for name, node in original_nodes.items():
        candidate = canonical_nodes[name]
        if node.op_type == "MatMul":
            if len(node.input) != 2 or node.input[0] not in original_weights:
                raise ValueError("expected constant-left matrix product")
            transformed = onnx.NodeProto()
            transformed.CopyFrom(node)
            transformed.input[:] = list(reversed(node.input))
            if transformed != candidate:
                raise ValueError("canonical matrix node differs beyond operand order")
            transposed.add(node.input[0])
        elif candidate != node:
            raise ValueError("canonical graph changed a non-matrix operation")
    if len(transposed) != 4:
        raise ValueError("expected four distinct matrix weights")
    for name, value in original_weights.items():
        np.testing.assert_array_equal(
            canonical_weights[name], value.T if name in transposed else value
        )


def session(path, identity, precision):
    graph = onnx.load(path)
    metadata = {item.key: item.value for item in graph.metadata_props}
    if (
        metadata.get("precision") != precision
        or metadata.get("frozen_fingerprint") != identity
        or metadata.get("neural_decoder_format") != "explicit_tcn_step_v1"
    ):
        raise ValueError("graph metadata does not bind requested precision and model")
    if (
        precision == "dynamic_int8"
        and sum(n.op_type == "MatMulInteger" for n in graph.graph.node) != 4
    ):
        raise ValueError("expected all four matrix products quantized")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def trace(engine, counts):
    """Independent explicit-state replay, with a fresh zero history at each file."""
    state = [np.zeros(shape, dtype=np.float32) for shape in ((2, 176), (4, 32), (8, 32))]
    predictions = np.empty((len(counts), 7), dtype=np.float32)
    for t, raw in enumerate(counts):
        feeds = {"counts": raw.astype(np.float32)}
        feeds.update({f"history{i}": history for i, history in enumerate(state)})
        prediction, *state = engine.run(None, feeds)
        predictions[t] = prediction
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite graph prediction")
    return predictions


def check_scores(saved, day, targets, predictions, *, full_bins, file_ids):
    aggregate = float(r2_score(targets, predictions, multioutput="variance_weighted"))
    per_output = r2_score(targets, predictions, multioutput="raw_values")
    np.testing.assert_allclose(saved["r2_mean"], aggregate, atol=1e-12, rtol=1e-12)
    values = saved["per_day"][day]
    np.testing.assert_allclose(values["r2_variance_weighted"], aggregate, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(values["r2_per_output"], per_output, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        values["sse_per_output"], ((targets - predictions) ** 2).sum(0), atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        values["sst_per_output"], ((targets - targets.mean(0)) ** 2).sum(0), atol=1e-12, rtol=1e-12
    )
    if (
        saved["num_bins"] != full_bins
        or saved["num_scored_bins"] != len(targets)
        or values["file_ids"] != file_ids
        or saved["num_outputs"] != 7
    ):
        raise ValueError("score accounting differs from full files and supplied masks")
    return aggregate, per_output.tolist()


def audit(quality_run, development_run, index_path):
    quality_run, development_run, index_path = map(Path, (quality_run, development_run, index_path))
    manifest = read_json(quality_run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("requires complete compression quality run")
    verify(quality_run / "summary.json", manifest["summary_sha256"])
    verify(development_run / "manifest.json", manifest["development_manifest_sha256"])
    verify(index_path, manifest["index_sha256"])
    for relative, expected in manifest["source"]["files"].items():
        verify(quality_run / "source" / relative, expected)
    verify(quality_run / "source.diff", manifest["source"]["diff_sha256"])
    index = read_json(index_path)
    if index.get("role") != "development" or any(
        item["day"] not in {"S0", "S1", "S2", "S3", "S4"}
        or item["provider_split"] != "held-in-calib"
        for item in index["files"]
    ):
        raise ValueError("refusing non-development sample arrays")
    summary = read_json(quality_run / "summary.json")
    expected_rows = [(seed, day) for seed in (0, 1, 2) for day in ("S2", "S3", "S4")]
    if [(r["seed"], r["day"]) for r in summary["rows"]] != expected_rows:
        raise ValueError("nine-model inventory changed")
    reports, observed_prediction_paths = [], set()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for row in summary["rows"]:
            seed, day = row["seed"], row["day"]
            source = development_run / f"seed-{seed}" / day
            folder = quality_run / f"seed-{seed}" / day
            identity = read_json(source / "identity.json")
            model = source / "model.npz"
            verify(model, identity["model_sha256"])
            verify(model, row["model_sha256"])
            with np.load(model, allow_pickle=False) as bundle:
                weights = {key: bundle[key] for key in bundle.files}
            paths = [
                folder / filename
                for filename in ("float32.onnx", "quantization-input.onnx", "dynamic-int8.onnx")
            ]
            for path, key in zip(
                paths,
                ("float32_graph_sha256", "canonical_float32_graph_sha256", "int8_graph_sha256"),
                strict=True,
            ):
                verify(path, row[key])
            verify_transpose(onnx.load(paths[0]), onnx.load(paths[1]))
            engines = [
                session(path, fingerprint(weights), precision)
                for path, precision in zip(
                    paths, ("float32", "float32", "dynamic_int8"), strict=True
                )
            ]
            selected = sorted(
                (item for item in index["files"] if item["day"] == day),
                key=lambda item: item["file_id"],
            )
            ys, originals, compressed, full_bins = [], [], [], 0
            max_canonical = max_quantized = max_reference = 0.0
            for item in selected:
                for key in ("features", "targets"):
                    verify(index_path.parent / item[key], item[f"{key}_sha256"])
                with np.load(index_path.parent / item["features"], allow_pickle=False) as bundle:
                    counts = bundle["counts"]
                    np.testing.assert_array_equal(bundle["sequence_id"], np.arange(len(counts)))
                with np.load(index_path.parent / item["targets"], allow_pickle=False) as bundle:
                    targets, mask = bundle["targets"], bundle["eval_mask"]
                if counts.shape != (item["n_bins"], 176) or mask.dtype != np.bool_:
                    raise ValueError("input length or score mask changed")
                file_id = item["file_id"]
                original_path, quantized_path = source / f"{file_id}.npy", folder / f"{file_id}.npy"
                verify(original_path, identity["prediction_sha256"][file_id])
                relative = str(quantized_path.relative_to(quality_run))
                verify(quantized_path, summary["prediction_sha256"][relative])
                observed_prediction_paths.add(relative)
                original = np.load(original_path, allow_pickle=False)
                quantized = np.load(quantized_path, allow_pickle=False)
                original_graph, canonical_graph, quantized_graph = [
                    trace(e, counts) for e in engines
                ]
                independent = reference_trace(counts, weights)
                np.testing.assert_allclose(original, independent, atol=1e-7, rtol=3e-5)
                np.testing.assert_allclose(original_graph, original, atol=1e-7, rtol=3e-5)
                np.testing.assert_allclose(canonical_graph, original_graph, atol=1e-7, rtol=3e-5)
                np.testing.assert_array_equal(quantized_graph, quantized)
                max_reference = max(max_reference, float(np.max(np.abs(independent - original))))
                max_canonical = max(
                    max_canonical, float(np.max(np.abs(canonical_graph - original_graph)))
                )
                max_quantized = max(max_quantized, float(np.max(np.abs(quantized - original))))
                ys.append(targets[mask].astype(np.float64))
                originals.append(original[mask].astype(np.float64))
                compressed.append(quantized[mask].astype(np.float64))
                full_bins += len(counts)
            labels = np.concatenate(ys)
            file_ids = [item["file_id"] for item in selected]
            float_r2, _ = check_scores(
                row["float32"],
                day,
                labels,
                np.concatenate(originals),
                full_bins=full_bins,
                file_ids=file_ids,
            )
            int8_r2, per_output = check_scores(
                row["dynamic_int8"],
                day,
                labels,
                np.concatenate(compressed),
                full_bins=full_bins,
                file_ids=file_ids,
            )
            np.testing.assert_allclose(row["delta_r2"], int8_r2 - float_r2, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(
                row["canonical_float32_max_trace_difference"], max_canonical, atol=1e-12, rtol=1e-12
            )
            np.testing.assert_allclose(
                row["max_absolute_trace_difference"], max_quantized, atol=1e-12, rtol=1e-12
            )
            if (
                row["float32_graph_bytes"] != paths[0].stat().st_size
                or row["int8_graph_bytes"] != paths[2].stat().st_size
            ):
                raise ValueError("reported graph bytes changed")
            reports.append(
                {
                    "seed": seed,
                    "day": day,
                    "float32_r2": float_r2,
                    "int8_r2": int8_r2,
                    "delta_r2": int8_r2 - float_r2,
                    "int8_per_output_r2": per_output,
                    "canonical_max_abs_difference": max_canonical,
                    "source_reference_max_abs_difference": max_reference,
                    "int8_trace_exact_match": True,
                }
            )
        if observed_prediction_paths != set(summary["prediction_sha256"]):
            raise ValueError("prediction evidence inventory changed")
        deltas = [r["delta_r2"] for r in reports]
        np.testing.assert_allclose(
            summary["mean_delta_r2"], np.mean(deltas), atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            summary["worst_seed_day_delta"], min(deltas), atol=1e-12, rtol=1e-12
        )
        if summary["quality_gate"] != (min(deltas) >= -0.005):
            raise ValueError("quality gate does not match prespecified worst seed/day rule")
        return {
            "status": "passed",
            "audited_utc": datetime.now(UTC).isoformat(),
            "auditor_sha256": digest(__file__),
            "reference_auditor_sha256": digest(Path(__file__).with_name("audit_runtime.py")),
            "quality_manifest_sha256": digest(quality_run / "manifest.json"),
            "quality_summary_sha256": digest(quality_run / "summary.json"),
            "models": reports,
            "quality_gate": summary["quality_gate"],
            "scope": (
                "source/graph/full-trace identities, exact transpose, sklearn day/output scores"
            ),
        }
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-run", type=Path, required=True)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.quality_run, args.development_run, args.index)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "models": len(result["models"])}), flush=True)


if __name__ == "__main__":
    main()
