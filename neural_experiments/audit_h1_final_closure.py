"""Independent final-H1 conversion audit: identities, graph arithmetic and scores.

The auditor never imports the quality producer or production scoring functions.
It uses the frozen authorized assessment loader, an independent whole-file Torch
reference, direct ONNX Runtime feeds, and sklearn plus separate float64 sums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from neural_decoder.assessment_data import load_assessment_files
from neural_experiments.audit_runtime import reference_trace

ARMS = ("A", "B", "C", "D", "E")
FLOAT_ARMS = ARMS[:4]
RTOL, ATOL, SCORE_ATOL, MAX_LOSS = 3e-5, 1e-7, 1e-5, 0.005
AUDITOR_FILES = (
    "neural_experiments/audit_h1_final_closure.py",
    "neural_experiments/audit_runtime.py",
    "src/neural_decoder/assessment_data.py",
    "src/neural_decoder/data.py",
    "src/neural_decoder/temporal.py",
)
MODEL_WEIGHTS = ("conv1_weight", "conv2_weight", "conv3_weight", "head_weight")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def bound_path(base, relative):
    base, relative = Path(base).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("artifact path must be relative and contained")
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError("artifact symlink escapes its evidence directory")
    return path


def verify(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise ValueError(f"changed evidence: {path}")


def assert_equal(actual, expected, location="evidence"):
    """Exact structures, tolerant independent floating arithmetic, no extra fields."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{location}: field inventory differs")
        for key in expected:
            assert_equal(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{location}: list inventory differs")
        for index, (value, wanted) in enumerate(zip(actual, expected, strict=True)):
            assert_equal(value, wanted, f"{location}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        if not isinstance(actual, (int, float)) or not np.isfinite(actual):
            raise ValueError(f"{location}: nonfinite/non-numeric metric")
        if not np.isclose(actual, expected, atol=1e-12, rtol=1e-12):
            raise ValueError(f"{location}: independently recomputed metric differs")
    elif actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        raise ValueError(f"{location}: value differs")


def fingerprint(weights):
    result = hashlib.sha256()
    for name, value in sorted(weights.items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(value.shape).encode())
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def independent_scores(files, predictions):
    """Concatenate scored rows per day; retain every count and output statistic."""
    rows = sorted(files, key=lambda row: (row[0]["day"], row[0]["file_id"]))
    if {item["file_id"] for item, *_ in rows} != set(predictions):
        raise ValueError("prediction file inventory changed")
    grouped = {}
    for item, counts, targets, mask in rows:
        pred = np.asarray(predictions[item["file_id"]])
        if (
            pred.shape != targets.shape
            or pred.shape != (len(counts), 7)
            or mask.dtype != np.bool_
            or mask.shape != (len(counts),)
            or not np.isfinite(pred).all()
            or not np.isfinite(targets[mask]).all()
        ):
            raise ValueError("invalid full trace, target or scoring mask")
        grouped.setdefault(item["day"], []).append((item, pred, targets, mask))
    per_day = {}
    for day, group in grouped.items():
        y = np.concatenate([row[2][row[3]] for row in group]).astype(np.float64)
        p = np.concatenate([row[1][row[3]] for row in group]).astype(np.float64)
        if len(y) < 2:
            raise ValueError("day needs at least two scored rows")
        sse = np.einsum("ij,ij->j", y - p, y - p)
        centered = y - np.mean(y, axis=0)
        sst = np.einsum("ij,ij->j", centered, centered)
        bins = sum(len(row[1]) for row in group)
        per_day[day] = {
            "r2_variance_weighted": float(r2_score(y, p, multioutput="variance_weighted")),
            "r2_per_output": r2_score(y, p, multioutput="raw_values").tolist(),
            "sse_per_output": sse.tolist(),
            "sst_per_output": sst.tolist(),
            "zero_variance_outputs": np.flatnonzero(sst == 0).tolist(),
            "num_scored_bins": len(y),
            "num_bins": bins,
            "num_unscored_bins": bins - len(y),
            "num_files": len(group),
            "file_ids": [row[0]["file_id"] for row in group],
            "nominal_duration_seconds": bins * 0.02,
            "scored_duration_seconds": len(y) * 0.02,
        }
    values = [row["r2_variance_weighted"] for row in per_day.values()]
    total = sum(row["num_bins"] for row in per_day.values())
    scored = sum(row["num_scored_bins"] for row in per_day.values())
    return {
        "schema_version": 1,
        "metric": "mean_day_variance_weighted_r2",
        "evaluation_kind": "local_evaluation",
        "upstream_pin": "770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f",
        "canonical_day_order": list(per_day),
        "chronological_day_order": sorted(per_day, key=lambda day: int(day[1:])),
        "num_days": len(per_day),
        "num_files": len(rows),
        "num_outputs": 7,
        "num_bins": total,
        "num_scored_bins": scored,
        "num_unscored_bins": total - scored,
        "nominal_bin_seconds": 0.02,
        "nominal_duration_seconds": total * 0.02,
        "scored_duration_seconds": scored * 0.02,
        "r2_mean": float(np.mean(values)),
        "r2_population_std": float(np.std(values)),
        "r2_worst_day": min(values),
        "worst_day": min(per_day, key=lambda day: per_day[day]["r2_variance_weighted"]),
        "per_day": per_day,
    }


def check_trace(saved, reference, *, exact=False):
    if saved.dtype != np.float32 or saved.shape != reference.shape or not np.isfinite(saved).all():
        raise ValueError("invalid prediction trace shape, precision or values")
    match = (
        np.array_equal(saved, reference)
        if exact
        else np.allclose(saved, reference, rtol=RTOL, atol=ATOL)
    )
    if not match:
        raise ValueError("prediction trace differs from independent inference")
    return float(np.max(np.abs(saved - reference)))


def _unique_nodes(graph):
    nodes = {node.name: node for node in graph.graph.node}
    if len(nodes) != len(graph.graph.node):
        raise ValueError("duplicate graph node identity")
    return nodes


def _weights(graph):
    values = {p.name: onnx.numpy_helper.to_array(p) for p in graph.graph.initializer}
    if len(values) != len(graph.graph.initializer):
        raise ValueError("duplicate graph initializer identity")
    return values


def verify_graph_identity(graph, weights, precision, *, rewritten=False):
    onnx.checker.check_model(graph, full_check=True)
    metadata = {item.key: item.value for item in graph.metadata_props}
    if len(metadata) != len(graph.metadata_props):
        raise ValueError("duplicate graph metadata")
    expected = {
        "neural_decoder_format": "explicit_tcn_step_v1",
        "frozen_fingerprint": fingerprint(weights),
        "precision": precision,
        "state_contract": "chronological_input_histories_2x176_4x32_8x32",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("graph metadata/model identity differs")
    if [(op.domain, op.version) for op in graph.opset_import] != [("", 17)]:
        raise ValueError("graph opset differs")
    names = (
        ("counts", "history0", "history1", "history2"),
        ("prediction", "next_history0", "next_history1", "next_history2"),
    )
    shapes = (((176,), (2, 176), (4, 32), (8, 32)), ((7,), (2, 176), (4, 32), (8, 32)))
    for nodes, expected_names, expected_shapes in zip(
        (graph.graph.input, graph.graph.output), names, shapes, strict=True
    ):
        if len(nodes) != 4:
            raise ValueError("graph interface inventory differs")
        for node, name, shape in zip(nodes, expected_names, expected_shapes, strict=True):
            tensor = node.type.tensor_type
            if (
                node.name != name
                or tensor.elem_type != onnx.TensorProto.FLOAT
                or tuple(dim.dim_value for dim in tensor.shape.dim) != shape
            ):
                raise ValueError("graph must retain float32 input/output/state interface")
    _unique_nodes(graph)
    _weights(graph)
    if precision == "float32":
        actual = _weights(graph)
        wanted = {name: value for name, value in weights.items() if name != "format_version"}
        if actual.keys() != wanted.keys():
            raise ValueError("graph/model initializer inventory differs")
        for name, value in wanted.items():
            if name.startswith("conv") and name.endswith("weight"):
                value = value.transpose(0, 2, 1).reshape(32, -1)
            if rewritten and name in MODEL_WEIGHTS:
                value = value.T
            if not np.array_equal(value, actual[name]):
                raise ValueError(f"graph weight differs from frozen model: {name}")


def verify_transpose(original, rewritten):
    """D may differ only in four exact weight transposes/operands and inferred shapes."""
    original_nodes, rewritten_nodes = _unique_nodes(original), _unique_nodes(rewritten)
    if original_nodes.keys() != rewritten_nodes.keys():
        raise ValueError("layout rewrite changed node inventory")
    expected = onnx.ModelProto()
    expected.CopyFrom(original)
    tensors = {value.name: value for value in expected.graph.initializer}
    transposed = []
    for node in expected.graph.node:
        if node.op_type != "MatMul":
            continue
        if len(node.input) != 2 or node.input[0] not in tensors:
            raise ValueError("expected constant-left matrix product")
        name, vector = node.input
        value = onnx.numpy_helper.to_array(tensors[name])
        tensors[name].CopyFrom(onnx.numpy_helper.from_array(value.T.copy(), name))
        node.input[:] = [vector, name]
        transposed.append(name)
    if sorted(transposed) != sorted(MODEL_WEIGHTS):
        raise ValueError("expected exactly four distinct matrix weight transposes")
    expected = onnx.shape_inference.infer_shapes(expected)
    if expected != rewritten:
        raise ValueError("layout graph changed beyond exact transposes and inferred shapes")


def verify_quantization(rewritten, compressed):
    """Check fixed signed per-channel weights and the four dynamic MatMul expansions."""
    original_nodes, actual_nodes = _unique_nodes(rewritten), _unique_nodes(compressed)
    original_weights, actual_weights = _weights(rewritten), _weights(compressed)
    expected_nodes = {}
    expected_weights = {}
    for name, value in original_weights.items():
        if name not in MODEL_WEIGHTS:
            expected_weights[name] = value
            continue
        scale = (np.max(np.abs(value), axis=0).astype(np.float64) / 127).astype(np.float32)
        scale[scale < np.finfo(np.float32).tiny] = 1
        quantized = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
        expected_weights.update(
            {
                name + "_scale": scale,
                name + "_zero_point": np.zeros(value.shape[1], dtype=np.int8),
                name + "_quantized": quantized,
            }
        )
    if expected_weights.keys() != actual_weights.keys():
        raise ValueError("quantized initializer inventory differs")
    for name, value in expected_weights.items():
        if actual_weights[name].dtype != value.dtype or not np.array_equal(
            actual_weights[name], value
        ):
            raise ValueError(f"fixed full-range per-channel quantization differs: {name}")
    make = onnx.helper.make_node
    for node in original_nodes.values():
        if node.op_type != "MatMul":
            expected_nodes[node.name] = node
            continue
        vector, weight = node.input
        output = node.output[0]
        dynamic = make(
            "DynamicQuantizeLinear",
            [vector],
            [vector + "_quantized", vector + "_scale", vector + "_zero_point"],
            name=vector + "_QuantizeLinear",
        )
        scales = node.name + "_quant_scales_mul"
        integer = output + "_output_quantized"
        cast = integer + "_cast"
        additions = [
            dynamic,
            make("Mul", [vector + "_scale", weight + "_scale"], [scales + ":0"], name=scales),
            make(
                "MatMulInteger",
                [
                    vector + "_quantized",
                    weight + "_quantized",
                    vector + "_zero_point",
                    weight + "_zero_point",
                ],
                [integer],
                name=node.name + "_quant",
            ),
            make("Cast", [integer], [cast + "_output"], name=cast, to=onnx.TensorProto.FLOAT),
            make(
                "Mul",
                [cast + "_output", scales + ":0"],
                [output],
                name=node.name + "_quant_output_scale_mul",
            ),
        ]
        expected_nodes.update({addition.name: addition for addition in additions})
    if expected_nodes.keys() != actual_nodes.keys() or any(
        actual_nodes[name] != node for name, node in expected_nodes.items()
    ):
        raise ValueError("quantized graph changed non-layout operations or dynamic arithmetic")
    metadata = {item.key: item.value for item in compressed.metadata_props}
    if metadata.get("compression") != "per_channel_QInt8_weights_dynamic_activation_quantization":
        raise ValueError("quantization setting metadata differs")


def graph_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def graph_trace(engine, counts):
    """No production ONNX wrapper: explicitly advance all three histories every bin."""
    state = [np.zeros(shape, np.float32) for shape in ((2, 176), (4, 32), (8, 32))]
    predictions = np.empty((len(counts), 7), np.float32)
    for index, raw in enumerate(counts):
        feeds = {"counts": raw.astype(np.float32)}
        feeds.update({f"history{i}": value for i, value in enumerate(state)})
        prediction, *state = engine.run(None, feeds)
        if not all(np.isfinite(value).all() for value in [prediction, *state]):
            raise ValueError("nonfinite independent ONNX arithmetic")
        predictions[index] = prediction
    return predictions


def verify_run_envelope(run, root):
    """Authenticate the immutable attempt before opening any sample arrays."""
    run, root = Path(run).resolve(), Path(root).resolve()
    manifest = read_json(run / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "h1_final_closure_quality"
        or manifest.get("status") != "complete"
        or manifest.get("run") != str(run)
        or manifest["statuses"].get("artifact_completion") != "complete"
        or manifest["statuses"].get("implementation_execution")
        not in {"complete", "partial_failure"}
        or manifest["statuses"].get("independent_verification") != "not_run"
        or manifest["statuses"].get("runtime_goals") != "not_run"
    ):
        raise ValueError(
            "quality attempt is not artifact-complete or its status/run binding differs"
        )
    start = read_json(run / "start.json")
    for key in ("schema_version", "kind", "run", "started_utc", "command"):
        assert_equal(start[key], manifest[key], f"start.{key}")
    if start["status"] != "running":
        raise ValueError("attempt start is not the retained original running status")
    started = datetime.fromisoformat(manifest["started_utc"])
    finished = datetime.fromisoformat(manifest["finished_utc"])
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise ValueError("attempt timestamp order/timezone differs")
    actual_paths = {
        str(path.relative_to(run))
        for path in run.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_paths != set(manifest["artifacts"]):
        raise ValueError("attempt artifact inventory differs")
    for relative, checksum in manifest["artifacts"].items():
        verify(bound_path(run, relative), checksum)
    for relative, checksum in manifest["source"]["files"].items():
        verify(bound_path(run / "source", relative), checksum)
        committed = subprocess.check_output(
            ["git", "show", f"{manifest['source']['head']}:{relative}"], cwd=root
        )
        if hashlib.sha256(committed).hexdigest() != checksum:
            raise ValueError(f"producer source not bound to committed bytes: {relative}")
    verify(run / "source.diff", manifest["source"]["diff_sha256"])
    required = {
        "neural_experiments/h1_final_closure.py",
        "neural_experiments/compression.py",
        "neural_experiments/H1_FINAL_CLOSURE_PROTOCOL.md",
        manifest["config"]["path"],
        *AUDITOR_FILES,
    }
    if not required.issubset(manifest["source"]["files"]):
        raise ValueError("producer source inventory omits study or auditor dependencies")
    config = manifest["configuration"]
    verify(run / "configuration.json", manifest["config"]["sha256"])
    verify(bound_path(run / "source", manifest["config"]["path"]), manifest["config"]["sha256"])
    assert_equal(read_json(run / "configuration.json"), config, "configuration")
    verify(run / "summary.json", manifest["summary_sha256"])
    verify(run / "inventory.json", manifest["inventory_sha256"])
    return manifest, config, read_json(run / "summary.json"), read_json(run / "inventory.json")


def check_config(config):
    """Thresholds and final identities are independently pinned, not producer imports."""
    if (
        config["schema_version"] != 1
        or config["seeds"] != [0, 1, 2]
        or config["timing_seed"] != 0
        or config["execution_paths"] != list(ARMS)
        or config["tolerances"]
        != {
            "rtol": RTOL,
            "atol": ATOL,
            "day_r2_atol": SCORE_ATOL,
            "int8_day_r2_min_delta": -MAX_LOSS,
        }
        or config["runtime_versions"] != {"onnx": "1.19.1", "onnxruntime": "1.23.2"}
        or config["quality"]
        != {
            "parity_rtol": RTOL,
            "parity_atol": ATOL,
            "score_atol": SCORE_ATOL,
            "max_int8_r2_loss": MAX_LOSS,
        }
    ):
        raise ValueError("fixed conversion quality configuration changed")
    checksums = (
        "95f6a0b3131e1971f146298e23c95a42f7da28467d8acc24f37f916d9dc049ec",
        "b33b6ed1b38bad2d82a3605f6be4ae0707faae77361596245f1dd484f621579b",
        "12aed331e7bac3c9f42f67d75ff56dac32f7dcbffde0c942d96215dac8263627",
    )
    if [model["seed"] for model in config["models"]] != [0, 1, 2] or tuple(
        model["sha256"] for model in config["models"]
    ) != checksums:
        raise ValueError("predetermined final-model identities changed")
    pinned = {
        "freeze": "fd458c649b2af3c78e90ab2298821b2e28a1dc38011c7cc5ee3b29ef19dcea1c",
        "parent_freeze": "471c01b3fdc030c432b61c8188641c110b56f97aeb25ea64f5455312849a1fff",
        "assessment_index": "dd27945ae0f2b9aaab9e2ae6d6f35e77ae811f2cc0b95e19286c75eb7f2dc717",
        "development_index": "a87f1e638114ec41c5938366876c1789ceea2150b1c3a144e88f023b05d562c2",
        "historical_manifest": "00eabebd9c456c8a5378f8cf0f9e261bfb8472c83159c1b7510491a22b0b4b86",
        "historical_summary": "c5657c2753dae25972ad50872bb55067a9cfe4650c2faeb8eb2d65239e5968de",
        "public_final": "c5657c2753dae25972ad50872bb55067a9cfe4650c2faeb8eb2d65239e5968de",
    }
    if {key: value["sha256"] for key, value in config["inputs"].items()} != pinned:
        raise ValueError("independently pinned historical input identities changed")


def score_deltas(scores, reference):
    return {
        day: {
            "r2_variance_weighted": values["r2_variance_weighted"]
            - reference["per_day"][day]["r2_variance_weighted"],
            "r2_per_output": [
                value - baseline
                for value, baseline in zip(
                    values["r2_per_output"], reference["per_day"][day]["r2_per_output"], strict=True
                )
            ],
        }
        for day, values in scores["per_day"].items()
    }


def trace_statistics(actual, reference):
    if (
        actual.shape != reference.shape
        or actual.dtype != np.float32
        or not np.isfinite(actual).all()
    ):
        raise ValueError("malformed saved full trace")
    return {
        "allclose": bool(np.allclose(actual, reference, rtol=RTOL, atol=ATOL)),
        "max_absolute_error": float(np.max(np.abs(actual - reference))),
        "num_bins": len(actual),
    }


def decide_gate(arm, checks, deltas):
    if not checks or not deltas:
        raise ValueError("cannot qualify incomplete evidence")
    trace_gate = None if arm == "E" else all(row["allclose"] for row in checks.values())
    differences = [row["r2_variance_weighted"] for row in deltas.values()]
    score_gate = (
        all(value >= -MAX_LOSS for value in differences)
        if arm == "E"
        else all(abs(value) <= SCORE_ATOL for value in differences)
    )
    return {
        "trace_gate": trace_gate,
        "score_gate": score_gate,
        "qualified": score_gate and trace_gate is not False,
    }


def load_prediction_records(base, records, file_ids):
    if set(records) != set(file_ids):
        raise ValueError("full prediction file inventory differs")
    result = {}
    seen = set()
    for file_id, record in records.items():
        path = bound_path(base, record["path"])
        if path in seen:
            raise ValueError("different files alias the same prediction artifact")
        seen.add(path)
        verify(path, record["sha256"])
        result[file_id] = np.load(path, allow_pickle=False)
    return result


def _audit_models(run, root, config, summary, inventory, files, historical):
    from neural_decoder.temporal import TCNStream, load_tcn

    if [model["seed"] for model in summary["models"]] != [0, 1, 2]:
        raise ValueError("quality summary model/seed inventory differs")
    file_ids = [item["file_id"] for item, *_ in files]
    historical_by_name = {row["model"]["name"]: row for row in historical["models"]}
    audited_rows, eligibility = [], {arm: True for arm in ARMS}
    any_failure = False
    for requested, model, inv in zip(
        config["models"], summary["models"], inventory["models"], strict=True
    ):
        seed = requested["seed"]
        if inv["seed"] != seed:
            raise ValueError("inventory seed identity differs")
        original = historical_by_name[requested["name"]]
        assert_equal(model["model"], original["model"], "historical model")
        assert_equal(inv["model"], original["model"], "inventory model")
        for key in ("name", "path", "sha256", "seed"):
            assert_equal(requested[key], model["model"][key], f"requested model.{key}")
        path = bound_path(root, requested["path"])
        verify(path, requested["sha256"])
        with np.load(path, allow_pickle=False) as bundle:
            weights = {key: bundle[key] for key in bundle.files}
        frozen = load_tcn(path)
        if model["fingerprint"] != fingerprint(weights) or model["parameter_count"] != 23367:
            raise ValueError("model fingerprint or parameter count differs")
        if inv["model_bytes"] != path.stat().st_size:
            raise ValueError("inventory model size differs")
        historical_root = bound_path(root, config["inputs"]["historical_manifest"]["path"]).parent
        expected_historical = {
            fid: {
                "path": str((historical_root / requested["name"] / f"{fid}.npy").relative_to(root)),
                "sha256": original["prediction_sha256"][fid],
            }
            for fid in file_ids
        }
        assert_equal(
            inv["historical_predictions"], expected_historical, "historical prediction identities"
        )
        historical_predictions = load_prediction_records(root, expected_historical, file_ids)
        historical_scores = independent_scores(files, historical_predictions)
        assert_equal(historical_scores, original["scores"], "historical scores")
        assert_equal(inv["historical_scores"], historical_scores, "inventory historical scores")
        assert_equal(
            read_json(historical_root / requested["name"] / "scores.json"),
            historical_scores,
            "historical standalone scores",
        )
        reference = model["reference"]
        assert_equal(
            read_json(run / f"seed-{seed}/reference.json"), reference, "reference standalone"
        )
        references = load_prediction_records(run, reference["predictions"], file_ids)
        checks = {}
        for item, counts, _, _ in files:
            fid = item["file_id"]
            independent = reference_trace(counts, weights)
            check_trace(references[fid], independent)
            checks[fid] = trace_statistics(references[fid], historical_predictions[fid])
        scores = independent_scores(files, references)
        deltas = score_deltas(scores, historical_scores)
        gate = decide_gate("A", checks, deltas)
        assert_equal(reference["scores"], scores, "reference scores")
        assert_equal(reference["historical_trace"], checks, "historical full-trace checks")
        assert_equal(reference["historical_delta_per_day"], deltas, "historical score deltas")
        assert_equal(reference["historical_gate"], gate, "historical gate")
        if reference["historical_scores_match"] is not True or not gate["qualified"]:
            raise ValueError("historical float32 reconstruction did not pass")
        if set(model["arms"]) != set(ARMS):
            raise ValueError("execution path inventory differs")
        graphs = {}
        for arm in ("C", "D", "E"):
            record = model["arms"][arm].get("graph")
            if record is None:
                continue
            graph_path = bound_path(run, record["path"])
            verify(graph_path, record["sha256"])
            graph = onnx.load(graph_path)
            verify_graph_identity(
                graph, weights, "dynamic_int8" if arm == "E" else "float32", rewritten=arm == "D"
            )
            ops = [node.op_type for node in graph.graph.node]
            assert_equal(
                record["operations"],
                {op: ops.count(op) for op in sorted(set(ops))},
                "graph operators",
            )
            assert_equal(
                record["metadata"],
                {prop.key: prop.value for prop in graph.metadata_props},
                "graph metadata",
            )
            if record["bytes"] != graph_path.stat().st_size:
                raise ValueError("graph size differs")
            graphs[arm] = graph
        if "D" in graphs:
            if "C" not in graphs:
                raise ValueError("rewritten graph lacks original control")
            verify_transpose(graphs["C"], graphs["D"])
        if "E" in graphs:
            if "D" not in graphs:
                raise ValueError("int8 graph lacks layout control")
            verify_quantization(graphs["D"], graphs["E"])
        for arm in ARMS:
            result = model["arms"][arm]
            assert_equal(
                read_json(run / f"seed-{seed}/{arm}-result.json"), result, "standalone arm result"
            )
            if result["precision"] != ("dynamic_int8" if arm == "E" else "float32"):
                raise ValueError("arm precision label changed")
            if result["status"] == "failed":
                if (
                    result["qualified"] is not False
                    or not result.get("error")
                    or not result.get("traceback")
                ):
                    raise ValueError("failed path lacks explicit failure evidence")
                eligibility[arm] = False
                any_failure = True
                continue
            if result["status"] != "complete":
                raise ValueError("arm execution is not terminal")
            predictions = load_prediction_records(run, result["predictions"], file_ids)
            if arm in ("A", "B"):
                if result["graph"] is not None:
                    raise ValueError("non-ONNX arm unexpectedly binds an ONNX graph")
                engine = TCNStream(frozen, implementation="window" if arm == "A" else "incremental")
                expected_state_bytes = engine.state_nbytes
            else:
                engine = graph_session(bound_path(run, result["graph"]["path"]))
                expected_state_bytes = 4 * (2 * 176 + 4 * 32 + 8 * 32)
            if result["numeric_state_bytes"] != expected_state_bytes:
                raise ValueError("persistent numeric state accounting differs")
            checks = {}
            for item, counts, _, _ in files:
                fid = item["file_id"]
                if arm in ("A", "B"):
                    engine.reset()
                    independent = engine.predict_chunk(counts)
                else:
                    independent = graph_trace(engine, counts)
                check_trace(predictions[fid], independent, exact=True)
                checks[fid] = trace_statistics(predictions[fid], references[fid])
            scores = independent_scores(files, predictions)
            deltas = score_deltas(scores, reference["scores"])
            gate = decide_gate(arm, checks, deltas)
            assert_equal(result["scores"], scores, f"seed-{seed}/{arm} scores")
            assert_equal(result["trace_checks"], checks, f"seed-{seed}/{arm} full-trace checks")
            assert_equal(result["delta_per_day"], deltas, f"seed-{seed}/{arm} deltas")
            for key, value in gate.items():
                assert_equal(result[key], value, f"seed-{seed}/{arm} {key}")
            eligibility[arm] &= gate["qualified"]
            audited_rows.extend(
                {"seed": seed, "arm": arm, "day": day, **delta} for day, delta in deltas.items()
            )
        print(f"Independently verified all retained quality paths: seed {seed}", flush=True)
    return audited_rows, eligibility, any_failure


def audit(run, *, root):
    run, root = Path(run).resolve(), Path(root).resolve()
    manifest, config, summary, inventory = verify_run_envelope(run, root)
    check_config(config)
    if {"onnx": onnx.__version__, "onnxruntime": ort.__version__} != config["runtime_versions"]:
        raise ValueError("independent replay requires the pinned installed runtime versions")
    assert_equal(manifest["inputs"], config["inputs"], "manifest input bindings")
    assert_equal(inventory["inputs"], config["inputs"], "inventory input bindings")
    paths = {}
    for name, record in config["inputs"].items():
        if name == "development_index":
            # Historical development cache is shared via a pre-existing symlink.
            # Its exact index bytes are pinned; assessment/model roots stay strict.
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("development index must use its repository-relative location")
            paths[name] = root / relative
        else:
            paths[name] = bound_path(root, record["path"])
        verify(paths[name], record["sha256"])
    index, files = load_assessment_files(
        paths["assessment_index"],
        paths["freeze"],
        config["inputs"]["freeze"]["sha256"],
        root=root,
        cache=root / config["cache"],
    )
    assert_equal(
        inventory["assessment_files"], index["files"], "authenticated assessment inventory"
    )
    assert_equal(
        inventory["development_files"],
        read_json(paths["development_index"])["files"],
        "authenticated timing inventory",
    )
    if (len(files), sum(len(row[1]) for row in files), sum(int(row[3].sum()) for row in files)) != (
        14,
        34490,
        33613,
    ):
        raise ValueError("fixed assessment file/bin/mask counts changed")
    historical = read_json(paths["historical_manifest"])
    historical_summary = read_json(paths["historical_summary"])
    freeze = read_json(paths["freeze"])
    if (
        historical["status"] != "complete"
        or historical["freeze_sha256"] != config["inputs"]["freeze"]["sha256"]
        or historical["index_sha256"] != config["inputs"]["assessment_index"]["sha256"]
        or historical["summary_sha256"] != digest(paths["historical_summary"])
        or historical_summary != read_json(paths["public_final"])
        or historical_summary["models"] != historical["models"]
        or [row["model"] for row in historical["models"]] != freeze["models"]
    ):
        raise ValueError("historical completed evaluation no longer matches its freeze")
    for relative, checksum in historical["source"]["files"].items():
        verify(bound_path(paths["historical_manifest"].parent / "source", relative), checksum)
    verify(paths["historical_manifest"].parent / "source.diff", historical["source"]["diff_sha256"])
    for relative, checksum in freeze["source"]["files_sha256"].items():
        if historical["source"]["files"].get(relative) != checksum:
            raise ValueError("historical evaluation source differs from frozen production source")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            rows, eligibility, any_failure = _audit_models(
                run, root, config, summary, inventory, files, historical
            )
    finally:
        torch.set_num_threads(old_threads)
    gates = {
        "historical_reference": True,
        "float32_equivalence": {arm: eligibility[arm] for arm in FLOAT_ARMS},
        "compression_quality": eligibility["E"],
    }
    assert_equal(summary["gates"], gates, "aggregate gates")
    assert_equal(summary["seed_day_rows"], rows, "every seed/arm/day delta")
    int8 = [row["r2_variance_weighted"] for row in rows if row["arm"] == "E"]
    assert_equal(
        summary["mean_delta_r2"], float(np.mean(int8)) if int8 else None, "mean int8 delta"
    )
    assert_equal(summary["worst_seed_day_delta"], min(int8) if int8 else None, "worst int8 delta")
    if summary["completed_seed_day_cells"] != len(int8) or summary["timing_seed"] != 0:
        raise ValueError("summary cell count or predetermined timing seed differs")
    expected_statuses = {
        "implementation_execution": "partial_failure" if any_failure else "complete",
        "artifact_completion": "complete",
        "independent_verification": "not_run",
        "float32_equivalence": "passed"
        if all(eligibility[arm] for arm in FLOAT_ARMS)
        else "failed",
        "compression_quality": "passed" if eligibility["E"] else "failed",
        "runtime_goals": "not_run",
    }
    assert_equal(manifest["statuses"], expected_statuses, "execution/evidence/gate statuses")
    return {
        "schema_version": 1,
        "kind": "h1_final_closure_quality_audit",
        "status": "passed",
        "run": str(run),
        "audited_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
        "config_sha256": manifest["config"]["sha256"],
        "source": manifest["source"],
        "auditor_source": {name: digest(root / name) for name in AUDITOR_FILES},
        "artifact_sha256": manifest["artifacts"],
        "qualified_arms": eligibility,
        "gates": gates,
        "implementation_execution": expected_statuses["implementation_execution"],
        "completed_seed_day_cells": len(int8),
        "seed_day_rows": rows,
        "checks": [
            "authorized_inputs_and_historical_identity",
            "independent_whole_file_reference",
            "four_exact_layout_transposes",
            "fixed_per_channel_int8_arithmetic",
            "every_full_trace_including_unscored_bins",
            "independent_per_day_per_output_scores",
            "exact_int8_to_int8_replay",
            "all_21_seed_day_quality_cells",
            "status_source_artifact_binding",
        ],
    }


def validate_audit_receipt(run, receipt_path, root):
    """Read-only runtime admission: stale or mixed evidence cannot qualify an arm."""
    run, root = Path(run).resolve(), Path(root).resolve()
    receipt = read_json(receipt_path)
    manifest, _, summary, _ = verify_run_envelope(run, root)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "h1_final_closure_quality_audit"
        or receipt.get("status") != "passed"
        or receipt.get("run") != str(run)
    ):
        raise ValueError("quality audit receipt is not verified for this run")
    verify(run / "manifest.json", receipt["manifest_sha256"])
    verify(run / "summary.json", receipt["summary_sha256"])
    assert_equal(receipt["config_sha256"], manifest["config"]["sha256"], "audit config binding")
    assert_equal(receipt["source"], manifest["source"], "audit source binding")
    assert_equal(receipt["artifact_sha256"], manifest["artifacts"], "audit artifact bindings")
    assert_equal(receipt["gates"], summary["gates"], "audit gate bindings")
    assert_equal(
        receipt["implementation_execution"],
        manifest["statuses"]["implementation_execution"],
        "audit execution status binding",
    )
    expected = {
        **summary["gates"]["float32_equivalence"],
        "E": summary["gates"]["compression_quality"],
    }
    assert_equal(receipt["qualified_arms"], expected, "audit arm qualification")
    if set(receipt["auditor_source"]) != set(AUDITOR_FILES):
        raise ValueError("audit source dependency inventory differs")
    for relative, checksum in receipt["auditor_source"].items():
        verify(bound_path(root, relative), checksum)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    # Reserve the receipt before inference, retaining explicit failed-audit evidence.
    with args.output.open("x") as handle:
        json.dump(
            {
                "status": "running",
                "run": str(args.run.resolve()),
                "started_utc": datetime.now(UTC).isoformat(),
            },
            handle,
        )
    try:
        result = audit(args.run, root=args.root)
    except BaseException as error:
        result = {
            "status": "failed",
            "run": str(args.run.resolve()),
            "error": f"{type(error).__name__}: {error}",
            "finished_utc": datetime.now(UTC).isoformat(),
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "qualified_arms": result["qualified_arms"]}))


if __name__ == "__main__":
    main()
