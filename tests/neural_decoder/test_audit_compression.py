"""The independent compression auditor rejects graph and aggregate evidence mutations."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture
def auditor(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "neural_experiments"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "compression_auditor", directory / "audit_compression.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_pair():
    nodes, initializers = [], []
    for i in range(4):
        weights = np.array([[1, 2], [3, 4]], dtype=np.float32) + i
        initializers.append(onnx.numpy_helper.from_array(weights, f"w{i}"))
        nodes.append(onnx.helper.make_node("MatMul", [f"w{i}", "x"], [f"h{i}"], name=f"matrix{i}"))
    nodes.append(onnx.helper.make_node("Relu", ["h3"], ["result"], name="activation"))
    original = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, "fixture", [], [], initializers)
    )
    canonical = onnx.ModelProto()
    canonical.CopyFrom(original)
    for node in canonical.graph.node[:4]:
        node.input[:] = list(reversed(node.input))
    for tensor in canonical.graph.initializer:
        tensor.CopyFrom(
            onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(tensor).T.copy(), tensor.name)
        )
    return original, canonical


def test_transpose_audit_rejects_changed_weights_and_non_matrix_operations(auditor):
    original, canonical = graph_pair()
    auditor.verify_transpose(original, canonical)
    tensor = canonical.graph.initializer[0]
    tensor.CopyFrom(onnx.numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), tensor.name))
    with pytest.raises(AssertionError):
        auditor.verify_transpose(original, canonical)
    original, canonical = graph_pair()
    canonical.graph.node[-1].op_type = "Sigmoid"
    with pytest.raises(ValueError, match="non-matrix"):
        auditor.verify_transpose(original, canonical)


def test_score_audit_uses_known_zero_r2_and_rejects_false_counts(auditor):
    targets = np.array([-1, 1, -1, 1])[:, None] * np.arange(1, 8)[None]
    predictions = np.zeros((4, 7))
    day = {
        "r2_variance_weighted": 0,
        "r2_per_output": [0] * 7,
        "sse_per_output": (4 * np.arange(1, 8) ** 2).tolist(),
        "sst_per_output": (4 * np.arange(1, 8) ** 2).tolist(),
        "file_ids": ["S2_set_1"],
    }
    saved = {
        "r2_mean": 0,
        "per_day": {"S2": day},
        "num_bins": 8,
        "num_scored_bins": 4,
        "num_outputs": 7,
    }
    value, outputs = auditor.check_scores(
        saved, "S2", targets, predictions, full_bins=8, file_ids=["S2_set_1"]
    )
    assert value == 0 and outputs == [0] * 7
    saved["num_scored_bins"] = 8
    with pytest.raises(ValueError, match="score accounting"):
        auditor.check_scores(saved, "S2", targets, predictions, full_bins=8, file_ids=["S2_set_1"])
