"""Optional standard-runtime correctness; no performance claims from these tests."""

import numpy as np
import pytest
import torch

from neural_decoder.optimized import (
    STATE_SHAPES,
    ExplicitTCNStep,
    ONNXTCNStream,
    export_step_onnx,
    model_fingerprint,
)
from neural_decoder.temporal import CausalTCN, TCNStream, freeze_tcn


@pytest.fixture
def frozen():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(88)
        model = CausalTCN()
    with torch.no_grad():
        for layer in model.convs:
            layer.bias.copy_(torch.linspace(-0.1, 0.5, 32))
    return freeze_tcn(
        model,
        input_mean=np.full(176, 0.7),
        input_scale=np.full(176, 1.2),
        target_mean=np.linspace(-0.01, 0.01, 7),
        target_scale=np.full(7, 0.02),
    )


@pytest.fixture
def onnx_path(tmp_path, frozen):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    path = tmp_path / "model.onnx"
    export_step_onnx(frozen, path)
    return path


def test_explicit_state_step_matches_biased_startup_and_whole_file(frozen):
    counts = np.random.default_rng(12).poisson(3, (91, 176)).astype(np.float32)
    expected = TCNStream(frozen).predict_chunk(counts)
    state = tuple(torch.zeros(shape) for shape in STATE_SHAPES)
    model = ExplicitTCNStep(frozen).eval()
    outputs = []
    with torch.inference_mode():
        for row in counts:
            result = model(torch.from_numpy(row), *state)
            outputs.append(result[0].numpy())
            state = result[1:]
    np.testing.assert_allclose(outputs, expected, rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize("kind", ["counts", "zero", "large", "impulse"])
def test_onnx_complete_trace_arithmetic_and_reset(onnx_path, frozen, kind):
    counts = np.random.default_rng(13).poisson(3, (123, 176)).astype(np.float32)
    if kind == "zero":
        counts.fill(0)
    elif kind == "large":
        counts *= 1000
    elif kind == "impulse":
        counts.fill(0)
        counts[[0, 14, 38, 122], [0, 17, 32, 175]] = 20
    stream = ONNXTCNStream(onnx_path, expected_model_identity=model_fingerprint(frozen))
    expected = TCNStream(frozen).predict_chunk(counts)
    actual = stream.predict_chunk(counts)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=5e-5 if kind == "large" else 1e-7)
    stream.reset()
    np.testing.assert_array_equal(actual, stream.predict_chunk(counts))
    assert stream.state_nbytes == 2944
    assert stream.session.get_providers() == ["CPUExecutionProvider"]


def test_onnx_future_chunk_partition_and_zero_missing_history(onnx_path, frozen):
    counts = np.random.default_rng(14).poisson(2, (81, 176))
    first = ONNXTCNStream(onnx_path, expected_model_identity=model_fingerprint(frozen))
    second = ONNXTCNStream(onnx_path, expected_model_identity=model_fingerprint(frozen))
    a = first.predict_chunk(counts[:13])
    b = np.r_[second.predict_chunk(counts[:3]), second.predict_chunk(counts[3:13])]
    np.testing.assert_array_equal(a, b)
    for left, right in zip(first.state_snapshot(), second.state_snapshot(), strict=True):
        np.testing.assert_array_equal(left, right)
    a = first.predict_chunk(counts[13:])
    b = second.predict_chunk(counts[13:] + 20)
    assert not np.allclose(a, b)
    first.reset()
    whole = first.predict_chunk(counts)
    second.reset()
    chunked = np.r_[
        second.predict_chunk(counts[:1]),
        second.predict_chunk(counts[1:30]),
        second.predict_chunk(counts[30:]),
    ]
    np.testing.assert_array_equal(whole, chunked)


def test_onnx_identity_input_failure_and_export_no_overwrite(onnx_path, frozen):
    with pytest.raises(ValueError, match="identity"):
        ONNXTCNStream(onnx_path, expected_model_identity="different-model")
    with pytest.raises(FileExistsError):
        export_step_onnx(frozen, onnx_path)
    stream = ONNXTCNStream(onnx_path, expected_model_identity=model_fingerprint(frozen))
    stream.step(np.ones(176))
    before = stream.state_snapshot()
    for bad in (np.zeros(175), np.zeros((1, 176)), np.full(176, np.inf)):
        with pytest.raises(ValueError):
            stream.step(bad)
    for left, right in zip(before, stream.state_snapshot(), strict=True):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(FloatingPointError, match="reset"):
        stream.step(np.full(176, np.finfo(np.float64).max))
    with pytest.raises(RuntimeError, match="reset"):
        stream.step(np.ones(176))
    stream.reset()
    assert np.isfinite(stream.step(np.ones(176))).all()


def test_model_identity_changes_for_affine_or_weight_changes(frozen):
    from neural_decoder.temporal import FrozenTCN

    original = model_fingerprint(frozen)
    for field in ("input_mean", "head_weight"):
        values = {name: array.copy() for name, array in frozen.arrays.items()}
        values[field].flat[0] += 0.01
        assert model_fingerprint(FrozenTCN(values)) != original


def test_precision_change_requires_explicit_approximation_opt_in(onnx_path, frozen, tmp_path):
    import onnx

    graph = onnx.load(onnx_path)
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata["precision"] = "dynamic_int8"
    onnx.helper.set_model_props(graph, metadata)
    path = tmp_path / "approximation-tag.onnx"
    onnx.save(graph, path)
    with pytest.raises(ValueError, match="precision"):
        ONNXTCNStream(path, expected_model_identity=model_fingerprint(frozen))
    # This fixture tests the metadata gate only; it is not a quantized quality experiment.
    approximate = ONNXTCNStream(
        path, expected_model_identity=model_fingerprint(frozen), precision="dynamic_int8"
    )
    assert approximate.precision == "dynamic_int8"
