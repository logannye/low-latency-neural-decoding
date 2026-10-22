"""Compression must change arithmetic explicitly and keep replay causal."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_decoder.optimized import ONNXTCNStream, model_fingerprint
from neural_decoder.temporal import CausalTCN, freeze_tcn

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def test_explicit_quantization_identity_and_causal_replay(tmp_path):
    path = Path(__file__).resolve().parents[2] / "neural_experiments/compression.py"
    spec = importlib.util.spec_from_file_location("compression_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(8)
    frozen = freeze_tcn(CausalTCN())
    original, quantized, operations = module.convert(frozen, tmp_path)
    assert operations["MatMulInteger"] == 4
    identity = model_fingerprint(frozen)
    with pytest.raises(ValueError, match="precision"):
        ONNXTCNStream(quantized, expected_model_identity=identity)
    decoder = ONNXTCNStream(quantized, expected_model_identity=identity, precision="dynamic_int8")
    values = np.random.default_rng(9).poisson(0.2, (50, 176))
    full = decoder.predict_chunk(values)
    decoder.reset()
    modified = values.copy()
    modified[25:] += 5
    altered = decoder.predict_chunk(modified)
    np.testing.assert_array_equal(full[:25], altered[:25])
    assert full.shape == (50, 7) and np.isfinite(full).all()
    reference = ONNXTCNStream(original, expected_model_identity=identity).predict_chunk(values)
    canonical = ONNXTCNStream(
        tmp_path / "quantization-input.onnx", expected_model_identity=identity
    ).predict_chunk(values)
    np.testing.assert_allclose(canonical, reference, rtol=3e-5, atol=1e-6)
    assert not np.array_equal(full, reference)
    assert np.max(np.abs(full - reference)) < 0.03
