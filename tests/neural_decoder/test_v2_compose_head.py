"""Algebraic head composition preserves causal features and uses no labels."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "v2_compose_head", ROOT / "neural_experiments/v2/compose_head.py"
)
composition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(composition)


@pytest.mark.parametrize("dilations", [(1, 2, 4), (1, 2, 4, 8, 16)])
def test_composition_matches_retained_correction_and_preserves_backbone(tmp_path, dilations):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(96)
        torch.manual_seed(42)
        model = composition.accuracy.TemporalCore(4, 2, dilations, 5).eval()
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}
        affine = {
            "input_mean": np.array([2, 1, 3, 0], dtype=np.float32),
            "input_scale": np.array([1, 2, 3, 1], dtype=np.float32),
            "target_mean": np.array([0.3, -0.2], dtype=np.float32),
            "target_scale": np.array([0.7, 0.2], dtype=np.float32),
        }
        counts = rng.poisson(2, (83, 4)).astype(np.float32)
        mean, scale = rng.normal(size=5), rng.uniform(0.5, 2, size=5)
        coefficient, intercept = rng.normal(size=(5, 2)), rng.normal(size=2)
        base, latent = composition.accuracy.predict_with_features(model, affine, counts)
        expected = base.astype(np.float64) + ((latent - mean) / scale) @ coefficient + intercept
        composed = composition.compose_head(model, affine, mean, scale, coefficient, intercept)
        assert composed.receptive_field == model.receptive_field
        assert composed.dilations == model.dilations
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
            if name.startswith("convs."):
                torch.testing.assert_close(
                    composed.state_dict()[name], original[name], atol=0, rtol=0
                )
        path = tmp_path / "composed.npz"
        composition.accuracy.save_model(path, composed, affine)
        loaded, transforms, _ = composition.accuracy.load_model(path)
        actual = composition.accuracy.predict(loaded, transforms, counts)
        np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=3e-6)
        stream = composition.accuracy.CausalStream(loaded, transforms)
        np.testing.assert_allclose(stream.predict_chunk(counts), actual, atol=5e-7, rtol=3e-6)
        invalid = coefficient.copy()
        invalid[0, 0] = np.nan
        with pytest.raises(ValueError, match="calibration arrays"):
            composition.compose_head(model, affine, mean, scale, invalid, intercept)
        with pytest.raises(ValueError, match="positive"):
            composition.compose_head(model, affine, mean, scale * 0, coefficient, intercept)
    finally:
        torch.set_num_threads(old_threads)
