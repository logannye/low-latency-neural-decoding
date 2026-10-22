"""The named V3 daily comparator reproduces the frozen V3 mean-loss ridge fit."""

import numpy as np
import pytest

from neural_experiments.v3 import model as v3
from neural_experiments.v4 import model as v4


@pytest.mark.parametrize("prefix", [1500, 6000], ids=["30s", "120s"])
def test_v4_daily_is_exact_v3_recipe_at_each_prefix(prefix):
    rng = np.random.default_rng(924)
    n, channels = 6500, 3
    # Both floor-limited and data-scaled channels; a reset occurs inside the120s prefix.
    counts = rng.poisson([0.001, 0.4, 3.0], (n, channels)).astype(np.float32)
    power = rng.lognormal([0, 1, 3], [0.001, 0.3, 1.0], (n, channels)).astype(np.float32)
    features = {"x_tx3": counts + 1, "x_tx45": counts, "x_sbp": power}
    segments = [(0, 3100), (3100, n)]
    clipped_segments = [(a, min(b, prefix)) for a, b in segments if a < prefix]
    angle = rng.normal(size=n)
    target = np.column_stack((np.cos(angle), np.sin(angle)))
    mask = rng.random(n) > 0.2
    config3 = {
        "kind": "regularized",
        "family": "tx_sbp",
        "tau_seconds": 0.2,
        "baseline_tau_seconds": None,
        "ridge_lambda": 1.0,
    }
    original, receipt = v3.fit(
        counts[:prefix], power[:prefix], target[:prefix], mask[:prefix], clipped_segments, config3
    )
    config4 = {
        "family": "tx45_sbp",
        "tau_seconds": 0.2,
        "baseline_tau_seconds": None,
        "adapt_gamma": 1.0,
    }
    raw = v4.representation(features, segments, config4)
    z, mean, scale = v4.design(raw, prefix, channels)
    usable = v4.training_mask(mask, segments, prefix)
    statistics = v4.sufficient(z[:prefix], target[:prefix], usable)
    weights = v4.solve_daily(statistics, penalty=1.0, bias=1)
    reproduced = v4.Decoder(config4, mean, scale, weights)
    assert statistics["n"] == receipt["n_training_labels"]
    np.testing.assert_array_equal(usable, v3.training_mask(mask[:prefix], clipped_segments))
    np.testing.assert_allclose(mean, original.mean, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(scale, original.scale, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(weights[:-1], original.coefficients, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(weights[-1], original.intercept, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(
        reproduced.predict(features, segments),
        original.predict(counts, power, segments),
        atol=1e-12,
        rtol=1e-12,
    )
