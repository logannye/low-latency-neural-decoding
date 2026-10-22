import numpy as np
import pytest
from sklearn.linear_model import Ridge

from neural_experiments.v6 import model


@pytest.mark.parametrize("normalization", ["first30", "fit_available"])
def test_prefix_only_fit_no_future_normalization_or_label_leak(normalization, tmp_path):
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(2500, 3))
    y = rng.normal(size=(2500, 2))
    mask = np.ones(2500, bool)
    segments = [(0, 1750), (1750, 2500)]
    stop = 2137
    candidate = next(
        c for c in model.candidate_grid() if c["candidate_id"] == f"tx45_scratch1_{normalization}"
    )
    fitted, receipt = model.fit_candidate(
        raw, y[:stop], mask[:stop], segments, stop, candidate, None, 3
    )
    changed_raw, changed_y = raw.copy(), y.copy()
    changed_raw[stop:] *= 1e6
    changed_y[stop:] = np.nan
    other, _ = model.fit_candidate(changed_raw, changed_y, mask, segments, stop, candidate, None, 3)
    np.testing.assert_array_equal(fitted.weights, other.weights)
    np.testing.assert_array_equal(fitted.mean, other.mean)
    assert receipt["normalization_stop_bin"] == (1500 if normalization == "first30" else stop)
    valid = mask[:stop].copy()
    valid[:1500], valid[1750:1812] = False, False
    z = np.clip((raw[:stop] - fitted.mean) / fitted.scale, -8, 8)
    reference = Ridge(alpha=valid.sum(), fit_intercept=True).fit(z[valid], y[:stop][valid])
    np.testing.assert_allclose(fitted.weights[:-1], reference.coef_.T, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(fitted.weights[-1], reference.intercept_, atol=1e-12, rtol=1e-12)
    assert receipt["training_rows"] == int(valid.sum())
    path = tmp_path / "model.npz"
    fitted.save(path)
    exported = model.Decoder.load(path)
    np.testing.assert_array_equal(exported.weights, fitted.weights)
    np.testing.assert_array_equal(exported.scale, fitted.scale)


def test_incomplete_prefix_cannot_be_silently_confirmed_or_fitted():
    candidate = model.candidate_grid()[0]
    raw, y, mask = np.ones((2000, 3)), np.ones((2000, 2)), np.ones(2000, bool)
    with pytest.raises(ValueError, match="complete prospective"):
        model.fit_candidate(raw, y[:1800], mask[:1800], [(0, 2000)], 1900, candidate, None, 3)
    with pytest.raises(ValueError, match="complete prospective"):
        model.fit_candidate(raw, y, mask, [(0, 2000)], 2100, candidate, None, 3)
