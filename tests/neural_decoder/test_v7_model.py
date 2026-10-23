import numpy as np
import pytest
from sklearn.linear_model import Ridge

from neural_experiments.v7 import model


@pytest.fixture
def problem():
    rng = np.random.default_rng(731)
    raw = rng.normal(size=(14000, 4))
    y = rng.normal(size=(14000, 2))
    mask = rng.random(14000) > 0.1
    segments = [(0, 7000), (7000, 14000)]
    return raw, y, mask, segments


@pytest.mark.parametrize("method", model.METHODS)
def test_independent_weighted_ridge_parity_and_past_only_fit(problem, method, tmp_path):
    raw, y, mask, segments = problem
    fitted, receipt = model.fit(raw, y, mask, segments, 2, method, 13600)
    start, stop = receipt["label_start_bin"], receipt["label_stop_bin"]
    selected = mask[:stop].copy()
    selected[:start] = False
    selected[:62] = False
    selected[7000:7062] = False
    rows = np.flatnonzero(selected)
    z = np.clip((raw[:stop][selected] - fitted.mean) / fitted.scale, -8, 8)
    w = np.exp2(-(stop - 1 - rows) * 0.02 / 60) if method == "exponential60" else np.ones(len(rows))
    reference = Ridge(alpha=w.sum(), fit_intercept=True).fit(z, y[:stop][selected], sample_weight=w)
    np.testing.assert_allclose(reference.coef_.T, fitted.weights[:-1], atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(reference.intercept_, fitted.weights[-1], atol=2e-13, rtol=2e-13)
    changed_x, changed_y = raw.copy(), y.copy()
    changed_x[stop:] = np.nan
    changed_y[stop:] = np.nan
    changed_y[:start] = np.nan
    replay, _ = model.fit(changed_x, changed_y, mask, segments, 2, method, 13600)
    np.testing.assert_array_equal(replay.weights, fitted.weights)
    np.testing.assert_array_equal(replay.mean, fitted.mean)
    assert receipt["last_training_bin"] < stop <= 13500
    assert receipt["training_rows"] == selected.sum()
    normal = raw[start:stop] if method.endswith("local") else raw[:1500]
    np.testing.assert_array_equal(fitted.mean, normal.mean(0))
    path = tmp_path / "decoder.npz"
    fitted.save(path)
    np.testing.assert_array_equal(model.Decoder.load(path).weights, fitted.weights)


def test_masked_nonfinite_labels_do_not_enter_fit(problem):
    raw, y, mask, segments = problem
    first, _ = model.fit_mask(raw, y, mask, segments, 2, 13500)
    y[~mask] = np.nan
    y[:1500] = np.nan
    y[7000:7062] = np.nan
    second, _ = model.fit_mask(raw, y, mask, segments, 2, 13500)
    np.testing.assert_array_equal(first.weights, second.weights)


def test_insufficient_labels_fallback_exact_zero_and_empty_weights(problem):
    raw, y, mask, segments = problem
    mask[:] = False
    fitted, receipt = model.fit(raw, y, mask, segments, 2, "rolling60", 13600)
    assert receipt["fallback_zero"] and receipt["effective_sample_size"] == 0
    assert receipt["first_training_bin"] is None
    assert not np.any(model.predict_raw(fitted, raw))


def test_cached_representation_prediction_equals_portable_decoder():
    rng = np.random.default_rng(77)
    features = {key: rng.random((14000, 2)) for key in ("x_tx3", "x_tx45", "x_sbp")}
    segments = [(0, 8000), (8000, 14000)]
    raw = model.representation(features, segments, model.CONFIG)
    y = rng.normal(size=(14000, 2))
    fitted, _ = model.fit(raw, y, np.ones(14000, bool), segments, 2, "rolling120", 13600)
    np.testing.assert_allclose(
        model.predict_raw(fitted, raw), fitted.predict(features, segments), atol=2e-15, rtol=2e-15
    )


@pytest.mark.parametrize("stop", [1499, 14001, True, 1500.0])
def test_reject_invalid_fit_prefix(problem, stop):
    raw, y, mask, segments = problem
    with pytest.raises(ValueError, match="past-only"):
        model.fit_mask(raw, y, mask, segments, 2, stop)


def test_reject_incomplete_prefix_and_unplanned_early_update(problem):
    raw, y, mask, segments = problem
    with pytest.raises(ValueError, match="past-only"):
        model.fit_mask(raw, y[:1500], mask[:1500], segments, 2, 3000)
    with pytest.raises(ValueError, match="272-second"):
        model.fit(raw, y, mask, segments, 2, "rolling240", 13599)
    with pytest.raises(ValueError, match="known V7"):
        model.fit(raw, y, mask, segments, 2, "rolling30", 13600)


def test_explicit_mask_update_can_select_sparse_paid_labels(problem):
    raw, y, _, segments = problem
    mask = np.zeros(len(y), bool)
    mask[1500:2000] = True
    mask[9000:10000] = True
    fitted, receipt = model.fit_mask(raw, y, mask, segments, 2, 12000)
    assert receipt["training_rows"] == 1500
    assert receipt["first_training_bin"] == 1500
    assert receipt["last_training_bin"] == 9999
    assert np.isfinite(fitted.weights).all()
