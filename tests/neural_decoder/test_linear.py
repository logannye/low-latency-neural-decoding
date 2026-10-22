import inspect

import numpy as np
import pytest

from neural_decoder.features import CausalFeatures, filter_file, fir_weights, history_matrix
from neural_decoder.models import LinearDecoder, fit_linear, load_model


def fixture():
    rng = np.random.default_rng(19)
    x = rng.poisson(2, (130, 4))
    y = np.c_[x[:, 0] - x[:, 1], x[:, 2] + 0.5 * x[:, 3]]
    return x, y, np.ones(len(x), dtype=bool)


def test_offline_features_equal_causal_steps_across_file_start():
    x, _, _ = fixture()
    weights = fir_weights()
    mean, scale = np.arange(4), np.arange(1, 5)
    features = CausalFeatures(mean, scale, 5, weights)
    actual = np.array([features.step(row, i) for i, row in enumerate(x)])
    expected = history_matrix(filter_file(x, weights), mean, scale, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    features.reset()
    np.testing.assert_allclose(features.step(x[0], 0), expected[0])


def test_future_intervention_chunking_reset_and_state():
    x, y, mask = fixture()
    model = fit_linear([(x, y, mask)], history=5)
    first, second = LinearDecoder(model), LinearDecoder(model)
    altered = x.copy()
    altered[65:] = 1234
    a, b = first.predict_chunk(x[:65]), second.predict_chunk(altered[:65])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.features.past, second.features.past)
    first.predict_chunk(x[65:])
    second.predict_chunk(altered[65:])
    whole = LinearDecoder(model).predict_chunk(x)
    chunked = LinearDecoder(model)
    parts = [chunked.predict_chunk(x[start:stop]) for start, stop in ((0, 3), (3, 28), (28, 130))]
    np.testing.assert_array_equal(whole, np.concatenate(parts))
    chunked.reset()
    np.testing.assert_array_equal(whole, chunked.predict_chunk(x))
    assert chunked.features.raw.shape == (12, 4)
    assert chunked.features.past.shape == (5, 4)


def test_scorer_only_label_and_mask_mutation_cannot_enter_replay(tmp_path):
    x, y, mask = fixture()
    model = fit_linear([(x, y, mask)])
    path = tmp_path / "ridge.npz"
    model.save(path)
    a = LinearDecoder(load_model(path))
    predictions = a.predict_chunk(x)
    y[:] = 1e6
    mask[:] = False
    b = LinearDecoder(load_model(path))
    np.testing.assert_array_equal(predictions, b.predict_chunk(x))
    np.testing.assert_array_equal(a.features.raw, b.features.raw)
    np.testing.assert_array_equal(a.features.past, b.features.past)
    assert set(inspect.signature(a.step).parameters) == {"counts", "sequence_id"}


def test_no_trial_reset_and_file_reset():
    x, y, mask = fixture()
    model = fit_linear([(x, y, mask)], history=5)
    decoder = LinearDecoder(model)
    prefix = decoder.predict_chunk(x[:17])
    suffix = decoder.predict_chunk(x[17:])
    np.testing.assert_array_equal(np.r_[prefix, suffix], LinearDecoder(model).predict_chunk(x))
    decoder.reset()
    np.testing.assert_array_equal(decoder.predict_chunk(x), np.r_[prefix, suffix])


def test_invalid_input_does_not_mutate_state():
    x, y, mask = fixture()
    decoder = LinearDecoder(fit_linear([(x, y, mask)]))
    for bad in (np.zeros(3), np.full(4, np.nan), np.full(4, -1), np.zeros((1, 4))):
        with pytest.raises(ValueError):
            decoder.step(bad)
    with pytest.raises(ValueError):
        decoder.step(x[0], sequence_id=2)
    assert decoder.features.next_sequence_id == 0
    decoder.step(x[0], sequence_id=0)
    with pytest.raises(ValueError):
        decoder.step(x[0], sequence_id=0)


def test_ridge_matches_independent_augmented_least_squares():
    x, y, mask = fixture()
    model = fit_linear([(x, y, mask)], history=2, regularization=0.03)
    design = history_matrix(filter_file(x, model.weights), model.mean, model.scale, 2)
    centered = design - design.mean(axis=0)
    augmented = np.r_[centered, np.sqrt(len(x) * 0.03) * np.eye(centered.shape[1])]
    target = np.r_[y - y.mean(axis=0), np.zeros((centered.shape[1], 2))]
    coefficient = np.linalg.lstsq(augmented, target, rcond=None)[0]
    np.testing.assert_allclose(model.coefficient, coefficient, atol=1e-11)
    np.testing.assert_allclose(
        LinearDecoder(model).predict_chunk(x), design @ coefficient + model.intercept, atol=1e-11
    )


def test_constant_is_training_target_mean_and_bad_artifact_rejected(tmp_path):
    x, y, mask = fixture()
    mask[::3] = False
    model = fit_linear([(x, y, mask)], constant=True)
    np.testing.assert_allclose(
        LinearDecoder(model).predict_chunk(x), np.tile(y[mask].mean(0), (len(x), 1))
    )
    path = tmp_path / "bad.npz"
    np.savez(path, schema_version=9)
    with pytest.raises(ValueError):
        load_model(path)
