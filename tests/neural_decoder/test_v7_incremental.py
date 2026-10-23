import numpy as np
import pytest

from neural_experiments.v7 import model
from neural_experiments.v7.incremental import CenteredRidge


def example():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(2200, 6))
    y = np.column_stack((0.4 * raw[:, 0] + 0.3, -0.3 * raw[:, 2] - 0.1))
    paid = np.arange(len(raw)) % 5 != 0
    valid = np.arange(len(raw)) % 7 != 0
    starts = np.where(np.arange(len(raw)) < 1800, 0, 1800)
    y[~(paid & valid)] = np.nan
    return raw, y, paid, valid, starts


@pytest.mark.parametrize("chunk", [1, 17, 64])
def test_centered_updates_match_batch_ridge_with_burn_and_unpurchased_nan(chunk):
    raw, y, paid, valid, starts = example()
    expected, receipt = model.fit_mask(raw, y, paid & valid, [(0, 1800), (1800, 2200)], 3, 2200)
    state = CenteredRidge(expected.mean, expected.scale)
    size = state.moment_nbytes + state.normalizer_nbytes
    for a in range(1500, 2200, chunk):
        b = min(a + chunk, 2200)
        state.add(
            raw[a:b], y[a:b], np.arange(a, b), paid[a:b], valid[a:b], starts[a:b], observed_stop=b
        )
        assert state.moment_nbytes + state.normalizer_nbytes == size
    assert state.n == receipt["training_rows"]
    np.testing.assert_allclose(state.decoder().weights, expected.weights, atol=2e-14, rtol=1e-12)
    assert all(value.ndim <= 2 for value in state.snapshot().values())
    assert not any(value.shape == raw.shape for value in state.snapshot().values())


def test_future_repeat_and_large_block_rejected_without_state_change():
    state = CenteredRidge(np.zeros(4), np.ones(4))
    raw, y = np.zeros((3, 4)), np.zeros((3, 2))
    indices = np.arange(1500, 1503)
    flags = np.ones(3, bool)
    starts = np.zeros(3, int)
    with pytest.raises(ValueError, match="chronological"):
        state.add(raw, y, indices, flags, flags, starts, observed_stop=1502)
    assert state.n == 0 and state.last_bin == 1499
    state.add(raw, y, indices, flags, flags, starts, observed_stop=1503)
    with pytest.raises(ValueError, match="chronological"):
        state.add(raw, y, indices, flags, flags, starts, observed_stop=1503)
    with pytest.raises(ValueError, match="bounded"):
        state.add(
            np.zeros((65, 4)),
            np.zeros((65, 2)),
            np.arange(1503, 1568),
            np.ones(65, bool),
            np.ones(65, bool),
            np.zeros(65, int),
            observed_stop=1568,
        )


def test_unpurchased_and_burn_labels_do_not_enter_state():
    state = CenteredRidge(np.zeros(4), np.ones(4))
    y = np.full((64, 2), np.nan)
    raw, indices = np.ones((64, 4)), np.arange(1500, 1564)
    flags = np.ones(64, bool)
    paid = np.zeros(64, bool)
    state.add(raw, y, indices, paid, flags, np.zeros(64, int), observed_stop=1564)
    assert state.n == 0
    np.testing.assert_array_equal(state.decoder().weights, np.zeros((5, 2)))
    y[:] = 0
    y[:62] = np.nan
    state.add(raw, y, indices + 64, flags, flags, np.full(64, 1564), observed_stop=1628)
    assert state.n == 2
    np.testing.assert_array_equal(state.decoder().weights, np.zeros((5, 2)))


def test_fixed_normalizer_is_copied_and_constant_targets_have_free_intercept():
    mean, scale = np.zeros(4), np.ones(4)
    state = CenteredRidge(mean, scale)
    mean[:] = 123
    scale[:] = 456
    rng = np.random.default_rng(22)
    y = np.tile([1e9, -1e9], (64, 1))
    state.add(
        rng.normal(size=(64, 4)),
        y,
        np.arange(1500, 1564),
        np.ones(64, bool),
        np.ones(64, bool),
        np.zeros(64, int),
        observed_stop=1564,
    )
    fitted = state.decoder()
    np.testing.assert_array_equal(fitted.weights[:-1], np.zeros((4, 2)))
    np.testing.assert_array_equal(fitted.weights[-1], [1e9, -1e9])
    np.testing.assert_array_equal(fitted.mean, np.zeros(4))


def test_selected_nonfinite_label_rejected_transactionally():
    state = CenteredRidge(np.zeros(4), np.ones(4))
    with pytest.raises(ValueError, match="finite"):
        state.add(
            np.zeros((1, 4)),
            np.full((1, 2), np.nan),
            np.array([1500]),
            np.ones(1, bool),
            np.ones(1, bool),
            np.zeros(1, int),
            observed_stop=1501,
        )
    assert state.n == 0 and state.last_bin == 1499


def test_unsigned_descending_bins_cannot_regress_or_duplicate_history():
    state = CenteredRidge(np.zeros(4), np.ones(4))
    with pytest.raises(ValueError, match="chronological"):
        state.add(
            np.zeros((2, 4)),
            np.zeros((2, 2)),
            np.array([2000, 1999], dtype=np.uint64),
            np.ones(2, bool),
            np.ones(2, bool),
            np.zeros(2, np.uint64),
            observed_stop=2001,
        )
    assert state.n == 0 and state.last_bin == 1499
