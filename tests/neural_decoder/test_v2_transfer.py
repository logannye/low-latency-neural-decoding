"""Causal representation, calibration fairness, and bounded streaming contracts."""

import copy

import numpy as np
import pytest
import torch

from neural_experiments.v2.transfer import (
    AdapterTCN,
    IncrementalDecoder,
    Recording,
    adapt,
    export_model,
    filtered,
    fit_transform,
    load_model,
    make_chunks,
    predict,
    score,
    training_mask,
)


@pytest.fixture
def record():
    rng = np.random.default_rng(17)
    x = rng.poisson(1.1, (420, 6)).astype(np.float32)
    angle = np.linspace(-2.8, 2.8, len(x))
    y = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
    mask = np.ones(len(x), dtype=bool)
    mask[30:55] = False
    return Recording(
        {"session_id": "test", "participant_id": "A", "post_implant_day": 0},
        x,
        y,
        mask,
        [(0, 210), (210, 420)],
    )


def test_future_observations_cannot_change_past_predictions(record):
    torch.manual_seed(4)
    model = AdapterTCN(6, 8, (1, 2, 4))
    stats = fit_transform([record.prefix(100)])
    original = predict(model, record.x, record.segments, stats)
    changed = record.x.copy()
    changed[250:] = 1000
    result = predict(model, changed, record.segments, stats)
    np.testing.assert_array_equal(original[:250], result[:250])


def test_incremental_matches_full_causal_network_and_reset(record):
    torch.manual_seed(9)
    model = AdapterTCN(6, 8, (1, 2, 4, 8, 16))
    stats = fit_transform([record.prefix(100)])
    expected = predict(model, record.x, record.segments, stats)
    stream = IncrementalDecoder(model, stats)
    actual = []
    nbytes = stream.state_nbytes
    for start, stop in record.segments:
        stream.reset()
        actual.extend(stream.step(row) for row in record.x[start:stop])
        assert stream.state_nbytes == nbytes
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
    stream.reset()
    np.testing.assert_allclose(stream.step(record.x[0]), expected[0], atol=1e-6)


def test_block_reset_removes_entire_previous_recording(record):
    model = AdapterTCN(6, 4, (1, 2))
    stats = fit_transform([record.prefix(100)])
    original = predict(model, record.x, record.segments, stats)
    changed = record.x.copy()
    changed[:210] = 999
    result = predict(model, changed, record.segments, stats)
    np.testing.assert_array_equal(original[210:], result[210:])


def test_training_chunks_are_block_local_and_use_identical_effective_labels(record):
    stats = fit_transform([record])
    x, y, masks = make_chunks([record], stats, burn=14, length=64)
    assert x.shape[1] == 78
    assert not masks[:, :14].any()
    assert masks.sum().item() == training_mask(record, 14).sum()
    expected_labels = record.y[training_mask(record, 14)]
    np.testing.assert_array_equal(y[masks].numpy(), expected_labels)
    # Changing the previous block cannot enter a later block's chunks.
    changed = copy.deepcopy(record)
    changed.x[:210] = 999
    updated = make_chunks([changed], stats, burn=14, length=64)[0]
    np.testing.assert_array_equal(x[4:].numpy(), updated[4:].numpy())


def test_prefix_statistics_and_fit_do_not_see_future_labels(record):
    plan = {
        "model": {"width": 4, "dilations": [1, 2], "channel_dropout": 0.1},
        "adaptation": {
            "epochs": 2,
            "batch_chunks": 2,
            "chunk_bins": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
    }
    changed = copy.deepcopy(record)
    changed.x[100:] = 9000
    changed.y[100:] = -7000
    changed.mask[100:] = ~changed.mask[100:]
    first, stats, _ = adapt(record.prefix(100), plan, 1, "scratch", None)
    second, changed_stats, _ = adapt(changed.prefix(100), plan, 1, "scratch", None)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(stats, changed_stats)


def test_frozen_core_is_unchanged_by_target_calibration(record):
    model = AdapterTCN(6, 4, (1, 2))
    before = copy.deepcopy(model.core.state_dict())
    plan = {
        "model": {"width": 4, "dilations": [1, 2], "channel_dropout": 0.1},
        "adaptation": {
            "epochs": 2,
            "batch_chunks": 2,
            "chunk_bins": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
    }
    fitted, _, _ = adapt(record.prefix(100), plan, 1, "pretrained_frozen_core", before)
    for key, value in fitted.core.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in fitted.core.parameters())


def test_model_export_restore_preserves_predictions(record, tmp_path):
    model = AdapterTCN(6, 4, (1, 2))
    stats = fit_transform([record.prefix(100)])
    path = tmp_path / "model.npz"
    export_model(model, stats, path)
    restored, restored_stats = load_model(path)
    np.testing.assert_array_equal(
        predict(model, record.x, record.segments, stats),
        predict(restored, record.x, record.segments, restored_stats),
    )
    with pytest.raises(FileExistsError):
        export_model(model, stats, path)


@pytest.mark.parametrize("replacement", [np.nan, np.inf, -1.0])
def test_invalid_features_do_not_advance_stream(record, replacement):
    model = AdapterTCN(6, 4, (1, 2))
    stats = fit_transform([record.prefix(100)])
    stream = IncrementalDecoder(model, stats)
    invalid = record.x[0].copy()
    invalid[0] = replacement
    with pytest.raises(ValueError):
        stream.step(invalid)
    np.testing.assert_allclose(
        stream.step(record.x[0]), IncrementalDecoder(model, stats).step(record.x[0]), atol=0, rtol=0
    )


def test_linear_filter_is_causal_and_resets_per_block(record):
    stats = fit_transform([record.prefix(100)])
    cfg = {"fir_taps": 12, "fir_tau_bins": 12}
    original = filtered(record.x, record.segments, stats, cfg)
    changed = record.x.copy()
    changed[:210] = 1000
    result = filtered(changed, record.segments, stats, cfg)
    np.testing.assert_array_equal(original[210:], result[210:])
    changed = record.x.copy()
    changed[100:] = 1000
    result = filtered(changed, record.segments, stats, cfg)
    np.testing.assert_array_equal(original[:100], result[:100])


def test_score_penalizes_constant_coordinate_error_and_zero_vectors():
    y = np.array([[0.0, -1.0], [0.0, 1.0]])
    prediction = np.array([[10.0, -1.0], [10.0, 1.0]])
    result = score(y, prediction, np.ones(2, dtype=bool))
    assert result["r2"] == -99
    assert result["per_output_r2"] == [None, 1.0]
    result = score(y, np.zeros_like(y), np.ones(2, dtype=bool))
    assert result["angular_error_degrees"] == 90
    assert result["zero_prediction_count"] == 2
