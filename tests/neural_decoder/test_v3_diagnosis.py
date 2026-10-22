import numpy as np
import pytest

from neural_experiments.v3.diagnosis import decompose, project_unit_ball, summarize_rows


def directions():
    angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    return np.stack((np.cos(angle), np.sin(angle)), axis=1)


def test_constant_predictor_has_undefined_correlation_and_correct_error_components():
    y = directions()
    prediction = np.broadcast_to([0.4, -0.3], y.shape)
    result = decompose(y, prediction, np.ones(len(y), bool), [(0, len(y))])
    assert result["centered_xy_correlation"] is None
    assert result["r2"] == pytest.approx(-0.25)
    assert result["centered_r2_diagnostic"] == pytest.approx(0)
    assert result["mean_bias_sse"] == pytest.approx(8)
    assert result["mean_bias_sse"] + result["centered_error_sse"] == pytest.approx(result["sse"])


def test_bias_only_prediction_error_and_block_decomposition():
    y = directions()
    prediction = y + [0.6, -0.2]
    result = decompose(y, prediction, np.ones(len(y), bool), [(0, 16), (16, 32)])
    assert result["bias_fraction_sse"] == pytest.approx(1)
    assert result["block_bias_fraction_sse"] == pytest.approx(1)
    assert result["centered_error_sse"] == pytest.approx(0, abs=1e-25)
    assert result["centered_r2_diagnostic"] == pytest.approx(1)
    assert result["centered_xy_correlation"] == pytest.approx(1)
    prediction[:16] = y[:16] + [0.5, 0]
    prediction[16:] = y[16:] - [0.5, 0]
    result = decompose(y, prediction, np.ones(len(y), bool), [(0, 16), (16, 32)])
    assert result["bias_fraction_sse"] == pytest.approx(0, abs=1e-25)
    assert result["block_bias_fraction_sse"] == pytest.approx(1)


def test_fixed_unit_ball_projection_preserves_interior_and_cannot_increase_error():
    rng = np.random.default_rng(41)
    y = directions()
    prediction = rng.normal(size=y.shape) * 4
    prediction[0] = [0.3, -0.4]
    projected = project_unit_ball(prediction)
    np.testing.assert_array_equal(projected[0], prediction[0])
    assert np.all(np.linalg.norm(projected, axis=1) <= 1 + 1e-15)
    assert np.all(np.square(projected - y).sum(1) <= np.square(prediction - y).sum(1) + 1e-12)
    assert not np.shares_memory(projected, prediction)


def test_mask_excludes_unscored_values_and_requires_complete_segment_coverage():
    y = directions()
    prediction = y.copy()
    mask = np.ones(len(y), bool)
    mask[0] = False
    y[0], prediction[0] = np.nan, np.nan
    assert decompose(y, prediction, mask, [(0, 32)])["r2"] == pytest.approx(1)
    with pytest.raises(ValueError, match="cover"):
        decompose(y, prediction, mask, [(0, 31)])
    with pytest.raises(ValueError, match="overlapping"):
        decompose(y, prediction, mask, [(0, 32), (1, 2)])


def test_equal_person_summary_does_not_pool_rows_or_samples():
    rows = [
        {"method": "model", "partition": "future_day", "participant": "A", "r2": 0.2},
        {"method": "model", "partition": "future_day", "participant": "A", "r2": 0.4},
        {"method": "model", "partition": "future_day", "participant": "B", "r2": 0.9},
    ]
    result = summarize_rows(rows)[0]
    assert result["r2"] == pytest.approx(0.6)
    assert result["people"] == 2
