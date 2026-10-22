import json

import numpy as np
import pytest

from neural_experiments.v4.diagnosis import metric, read_exposed_index, target_stats


def test_diagnosis_rejects_new_or_changed_index_before_array_loading(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"role": "t3_blocks", "sessions": []}))
    with pytest.raises(ValueError, match="exact previously exposed"):
        read_exposed_index(path, "v3_assessment")
    with pytest.raises(ValueError, match="exact previously exposed"):
        read_exposed_index(path, None)


def test_diagnostic_zero_and_bias_decomposition():
    y = np.tile(np.eye(2), (50, 1))
    mask = np.ones(100, dtype=bool)
    zero = metric(y, np.zeros_like(y), mask)
    assert zero["r2"] == pytest.approx(-1)
    assert zero["angular_degrees"] == 90
    biased = metric(y, y + [2, -3], mask)
    assert biased["mean_bias_sse_fraction"] == pytest.approx(1)
    assert biased["centered_correlations"] == pytest.approx([1, 1])


def test_direction_autocorrelation_excludes_block_crossing_and_mask_gaps():
    y = np.tile([1.0, 0.0], (20, 1))
    y[10:] = [-1, 0]
    q = np.r_[np.zeros(10), np.ones(10)]
    mask = np.ones(20, dtype=bool)
    mask[4] = False
    result = target_stats(y, mask, q, 0, 20)
    lag1 = result["autocorrelation"][0]
    assert lag1["n_contiguous_go_pairs"] == 16
    assert lag1["mean_direction_dot"] == pytest.approx(1)
