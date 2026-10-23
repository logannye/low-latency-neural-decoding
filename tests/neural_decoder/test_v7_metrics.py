import copy

import pytest

from neural_experiments.v7 import metrics


def example():
    rows, windows = [], []
    for day in range(8):
        for method in metrics.METHODS:
            value = 0.15 if method == "rolling120" else 0.01
            row = {
                "session_id": f"T2_{day}",
                "participant": "T2",
                "post_implant_day": day,
                "method": method,
                "score_mask_sha256": "same",
                "unique_requested_label_seconds": 120,
                "metrics": {"r2": value, "summed_mse": 0.5 - value, "angular_error_degrees": 60.0},
            }
            rows.append(row)
            windows.append({**row, "start_bin": 13600, "valid_rows": 100})
    return rows, windows


def test_gate_uses_fixed_latest_six_and_paired_gain():
    result = metrics.summarize(*example())
    assert result["primary_T2_session_ids"] == [f"T2_{d}" for d in range(2, 8)]
    assert result["gate"]["passing_methods"] == ["rolling120"]
    assert result["primary_T2"]["rolling120"]["mean_gain_over_frozen120"] == pytest.approx(0.14)


def test_unavailable_primary_is_retained_and_gate_fails():
    rows, windows = example()
    for row in rows + windows:
        if row["session_id"] == "T2_7":
            row["metrics"] = None
    result = metrics.summarize(rows, windows)
    assert result["primary_T2"]["rolling120"]["selected_sessions"] == 6
    assert result["primary_T2"]["rolling120"]["scored_sessions"] == 5
    assert not result["gate"]["passed"]


def test_mismatched_masks_and_duplicate_rows_rejected():
    rows, windows = example()
    bad = copy.deepcopy(rows)
    bad[0]["score_mask_sha256"] = "other"
    with pytest.raises(ValueError, match="matched"):
        metrics.summarize(bad, windows)
    with pytest.raises(ValueError, match="duplicate"):
        metrics.summarize(rows + [rows[0]], windows)


def test_positive_accuracy_without_improvement_fails():
    rows, windows = example()
    for row in rows + windows:
        row["metrics"]["r2"] = 0.2
    assert not metrics.summarize(rows, windows)["gate"]["passed"]


def test_bootstrap_is_deterministic_and_rejects_nonfinite():
    assert metrics.interval([1, 2, 3]) == metrics.interval([1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        metrics.interval([float("nan")])


def test_constant_direction_window_still_counts_mse_harm():
    rows, windows = example()
    for window in windows:
        window["metrics"] = dict(window["metrics"])
        window["metrics"]["r2"] = None
        window["metrics"]["summed_mse"] = 1.0 if window["method"] == "rolling120" else 0.5
    result = metrics.summarize(rows, windows)
    unit = result["methods"]["rolling120"]["participants"][0]
    assert unit["fraction_valid_time_in_windows_worse_than_zero"] == 1.0
