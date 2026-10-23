import copy

import pytest

from neural_experiments.v7 import continual_metrics as metrics


def rows():
    result = []
    for person in ("T2", "T5", "T6", "T7", "T8", "T9", "T10", "T11"):
        for day in range(3):
            for method in metrics.METHODS:
                value = {
                    "zero": -0.01,
                    "frozen120": 0.1,
                    "fixed240": 0.15,
                    "periodic240": 0.2,
                    "triggered240": 0.22,
                }[method]
                score = {"r2": value, "summed_mse": 1 - value, "angular_error_degrees": 60.0}
                cost = 0 if method == "zero" else 120 if method == "frozen120" else 240
                result.append(
                    {
                        "session_id": f"{person}_{day}",
                        "participant": person,
                        "post_implant_day": day,
                        "method": method,
                        "n_bins": 30000,
                        "score_mask_sha256": "same",
                        "metrics": score,
                        "planned_label_seconds": cost,
                        "available_label_seconds": cost,
                        "interruption_seconds": 128
                        if method in ("periodic240", "triggered240")
                        else 0,
                        "diagnostic_windows": [
                            {
                                "start_bin": 7600,
                                "stop_bin": 9100,
                                "valid_rows": 100,
                                "metrics": score,
                            }
                        ],
                    }
                )
    return result


def test_fresh_gates_use_paired_outcomes_and_equal_budgets():
    result = metrics.summarize(rows())
    assert all(result["gates"].values())
    assert result["T2"]["periodic240"]["mean_r2"] == pytest.approx(0.2)
    assert result["contrasts"]["triggered240_minus_periodic240"]["mean_gain"] == pytest.approx(0.02)


def test_unequal_cost_does_not_pass_timing_gate():
    data = rows()
    next(r for r in data if r["method"] == "triggered240")["available_label_seconds"] = 239
    result = metrics.summarize(data)
    assert not result["complete_sessions_matched_feedback_budget"]
    assert not result["gates"]["triggered_timing_value"]


def test_unavailable_t2_case_retained_without_replacement():
    data = rows()
    for row in data:
        if row["session_id"] == "T2_2":
            row["metrics"] = {"r2": None, "summed_mse": None, "angular_error_degrees": None}
            row["diagnostic_windows"] = []
    result = metrics.summarize(data)
    assert result["T2"]["periodic240"]["selected_sessions"] == 3
    assert result["T2"]["periodic240"]["scored_sessions"] == 2
    assert not result["gates"]["periodic_T2_recovery"]


def test_entire_person_excluded_from_threshold_ranking():
    grid = [{**copy.deepcopy(r), "threshold": t} for r in rows() for t in metrics.THRESHOLDS]
    for row in grid:
        if row["method"] == "triggered240":
            row["metrics"]["r2"] = 1 if row["threshold"] == 0.1 else 0
            if row["participant"] == "T2":
                row["metrics"]["r2"] = 1000 if row["threshold"] == 1 else -1000
    choice = metrics.select_source(grid)
    t2 = next(c for c in choice["choices"] if c["participant"] == "T2")
    assert t2["threshold"] == 0.1
    assert all(u["participant"] != "T2" for c in t2["candidates"] for u in c["source_units"])
    with pytest.raises(ValueError, match="complete"):
        metrics.select_source(grid[:-1])


def test_matched_masks_and_no_duplicates_required():
    data = rows()
    with pytest.raises(ValueError, match="duplicate"):
        metrics.summarize(data + [data[0]])
    data[0]["score_mask_sha256"] = "different"
    with pytest.raises(ValueError, match="common"):
        metrics.summarize(data)
