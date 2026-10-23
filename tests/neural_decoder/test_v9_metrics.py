"""Whole-session recovery decisions reject missing, mismatched or weak evidence."""

import copy

import pytest

from neural_experiments.v9 import metrics


def rows():
    result = []
    for i in range(6):
        for method in metrics.METHODS:
            r2 = 0.12 + i / 100 if method == metrics.PRIMARY else 0.01
            result.append(
                {
                    "participant": "T2",
                    "session_id": f"T2_day_{i}",
                    "post_implant_day": i,
                    "method": method,
                    "score_mask_sha256": str(i),
                    "metrics": {"r2": r2, "summed_mse": 1 - r2},
                    "calibration": {"quota_attained": False, "timed_out": True},
                }
            )
    return result


def test_positive_session_level_evidence_and_retained_timeouts():
    result = metrics.summarize(rows())
    assert all(result["gates"].values())
    assert result["methods"][metrics.PRIMARY]["participants"][0]["scored_sessions"] == 6


def test_missing_fresh_session_cannot_pass_recovery():
    result = metrics.summarize([r for r in rows() if r["post_implant_day"] != 5])
    assert not any(result["gates"].values())


def test_more_labels_explain_gain_without_incremental_trial_benefit():
    data = rows()
    for row in data:
        if row["method"] == "duration":
            row["metrics"]["r2"] = 0.3
    result = metrics.summarize(data)
    assert result["gates"]["useful_T2_recovery"]
    assert not result["gates"]["incremental_benefit_over_more_labels"]


def test_mean_gain_does_not_override_weak_session_pairing():
    data = rows()
    for row in data:
        if row["method"] == metrics.PRIMARY:
            row["metrics"]["r2"] = -0.1 if row["post_implant_day"] < 2 else 0.2
    result = metrics.summarize(data)
    assert result["methods"][metrics.PRIMARY]["participants"][0]["mean_r2"] > 0.05
    assert not result["gates"]["useful_T2_recovery"]


@pytest.mark.parametrize("mutation", ["mask", "missing", "duplicate", "availability"])
def test_incompatible_comparisons_fail_closed(mutation):
    data = copy.deepcopy(rows())
    if mutation == "mask":
        data[0]["score_mask_sha256"] = "changed"
    elif mutation == "missing":
        data.pop()
    elif mutation == "duplicate":
        data.append(copy.deepcopy(data[0]))
    else:
        data[0]["metrics"]["r2"] = None
    with pytest.raises(ValueError):
        metrics.summarize(data)


def test_participants_have_equal_weight_despite_unequal_session_counts():
    data = rows()
    other = copy.deepcopy(data[: len(metrics.METHODS)])
    for row in other:
        row.update(participant="T7", session_id="T7_one")
        row["metrics"]["r2"] = 0.5
    result = metrics.summarize(data + other)
    assert result["methods"]["baseline"]["mean_r2"] == pytest.approx(0.255)
