"""Scoring cannot hide abstentions, unavailable rows, or false acceptances."""

import copy

import pytest

from neural_experiments.v6 import metrics


def rows():
    result = []
    for person in metrics.PEOPLE:
        for day in range(3):
            for method in metrics.METHODS:
                r2 = {
                    "zero": -0.02,
                    "history0": 0.1,
                    "daily120": 0.1,
                    "daily240": 0.2,
                    "daily360": 0.25,
                    "v5_adaptive": 0.15,
                    "confirmed": 0.2,
                }[method]
                result.append(
                    {
                        "participant": person,
                        "session_id": f"{person}-{day}",
                        "method": method,
                        "action": "abstain" if method == "zero" else "accept",
                        "training_label_seconds": 240,
                        "label_seconds": 300,
                        "available_label_seconds": 300,
                        "eval_mask_sha256": "same",
                        "metrics": {"r2": r2, "summed_mse": 1 - r2, "angular_error_degrees": 60},
                    }
                )
    return result


def test_all_abstain_cannot_pass_useful_or_false_acceptance_gate():
    data = rows()
    for r in data:
        if r["method"] == "confirmed":
            r["action"] = "abstain"
            r["metrics"].update(r2=-0.02, summed_mse=1.02)
    s = metrics.summarize(data)
    assert s["methods"]["confirmed"]["accepted_scored_sessions"] == 0
    assert not s["gates"]["useful_cohort_accuracy"]
    assert not s["gates"]["fewer_false_acceptances_with_useful_coverage"]
    assert not s["gates"]["T2_recovered_without_harmful_acceptance"]


def test_opportunity_and_failure_accounting_preserves_cost_and_short_cases():
    data = rows()
    for r in data:
        if r["session_id"] == "T6-0":
            r["metrics"].update(r2=None, summed_mse=None)
        if r["session_id"] == "T2-0" and r["method"] == "confirmed":
            r["metrics"].update(r2=-0.01, summed_mse=1.01)
        if r["session_id"] == "T2-1" and r["method"] == "confirmed":
            r["metrics"].update(r2=-0.1, summed_mse=1.1)
    s = metrics.summarize(data)
    c = s["methods"]["confirmed"]
    assert (c["requested_sessions"], c["scored_sessions"], c["unavailable_sessions"]) == (24, 23, 1)
    assert c["mean_requested_label_seconds"] == 300
    assert c["nonpositive_accepted_sessions"] == ["T2-0", "T2-1"]
    assert c["harmful_vs_zero_accepted_sessions"] == ["T2-1"]
    assert s["later_label_opportunities"]["n"] == 23
    assert s["later_label_opportunities"]["captured"] == 21
    assert s["risk_contrasts_vs_v5"]["harmful_burden"]["mean_difference"] > 0


def test_masks_inventory_and_availability_must_match():
    original = rows()
    for change in ("mask", "availability", "duplicate"):
        data = copy.deepcopy(original)
        if change == "mask":
            data[0]["eval_mask_sha256"] = "different"
        elif change == "availability":
            data[0]["metrics"]["r2"] = None
        else:
            data.append(data[0])
        with pytest.raises(ValueError):
            metrics.summarize(data)
