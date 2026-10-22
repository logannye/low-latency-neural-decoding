"""Aggregate figure semantics: fixed cohort, equal-person means, no invented rows."""

import copy

import numpy as np
import pytest

from neural_experiments.v2 import figures


def transfer_summary():
    groups = []
    for method_index, method in enumerate(figures.METHODS):
        for budget in (30, 120, 300):
            people = []
            for person_index, person in enumerate(figures.PEOPLE):
                if person == "T6" and budget == 300:
                    continue
                people.append(
                    {
                        "participant": person,
                        "r2": person_index + budget / 1000 + method_index,
                        "n_seeds": 1 if method == "ridge" else 3,
                    }
                )
            groups.append(
                {
                    "method": method,
                    "budget_seconds": budget,
                    "partition": "future_day",
                    "people": people,
                }
            )
    return {
        "status": "complete",
        "groups": groups,
        "primary_contrast": {
            "effects": [{"participant": p, "difference": 0.1} for p in figures.PEOPLE]
        },
    }


def test_calibration_curves_keep_identical_people_at_every_budget():
    source = transfer_summary()
    untouched = copy.deepcopy(source)
    summary, budgets, common, means = figures.transfer_curves(source)
    assert budgets == [30, 120, 300]
    assert common == [p for p in figures.PEOPLE if p != "T6"]
    eligible_person_mean = np.mean([i for i, p in enumerate(figures.PEOPLE) if p != "T6"])
    np.testing.assert_allclose(means["ridge"], eligible_person_mean + np.array(budgets) / 1000)
    # The 120-second paired panel still retains T6 and all nine eligible people.
    assert len(summary["primary_contrast"]["effects"]) == 9
    assert source == untouched


def test_missing_neural_seed_excludes_person_from_entire_curve():
    source = transfer_summary()
    row = next(
        g for g in source["groups"] if g["method"] == "scratch" and g["budget_seconds"] == 30
    )
    row["people"][0]["n_seeds"] = 2
    _, _, common, _ = figures.transfer_curves(source)
    assert "T2" not in common and "T6" not in common
    assert len(common) == 7


def test_missing_method_budget_group_is_not_silently_interpolated():
    source = transfer_summary()
    source["groups"].pop()
    with pytest.raises(ValueError, match="Missing or duplicate"):
        figures.transfer_curves(source)


def test_h1_means_require_all_days_and_seeds():
    rows = [
        {"day": day, "seed": seed, "scores": {"r2_mean": index + seed / 10}}
        for index, day in enumerate(("S2", "S3", "S4"))
        for seed in range(3)
    ]
    mean, days = figures.h1_mean(rows)
    assert mean == pytest.approx(1.1)
    assert days == pytest.approx({"S2": 0.1, "S3": 1.1, "S4": 2.1})
    with pytest.raises(ValueError, match="all three seeds"):
        figures.h1_mean(rows[:-1])
    with pytest.raises(ValueError, match="all three seeds"):
        figures.h1_mean(rows + [rows[0]])
