"""Outer-person exclusion, balanced scoring, and fixed study inventory."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import r2_score

from neural_experiments.v3 import study


def test_numpy_scores_match_sklearn_and_zero_direction_angle():
    rng = np.random.default_rng(17)
    actual = rng.normal(size=(100, 2))
    actual /= np.linalg.norm(actual, axis=1)[:, None]
    predicted = actual * 0.4 + rng.normal(size=actual.shape) * 0.3
    mask = np.arange(100) % 3 != 0
    result = study.score(actual, predicted, mask)
    assert result["r2"] == pytest.approx(
        r2_score(actual[mask], predicted[mask], multioutput="variance_weighted")
    )
    assert study.score(actual, np.zeros_like(actual), mask)["angular_error_degrees"] == 90


def test_locked_grid_has_36_distinct_configs_and_stable_order():
    path = Path(__file__).parents[2] / "neural_experiments/v3/plan.json"
    plan = json.loads(path.read_text())
    candidates = study.candidate_grid(plan)
    assert len(candidates) == 36
    assert len({json.dumps(c["config"], sort_keys=True) for c in candidates}) == 36
    assert candidates[0]["config"]["family"] == "tx"
    assert candidates[-1]["config"]["family"] == "tx_sbp"


def selector_fixture():
    plan = {"participants": ["a", "b", "c"], "calibration_budgets_seconds": [120]}
    candidates = [
        {"candidate_id": "x", "config": {"family": "tx"}},
        {"candidate_id": "y", "config": {"family": "tx_sbp"}},
    ]
    rows = []
    for person in plan["participants"]:
        for candidate, value in (("x", 0.2), ("y", 0.3)):
            rows.append(
                {
                    "participant": person,
                    "session_id": person,
                    "budget_seconds": 120,
                    "candidate_id": candidate,
                    "metrics": {"r2": value},
                }
            )
    return plan, candidates, rows


def test_target_person_outcomes_cannot_change_its_selected_recipe():
    plan, candidates, rows = selector_fixture()
    baseline = study.choose(rows, candidates, plan)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["participant"] == "a":
            row["metrics"]["r2"] = 1000 if row["candidate_id"] == "x" else -1000
    after = study.choose(changed, candidates, plan)
    a_before = [c for c in baseline if c["target_participant"] == "a"]
    a_after = [c for c in after if c["target_participant"] == "a"]
    assert a_before == a_after
    assert all("a" not in c["source_participants"] for c in a_after)


def test_participants_are_weighted_equally_despite_session_counts():
    plan, candidates, rows = selector_fixture()
    # For held-out a, b has 20 sessions favoring x, c has one favoring y more.
    rows = []
    for person, n, vx, vy in (("b", 20, 0.4, 0), ("c", 1, -0.5, 0.3)):
        for day in range(n):
            for candidate, value in (("x", vx), ("y", vy)):
                rows.append(
                    {
                        "participant": person,
                        "session_id": f"{person}{day}",
                        "budget_seconds": 120,
                        "candidate_id": candidate,
                        "metrics": {"r2": value},
                    }
                )
    choices = study.choose(rows, candidates, plan)
    choice = next(
        c for c in choices if c["target_participant"] == "a" and c["method"] == "selected_daily"
    )
    assert choice["candidate_id"] == "y"
    assert choice["objective"] == pytest.approx(0.15)


def test_exact_ties_keep_first_grid_candidate():
    plan, candidates, rows = selector_fixture()
    for row in rows:
        row["metrics"]["r2"] = 0.1
    choices = study.choose(rows, candidates, plan)
    assert all(c["candidate_id"] == "x" for c in choices if c["method"] == "selected_daily")


def test_prefix_cannot_silently_expand_budget_and_drops_empty_tail_segments():
    case = {
        "tx": np.zeros((130, 2)),
        "sbp": np.zeros((130, 2)),
        "y": np.zeros((130, 2)),
        "mask": np.ones(130, bool),
        "segments": [(0, 70), (70, 130)],
    }
    sliced = study.prefix(case, 70)
    assert sliced["segments"] == [(0, 70)]
    assert len(sliced["y"]) == 70
    with pytest.raises(ValueError, match="exceeds"):
        study.prefix(case, 131)
