"""Synthetic adversarial tests for the separate V5 arithmetic reference."""

import copy
import inspect

import numpy as np
import pytest

from neural_experiments.v5 import audit, policy


def prefix(budget=30):
    n = 1500 + budget * 50
    angle = np.arange(n) * 2 * np.pi / 100
    y = np.column_stack((np.cos(angle), np.sin(angle)))
    mask = np.ones(n, bool)
    mask[:1500] = False
    segments = [{"start": 0, "end": n}]
    predictions = {c["candidate_id"]: y * 0.75 for c in audit.grids()[0]}
    return y, mask, segments, predictions


def history():
    return {
        "current_session_id": "current",
        "current_post_implant_day": 9,
        "warmup_available": True,
        "prior_candidates": [
            {"candidate_id": f + "_prior10", "prior_available": False, "probes": []}
            for f in audit.FAMILIES
        ],
    }


def test_audit_does_not_import_v5_producer_modules():
    source = inspect.getsource(audit)
    assert "from neural_experiments.v5" not in source
    assert "import neural_experiments.v5" not in source


@pytest.mark.parametrize("budget", audit.BUDGETS)
def test_independent_group_evidence_matches_producer(budget):
    y, mask, segments, predictions = prefix(budget)
    # Make a boundary cut through a group, so that the group and burn must be omitted.
    boundary = 1500 + budget * 25 + 220
    segments = [{"start": 0, "end": boundary}, {"start": boundary, "end": len(y)}]
    reference = audit.stage_reference(y, mask, segments, budget, predictions)
    actual = policy.stage_evidence(
        y, mask, segments, 1500 + budget * 25, 1600 + budget * 25, len(y), predictions
    )
    audit.compare_structure(actual, reference)
    assert all(not (g["start"] < boundary < g["stop"]) for g in reference["groups"])


def test_future_label_or_prediction_arrays_rejected():
    y, mask, segments, predictions = prefix()
    with pytest.raises(ValueError, match="prefix"):
        audit.stage_reference(
            np.pad(y, ((0, 1), (0, 0))),
            np.pad(mask, (0, 1)),
            [{"start": 0, "end": len(y) + 1}],
            30,
            predictions,
        )
    predictions[next(iter(predictions))] = np.pad(y, ((0, 1), (0, 0)))
    with pytest.raises(ValueError, match="prediction"):
        audit.stage_reference(y, mask, segments, 30, predictions)


def test_warmup_labels_cannot_change_stage_validation():
    y, mask, segments, predictions = prefix()
    before = audit.stage_reference(y, mask, segments, 30, predictions)
    y[:1500] = 999
    after = audit.stage_reference(y, mask, segments, 30, predictions)
    audit.compare_structure(before, after)


def test_float32_negative_axis_uses_broker_float64_sector_arithmetic():
    y = np.array([[-1, -1e-8], [-1, 0], [-1, 1e-8]], dtype=np.float32)
    mask = np.ones(3, bool)
    assert audit.sectors(y, mask) == audit.sectors(y.astype(np.float64), mask)
    assert audit.sectors(y, mask) == [2, 0, 0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize("config", audit.grids()[1])
def test_decision_separate_reference_has_same_early_stop(config):
    y, mask, segments, predictions = prefix()
    stage = audit.stage_reference(y, mask, segments, 30, predictions)
    expected, requested = audit.decision_reference(history(), {30: stage}, config)
    calls = []

    def get_stage(budget):
        calls.append(budget)
        assert budget == 30
        return stage

    actual = policy.decide(history(), get_stage, config)
    audit.compare_structure(actual, expected)
    assert calls == requested == [30]


def test_history_acceptance_never_needs_stage_evidence():
    h = history()
    h["prior_candidates"][0].update(
        prior_available=True,
        probes=[
            {"session_id": f"p{i}", "post_implant_day": i, "n": 100, "mse": 0.2} for i in (1, 2)
        ],
    )
    expected, requests = audit.decision_reference(h, {}, audit.grids()[1][0])
    assert requests == []
    assert expected["budget_seconds"] == 0
    assert expected["candidate_id"] == "tx45_prior10"


def test_short_checkpoint_records_requested_and_observed_cost():
    stage = {
        "budget_seconds": 30,
        "unavailable": "requested_checkpoint_not_observed",
        "available_label_seconds": 12.8,
    }
    result, requests = audit.decision_reference(history(), {30: stage}, audit.grids()[1][0])
    assert requests == [30]
    assert result["budget_seconds"] == 30
    assert result["available_label_seconds"] == 12.8
    assert result["decision_time_seconds"] == 42.8
    assert result["action"] == "abstain"


def source_rows():
    return [
        {
            "participant": p,
            "session_id": f"{p}-{i}",
            "policy_id": f"p{k}",
            "label_seconds": 240 if i else 30 * k,
            "metrics": {"r2": None if i else 0.3 - 0.005 * k},
            "daily120_r2": 0.3,
        }
        for p in ("a", "b", "c")
        for i in range(2)
        for k in range(4)
    ]


def test_policy_ranking_excludes_entire_target_and_keeps_unavailable_cost():
    rows = source_rows()
    before = audit.selection_reference(rows, ["a", "b", "c"])
    assert before[0]["ranking"][0]["mean_label_seconds"] == 120
    altered = copy.deepcopy(rows)
    for row in altered:
        if row["participant"] == "a":
            row["metrics"]["r2"] = -100
            row["daily120_r2"] = 20
            row["label_seconds"] = 999
    after = audit.selection_reference(altered, ["a", "b", "c"])
    assert before[0] == after[0]
    assert before[1] != after[1]


def test_duplicate_or_missing_artifact_rejected():
    with pytest.raises(ValueError, match="inventory"):
        audit.unique([{"id": 1}, {"id": 1}], lambda r: r["id"])
    with pytest.raises(ValueError, match="inventory"):
        audit.unique([{"id": 1}], lambda r: r["id"], [1, 2])


def test_fit_mask_excludes_warmup_and_reburns_every_block():
    mask = audit.label_mask(np.ones(4000, bool), [(0, 2000), (2000, 4000)], 3000)
    assert len(mask) == 3000
    assert not mask[:1500].any()
    assert not mask[2000:2062].any()
    assert mask[1500:2000].all() and mask[2062:].all()
