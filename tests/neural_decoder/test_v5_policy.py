"""Lazy prefix boundaries, grouped evidence and failure dispositions."""

import copy

import numpy as np
import pytest

from neural_experiments.v5 import policy


def history(good=False):
    return {
        "current_session_id": "query",
        "current_post_implant_day": 10,
        "warmup_available": True,
        "prior_candidates": [
            {
                "candidate_id": candidate,
                "prior_available": True,
                "probes": [
                    {
                        "session_id": f"old{day}",
                        "post_implant_day": day,
                        "n": 100,
                        "mse": 0.3 if good else 0.6,
                    }
                    for day in (1, 2)
                ],
            }
            for candidate in policy.PRIORS
        ],
    }


def arrays(budget, good=True):
    stop = 1500 + budget * 50
    angle = (np.arange(stop) // 75 % 8 + 0.5) * np.pi / 4
    y = np.column_stack((np.cos(angle), np.sin(angle)))
    mask = np.ones(stop, bool)
    predictions = {key: np.zeros_like(y) for key in policy.CANDIDATES}
    if good:
        predictions[policy.CANDIDATES[0]] = y * 0.8
    return y, mask, predictions


def stage(budget, good=True, segments=None):
    y, mask, predictions = arrays(budget, good)
    stop, cut = len(y), 1500 + budget * 25
    return policy.stage_evidence(
        y, mask, segments or [(0, stop)], cut, cut + 100, stop, predictions
    )


def test_finite_grid_exact_four_policies():
    assert [
        (c["history_mean_mse_ceiling"], c["validation_margin"]) for c in policy.policy_grid()
    ] == [(0.4, 0.025), (0.4, 0.05), (0.45, 0.025), (0.45, 0.05)]
    assert [c["policy_id"] for c in policy.policy_grid()] == ["p0", "p1", "p2", "p3"]


def test_zero_label_acceptance_never_requests_any_current_labels():
    def forbidden(_):
        pytest.fail("Zero-label acceptance requested labels")

    result = policy.decide(history(True), forbidden, policy.policy_grid()[0])
    assert result["action"] == "accept"
    assert result["candidate_id"] == policy.PRIORS[0]
    assert result["budget_seconds"] == result["available_label_seconds"] == 0
    assert result["decision_time_seconds"] == 30


@pytest.mark.parametrize(
    "change", ["same_day", "self", "future_metric", "unsorted", "training_loss"]
)
def test_rejects_current_future_or_ineligible_reliability_evidence(change):
    value = history(True)
    probe = value["prior_candidates"][0]["probes"][0]
    if change == "same_day":
        probe["post_implant_day"] = 10
    if change == "self":
        probe["session_id"] = "query"
    if change == "future_metric":
        value["future_r2"] = 1.0
    if change == "unsorted":
        value["prior_candidates"][0]["probes"].reverse()
    if change == "training_loss":
        probe["training_mse"] = 0.0
    with pytest.raises(ValueError):
        policy.decide(value, lambda _: stage(30), policy.policy_grid()[0])


def test_single_history_probe_cannot_trigger_zero_label_acceptance():
    evidence = history(True)
    for candidate in evidence["prior_candidates"]:
        candidate["probes"] = candidate["probes"][:1]
    called = []
    result = policy.decide(
        evidence, lambda b: called.append(b) or stage(b), policy.policy_grid()[0]
    )
    assert called == [30]
    assert result["budget_seconds"] == 30


def test_lazy_acceptance_does_not_materialize_later_prefixes():
    def get_stage(budget):
        assert budget == 30
        return stage(budget)

    result = policy.decide(history(), get_stage, policy.policy_grid()[0])
    assert result["action"] == "accept"
    assert result["candidate_id"] == policy.CANDIDATES[0]
    assert result["requested_label_seconds"] == 30
    assert result["decision_time_seconds"] == 60


def test_low_directional_coverage_forces_further_labels_despite_low_loss():
    first = stage(30)
    first["training_sector_counts"] = [first["training_rows"]] + [0] * 7
    calls = []

    def get_stage(budget):
        calls.append(budget)
        return first if budget == 30 else stage(budget)

    result = policy.decide(history(), get_stage, policy.policy_grid()[0])
    assert calls == [30, 60]
    assert result["budget_seconds"] == 60
    assert result["events"][2]["geometry_passed"] is False


def test_cap_failure_abstains_instead_of_forcing_history_or_daily_fit():
    calls = []
    result = policy.decide(
        history(), lambda b: calls.append(b) or stage(b, False), policy.policy_grid()[0]
    )
    assert calls == [30, 60, 120, 240]
    assert result["action"] == "abstain" and result["candidate_id"] == "zero"
    assert result["budget_seconds"] == 240
    assert result["decision_time_seconds"] == 270


def test_short_checkpoint_retains_requested_and_observed_cost_separately():
    def get_stage(budget):
        return (
            {"budget_seconds": budget, "unavailable": "duration", "available_label_seconds": 83.9}
            if budget == 120
            else stage(budget, False)
        )

    result = policy.decide(history(), get_stage, policy.policy_grid()[0])
    assert result["action"] == "abstain"
    assert result["budget_seconds"] == result["requested_label_seconds"] == 120
    assert result["available_label_seconds"] == 83.9
    assert result["decision_time_seconds"] == 113.9


def test_stage_helper_never_accepts_future_labels_or_predictions():
    y, mask, prediction = arrays(60)
    with pytest.raises(ValueError, match="requested.*prefix"):
        policy.stage_evidence(y, mask, [(0, len(y))], 2250, 2350, 3000, prediction)
    y, mask, prediction = arrays(30)
    prediction[policy.CANDIDATES[0]] = np.zeros((3100, 2))
    with pytest.raises(ValueError, match="requested prefixes"):
        policy.stage_evidence(y, mask, [(0, len(y))], 2250, 2350, 3000, prediction)


def test_warmup_labels_do_not_affect_training_mean_or_validation_evidence():
    y, mask, prediction = arrays(30)
    expected = policy.stage_evidence(y, mask, [(0, len(y))], 2250, 2350, 3000, prediction)
    y[:1500] = [1.0, 0.0]
    actual = policy.stage_evidence(y, mask, [(0, len(y))], 2250, 2350, 3000, prediction)
    assert actual == expected


def test_groups_are_complete_anchored_and_do_not_cross_acquisition_boundaries():
    value = stage(30, segments=[(0, 2600), (2600, 3000)])
    assert [g["start"] for g in value["groups"]] == [2350, 2450, 2650, 2750, 2850]
    assert value["groups"][2]["n"] == 88  # New block burn2600:2662.
    assert value["validation_rows"] == 488
    assert sum(value["validation_sector_counts"]) == 488


def test_pooled_gain_alone_is_insufficient_without_group_agreement():
    value = stage(30)
    for group in value["groups"]:
        for key in policy.CANDIDATES:
            group["candidate_mse"][key] = 0.51
    # Two spectacular groups still give good pooled loss, but only2/6 groupwins.
    for group in value["groups"][:2]:
        group["candidate_mse"][policy.CANDIDATES[0]] = 0.0
    chosen, event = policy._stage_decision(value, policy.policy_grid()[0])
    assert chosen is None
    assert event["candidates"][0]["mse"] < 0.4


@pytest.mark.parametrize("field", ["held_out_r2", "future_scores", "training_mse"])
def test_decision_rejects_outcome_fields_outside_stage_contract(field):
    value = stage(30)
    value[field] = 1.0
    with pytest.raises(ValueError, match="future metrics forbidden"):
        policy.decide(history(), lambda _: value, policy.policy_grid()[0])


def test_rehashed_evidence_with_inconsistent_group_counts_is_rejected():
    value = copy.deepcopy(stage(30))
    value["groups"][0]["n"] -= 1
    with pytest.raises(ValueError, match="row count"):
        policy.decide(history(), lambda _: value, policy.policy_grid()[0])
