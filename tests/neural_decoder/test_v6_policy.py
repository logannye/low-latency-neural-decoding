"""Controller evidence has no future labels; source ranking excludes the target person."""

import copy

import numpy as np
import pytest

from neural_experiments.v6 import policy


def labels(n=4500):
    angle = np.arange(n) * 2 * np.pi / 100
    return np.column_stack((np.cos(angle), np.sin(angle))), np.ones(n, bool), [(0, n)]


def test_inner_choice_is_prefix_only_and_uses_fixed_tie_order():
    y, mask, segments = labels()
    predictions = {cid: y * 0.5 for cid in policy.CANDIDATES}
    result = policy.inner_selection(y, mask, segments, 3000, 3100, 4500, predictions)
    assert result["selected_candidate"] == policy.CANDIDATES[0]
    predictions[policy.CANDIDATES[7]] = y * 0.9
    assert (
        policy.inner_selection(y, mask, segments, 3000, 3100, 4500, predictions)[
            "selected_candidate"
        ]
        == policy.CANDIDATES[7]
    )


def test_inner_invalid_geometry_has_no_hindsight_fallback():
    y, mask, segments = labels()
    y[:] = (1, 0)
    result = policy.inner_selection(
        y, mask, segments, 3000, 3100, 4500, {cid: y.copy() for cid in policy.CANDIDATES}
    )
    assert not result["geometry_passed"]
    assert result["selected_candidate"] is None


def test_inner_drops_groups_crossing_acquisition_boundary_and_burns():
    y, mask, _ = labels()
    result = policy.inner_selection(
        y,
        mask,
        [(0, 3250), (3250, 4500)],
        3000,
        3100,
        4500,
        {cid: y.copy() for cid in policy.CANDIDATES},
    )
    assert result["validation_rows"] < 1400
    assert result["validation_groups"] < 14


def test_confirmation_accepts_only_single_frozen_prefix_prediction():
    y, mask, segments = labels()
    with pytest.raises(ValueError, match="prefix"):
        policy.confirmation_evidence(
            np.pad(y, ((0, 1), (0, 0))), np.pad(mask, (0, 1)), segments, 1500, 4500, y, np.zeros(2)
        )
    with pytest.raises(ValueError, match="prediction"):
        policy.confirmation_evidence(
            y, mask, segments, 1500, 4500, np.pad(y, ((0, 1), (0, 0))), np.zeros(2)
        )


def test_confirmation_halves_preserve_temporal_failure():
    y, mask, segments = labels()
    prediction = y.copy()
    prediction[3000:] *= -1
    evidence = policy.confirmation_evidence(y, mask, segments, 1500, 4500, prediction, np.zeros(2))
    assert evidence["first_half"]["mse"] == 0
    assert evidence["last_half"]["mse"] == pytest.approx(2)
    assert evidence["pooled"]["mse"] == pytest.approx(1)


def ranking_rows():
    result = []
    for person in ("a", "b", "c"):
        for i in range(3):
            for p in range(4):
                unavailable = i == 2
                r2 = [-0.01, 0.30, (-0.10 if i == 0 else 0.70), 0.28][p]
                result.append(
                    {
                        "participant": person,
                        "session_id": f"{person}-{i}",
                        "policy_id": f"p{p}",
                        "action": "abstain" if p == 0 else "accept",
                        "label_seconds": 420 if unavailable else [420, 300, 60, 60][p],
                        "daily240_r2": None if unavailable else 0.30,
                        "zero_mse": None if unavailable else 0.5,
                        "metrics": {
                            "r2": None if unavailable else r2,
                            "summed_mse": None if unavailable else 1 - r2,
                        },
                    }
                )
    return result


def configs():
    return [{"policy_id": f"p{i}"} for i in range(4)]


def test_source_ranking_avoids_all_abstain_and_minimizes_harm_before_cost():
    choices = policy.choose_policy(ranking_rows(), configs(), ["a", "b", "c"])
    assert all(c["policy_id"] == "p1" for c in choices)
    rows = choices[0]["ranking"]
    assert not rows[0]["feasible"]
    assert rows[1]["mean_label_seconds"] == 340
    assert rows[2]["mean_harmful_burden"] == 0.5
    assert rows[2]["mean_nonpositive_burden"] == 0.5


def test_entire_target_person_is_excluded_from_source_choice():
    rows = ranking_rows()
    before = policy.choose_policy(rows, configs(), ["a", "b", "c"])[0]
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["participant"] == "a":
            row["label_seconds"] = 420
            if row["metrics"]["r2"] is not None:
                row["metrics"].update(r2=-100, summed_mse=101)
    assert policy.choose_policy(changed, configs(), ["a", "b", "c"])[0] == before


def test_unavailable_case_cannot_be_removed_to_reduce_cost():
    rows = ranking_rows()
    rows = [
        r
        for r in rows
        if not (r["participant"] == "a" and r["session_id"] == "a-2" and r["policy_id"] == "p1")
    ]
    with pytest.raises(ValueError, match="identical"):
        policy.choose_policy(rows, configs(), ["a", "b", "c"])


def test_policy_cannot_change_scoring_availability_or_comparators():
    rows = ranking_rows()
    rows[0]["metrics"]["r2"] = None
    with pytest.raises(ValueError, match="availability"):
        policy.choose_policy(rows, configs(), ["a", "b", "c"])
    rows = ranking_rows()
    rows[0]["zero_mse"] = 1.0
    with pytest.raises(ValueError, match="comparators"):
        policy.choose_policy(rows, configs(), ["a", "b", "c"])


def test_no_feasible_policy_uses_locked_utility():
    rows = ranking_rows()
    for row in rows:
        row["daily240_r2"] = 0.9 if row["metrics"]["r2"] is not None else None
    choices = policy.choose_policy(rows, configs(), ["a", "b", "c"])
    assert all(c["selection_rule"] == "fallback_fixed_utility" for c in choices)
    for choice in choices:
        best = max(choice["ranking"], key=lambda r: r["objective"])
        assert choice["policy_id"] == best["policy_id"]


def history(curves=()):
    return {
        "current_session_id": "query",
        "current_post_implant_day": 100,
        "past_learning_curves": [
            {
                "session_id": f"past-{i}",
                "post_implant_day": i,
                "budgets": {
                    str(b): {
                        "r2": r,
                        "coordinate_mse": None if r is None else (1 - r) / 2,
                        "n": 1000,
                    }
                    for b, r in zip((120, 240, 360), values, strict=True)
                },
            }
            for i, values in enumerate(curves)
        ],
    }


def stage(budget, gain=0.1):
    stop = 4500 + budget * 50
    y, mask, segments = labels(stop)
    prediction = y * (1 - np.sqrt(1 - 2 * gain))
    return {
        "training_seconds": budget,
        "requested_label_seconds": budget + 60,
        "available_label_seconds": budget + 60,
        "decision_time_seconds": budget + 90,
        "candidate_id": "tx45_sbp_prior0" if budget == 0 else policy.FIXED_CANDIDATE,
        "model_sha256": f"{budget:064x}",
        "fit_stop_bin": 1500 + budget * 50,
        "confirmation_start_bin": 1600 + budget * 50,
        "confirmation_stop_bin": stop,
        "confirmation": policy.confirmation_evidence(
            y, mask, segments, 1600 + budget * 50, stop, prediction, np.zeros(2)
        ),
    }


def test_confirmation_reports_own_mean_r2_and_undefined_constant_axis_pair():
    y, mask, segments = labels()
    result = policy.confirmation_evidence(y, mask, segments, 1600, 4500, y * 0.5, np.zeros(2))
    assert result["pooled"]["r2"] == pytest.approx(0.75)
    y[:] = (1, 0)
    result = policy.confirmation_evidence(y, mask, segments, 1600, 4500, y.copy(), np.zeros(2))
    assert result["pooled"]["r2"] is None


def test_policy_grid_is_exact_four_locked_rules():
    assert [(c["gain_margin"], c["pooled_r2_min"]) for c in policy.policy_grid()] == [
        (0, 0),
        (0, 0.05),
        (0.025, 0),
        (0.025, 0.05),
    ]


def test_budget_floor_uses_paired_strict_past_and_requires_two_cases():
    assert policy.budget_floor(history([(-0.1, 0.1, 0.12)]))["training_seconds"] == 0
    assert policy.budget_floor(history([(-0.1, 0.1, 0.12)] * 2))["training_seconds"] == 240
    assert policy.budget_floor(history([(-0.1, 0.1, 0.2)] * 3))["training_seconds"] == 360
    bad = history([(-0.1, 0.1, 0.2)] * 2)
    bad["past_learning_curves"][0]["post_implant_day"] = 100
    with pytest.raises(ValueError, match="strictly earlier"):
        policy.budget_floor(bad)
    bad = history([(-0.1, 0.1, 0.2)] * 2)
    bad["past_learning_curves"][0]["session_id"] = "query"
    with pytest.raises(ValueError, match="strictly earlier"):
        policy.budget_floor(bad)


def test_budget_floor_requires_weak_early_accuracy_and_positive_late_accuracy():
    assert policy.budget_floor(history([(0.1, 0.3, 0.4)] * 3))["training_seconds"] == 0
    assert policy.budget_floor(history([(-0.3, -0.2, -0.1)] * 3))["training_seconds"] == 0
    assert policy.budget_floor(history([(-0.1, -0.2, 0.1)] * 3))["training_seconds"] == 360
    # Missing old cases stay in inventory; no older replacement is sought.
    assert (
        policy.budget_floor(history([(-0.1, 0.2, 0.3), (None, None, None)]))["training_seconds"]
        == 0
    )


def test_budget_floor_requires_two_thirds_positive_gains_and_matched_availability():
    curves = [(-0.1, 0.3, 0.3), (0.1, 0, 0), (0.1, 0, 0)]
    assert policy.budget_floor(history(curves))["training_seconds"] == 0
    bad = history([(-0.1, 0.1, 0.2)] * 2)
    bad["past_learning_curves"][0]["budgets"]["240"]["r2"] = None
    with pytest.raises(ValueError, match="availability"):
        policy.budget_floor(bad)


def test_lazy_acceptance_never_requests_later_labels():
    visited = []

    def callback(b):
        visited.append(b)
        if b > 0:
            pytest.fail("Accepted prior must not request later labels")
        return stage(b)

    result = policy.decide(history(), callback, policy.policy_grid()[3])
    assert visited == [0]
    assert result["action"] == "accept"
    assert result["requested_label_seconds"] == 60
    assert result["model_sha256"] == stage(0)["model_sha256"]


def test_budget_floor_applies_before_prior_confirmation():
    visited = []

    def callback(b):
        visited.append(b)
        return stage(b)

    result = policy.decide(history([(-0.1, 0.1, 0.12)] * 2), callback, policy.policy_grid()[0])
    assert visited == [240]
    assert result["training_seconds"] == 240
    assert result["requested_label_seconds"] == 300


def test_zero_gain_and_zero_r2_are_strict_rejections():
    result = policy.decide(history(), lambda b: stage(b, 0), policy.policy_grid()[0])
    assert result["action"] == "abstain"
    assert len(result["events"]) == 5
    assert result["requested_label_seconds"] == 420


def test_temporal_half_or_coverage_failure_cannot_be_hidden_by_pooled_accuracy():
    for bad_key in ("gain_vs_zero", "gain_vs_training_mean", "direction_sector_counts"):

        def callback(b):
            value = stage(b)
            half = value["confirmation"]["last_half"]
            half[bad_key] = (
                [half["n"], 0, 0, 0, 0, 0, 0, 0] if bad_key == "direction_sector_counts" else -0.01
            )
            return value

        assert policy.decide(history(), callback, policy.policy_grid()[0])["action"] == "abstain"


def test_short_checkpoint_abstains_with_actual_and_requested_costs():
    value = {
        "training_seconds": 240,
        "requested_label_seconds": 300,
        "available_label_seconds": 235.36,
        "decision_time_seconds": 265.36,
        "unavailable": "complete_fit_and_confirmation_not_observed",
    }
    result = policy.decide(
        history([(-0.1, 0.1, 0.12)] * 2), lambda b: value, policy.policy_grid()[0]
    )
    assert result["action"] == "abstain"
    assert result["requested_label_seconds"] == 300
    assert result["available_label_seconds"] == 235.36
    assert result["model_sha256"] is None


def test_decision_rejects_changed_candidate_geometry_or_artifact_identity():
    for key, replacement in (
        ("candidate_id", "hindsight_winner"),
        ("confirmation_start_bin", 1500),
        ("model_sha256", "unbound"),
    ):
        value = stage(0)
        value[key] = replacement
        with pytest.raises(ValueError, match="frozen artifact"):
            policy.decide(history(), lambda b: value, policy.policy_grid()[0])
