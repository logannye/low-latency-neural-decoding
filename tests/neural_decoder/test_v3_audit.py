"""Independent arithmetic attacks for prospective V3 verification."""

import copy
import json

import numpy as np
import pytest

from neural_experiments.v3 import audit, model


def synthetic():
    rng = np.random.default_rng(17)
    tx = rng.poisson(2, (320, 4)).astype(float)
    sbp = rng.uniform(0, 20, tx.shape)
    angle = np.linspace(0, 7, len(tx))
    y = np.column_stack((np.cos(angle), np.sin(angle)))
    mask = np.arange(len(tx)) % 7 != 0
    return tx, sbp, y, mask, [(0, 160), (160, 320)]


@pytest.mark.parametrize("family", ["tx", "tx_sbp"])
@pytest.mark.parametrize("baseline", [None, 30.0])
def test_independent_recurrence_matches_causal_production_and_resets(family, baseline):
    tx, sbp, _, _, segments = synthetic()
    config = {
        "kind": "regularized",
        "family": family,
        "tau_seconds": 0.2,
        "baseline_tau_seconds": baseline,
        "ridge_lambda": 1.0,
    }
    expected = audit.reference_representation(tx, sbp, segments, config)
    np.testing.assert_allclose(
        model.representation(tx, sbp, segments, config), expected, rtol=1e-12, atol=1e-12
    )
    changed = tx.copy()
    changed[150:] += 100
    observed = audit.reference_representation(changed, sbp, segments, config)
    np.testing.assert_array_equal(observed[:150], expected[:150])
    changed = tx.copy()
    changed[:160] += 100
    observed = audit.reference_representation(changed, sbp, segments, config)
    np.testing.assert_array_equal(observed[160:], expected[160:])


@pytest.mark.parametrize("kind", ["regularized", "v2_ridge"])
def test_independent_optimality_accepts_exact_prefix_and_rejects_changed_weights_or_labels(kind):
    tx, sbp, y, mask, segments = synthetic()
    config = {
        "kind": kind,
        "family": "tx",
        "tau_seconds": 0.2,
        "baseline_tau_seconds": None,
        "ridge_lambda": 1.0,
    }
    if kind == "v2_ridge":
        config = {"kind": "v2_ridge", "family": "tx"}
    fitted, _ = model.fit(tx, sbp, y, mask, segments, config)
    arrays = {
        "format_version": np.asarray(3),
        "config": np.asarray(json.dumps(config)),
        "mean": fitted.mean,
        "scale": fitted.scale,
        "coefficients": fitted.coefficients,
        "intercept": fitted.intercept,
    }
    result = audit.validate_fitted_model(arrays, tx, sbp, y, mask, segments, config)
    assert result["normal_equation_relative_residual"] < 1e-10
    changed = copy.deepcopy(arrays)
    changed["coefficients"][0, 0] += 0.1
    with pytest.raises(ValueError, match="ridge optimum"):
        audit.validate_fitted_model(changed, tx, sbp, y, mask, segments, config)
    shifted = np.roll(y, 20, axis=0)
    with pytest.raises(ValueError, match="ridge optimum"):
        audit.validate_fitted_model(arrays, tx, sbp, shifted, mask, segments, config)
    changed = copy.deepcopy(arrays)
    changed["mean"][0] += 1
    with pytest.raises(ValueError, match="normalization"):
        audit.validate_fitted_model(changed, tx, sbp, y, mask, segments, config)


def test_training_burn_never_stitches_mask_intervals():
    mask = np.ones(140, bool)
    mask[65:70] = False
    actual = audit.training_mask(mask, [(0, 70), (70, 70), (70, 140)])
    assert np.flatnonzero(actual).tolist() == [62, 63, 64, 132, 133, 134, 135, 136, 137, 138, 139]


def test_grid_has_exact_36_distinct_candidates_in_plan_order():
    plan = audit.read(audit.ROOT / "neural_experiments/v3/plan.json")
    configs = audit.grid(plan)
    assert len(configs) == len({json.dumps(c, sort_keys=True) for c in configs}) == 36
    assert configs[0] == {
        "kind": "regularized",
        "family": "tx",
        "tau_seconds": 0.1,
        "baseline_tau_seconds": None,
        "ridge_lambda": 0.1,
    }


@pytest.mark.parametrize("kind", ["v2_ridge", "regularized"])
def test_saved_model_replay_uses_frozen_prefix_stats_and_preserves_causality(kind):
    tx, sbp, y, mask, segments = synthetic()
    config = {"kind": "v2_ridge", "family": "tx"}
    if kind == "regularized":
        config = {
            "kind": kind,
            "family": "tx_sbp",
            "tau_seconds": 0.2,
            "baseline_tau_seconds": 30.0,
            "ridge_lambda": 0.1,
        }
    fitted, _ = model.fit(tx[:160], sbp[:160], y[:160], mask[:160], [(0, 160)], config)
    arrays = {
        "mean": fitted.mean,
        "scale": fitted.scale,
        "coefficients": fitted.coefficients,
        "intercept": fitted.intercept,
    }

    def reference(values):
        representation = (
            audit.transformed(values, sbp, "tx")
            if kind == "v2_ridge"
            else audit.reference_representation(values, sbp, segments, config)
        )
        return audit.replay_prediction(arrays, representation, segments, config)

    prediction = reference(tx)
    np.testing.assert_allclose(prediction, fitted.predict(tx, sbp, segments), atol=1e-10)
    changed = tx.copy()
    changed[250:] += 1000
    np.testing.assert_array_equal(reference(changed)[:250], prediction[:250])
    changed[:160] += 1000
    np.testing.assert_array_equal(reference(changed)[160:250], prediction[160:250])


@pytest.mark.parametrize("n", [0, 1, 49, 50])
def test_independent_score_minimum_count_and_zero_prediction_semantics(n):
    from neural_experiments.v3.study import score

    angles = np.linspace(0, 4, 100)
    y = np.column_stack((np.cos(angles), np.sin(angles)))
    mask = np.arange(len(y)) < n
    result, detailed = audit.score_reference(y, np.zeros_like(y), mask)
    audit.compare_structure(result, score(y, np.zeros_like(y), mask), "score")
    assert result == detailed
    assert result["zero_prediction_count"] == n
    assert (result["r2"] is None) == (n < 50)
    assert result["angular_error_degrees"] == (90.0 if n >= 50 else None)


def test_whole_person_excluded_choices_do_not_use_their_target_scores():
    plan = audit.read(audit.ROOT / "neural_experiments/v3/plan.json")
    candidates = [
        {"candidate_id": f"c{i:02d}", "config": c} for i, c in enumerate(audit.grid(plan))
    ]
    cases = [
        {
            "participant": p,
            "budget_seconds": b,
            "candidate_id": c["candidate_id"],
            "metrics": {"r2": 0.2},
        }
        for p in plan["participants"]
        for b in plan["calibration_budgets_seconds"]
        for c in candidates
    ]
    original = audit.reconstruct_choices(cases, candidates, plan)
    for row in cases:
        if row["participant"] == "T2" and row["candidate_id"] == "c35":
            row["metrics"]["r2"] = 1e9
    changed = audit.reconstruct_choices(cases, candidates, plan)
    for before, after in zip(original, changed, strict=True):
        assert before["target_participant"] not in before["source_participants"]
        if before["target_participant"] == "T2":
            assert before == after
            assert before["candidate_id"] in ("c00", "c18")
    assert any(before != after for before, after in zip(original, changed, strict=True))


def test_person_aggregation_requires_all_nine_and_matches_runner_bootstrap():
    from neural_experiments.v3.study import summarize

    plan = audit.read(audit.ROOT / "neural_experiments/v3/plan.json")
    models = []
    for i, person in enumerate(plan["participants"]):
        # Unequal session counts must not give T2 extra cohort weight.
        for _ in range(3 if i == 0 else 1):
            for method in ("selected_daily", "zero", "prefix_mean", "v2_ridge_daily"):
                value = (0.2 if i else 0.8) if method == "selected_daily" else -0.1
                models.append(
                    {
                        "target_participant": person,
                        "method": method,
                        "budget_seconds": 120,
                        "traces": [
                            {
                                "partition": "primary",
                                "metrics": {"r2": value, "angular_error_degrees": 60.0},
                            }
                        ],
                    }
                )
    result = audit.reconstruct_summary(models, plan)
    audit.compare_structure(result, summarize(models, plan), "summary")
    primary = next(g for g in result["groups"] if g["method"] == "selected_daily")
    assert primary["mean_r2"] == pytest.approx((0.8 + 8 * 0.2) / 9)
    assert result["operational_gate"]["passed"]
    incomplete = audit.reconstruct_summary(
        [m for m in models if m["target_participant"] != "T6"], plan
    )
    assert not incomplete["operational_gate"]["passed"]
    assert not incomplete["operational_gate"]["criteria"]["nine_eligible_people"]


def test_changed_hash_or_escaping_path_cannot_pass_identity_check(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}")
    ref = {"path": path.name, "sha256": audit.sha256(path)}
    assert audit.checked_reference(tmp_path, ref) == path
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.checked_reference(tmp_path, ref)
    for value in ("../artifact.json", str(path)):
        with pytest.raises(ValueError, match="relative"):
            audit.contained(tmp_path, value)
