"""Independent references reject leakage, chronology changes and artifact tampering."""

import json

import numpy as np
import pytest

from neural_experiments.v4 import audit, model


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(19)
    features = {
        name: rng.uniform(0, 5, (400, 4)).astype(np.float32)
        for name in ("x_tx3", "x_tx45", "x_sbp")
    }
    labels = rng.normal(size=(400, 2))
    labels /= np.linalg.norm(labels, axis=1)[:, None]
    mask = np.ones(400, dtype=bool)
    mask[::7] = False
    return features, labels, mask, [(0, 240), (240, 400)]


def config(family="tx45", baseline=None, gamma=1.0):
    return {
        "family": family,
        "tau_seconds": 0.2 if family.endswith("sbp") else 0.4,
        "baseline_tau_seconds": baseline,
        "adapt_gamma": gamma,
    }


@pytest.mark.parametrize("family", ["tx3", "tx45", "tx3_sbp", "tx45_sbp"])
@pytest.mark.parametrize("baseline", [None, 5.0])
def test_direct_recurrence_matches_production_and_preserves_causality(synthetic, family, baseline):
    features, _, _, segments = synthetic
    cfg = config(family, baseline)
    expected = audit.reference_representation(features, segments, cfg)
    np.testing.assert_allclose(model.representation(features, segments, cfg), expected, atol=1e-12)
    altered = {k: v.copy() for k, v in features.items()}
    for v in altered.values():
        v[180:] *= 20
    changed = audit.reference_representation(altered, segments, cfg)
    np.testing.assert_array_equal(changed[:180], expected[:180])
    if baseline is not None:
        np.testing.assert_array_equal(expected[[0, 240]], 0)


def test_mask_uses_observed_global_stride_after_block_burn():
    mask = np.ones(180, dtype=bool)
    mask[70:82] = False
    actual = audit.reference_mask(mask, [(0, 73), (73, 180)], stride=5)
    assert np.flatnonzero(actual).tolist() == [65, 135, 140, 145, 150, 155, 160, 165, 170, 175]
    assert not audit.reference_mask(mask, [(0, 73), (73, 180)], 62).any()


def test_equal_session_history_does_not_pool_unequal_counts():
    a = {"gram": np.eye(3), "rhs": np.ones((3, 2)), "n": 50}
    b = {"gram": 3 * np.eye(3), "rhs": 5 * np.ones((3, 2)), "n": 5000}
    b["gram"][-1, -1] = 1
    got = audit.reference_history([a, b], 3)
    np.testing.assert_allclose(got, [[1, 1], [1, 1], [3, 3]])
    np.testing.assert_allclose(model.solve_history([a, b], 3), got)
    with pytest.raises(ValueError, match="Insufficient history"):
        audit.reference_history([{**a, "n": 49}], 3)


@pytest.mark.parametrize("gamma", [1.0, 10.0])
def test_adaptation_optimum_and_low_label_fallback(synthetic, gamma):
    features, y, mask, segments = synthetic
    raw = audit.reference_representation(features, segments, config())
    z, _, _ = audit.reference_design(raw, 150, 4)
    training = audit.reference_mask(mask, segments, 150)
    stats = audit.reference_statistics(z[:150], y[:150], training)
    prior = np.full((5, 2), 0.03)
    weights = model.adapt(stats, prior, gamma)
    assert (
        audit.validate_weights(weights, stats, prior, gamma)["normal_equation_relative_residual"]
        < 1e-12
    )
    wrong = weights.copy()
    wrong[-1, 0] += 0.02
    with pytest.raises(ValueError, match="optimum"):
        audit.validate_weights(wrong, stats, prior, gamma)
    for count in (0, 49):
        low = {**stats, "n": count}
        assert audit.validate_weights(prior.copy(), low, prior, gamma)["fallback"]
        with pytest.raises(ValueError, match="fallback"):
            audit.validate_weights(np.zeros_like(prior), low, prior, gamma)


def test_post_prefix_labels_and_mask_cannot_change_statistics(synthetic):
    features, y, mask, segments = synthetic
    raw = audit.reference_representation(features, segments, config())
    z, _, _ = audit.reference_design(raw, 150, 4)
    before = audit.reference_statistics(z[:150], y[:150], audit.reference_mask(mask, segments, 150))
    y[150:] = np.nan
    mask[150:] = False
    after = audit.reference_statistics(z[:150], y[:150], audit.reference_mask(mask, segments, 150))
    np.testing.assert_array_equal(before["rhs"], after["rhs"])


@pytest.mark.parametrize("bias", [0, 1])
def test_daily_fit_export_and_full_replay(synthetic, tmp_path, bias):
    features, y, mask, segments = synthetic
    cfg = config("tx3_sbp", 5.0)
    raw = audit.reference_representation(features, segments, cfg)
    z, mean, scale = audit.reference_design(raw, 150, 4)
    stats = audit.reference_statistics(z[:150], y[:150], audit.reference_mask(mask, segments, 150))
    weights = model.solve_daily(stats, 10.0, bias)
    audit.validate_weights(weights, stats, daily_penalty=10.0, free_bias=bool(bias))
    fitted = model.Decoder(cfg, mean, scale, weights)
    path = tmp_path / "model.npz"
    fitted.save(path)
    arrays = audit.numeric(path)
    audit.validate_normalization(arrays, raw, 150, 4, cfg)
    np.testing.assert_allclose(
        audit.replay(arrays, raw), fitted.predict(features, segments), atol=1e-12
    )
    with pytest.raises(ValueError, match="normalization"):
        audit.validate_normalization(arrays, raw, 151, 4, cfg)
    arrays["mean"][0] += 0.1
    with pytest.raises(ValueError, match="normalization"):
        audit.validate_normalization(arrays, raw, 150, 4, cfg)


def test_history_selection_cannot_replace_invalid_or_include_query():
    items = [
        {"participant_id": "T3", "session_id": str(i), "post_implant_day": i} for i in range(6)
    ]
    layouts = {str(i): (1, 2) for i in range(6)}
    layouts["4"] = (4, 5)
    stats = {str(i): {"gram": np.eye(2), "rhs": np.ones((2, 2)), "n": 100} for i in range(6)}
    stats["2"]["n"] = 0
    _, receipt = audit.checked_prior(items[-1], items, layouts, stats, 2)
    assert [r["session_id"] for r in receipt] == ["1", "2", "3"]
    assert receipt[1]["disposition"] == "insufficient_labels"
    assert all(r["post_implant_day"] < 5 for r in receipt)


def test_selection_excludes_whole_person_and_targeted_query_future():
    grid = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    rows = [
        {
            "participant": person,
            "session_id": sid,
            "post_implant_day": day,
            "candidate_id": candidate,
            "kind": kind,
            "metrics": {"r2": value},
        }
        for kind in ("global", "targeted")
        for person, sid, day, values in [
            ("T2", "other", 0, [0.0, 1.0]),
            ("T3", "early", 1, [2.0, 0.0]),
            ("T3", "later", 2, [-100.0, 100.0]),
        ]
        for candidate, value in zip(["a", "b"], values, strict=True)
    ]
    items = [
        {"participant_id": "T3", "session_id": sid, "post_implant_day": day}
        for sid, day in [("early", 1), ("later", 2)]
    ]
    choices, targeted = audit.reconstruct_choices(rows, grid, grid, items, ["T2", "T3"])
    assert choices[1]["candidate_id"] == "b"
    assert targeted[0]["selection_scope"] == "other_people"
    assert targeted[0]["candidate_id"] == "b"
    assert targeted[1]["source_session_ids"] == ["early"]
    assert targeted[1]["candidate_id"] == "a"


def test_locked_grids_have_exact_order_and_count():
    plan = json.loads((audit.ROOT / "neural_experiments/v4/plan.json").read_text())
    global_grid, targeted = audit.candidate_grids(plan)
    assert len(global_grid) == len(targeted) == 16
    assert global_grid[0]["config"] == config("tx45", None, 1.0)
    assert global_grid[-1]["config"] == config("tx3_sbp", 5.0, 10.0)
    assert targeted[0]["bias"] == 0
    assert targeted[-1]["bias"] == 1
