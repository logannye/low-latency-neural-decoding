"""Synthetic independent numerical references, information boundaries and tampering."""

import ast
import copy
import inspect

import numpy as np
import pytest

from neural_experiments.v8 import audit, development, metrics, model


def fixture(n=9500):
    rng = np.random.default_rng(443)
    t = np.arange(n)
    trial = t // 173
    start = trial * 173
    a = trial * 1.731 + 0.2 * np.sin(t * 0.07)
    y = np.c_[np.cos(a), np.sin(a)].astype(np.float32)
    return {
        "item": {
            "participant_id": "T2",
            "session_id": "synthetic",
            "post_implant_day": 1,
            "n_bins": n,
            "n_channels": 2,
            "far_distance": 0.2,
        },
        "features": {
            "x_tx3": rng.poisson(1, (n, 2)).astype(np.float32),
            "x_tx45": rng.poisson(0.3, (n, 2)).astype(np.float32),
            "x_sbp": rng.gamma(2, 1, (n, 2)).astype(np.float32),
        },
        "segments": [(0, n)],
        "y": y,
        "eval_mask": np.ones(n, bool),
        "context": {
            "trial_id": trial,
            "trial_age_seconds": (t - start) * 0.02 + 0.01,
            "trial_start_bin": start,
            "trial_stop_bin": np.minimum(start + 173, n),
            "trial_onset_seconds": start * 0.02,
            "trial_right_censored": start + 173 > n,
            "reference40_distance": np.ones(n),
            "reference40_available": np.ones(n, bool),
            "reference40_direction": np.c_[np.cos(trial * 1.731), np.sin(trial * 1.731)],
        },
    }


def exported(tmp_path, monkeypatch, n=9500):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    case = fixture(n)
    chosen = {**audit.BASELINE, "family": "tx3_sbp", "taus_seconds": [0.05, 0.5], "ridge": 10.0}
    factorials = development.factorial_configs(chosen)
    configs = development.assessment_configs(factorials, "selected_clean_equal_trial")
    result = model.evaluate_session(case, configs, artifact_dir=tmp_path / "artifacts")
    return case, configs, result


def test_no_producer_imports():
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("neural_experiments.v8")
        if isinstance(node, ast.Import):
            assert not any(n.name.startswith("neural_experiments.v8") for n in node.names)


@pytest.mark.parametrize("n", [1000, 5000, 7499, 7500, 8000, 9500])
def test_actual_producer_all_twelve_exports_independently_verified(tmp_path, monkeypatch, n):
    case, configs, result = exported(tmp_path, monkeypatch, n)
    rebuilt, errors, residuals = audit.check_session(case, configs, result["rows"], result)
    assert len(rebuilt) == 12
    assert max(errors) < 1e-10 and max(residuals) < 1e-10
    if n < 7500:
        assert all(r["training"]["fallback_zero"] for r in rebuilt)
    if n < 7600:
        assert all(r["metrics"]["r2"] is None for r in rebuilt)


def test_independent_future_and_unpurchased_label_isolation():
    case = fixture()
    config = {**audit.BASELINE, "phase": "clean", "weighting": "equal_trial"}
    raw = audit.representation_reference(case, config)
    before = audit.fit_reference(case, config, raw)
    other = copy.deepcopy(case)
    other["y"][~before["mask"]] = np.nan
    other["context"]["reference40_direction"][~before["mask"]] = np.nan
    raw[7500:] = np.nan
    after = audit.fit_reference(other, config, raw)
    for key in ("mean", "scale", "gram", "rhs", "sample_weights"):
        np.testing.assert_array_equal(before[key], after[key])
    assert before["receipt"] == after["receipt"]


def test_causal_reference_future_features_and_block_resets():
    case = fixture()
    case["segments"] = [(0, 5000), (5000, 9500)]
    config = {**audit.BASELINE, "family": "tx3_sbp", "taus_seconds": [0.05, 0.5]}
    raw = audit.representation_reference(case, config)
    other = copy.deepcopy(case)
    for key in other["features"]:
        other["features"][key][7500:] += 500
    changed = audit.representation_reference(other, config)
    np.testing.assert_array_equal(raw[:7500], changed[:7500])
    value = np.r_[
        np.sqrt(case["features"]["x_tx3"][5000].astype(float)),
        np.log1p(case["features"]["x_sbp"][5000].astype(float)),
    ]
    np.testing.assert_allclose(raw[5000], np.tile(value, 2), atol=1e-12)


def test_partial_paid_trial_cannot_score_and_right_censor_remains():
    case = fixture()
    _, primary, secondary, purchased = audit.masks_reference(case)
    assert not np.isin(case["context"]["trial_id"][primary], purchased).any()
    assert primary[-1] and case["context"]["trial_right_censored"][-1]
    assert np.all(primary[secondary])
    c = case["context"]
    c["trial_start_bin"][1510:1540] = 1500
    c["trial_onset_seconds"][1510:1540] = 29.99
    c["trial_age_seconds"][1510:1540] = 0.3
    assert not audit.masks_reference(case, {**audit.BASELINE, "phase": "clean"})[0][1510:1540].any()


@pytest.mark.parametrize("field", ["mean", "scale", "weights", "version"])
def test_rehashed_model_tampering_is_rejected(tmp_path, monkeypatch, field):
    case, configs, result = exported(tmp_path, monkeypatch)
    row = next(r for r in result["rows"] if r["method"] == "baseline")
    path = row["model"]["model_path"]
    bundle = audit.numeric(path)
    if field == "version":
        bundle[field] = bundle[field].astype(float)
    else:
        bundle[field].flat[0] += 0.01
    np.savez(path, **bundle)
    row["model"]["model_sha256"] = audit.sha256(path)
    with pytest.raises(ValueError, match="version|normalization|optimum"):
        audit.check_session(case, configs, result["rows"], result)


@pytest.mark.parametrize(
    "field",
    [
        "baseline__prediction",
        "primary_mask",
        "baseline__training_mask",
        "baseline__training_weights",
    ],
)
def test_rehashed_trace_tampering_is_rejected(tmp_path, monkeypatch, field):
    case, configs, result = exported(tmp_path, monkeypatch)
    path = result["rows"][0]["trace"]["trace_path"]
    bundle = audit.numeric(path)
    if bundle[field].dtype == bool:
        bundle[field].flat[-1] = not bundle[field].flat[-1]
    else:
        bundle[field].flat[-1] += 0.1
    np.savez(path, **bundle)
    for r in result["rows"]:
        r["trace"]["trace_sha256"] = audit.sha256(path)
    with pytest.raises(ValueError, match="replay|mask|weights"):
        audit.check_session(case, configs, result["rows"], result)


def test_trial_shuffle_is_chronological_rotation_not_bin_shuffle():
    y = np.arange(18, dtype=float).reshape(9, 2)
    ids = np.array([7] * 2 + [3] * 4 + [9] * 3)
    out, mapping = audit.permutation_reference(y, ids)
    np.testing.assert_array_equal(out[:2], y[[2, 4]])
    assert [(m["recipient_trial"], m["donor_trial"]) for m in mapping] == [(7, 3), (3, 9), (9, 7)]


def test_primary_and_alternative_target_cannot_be_substituted(tmp_path, monkeypatch):
    case, configs, result = exported(tmp_path, monkeypatch)
    row = next(r for r in result["rows"] if r["method"] == "primary_onset")
    row["metrics"] = row["secondary"]["onset"]
    with pytest.raises(ValueError, match="scores"):
        audit.check_session(case, configs, result["rows"], result)


def fake_rows():
    configs = audit.configurations_reference(
        audit.factorial_reference(audit.BASELINE), "selected_clean_equal_trial"
    )
    rows = []
    for person in ("T2", "T5"):
        for i in range(3):
            for k, method in enumerate(configs):
                value = 0.3 if method == "selected_clean_equal_trial" else 0.1 + k * 0.001
                rows.append(
                    {
                        "session_id": f"{person}-{i}",
                        "participant": person,
                        "post_implant_day": i,
                        "method": method,
                        "score_mask_sha256": "same",
                        "metrics": {"r2": value, "summed_mse": 1 - value},
                    }
                )
    return configs, rows


def test_selection_ties_and_aggregate_gate_match_independent_reference():
    configs, rows = fake_rows()
    actual = development.choose(rows, configs)
    audit.compare_structure(actual, audit.choose_reference(rows, configs))
    expected = audit.summary_reference(rows, "selected_clean_equal_trial")
    audit.compare_structure(metrics.summarize(rows, "selected_clean_equal_trial"), expected)
    assert expected["gates"]["useful_T2_primary_recovery"]
    unavailable = copy.deepcopy(rows)
    for r in unavailable:
        if r["session_id"] == "T2-2":
            r["metrics"] = {"r2": None, "summed_mse": None}
    assert not audit.summary_reference(unavailable, "selected_clean_equal_trial")["gates"][
        "useful_T2_primary_recovery"
    ]
    rows[0]["score_mask_sha256"] = "different"
    with pytest.raises(ValueError, match="paired"):
        audit.choose_reference(rows, configs)


def test_zero_prediction_float32_targets_and_constant_axis_keep_correct_error():
    t = np.linspace(0.1, 5, 100)
    y = np.c_[np.cos(t), np.sin(t)].astype(np.float32)
    r = audit.score_reference(y, np.zeros_like(y, dtype=float), np.ones(len(y), bool))
    assert r["summed_mse"] == float(np.square(y.astype(float)).sum() / len(y))
    target = np.c_[np.linspace(1, 2, 100), np.ones(100)]
    pred = target.copy()
    pred[:, 1] += 1
    r = audit.score_reference(target, pred, np.ones(100, bool))
    assert r["r2"] < 0 and r["per_output_r2"][1] is None
