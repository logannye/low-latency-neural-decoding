import copy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from neural_experiments.v7 import model as v7
from neural_experiments.v8 import development, model


@pytest.fixture
def case():
    rng = np.random.default_rng(7841)
    n = 10000
    trial = np.arange(n) // 137
    start = trial * 137
    angle = trial * 2.399 + 0.3 * np.sin((np.arange(n) - start) * 0.08)
    onset_angle = trial * 2.399
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    onset_direction = np.column_stack((np.cos(onset_angle), np.sin(onset_angle)))
    return {
        "item": {
            "participant_id": "T2",
            "session_id": "synthetic",
            "post_implant_day": 10,
            "n_bins": n,
            "n_channels": 2,
            "far_distance": 0.2,
        },
        "features": {
            "x_tx3": rng.poisson(0.4, (n, 2)),
            "x_tx45": rng.poisson(0.1, (n, 2)),
            "x_sbp": rng.gamma(2, 1, (n, 2)),
        },
        "segments": [(0, 4000), (4000, n)],
        "y": direction.astype(np.float32),
        "eval_mask": np.ones(n, bool),
        "context": {
            "trial_id": trial,
            "trial_age_seconds": (np.arange(n) - start) * 0.02 + 0.01,
            "trial_start_bin": start,
            "trial_stop_bin": np.minimum(start + 137, n),
            "trial_onset_seconds": start * 0.02,
            "trial_right_censored": start + 137 > n,
            "reference40_distance": np.ones(n),
            "reference40_available": np.ones(n, bool),
            "reference40_direction": onset_direction,
        },
    }


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"weighting": "equal_trial"},
        {"phase": "clean", "weighting": "equal_trial"},
        {"family": "tx3_sbp", "taus_seconds": [0.05, 0.5], "ridge": 10.0},
        {"phase": "clean", "target": "onset"},
    ],
)
def test_weighted_mean_ridge_matches_independent_sklearn(case, change):
    config = {**model.BASELINE, **change}
    raw = model.representation(case["features"], case["segments"], config)
    fitted, receipt, selected, weights = model.fit(case, config, raw=raw)
    z = np.clip((raw[selected] - fitted.mean) / fitted.scale, -8, 8)
    z /= np.sqrt(len(config["taus_seconds"]))
    y = (
        case["y"]
        if config["target"] == "instantaneous"
        else case["context"]["reference40_direction"]
    )
    reference = Ridge(alpha=config["ridge"] * weights.sum(), fit_intercept=True).fit(
        z, y[selected].astype(float), sample_weight=weights
    )
    np.testing.assert_allclose(fitted.weights[:-1], reference.coef_.T, atol=3e-14, rtol=3e-13)
    np.testing.assert_allclose(fitted.weights[-1], reference.intercept_, atol=3e-14, rtol=3e-13)
    assert receipt["last_training_bin"] < 7500
    assert not selected[:1500].any() and not selected[7500:].any()
    if config["weighting"] == "equal_trial":
        ids = case["context"]["trial_id"][selected]
        totals = [weights[ids == value].sum() for value in np.unique(ids)]
        np.testing.assert_allclose(totals, np.mean(totals))
        assert receipt["effective_trials"] == pytest.approx(len(totals))


def test_original_baseline_is_exact_v7_fit(case):
    raw = model.representation(case["features"], case["segments"], model.BASELINE)
    first, _, _, _ = model.fit(case, model.BASELINE, raw=raw)
    second, _ = v7.fit_mask(raw, case["y"], case["eval_mask"], case["segments"], 2, 7500)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)


@pytest.mark.parametrize("target", ["instantaneous", "onset"])
def test_future_or_unpurchased_labels_cannot_change_fit(case, target):
    config = {**model.BASELINE, "phase": "clean", "target": target, "weighting": "equal_trial"}
    raw = model.representation(case["features"], case["segments"], config)
    first, receipt, selected, weights = model.fit(case, config, raw=raw)
    poisoned = copy.deepcopy(case)
    poisoned["y"][~selected] = np.nan
    poisoned["context"]["reference40_direction"][~selected] = np.nan
    changed_raw = raw.copy()
    changed_raw[7500:] = np.nan
    second, other, _, next_weights = model.fit(poisoned, config, raw=changed_raw)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(weights, next_weights)
    assert receipt == other


def test_clean_phase_uses_exact_raw_onset_not_rounded_paired_bound(case):
    context = case["context"]
    context["trial_start_bin"][1510:1540] = 1500
    context["trial_onset_seconds"][1510:1540] = 29.99
    context["trial_age_seconds"][1510:1540] = 0.3
    selected = model.training_mask(case, {**model.BASELINE, "phase": "clean"})
    assert not selected[1510:1540].any()
    context["trial_onset_seconds"][1510:1540] = 30.0
    assert model.training_mask(case, {**model.BASELINE, "phase": "clean"})[1510:1540].all()


def test_common_masks_exclude_purchased_trial_and_keep_right_censor(case):
    primary, secondary, purchased = model.score_masks(case)
    ids = case["context"]["trial_id"]
    assert np.isin(ids[primary], purchased).sum() == 0
    assert not primary[:7600].any()
    assert primary[-1] and case["context"]["trial_right_censored"][-1]
    assert np.all(primary[secondary])
    for a, _ in case["segments"]:
        assert not primary[a : a + 62].any()
    crossing = ids[7499]
    assert not primary[ids == crossing].any()


@pytest.mark.parametrize("taus", [[0.05], [0.2], [0.5], [0.05, 0.5]])
def test_streaming_causal_representation_and_resets_match_batch(case, taus):
    config = {**model.BASELINE, "family": "tx3_sbp", "taus_seconds": taus}
    fitted, _, _, _ = model.fit(case, config)
    batch = fitted.predict(case["features"], case["segments"])
    stream = model.StreamingDecoder(fitted)
    replay = np.empty_like(batch)
    for a, b in case["segments"]:
        stream.reset()
        for i in range(a, b):
            replay[i] = stream.step({key: value[i] for key, value in case["features"].items()})
    np.testing.assert_allclose(replay, batch, atol=5e-15, rtol=5e-14)
    future = {key: value.copy() for key, value in case["features"].items()}
    for key in future:
        future[key][7500:] *= 5
    np.testing.assert_array_equal(fitted.predict(future, case["segments"])[:7500], batch[:7500])
    with pytest.raises(ValueError, match="labels or unknown"):
        fitted.predict({**case["features"], "target_position": case["y"]}, case["segments"])


def test_trial_permutation_is_paid_only_deterministic_whole_trial_rotation(case):
    config = {**model.BASELINE, "control": "trial_permutation"}
    _, receipt, selected, _ = model.fit(case, config)
    ids, y = case["context"]["trial_id"][selected], case["y"][selected].astype(float)
    reference = y.copy()
    for mapping in receipt["permutation"]:
        recipient = np.flatnonzero(ids == mapping["recipient_trial"])
        donor = np.flatnonzero(ids == mapping["donor_trial"])
        reference[recipient] = y[donor[np.arange(len(recipient)) * len(donor) // len(recipient)]]
        assert mapping["recipient_trial"] != mapping["donor_trial"]
    assert model.digest_array(reference) == receipt["training_labels_sha256"]
    assert not np.array_equal(reference, y)


def test_export_replay_all_controls_shared_masks_and_different_target_heads(case, tmp_path):
    factorials = development.factorial_configs({**model.BASELINE, "family": "tx3_sbp"})
    configurations = development.assessment_configs(factorials, "selected_all_equal_trial")
    result = model.evaluate_session(case, configurations, artifact_dir=tmp_path / "run")
    rows = result["rows"]
    assert len(rows) == 12 and len({r["score_mask_sha256"] for r in rows}) == 1
    assert len({r["secondary"]["score_mask_sha256"] for r in rows}) == 1
    with np.load(rows[0]["trace"]["trace_path"], allow_pickle=False) as trace:
        np.testing.assert_array_equal(trace["primary_mask"], result["traces"]["primary_mask"])
    for row in rows:
        fitted = model.Decoder.load(row["model"]["model_path"])
        predicted = fitted.predict(case["features"], case["segments"])
        predicted[:7600] = 0
        np.testing.assert_array_equal(predicted, result["traces"][row["method"] + "__prediction"])
        assert row["training"]["export_prediction_max_error"] == 0
    zero = next(r for r in rows if r["method"] == "zero")
    assert zero["metrics"]["angular_error_degrees"] == 90.0
    assert not result["traces"]["zero__prediction"].any()
    assert next(r for r in rows if r["method"] == "primary_onset")["target"] == "onset"
    constant = model.Decoder.load(
        next(r for r in rows if r["method"] == "constant")["model"]["model_path"]
    )
    assert not constant.weights[:-1].any()
    with pytest.raises(FileExistsError):
        model.evaluate_session(case, configurations, artifact_dir=tmp_path / "run")


@pytest.mark.parametrize("n", [1000, 1500, 7499])
def test_short_recording_retained_as_zero_with_unavailable_scores(case, n):
    short = copy.deepcopy(case)
    for name in ("features", "context"):
        short[name] = {key: value[:n] for key, value in short[name].items()}
    for key in ("y", "eval_mask"):
        short[key] = short[key][:n]
    short["segments"] = [(0, n)]
    result = model.evaluate_session(short, {"baseline": model.BASELINE})
    assert result["rows"][0]["metrics"]["r2"] is None
    assert result["rows"][0]["training"]["fallback_zero"]
    assert not result["traces"]["baseline__prediction"].any()


def test_secondary_trial_means_require_two_trials_and_are_not_bin_scores():
    y = np.array([[1.0, 0], [1.0, 0], [0.0, 1]])
    prediction = np.array([[1.0, 0], [-1.0, 0], [0.0, 1]])
    mask, ids = np.ones(3, bool), np.array([0, 0, 1])
    result = model._trial_metrics(y, prediction, mask, ids)
    assert result["trials"] == 2 and result["summed_mse"] == 0.5
    assert model._trial_metrics(y, prediction, mask, np.zeros(3, int))["r2"] is None


def test_development_declared_tie_order_and_matched_availability():
    configurations = {"first": {}, "second": {}}
    rows = [
        {"session_id": "a", "method": m, "score_mask_sha256": "same", "metrics": {"r2": 0.1}}
        for m in configurations
    ]
    assert development.choose(rows, configurations)["selected_method"] == "first"
    changed = copy.deepcopy(rows)
    changed[1]["metrics"]["r2"] = None
    with pytest.raises(ValueError, match="availability"):
        development.choose(changed, configurations)
    changed = copy.deepcopy(rows)
    changed[1]["score_mask_sha256"] = "different"
    with pytest.raises(ValueError, match="identical masks"):
        development.choose(changed, configurations)
    plan = json.loads(Path("neural_experiments/v8/study-plan.json").read_text())
    configs, ids = development.representation_grid(plan)
    assert len(configs) == 8 and next(iter(ids.values())) == "baseline_fallback"


def test_development_orchestration_frozen_two_stage_choice_and_controls(tmp_path, monkeypatch):
    from neural_experiments.v8 import assessment

    index_path = tmp_path / "index.json"
    index_path.write_text("{}")
    items = [
        {
            "participant_id": "T2",
            "session_id": f"t2_{i:02d}",
            "source": {"archive_ordinal": i},
            "post_implant_day": 1000 - i,
        }
        for i in range(1, 22)
    ]
    items += [{"participant_id": "T5"}] * 155
    calls = []

    def load_index(path, **kwargs):
        assert kwargs == {"expected_role": "development", "verify_hashes": True}
        return {"sessions": items}

    def load_case(path, item, **kwargs):
        assert kwargs == {"labels": True, "context": True}
        return {"item": item}

    def evaluate(case, configurations, artifact_dir):
        assert not artifact_dir.exists()
        artifact_dir.mkdir(parents=True)
        item = case["item"]
        calls.append((artifact_dir.parent.name, item["session_id"], tuple(configurations)))
        rows = []
        for method in configurations:
            r2 = 0.1 if method == "representation_06" else 0.0
            if method == "selected_clean_equal_trial":
                r2 = 0.2
            if method == "primary_onset":
                r2 = 100.0  # A diagnostic can never substitute for primary selection.
            rows.append(
                {
                    "session_id": item["session_id"],
                    "participant": "T2",
                    "method": method,
                    "metrics": {"r2": r2},
                    "score_mask_sha256": "same",
                }
            )
        return {"rows": rows, "traces": {}, "fit_receipts": {}, "model_refs": {}}

    monkeypatch.setattr(development.data, "load_index", load_index)
    monkeypatch.setattr(development.data, "load_case", load_case)
    monkeypatch.setattr(development.model, "evaluate_session", evaluate)
    monkeypatch.setattr(development, "_study_lock", lambda *args: None)
    monkeypatch.setattr(development, "relative", lambda path: str(path))
    monkeypatch.setattr(assessment, "source_hashes", lambda: {})
    monkeypatch.setattr(assessment, "scientific_sources", lambda: ())
    output = tmp_path / "run"
    manifest = development.run(index_path, output)
    selection = json.loads((output / "selection-provisional.json").read_text())
    assert manifest["status"] == "complete"
    assert selection["primary_method"] == "selected_clean_equal_trial"
    assert selection["representation_choice"]["method"] == "representation_06"
    assert selection["source_audit_status"] == "PENDING"
    assert selection["source_manifest_sha256"] == development.digest(output / "manifest.json")
    assert len(manifest["rows"]) == 15 * 8 + 6 * 8 + 6 * 4
    assert len(calls) == 27
    assert [sid for stage, sid, _ in calls if stage == "representation"] == [
        f"t2_{i:02d}" for i in range(1, 16)
    ]
    assert [sid for stage, sid, _ in calls if stage == "factorial"] == [
        f"t2_{i:02d}" for i in range(16, 22)
    ]
    assert len(selection["configurations"]) == 12
    assert selection["onset_matched_instantaneous_method"] == "selected_clean_equal_trial"
