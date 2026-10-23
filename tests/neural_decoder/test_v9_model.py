import copy
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from neural_experiments.v8 import model as previous
from neural_experiments.v9 import development, metrics, model


@pytest.fixture
def case():
    rng = np.random.default_rng(19039)
    n, period = 24000, 400
    bins = np.arange(n)
    trial = bins // period
    start = trial * period
    theta = (trial % 4) * np.pi / 2
    target = np.column_stack((np.cos(theta), np.sin(theta)))
    progress = (bins - start) / period
    cursor = 0.15 * progress[:, None] * target
    cursor += 0.02 * np.sin(progress * 5)[:, None] * target[:, ::-1]
    displacement = target - cursor
    y = displacement / np.linalg.norm(displacement, axis=1)[:, None]
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
        "segments": [(0, 12000), (12000, n)],
        "y": y.astype(np.float32),
        "eval_mask": np.ones(n, bool),
        "context": {
            "trial_id": trial,
            "trial_age_seconds": (bins - start) * 0.02 + 0.01,
            "trial_start_bin": start,
            "trial_stop_bin": start + period,
            "trial_onset_seconds": start * 0.02,
            "trial_right_censored": start + period > n,
            "reference40_distance": np.ones(n),
            "reference40_available": np.ones(n, bool),
            "reference40_direction": target.copy(),
            "target_position": target,
            "cursor_position": cursor,
        },
    }


@pytest.mark.parametrize(
    "config",
    [
        model.BASELINE,
        development.trial_config(0.2, 0.1),
        development.trial_config(0.8, 1.0, weighting="uniform"),
        development.trial_config(0.8, 0.1, target="endpoint"),
    ],
)
def test_weighted_mean_ridge_matches_independent_sklearn(case, config):
    raw = model.representation(case["features"], case["segments"], config)
    fitted, receipt, arrays = model.fit(case, config, raw=raw)
    indices, weights = arrays["training_indices"], arrays["training_weights"]
    z = np.clip((raw[indices] - fitted.mean) / fitted.scale, -8, 8)
    reference = Ridge(alpha=config["ridge"] * weights.sum(), fit_intercept=True).fit(
        z,
        arrays["training_labels"],
        sample_weight=weights,
    )
    np.testing.assert_allclose(fitted.weights[:-1], reference.coef_.T, atol=3e-14, rtol=3e-13)
    np.testing.assert_allclose(fitted.weights[-1], reference.intercept_, atol=3e-14, rtol=3e-13)
    assert receipt["last_training_bin"] < receipt["label_stop_bin"]
    if config["weighting"] == "sector":
        sectors = arrays["training_sectors"]
        sums = [weights[sectors == sector].sum() for sector in np.unique(sectors)]
        np.testing.assert_allclose(sums, np.mean(sums))


def test_startup_control_is_exact_previous_decoder(case):
    first, _, _ = model.fit(case, model.BASELINE)
    second, _, _, _ = previous.fit(case, previous.BASELINE)
    for name in ("mean", "scale", "weights"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_quota_uses_completed_common_eligible_trials_and_fixed_sectors(case):
    trials, receipt = model.calibration(case, "quota24")
    assert receipt["quota_met"] and len(trials) == 24
    assert receipt["label_stop_bin"] == trials[-1]["completion_bin"]
    assert receipt["label_stop_bin"] < model.MAX_LABEL_STOP
    assert min(receipt["sector_trial_counts"]) >= 4
    for trial in trials:
        assert trial["common_stop_bin"] - trial["common_start_bin"] == 45
        for w in trial["windows"]:
            assert w["stop_bin"] - w["start_bin"] == 15
            assert w["first_raw_age_seconds"] >= w["rt_seconds"] - 1e-12
            assert w["first_raw_age_seconds"] < w["rt_seconds"] + 0.011
        rows = case["context"]["trial_id"] == trial["trial_id"]
        assert trial["completion_bin"] == case["context"]["trial_stop_bin"][rows][0] + 1


def test_quota_and_fit_ignore_unpurchased_future_behavior(case):
    config = development.trial_config(0.8, 0.1)
    raw = model.representation(case["features"], case["segments"], config)
    fitted, receipt, arrays = model.fit(case, config, raw=raw)
    stop = receipt["label_stop_bin"]
    poisoned = copy.deepcopy(case)
    poisoned["y"][stop:] = np.nan
    for key in (
        "target_position",
        "cursor_position",
        "reference40_direction",
        "reference40_distance",
    ):
        poisoned["context"][key][stop:] = np.nan
    changed_raw = raw.copy()
    changed_raw[stop:] = np.nan
    second, second_receipt, second_arrays = model.fit(poisoned, config, raw=changed_raw)
    np.testing.assert_array_equal(fitted.weights, second.weights)
    assert receipt == second_receipt
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], second_arrays[name])


def test_common_geometry_rejects_motion_between_candidate_windows(case):
    first = next(model.eligible_trials(case, limit_bin=model.MAX_LABEL_STOP))
    gap = first["windows"][0]["stop_bin"] + 1
    case["context"]["target_position"][gap] += [0.03, 0.0]
    remaining = list(model.eligible_trials(case, limit_bin=model.MAX_LABEL_STOP))
    assert first["trial_id"] not in {t["trial_id"] for t in remaining}


def test_exact_onset_and_raw_pair_anchor(case):
    first = next(model.eligible_trials(case, limit_bin=model.MAX_LABEL_STOP))
    ident = first["trial_id"]
    rows = case["context"]["trial_id"] == ident
    case["context"]["trial_onset_seconds"][rows] = 29.99
    assert ident not in {
        t["trial_id"] for t in model.eligible_trials(case, limit_bin=model.MAX_LABEL_STOP)
    }
    case["context"]["trial_onset_seconds"][rows] = 30.0
    case["context"]["trial_age_seconds"][rows] -= 0.01
    other = next(model.eligible_trials(case, limit_bin=model.MAX_LABEL_STOP))
    for old, new in zip(first["windows"], other["windows"], strict=True):
        assert new["start_bin"] == old["start_bin"] + 1
        assert new["first_raw_age_seconds"] == pytest.approx(new["rt_seconds"] + 0.01)


def test_completion_needs_observed_confirmation_bin(case):
    first = next(model.eligible_trials(case, limit_bin=model.MAX_LABEL_STOP))
    end = first["completion_bin"]
    assert not list(model.eligible_trials(case, limit_bin=end - 1))
    assert len(list(model.eligible_trials(case, limit_bin=end))) == 1
    case["context"]["trial_right_censored"][case["context"]["trial_id"] == first["trial_id"]] = True
    assert not list(model.eligible_trials(case, limit_bin=end))


def test_target_is_normalized_mean_displacement_not_mean_unit_vectors(case):
    first = next(model.eligible_trials(case, limit_bin=model.MAX_LABEL_STOP))
    window = first["windows"][0]
    a, b = window["start_bin"], window["stop_bin"]
    target = case["context"]["target_position"][a:b]
    vectors = np.tile([1.0, 0.0], (15, 1))
    vectors[:5] = [0.3, 0.04]
    case["context"]["cursor_position"][a:b] = target - vectors
    case["y"][a:b] = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    updated = next(model.eligible_trials(case, limit_bin=model.MAX_LABEL_STOP))["windows"][0]
    expected = vectors.mean(0) / np.linalg.norm(vectors.mean(0))
    np.testing.assert_allclose(updated["window_mean"], expected, atol=1e-15)
    wrong = (vectors / np.linalg.norm(vectors, axis=1)[:, None]).mean(0)
    wrong /= np.linalg.norm(wrong)
    assert np.linalg.norm(expected - wrong) > 0.01


@pytest.mark.parametrize("kind", ["ewma200", "boxcar300"])
def test_streaming_batch_reset_future_and_feature_only_contract(case, kind):
    config = model.BASELINE if kind == "ewma200" else development.trial_config(0.2, 1.0)
    fitted, _, _ = model.fit(case, config)
    raw = model.representation(case["features"], case["segments"], config)
    batch = fitted.predict_raw(raw)
    streaming = model.StreamingDecoder(fitted)
    output = np.empty_like(batch)
    for a, b in case["segments"]:
        streaming.reset()
        for i in range(a, b):
            output[i] = streaming.step({key: values[i] for key, values in case["features"].items()})
    np.testing.assert_allclose(output, batch, atol=3e-15, rtol=3e-14)
    changed = {k: v.copy() for k, v in case["features"].items()}
    for value in changed.values():
        value[14000:] *= 7
    np.testing.assert_array_equal(fitted.predict(changed, case["segments"])[:14000], batch[:14000])
    if kind == "boxcar300":
        transformed = np.column_stack(
            (np.sqrt(case["features"]["x_tx45"]), np.log1p(case["features"]["x_sbp"]))
        )
        for i in (0, 7, 63, 11999, 12000, 12010, 22000):
            start = max(0 if i < 12000 else 12000, i - 14)
            np.testing.assert_allclose(raw[i], transformed[start : i + 1].mean(0), atol=2e-14)
    with pytest.raises(ValueError):
        fitted.predict({**case["features"], "target_position": case["y"]}, case["segments"])


def test_common_primary_purge_and_right_censored_final_trial(case):
    mask, bought = model.score_mask(case)
    ids = case["context"]["trial_id"]
    assert not mask[:19600].any() and not np.isin(ids[mask], bought).any()
    final = ids[-1]
    case["context"]["trial_right_censored"][ids == final] = True
    changed, _ = model.score_mask(case)
    np.testing.assert_array_equal(mask, changed)
    assert mask[-1]
    secondary = list(
        model.eligible_trials(case, limit_bin=len(mask), training=False, required_mask=mask)
    )
    assert final not in {t["trial_id"] for t in secondary}
    for t in secondary:
        assert mask[t["common_start_bin"] : t["common_stop_bin"]].all()


def test_raw_onset_crossing_purchase_end_is_purged_before_first_complete_pair(case):
    context = case["context"]
    context["trial_id"][19500:19800] = 999
    context["trial_start_bin"][19500:19800] = 19500
    context["trial_stop_bin"][19500:19800] = 19800
    context["trial_onset_seconds"][19500:19800] = 389.99
    mask, purchased = model.score_mask(case)
    assert 999 in purchased and not mask[19500:19800].any()
    context["trial_onset_seconds"][19500:19800] = 390.0
    mask, purchased = model.score_mask(case)
    assert 999 not in purchased and mask[19600:19800].all()


def test_timeout_fits_available_trials_and_zero_uses_no_labels(case):
    # Sparse direction support never passes the balanced quota, but remains a fit.
    case["context"]["reference40_direction"][:] = [1.0, 0.0]
    fitted, receipt, arrays = model.fit(case, development.trial_config(0.2, 1.0))
    assert receipt["stop_reason"] == "deadline" and not receipt["quota_met"]
    assert receipt["label_stop_bin"] == model.MAX_LABEL_STOP
    assert receipt["training_trials"] > 0 and not receipt["fallback_zero"]
    assert np.any(fitted.weights != 0)
    empty = copy.deepcopy(case)
    empty["y"][:] = np.nan
    for key in ("target_position", "cursor_position", "reference40_direction"):
        empty["context"][key][:] = np.nan
    zero, zreceipt, zarrays = model.fit(empty, {**model.BASELINE, "control": "zero"})
    assert zreceipt["available_label_seconds"] == 0 and zreceipt["training_rows"] == 0
    assert not zero.weights.any() and not zarrays["training_mask"].any()


def test_twelve_exported_arms_and_paired_secondary_replay(case, tmp_path):
    configs = development.assessment_configs(development.trial_config(0.8, 0.1))
    result = model.evaluate_session(case, configs, artifact_dir=tmp_path / "artifacts")
    assert len(result["rows"]) == 12
    assert len({r["score_mask_sha256"] for r in result["rows"]}) == 1
    assert len({r["secondary"]["score_mask_sha256"] for r in result["rows"]}) == 1
    summary = metrics.summarize(result["rows"])
    assert summary["selected_sessions"] == 1
    assert set(summary["methods"]) == set(configs)
    for row in result["rows"]:
        method = row["method"]
        receipt = row["training"]
        restored = model.Decoder.load(row["model"]["model_path"])
        prediction = restored.predict(case["features"], case["segments"])
        prediction[: receipt["deployment_bin"]] = 0
        np.testing.assert_array_equal(result["traces"][method + "__prediction"], prediction)
        indices = result["traces"][method + "__secondary_indices"]
        expected = model.trial_metrics(
            result["traces"][method + "__secondary_window_mean"], prediction[indices]
        )
        assert row["secondary"]["window_mean"] == expected
        assert isinstance(row["secondary"]["endpoint"], dict)
        assert row["calibration"]["observed_label_seconds"] == row["available_label_seconds"]
        json.dumps(row, allow_nan=False)
    primary = result["traces"]["quota_sector_mean__training_labels"]
    permuted = result["traces"]["trialshuffle__training_labels"]
    np.testing.assert_array_equal(permuted, np.roll(primary, -1, axis=0))
    with pytest.raises(FileExistsError):
        model.evaluate_session(case, configs, artifact_dir=tmp_path / "artifacts")


def test_source_choice_preserves_first_tie_and_common_availability():
    configs = {"a": {}, "b": {}}
    rows = [
        {"session_id": sid, "method": method, "score_mask_sha256": sid, "metrics": {"r2": value}}
        for sid, value in [("one", 0.2), ("two", -0.1)]
        for method in configs
    ]
    choice = development.choose(rows, configs)
    assert choice["selected_method"] == "a"
    rows[-1]["metrics"]["r2"] = None
    with pytest.raises(ValueError, match="availability"):
        development.choose(rows, configs)


def test_export_rejects_noninteger_version(case, tmp_path):
    fitted, _, _ = model.fit(case, model.BASELINE)
    path = tmp_path / "model.npz"
    fitted.save(path)
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    values["version"] = np.asarray(9.0)
    np.savez(path, **values)
    with pytest.raises(ValueError, match="V9"):
        model.Decoder.load(path)


def test_source_runner_retains_grid_and_final_stages_with_locked_sources(tmp_path, monkeypatch):
    from pathlib import Path

    from neural_experiments.v9 import assessment

    plan = json.loads(Path("neural_experiments/v9/study-plan.json").read_text())
    people = plan["development"]["study_participants"]
    items = [
        {
            "participant_id": person,
            "session_id": f"{person}_{rank}",
            "post_implant_day": rank,
            "source": {"archive_ordinal": rank},
        }
        for person in people
        for rank in range(1, 25)
    ]
    items += [{"participant_id": "T3", "session_id": f"unused{i}"} for i in range(104)]
    index = {"sessions": items}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    plan["development"]["index_sha256"] = development.digest(index_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    source = tmp_path / "fakecore.py"
    source.write_text("constant = 1\n")
    hashes = {"fakecore.py": development.digest(source)}
    locks, calls = [], []

    def lock(path, expected):
        assert development.digest(path) == expected
        locks.append(Path(path).name)

    def load(path, **kwargs):
        assert kwargs == {"expected_role": "development", "verify_hashes": True}
        return index

    def evaluate(case, configs, *, artifact_dir):
        artifact_dir.mkdir(parents=True, exist_ok=False)
        item = case["item"]
        calls.append((item["session_id"], tuple(configs), artifact_dir))
        rows = []
        for name in configs:
            value = 0.3 if name == "grid_02" else 0.1
            rows.append(
                {
                    "participant": item["participant_id"],
                    "session_id": item["session_id"],
                    "post_implant_day": item["post_implant_day"],
                    "method": name,
                    "score_mask_sha256": "fixed",
                    "metrics": {"r2": value},
                }
            )
        return {"rows": rows, "fit_receipts": {}, "model_refs": {}, "secondary_trials": []}

    def summarize(rows):
        assert len(rows) == 1152 and {r["method"] for r in rows} == set(plan["methods"])
        return {"selected_sessions": len({r["session_id"] for r in rows})}

    monkeypatch.setattr(development, "ROOT", tmp_path)
    monkeypatch.setattr(development, "relative", lambda path: str(path))
    monkeypatch.setattr(development, "_study_lock", lock)
    monkeypatch.setattr(development.subprocess, "check_output", lambda *a, **k: "synthetic-head\n")
    monkeypatch.setattr(development.platform, "platform", lambda: "synthetic-platform")
    monkeypatch.setattr(assessment, "scientific_sources", lambda: tuple(hashes))
    monkeypatch.setattr(assessment, "source_hashes", lambda: hashes)
    monkeypatch.setattr(development.data, "load_index", load)
    monkeypatch.setattr(development.data, "load_case", lambda p, item, **kw: {"item": item})
    monkeypatch.setattr(model, "evaluate_session", evaluate)
    monkeypatch.setattr(metrics, "summarize", summarize)
    result = development.run(
        index_path, tmp_path / "run", plan_path=plan_path, plan_sha256=development.digest(plan_path)
    )
    assert result["status"] == "complete" and locks == ["plan.json", "fakecore.py"]
    assert len(result["grid_rows"]) == 36 and len(result["final_rows"]) == 1152
    assert len(result["rows"]) == 1188 and len(result["sessions"]) == 105
    assert result["grid_selection"]["selected_method"] == "grid_02"
    assert len({path for _, _, path in calls}) == 105
    assert [sid for sid, _, _ in calls[:9]] == [f"T2_{rank}" for rank in range(16, 25)]
    provisional = json.loads((tmp_path / "run/selection-provisional.json").read_text())
    assert provisional["source_audit_status"] == "PENDING"
    assert provisional["source_manifest_sha256"] == development.digest(
        tmp_path / "run/manifest.json"
    )
    assert (tmp_path / "run/source_snapshot/fakecore.py").read_bytes() == source.read_bytes()
