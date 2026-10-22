"""Synthetic fixed-recipe and per-file supervised-prefix contract checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neural_decoder import assessment
from neural_decoder.artifacts import sha256, write_json


def test_prefix_offset_uses_exact_declared_rows_and_only_available_labels():
    prediction = np.zeros((250, 7))
    target = np.arange(250)[:, None] + np.arange(7)[None, :]
    mask = np.arange(250) % 2 == 0
    expected = target[mask].mean(axis=0)
    np.testing.assert_array_equal(assessment.prefix_offset(prediction, target, mask, 250), expected)
    target = target.astype(float)
    target[~mask] = 1e9
    np.testing.assert_array_equal(assessment.prefix_offset(prediction, target, mask, 250), expected)
    np.testing.assert_array_equal(
        assessment.prefix_offset(np.empty((0, 7)), np.empty((0, 7)), np.empty(0, dtype=bool), 0),
        np.zeros(7),
    )
    with pytest.raises(ValueError, match="exactly its declared prefix"):
        assessment.prefix_offset(np.zeros((251, 7)), np.zeros((251, 7)), np.ones(251, bool), 250)
    with pytest.raises(ValueError, match="finite scored rows"):
        assessment.prefix_offset(prediction, target, np.zeros(250, bool), 250)


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    index.write_text('{"synthetic": true}\n')
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text('{"synthetic": true}\n')
    digest = sha256(freeze_path)
    model = {
        "name": "ridge",
        "kind": "linear",
        "path": "synthetic-model.npz",
        "sha256": "a" * 64,
        "training_days": [f"S{i}" for i in range(6)],
        "regime": "frozen_zero_calibration",
    }
    freeze = {"stage": "later", "models": [model]}
    primary = tmp_path / "primary"
    (primary / "ridge").mkdir(parents=True)
    files, hashes, predictions = [], {}, {}
    for number, day in enumerate(["S6", "S10"]):
        n = 1613 + number * 17
        t = np.arange(n)[:, None]
        prediction = np.cos(t * 0.013 + np.arange(7)[None, :])
        target = prediction + (number * 10 + np.arange(7)[None, :] + t / 1000)
        mask = np.arange(n) % (3 + number) != 0
        item = {"file_id": f"{day}_set_1", "day": day}
        files.append((item, np.zeros((n, 176), np.uint32), target, mask))
        path = primary / "ridge" / f"{item['file_id']}.npy"
        np.save(path, prediction)
        hashes[item["file_id"]] = sha256(path)
        predictions[item["file_id"]] = prediction
    entries = [{"model": model, "prediction_sha256": hashes, "scores": {"r2_mean": -1.0}}]
    summary = {
        "stage": "later",
        "freeze_sha256": digest,
        "index_sha256": sha256(index),
        "models": entries,
    }
    write_json(primary / "summary.json", summary)
    manifest = {
        "status": "complete",
        "freeze_sha256": digest,
        "index_sha256": sha256(index),
        "models": entries,
        "summary_sha256": sha256(primary / "summary.json"),
    }
    write_json(primary / "manifest.json", manifest)
    monkeypatch.setattr(assessment, "validate_freeze", lambda *args, **kwargs: freeze)
    monkeypatch.setattr(assessment, "assessment_files", lambda *args, **kwargs: ({}, files))
    monkeypatch.setattr(assessment, "source_identity", lambda *args: {"synthetic": True})
    monkeypatch.setattr(assessment, "environment", lambda: {"synthetic": True})
    return {
        "root": tmp_path,
        "index": index,
        "freeze_path": freeze_path,
        "digest": digest,
        "freeze": freeze,
        "primary": primary,
        "files": files,
        "predictions": predictions,
    }


def _run(calibration, name="calibration"):
    data = calibration
    return assessment.calibration_study(
        data["index"],
        data["freeze_path"],
        data["digest"],
        data["primary"],
        data["root"] / name,
        root=data["root"],
        cache=data["root"] / "cache",
    )


def test_all_budget_arms_receive_own_prefix_and_score_identical_suffix(calibration, monkeypatch):
    calls = []
    real_prefix = assessment.prefix_offset
    real_score = assessment.score_files
    scored = []

    def prefix(prediction, target, mask, budget):
        calls.append((budget, prediction.copy(), target.copy(), mask.copy()))
        return real_prefix(prediction, target, mask, budget)

    def score(records):
        scored.append([(r.file_id, r.targets.copy(), r.eval_mask.copy()) for r in records])
        return real_score(records)

    monkeypatch.setattr(assessment, "prefix_offset", prefix)
    monkeypatch.setattr(assessment, "score_files", score)
    output = _run(calibration)
    summary = json.loads((output / "summary.json").read_text())
    assert len(calls) == 8
    assert len(scored) == 4
    assert [r["nominal_label_seconds_per_day"] for r in summary["results"]] == [0, 10, 30, 60]
    for arm, budget in enumerate([0, 250, 750, 1500]):
        for position, (item, _, target, mask) in enumerate(calibration["files"]):
            received_budget, pred_prefix, target_prefix, mask_prefix = calls[arm * 2 + position]
            prediction = calibration["predictions"][item["file_id"]]
            assert received_budget == budget
            np.testing.assert_array_equal(pred_prefix, prediction[:budget])
            np.testing.assert_array_equal(target_prefix, target[:budget])
            np.testing.assert_array_equal(mask_prefix, mask[:budget])
            scored_id, scored_targets, scored_mask = scored[arm][position]
            assert scored_id == item["file_id"]
            np.testing.assert_array_equal(scored_targets, target[1500:])
            np.testing.assert_array_equal(scored_mask, mask[1500:])
            offset = (
                np.zeros(7)
                if budget == 0
                else (target[:budget][mask[:budget]] - prediction[:budget][mask[:budget]]).mean(0)
            )
            saved = np.load(output / "ridge" / f"prefix-{budget}" / f"{item['file_id']}.npy")
            np.testing.assert_array_equal(saved, prediction[1500:] + offset)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["summary_sha256"] == sha256(output / "summary.json")


@pytest.mark.parametrize("corruption", ["index", "summary", "model_inventory", "failed"])
def test_calibration_requires_exact_completed_primary_evidence(calibration, corruption):
    primary = calibration["primary"]
    path = primary / "manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "index":
        manifest["index_sha256"] = "0" * 64
    elif corruption == "failed":
        manifest["status"] = "failed"
    elif corruption == "summary":
        (primary / "summary.json").write_text("{}")
    else:
        manifest["models"] = []
        summary = json.loads((primary / "summary.json").read_text())
        summary["models"] = []
        write_json(primary / "summary.json", summary)
        manifest["summary_sha256"] = sha256(primary / "summary.json")
    write_json(path, manifest)
    with pytest.raises(ValueError):
        _run(calibration)
    assert not (calibration["root"] / "calibration").exists()


def test_changed_prediction_retains_failed_run_without_completed_summary(calibration):
    path = calibration["primary"] / "ridge/S6_set_1.npy"
    values = np.load(path)
    values[0, 0] += 99
    np.save(path, values)
    with pytest.raises(ValueError, match="trace changed"):
        _run(calibration)
    output = calibration["root"] / "calibration"
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("problem", ["missing_freeze", "changed_recipe", "wrong_refit_day"])
def test_later_refit_guard_runs_before_fitting(tmp_path, monkeypatch, problem):
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}\n")
    index = tmp_path / "index.json"
    index.write_text("{}\n")
    files = [({"day": f"S{i}"}, None, None, None) for i in range(5)]
    monkeypatch.setattr(assessment, "load_development_files", lambda path: ({}, files))
    monkeypatch.setattr(
        assessment, "fit_linear", lambda *args, **kwargs: pytest.fail("Fitting began")
    )
    kwargs = (
        {}
        if problem == "missing_freeze"
        else {
            "early_index": tmp_path / "early-index.json",
            "early_freeze": tmp_path / "early.json",
            "early_sha256": "a" * 64,
        }
    )
    monkeypatch.setattr(
        assessment,
        "validate_freeze",
        lambda *args, **kwargs: {
            "stage": "early",
            "recipe": {"sha256": "b" * 64 if problem == "changed_recipe" else sha256(recipe)},
        },
    )
    monkeypatch.setattr(
        assessment,
        "assessment_files",
        lambda *args, **kwargs: ({}, [({"day": "S6"}, None, None, None)]),
    )
    if problem == "wrong_refit_day":
        kwargs["early_index"].write_text("{}\n")
    with pytest.raises(ValueError):
        assessment.fit_stage(
            index,
            recipe,
            tmp_path / "fit",
            stage="later",
            root=tmp_path,
            cache=tmp_path / "cache",
            **kwargs,
        )
    assert not (tmp_path / "fit").exists()


def test_incomplete_refit_cannot_be_frozen(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}\n")
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "manifest.json", {"status": "failed", "recipe_sha256": sha256(recipe)})
    with pytest.raises(ValueError, match="incomplete"):
        assessment.freeze_stage(
            run, recipe, tmp_path / "freeze.json", root=tmp_path, cache=tmp_path / "cache"
        )
    assert not (tmp_path / "freeze.json").exists()


def test_unknown_model_kind_is_not_silently_treated_as_tcn(tmp_path):
    model = tmp_path / "model.npz"
    model.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="Unknown frozen decoder kind"):
        assessment.make_decoder(
            {"path": "model.npz", "sha256": sha256(model), "kind": "gru"}, tmp_path
        )


def test_independent_audit_recomputes_budget_offsets_and_common_suffix(calibration, monkeypatch):
    from neural_experiments import audit_assessment as independent

    output = _run(calibration)
    monkeypatch.setattr(independent, "verify_snapshot", lambda *args: None)
    retained = {
        ("ridge", item["file_id"]): (item, calibration["predictions"][item["file_id"]], y, mask)
        for item, _, y, mask in calibration["files"]
    }
    primary_result = {
        "freeze_sha256": calibration["digest"],
        "index_sha256": sha256(calibration["index"]),
    }
    result = independent.audit_calibration(
        output, calibration["primary"], primary_result, retained, calibration["freeze"]
    )
    assert result["arms_checked"] == 4
    assert result["prediction_traces_checked"] == 8
    assert result["max_absolute_prediction_difference"] == 0
    # Even coherently rehashing a wrong calibrated trace cannot fool arithmetic recomputation.
    summary = json.loads((output / "summary.json").read_text())
    relative = "ridge/prefix-250/S6_set_1.npy"
    path = output / relative
    wrong = np.load(path)
    wrong[0, 0] += 1
    np.save(path, wrong)
    summary["prediction_sha256"][relative] = sha256(path)
    write_json(output / "summary.json", summary)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["summary_sha256"] = sha256(output / "summary.json")
    write_json(output / "manifest.json", manifest)
    with pytest.raises(AssertionError):
        independent.audit_calibration(
            output, calibration["primary"], primary_result, retained, calibration["freeze"]
        )


def test_independent_primary_audit_checks_inference_and_day_scores(tmp_path, monkeypatch):
    from neural_decoder.metrics import FileEvaluation, score_files
    from neural_decoder.models import LinearDecoder, LinearModel
    from neural_experiments import audit_assessment as independent

    rng = np.random.default_rng(903)
    model = LinearModel(
        np.zeros(176),
        np.ones(176),
        np.array([0.6, 0.3, 0.1]),
        rng.normal(0, 0.001, (352, 7)),
        np.zeros(7),
        2,
    )
    model_path = tmp_path / "ridge.npz"
    model.save(model_path)
    model_record = {
        "name": "ridge",
        "kind": "linear",
        "path": "ridge.npz",
        "sha256": sha256(model_path),
    }
    freeze = {"stage": "later", "models": [model_record], "source": {"files_sha256": {}}}
    freeze_path = tmp_path / "freeze.json"
    write_json(freeze_path, freeze)
    freeze_digest = sha256(freeze_path)
    run = tmp_path / "primary"
    (run / "ridge").mkdir(parents=True)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows, evaluated, prediction_hashes = [], [], {}
    for day, n in [("S6", 23), ("S10", 31)]:
        file_id = f"{day}_set_1"
        x = rng.poisson(2, (n, 176)).astype(np.uint32)
        saved = LinearDecoder(model).predict_chunk(x)
        y = saved + rng.normal(0, 0.01, saved.shape)
        mask = np.arange(n) % 3 != 0
        feature_path, target_path = (
            prepared / f"{file_id}-features.npz",
            prepared / f"{file_id}-targets.npz",
        )
        np.savez(
            feature_path,
            counts=x,
            timestamps=np.arange(n) * 0.02,
            sequence_id=np.arange(n),
            channel_ids=np.arange(176),
        )
        np.savez(target_path, targets=y, eval_mask=mask, trial_change=np.zeros(n, bool))
        prediction_path = run / "ridge" / f"{file_id}.npy"
        np.save(prediction_path, saved)
        prediction_hashes[file_id] = sha256(prediction_path)
        rows.append(
            {
                "file_id": file_id,
                "day": day,
                "n_bins": n,
                "features": feature_path.name,
                "features_sha256": sha256(feature_path),
                "targets": target_path.name,
                "targets_sha256": sha256(target_path),
            }
        )
        evaluated.append(FileEvaluation(file_id, day, saved, y, mask))
    index = {"files": rows}
    index_path = prepared / "index.json"
    write_json(index_path, index)
    scores = score_files(evaluated)
    entries = [{"model": model_record, "prediction_sha256": prediction_hashes, "scores": scores}]
    write_json(run / "ridge/scores.json", scores)
    summary = {
        "stage": "later",
        "models": entries,
        "freeze_sha256": freeze_digest,
        "index_sha256": sha256(index_path),
    }
    write_json(run / "summary.json", summary)
    (run / "source.diff").write_text("")
    manifest = {
        "status": "complete",
        "models": entries,
        "freeze_sha256": freeze_digest,
        "index_sha256": sha256(index_path),
        "summary_sha256": sha256(run / "summary.json"),
        "source": {"files": {}, "diff_sha256": sha256(run / "source.diff")},
    }
    write_json(run / "manifest.json", manifest)
    monkeypatch.setattr(independent, "validate_freeze", lambda *args, **kwargs: freeze)
    monkeypatch.setattr(independent, "load_assessment_index", lambda *args, **kwargs: index)
    result, _, _ = independent.audit_primary(
        run, index_path, freeze_path, freeze_digest, root=tmp_path, cache=tmp_path / "cache"
    )
    assert result["prediction_traces_checked"] == 2
    assert result["max_absolute_prediction_difference"] < 1e-12
    assert result["models"][0]["scores"]["canonical_day_order"] == ["S10", "S6"]
