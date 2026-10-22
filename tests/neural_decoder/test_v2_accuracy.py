"""Synthetic checks for the V2 experiment's temporal and label boundaries."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_decoder.temporal import CausalTCN

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "v2_accuracy", ROOT / "neural_experiments/v2/accuracy.py"
)
accuracy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accuracy)


@pytest.fixture(autouse=True)
def bounded_runtime():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    state = torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.set_rng_state(state)


def synthetic(length=103, offset=0, channels=4, outputs=2):
    rng = np.random.default_rng(100 + offset)
    x = rng.poisson(2, (length, channels)).astype(np.float32)
    y = np.arange(length)[:, None] + offset + np.arange(outputs)[None] / 8
    mask = np.ones(length, dtype=bool)
    mask[::7] = False
    return x, y, mask


@pytest.mark.parametrize("dilations", [(1, 2, 4), (1, 2, 4, 8, 16)])
def test_common_chunk_loss_positions_and_whole_file_parity(dilations):
    training = [synthetic(), synthetic(87, 200)]
    model = accuracy.TemporalCore(4, 2, dilations, 5).eval()
    transforms, (x, y, mask) = accuracy.training_arrays(
        training,
        receptive_field=model.receptive_field,
        chunk_scored_bins=13,
        minimum_loss_start=62,
    )
    identities = np.rint(
        (y.numpy() * transforms["target_scale"] + transforms["target_mean"])[:, :, 0]
    ).astype(int)
    expected = np.r_[
        np.arange(103)[training[0][2] & (np.arange(103) >= 62)],
        (200 + np.arange(87))[training[1][2] & (np.arange(87) >= 62)],
    ]
    np.testing.assert_array_equal(np.sort(identities[mask]), expected)
    assert not mask[:, : model.receptive_field - 1].any()
    assert len(identities[mask]) == len(set(identities[mask]))
    full = {}
    for counts, target, _ in training:
        pred = accuracy.predict(model, transforms, counts)
        full.update(zip(target[:, 0].astype(int), pred, strict=True))
    with torch.inference_mode():
        chunks = model(x).numpy() * transforms["target_scale"] + transforms["target_mean"]
        changed = x.clone()
        # Every false trailing padding element can be arbitrary without affecting any scored bin.
        for row in range(len(mask)):
            last = int(torch.nonzero(mask[row])[-1])
            changed[row, last + 1 :] = 1e5
        altered = model(changed).numpy()
        original = model(x).numpy()
    np.testing.assert_allclose(altered[mask], original[mask], atol=1e-6, rtol=1e-6)
    for identity, prediction in zip(identities[mask], chunks[mask], strict=True):
        np.testing.assert_allclose(prediction, full[identity], atol=2e-5, rtol=2e-6)


def test_rf15_matches_original_architecture():
    original = CausalTCN().eval()
    model = accuracy.TemporalCore(176, 7).eval()
    model.load_state_dict(original.state_dict())
    x = torch.randn(1, 83, 176)
    torch.testing.assert_close(model(x), original(x), rtol=0, atol=0)


@pytest.mark.parametrize("dilations", [(1, 2, 4), (1, 2, 4, 8, 16)])
def test_stream_causality_partition_reset_and_export(tmp_path, dilations):
    training = [synthetic(111)]
    transforms, _ = accuracy.training_arrays(training, receptive_field=63)
    model = accuracy.TemporalCore(4, 2, dilations, 5).eval()
    counts = training[0][0]
    expected = accuracy.predict(model, transforms, counts)
    changed = counts.copy()
    changed[58:] = 1e4
    np.testing.assert_array_equal(accuracy.predict(model, transforms, changed)[:58], expected[:58])
    stream = accuracy.CausalStream(model, transforms)
    actual = np.concatenate(
        [
            stream.predict_chunk(counts[:3]),
            stream.predict_chunk(counts[3:59]),
            stream.predict_chunk(counts[59:]),
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-6)
    stream.reset()
    np.testing.assert_array_equal(stream.predict_chunk(counts), actual)
    path = tmp_path / "model.npz"
    accuracy.save_model(path, model, transforms)
    loaded, affine, _ = accuracy.load_model(path)
    np.testing.assert_array_equal(accuracy.predict(loaded, affine, counts), expected)
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    arrays["state__convs.0.weight"][0, 0, 0] = np.nan
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="model tensor"):
        accuracy.load_model(tmp_path / "bad.npz")


def test_fixed_fit_checkpoints_training_transforms_and_source_loss(tmp_path):
    training = [synthetic(90)]
    model, affine, losses, checkpoints = accuracy.fit_model(
        training,
        epochs=2,
        checkpoint_epochs=(1, 2),
        width=4,
        minimum_loss_start=62,
        chunk_scored_bins=11,
        batch_chunks=2,
    )
    assert [row["epoch"] for row in losses] == [1, 2]
    assert set(checkpoints) == {1, 2}
    expected_rows = int(training[0][2][62:].sum())
    assert all(row["scored_rows"] == expected_rows for row in losses)
    np.testing.assert_allclose(affine["target_mean"], training[0][1][training[0][2]].mean(0))
    np.testing.assert_allclose(affine["input_mean"], training[0][0].mean(0))
    residual = np.asarray([[1.0, 2.0], [-3.0, 4.0]])
    weights = accuracy.loss_weights(affine, "source_units")
    np.testing.assert_allclose(
        np.mean(residual**2 * weights),
        np.mean((residual * affine["target_scale"]) ** 2) / np.mean(affine["target_scale"] ** 2),
    )
    accuracy.save_model(tmp_path / "trained.npz", model, affine)
    loaded, transforms, _ = accuracy.load_model(tmp_path / "trained.npz")
    np.testing.assert_allclose(
        accuracy.CausalStream(loaded, transforms).predict_chunk(training[0][0]),
        accuracy.predict(model, affine, training[0][0]),
        atol=2e-5,
        rtol=2e-6,
    )


def test_matched_exposure_exact_duration_labels_and_disjoint_files():
    same = ({"file_id": "S2_set_1"}, *synthetic(400))
    earlier = [
        ({"file_id": "S1_set_1"}, *synthetic(180, 1000)),
        ({"file_id": "S1_set_2"}, *synthetic(250, 2000)),
    ]
    for fraction in (0.25, 0.5, 1.0):
        conditions, metadata = accuracy.match_exposure(same, earlier, fraction)
        assert all(
            sum(len(x) for x, _, _ in records) == int(400 * fraction)
            for records in conditions.values()
        )
        counts = [sum(int(m.sum()) for _, _, m in records) for records in conditions.values()]
        assert counts == [metadata["eligible_labels_each"]] * 2
        assert metadata["same_day_sources"] == [
            {"file_id": "S2_set_1", "start": 0, "stop": int(400 * fraction)}
        ]
        assert all(
            source["file_id"].startswith("S1") for source in metadata["earlier_days_sources"]
        )
        for records in conditions.values():
            assert all(not m[:62].any() for _, _, m in records)
    assert same[3][1]  # Matching never mutates original masks.


def test_calibration_uses_only_prefix_and_matches_independent_closed_form():
    rng = np.random.default_rng(5)
    latent = rng.normal(size=(50, 4))
    base = rng.normal(size=(50, 2))
    target = base + latent @ rng.normal(size=(4, 2)) + 2
    mask = np.arange(50) % 3 != 0
    boundary = 17
    coefficient, intercept = accuracy.fit_calibration(
        latent[:boundary],
        base[:boundary],
        target[:boundary],
        mask[:boundary],
        method="ridge_delta_head",
    )
    x = np.c_[latent[:boundary][mask[:boundary]], np.ones(mask[:boundary].sum())]
    regularizer = np.diag([1.0] * 4 + [0.0])
    expected = np.linalg.solve(
        x.T @ x / len(x) + regularizer,
        x.T @ (target[:boundary] - base[:boundary])[mask[:boundary]] / len(x),
    )
    np.testing.assert_allclose(np.r_[coefficient, intercept[None]], expected, atol=1e-12)
    poisoned = target.copy()
    poisoned[boundary:] = 1e100
    alternative = accuracy.fit_calibration(
        latent[:boundary],
        base[:boundary],
        poisoned[:boundary],
        mask[:boundary],
        method="ridge_delta_head",
    )
    np.testing.assert_array_equal(coefficient, alternative[0])
    np.testing.assert_array_equal(intercept, alternative[1])
    for method in ("unadapted", "intercept", "ridge_delta_head"):
        a, b = accuracy.fit_calibration(latent[:0], base[:0], target[:0], mask[:0], method=method)
        assert not a.any() and not b.any()


def test_candidate_selection_uses_all_seeds_and_day_harm_gate():
    plan = json.loads((ROOT / "neural_experiments/v2/accuracy_plan.json").read_text())
    rows = []
    gains = {
        "rf15_standardized": [0, 0, 0],
        "rf63_standardized": [0.03, 0.03, -0.02],
        "rf15_source_units": [0.01, 0.01, 0.01],
        "rf63_source_units": [0.01, 0.01, 0.01],
    }
    for order, arm in enumerate(plan["arms"]):
        for seed in plan["seeds"]:
            for i, day in enumerate(("S2", "S3", "S4")):
                rows.append(
                    {
                        "arm": arm["name"],
                        "seed": seed,
                        "day": day,
                        "scores": {"r2_mean": 0.2 + gains[arm["name"]][i]},
                        "model": {"path": f"{arm['name']}/{seed}/{day}"},
                        "parameter_count": 10 + order,
                        "receptive_field": 15,
                    }
                )
    selected = accuracy.select_candidate(rows, plan)
    assert selected["selected_arm"] == "rf15_source_units"
    assert selected["runtime_candidate"]["seed"] == 0
    assert selected["runtime_candidate"]["day"] == "S4"
    assert not selected["decisions"][1]["eligible"]
    with pytest.raises(ValueError, match="every arm seed/day"):
        accuracy.select_candidate(rows[:-1], plan)


def test_published_trace_audit_recomputes_scores_beyond_hash_checks(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "v2_accuracy_auditor", ROOT / "neural_experiments/v2/audit_accuracy.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    run = tmp_path / "run"
    run.mkdir()
    plan_path = run / "source/neural_experiments/v2/fresh-plan-name.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps({"arms": [{"name": "rf15_standardized"}]}))
    index = {"files": []}
    rows = []
    target = np.arange(24, dtype=np.float64).reshape(12, 2) ** 1.2
    prediction = target + 0.1
    sse = ((prediction - target) ** 2).sum(axis=0)
    sst = ((target - target.mean(0)) ** 2).sum(axis=0)
    score, per_output = float(1 - sse.sum() / sst.sum()), (1 - sse / sst).tolist()
    for day in ("S2", "S3", "S4"):
        file_id = f"{day}_set_1"
        target_path = tmp_path / f"{file_id}.targets.npz"
        np.savez(target_path, targets=target, eval_mask=np.ones(12, dtype=bool))
        prediction_path = run / f"{file_id}.npy"
        np.save(prediction_path, prediction, allow_pickle=False)
        index["files"].append(
            {
                "file_id": file_id,
                "day": day,
                "targets": target_path.name,
                "targets_sha256": auditor.digest(target_path),
            }
        )
        for seed in (0, 1, 2):
            rows.append(
                {
                    "arm": "rf15_standardized",
                    "seed": seed,
                    "day": day,
                    "parameter_count": 3,
                    "predictions": {
                        file_id: {
                            "path": prediction_path.name,
                            "sha256": auditor.digest(prediction_path),
                        }
                    },
                    "scores": {
                        "r2_mean": score,
                        "num_scored_bins": 12,
                        "per_day": {day: {"r2_per_output": per_output}},
                    },
                }
            )
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    summary = {
        "main": rows,
        "diagnostics": [],
        "calibration": [],
        "schedule_decision": {"results": []},
        "selection": {"runtime_candidate": {"arm": "rf15_standardized", "seed": 0, "day": "S4"}},
    }

    def retain():
        (run / "summary.json").write_text(json.dumps(summary))
        artifacts = {
            str(path.relative_to(run)): auditor.digest(path)
            for path in run.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", "artifact-hashes.json"}
        }
        (run / "artifact-hashes.json").write_text(json.dumps(artifacts))
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "summary_sha256": auditor.digest(run / "summary.json"),
                    "artifacts_sha256": auditor.digest(run / "artifact-hashes.json"),
                    "index_sha256": auditor.digest(index_path),
                    "plan_sha256": auditor.digest(plan_path),
                }
            )
        )

    retain()
    receipt = auditor.audit(run, index_path)
    assert receipt["recomputed_score_rows"] == 9
    assert receipt["maximum_r2_absolute_error"] < 1e-12
    with pytest.raises(ValueError, match="no retained plan"):
        auditor.retained_plan(run, "0" * 64)
    summary["main"][0]["scores"]["r2_mean"] -= 0.01
    retain()  # Hashes now agree with a false scalar; independent recomputation must fail.
    with pytest.raises(AssertionError):
        auditor.audit(run, index_path)
