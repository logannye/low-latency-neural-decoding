"""Train/export/replay plumbing and scientific boundary tests on synthetic arrays."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_decoder import train_temporal
from neural_decoder.metrics import FileEvaluation, score_files
from neural_decoder.temporal import CausalTCN, TCNStream, export_tcn, load_tcn


@pytest.fixture(autouse=True)
def bounded_training_runtime():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    rng_state = torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.set_rng_state(rng_state)


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[2] / "neural_experiments/configs/tcn-v1.json"
    return json.loads(path.read_text())


def _file(length, offset=0):
    rng = np.random.default_rng(400 + offset)
    x = rng.poisson(2, (length, 176)).astype(np.uint32)
    # These distinct values let the independent audit recover original row/file identity.
    x[:, 0] = np.arange(length) + offset
    y = np.arange(length)[:, None] + offset + np.arange(7)[None] / 8
    mask = np.ones(length, dtype=bool)
    return x, y, mask


def test_chunks_preserve_each_eligible_loss_row_and_whole_file_predictions(config):
    config = {**config, "chunk_scored_bins": 7}
    first, second = _file(41), _file(29, 100)
    first[2][21:28] = False  # An entire unscored loss chunk must not create phantom rows.
    second[2][[15, 27]] = False
    training = [first, second]
    transforms, (features, targets, selected) = train_temporal.training_arrays(training, config)
    assert features.dtype == targets.dtype == torch.float32
    assert selected.dtype == torch.bool
    assert not selected[:, :14].any()
    assert features.shape[1:] == (21, 176)

    target_ids = (
        targets[:, :, 0].numpy() * transforms["target_scale"][0] + transforms["target_mean"][0]
    )
    actual_ids = np.rint(target_ids[selected.numpy()]).astype(int)
    expected_ids = np.r_[
        np.arange(41)[first[2] & (np.arange(41) >= 14)],
        (100 + np.arange(29))[second[2] & (np.arange(29) >= 14)],
    ]
    np.testing.assert_array_equal(np.sort(actual_ids), np.sort(expected_ids))
    assert len(actual_ids) == len(set(actual_ids))

    torch.manual_seed(32)
    model = CausalTCN().eval()
    full = {}
    with torch.inference_mode():
        chunk_predictions = model(features).numpy()
        for raw, _, _ in training:
            standardized = raw.astype(np.float32) - transforms["input_mean"]
            standardized /= transforms["input_scale"]
            prediction = model(torch.from_numpy(standardized)[None])[0].numpy()
            full.update(zip(raw[:, 0], prediction, strict=True))
    independent = np.array([full[key] for key in actual_ids])
    np.testing.assert_allclose(
        chunk_predictions[selected.numpy()], independent, atol=2e-5, rtol=2e-5
    )


def test_right_padding_and_future_chunk_values_cannot_change_prior_outputs(config):
    config = {**config, "chunk_scored_bins": 11}
    transforms, (features, _, selected) = train_temporal.training_arrays([_file(29)], config)
    assert transforms["input_mean"].shape == (176,)
    modified = features.clone()
    # Last chunk has four real loss bins. All later bins are synthetic right padding.
    last_real = 14 + 4
    assert not selected[-1, last_real:].any()
    modified[-1, last_real:] = 1000
    torch.manual_seed(20)
    model = CausalTCN().eval()
    with torch.inference_mode():
        expected = model(features)
        actual = model(modified)
    torch.testing.assert_close(expected[selected], actual[selected], rtol=0, atol=0)
    # With fitted transforms fixed, a changed suffix cannot affect its causal prefix.
    changed_suffix = features.clone()
    changed_suffix[:, 17:] = -1000
    with torch.inference_mode():
        torch.testing.assert_close(model(changed_suffix)[:, :17], expected[:, :17], rtol=0, atol=0)


def test_normalization_uses_training_features_and_only_masked_training_targets(config):
    first, second = _file(31), _file(28, 100)
    first[2][::3] = False
    second[2][::4] = False
    transforms, _ = train_temporal.training_arrays([first, second], config)
    x_all = np.r_[first[0], second[0]].astype(np.float64)
    y_all = np.r_[first[1][first[2]], second[1][second[2]]]
    np.testing.assert_array_equal(transforms["input_mean"], x_all.mean(0).astype(np.float32))
    np.testing.assert_array_equal(transforms["input_scale"], x_all.std(0).astype(np.float32))
    np.testing.assert_array_equal(transforms["target_mean"], y_all.mean(0).astype(np.float32))
    np.testing.assert_array_equal(transforms["target_scale"], y_all.std(0).astype(np.float32))
    first[1][~first[2]] = 1e20
    second[1][~second[2]] = -1e20
    changed, _ = train_temporal.training_arrays([first, second], config)
    for name in transforms:
        np.testing.assert_array_equal(transforms[name], changed[name])


def test_twenty_epoch_synthetic_train_export_replay_and_first_bins_scored(tmp_path, config):
    training = [_file(34), _file(37, 100)]
    # One tiny batch per epoch exercises the actual optimizer and final-epoch export.
    config = {**config, "chunk_scored_bins": 32, "batch_chunks": 16}
    assert config["epochs"] == 20
    model, transforms, losses = train_temporal.fit_temporal(training, config, seed=0, epochs=20)
    assert [entry["epoch"] for entry in losses] == list(range(1, 21))
    values = np.array([entry["masked_standardized_mse"] for entry in losses])
    assert np.isfinite(values).all()
    assert values[-1] < values[0]
    path = tmp_path / "synthetic.npz"
    export_tcn(model, path, **transforms)
    frozen = load_tcn(path)
    evaluation = [({"file_id": "synthetic-S2", "day": "S2"}, *training[0])]
    records, parity = train_temporal.evaluate_temporal(model, frozen, evaluation, tmp_path)
    assert parity["synthetic-S2"] < 3e-5
    assert records[0].predictions.shape == (34, 7)
    # First 14 bins are excluded only from training loss, not evaluation or replay.
    assert score_files(records)["num_scored_bins"] == 34
    incremental = TCNStream(frozen).predict_chunk(training[0][0])
    window = TCNStream(frozen, "window").predict_chunk(training[0][0])
    np.testing.assert_array_equal(records[0].predictions, incremental)
    np.testing.assert_array_equal(np.load(tmp_path / "synthetic-S2.npy"), incremental)
    np.testing.assert_allclose(window, incremental, rtol=3e-5, atol=3e-5)

    mutation_dir = tmp_path / "labels-changed"
    mutation_dir.mkdir()
    changed = [(evaluation[0][0], training[0][0], np.full((34, 7), 12345), np.zeros(34, bool))]
    changed_records, _ = train_temporal.evaluate_temporal(model, frozen, changed, mutation_dir)
    np.testing.assert_array_equal(changed_records[0].predictions, incremental)


@pytest.mark.parametrize("pilot", [False, True])
def test_run_uses_fixed_epochs_and_evaluates_only_after_training(
    tmp_path, monkeypatch, config, pilot
):
    files = [({"file_id": f"file-S{i}", "day": f"S{i}"}, *_file(17, 100 * i)) for i in range(5)]
    events = []
    models = {}

    def fit(training, supplied_config, seed, epochs):
        days = tuple(int(x[0, 0] // 100) for x, _, _ in training)
        events.append(("fit", seed, days, epochs))
        # Deliberately avoid optimizer work: this test verifies orchestration boundaries.
        model = CausalTCN().eval()
        models[id(model)] = (seed, days, epochs)
        return model, {}, [{"epoch": epochs, "masked_standardized_mse": 1.0}]

    def evaluate(model, frozen, evaluation, destination):
        seed, days, epochs = models[id(model)]
        assert events[-1] == ("fit", seed, days, epochs)
        assert epochs == 20
        day = evaluation[0][0]["day"]
        assert int(day[1:]) == max(days) + 1
        events.append(("evaluate", seed, day))
        records = []
        for item, _, y, mask in evaluation:
            np.save(destination / f"{item['file_id']}.npy", y, allow_pickle=False)
            records.append(FileEvaluation(item["file_id"], item["day"], y, y, mask))
        return records, {record.file_id: 0.0 for record in records}

    monkeypatch.setattr(train_temporal, "fit_temporal", fit)
    monkeypatch.setattr(train_temporal, "evaluate_temporal", evaluate)
    monkeypatch.setattr(train_temporal, "load_development_files", lambda _: ({}, files))
    monkeypatch.setattr(train_temporal, "environment", lambda: {"synthetic": True})
    monkeypatch.setattr(train_temporal, "source_identity", lambda *_: {"synthetic": True})
    index_path, config_path = tmp_path / "index.json", tmp_path / "config.json"
    index_path.write_text("{}")
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    train_temporal.run_temporal(index_path, config_path, output, root=tmp_path, pilot=pilot)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    if pilot:
        assert events == [("fit", 0, (0, 1), 1)]
    else:
        assert events == [
            event
            for seed in (0, 1, 2)
            for days, day in (((0, 1), "S2"), ((0, 1, 2), "S3"), ((0, 1, 2, 3), "S4"))
            for event in (("fit", seed, days, 20), ("evaluate", seed, day))
        ]
    assert json.loads((output / "summary.json").read_text())["sealed_days_scored"] == []
