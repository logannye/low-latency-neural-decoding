"""Synthetic end-to-end guards at the training and replay entry points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_decoder import data, replay
from neural_decoder.features import fir_weights
from neural_decoder.models import LinearDecoder, LinearModel
from neural_decoder.train import checked_development_index


def _synthetic_stream(tmp_path: Path):
    rng = np.random.default_rng(1203)
    counts = rng.poisson(2, size=(37, 4)).astype(np.uint32)
    features = {
        "counts": counts,
        "timestamps": np.arange(len(counts)) * 0.02,
        "sequence_id": np.arange(len(counts), dtype=np.int64),
        "channel_ids": np.arange(counts.shape[1], dtype=np.int64),
    }
    feature_path = tmp_path / "features.npz"
    np.savez(feature_path, **features)
    model = LinearModel(
        mean=np.arange(4) * 0.1,
        scale=np.arange(1, 5),
        weights=fir_weights(taps=3, tau_bins=2),
        coefficient=np.arange(24).reshape(12, 2) * 0.01,
        intercept=np.array([0.2, -0.3]),
        history=3,
    )
    model_path = tmp_path / "model.npz"
    model.save(model_path)
    return model_path, feature_path, features


def test_training_rejects_relabelled_sealed_asset_before_any_arrays_load(
    tmp_path: Path, monkeypatch
) -> None:
    """The index claims five legal days; trusted source identity exposes S5."""
    upstream = tmp_path / "upstream/falcon_challenge"
    upstream.mkdir(parents=True)
    mapping = {f"synthetic_session_{i}": f"S{i}_set_1" for i in range(1, 6)}
    config = upstream / "config.py"
    config.write_text(f"H1_NEW_TO_OLD = {mapping!r}\n")
    monkeypatch.setitem(data.SOURCE_SHA256, "falcon_challenge/config.py", data.sha256_file(config))
    assets, records = [], []
    # S1-S4 are honestly labelled; the asset from S5 is presented as S0.
    for source_day in [5, 1, 2, 3, 4]:
        claimed_day = 0 if source_day == 5 else source_day
        asset_id = f"synthetic-asset-S{source_day}"
        assets.append(
            {
                "asset_id": asset_id,
                "sha256": f"raw-hash-{source_day}",
                "path": "sub-HumanPitt-held-in-calib/"
                f"sub-HumanPitt-held-in-calib_ses-synthetic_session_{source_day}.nwb",
            }
        )
        records.append(
            {
                "asset_id": asset_id,
                "day": f"S{claimed_day}",
                "file_id": f"S{claimed_day}_set_1",
                "provider_split": "held-in-calib",
                "raw_sha256": f"raw-hash-{source_day}",
                "features": f"features/S{claimed_day}_set_1.npz",
                "targets": f"targets/S{claimed_day}_set_1.npz",
                "features_sha256": "sample-files-must-not-be-opened",
                "targets_sha256": "sample-files-must-not-be-opened",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": assets}))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    index_path = prepared / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "development",
                "upstream_pin": data.UPSTREAM_PIN,
                "source_manifest_sha256": data.sha256_file(manifest),
                "files": records,
            }
        )
    )

    def forbid_array_load(*args, **kwargs):
        pytest.fail("Sample arrays were opened before checking source-day authorization")

    monkeypatch.setattr(np, "load", forbid_array_load)
    with pytest.raises(ValueError, match="Source role/identity mismatch"):
        checked_development_index(index_path)
    # No sample files exist: even hashing one before role validation would fail.
    assert not (prepared / "features").exists()
    assert not (prepared / "targets").exists()


@pytest.mark.parametrize("corruption", ["fractional_sequence_ids", "disordered_channels"])
def test_replay_rejects_invalid_feature_metadata_before_writing_predictions(
    tmp_path: Path, corruption: str
) -> None:
    model, feature_path, features = _synthetic_stream(tmp_path)
    if corruption == "fractional_sequence_ids":
        features["sequence_id"] = features["sequence_id"].astype(float) + 0.5
        expected_error = "sequence ID"
    else:
        features["channel_ids"] = np.array([1, 0, 2, 3])
        expected_error = "channel map"
    np.savez(feature_path, **features)
    output = tmp_path / "predictions.npy"
    with pytest.raises(ValueError, match=expected_error):
        replay.replay_features(model, feature_path, output)
    assert not output.exists()


@pytest.mark.parametrize("changed_field", ["targets", "eval_mask", "trial_change"])
def test_sidefile_intervention_leaves_saved_replay_and_final_state_exactly_unchanged(
    tmp_path: Path, monkeypatch, changed_field: str
) -> None:
    model, feature_path, features = _synthetic_stream(tmp_path)
    count = len(features["counts"])
    target_path = tmp_path / "targets.npz"
    targets = {
        "targets": np.linspace(-2, 2, count * 2).reshape(count, 2),
        "eval_mask": np.arange(count) % 3 != 0,
        "trial_change": np.arange(count) % 7 == 0,
    }
    np.savez(target_path, **targets)
    model_sha256 = data.sha256_file(model)
    feature_sha256 = data.sha256_file(feature_path)
    original_load = np.load
    opened = []

    def guarded_load(path, *args, **kwargs):
        resolved = Path(path).resolve()
        opened.append(resolved)
        if resolved == target_path.resolve():
            pytest.fail("Feature-only replay attempted to open a scorer-only sidefile")
        return original_load(path, *args, **kwargs)

    instances = []

    class ObservedDecoder(LinearDecoder):
        def __init__(self, fitted):
            super().__init__(fitted)
            instances.append(self)

    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(replay, "LinearDecoder", ObservedDecoder)
    first_output, second_output = tmp_path / "first.npy", tmp_path / "second.npy"
    first = replay.replay_features(model, feature_path, first_output)
    if changed_field == "targets":
        targets[changed_field] = -targets[changed_field] + 1e9
    else:
        targets[changed_field] = ~targets[changed_field]
    np.savez(target_path, **targets)
    second = replay.replay_features(model, feature_path, second_output)

    np.testing.assert_array_equal(first, second)
    assert first_output.read_bytes() == second_output.read_bytes()
    assert len(instances) == 2
    for attribute in ("raw", "past"):
        np.testing.assert_array_equal(
            getattr(instances[0].features, attribute), getattr(instances[1].features, attribute)
        )
    assert instances[0].features.next_sequence_id == instances[1].features.next_sequence_id == count
    assert set(opened) == {model.resolve(), feature_path.resolve()}
    assert data.sha256_file(model) == model_sha256
    assert data.sha256_file(feature_path) == feature_sha256
