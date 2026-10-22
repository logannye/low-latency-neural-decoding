from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_decoder.data import (
    FEATURE_KEYS,
    bin_spikes,
    load_feature_file,
    load_target_file,
    sha256_file,
)


def test_histogram_endpoints_short_bins_and_discarded_gap() -> None:
    stamps = np.array([0.02, 0.04, 0.08, 0.09])
    spikes = np.array([0, 0.019, 0.02, 0.039, 0.04, 0.05, 0.06, 0.079, 0.08, 0.09])
    # [0,.02), [.02,.04), discard [.04,.06), [.06,.08), [.08,.09]
    np.testing.assert_array_equal(bin_spikes([spikes], stamps).ravel(), [2, 2, 2, 2])


def test_counts_do_not_silently_wrap_at_256() -> None:
    counts = bin_spikes([np.repeat(0.01, 300)], np.array([0.02]))
    assert counts.dtype == np.uint32
    assert counts[0, 0] == 300


@pytest.mark.parametrize("timestamps", [[0, 0], [0.04, 0.02], [0, np.nan], []])
def test_invalid_timestamps_fail(timestamps: list[float]) -> None:
    with pytest.raises(ValueError):
        bin_spikes([np.array([0.01])], np.array(timestamps))


def _features() -> dict[str, np.ndarray]:
    return {
        "counts": np.zeros((3, 2), dtype=np.uint32),
        "timestamps": np.arange(3) * 0.02,
        "sequence_id": np.arange(3),
        "channel_ids": np.arange(2),
    }


def test_feature_export_rejects_any_label_or_mask(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    for key in ["targets", "eval_mask", "trial_change"]:
        np.savez(path, **_features(), **{key: np.ones(3)})
        with pytest.raises(ValueError, match="information boundary"):
            load_feature_file(path)


def test_duplicate_sequence_and_channel_ids_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    for key, value in [("sequence_id", [0, 1, 1]), ("channel_ids", [0, 0])]:
        arrays = _features()
        arrays[key] = np.array(value)
        np.savez(path, **arrays)
        with pytest.raises(ValueError):
            load_feature_file(path)


def test_hash_change_invalidates_prepared_evidence(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    arrays = _features()
    np.savez(path, **arrays)
    digest = sha256_file(path)
    assert set(load_feature_file(path, digest)) == FEATURE_KEYS
    arrays["counts"][0, 0] = 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="identity changed"):
        load_feature_file(path, digest)


def test_targets_are_separate_and_checked(tmp_path: Path) -> None:
    path = tmp_path / "targets.npz"
    np.savez(
        path,
        targets=np.zeros((3, 7)),
        eval_mask=np.ones(3, dtype=bool),
        trial_change=np.array([False, True, False]),
    )
    assert load_target_file(path)["targets"].shape == (3, 7)
    with pytest.raises(ValueError, match="information boundary"):
        load_feature_file(path)


def test_read_only_real_data_manifest_when_available() -> None:
    """Opt-in local artifact check; routine CI does not require external data."""
    import os

    if "NEURAL_H1_INDEX" not in os.environ:
        pytest.skip("Set NEURAL_H1_INDEX to verify an acquired local development cache")
    from neural_decoder.data import load_index

    path = Path(os.environ["NEURAL_H1_INDEX"])
    index = load_index(path)
    assert len(index["files"]) == 11
    assert {r["day"] for r in index["files"]} == {f"S{i}" for i in range(5)}
    for row in index["files"]:
        features = load_feature_file(path.parent / row["features"], row["features_sha256"])
        targets = load_target_file(path.parent / row["targets"], row["targets_sha256"])
        assert features["counts"].shape == (row["n_bins"], 176)
        assert targets["targets"].shape == (row["n_bins"], 7)
    audit = json.loads((path.parent.parent / "schema_audit.json").read_text())
    sealed = [r for r in audit["files"] if r["access_role"] == "sealed_metadata_only"]
    assert sealed
    assert all("count_max" not in row and "scored_bins" not in row for row in sealed)


def test_relabeling_a_sealed_asset_does_not_bypass_guard(tmp_path: Path, monkeypatch) -> None:
    from neural_decoder import data

    upstream = tmp_path / "upstream/falcon_challenge"
    upstream.mkdir(parents=True)
    config = upstream / "config.py"
    config.write_text("H1_NEW_TO_OLD = {'19250120T115044': 'S5_set_1'}\n")
    monkeypatch.setitem(data.SOURCE_SHA256, "falcon_challenge/config.py", sha256_file(config))
    asset = {
        "asset_id": "sealed-asset",
        "sha256": "raw-hash",
        "path": "sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-19250120T115044.nwb",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": [asset]}))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    index = {
        "schema_version": 1,
        "role": "development",
        "upstream_pin": data.UPSTREAM_PIN,
        "source_manifest_sha256": sha256_file(manifest),
        "files": [
            {
                "asset_id": "sealed-asset",
                "day": "S0",
                "file_id": "S0_set_1",
                "provider_split": "held-in-calib",
                "raw_sha256": "raw-hash",
                "features": "features/S0_set_1.npz",
                "targets": "targets/S0_set_1.npz",
            }
        ],
    }
    index_path = prepared / "index.json"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Source role/identity mismatch"):
        data.load_index(index_path)
