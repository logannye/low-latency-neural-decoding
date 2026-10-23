"""V7 exposed-prefix provenance and feature/label separation contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from neural_experiments.v7 import data


@pytest.fixture
def parents(tmp_path, monkeypatch):
    """Tiny synthetic artifacts, with the real 128+24 participant inventory."""
    paths = []
    for role in ("development", "new_sessions"):
        root = tmp_path / role
        root.mkdir()
        feature = root / "features.npz"
        target = root / "targets.npz"
        np.savez(
            feature,
            x_tx3=np.ones((3, 96), dtype=np.float32),
            x_tx45=np.ones((3, 96), dtype=np.float32),
            x_sbp=np.ones((3, 96), dtype=np.float32),
            timestamps=np.arange(1, 4) * 0.02,
            sequence_id=np.zeros(3, dtype=np.int64),
            channel_ids=np.arange(96, dtype=np.int64),
        )
        np.savez(
            target,
            y=np.tile(np.array([1.0, 0.0], dtype=np.float32), (3, 1)),
            eval_mask=np.ones(3, dtype=bool),
        )
        rows = []
        for participant in data.ALL_PARTICIPANTS:
            days = (
                range(8 if participant == "T3" else 15)
                if role == "development"
                else range(15, 18)
                if participant != "T3"
                else []
            )
            for day in days:
                session_id = f"{participant}_day_{day}_decoding"
                rows.append(
                    {
                        "session_id": session_id,
                        "participant_id": participant,
                        "post_implant_day": day,
                        "role": role,
                        "n_bins": 3,
                        "full_prepared_n_bins": 3,
                        "n_channels": 96,
                        "bin_seconds": 0.02,
                        "source": {"raw_path": "MUST_NOT_BE_OPENED.mat"},
                        "raw_source_root": str(tmp_path / "missing-raw-root"),
                        "feature_path": feature.name,
                        "target_path": target.name,
                        "feature_sha256": data.v2.sha256_file(feature),
                        "target_sha256": data.v2.sha256_file(target),
                        "segments": [
                            {
                                "sequence_id": 0,
                                "start": 0,
                                "end": 3,
                                "global_start": 0,
                                "global_end": 3,
                            }
                        ],
                    }
                )
        index = {
            "schema_version": 1,
            "role": role,
            "provider": {"doi": data.v2.DOI, "version_id": data.v2.VERSION_ID},
            "feature_definition": data.v6.FEATURE_DEFINITION,
            "sessions": rows,
        }
        path = root / "index.json"
        path.write_text(json.dumps(index))
        paths.append(path)
    monkeypatch.setattr(data, "V6_DEVELOPMENT_INDEX_SHA256", data.v2.sha256_file(paths[0]))
    monkeypatch.setattr(data, "V6_NEW_INDEX_SHA256", data.v2.sha256_file(paths[1]))
    return paths


def test_prepare_exact_exposed_union_never_needs_raw_sources(tmp_path, parents):
    destination = tmp_path / "v7"
    index = data.prepare_development(destination, *parents)
    loaded = data.load_index(destination / "index.json", expected_role="development")
    assert index == loaded
    assert len(index["sessions"]) == 152
    assert all(s["role"] == "development" for s in index["sessions"])
    assert {
        p: sum(s["participant_id"] == p for s in index["sessions"]) for p in data.ALL_PARTICIPANTS
    } == data.EXPECTED_COUNTS
    assert index["exposure"]["fresh_data_included"] is False
    assert index["exposure"]["continuation_tails_included"] is False
    for item in index["sessions"]:
        for kind in ("feature", "target"):
            original = Path(item["reused_parent_artifacts"][kind]["path"])
            linked = destination / item[kind + "_path"]
            assert linked.stat().st_ino == original.stat().st_ino


def test_default_case_loading_never_opens_targets(tmp_path, parents, monkeypatch):
    destination = tmp_path / "v7"
    item = data.prepare_development(destination, *parents)["sessions"][0]
    target_path = destination / item["target_path"]
    target_path.unlink()
    monkeypatch.setattr(data, "load_targets", lambda *_: pytest.fail("target loader called"))
    case = data.load_case(destination / "index.json", item)
    assert set(case) == {"item", "features", "segments"}
    assert set(case["features"]) == data.FEATURE_KEYS
    assert "y" not in case and "eval_mask" not in case


def test_explicit_target_request_is_separate(tmp_path, parents):
    destination = tmp_path / "v7"
    item = data.prepare_development(destination, *parents)["sessions"][0]
    case = data.load_case(destination / "index.json", item, labels=True)
    assert set(case) == {"item", "features", "segments", "y", "eval_mask"}
    assert case["y"].shape == (3, 2)


def test_changed_parent_refused_before_output_creation(tmp_path, parents):
    parents[0].write_text(parents[0].read_text() + " ")
    destination = tmp_path / "v7"
    with pytest.raises(ValueError, match="locked exposed cohort"):
        data.prepare_development(destination, *parents)
    assert not destination.exists()


def test_prepared_index_is_immutable(tmp_path, parents):
    destination = tmp_path / "v7"
    data.prepare_development(destination, *parents)
    with pytest.raises(ValueError, match="immutable"):
        data.prepare_development(destination, *parents)


def test_tampered_manifest_record_rejected_even_without_array_validation(tmp_path, parents):
    destination = tmp_path / "v7"
    index = data.prepare_development(destination, *parents)
    index["sessions"][0]["n_bins"] += 1
    (destination / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="immutable V6 parent"):
        data.load_index(destination / "index.json", verify_hashes=False)


def test_wrong_role_rejected_before_references(tmp_path):
    index = tmp_path / "index.json"
    index.write_text('{"role": "new_sessions"}')
    with pytest.raises(ValueError, match="role"):
        data.load_index(index)


def test_changed_feature_hash_rejected_before_loading(tmp_path, parents):
    destination = tmp_path / "v7"
    item = data.prepare_development(destination, *parents)["sessions"][0]
    path = destination / item["feature_path"]
    path.unlink()
    path.write_bytes(b"not a feature archive")
    with pytest.raises(ValueError, match="Feature identity"):
        data.load_case(destination / "index.json", item)
    with pytest.raises(ValueError, match="artifact hash"):
        data.load_index(destination / "index.json")


def test_changed_target_never_changes_feature_only_results(tmp_path, parents):
    destination = tmp_path / "v7"
    item = data.prepare_development(destination, *parents)["sessions"][0]
    before = data.load_case(destination / "index.json", item)
    path = destination / item["target_path"]
    path.unlink()
    path.write_bytes(b"not a target archive")
    after = data.load_case(destination / "index.json", item)
    for key in data.FEATURE_KEYS:
        np.testing.assert_array_equal(before["features"][key], after["features"][key])
    with pytest.raises(ValueError, match="Target identity"):
        data.load_case(destination / "index.json", item, labels=True)


def test_geometry_claim_checked_against_features(tmp_path, parents):
    destination = tmp_path / "v7"
    item = copy.deepcopy(data.prepare_development(destination, *parents)["sessions"][0])
    item["n_bins"] = 4
    with pytest.raises(ValueError, match="geometry"):
        data.load_case(destination / "index.json", item)


@pytest.mark.parametrize("violation", ["too_long", "overlap", "duplicate", "path_escape"])
def test_invalid_reparenting_refused(tmp_path, parents, monkeypatch, violation):
    index = json.loads(parents[0].read_text())
    row = index["sessions"][0]
    if violation == "too_long":
        row["n_bins"] = row["full_prepared_n_bins"] = 30001
    elif violation == "overlap":
        row["segments"][0]["start"] = 1
    elif violation == "duplicate":
        index["sessions"][1]["session_id"] = row["session_id"]
    else:
        row["feature_path"] = "../outside.npz"
    parents[0].write_text(json.dumps(index))
    monkeypatch.setattr(data, "V6_DEVELOPMENT_INDEX_SHA256", data.v2.sha256_file(parents[0]))
    with pytest.raises(ValueError):
        data.prepare_development(tmp_path / "v7", *parents)


def test_known_parent_hashes_are_fixed():
    assert (
        data.V6_DEVELOPMENT_INDEX_SHA256
        == "357f8b917abec17f6354dac3cd93b4d87d24df0db4bf60afeee8bdbefbf84678"
    )
    assert (
        data.V6_NEW_INDEX_SHA256
        == "dfe5ed95bf7f43b0bb51f3a11d163505e99641a4affc86daf38591fbdc5af534"
    )
