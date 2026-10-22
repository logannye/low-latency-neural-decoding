"""Synthetic ingestion contract tests: no human recording downloads required."""

from __future__ import annotations

import copy
import io
import json
import tarfile

import numpy as np
import pytest
from scipy.io import savemat

from neural_experiments.v2 import data


def session(participant="T2", day=100):
    counts = np.zeros((12, 96), dtype=np.uint16)
    counts[:, 0] = np.arange(12)
    counts[:, 1] = 250  # sum must not overflow uint8
    cursor = np.zeros((12, 2))
    target = np.tile([3.0, 4.0], (12, 1))
    target[1] = [0.0, 1.0]
    target[3] = [0.0, 0.0]
    target[9] = [np.nan, 1.0]
    return {
        "participant_id": participant,
        "post_implant_day": day,
        "bin_size": 0.01,
        "electrodes": {"electrode_id": np.arange(96)},
        "neural": {"tx_4_5": counts},
        "cursor_position": cursor,
        "target_position": target,
        "blocks": {"block_id": [5, 9], "sample_start_index": [0, 7], "sample_end_index": [5, 12]},
        "trials": {
            "trial_start_index": [0, 2, 7, 10],
            "trial_end_index": [2, 5, 10, 12],
            "block_id": [5, 5, 9, 9],
        },
    }


def archive_bytes(rows):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as tar:
        for name, value in rows:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            tar.addfile(info, io.BytesIO(value))
    return result.getvalue()


def test_pairing_target_endpoint_go_mask_and_block_resets():
    features, targets, metadata = data.convert_session(session(), "T2", 100)
    assert set(features) == data.FEATURE_KEYS
    assert set(targets) == data.TARGET_KEYS
    np.testing.assert_array_equal(features["x"][:, 0], [1, 5, 15, 19])
    np.testing.assert_array_equal(features["x"][:, 1], [500, 500, 500, 500])
    np.testing.assert_array_equal(features["sequence_id"], [0, 0, 1, 1])
    np.testing.assert_allclose(features["timestamps"], [0.02, 0.04, 0.09, 0.11])
    np.testing.assert_allclose(targets["y"][0], [0, 1])
    np.testing.assert_allclose(targets["y"][2:], [[0.6, 0.8], [0.6, 0.8]])
    # Last pair straddles two different go intervals; both samples are go, but not one trial.
    np.testing.assert_array_equal(targets["eval_mask"], [True, False, True, False])
    assert metadata["discarded_trailing_10ms_bins"] == 2
    assert metadata["discarded_unblocked_10ms_bins"] == 2
    assert metadata["segments"][1]["source_gap_before_bins"] == 2
    assert metadata["segments"][1]["start"] == 2


def test_targets_and_trial_mutation_cannot_change_features():
    raw = session()
    first, _, _ = data.convert_session(raw, "T2", 100)
    raw["target_position"] *= 7
    raw["cursor_position"] += 3
    raw["trials"] = {"trial_start_index": [], "trial_end_index": [], "block_id": []}
    second, targets, metadata = data.convert_session(raw, "T2", 100)
    for key in data.FEATURE_KEYS:
        np.testing.assert_array_equal(first[key], second[key])
    assert not targets["eval_mask"].any()
    assert metadata["n_eval_bins"] == 0


def test_source_future_mutation_cannot_change_earlier_output():
    raw = session()
    before, target_before, _ = data.convert_session(raw, "T2", 100)
    raw["neural"]["tx_4_5"][7:] *= 3
    raw["target_position"][7:] = [1, 0]
    after, target_after, _ = data.convert_session(raw, "T2", 100)
    np.testing.assert_array_equal(before["x"][:2], after["x"][:2])
    np.testing.assert_array_equal(target_before["y"][:2], target_after["y"][:2])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(participant_id="T3"),
        lambda r: r.update(post_implant_day=100.5),
        lambda r: r.update(bin_size=0.02),
        lambda r: r["neural"].pop("tx_4_5"),
        lambda r: r["neural"].update(tx_4_5=np.full((12, 96), 0.5)),
        lambda r: r["electrodes"].update(electrode_id=np.arange(96)[::-1]),
        lambda r: r["blocks"].update(sample_start_index=[0, 3]),
        lambda r: r["trials"].update(trial_end_index=[3, 5, 10, 12]),
        lambda r: r["trials"].update(block_id=[9, 5, 9, 9]),
    ],
)
def test_malformed_raw_schema_fails_closed(mutation):
    raw = session()
    mutation(raw)
    with pytest.raises(ValueError):
        data.convert_session(raw, "T2", 100)


def test_short_and_zero_go_session_retained():
    raw = session()
    raw["blocks"] = {"block_id": [5], "sample_start_index": [0], "sample_end_index": [1]}
    raw["trials"] = {"trial_start_index": [], "trial_end_index": [], "block_id": []}
    features, targets, metadata = data.convert_session(raw, "T2", 100)
    assert features["x"].shape == (0, 96)
    assert targets["y"].shape == (0, 2)
    assert metadata["n_bins"] == metadata["n_eval_bins"] == 0
    assert metadata["discarded_trailing_10ms_bins"] == 1


def test_archive_order_preserved_and_stops_before_later_bad_member(tmp_path):
    payload = archive_bytes(
        [
            ("decoding/T2/T2_day_99_decoding.mat", b"ninety-nine"),
            ("decoding/T2/T2_day_1_decoding.mat", b"one"),
            ("../../unsafe.mat", b"must never reach third member"),
        ]
    )
    receipt = data.extract_first_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", sessions=2
    )
    assert [r["post_implant_day"] for r in receipt["selected"]] == [99, 1]
    assert receipt["status"] == "complete"
    assert not receipt["full_archive_sha256_verified"]
    assert not receipt["tar_end_observed"]
    assert (tmp_path / "raw/T2/T2_day_1_decoding.mat").read_bytes() == b"one"
    for row in receipt["selected"]:
        assert data.sha256_file(tmp_path / row["raw_path"]) == row["member_sha256"]


@pytest.mark.parametrize(
    "name", ["../../unsafe.mat", "/tmp/unsafe.mat", "decoding/T3/T3_day_1_decoding.mat"]
)
def test_unsafe_or_wrong_person_archive_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        data.extract_first_sessions(
            io.BytesIO(archive_bytes([(name, b"x")])), cache=tmp_path, participant="T2", sessions=1
        )
    receipt = json.loads((tmp_path / "acquisition/T2.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["selected"] == []
    assert not receipt["full_archive_sha256_verified"]


def test_disk_budget_fails_before_session_copy(tmp_path):
    with pytest.raises(ValueError, match="disk budget"):
        data.extract_first_sessions(
            io.BytesIO(archive_bytes([("decoding/T2/T2_day_1_decoding.mat", b"12345")])),
            cache=tmp_path,
            participant="T2",
            sessions=1,
            max_cache_bytes=3,
        )
    assert not list((tmp_path / "raw/T2").glob("*.mat"))


def test_compressed_prefix_budget_is_bounded_and_records_failure(tmp_path):
    payload = archive_bytes([("decoding/T2/T2_day_1_decoding.mat", b"12345")])
    with pytest.raises((ValueError, tarfile.ReadError)):
        data.extract_first_sessions(
            io.BytesIO(payload), cache=tmp_path, participant="T2", sessions=1, max_stream_bytes=10
        )
    receipt = json.loads((tmp_path / "acquisition/T2.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["compressed_prefix_bytes"] <= 10


def complete_cache(tmp_path):
    manifest = {"doi": data.DOI, "version_id": data.VERSION_ID}
    data._json(tmp_path / "provider-manifest.json", manifest)
    for participant in data.PARTICIPANTS:
        buffer = io.BytesIO()
        savemat(buffer, session(participant, 100))
        file_id, size, digest = data.PINS[participant]
        data.extract_first_sessions(
            io.BytesIO(
                archive_bytes(
                    [
                        (
                            f"decoding/{participant}/{participant}_day_100_decoding.mat",
                            buffer.getvalue(),
                        )
                    ]
                )
            ),
            cache=tmp_path,
            participant=participant,
            sessions=1,
            archive={"file_id": file_id, "size": size, "provider_sha256": digest},
        )
    return data.prepare(tmp_path)


def test_complete_nine_person_fixture_is_label_separated_and_hash_verified(tmp_path):
    complete_cache(tmp_path)
    index = data.load_index(tmp_path / "index.json")
    assert index["participants"] == list(data.PARTICIPANTS)
    assert len(index["sessions"]) == 9
    for record in index["sessions"]:
        features = data.load_features(tmp_path / record["feature_path"])
        assert set(features) == data.FEATURE_KEYS
    side = tmp_path / index["sessions"][0]["target_path"]
    side.write_bytes(side.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="target artifact hash"):
        data.load_index(tmp_path / "index.json")


def test_index_provider_forgery_rejected_before_array_open(tmp_path, monkeypatch):
    index = complete_cache(tmp_path)
    index["sessions"][0]["source"]["archive_file_id"] = data.PINS["T3"][0]
    data._json(tmp_path / "index.json", index)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("opened array before identity gate")
    )
    with pytest.raises(ValueError, match="provider identity"):
        data.load_index(tmp_path / "index.json")


def test_index_segments_must_partition_and_all_humans_retained(tmp_path):
    index = complete_cache(tmp_path)
    changed = copy.deepcopy(index)
    changed["sessions"][0]["segments"][1]["start"] = 0
    data._json(tmp_path / "index.json", changed)
    with pytest.raises(ValueError, match="partition"):
        data.load_index(tmp_path / "index.json", verify_hashes=False)
    index["sessions"] = index["sessions"][:-1]
    data._json(tmp_path / "index.json", index)
    with pytest.raises(ValueError, match="participant"):
        data.load_index(tmp_path / "index.json", verify_hashes=False)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a.update(eval_mask=np.ones(4, dtype=bool)),
        lambda a: a.update(sequence_id=a["sequence_id"].astype(float) + 0.5),
        lambda a: a.update(channel_ids=a["channel_ids"][::-1]),
        lambda a: a.update(timestamps=np.array([0.02, 0.05, 0.09, 0.11])),
    ],
)
def test_inference_loader_rejects_labels_disorder_and_unmarked_gap(tmp_path, mutation):
    features, _, _ = data.convert_session(session(), "T2", 100)
    mutation(features)
    path = tmp_path / "features.npz"
    np.savez(path, **features)
    with pytest.raises(ValueError):
        data.load_features(path)


def test_receipt_selection_cannot_be_replaced_by_later_session(tmp_path):
    complete_cache(tmp_path)
    receipt_path = tmp_path / "acquisition/T2.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["selected"][0]["archive_ordinal"] += 1
    with pytest.raises(ValueError, match="provenance"):
        data.validate_receipt(receipt, "T2")


def test_index_raw_hash_cannot_be_forged_independently_of_receipt(tmp_path):
    index = complete_cache(tmp_path)
    index["sessions"][0]["source"]["member_sha256"] = "0" * 64
    data._json(tmp_path / "index.json", index)
    with pytest.raises(ValueError, match="differs from acquisition receipt"):
        data.load_index(tmp_path / "index.json", verify_hashes=False)


def test_prediction_features_identical_after_target_sidefile_mutation(tmp_path):
    index = complete_cache(tmp_path)
    row = index["sessions"][0]
    feature_path = tmp_path / row["feature_path"]
    before = data.load_features(feature_path)
    target_path = tmp_path / row["target_path"]
    np.savez(target_path, y=np.zeros((4, 2), dtype=np.float32), eval_mask=np.zeros(4, dtype=bool))
    after = data.load_features(feature_path)
    for key in data.FEATURE_KEYS:
        np.testing.assert_array_equal(before[key], after[key])
    assert data.sha256_file(feature_path) == row["feature_sha256"]


def test_scorer_loader_rejects_nonunit_scored_direction(tmp_path):
    path = tmp_path / "targets.npz"
    np.savez(path, y=np.ones((2, 2), dtype=np.float32), eval_mask=np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match="not unit"):
        data.load_targets(path, n_bins=2)


def test_observed_redirect_must_match_participant_and_failure_hides_query(tmp_path, monkeypatch):
    participant = "T2"
    file_id, size, digest = data.PINS[participant]
    archive = {
        "participant_id": participant,
        "file_id": file_id,
        "size": size,
        "provider_sha256": digest,
        "filename": "decoding_T2.tar.gz",
        "download_url": "https://datadryad.org/downloads/file_stream/4952113",
    }
    monkeypatch.setattr(data, "pin_metadata", lambda cache: {"archives": [archive]})
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: pytest.fail("unverified redirect opened"))
    private = tmp_path / "private.json"
    private.write_text(json.dumps({"T2": "https://untrusted.example/data?secret=not-for-logs"}))
    with pytest.raises(ValueError, match="observed matching"):
        data.acquire(tmp_path, participants=("T2",), url_file=private)
    failure = next((tmp_path / "failures").glob("*.json")).read_text()
    assert "not-for-logs" not in failure
    assert "untrusted.example" not in failure
