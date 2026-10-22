"""Synthetic V3 acquisition and pure-feature conversion contracts."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from neural_experiments.v3 import data


def raw():
    tx = np.tile(np.arange(12, dtype=np.float32)[:, None], (1, 96))
    return {
        "participant_id": "T2",
        "post_implant_day": 100,
        "bin_size": 0.01,
        "electrodes": {"electrode_id": np.arange(96)},
        "neural": {"tx_4_5": tx, "sbp": tx * 2},
        "cursor_position": np.zeros((12, 2)),
        "target_position": np.tile([3.0, 4.0], (12, 1)),
        "blocks": {"block_id": [1, 2], "sample_start_index": [0, 7], "sample_end_index": [5, 12]},
        "trials": {"block_id": [1, 2], "trial_start_index": [0, 7], "trial_end_index": [5, 12]},
    }


def test_pair_sum_power_mean_gap_and_cap():
    f, t, m = data.convert_session(raw(), "T2", 100, cap_bins=3)
    assert set(f) == data.FEATURE_KEYS
    assert set(t) == {"y", "eval_mask"}
    np.testing.assert_array_equal(f["x_tx"][:, 0], [1, 5, 15])
    np.testing.assert_array_equal(f["x_sbp"][:, 0], [1, 5, 15])
    np.testing.assert_array_equal(f["sequence_id"], [0, 0, 1])
    np.testing.assert_allclose(f["timestamps"], [0.02, 0.04, 0.09])
    assert m["n_bins"] == 3 and m["full_prepared_n_bins"] == 4
    assert m["full_raw_n_source_bins"] == 12
    assert m["segments"][-1]["end"] == 3
    assert m["discarded_trailing_10ms_bins"] == 1


def test_discarded_future_cannot_affect_capped_export():
    r = raw()
    before = data.convert_session(r, "T2", 100, cap_bins=2)
    r["neural"]["tx_4_5"][7:] = np.nan
    r["neural"]["sbp"][7:] = -1
    r["target_position"][7:] = np.nan
    after = data.convert_session(r, "T2", 100, cap_bins=2)
    for k in data.FEATURE_KEYS:
        np.testing.assert_array_equal(before[0][k], after[0][k])
    for k in before[1]:
        np.testing.assert_array_equal(before[1][k], after[1][k])


def test_labels_cannot_enter_features(tmp_path):
    r = raw()
    a = data.convert_session(r, "T2", 100)[0]
    r["target_position"] *= -5
    r["trials"] = {"block_id": [], "trial_start_index": [], "trial_end_index": []}
    b = data.convert_session(r, "T2", 100)[0]
    for k in data.FEATURE_KEYS:
        np.testing.assert_array_equal(a[k], b[k])
    np.savez(tmp_path / "x.npz", **a, eval_mask=np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="forbidden"):
        data.load_features(tmp_path / "x.npz")


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_exposed_power_rejected(value):
    r = raw()
    r["neural"]["sbp"][0, 0] = value
    with pytest.raises(ValueError, match="SBP"):
        data.convert_session(r, "T2", 100)


def test_power_zero_is_valid_and_channel_order_strict(tmp_path):
    r = raw()
    r["neural"]["sbp"][:] = 0
    f, _, _ = data.convert_session(r, "T2", 100)
    np.savez(tmp_path / "x.npz", **f)
    assert not data.load_features(tmp_path / "x.npz")["x_sbp"].any()
    f["channel_ids"] = f["channel_ids"][::-1]
    np.savez(tmp_path / "x.npz", **f)
    with pytest.raises(ValueError, match="channel"):
        data.load_features(tmp_path / "x.npz")


def test_empty_single_source_bin_retained():
    r = raw()
    r["blocks"] = {"block_id": [1], "sample_start_index": [0], "sample_end_index": [1]}
    r["trials"] = {"block_id": [], "trial_start_index": [], "trial_end_index": []}
    f, t, m = data.convert_session(r, "T2", 100)
    assert f["x_tx"].shape == (0, 96) and f["x_sbp"].shape == (0, 96)
    assert m["n_bins"] == 0 and t["y"].shape == (0, 2)


def tar_fixture(n=9):
    rows = [
        (f"decoding/T2/T2_day_{100 + i}_decoding.mat", f"payload{i}".encode()) for i in range(n)
    ]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tf:
        for name, payload in rows:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    prior = [
        {
            "member_name": name,
            "member_size": len(payload),
            "archive_ordinal": i,
            "session_id": name.split("/")[-1].removesuffix(".mat"),
            "member_sha256": hashlib.sha256(payload).hexdigest(),
        }
        for i, (name, payload) in enumerate(rows[:6])
    ]
    file_id, size, digest = data.v2.PINS["T2"]
    return stream.getvalue(), prior, {"file_id": file_id, "size": size, "provider_sha256": digest}


def test_extract_exact_next_three_and_authenticate_old_without_copy(tmp_path):
    payload, prior, archive = tar_fixture()
    r = data.extract_next_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", prior_members=prior, archive=archive
    )
    data.validate_receipt(r, "T2", prior)
    assert [s["regular_session_rank"] for s in r["selected"]] == [7, 8, 9]
    assert len(list((tmp_path / "raw/T2").glob("*.mat"))) == 3
    assert len(r["verified_prior_members"]) == 6
    assert not r["full_archive_sha256_verified"] and not r["tar_end_observed"]


def test_short_archive_retains_disposition_no_replacement(tmp_path):
    payload, prior, archive = tar_fixture(7)
    r = data.extract_next_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", prior_members=prior, archive=archive
    )
    data.validate_receipt(r, "T2", prior)
    assert r["missing_requested_ranks"] == [8, 9] and len(r["selected"]) == 1
    assert r["tar_end_observed"]


def test_changed_old_member_fails_before_new_data(tmp_path):
    payload, prior, archive = tar_fixture()
    prior[0]["member_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="V2 member hash"):
        data.extract_next_sessions(
            io.BytesIO(payload),
            cache=tmp_path,
            participant="T2",
            prior_members=prior,
            archive=archive,
        )
    assert not list((tmp_path / "raw/T2").glob("*.mat"))
    assert json.loads((tmp_path / "acquisition/T2.json").read_text())["status"] == "failed"


def test_forged_rank_and_shortage_rejected(tmp_path):
    payload, prior, archive = tar_fixture()
    r = data.extract_next_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", prior_members=prior, archive=archive
    )
    forged = copy.deepcopy(r)
    forged["selected"][0]["regular_session_rank"] = 6
    with pytest.raises(ValueError, match="rank"):
        data.validate_receipt(forged, "T2", prior)
    forged = copy.deepcopy(r)
    forged["missing_requested_ranks"] = [9]
    with pytest.raises(ValueError, match="shortage"):
        data.validate_receipt(forged, "T2", prior)


def test_role_gate_precedes_any_referenced_file(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"role": "assessment"}))
    with pytest.raises(ValueError, match="role"):
        data.load_index(path, expected_role="development")
