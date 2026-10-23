"""Synthetic V9 acquisition and pure-feature conversion contracts."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from neural_experiments.v9 import fresh_data as data


def raw():
    tx = np.tile(np.arange(12, dtype=np.float32)[:, None], (1, 96))
    return {
        "participant_id": "T2",
        "post_implant_day": 100,
        "bin_size": 0.01,
        "electrodes": {"electrode_id": np.arange(96)},
        "neural": {"tx_3": tx * 3, "tx_4_5": tx, "sbp": tx * 2},
        "cursor_position": np.zeros((12, 2)),
        "target_position": np.tile([3.0, 4.0], (12, 1)),
        "blocks": {"block_id": [1, 2], "sample_start_index": [0, 7], "sample_end_index": [5, 12]},
        "trials": {"block_id": [1, 2], "trial_start_index": [0, 7], "trial_end_index": [5, 12]},
    }


def test_pair_sum_power_mean_gap_and_cap():
    f, t, m = data.convert_session(raw(), "T2", 100, cap_bins=3)
    assert set(f) == data.FEATURE_KEYS
    assert set(t) == {"y", "eval_mask"}
    np.testing.assert_array_equal(f["x_tx45"][:, 0], [1, 5, 15])
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
    with pytest.raises(ValueError, match="sbp"):
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
    assert f["x_tx45"].shape == (0, 96) and f["x_sbp"].shape == (0, 96)
    assert m["n_bins"] == 0 and t["y"].shape == (0, 2)


def tar_fixture(n=30):
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
        for i, (name, payload) in enumerate(rows[:24])
    ]
    file_id, size, digest = data.v2.PINS["T2"]
    return (stream.getvalue(), prior, {"file_id": file_id, "size": size, "provider_sha256": digest})


def test_extract_exact_next_six_and_authenticate_old_without_copy(tmp_path):
    payload, prior, archive = tar_fixture()
    r = data.extract_next_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", prior_members=prior, archive=archive
    )
    data.validate_receipt(r, "T2", prior)
    assert [s["regular_session_rank"] for s in r["selected"]] == [25, 26, 27, 28, 29, 30]
    assert len(list((tmp_path / "raw/T2").glob("*.mat"))) == 6
    assert len(r["verified_prior_members"]) == 24
    assert not r["full_archive_sha256_verified"] and (not r["tar_end_observed"])


def test_short_archive_retains_disposition_no_replacement(tmp_path):
    payload, prior, archive = tar_fixture(25)
    r = data.extract_next_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", prior_members=prior, archive=archive
    )
    data.validate_receipt(r, "T2", prior)
    assert r["missing_requested_ranks"] == [26, 27, 28, 29, 30] and len(r["selected"]) == 1
    assert r["tar_end_observed"]


def test_changed_old_member_fails_before_new_data(tmp_path):
    payload, prior, archive = tar_fixture()
    prior[0]["member_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="V8 member hash"):
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
    forged["selected"][0]["regular_session_rank"] = 9
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


def test_three_features_use_identical_pairs():
    f, _, _ = data.convert_session(raw(), "T2", 100, cap_bins=3)
    np.testing.assert_array_equal(f["x_tx3"][:, 0], [3, 15, 45])
    np.testing.assert_array_equal(f["x_tx45"][:, 0], [1, 5, 15])
    np.testing.assert_array_equal(f["x_sbp"][:, 0], [1, 5, 15])


def test_all_twenty_four_prior_members_required(tmp_path):
    payload, prior, archive = tar_fixture()
    with pytest.raises(ValueError, match="twenty-four"):
        data.extract_next_sessions(
            io.BytesIO(payload),
            cache=tmp_path,
            participant="T2",
            prior_members=prior[:9],
            archive=archive,
        )


def test_twenty_fourth_member_hash_is_authenticated(tmp_path):
    payload, prior, archive = tar_fixture()
    prior[23]["member_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="member hash"):
        data.extract_next_sessions(
            io.BytesIO(payload),
            cache=tmp_path,
            participant="T2",
            prior_members=prior,
            archive=archive,
        )
    assert not list((tmp_path / "raw/T2").glob("*.mat"))


def test_disk_limit_preserves_failure_receipt(tmp_path):
    payload, prior, archive = tar_fixture()
    with pytest.raises(ValueError, match="disk budget"):
        data.extract_next_sessions(
            io.BytesIO(payload),
            cache=tmp_path,
            participant="T2",
            prior_members=prior,
            archive=archive,
            max_cache_bytes=1,
        )
    assert json.loads((tmp_path / "acquisition/T2.json").read_text())["status"] == "failed"


def test_fresh_export_guard_before_parent_or_data_open(tmp_path):
    with pytest.raises(ValueError, match="locked study-plan"):
        data.prepare(tmp_path, tmp_path / "missing-v2", tmp_path / "missing-v3", role="fresh")
    assert not list(tmp_path.iterdir())


def test_changed_study_plan_refused(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        data._study_lock(p, "0" * 64)


@pytest.mark.parametrize("value", [-1, np.nan, 1.5])
def test_tx3_requires_finite_integer_counts(value):
    r = raw()
    r["neural"]["tx_3"][0, 0] = value
    with pytest.raises(ValueError):
        data.convert_session(r, "T2", 100)


def test_v9_fresh_rejects_continuation_role():
    with pytest.raises(ValueError, match="only permits the fresh"):
        data.convert_session(raw(), "T2", 100, role="t3_blocks")


def test_global_budget_reserves_concurrent_reads_and_raw(tmp_path):
    from neural_experiments.v9.acquire_parallel import SharedBudget

    b = SharedBudget(tmp_path, compressed_limit=100, raw_limit=20)
    b.start("T2")
    b.start("T5")
    assert b.reserve_read(80) == 80
    assert b.reserve_read(80) == 20
    with pytest.raises(ValueError, match="exhausted"):
        b.reserve_read(1)
    b.finish_read("T2", 80, 50)
    b.finish_read("T5", 20, 20)
    assert b.state["compressed_bytes_actual_total"] == 70
    assert b.reserve_read(80) == 30
    b.finish_read("T2", 30, 30)
    b.reserve_member("T2", tmp_path / "raw/T2/a.mat", 12)
    with pytest.raises(ValueError, match="raw-storage"):
        b.reserve_member("T5", tmp_path / "raw/T5/b.mat", 9)
    b.reserve_member("T5", tmp_path / "raw/T5/b.mat", 8)
    assert b.state["raw_bytes_reserved_total"] == 20


def test_unclean_stop_refuses_resume_without_mutating_receipts(tmp_path):
    from neural_experiments.v9.acquire_parallel import SharedBudget

    b = SharedBudget(tmp_path, compressed_limit=100, raw_limit=20)
    b.start("T2")
    b.reserve_read(40)
    b.finish_read("T2", 40, 25)
    b.reserve_read(30)
    b.save(True)
    before = (tmp_path / "parallel-acquisition.json").read_bytes()
    with pytest.raises(ValueError, match="Unclean acquisition stop"):
        SharedBudget(tmp_path, compressed_limit=100, raw_limit=20)
    assert (tmp_path / "parallel-acquisition.json").read_bytes() == before
    assert not (tmp_path / "failures").exists()


def test_clean_failed_attempt_keeps_consumed_transfer_on_resume(tmp_path):
    from neural_experiments.v9.acquire_parallel import SharedBudget

    b = SharedBudget(tmp_path, compressed_limit=100, raw_limit=20)
    b.start("T2")
    b.reserve_read(40)
    b.finish_read("T2", 40, 25)
    b.state["status"] = "failed"
    b.save(True)
    later = SharedBudget(tmp_path, compressed_limit=100, raw_limit=20)
    assert later.state["compressed_bytes_actual_total"] == 25
    assert len(list((tmp_path / "failures").glob("parallel-*.json"))) == 1


def test_replacing_existing_member_reserves_duplicate_temporary_bytes(tmp_path):
    from neural_experiments.v9.acquire_parallel import SharedBudget

    target = tmp_path / "raw/T2/a.mat"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * 12)
    b = SharedBudget(tmp_path, compressed_limit=100, raw_limit=20)
    b.start("T2")
    with pytest.raises(ValueError, match="raw-storage"):
        b.reserve_member("T2", target, 12)
    assert target.read_bytes() == b"x" * 12


def test_existing_partial_bytes_are_already_in_raw_inventory(tmp_path):
    from neural_experiments.v9.acquire_parallel import SharedBudget

    target = tmp_path / "raw/T2/a.mat"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * 12)
    target.with_suffix(".mat.part").write_bytes(b"y" * 8)
    b = SharedBudget(tmp_path, compressed_limit=100, raw_limit=24)
    b.start("T2")
    b.reserve_member("T2", target, 12)
    assert b.state["raw_bytes_reserved_total"] == 24
    with pytest.raises(ValueError, match="raw-storage"):
        b.reserve_member("T2", tmp_path / "raw/T2/b.mat", 1)


def test_public_parallel_worker_cap_before_metadata(tmp_path):
    from neural_experiments.v9.acquire_parallel import acquire_parallel

    with pytest.raises(ValueError, match="three"):
        acquire_parallel(tmp_path, tmp_path / "missing", tmp_path / "missing", workers=4)


@pytest.mark.parametrize(
    "n, missing", [(24, [25, 26, 27, 28, 29, 30]), (25, [26, 27, 28, 29, 30]), (29, [30])]
)
def test_fixed_selection_retains_every_archive_end_shortage(tmp_path, n, missing):
    payload, prior, archive = tar_fixture(n)
    receipt = data.extract_next_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", prior_members=prior, archive=archive
    )
    data.validate_receipt(receipt, "T2", prior)
    assert receipt["missing_requested_ranks"] == missing
    assert receipt["tar_end_observed"]
    assert len(receipt["selected"]) == n - 24


def test_prepare_and_validate_fresh_artifacts_and_no_substitution(tmp_path, monkeypatch):
    payload, prior, archive = tar_fixture(25)
    receipt = data.extract_next_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", prior_members=prior, archive=archive
    )
    monkeypatch.setattr(data, "PARTICIPANTS", ("T2",))
    monkeypatch.setattr(
        data, "_study_lock", lambda *_: {"path": "synthetic-plan", "sha256": "0" * 64}
    )
    monkeypatch.setattr(
        data,
        "_read_parents",
        lambda *_: {"sessions": [{"participant_id": "T2", "source": p} for p in prior]},
    )
    synthetic_raw = raw()
    synthetic_raw["post_implant_day"] = receipt["selected"][0]["post_implant_day"]
    monkeypatch.setattr(data, "loadmat", lambda *_args, **_kwargs: synthetic_raw)
    index = data.prepare(tmp_path, tmp_path / "parent-a", tmp_path / "parent-b")
    loaded = data.load_index(tmp_path / "index.json", expected_role="fresh")
    assert loaded == index
    assert len(index["sessions"]) == 1
    assert index["sessions"][0]["role"] == "fresh"
    assert index["participant_dispositions"][0]["missing_requested_ranks"] == [26, 27, 28, 29, 30]
    assert (
        set(data.load_features(tmp_path / index["sessions"][0]["feature_path"]))
        == data.FEATURE_KEYS
    )
    assert set(data.load_targets(tmp_path / index["sessions"][0]["target_path"])) == {
        "y",
        "eval_mask",
    }
    index["sessions"][0]["n_bins"] = 30001
    (tmp_path / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="exposure"):
        data.load_index(tmp_path / "index.json", expected_role="fresh", verify_hashes=False)


def test_parallel_plan_guard_precedes_metadata_or_download(tmp_path, monkeypatch):
    from neural_experiments.v9 import acquire_parallel

    def blocked(*_):
        raise ValueError("uncommitted synthetic plan")

    monkeypatch.setattr(data, "_study_lock", blocked)
    monkeypatch.setattr(data.v2, "pin_metadata", lambda *_: pytest.fail("network metadata called"))
    with pytest.raises(ValueError, match="uncommitted"):
        acquire_parallel.acquire_parallel(tmp_path, tmp_path / "parent-a", tmp_path / "parent-b")


def test_provider_archive_mismatch_fails_receipt(tmp_path):
    payload, prior, archive = tar_fixture()
    receipt = data.extract_next_sessions(
        io.BytesIO(payload), cache=tmp_path, participant="T2", prior_members=prior, archive=archive
    )
    receipt["archive"]["provider_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Provider archive"):
        data.validate_receipt(receipt, "T2", prior)


def test_public_limits_are_bounded_and_roles_explicit(tmp_path):
    assert data.MAX_COMPRESSED_BYTES == 32_000_000_000
    assert data.MAX_CACHE_BYTES == 12_000_000_000
    assert data.SELECTION["regular_session_ranks_one_based"] == [25, 26, 27, 28, 29, 30]
    with pytest.raises(ValueError, match="role=fresh"):
        data.prepare(tmp_path, tmp_path / "a", tmp_path / "b", role="development")


def test_fixed_four_participants_and_exact_acquisition_plan():
    from pathlib import Path

    assert data.PARTICIPANTS == ("T2", "T7", "T9", "T10")
    path = Path(data.__file__).with_name("acquisition-plan.json")
    assert data.v2.sha256_file(path) == data.PLAN_SHA256
    plan = json.loads(path.read_text())
    assert plan["participants"] == list(data.PARTICIPANTS)
    assert plan["authenticate_prior_regular_members"] == 24
    assert plan["regular_session_ranks_one_based"] == list(range(25, 31))


def test_unselected_participant_rejected_before_payload_access(tmp_path):
    class Unreadable:
        def read(self, *_):
            pytest.fail("must reject before reading archive")

    with pytest.raises(ValueError, match="pinned participant"):
        data.extract_next_sessions(
            Unreadable(), cache=tmp_path, participant="T5", prior_members=[{}] * 24, archive={}
        )
