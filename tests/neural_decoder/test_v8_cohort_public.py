"""Cohort publication reads allowlisted JSON metadata, never sample artifacts."""

from __future__ import annotations

import copy
import json

import pytest

from neural_experiments.v2.data import _json, sha256_file
from neural_experiments.v8 import cohort_public as public


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    cache = tmp_path / "data/neural_v8/fresh"
    cache.mkdir(parents=True)
    for name in ("fresh_data.py", "acquire_parallel.py"):
        path = tmp_path / "neural_experiments/v8" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic execution snapshot\n")
    _json(tmp_path / "data/neural_v8/development/index.json", {"synthetic": True})
    sessions, dispositions = [], []
    for participant in public.PARTICIPANTS:
        selected = []
        for rank in (22, 23, 24):
            source = {
                "participant_id": participant,
                "session_id": f"{participant}_day_{rank}_decoding",
                "post_implant_day": rank,
                "regular_session_rank": rank,
                "archive_ordinal": rank,
                "member_size": 1000,
                "member_sha256": "c" * 64,
            }
            selected.append(source)
            sessions.append(
                {
                    "participant_id": participant,
                    "session_id": source["session_id"],
                    "post_implant_day": rank,
                    "n_bins": 30000,
                    "n_channels": 96,
                    "n_eval_bins": 25000,
                    "full_prepared_n_bins": 50000,
                    "role": "fresh",
                    "feature_sha256": "a" * 64,
                    "target_sha256": "b" * 64,
                    "context_sha256": "c" * 64,
                    "source": source,
                    "segments": [{"start": 0, "end": 30000}],
                    "feature_path": "MUST_NOT_BE_READ.features.npz",
                    "target_path": "MUST_NOT_BE_READ.targets.npz",
                }
            )
        _json(
            cache / "acquisition" / f"{participant}.json",
            {
                "archive": {
                    "file_id": 1,
                    "filename": f"decoding_{participant}.tar.gz",
                    "size": 100000,
                    "provider_sha256": "d" * 64,
                    "transport": {"private_url_file_sha256": "e" * 64},
                    "download_url": "https://private.invalid/?Credential=do-not-publish",
                },
                "status": "complete",
                "compressed_prefix_bytes": 100,
                "compressed_prefix_sha256": "f" * 64,
                "full_archive_sha256_verified": False,
                "verified_prior_members": [{}] * 21,
                "selected": selected,
                "missing_requested_ranks": [],
                "tar_end_observed": False,
            },
        )
        dispositions.append(
            {"participant_id": participant, "selected_sessions": 3, "missing_requested_ranks": []}
        )
    ledger = {
        "status": "complete",
        "started_at": "2026-10-23T00:00:00+00:00",
        "finished_at": "2026-10-23T00:01:00+00:00",
        "maximum_compressed_bytes_total": 32000000000,
        "maximum_retained_raw_bytes_total": 12000000000,
        "maximum_parallel_workers": 3,
        "compressed_bytes_actual_total": 800,
        "raw_bytes_actual_final": 24000,
        "orchestrator_sha256": sha256_file(tmp_path / "neural_experiments/v8/acquire_parallel.py"),
        "importer_sha256": sha256_file(tmp_path / "neural_experiments/v8/fresh_data.py"),
        "compressed_bytes_before_orchestration": 0,
        "compressed_bytes_claimed_pending": 0,
    }
    _json(cache / "parallel-acquisition.json", ledger)
    index = {
        "role": "fresh",
        "acquisition_plan_sha256": "1" * 64,
        "study_lock": {"sha256": "2" * 64},
        "v7_development_index_sha256": "3" * 64,
        "v7_fresh_index_sha256": "4" * 64,
        "participant_dispositions": dispositions,
        "sessions": sessions,
    }
    _json(cache / "index.json", index)
    monkeypatch.setattr(public, "ROOT", tmp_path)
    monkeypatch.setattr(public, "CACHE", cache)
    monkeypatch.setattr(public, "DEST", tmp_path / "cohort.json")
    return cache, index, ledger


def test_full_metadata_export_excludes_private_fields_and_never_opens_samples(cohort):
    _, _, _ = cohort
    result = public.main()
    assert result["selected_session_count"] == 24
    assert result["recorded_bins"] == 720000
    assert result["metadata_exporter_sha256"] == sha256_file(public.__file__)
    assert result["source_identity"]["importer_bytes_unchanged_since_acquisition"]
    text = public.DEST.read_text()
    assert "Credential=" not in text and "MUST_NOT_BE_READ" not in text
    assert str(public.ROOT) not in text
    assert all(s["scoring_potential_is_not_valid_go_confirmation"] for s in result["sessions"])


def test_short_recording_potential_is_geometry_only(cohort):
    cache, index, _ = cohort
    row = index["sessions"][0]
    row.update(n_bins=7620, n_eval_bins=7620, segments=[{"start": 0, "end": 7620}])
    _json(cache / "index.json", index)
    result = public.main()
    assert result["sessions"][0]["maximum_potential_shared_scoring_bins_from_geometry"] == 20
    assert not result["sessions"][0]["potentially_scorable_by_geometry"]


def test_acquisition_block_burn_applies_to_geometry_potential(cohort):
    cache, index, _ = cohort
    index["sessions"][0]["segments"] = [{"start": 0, "end": 7600}, {"start": 7600, "end": 7700}]
    _json(cache / "index.json", index)
    result = public.main()
    assert result["sessions"][0]["maximum_potential_shared_scoring_bins_from_geometry"] == 38


def test_dropped_acquired_session_rejected(cohort):
    cache, index, _ = cohort
    index["sessions"].pop()
    _json(cache / "index.json", index)
    with pytest.raises(ValueError, match="inventory"):
        public.main()


def test_explicit_missing_ranks_are_retained_without_replacement(cohort):
    cache, index, _ = cohort
    path = cache / "acquisition/T2.json"
    receipt = json.loads(path.read_text())
    receipt["selected"] = receipt["selected"][:1]
    receipt.update(missing_requested_ranks=[23, 24], tar_end_observed=True)
    _json(path, receipt)
    index["sessions"] = [
        s
        for s in index["sessions"]
        if s["session_id"] not in {"T2_day_23_decoding", "T2_day_24_decoding"}
    ]
    index["participant_dispositions"][0].update(
        selected_sessions=1, missing_requested_ranks=[23, 24]
    )
    _json(cache / "index.json", index)
    result = public.main()
    assert result["selected_session_count"] == 22
    assert result["archives"][0]["missing_requested_ranks"] == [23, 24]


def test_failed_attempt_provenance_and_bytes_are_retained(cohort):
    cache, _, ledger = cohort
    failed = copy.deepcopy(ledger)
    failed.update(status="failed", compressed_bytes_actual_total=10)
    _json(cache / "failures/parallel-first.json", failed)
    _json(
        cache / "failures/T2-first.json",
        {
            "participant_id": "T2",
            "status": "failed",
            "type": "HTTPError",
            "compressed_prefix_bytes": 10,
            "message": "https://private.invalid/?Signature=secret",
        },
    )
    ledger.update(compressed_bytes_actual_total=810, compressed_bytes_before_orchestration=10)
    _json(cache / "parallel-acquisition.json", ledger)
    result = public.main()
    assert len(result["orchestration_attempts"]) == 2
    assert result["failed_archive_attempts"][0]["compressed_prefix_bytes"] == 10
    assert "Signature=" not in public.DEST.read_text()


@pytest.mark.parametrize("change", ["mismatch", "pending", "running"])
def test_incomplete_or_inconsistent_transfer_rejected(cohort, change):
    cache, _, ledger = cohort
    if change == "mismatch":
        ledger["compressed_bytes_actual_total"] += 1
    elif change == "pending":
        ledger["compressed_bytes_claimed_pending"] = 1
    else:
        ledger["status"] = "running"
    _json(cache / "parallel-acquisition.json", ledger)
    with pytest.raises(ValueError):
        public.main()


def test_private_string_in_allowlisted_field_rejected(cohort):
    cache, _, _ = cohort
    path = cache / "acquisition/T2.json"
    receipt = json.loads(path.read_text())
    receipt["archive"]["filename"] = "/Users/private"
    _json(path, receipt)
    with pytest.raises(ValueError, match="Forbidden"):
        public.main()


def test_transport_observations_are_receipt_bound_and_private_extras_omitted(cohort):
    cache, _, _ = cohort
    path = cache / "transport-observations.json"
    _json(
        path,
        {
            "count": 8,
            "http_status": 403,
            "archive_body_bytes": 0,
            "ordinary_browser_validation": "automatic validation pending",
            "private_url": "https://private.invalid/?Signature=secret",
        },
    )
    result = public.main()
    record = result["separate_public_link_resolution_attempts"]
    assert record["count"] == 8 and record["receipt_sha256"] == sha256_file(path)
    assert "Signature=" not in public.DEST.read_text()


def test_successful_direct_transport_does_not_invent_browser_failures(cohort):
    cache, _, _ = cohort
    path = cache / "acquisition/T2.json"
    receipt = json.loads(path.read_text())
    del receipt["archive"]["transport"]
    _json(path, receipt)
    result = public.main()
    assert result["archives"][0]["transport_map_sha256"] is None
    assert result["separate_public_link_resolution_attempts"] == {"disposition": "not_recorded"}
