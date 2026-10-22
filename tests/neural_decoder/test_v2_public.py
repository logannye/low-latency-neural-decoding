"""Publication gate: exact completed-run audits and aggregate-only field selection."""

import copy
import json

import pytest

from neural_experiments.v2 import public

HASH = "a" * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return public.digest(path)


@pytest.fixture
def transfer_run(tmp_path):
    run = tmp_path / "run"
    case = {"target_participant": "T2", "method": "ridge", "seed": 0, "budget_seconds": 30}
    metrics = {
        "r2": 0.1,
        "per_output_r2": [0.2, 0.0],
        "angular_error_degrees": 30.0,
        "zero_prediction_count": 0,
        "n": 100,
    }
    prefix = {"session_id": "T2_day_1", "start_bin": 0, "stop_bin": 1500}
    identity = {
        "model_id": "T2-ridge-30s-seed0",
        **case,
        "source_participants": [],
        "source_train_session_ids": [],
        "source_validation_session_ids": [],
        "calibration": {
            **prefix,
            "eligible_count": 1400,
            "mask_sha256": HASH,
            "training_count": 1340,
            "training_mask_sha256": HASH,
            "targets": [999.0],
        },
        "normalization": {**prefix, "input_mean": [123.0]},
        "model_sha256": HASH,
        "core_sha256": None,
        "training_seconds": 0.5,
        "parameter_count": 194,
        "model_path": "private/model.npz",
        "weights": [123.0],
        "traces": [
            {
                "session_id": "T2_day_2",
                "partition": "future_day",
                "start_bin": 0,
                "stop_bin": 200,
                "prediction_sha256": HASH,
                "score_mask_sha256": HASH,
                "prediction_path": "private/prediction.npy",
                "metrics": metrics,
            }
        ],
    }
    identity_hash = write(run / "identity.json", identity)
    manifest = {
        "status": "complete",
        "source": {"neural_experiments/v2/transfer.py": HASH},
        "config_sha256": HASH,
        "index_sha256": HASH,
        "environment": {"threads": 2, "python": "3.12"},
        "models": [{"identity_path": "run/identity.json", "identity_sha256": identity_hash}],
        "pretraining": [],
        "unavailable": [],
    }
    manifest_hash = write(run / "manifest.json", manifest)
    summary = {
        "status": "complete",
        "manifest_sha256": manifest_hash,
        "rows": [],
        "groups": [],
        "primary_contrast": {"mean_difference": 0.1},
        "unavailable": [],
    }
    summary_hash = write(run / "summary.json", summary)
    audit = {
        "status": "passed",
        "manifest_sha256": manifest_hash,
        "summary_sha256": summary_hash,
        "config_sha256": HASH,
        "index_sha256": HASH,
        "audit_source_sha256": HASH,
        "models_checked": 1,
        "traces_checked": 1,
        "unavailable_models": 0,
        "selected_members_checked": 54,
        "source_cores_checked": 27,
        "participants": ["T2"],
        "reference": "independent sklearn R2",
        "private_probe": [123.0],
        "independent_summary": {k: summary[k] for k in ("rows", "groups", "primary_contrast")},
        "independent_scores": [
            {
                **case,
                "session_id": "T2_day_2",
                "partition": "future_day",
                "metrics": {**metrics, "r2": 0.10000000000000002},
            }
        ],
    }
    audit_path = run / "audit.json"
    write(audit_path, audit)
    return run, audit_path, tmp_path


def test_transfer_publication_preserves_aggregate_identity_not_private_fields(transfer_run):
    result = public.export_transfer(*transfer_run)
    model = result["models"][0]
    assert model["calibration"]["training_count"] == 1340
    assert model["model_sha256"] == HASH
    assert model["traces"][0]["metrics"]["r2"] == 0.10000000000000002
    text = json.dumps(result)
    for forbidden in (
        "model_path",
        "prediction_path",
        "input_mean",
        "weights",
        "targets",
        "private_probe",
    ):
        assert forbidden not in text
    assert result["primary_contrast"]["mean_difference"] == 0.1


@pytest.mark.parametrize(
    "damage", ["incomplete", "failed_audit", "manifest_changed", "summary_changed"]
)
def test_transfer_publication_rejects_unbound_or_incomplete_evidence(transfer_run, damage):
    run, audit_path, root = transfer_run
    if damage == "failed_audit":
        value = public.read(audit_path)
        value["status"] = "failed"
        write(audit_path, value)
    else:
        path = run / ("summary.json" if damage == "summary_changed" else "manifest.json")
        value = public.read(path)
        value["status"] = "running" if damage == "incomplete" else "complete"
        value["changed"] = True
        write(path, value)
    with pytest.raises(ValueError):
        public.export_transfer(run, audit_path, root)


def test_publication_rejects_missing_independent_trace(transfer_run):
    run, audit_path, root = transfer_run
    value = public.read(audit_path)
    value["independent_scores"] = []
    write(audit_path, value)
    with pytest.raises(ValueError, match="independently scored trace"):
        public.export_transfer(run, audit_path, root)


def test_runtime_export_sanitizes_library_paths_and_preserves_clock_scope(tmp_path):
    run = tmp_path / "runtime"
    metrics = {
        "mode": "paced",
        "repeat": 0,
        "implementation": "numpy_incremental",
        "n": 3000,
        "service_p50_ms": 0.1,
        "service_p99_ms": 0.2,
        "service_max_ms": 0.3,
        "queue_p99_ms": 0.1,
        "release_to_completion_p99_ms": 0.3,
        "deadline_misses": 0,
        "negative_queue_count": 0,
    }
    block = {
        **metrics,
        "artifact_path": "private/raw.npz",
        "artifact_sha256": HASH,
        "initialization_ms": 0.5,
        "persistent_numeric_state_bytes": 100,
        "state_measurement": "numeric caches only",
    }
    manifest = {
        "kind": "h1",
        "status": "complete",
        "model_sha256": HASH,
        "feature_sha256": HASH,
        "runtime_source_sha256": HASH,
        "model_path": "private/model.npz",
        "feature_path": "private/x.npz",
        "source": {"neural_experiments/v2/runtime.py": HASH},
        "protocol": {"period_ms": 20},
        "parameter_count": 100,
        "numeric_weight_bytes": 400,
        "full_trace_parity": {"numpy_incremental": 1e-8},
        "blocks": [block],
        "environment": {
            "python": "3.12",
            "username": "private user",
            "threadpools": [
                {
                    "filepath": "/Users/private/.venv/libblas.so",
                    "num_threads": 1,
                    "internal_api": "openblas",
                    "architecture": "ARMV8",
                    "version": "0.3",
                }
            ],
        },
    }
    manifest_hash = write(run / "manifest.json", manifest)
    summary_hash = write(run / "summary.json", {**manifest, "manifest_sha256": manifest_hash})
    audit = {
        "status": "pass",
        "auditor_source_sha256": HASH,
        "manifest_sha256": manifest_hash,
        "summary_sha256": summary_hash,
        "model_sha256": HASH,
        "feature_sha256": HASH,
        "source_snapshots_checked": 3,
        "blocks_checked": 1,
        "blocks": [copy.deepcopy(metrics)],
        "prediction_correctness_independently_verified": False,
        "prediction_check_scope": "shape and finiteness only",
        "clock_check_scope": "independent timestamps",
        "memory_check_scope": "numeric weights only",
    }
    audit_path = run / "audit.json"
    write(audit_path, audit)
    result = public.export_runtime(run, audit_path, tmp_path)
    pool = result["environment"]["threadpools"][0]
    assert pool["num_threads"] == 1 and pool["architecture"] == "ARMV8"
    assert "filepath" not in pool
    assert result["independent_audit"]["prediction_correctness_independently_verified"] is False
    assert result["blocks"][0]["release_to_completion_p99_ms"] == 0.3
    text = json.dumps(result)
    assert "/Users/" not in text and "private/" not in text and "username" not in text
    audit["summary_sha256"] = "b" * 64
    write(audit_path, audit)
    with pytest.raises(ValueError, match="hash mismatch"):
        public.export_runtime(run, audit_path, tmp_path)
