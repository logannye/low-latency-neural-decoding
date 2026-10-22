"""Fake-clock execution plus independently recomputed trace/clock tamper checks."""

import json
import shutil

import numpy as np
import pytest

from neural_experiments.v4.model import Decoder
from neural_experiments.v4.runtime import PERIOD_NS, measure
from neural_experiments.v6 import runtime
from neural_experiments.v6 import runtime_audit as audit


class FakeClock:
    def __init__(self):
        self.now = 0

    def perf_counter_ns(self):
        self.now += 1000
        return self.now

    def process_time_ns(self):
        return self.now // 100

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


def write(path, value):
    path.write_text(json.dumps(value))


def fixture_binding(tmp_path, *, zero=False):
    source_root = runtime.ROOT
    for name in runtime.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, path)
    shutil.copyfile(source_root / "neural_experiments/v6/plan.json", tmp_path / "plan.json")
    config = {
        "family": "tx45_sbp",
        "tau_seconds": 0.2,
        "baseline_tau_seconds": None,
        "adapt_gamma": 10.0,
    }
    weights = np.zeros((193, 2))
    if not zero:
        weights[:3] = [[0.10, -0.03], [-0.02, 0.03], [0.01, -0.04]]
        weights[-1] = [0.01, -0.02]
    decoder = Decoder(config, np.full(192, 0.5), np.full(192, 1.2), weights)
    decoder.save(tmp_path / "model.npz")
    n = 1500
    rng = np.random.default_rng(11)
    features = {
        "x_tx3": rng.poisson(1, (n, 96)).astype(np.float32),
        "x_tx45": rng.poisson(0.3, (n, 96)).astype(np.float32),
        "x_sbp": rng.random((n, 96), dtype=np.float32),
    }
    sequence = np.r_[np.zeros(700, np.int64), np.ones(n - 700, np.int64)]
    np.savez(
        tmp_path / "features.npz",
        **features,
        timestamps=np.arange(n) * 0.02,
        sequence_id=sequence,
        channel_ids=np.arange(96, dtype=np.int64),
    )
    expected = decoder.predict(features, [(0, 700), (700, n)])
    np.savez(tmp_path / "assessed.npz", prediction=expected, eval_mask=np.ones(n, bool))
    item = {
        "participant_id": "T2",
        "session_id": "T2_fixture",
        "source": {"regular_session_rank": 16},
        "n_bins": n,
        "feature_path": "features.npz",
        "feature_sha256": runtime.digest(tmp_path / "features.npz"),
    }
    plan = json.loads((tmp_path / "plan.json").read_text())
    index = {
        "role": "new_sessions",
        "study_lock": {"sha256": runtime.PLAN_SHA256},
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "sessions": [item],
    }
    write(tmp_path / "index.json", index)
    selection = {
        "plan": {"sha256": runtime.PLAN_SHA256},
        "code_sha256": {name: runtime.digest(tmp_path / name) for name in runtime.SOURCE_FILES},
    }
    write(tmp_path / "selection.json", selection)
    row = {
        "participant": "T2",
        "session_id": "T2_fixture",
        "domain": "new_sessions",
        "method": "confirmed",
        "action": "abstain" if zero else "accept",
        "training_label_seconds": 0 if zero else 60,
        "decision": {
            "action": "abstain" if zero else "accept",
            "candidate_id": None if zero else "tx45_sbp_scratch1_fit_available",
            "model_sha256": None if zero else runtime.digest(tmp_path / "model.npz"),
            "training_seconds": 0 if zero else 60,
            "requested_label_seconds": 420 if zero else 120,
            "available_label_seconds": 420 if zero else 120,
            "decision_time_seconds": 450 if zero else 150,
        },
        "label_seconds": 420 if zero else 120,
        "model_path": "model.npz",
        "model_sha256": runtime.digest(tmp_path / "model.npz"),
        "trace_path": "assessed.npz",
        "trace_sha256": runtime.digest(tmp_path / "assessed.npz"),
        "metrics": {"r2": -1000.0},
    }
    source_receipt = {
        "status": "PASS",
        "stage": "source",
        "selection_sha256": runtime.digest(tmp_path / "selection.json"),
        "plan_sha256": runtime.PLAN_SHA256,
    }
    write(tmp_path / "source-audit.json", source_receipt)
    assessment = {
        "status": "assessment_complete",
        "stage": "assessment",
        "cases": [row],
        "selection_path": "selection.json",
        "selection_sha256": runtime.digest(tmp_path / "selection.json"),
        "source_audit_path": "source-audit.json",
        "source_audit_sha256": runtime.digest(tmp_path / "source-audit.json"),
        "index": {"path": "index.json", "sha256": runtime.digest(tmp_path / "index.json")},
    }
    write(tmp_path / "manifest.json", assessment)
    write(tmp_path / "assessment.json", assessment)
    receipt = {
        "status": "PASS",
        "stage": "assessment",
        "source_audit_sha256": runtime.digest(tmp_path / "source-audit.json"),
        "index_sha256": runtime.digest(tmp_path / "index.json"),
        "manifest_sha256": runtime.digest(tmp_path / "manifest.json"),
        "assessment_sha256": runtime.digest(tmp_path / "assessment.json"),
        "selection_sha256": runtime.digest(tmp_path / "selection.json"),
        "plan_sha256": runtime.PLAN_SHA256,
    }
    write(tmp_path / "assessment-audit.json", receipt)
    return assessment, index, receipt


def fixture_run(tmp_path, monkeypatch, *, zero=False):
    fixture_binding(tmp_path, zero=zero)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(
        runtime, "measure", lambda *args, **kw: measure(*args, **kw, clock=FakeClock())
    )
    output = tmp_path / "run"
    manifest = runtime.run(
        tmp_path / "manifest.json",
        tmp_path / "index.json",
        tmp_path / "plan.json",
        output,
        assessment_audit_path=tmp_path / "assessment-audit.json",
    )
    assert PERIOD_NS == 20_000_000  # the frozen helper's constants were never changed
    return output, manifest


def save_manifest(output, manifest):
    write(output / "manifest.json", manifest)
    write(
        output / "summary.json",
        {**manifest, "manifest_sha256": runtime.digest(output / "manifest.json")},
    )


@pytest.mark.parametrize("zero", [False, True])
def test_fake_clock_full_run_and_independent_nonconstant_replay(tmp_path, monkeypatch, zero):
    output, manifest = fixture_run(tmp_path, monkeypatch, zero=zero)
    result = audit.audit(output, tmp_path)
    assert result["status"] == "passed" and result["blocks_checked"] == 5
    assert result["prediction_replay"]["measured_trace_max_absolute_error"] < 1e-12
    assert manifest["representative"]["action"] == ("abstain" if zero else "accept")
    assert result["representative"]["archive_rank"] == 16
    assert all(row["deadline_misses"] == 0 for row in result["blocks"])
    assert manifest["blocks"][3]["n"] == 3000
    assert manifest["blocks"][3]["service_p99_ms"] == pytest.approx(0.001)
    assert "not useful decoding" in result["scope"]


@pytest.mark.parametrize(
    "change",
    [
        "rank",
        "duplicate",
        "failed_audit",
        "audit_result",
        "model",
        "decision_model",
        "cost",
        "source_audit",
    ],
)
def test_binding_rejects_wrong_representative_or_unverified_artifacts(tmp_path, change):
    assessment, index, receipt = fixture_binding(tmp_path)
    if change == "rank":
        index["sessions"][0]["source"]["regular_session_rank"] = 14
    elif change == "duplicate":
        assessment["cases"].append(assessment["cases"][0].copy())
    elif change == "failed_audit":
        receipt["status"] = "FAIL"
    elif change == "audit_result":
        receipt["assessment_sha256"] = "0" * 64
    elif change == "decision_model":
        assessment["cases"][0]["decision"]["model_sha256"] = "0" * 64
    elif change == "cost":
        assessment["cases"][0]["decision"]["requested_label_seconds"] = 30
    elif change == "source_audit":
        source_path = tmp_path / "source-audit.json"
        source_receipt = audit.read(source_path)
        source_receipt["status"] = "FAIL"
        write(source_path, source_receipt)
        assessment["source_audit_sha256"] = receipt["source_audit_sha256"] = runtime.digest(
            source_path
        )
    else:
        with (tmp_path / "model.npz").open("ab") as stream:
            stream.write(b"tampered")
    write(tmp_path / "index.json", index)
    assessment["index"]["sha256"] = runtime.digest(tmp_path / "index.json")
    write(tmp_path / "manifest.json", assessment)
    write(tmp_path / "assessment.json", assessment)
    receipt["index_sha256"] = runtime.digest(tmp_path / "index.json")
    receipt["manifest_sha256"] = runtime.digest(tmp_path / "manifest.json")
    if change != "audit_result":
        receipt["assessment_sha256"] = runtime.digest(tmp_path / "assessment.json")
    write(tmp_path / "assessment-audit.json", receipt)
    with pytest.raises(ValueError):
        runtime.bind_representative(
            tmp_path / "manifest.json",
            tmp_path / "index.json",
            tmp_path / "plan.json",
            tmp_path / "assessment-audit.json",
            tmp_path,
        )


@pytest.mark.parametrize(
    "change",
    [
        "prediction",
        "queue",
        "release",
        "deadline",
        "overlap",
        "missing_block",
        "false_metric",
        "wrong_state",
    ],
)
def test_rehashed_trace_or_clock_tampering_rejected(tmp_path, monkeypatch, change):
    output, manifest = fixture_run(tmp_path, monkeypatch)
    block = manifest["blocks"][3]
    path = tmp_path / block["artifact_path"]
    with np.load(path, allow_pickle=False) as z:
        stamps, predictions = z["timestamps_ns"], z["predictions"]
    if change == "prediction":
        predictions[10, 0] += 0.01
    elif change == "queue":
        stamps[10, 1] = stamps[10, 0] - 1
    elif change == "release":
        stamps[10] += 1
    elif change == "deadline":
        stamps[10, 3] += 1
    elif change == "overlap":
        stamps[10, 2] = stamps[11, 1] + 1
    elif change == "missing_block":
        manifest["blocks"].pop()
    elif change == "false_metric":
        block["service_p99_ms"] = 0.0
    else:
        block["persistent_numeric_state_bytes"] = 0
    np.savez(path, timestamps_ns=stamps, predictions=predictions)
    block["artifact_sha256"] = runtime.digest(path)
    save_manifest(output, manifest)
    with pytest.raises(ValueError):
        audit.audit(output, tmp_path)


def test_independent_audit_rejects_rehashed_assessed_trace(tmp_path, monkeypatch):
    output, manifest = fixture_run(tmp_path, monkeypatch)
    path = tmp_path / "assessed.npz"
    with np.load(path, allow_pickle=False) as z:
        prediction, mask = z["prediction"], z["eval_mask"]
    prediction[100, 0] += 0.01
    np.savez(path, prediction=prediction, eval_mask=mask)
    manifest["assessment_prediction_sha256"] = runtime.digest(path)
    save_manifest(output, manifest)
    with pytest.raises(ValueError, match="assessed trace identity"):
        audit.audit(output, tmp_path)


def publication_fixture(tmp_path, monkeypatch):
    output, manifest = fixture_run(tmp_path, monkeypatch)
    manifest["private_note"] = "/Users/private/notes"
    manifest["raw_samples"] = [[5, 6]]
    manifest["environment"]["private_path"] = "/Users/private/runtime"
    # Accelerate environments may report no introspectable native pools.
    manifest["environment"]["threadpools"].append(
        {"num_threads": 1, "filepath": "/Users/private/library"}
    )
    manifest["protocol"]["private_path"] = "/Users/private/config"
    manifest["filter_response"]["raw_samples"] = [[7, 8]]
    for block in manifest["blocks"]:
        block["raw_samples"] = [[1, 2]]
    save_manifest(output, manifest)
    receipt = audit.audit(output, tmp_path)
    receipt["private_path"] = "/Users/private/audit"
    receipt["prediction_replay"]["raw_samples"] = [[3, 4]]
    receipt["blocks"][0]["service_p99_ms"] += 1e-16
    path = tmp_path / "runtime-audit.json"
    write(path, receipt)
    return output, manifest, path, receipt


def test_public_export_strips_unknown_paths_and_samples_and_retains_evidence(tmp_path, monkeypatch):
    from neural_experiments.v6 import runtime_public

    output, manifest, path, receipt = publication_fixture(tmp_path, monkeypatch)
    public = runtime_public.export(output, path, tmp_path)
    text = json.dumps(public)
    for forbidden in ("/Users/", "_path", "source_snapshots", "raw_samples", "neural_experiments/"):
        assert forbidden not in text
    assert public["representative"] == manifest["representative"]
    assert public["assessment_sha256"] == manifest["assessment_sha256"]
    assert (
        public["independent_audit"]["assessment_audit_sha256"]
        == manifest["assessment_audit_sha256"]
    )
    assert (
        public["independent_audit"]["prediction_replay"]["measured_trace_max_absolute_error"]
        == receipt["prediction_replay"]["measured_trace_max_absolute_error"]
    )
    assert (
        public["blocks"][0]["service_p99_ms"]
        != public["independent_audit"]["blocks"][0]["service_p99_ms"]
    )
    assert set(public["source_sha256"]) == set(runtime_public.COMPONENTS)


@pytest.mark.parametrize(
    "key",
    [
        "status",
        "manifest_sha256",
        "summary_sha256",
        "assessment_sha256",
        "plan_sha256",
        "index_sha256",
        "model_sha256",
        "feature_sha256",
        "auditor_source_sha256",
        "source_sha256",
        "source_audit_sha256",
        "prediction_reference_sha256",
    ],
)
def test_public_export_requires_exact_passed_runtime_audit(tmp_path, monkeypatch, key):
    from neural_experiments.v6 import runtime_public

    output, _, path, receipt = publication_fixture(tmp_path, monkeypatch)
    receipt[key] = "PASS" if key == "status" else "changed"
    write(path, receipt)
    with pytest.raises(ValueError, match="audit"):
        runtime_public.export(output, path, tmp_path)


def test_public_export_rejects_failed_quality_audit_even_if_rebound(tmp_path, monkeypatch):
    from neural_experiments.v6 import runtime_public

    output, manifest, path, receipt = publication_fixture(tmp_path, monkeypatch)
    quality_path = tmp_path / "assessment-audit.json"
    quality = audit.read(quality_path)
    quality["status"] = "FAIL"
    write(quality_path, quality)
    manifest["assessment_audit_sha256"] = runtime.digest(quality_path)
    save_manifest(output, manifest)
    receipt["assessment_audit_sha256"] = manifest["assessment_audit_sha256"]
    receipt["manifest_sha256"] = runtime.digest(output / "manifest.json")
    receipt["summary_sha256"] = runtime.digest(output / "summary.json")
    write(path, receipt)
    with pytest.raises(ValueError, match="assessment audit failed"):
        runtime_public.export(output, path, tmp_path)


def test_public_export_rechecks_bound_binary_and_source_files(tmp_path, monkeypatch):
    from neural_experiments.v6 import runtime_public

    output, manifest, path, _ = publication_fixture(tmp_path, monkeypatch)
    snapshot = tmp_path / manifest["source_snapshots"]["neural_experiments/v4/model.py"]
    snapshot.write_text("changed after audit")
    with pytest.raises(ValueError, match="changed source"):
        runtime_public.export(output, path, tmp_path)


def test_independent_audit_rejects_rebound_confirmation_model(tmp_path, monkeypatch):
    output, manifest = fixture_run(tmp_path, monkeypatch)
    assessment = audit.read(tmp_path / "assessment.json")
    assessment["cases"][0]["decision"]["model_sha256"] = "0" * 64
    write(tmp_path / "assessment.json", assessment)
    write(tmp_path / "manifest.json", assessment)
    quality = audit.read(tmp_path / "assessment-audit.json")
    quality["assessment_sha256"] = runtime.digest(tmp_path / "assessment.json")
    quality["manifest_sha256"] = runtime.digest(tmp_path / "manifest.json")
    write(tmp_path / "assessment-audit.json", quality)
    for key, filename in (
        ("assessment", "manifest.json"),
        ("assessment_result", "assessment.json"),
        ("assessment_audit", "assessment-audit.json"),
    ):
        manifest[key + "_sha256"] = runtime.digest(tmp_path / filename)
    save_manifest(output, manifest)
    with pytest.raises(ValueError, match="confirmation decision"):
        audit.audit(output, tmp_path)
