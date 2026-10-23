import ast
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from neural_experiments.v7 import audit_runtime_continual as audit


def refits(tmp_path):
    original = {
        "format_version": np.array(4, np.int64),
        "config": np.array(json.dumps(audit.CONFIG)),
        "mean": np.zeros(4),
        "scale": np.ones(4),
        "weights": np.zeros((5, 2)),
    }
    rows = []
    for i in range(10):
        path = tmp_path / f"refit-{i}.npz"
        np.savez(path, **original)
        rows.append(
            {
                "repeat": i,
                "fit_ns": 1000 + i,
                "export_reload_ns": 3000,
                "total_ns": 4000 + i,
                "training_rows": 50,
                "model_path": path.name,
                "model_sha256": audit.sha256(path),
                "numeric_max_absolute_error": 0.0,
            }
        )
    return rows, original


def test_ten_refit_models_and_recorded_duration_arithmetic(tmp_path):
    rows, original = refits(tmp_path)
    summary, error = audit.refits_reference(rows, original, 50, tmp_path, tmp_path)
    assert error == 0 and summary["n"] == 10
    assert summary["max_ms"] == 0.004009
    assert np.isclose(summary["p50_ms"], 0.0040045)
    assert summary["exceeded_two_second_allowance"] == 0


@pytest.mark.parametrize("tamper", ["model", "duration", "inventory", "training_rows", "claim"])
def test_refit_mutations_rejected(tmp_path, tamper):
    rows, original = refits(tmp_path)
    if tamper == "model":
        changed = {k: v.copy() for k, v in original.items()}
        changed["weights"][0, 0] = 0.1
        path = tmp_path / rows[0]["model_path"]
        np.savez(path, **changed)
        rows[0]["model_sha256"] = audit.sha256(path)
    elif tamper == "duration":
        rows[0]["total_ns"] += 1
    elif tamper == "inventory":
        rows.pop()
    elif tamper == "training_rows":
        rows[0]["training_rows"] += 1
    else:
        rows[0]["numeric_max_absolute_error"] = 0.1
    with pytest.raises(ValueError):
        audit.refits_reference(rows, original, 50, tmp_path, tmp_path)


def binding_fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    shutil.copyfile(Path(audit.__file__).with_name("continual-plan.json"), tmp_path / "plan.json")
    for name in ("model", "trace", "features", "targets"):
        np.savez(tmp_path / f"{name}.npz", dummy=np.arange(3))
    item = {
        "participant_id": "T2",
        "session_id": "T2_day_250_decoding",
        "post_implant_day": 250,
        "n_bins": 30000,
        "feature_path": "features.npz",
        "target_path": "targets.npz",
        "feature_sha256": audit.sha256(tmp_path / "features.npz"),
        "target_sha256": audit.sha256(tmp_path / "targets.npz"),
    }
    index = write("index.json", {"role": "development", "sessions": [item]})
    events = [
        {
            "session_id": item["session_id"],
            "method": "periodic240",
            "threshold": 0.1,
            "event": "model_fitted",
            "event_bin": deploy,
            "fit": {"fit_stop_bin": deploy - 100, "training_rows": 50},
            "model_path": "model.npz",
            "model_sha256": audit.sha256(tmp_path / "model.npz"),
        }
        for deploy in (7600, 12100, 18100, 24100, 27100)
    ]
    row = {
        "session_id": item["session_id"],
        "method": "periodic240",
        "threshold": 0.1,
        "trace_path": "trace.npz",
        "trace_sha256": audit.sha256(tmp_path / "trace.npz"),
    }
    source = write(
        "source.json",
        {
            "status": "complete",
            "phase": "source",
            "study": "V7B",
            "plan": {"path": "plan.json", "sha256": audit.PLAN_SHA},
            "code_sha256": {},
            "index": {"path": "index.json", "sha256": audit.sha256(index)},
            "rows": [row],
            "events": events,
        },
    )
    receipt = write(
        "quality.json",
        {
            "status": "PASS",
            "stage": "source",
            "manifest_sha256": audit.sha256(source),
            "index_sha256": audit.sha256(index),
            "plan_sha256": audit.PLAN_SHA,
            "code_sha256": {},
        },
    )
    manifest = {
        "source_manifest_path": source.name,
        "source_manifest_sha256": audit.sha256(source),
        "source_audit_path": receipt.name,
        "source_audit_sha256": audit.sha256(receipt),
        "index_path": index.name,
        "index_sha256": audit.sha256(index),
        "plan_sha256": audit.PLAN_SHA,
        "source_sha256": {},
        "representative": {
            "participant": "T2",
            "session_id": item["session_id"],
            "post_implant_day": 250,
            "method": "periodic240",
            "threshold_context": 0.1,
            "fit_stop_seconds": 540,
            "deployment_seconds": 542,
            "training_rows": 50,
            "selection_rule": (
                "greatest numeric exposed T2 day; last complete periodic fit; no outcome selection"
            ),
        },
        "artifacts": {
            name: {"path": f"{name}.npz", "sha256": audit.sha256(tmp_path / f"{name}.npz")}
            for name in ("model", "trace", "features", "targets")
        },
    }
    return manifest, source, receipt


def test_runtime_binds_source_quality_and_exact_last_deployed_model(tmp_path):
    manifest, _, _ = binding_fixture(tmp_path)
    paths, item, event = audit.binding(manifest, tmp_path)
    assert item["post_implant_day"] == 250 and event["event_bin"] == 27100
    assert paths["model"] == tmp_path / "model.npz"


@pytest.mark.parametrize("tamper", ["status", "last_fit", "representative", "trace"])
def test_runtime_binding_mutations_rejected(tmp_path, tamper):
    manifest, source, receipt = binding_fixture(tmp_path)
    if tamper == "status":
        value = audit.read(receipt)
        value["status"] = "FAIL"
        receipt.write_text(json.dumps(value))
        manifest["source_audit_sha256"] = audit.sha256(receipt)
    elif tamper == "last_fit":
        value = audit.read(source)
        value["events"][-1]["event_bin"] += 1
        source.write_text(json.dumps(value))
        manifest["source_manifest_sha256"] = audit.sha256(source)
        value = audit.read(receipt)
        value["manifest_sha256"] = audit.sha256(source)
        receipt.write_text(json.dumps(value))
        manifest["source_audit_sha256"] = audit.sha256(receipt)
    elif tamper == "representative":
        manifest["representative"]["threshold_context"] = 0.5
    else:
        (tmp_path / "trace.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        audit.binding(manifest, tmp_path)


def test_runtime_auditor_imports_only_independent_references():
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "neural_experiments"
        ):
            assert "audit" in node.module.split(".")[-1]
