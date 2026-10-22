import json

import pytest

from neural_experiments.v3.study import digest
from neural_experiments.v6.public import public_row, publish


def test_nested_decision_export_excludes_samples_and_private_paths():
    row = {
        "session_id": "T2_test",
        "model_path": "/Users/private/model.npz",
        "metrics": {"r2": 0.2, "prediction": [[1, 0]]},
        "decision": {
            "action": "accept",
            "model_path": "private.npz",
            "events": [{"reason": "pass", "weights": [1, 2]}],
            "floor": {
                "training_seconds": 240,
                "labels": [0, 1],
                "paired_comparisons": [{"triggered": True, "raw": [1, 2]}],
            },
        },
    }
    assert public_row(row) == {
        "session_id": "T2_test",
        "metrics": {"r2": 0.2},
        "decision": {
            "action": "accept",
            "events": [{"reason": "pass"}],
            "floor": {"training_seconds": 240, "paired_comparisons": [{"triggered": True}]},
        },
    }


def test_publication_requires_exact_completed_passed_artifact(tmp_path):
    artifact, audit, output = [tmp_path / f"{name}.json" for name in ("artifact", "audit", "out")]
    value = {
        "stage": "assessment",
        "status": "assessment_complete",
        "cases": [],
        "selection_sha256": "a" * 64,
        "source_audit_sha256": "b" * 64,
        "index": {"sha256": "c" * 64},
        "summary": {},
        "data_dispositions": [],
        "evidence": [],
    }
    artifact.write_text(json.dumps(value))
    audit.write_text(json.dumps({"status": "PASS", "assessment_sha256": digest(artifact)}))
    publish(artifact, audit, output)
    assert json.loads(output.read_text())["assessment_index_sha256"] == "c" * 64
    value["status"] = "running"
    artifact.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Exact passed"):
        publish(artifact, audit, output)
    audit.write_text(json.dumps({"status": "PASS", "assessment_sha256": digest(artifact)}))
    with pytest.raises(ValueError, match="complete"):
        publish(artifact, audit, output)
