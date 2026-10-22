"""Cost, abstention and publication boundaries for prospective calibration."""

import json

import pytest

from neural_experiments.v3.study import digest
from neural_experiments.v5 import metrics, public


def rows():
    result = []
    for person in ("T2", "T5", "T6", "T7", "T8", "T9", "T10", "T11"):
        for day in range(3):
            for method in metrics.METHODS:
                result.append(
                    {
                        "domain": "new_sessions",
                        "participant": person,
                        "session_id": f"{person}_{day}",
                        "method": method,
                        "label_seconds": 240 if method == "adaptive" else 120,
                        "available_label_seconds": 83 if day == 0 else 240,
                        "action": "abstain" if method == "adaptive" else "accept",
                        "eval_mask_sha256": "same",
                        "metrics": {
                            "r2": -0.01 if method == "adaptive" else 0.2,
                            "angular_error_degrees": 90 if method == "adaptive" else 50,
                        },
                    }
                )
    return result


def test_abstention_stays_in_accuracy_and_cannot_claim_recovery():
    result = metrics.summarize(rows())
    adaptive = result["domains"]["new_sessions"]["methods"]["adaptive"]
    assert adaptive["mean_r2"] == pytest.approx(-0.01)
    assert adaptive["abstained_sessions"] == 24
    assert adaptive["acceptance_coverage"] == 0
    assert adaptive["mean_requested_label_seconds"] == 240
    assert not any(result["gates"].values())


def test_unavailable_scores_still_contribute_requested_cost():
    data = rows()
    for row in data:
        if row["session_id"] == "T2_0":
            row["metrics"]["r2"] = None
    adaptive = metrics.summarize(data)["domains"]["new_sessions"]["methods"]["adaptive"]
    assert adaptive["scored_sessions"] == 23
    assert adaptive["requested_sessions"] == 24
    assert adaptive["mean_requested_label_seconds"] == 240


def test_paired_comparison_rejects_different_masks():
    data = rows()
    data[0]["eval_mask_sha256"] = "different"
    with pytest.raises(ValueError, match="masks"):
        metrics.summarize(data)


def test_publication_requires_exact_independent_audit(tmp_path):
    artifact = tmp_path / "artifact.json"
    audit = tmp_path / "audit.json"
    artifact.write_text(json.dumps({"stage": "source"}))
    audit.write_text(json.dumps({"status": "PASS", "selection_sha256": "wrong"}))
    with pytest.raises(ValueError, match="audit"):
        public.publish(artifact, audit, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_publication_strips_paths_and_unknown_sample_arrays(tmp_path):
    artifact = tmp_path / "artifact.json"
    audit = tmp_path / "audit.json"
    output = tmp_path / "out.json"
    artifact.write_text(
        json.dumps(
            {
                "stage": "source",
                "status": "complete",
                "cases": [
                    {
                        "participant": "T2",
                        "model_path": "/Users/private/model.npz",
                        "prediction": [1, 2],
                        "model_sha256": "hash",
                        "metrics": {"r2": 0.1},
                    }
                ],
                "plan": {},
                "code_sha256": {},
                "candidates": [],
                "policies": [],
                "choices": [],
                "fresh_labels_opened": False,
                "elapsed_seconds": 1,
                "development_index": {"sha256": "dev"},
                "fits": [],
                "history_probes": [],
                "evidence": [],
            }
        )
    )
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "selection_sha256": digest(artifact),
                "private_path": "/Users/private",
            }
        )
    )
    public.publish(artifact, audit, output)
    result = json.loads(output.read_text())
    assert set(result["cases"][0]) == {"participant", "model_sha256", "metrics"}
    assert "/Users/" not in output.read_text()
