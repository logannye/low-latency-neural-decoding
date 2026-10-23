import json

import pytest

from neural_experiments.v7 import public_continual as public


def test_fresh_export_omits_sample_arrays_and_paths():
    manifest = {
        "status": "complete",
        "study": "V7B",
        "phase": "assessment",
        "plan": {"sha256": "p"},
        "index": {"sha256": "i"},
        "code_sha256": {},
        "summary": {"gates": {}},
        "source_selection": {"sha256": "s"},
        "rows": [
            {
                "method": "periodic240",
                "metrics": {"r2": 0.2},
                "trace_path": "/Users/private/traces.npz",
                "prediction": [1, 2],
                "diagnostic_windows": [
                    {"start_bin": 7600, "metrics": {"r2": 0.2}, "samples": [3, 4]}
                ],
            }
        ],
        "events": [
            {
                "type": "fit",
                "model_path": "/Users/private/model.npz",
                "fit": {"training_rows": 100, "weights": [1, 2]},
                "evidence": {"kl_mean": 0.2, "raw_features": [1, 2]},
            }
        ],
    }
    result = public.aggregate(manifest, {"status": "PASS"})
    text = json.dumps(result)
    for secret in ("/Users/", "prediction", "samples", "weights", "raw_features"):
        assert secret not in text
    assert result["events"][0]["fit"] == {"training_rows": 100}


def test_failed_audit_prevents_publication():
    with pytest.raises(ValueError, match="passed"):
        public.aggregate({"status": "complete", "study": "V7B"}, {"status": "FAIL"})


def test_wrong_audit_manifest_cannot_seal_selection(tmp_path):
    manifest, audit = tmp_path / "manifest.json", tmp_path / "audit.json"
    manifest.write_text("{}")
    audit.write_text('{"manifest_sha256":"wrong"}')
    output = tmp_path / "public.json"
    with pytest.raises(ValueError, match="another manifest"):
        public.publish(manifest, audit, output, tmp_path / "selection.json")
    assert not output.exists()
