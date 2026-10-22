import json

import pytest

from neural_experiments.v4.public import publish
from neural_experiments.v4.study import digest


def test_export_requires_matching_independent_receipt_and_excludes_private_fields(tmp_path):
    source, audit, output = (tmp_path / f"{s}.json" for s in ("source", "audit", "public"))
    source.write_text(
        json.dumps(
            {
                "stage": "assessment",
                "status": "assessment_complete",
                "selection_sha256": "a",
                "source_audit_sha256": "b",
                "indices": {"new_sessions": {"path": "/Users/private", "sha256": "c"}},
                "summary": {},
                "cases": [
                    {
                        "participant": "T2",
                        "metrics": {"r2": 0.2},
                        "prediction": [[1, 0]],
                        "model_path": "/Users/private/model.npz",
                    }
                ],
            }
        )
    )
    audit.write_text(json.dumps({"status": "PASS", "assessment_sha256": "wrong"}))
    with pytest.raises(ValueError, match="exact passed"):
        publish(source, audit, output)
    assert not output.exists()
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "assessment_sha256": digest(source),
                "private_note": "/Users/private/labels",
            }
        )
    )
    publish(source, audit, output)
    text = output.read_text()
    assert "/Users" not in text
    assert '"prediction"' not in text
    assert json.loads(text)["cases"] == [{"participant": "T2", "metrics": {"r2": 0.2}}]
