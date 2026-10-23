import json

import pytest

from neural_experiments.v7.public import aggregate


def test_aggregate_omits_paths_arrays_and_unrecognized_fit_fields():
    row = dict(
        participant="T2",
        session_id="T2_example",
        post_implant_day=1,
        method="rolling120",
        n_bins=30000,
        metrics={"r2": 0.2},
        score_mask_sha256="hash",
        unique_requested_label_seconds=420,
    )
    window = {
        **row,
        "start_bin": 13600,
        "stop_bin": 16600,
        "disposition": "scored",
        "valid_rows": 100,
        "model_path": "/Users/private/model.npz",
        "prediction": [1, 2],
        "fit": {"weights": [3, 4], "training_rows": 100},
    }
    result = aggregate(
        {
            "status": "complete",
            "phase": "exposed_recoverability",
            "plan": {"sha256": "p"},
            "index": {"sha256": "i"},
            "code_sha256": {},
            "rows": [row],
            "window_rows": [window],
            "summary": {"gate": {"passed": False}},
        }
    )
    text = json.dumps(result)
    assert "/Users/" not in text and "prediction" not in text and '"weights"' not in text
    assert result["window_rows"][0]["fit"] == {"training_rows": 100}


def test_incomplete_study_is_not_publishable():
    with pytest.raises(ValueError, match="completed"):
        aggregate({"status": "running"})
