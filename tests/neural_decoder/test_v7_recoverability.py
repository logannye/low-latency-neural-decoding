import numpy as np
import pytest

from neural_experiments.v7 import metrics, recoverability


def case(n=30000):
    rng = np.random.default_rng(773)
    features = {key: rng.random((n, 1)) for key in ("x_tx3", "x_tx45", "x_sbp")}
    angles = np.arange(n) * 0.01
    return {
        "item": {
            "participant_id": "T2",
            "session_id": "T2_demo",
            "post_implant_day": 3,
            "n_channels": 1,
        },
        "features": features,
        "segments": [(0, n)],
        "y": np.column_stack((np.cos(angles), np.sin(angles))),
        "eval_mask": np.ones(n, bool),
    }


def test_shared_pooled_masks_requested_label_unions_and_artifact_geometry():
    source = case()
    artifacts = []

    def save(method, start, stop, fitted, prediction, mask):
        artifacts.append((method, start, stop, fitted, prediction.copy(), mask.copy()))
        return {"trace_test": True}

    rows, windows = recoverability.evaluate_session(source, save_artifact=save)
    assert len(rows) == 8 and len(windows) == len(artifacts) == 48
    assert len({row["score_mask_sha256"] for row in rows}) == 1
    expected_costs = {
        "zero": 0,
        "frozen120": 120,
        "expanding": 540,
        "rolling60": 360,
        "rolling120": 420,
        "rolling240": 540,
        "exponential60": 540,
        "rolling120_local": 420,
    }
    for row in rows:
        assert row["metrics"]["n"] == 16400
        assert row["unique_requested_label_seconds"] == expected_costs[row["method"]]
    for method, start, stop, fitted, prediction, mask in artifacts:
        assert prediction.shape == (stop - start, 2) and mask.shape == (stop - start,)
        assert (fitted is None) == (method == "zero")
        assert stop <= min(start + 3000, 30000)


def test_sparse_window_is_reported_but_excluded_from_pooled_score():
    source = case()
    source["eval_mask"][13600:16600] = False
    source["eval_mask"][13600:13649] = True
    rows, windows = recoverability.evaluate_session(source)
    assert all(r["metrics"]["n"] == 13400 for r in rows)
    sparse = [w for w in windows if w["start_bin"] == 13600]
    assert all(w["valid_rows"] == 49 and w["metrics"]["r2"] is None for w in sparse)
    assert all(w["disposition"] == "insufficient_labels" for w in sparse)


@pytest.mark.parametrize("n", [1000, 7500, 13600])
def test_short_sessions_retained_without_fitting(n):
    rows, windows = recoverability.evaluate_session(case(n))
    assert len(rows) == 8 and len(windows) == 48
    assert all(r["metrics"]["r2"] is None for r in rows)
    assert all(r["unique_requested_label_seconds"] == 0 for r in rows)
    assert all(w["disposition"] == "recording_ended_before_forecast_window" for w in windows)


def test_training_changes_only_after_scoring_window_has_passed():
    source = case()
    first, _ = recoverability.evaluate_session(source)
    source["y"][16600:] *= -1
    artifacts = []

    def capture(method, start, stop, fitted, prediction, mask):
        if start == 13600:
            artifacts.append((method, prediction))
        return {}

    _, changed = recoverability.evaluate_session(source, save_artifact=capture)
    # A future-label intervention changes later scores, while the first model and
    # its first-window prediction are unaffected for every method.
    original_artifacts = []
    source["y"][16600:] *= -1
    recoverability.evaluate_session(
        source,
        save_artifact=lambda method, start, stop, fitted, prediction, mask: (
            (original_artifacts.append((method, prediction.copy())) if start == 13600 else None)
            or {}
        ),
    )
    for (a, x), (b, z) in zip(artifacts, original_artifacts, strict=True):
        assert a == b
        np.testing.assert_array_equal(x, z)
    assert len(first) == 8 and len(changed) == 48


def test_evaluation_summary_integration_preserves_all_methods_and_primary_selection():
    rows, windows = [], []
    for day in range(8):
        source = case(14000 if day != 7 else 7500)
        source["item"].update(session_id=f"T2_day_{day}", post_implant_day=day)
        r, w = recoverability.evaluate_session(source)
        rows.extend(r)
        windows.extend(w)
    summary = metrics.summarize(rows, windows)
    assert summary["selected_sessions"] == 8
    assert summary["primary_T2_session_ids"] == [f"T2_day_{day}" for day in range(2, 8)]
    assert not summary["gate"]["passed"]
    for method in recoverability.METHODS:
        assert summary["methods"][method]["scored_sessions"] == 7
        assert summary["primary_T2"][method]["scored_sessions"] == 5
        assert not summary["primary_T2"][method]["conditions"]["all_six_scorable"]
