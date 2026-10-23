import numpy as np

from neural_experiments.v6 import diagnosis, diagnosis_public


def test_gain_over_zero_does_not_imply_positive_own_window_r2():
    angles = np.linspace(-0.05, 0.05, 400)
    y = np.column_stack((np.cos(angles), np.sin(angles)))
    case = {"y": y, "mask": np.ones(400, bool), "segments": [(0, 400)]}
    prediction = np.tile([0.9, 0.0], (400, 1))
    evidence = diagnosis.window(case, prediction, 0, 400, burned=False)
    assert evidence["gain_vs_zero"] > 0.49
    assert evidence["r2"] < -1
    assert sum(n >= 10 for n in evidence["direction_sector_counts"]) < 4


def test_pooled_confirmation_cannot_substitute_for_each_half():
    good = {
        "n": 100,
        "mse": 0.3,
        "direction_sector_counts": [25] * 4 + [0] * 4,
        "gain_vs_zero": 0.2,
        "gain_vs_training_mean": 0.1,
    }
    bad = {**good, "gain_vs_training_mean": -0.01}
    evidence = {"pooled": good, "first_half": bad, "last_half": good}
    assert diagnosis.passed(evidence, 0.025, False)
    assert not diagnosis.passed(evidence, 0.025, True)


def test_public_metric_allowlist_drops_arrays_and_paths():
    source = {
        "n": 100,
        "r2": -0.1,
        "mask_sha256": "a" * 64,
        "raw_predictions": [[1.0, 0.0]],
        "private_path": "/Users/private",
        "model_weights": [[0.5]],
    }
    assert diagnosis_public.metrics(source) == {"n": 100, "r2": -0.1, "mask_sha256": "a" * 64}
