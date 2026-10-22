"""Published diagnostic helpers preserve arithmetic and reject unpinned inputs."""

import json
from pathlib import Path

import numpy as np
import pytest

from neural_experiments.v5 import diagnosis
from neural_experiments.v5 import diagnosis_followup as followup
from neural_experiments.v5 import diagnosis_pilot as pilot


@pytest.mark.parametrize("module", [pilot, followup])
def test_unrecognized_index_rejected_before_any_arrays_open(tmp_path, monkeypatch, module):
    for role in ("development", "new_sessions"):
        path = tmp_path / "data/neural_v4" / role / "index.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Arrays opened before exposed input authentication")

    monkeypatch.setattr(module.study, "load_case", forbidden)
    with pytest.raises(ValueError, match="Exposed input identity"):
        module.main(tmp_path / "output.json")
    assert not (tmp_path / "output.json").exists()


def test_error_decomposition_and_direction_weighting_are_distinct():
    angles = np.arange(100) * (2 * np.pi / 100)
    y = np.column_stack((np.cos(angles), np.sin(angles)))
    mask = np.ones(100, dtype=bool)
    measured = pilot.metric(y, np.tile([0.2, -0.1], (100, 1)), mask)
    assert measured["mse"] == pytest.approx(measured["bias_mse"] + measured["centered_mse"])
    assert measured["bias_mse"] == pytest.approx(0.025)
    y = np.tile([1.0, 0.0], (101, 1))
    y[91:] = [-1, 0]
    mask = np.ones(101, dtype=bool)
    mask[90] = False
    pred = np.tile([1.0, 0.0], (101, 1))
    assert followup.balance_loss(y, pred, mask, "bins", [(0, 101)]) == pytest.approx(0.2)
    assert followup.balance_loss(y, pred, mask, "octants", [(0, 101)]) == pytest.approx(1.0)
    assert followup.balance_loss(y, pred, mask, "go_runs", [(0, 101)]) == pytest.approx(1.0)


def test_reproduction_comparison_excludes_only_declared_execution_fields(tmp_path, monkeypatch):
    original, reproduced = tmp_path / "original", tmp_path / "reproduced"
    original.mkdir()
    reproduced.mkdir()
    old_pilot = {"elapsed_seconds": 1.0, "source_sha256": "original", "metrics": [0.1, -0.2]}
    new_pilot = {
        **old_pilot,
        "elapsed_seconds": 2.0,
        "source_sha256": diagnosis.digest(pilot.__file__),
    }
    for folder, value in ((original, old_pilot), (reproduced, new_pilot)):
        (folder / "t2_pilot.json").write_text(json.dumps(value))
    old_followup = {
        "elapsed_seconds": 1.0,
        "source_sha256": "original",
        "helper_source_sha256": "original",
        "parent_pilot_sha256": diagnosis.digest(original / "t2_pilot.json"),
        "metrics": {"r2": -0.1},
    }
    new_followup = {
        **old_followup,
        "elapsed_seconds": 3.0,
        "source_sha256": diagnosis.digest(followup.__file__),
        "helper_source_sha256": diagnosis.digest(pilot.__file__),
        "parent_pilot_sha256": diagnosis.digest(reproduced / "t2_pilot.json"),
    }
    for folder, value in ((original, old_followup), (reproduced, new_followup)):
        (folder / "t2_followup.json").write_text(json.dumps(value))
    monkeypatch.setattr(
        diagnosis, "ORIGINAL", {p.name: diagnosis.digest(p) for p in Path(original).glob("*.json")}
    )
    assert diagnosis.verify_reproduction(original, reproduced)["status"] == "passed"
    new_followup["metrics"]["r2"] = 0.01
    (reproduced / "t2_followup.json").write_text(json.dumps(new_followup))
    with pytest.raises(ValueError, match="scientific fields differ"):
        diagnosis.verify_reproduction(original, reproduced)
