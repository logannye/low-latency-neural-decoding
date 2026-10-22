"""Published diagnostic helpers preserve arithmetic and reject unpinned inputs."""

import numpy as np
import pytest

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
