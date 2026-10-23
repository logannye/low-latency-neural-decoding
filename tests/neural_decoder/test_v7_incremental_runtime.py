import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neural_experiments.v7 import incremental_runtime as runner
from neural_experiments.v7 import model


def test_runner_replays_all_purchased_fits_and_preserves_raw_clocks(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    n = 2000
    features = {k: rng.random((n, 2)) for k in model.core.ARRAY_KEYS}
    features["sequence_id"] = np.zeros(n, int)
    raw = model.representation(
        {k: features[k] for k in model.core.ARRAY_KEYS}, [(0, n)], model.CONFIG
    )
    targets = {"y": rng.normal(size=(n, 2)), "eval_mask": np.ones(n, bool)}
    targets["eval_mask"][::6] = False
    events = []
    for ordinal, stop in enumerate((1600, 1700, 1800, 1900, 2000)):
        paid = targets["eval_mask"].copy()
        paid[:1500] = False
        paid[stop:] = False
        fitted, receipt = model.fit_mask(raw, targets["y"], paid, [(0, n)], 2, stop)
        filename = f"original-{ordinal}.npz"
        fitted.save(tmp_path / filename)
        events.append(
            {
                "session_id": "T2_test",
                "method": "periodic240",
                "threshold": 0.1,
                "event": "model_fitted",
                "event_bin": stop + 100,
                "model_path": filename,
                "model_sha256": runner.digest(tmp_path / filename),
                "fit": {**receipt, "purchased_intervals": [[1500, stop]]},
            }
        )
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"events": events}))
    index, audit = tmp_path / "index.json", tmp_path / "audit.json"
    index.write_text("{}")
    audit.write_text("{}")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SOURCES", ())
    monkeypatch.setattr(runner, "relative", lambda path: str(Path(path).relative_to(tmp_path)))
    monkeypatch.setattr(
        runner,
        "bind",
        lambda *args: (
            {"session_id": "T2_test", "n_bins": n, "n_channels": 2},
            None,
            {"features": Path("features"), "targets": Path("targets")},
        ),
    )
    monkeypatch.setattr(runner.data, "load_features", lambda path: features)
    monkeypatch.setattr(runner.data, "load_targets", lambda path, length: targets)
    out = tmp_path / "run"
    result = runner.run(source, index, audit, out)
    assert result["status"] == "complete" and len(result["fits"]) == 5
    assert result["memory"]["persistent_calibration_bytes"] == 256 + 64
    assert all(row["coefficient_max_error"] < 1e-12 for row in result["fits"])
    for row in result["fits"]:
        stop = row["fit_stop_bin"]
        expected = targets["eval_mask"][:stop].copy()
        expected[:1500] = False
        assert row["selected_mask_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    with np.load(out / "clocks.npz") as saved:
        clocks = saved["update_timestamps_ns"]
        geometry = saved["update_geometry"]
        assert np.all(clocks[:, 1] >= clocks[:, 0])
        assert np.all(clocks[:, 3] >= clocks[:, 2])
        assert np.all(geometry[:, 1] - geometry[:, 0] <= 64)
        assert saved["solve_timestamps_ns"].shape == (5, 6)
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(source, index, audit, out)
    assert (out / "manifest.json").read_bytes() == before
