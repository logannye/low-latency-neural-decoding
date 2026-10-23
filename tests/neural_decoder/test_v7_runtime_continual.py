import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from neural_experiments.v7 import model
from neural_experiments.v7 import runtime_continual as runtime


class FakeClock:
    def __init__(self):
        self.now = 0

    def perf_counter_ns(self):
        self.now += 1000
        return self.now


def fit_problem():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(30000, 4))
    y = rng.normal(size=(30000, 2))
    valid = rng.random(30000) > 0.1
    segments = [(0, 17000), (17000, 30000)]
    intervals = [[1500, 7500], [10500, 12000], [16500, 18000], [22500, 24000], [25500, 27000]]
    paid = np.zeros(30000, bool)
    for a, b in intervals:
        paid[a:b] = True
    expected, receipt = model.fit_mask(raw, y, valid & paid, segments, 2, 27000)
    event = {"fit": {**receipt, "purchased_intervals": intervals}}
    return raw, y, valid, paid, segments, event, expected


def test_ten_refits_export_exact_model_and_measure_complete_update(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "relative", lambda path: str(path))
    raw, y, valid, paid, segments, event, expected = fit_problem()
    # Targets after fitstop are inaccessible to fitting and may be unavailable.
    y[27000:] = np.nan
    measurements = runtime.refit_measurements(
        raw, y, valid, paid, segments, 2, event, expected, tmp_path, clock=FakeClock()
    )
    assert len(measurements) == 10
    assert all(r["fit_ns"] == r["export_reload_ns"] == 1000 for r in measurements)
    assert all(r["total_ns"] == 2000 for r in measurements)
    assert all(r["numeric_max_absolute_error"] == 0 for r in measurements)
    assert len(list(tmp_path.glob("refit-*.npz"))) == 10
    for receipt in measurements:
        assert runtime.digest(receipt["model_path"]) == receipt["model_sha256"]
        np.testing.assert_array_equal(
            model.Decoder.load(receipt["model_path"]).weights, expected.weights
        )


def test_refit_rejects_unpurchased_and_future_interval_tampering(tmp_path):
    raw, y, valid, paid, segments, event, expected = fit_problem()
    paid[9000] = True
    with pytest.raises(AssertionError):
        runtime.refit_measurements(raw, y, valid, paid, segments, 2, event, expected, tmp_path)
    paid[9000] = False
    event["fit"]["purchased_intervals"][-1][-1] = 27001
    with pytest.raises(ValueError, match="future"):
        runtime.refit_measurements(raw, y, valid, paid, segments, 2, event, expected, tmp_path)


def bound_fixture(tmp_path):
    plan = tmp_path / "plan.json"
    shutil.copyfile(Path(runtime.__file__).with_name("continual-plan.json"), plan)
    for name in ("model.npz", "trace.npz", "features.npz", "targets.npz"):
        np.savez(tmp_path / name, fixture=np.arange(3))
    index = {
        "role": "development",
        "sessions": [
            {
                "participant_id": "T2",
                "post_implant_day": day,
                "session_id": f"T2_day_{day}",
                "n_bins": 30000,
                "n_channels": 2,
                "feature_path": "features.npz",
                "target_path": "targets.npz",
                "feature_sha256": runtime.digest(tmp_path / "features.npz"),
                "target_sha256": runtime.digest(tmp_path / "targets.npz"),
            }
            for day in (214, 250)
        ],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    fits = [
        {
            "session_id": "T2_day_250",
            "method": "periodic240",
            "threshold": 0.1,
            "event": "model_fitted",
            "event_bin": second * 50,
            "fit": {"fit_stop_bin": second * 50 - 100},
            "model_path": "model.npz",
            "model_sha256": runtime.digest(tmp_path / "model.npz"),
        }
        for second in (152, 242, 362, 482, 542)
    ]
    source = {
        "status": "complete",
        "phase": "source",
        "study": "V7B",
        "plan": {"path": "plan.json", "sha256": runtime.PLAN_SHA256},
        "index": {"path": "index.json", "sha256": runtime.digest(index_path)},
        "code_sha256": {},
        "events": fits,
        "rows": [
            {
                "session_id": "T2_day_250",
                "method": "periodic240",
                "threshold": 0.1,
                "trace_path": "trace.npz",
                "trace_sha256": runtime.digest(tmp_path / "trace.npz"),
                "metrics": {"r2": -1e10},
            }
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    audit = {
        "status": "PASS",
        "stage": "source",
        "manifest_sha256": runtime.digest(source_path),
        "index_sha256": runtime.digest(index_path),
        "plan_sha256": runtime.PLAN_SHA256,
        "code_sha256": {},
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    return source_path, index_path, audit_path


def test_representative_bound_to_fixed_metadata_not_result_score(tmp_path):
    source_path, index_path, audit_path = bound_fixture(tmp_path)
    item, event, paths = runtime.bind(source_path, index_path, audit_path, root=tmp_path)
    assert item["post_implant_day"] == 250 and event["event_bin"] == 27100
    assert event["fit"]["fit_stop_bin"] == 27000
    assert paths["model"] == tmp_path / "model.npz"


@pytest.mark.parametrize("tamper", ["audit", "model", "escape", "last_fit"])
def test_binding_rejects_changed_model_audit_and_representative(tmp_path, tamper):
    source_path, index_path, audit_path = bound_fixture(tmp_path)
    if tamper == "audit":
        audit = json.loads(audit_path.read_text())
        audit["status"] = "FAIL"
        audit_path.write_text(json.dumps(audit))
    elif tamper == "model":
        (tmp_path / "model.npz").write_bytes(b"changed model")
    else:
        source = json.loads(source_path.read_text())
        if tamper == "escape":
            source["events"][-1]["model_path"] = "../outside.npz"
        else:
            source["events"][-1]["event_bin"] += 1
        source_path.write_text(json.dumps(source))
        audit = json.loads(audit_path.read_text())
        audit["manifest_sha256"] = runtime.digest(source_path)
        audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        runtime.bind(source_path, index_path, audit_path, root=tmp_path)
