import ast
import copy
import inspect
import json

import numpy as np
import pytest

from neural_experiments.v7 import audit_diagnostics_continual as audit
from neural_experiments.v7 import diagnostics_continual as producer


def windows(values):
    return [
        {
            "start_bin": 7600 + i * 1500,
            "stop_bin": 9100 + i * 1500,
            "metrics": None if v is None else {"summed_mse": v},
        }
        for i, v in enumerate(values)
    ]


def test_neutral_missing_and_bad_windows_do_not_fake_recovery():
    observed = windows([1.1, 1.0, None, 1.2, 0.9, 1.2])
    zero = windows([1.0, 1.0, None, 1.0, 1.0, 1.0])
    expected = audit.episodes_reference(observed, zero, 16600)
    assert expected == [
        {
            "onset_bin": 7600,
            "unavailable_windows": 1,
            "end_bin": 15100,
            "censored": False,
            "observed_seconds": 150.0,
        },
        {
            "onset_bin": 15100,
            "unavailable_windows": 0,
            "end_bin": 16600,
            "censored": True,
            "observed_seconds": 30.0,
        },
    ]
    assert producer.recovery_episodes(observed, zero, 16600) == expected


@pytest.mark.parametrize(
    "reason", ["neural_drift_threshold", "below_threshold", "scheduled_maximum_gap"]
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_active_prefix_error_and_forced_requests_remain_distinct(reason, sign):
    trace = {
        "prediction": np.full((14000, 2), sign * 0.2),
        "eval_mask": np.ones(14000, bool),
        "interruption_mask": np.zeros(14000, bool),
    }
    trace["interruption_mask"][10500:12100] = True
    y = np.ones((14000, 2), np.float32)
    event = {
        "event_bin": 13500,
        "participant": "T2",
        "session_id": "test",
        "reason": reason,
        "request_probe": reason != "below_threshold",
    }
    expected = audit.decision_reference(event, trace, y)
    audit.compare_structure(
        producer.pre_decision(
            event, trace["prediction"], y, trace["eval_mask"], trace["interruption_mask"]
        ),
        expected,
    )
    assert expected["active_valid_rows"] == 1400
    if reason == "scheduled_maximum_gap":
        assert expected["proxy_false_alarm"] is None
    trace["prediction"][13500:] = np.nan
    audit.compare_structure(audit.decision_reference(event, trace, y), expected)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    code = tmp_path / "neural_experiments/v7"
    code.mkdir(parents=True)
    (code / "DIAGNOSTICS.md").write_text("locked descriptive definitions")
    (code / "diagnostics_continual.py").write_bytes(inspect.getsource(producer).encode())
    n = 15100
    y = np.c_[np.ones(n), np.zeros(n)].astype(np.float32)
    target = tmp_path / "targets.npz"
    np.savez(target, y=y, eval_mask=np.ones(n, bool))
    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        prediction=np.tile([0.5, 0], (n, 1)),
        eval_mask=np.ones(n, bool),
        interruption_mask=np.zeros(n, bool),
    )
    item = {"session_id": "s", "target_path": target.name, "target_sha256": audit.sha256(target)}
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"sessions": [item]}))
    rows = [
        {
            "participant": "T2",
            "session_id": "s",
            "method": m,
            "n_bins": n,
            "diagnostic_windows": windows(v),
            "trace_path": trace.name,
            "trace_sha256": audit.sha256(trace),
        }
        for m, v in (("zero", [1, 1, 1, 1, 1]), ("triggered240", [1.1, 1, None, 1.2, 0.9]))
    ]
    rows[0]["diagnostic_windows"][2]["metrics"] = None
    events = [
        {
            "participant": "T2",
            "session_id": "s",
            "method": "triggered240",
            "event": "probe_decision",
            "event_bin": 10500,
            "reason": "neural_drift_threshold",
            "request_probe": True,
        }
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "phase": "assessment",
                "index": {"path": index.name, "sha256": audit.sha256(index)},
                "rows": rows,
                "events": events,
            }
        )
    )
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps(
            {"status": "PASS", "stage": "assessment", "manifest_sha256": audit.sha256(manifest)}
        )
    )
    output = tmp_path / "diagnostics.json"
    # Producer code/plan identities use its own source location; replace only these test paths.
    result = producer.run(manifest, quality, output)
    result["diagnostic_plan_sha256"] = audit.sha256(code / "DIAGNOSTICS.md")
    result["diagnostic_code_sha256"] = audit.sha256(code / "diagnostics_continual.py")
    output.write_text(json.dumps(result))
    return manifest, quality, output


def test_complete_diagnostics_reconstruct(tmp_path, monkeypatch):
    manifest, quality, output = fixture(tmp_path, monkeypatch)
    result = audit.audit_run(manifest, quality, output, tmp_path)
    assert result["status"] == "PASS"
    assert result["session_method_rows_checked"] == 2
    assert result["decisions_checked"] == 1


@pytest.mark.parametrize("field", ["episode", "false_alarm", "summary", "audit", "trace"])
def test_diagnostic_mutations_rejected(tmp_path, monkeypatch, field):
    manifest, quality, output = fixture(tmp_path, monkeypatch)
    result = copy.deepcopy(audit.read(output))
    if field == "episode":
        result["rows"][1]["episodes"][0]["censored"] = True
    elif field == "false_alarm":
        result["decisions"][0]["proxy_false_alarm"] = False
    elif field == "summary":
        result["summary"]["triggered240"]["mean_resolved_delay_seconds"] -= 30
    elif field == "audit":
        quality.write_text(
            json.dumps(
                {"status": "FAIL", "stage": "assessment", "manifest_sha256": audit.sha256(manifest)}
            )
        )
    else:
        (tmp_path / "trace.npz").write_bytes(b"changed")
    output.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        audit.audit_run(manifest, quality, output, tmp_path)


def test_diagnostics_auditor_has_no_producer_import():
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "neural_experiments"
        ):
            assert "audit" in node.module.split(".")[-1]
