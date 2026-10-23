import json

import numpy as np
import pytest

from neural_experiments.v7 import diagnostics_continual as diagnostics


def windows(mse):
    return [
        {
            "start_bin": 7600 + i * 1500,
            "stop_bin": 9100 + i * 1500,
            "metrics": {"summed_mse": value},
        }
        for i, value in enumerate(mse)
    ]


def test_recovery_requires_better_output_and_counts_missing_windows():
    episodes = diagnostics.recovery_episodes(
        windows([1.2, 1.0, None, 1.1, 0.8, 1.3]), windows([1.0] * 6), 16600
    )
    assert episodes == [
        {
            "onset_bin": 7600,
            "end_bin": 15100,
            "censored": False,
            "observed_seconds": 150.0,
            "unavailable_windows": 1,
        },
        {
            "onset_bin": 15100,
            "end_bin": 16600,
            "censored": True,
            "observed_seconds": 30.0,
            "unavailable_windows": 0,
        },
    ]


def test_absent_recording_windows_do_not_extend_censoring():
    source = windows([1.2, None, None])
    assert diagnostics.recovery_episodes(source, windows([1.0] * 3), 9100) == [
        {
            "onset_bin": 7600,
            "end_bin": 9100,
            "censored": True,
            "observed_seconds": 30.0,
            "unavailable_windows": 0,
        }
    ]
    assert diagnostics.recovery_episodes(windows([1.0]), windows([1.0]), 9100) == []


def event(reason="neural_drift_threshold"):
    return {
        "event_bin": 10500,
        "participant": "T2",
        "session_id": "synthetic",
        "reason": reason,
        "request_probe": reason != "below_threshold",
    }


def test_false_alarm_is_past_active_error_proxy_and_never_uses_suffix():
    y = np.ones((12000, 2))
    pred = y.copy()
    use = np.ones(12000, dtype=bool)
    interrupted = np.zeros(12000, dtype=bool)
    interrupted[9000:9100] = True
    before = diagnostics.pre_decision(event(), pred, y, use, interrupted)
    assert before["proxy_false_alarm"] is True
    assert before["active_valid_rows"] == 2800
    pred[10500:] = 1e100
    y[10500:] = 1e90
    assert diagnostics.pre_decision(event(), pred, y, use, interrupted) == before
    pred[7600:10500] = -1
    poor = diagnostics.pre_decision(event("below_threshold"), pred, y, use, interrupted)
    assert poor["below_threshold_after_poor_output"] is True
    assert poor["proxy_false_alarm"] is None


def test_forced_requests_and_sparse_active_rows_are_not_classified_false_alarms():
    y = pred = np.ones((12000, 2))
    use = np.ones(12000, dtype=bool)
    interrupted = np.zeros(12000, dtype=bool)
    forced = diagnostics.pre_decision(event("scheduled_maximum_gap"), pred, y, use, interrupted)
    assert forced["proxy_false_alarm"] is None
    use[:] = False
    use[7600:7649] = True
    sparse = diagnostics.pre_decision(event(), pred, y, use, interrupted)
    assert sparse["pre_decision_mse_minus_zero"] is None
    assert sparse["proxy_false_alarm"] is None


def test_reject_mismatched_recovery_geometry():
    baseline = windows([1.0])
    baseline[0]["stop_bin"] -= 1
    with pytest.raises(ValueError, match="matched"):
        diagnostics.recovery_episodes(windows([1.1]), baseline, 9100)


def test_publisher_binds_trace_targets_and_exports_only_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    y = np.ones((12000, 2))
    target = tmp_path / "targets.npz"
    np.savez(target, y=y)
    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        prediction=y,
        eval_mask=np.ones(len(y), bool),
        interruption_mask=np.zeros(len(y), bool),
    )
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session_id": "synthetic",
                        "target_path": target.name,
                        "target_sha256": diagnostics.digest(target),
                    }
                ]
            }
        )
    )
    rows = [
        {
            "participant": "T2",
            "session_id": "synthetic",
            "method": method,
            "n_bins": 12000,
            "diagnostic_windows": windows([1.0]),
            "trace_path": trace.name,
            "trace_sha256": diagnostics.digest(trace),
        }
        for method in ("zero", "triggered240")
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "phase": "assessment",
                "index": {"path": index.name, "sha256": diagnostics.digest(index)},
                "rows": rows,
                "events": [{**event(), "method": "triggered240", "event": "probe_decision"}],
            }
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({"status": "PASS", "manifest_sha256": diagnostics.digest(manifest)})
    )
    output = tmp_path / "public.json"
    result = diagnostics.run(manifest, audit, output)
    assert result["trigger_summary"]["proxy_false_alarms"] == 1
    text = output.read_text()
    for private in (str(tmp_path), "trace_path", "target_path", '"prediction"', '"y"'):
        assert private not in text
    trace.write_bytes(b"changed")
    with pytest.raises(ValueError, match="trace or targets changed"):
        diagnostics.run(manifest, audit, tmp_path / "other.json")
