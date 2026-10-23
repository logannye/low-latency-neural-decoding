"""Separate reconstruction of descriptive V7B monitoring diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v7.audit import (
    ROOT,
    average,
    compare_structure,
    contained,
    numeric,
    read,
    sha256,
    unique,
    verify,
)


def episodes_reference(windows, zeros, n):
    """Track unresolved bad-output intervals without interpreting missing bins."""
    if len(windows) != len(zeros):
        raise ValueError("Unpaired diagnostic window inventory")
    output, onset, unavailable = [], None, 0
    for w, z in zip(windows, zeros, strict=True):
        if (w["start_bin"], w["stop_bin"]) != (z["start_bin"], z["stop_bin"]):
            raise ValueError("Unpaired diagnostic boundaries")
        if w["start_bin"] >= n:
            continue
        a = None if w["metrics"] is None else w["metrics"]["summed_mse"]
        b = None if z["metrics"] is None else z["metrics"]["summed_mse"]
        if (a is None) != (b is None):
            raise ValueError("Unpaired diagnostic availability")
        if a is None:
            unavailable += int(onset is not None)
            continue
        difference = a - b
        if onset is None:
            if difference > 1e-12:
                onset, unavailable = w["start_bin"], 0
        elif difference < -1e-12:
            output.append(
                {
                    "onset_bin": onset,
                    "unavailable_windows": unavailable,
                    "end_bin": w["stop_bin"],
                    "censored": False,
                    "observed_seconds": (w["stop_bin"] - onset) / 50,
                }
            )
            onset, unavailable = None, 0
    if onset is not None:
        output.append(
            {
                "onset_bin": onset,
                "unavailable_windows": unavailable,
                "end_bin": n,
                "censored": True,
                "observed_seconds": (n - onset) / 50,
            }
        )
    return output


def decision_reference(event, trace, y):
    stop = event["event_bin"]
    start = max(7600, stop - 3000)
    indices = (
        np.flatnonzero(trace["eval_mask"][start:stop] & ~trace["interruption_mask"][start:stop])
        + start
    )
    result = {
        "participant": event["participant"],
        "session_id": event["session_id"],
        "event_bin": stop,
        "start_bin": start,
        "reason": event["reason"],
        "request_probe": event["request_probe"],
        "active_valid_rows": len(indices),
        "pre_decision_mse_minus_zero": None,
        "proxy_false_alarm": None,
        "below_threshold_after_poor_output": None,
    }
    if len(indices) >= 50:
        prediction, target = trace["prediction"][indices], y[indices]
        difference = float(np.sum((prediction - target) ** 2 - target**2) / len(indices))
        result["pre_decision_mse_minus_zero"] = difference
        if event["reason"] == "neural_drift_threshold":
            result["proxy_false_alarm"] = difference <= 1e-12
        if event["reason"] == "below_threshold":
            result["below_threshold_after_poor_output"] = difference > 1e-12
    return result


def reference(manifest, arrays):
    """Arrays is a callback for checked triggered traces and scorer-only targets."""
    zero = unique([r for r in manifest["rows"] if r["method"] == "zero"], lambda r: r["session_id"])
    rows, decisions = [], []
    for row in manifest["rows"]:
        episodes = episodes_reference(
            row["diagnostic_windows"], zero[row["session_id"]]["diagnostic_windows"], row["n_bins"]
        )
        resolved = [e["observed_seconds"] for e in episodes if not e["censored"]]
        rows.append(
            {
                "participant": row["participant"],
                "session_id": row["session_id"],
                "method": row["method"],
                "episodes": episodes,
                "resolved_episodes": len(resolved),
                "censored_episodes": sum(e["censored"] for e in episodes),
                "mean_resolved_delay_seconds": average(resolved),
            }
        )
        if row["method"] == "triggered240":
            trace, y = arrays(row)
            decisions.extend(
                decision_reference(e, trace, y)
                for e in manifest["events"]
                if e["session_id"] == row["session_id"]
                and e["method"] == "triggered240"
                and e["event"] == "probe_decision"
            )
    summary = {}
    for method in sorted({r["method"] for r in rows}):
        episodes = [e for r in rows if r["method"] == method for e in r["episodes"]]
        resolved = [e["observed_seconds"] for e in episodes if not e["censored"]]
        summary[method] = {
            "resolved_episodes": len(resolved),
            "censored_episodes": sum(e["censored"] for e in episodes),
            "mean_resolved_delay_seconds": average(resolved),
        }
    triggered = [d for d in decisions if d["proxy_false_alarm"] is not None]
    skipped = [d for d in decisions if d["below_threshold_after_poor_output"] is not None]
    totals = {
        "neural_threshold_requests": sum(
            d["reason"] == "neural_drift_threshold" for d in decisions
        ),
        "assessable_neural_threshold_requests": len(triggered),
        "proxy_false_alarms": sum(d["proxy_false_alarm"] for d in triggered),
        "forced_requests": sum(
            d["request_probe"] and d["reason"] != "neural_drift_threshold" for d in decisions
        ),
        "assessable_below_threshold_decisions": len(skipped),
        "below_threshold_after_poor_output": sum(
            d["below_threshold_after_poor_output"] for d in skipped
        ),
    }
    return {"rows": rows, "decisions": decisions, "summary": summary, "trigger_summary": totals}


def audit_run(manifest_path, quality_audit, diagnostics, root=ROOT):
    root = Path(root)
    manifest, passed, observed = read(manifest_path), read(quality_audit), read(diagnostics)
    if (
        manifest["status"] != "complete"
        or manifest["phase"] != "assessment"
        or passed["status"] != "PASS"
        or passed["stage"] != "assessment"
        or passed["manifest_sha256"] != sha256(manifest_path)
    ):
        raise ValueError("Passed bound fresh quality audit required")
    index_path = contained(root, manifest["index"]["path"])
    verify(index_path, manifest["index"]["sha256"])
    items = unique(read(index_path)["sessions"], lambda r: r["session_id"])

    def arrays(row):
        item = items[row["session_id"]]
        trace_path = contained(root, row["trace_path"])
        target_path = contained(index_path.parent, item["target_path"])
        verify(trace_path, row["trace_sha256"])
        verify(target_path, item["target_sha256"])
        return numeric(trace_path), numeric(target_path)["y"]

    expected = {
        "schema_version": 1,
        "evidence_role": "descriptive scorer-only diagnostics; no policy tuning",
        "manifest_sha256": sha256(manifest_path),
        "audit_sha256": sha256(quality_audit),
        "diagnostic_plan_sha256": sha256(root / "neural_experiments/v7/DIAGNOSTICS.md"),
        "diagnostic_code_sha256": sha256(root / "neural_experiments/v7/diagnostics_continual.py"),
        **reference(manifest, arrays),
    }
    compare_structure(observed, expected, "independent descriptive monitoring diagnostics")
    return {
        "status": "PASS",
        "auditor_source_sha256": sha256(__file__),
        "manifest_sha256": sha256(manifest_path),
        "quality_audit_sha256": sha256(quality_audit),
        "diagnostics_sha256": sha256(diagnostics),
        "session_method_rows_checked": len(expected["rows"]),
        "decisions_checked": len(expected["decisions"]),
        "scope": (
            "Separate-code episode/censoring and past active-output error reconstruction; "
            "binds passed numeric quality audit; descriptive only."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "quality-audit", "diagnostics", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit receipts cannot be overwritten")
    result = audit_run(args.manifest, args.quality_audit, args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
