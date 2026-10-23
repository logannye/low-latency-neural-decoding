"""Frozen descriptive recovery episodes and pre-request error diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v3.study import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-12


def recovery_episodes(windows, zero_windows, n_bins):
    """Neutral/unknown windows never resolve an episode; end-of-file censors it."""
    episodes, active = [], None
    for window, zero in zip(windows, zero_windows, strict=True):
        if (window["start_bin"], window["stop_bin"]) != (zero["start_bin"], zero["stop_bin"]):
            raise ValueError("matched diagnostic windows required")
        if window["start_bin"] >= n_bins:
            continue
        values, baseline = window["metrics"], zero["metrics"]
        if values is None or values["summed_mse"] is None:
            if active is not None:
                active["unavailable_windows"] += 1
            continue
        if baseline is None or baseline["summed_mse"] is None:
            raise ValueError("matched baseline availability required")
        delta = values["summed_mse"] - baseline["summed_mse"]
        if delta > TOLERANCE and active is None:
            active = {"onset_bin": window["start_bin"], "unavailable_windows": 0}
        elif delta < -TOLERANCE and active is not None:
            end = window["stop_bin"]
            episodes.append(
                {
                    **active,
                    "end_bin": end,
                    "censored": False,
                    "observed_seconds": (end - active["onset_bin"]) * 0.02,
                }
            )
            active = None
    if active is not None:
        episodes.append(
            {
                **active,
                "end_bin": n_bins,
                "censored": True,
                "observed_seconds": (n_bins - active["onset_bin"]) * 0.02,
            }
        )
    return episodes


def pre_decision(event, prediction, targets, score_mask, interrupted):
    stop = event["event_bin"]
    start = max(7600, stop - 3000)
    use = score_mask[start:stop] & ~interrupted[start:stop]
    record = {
        "participant": event["participant"],
        "session_id": event["session_id"],
        "event_bin": stop,
        "start_bin": start,
        "reason": event["reason"],
        "request_probe": event["request_probe"],
        "active_valid_rows": int(use.sum()),
        "pre_decision_mse_minus_zero": None,
        "proxy_false_alarm": None,
        "below_threshold_after_poor_output": None,
    }
    if use.sum() < 50:
        return record
    y = np.asarray(targets[start:stop][use], dtype=np.float64)
    pred = np.asarray(prediction[start:stop][use], dtype=np.float64)
    delta = float((np.square(y - pred).sum(1) - np.square(y).sum(1)).mean())
    record["pre_decision_mse_minus_zero"] = delta
    if event["reason"] == "neural_drift_threshold":
        record["proxy_false_alarm"] = bool(delta <= TOLERANCE)
    elif event["reason"] == "below_threshold":
        record["below_threshold_after_poor_output"] = bool(delta > TOLERANCE)
    return record


def run(manifest_path, audit_path, output):
    manifest = json.loads(Path(manifest_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if (
        manifest["status"] != "complete"
        or manifest["phase"] != "assessment"
        or audit["status"] != "PASS"
        or audit["manifest_sha256"] != digest(manifest_path)
    ):
        raise ValueError("completed fresh study and bound passed audit required")
    index_path = ROOT / manifest["index"]["path"]
    if digest(index_path) != manifest["index"]["sha256"]:
        raise ValueError("index changed")
    index = json.loads(index_path.read_text())
    items = {item["session_id"]: item for item in index["sessions"]}
    zeros = {r["session_id"]: r for r in manifest["rows"] if r["method"] == "zero"}
    rows, decisions = [], []
    for row in manifest["rows"]:
        episodes = recovery_episodes(
            row["diagnostic_windows"],
            zeros[row["session_id"]]["diagnostic_windows"],
            row["n_bins"],
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
                "mean_resolved_delay_seconds": float(np.mean(resolved)) if resolved else None,
            }
        )
        if row["method"] != "triggered240":
            continue
        trace_path = ROOT / row["trace_path"]
        item = items[row["session_id"]]
        target_path = index_path.parent / item["target_path"]
        if (
            digest(trace_path) != row["trace_sha256"]
            or digest(target_path) != item["target_sha256"]
        ):
            raise ValueError("trace or targets changed")
        with (
            np.load(trace_path, allow_pickle=False) as trace,
            np.load(target_path, allow_pickle=False) as targets,
        ):
            for event in manifest["events"]:
                if (
                    event["session_id"] == row["session_id"]
                    and event["method"] == "triggered240"
                    and event["event"] == "probe_decision"
                ):
                    decisions.append(
                        pre_decision(
                            event,
                            trace["prediction"],
                            targets["y"],
                            trace["eval_mask"],
                            trace["interruption_mask"],
                        )
                    )
    summary = {}
    for method in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == method]
        resolved = [e["observed_seconds"] for r in rr for e in r["episodes"] if not e["censored"]]
        summary[method] = {
            "resolved_episodes": len(resolved),
            "censored_episodes": sum(r["censored_episodes"] for r in rr),
            "mean_resolved_delay_seconds": float(np.mean(resolved)) if resolved else None,
        }
    assessable = [d for d in decisions if d["proxy_false_alarm"] is not None]
    skipped = [d for d in decisions if d["below_threshold_after_poor_output"] is not None]
    result = {
        "schema_version": 1,
        "evidence_role": "descriptive scorer-only diagnostics; no policy tuning",
        "manifest_sha256": digest(manifest_path),
        "audit_sha256": digest(audit_path),
        "diagnostic_plan_sha256": digest(Path(__file__).with_name("DIAGNOSTICS.md")),
        "diagnostic_code_sha256": digest(__file__),
        "rows": rows,
        "decisions": decisions,
        "summary": summary,
        "trigger_summary": {
            "neural_threshold_requests": sum(
                d["reason"] == "neural_drift_threshold" for d in decisions
            ),
            "assessable_neural_threshold_requests": len(assessable),
            "proxy_false_alarms": sum(d["proxy_false_alarm"] for d in assessable),
            "forced_requests": sum(
                d["request_probe"] and d["reason"] != "neural_drift_threshold" for d in decisions
            ),
            "assessable_below_threshold_decisions": len(skipped),
            "below_threshold_after_poor_output": sum(
                d["below_threshold_after_poor_output"] for d in skipped
            ),
        },
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.audit, args.output)
