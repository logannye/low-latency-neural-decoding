"""Publish audit-bound aggregate V6 evidence without private arrays or models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_experiments.v3.public import safe_write
from neural_experiments.v3.study import digest
from neural_experiments.v5.public import AUDIT_FIELDS

ROW_FIELDS = {
    "participant",
    "session_id",
    "post_implant_day",
    "domain",
    "method",
    "action",
    "training_label_seconds",
    "label_seconds",
    "available_label_seconds",
    "policy_id",
    "label_requests",
    "model_sha256",
    "trace_sha256",
    "eval_mask_sha256",
    "metrics",
    "zero_mse",
    "daily240_r2",
}
DECISION_FIELDS = {
    "action",
    "training_seconds",
    "requested_label_seconds",
    "available_label_seconds",
    "decision_time_seconds",
    "candidate_id",
    "model_sha256",
    "reason",
    "events",
    "floor",
}
EVENT_FIELDS = {"training_seconds", "model_sha256", "candidate_id", "accepted", "reason"}
FLOOR_FIELDS = {"training_seconds", "reason", "source_session_ids", "paired_comparisons"}
COMPARISON_FIELDS = {
    "earlier_training_seconds",
    "later_training_seconds",
    "paired_cases",
    "source_session_ids",
    "mean_120_r2",
    "mean_later_r2",
    "mean_paired_gain",
    "positive_paired_gains",
    "triggered",
}
METRIC_FIELDS = {
    "r2",
    "per_output_r2",
    "n",
    "sse",
    "sst",
    "angular_error_degrees",
    "prediction_norm_rms",
    "zero_prediction_count",
    "summed_mse",
}


def subset(value, keys):
    return {k: v for k, v in value.items() if k in keys}


def public_row(row):
    result = subset(row, ROW_FIELDS)
    result["metrics"] = subset(row["metrics"], METRIC_FIELDS)
    if "decision" in row:
        decision = subset(row["decision"], DECISION_FIELDS)
        decision["events"] = [subset(e, EVENT_FIELDS) for e in decision["events"]]
        floor = subset(decision["floor"], FLOOR_FIELDS)
        floor["paired_comparisons"] = [
            subset(c, COMPARISON_FIELDS) for c in floor["paired_comparisons"]
        ]
        decision["floor"] = floor
        result["decision"] = decision
    return result


def publish(artifact_path, audit_path, output):
    artifact = json.loads(Path(artifact_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    stage = artifact["stage"]
    if stage not in ("source", "assessment"):
        raise ValueError("Unrecognized study stage")
    binding = "selection_sha256" if stage == "source" else "assessment_sha256"
    if audit.get("status") != "PASS" or audit.get(binding) != digest(artifact_path):
        raise ValueError("Exact passed independent audit required for publication")
    if artifact["status"] != ("complete" if stage == "source" else "assessment_complete"):
        raise ValueError("Only complete studies can publish")
    result = {
        "schema_version": 1,
        "stage": stage,
        "status": "complete",
        binding: digest(artifact_path),
        "audit_sha256": digest(audit_path),
        "independent_audit": subset(audit, AUDIT_FIELDS | {"index_sha256"}),
        "cases": [public_row(row) for row in artifact["cases"]],
        "data_policy": "Aggregate metrics, decisions and hashes only. Raw neural features, "
        "labels, coefficients, models and sample traces remain local.",
    }
    if stage == "source":
        for key in (
            "plan",
            "code_sha256",
            "candidates",
            "policies",
            "choices",
            "fresh_labels_opened",
            "elapsed_seconds",
        ):
            result[key] = artifact[key]
        result["development_index_sha256"] = artifact["development_index"]["sha256"]
        result["source_evidence_counts"] = {
            "fits": len(artifact["fits"]),
            "history_probes": len(artifact["history_probes"]),
            "sessions": len(artifact["evidence"]),
            "stages": sum(len(e["stages"]) for e in artifact["evidence"]),
        }
    else:
        result.update(
            selection_sha256=artifact["selection_sha256"],
            source_audit_sha256=artifact["source_audit_sha256"],
            assessment_index_sha256=artifact["index"]["sha256"],
            summary=artifact["summary"],
            data_dispositions=artifact["data_dispositions"],
        )
        result["calibration_compute"] = [
            subset(e, {"participant", "session_id", "timing"}) for e in artifact["evidence"]
        ]
    safe_write(output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    publish(a.artifact, a.audit, a.output)


if __name__ == "__main__":
    main()
