"""Allowlisted aggregate publication and audited source-selection sealing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_experiments.v3.study import digest, write_json

ROW_FIELDS = (
    "participant",
    "session_id",
    "post_implant_day",
    "method",
    "threshold",
    "n_bins",
    "metrics",
    "score_mask_sha256",
    "planned_label_seconds",
    "unique_requested_label_seconds",
    "available_label_seconds",
    "interruption_seconds",
    "interrupted_valid_seconds",
    "interruption_mask_sha256",
    "purchased_mask_sha256",
    "completed_fits",
    "requested_probes",
    "trace_sha256",
)
EVENT_FIELDS = (
    "participant",
    "session_id",
    "post_implant_day",
    "method",
    "threshold",
    "event",
    "type",
    "event_bin",
    "label_start_bin",
    "label_stop_bin",
    "requested_label_seconds",
    "available_label_seconds",
    "reason",
    "scheduled_deployment_bin",
    "fit_stop_bin",
    "fit_ordinal",
    "model_sha256",
    "slot_index",
    "request_probe",
    "remaining_probes_before",
    "remaining_slots_including_current",
    "gap_since_last_probe_end_seconds",
    "forced_quota",
    "forced_gap",
)
FIT_FIELDS = (
    "fit_stop_bin",
    "normalization_start_bin",
    "normalization_stop_bin",
    "half_life_seconds",
    "training_rows",
    "training_mask_sha256",
    "training_weights_sha256",
    "weight_sum",
    "effective_sample_size",
    "first_training_bin",
    "last_training_bin",
    "feature_penalty",
    "free_intercept",
    "fallback_zero",
    "deployment_bin",
    "purchased_intervals",
    "purchased_mask_sha256",
)
EVIDENCE_FIELDS = (
    "kl_mean",
    "current_start_bin",
    "current_stop_bin",
    "reference_start_bin",
    "reference_stop_bin",
    "variance_floor",
    "kl_direction",
)


def aggregate(manifest, audit):
    if (
        manifest.get("status") != "complete"
        or manifest.get("study") != "V7B"
        or audit.get("status") != "PASS"
    ):
        raise ValueError("complete study and passed independent audit required")
    result = {
        "schema_version": 1,
        "study": "V7B",
        "phase": manifest["phase"],
        "plan_sha256": manifest["plan"]["sha256"],
        "index_sha256": manifest["index"]["sha256"],
        "scientific_code_sha256": manifest["code_sha256"],
        "summary": manifest["summary"],
        "rows": [{k: r[k] for k in ROW_FIELDS if k in r} for r in manifest["rows"]],
    }
    if manifest["phase"] == "source":
        result["selection"] = manifest["selection"]
        result["evidence_role"] = "exposed development; every threshold/session outcome retained"
    else:
        result["evidence_role"] = (
            "fresh sessions from known participants, following explicit V7A stopping-rule amendment"
        )
        result["source_selection_sha256"] = manifest["source_selection"]["sha256"]
        for exported, row in zip(result["rows"], manifest["rows"], strict=True):
            exported["diagnostic_windows"] = [
                {
                    k: w[k]
                    for k in ("start_bin", "stop_bin", "valid_rows", "metrics", "score_mask_sha256")
                    if k in w
                }
                for w in row["diagnostic_windows"]
            ]
        result["events"] = []
        for event in manifest["events"]:
            value = {k: event[k] for k in EVENT_FIELDS if k in event}
            if "fit" in event:
                value["fit"] = {k: event["fit"][k] for k in FIT_FIELDS if k in event["fit"]}
            if "evidence" in event:
                value["evidence"] = {
                    k: event["evidence"][k] for k in EVIDENCE_FIELDS if k in event["evidence"]
                }
            result["events"].append(value)
    return result


def publish(manifest_path, audit_path, output, selection_output=None):
    manifest, audit = (
        json.loads(Path(manifest_path).read_text()),
        json.loads(Path(audit_path).read_text()),
    )
    if audit.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("audit belongs to another manifest")
    result = aggregate(manifest, audit)
    result["private_manifest_sha256"] = digest(manifest_path)
    result["audit_sha256"] = digest(audit_path)
    write_json(output, result)
    if selection_output:
        if manifest["phase"] != "source":
            raise ValueError("only source runs produce policy choices")
        write_json(
            selection_output,
            {
                "schema_version": 1,
                "plan_sha256": manifest["plan"]["sha256"],
                "source_manifest_sha256": digest(manifest_path),
                "source_audit_sha256": digest(audit_path),
                "source_audit_status": audit["status"],
                "scientific_code_sha256": manifest["code_sha256"],
                "selection": manifest["selection"],
            },
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()
    publish(args.manifest, args.audit, args.output, args.selection_output)
