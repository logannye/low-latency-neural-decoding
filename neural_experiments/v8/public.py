"""Allowlisted aggregate publication. Never copy trace arrays or model coefficients."""

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
    "config",
    "target",
    "metrics",
    "secondary",
    "score_mask_sha256",
    "n_bins",
    "valid_rows",
    "planned_label_seconds",
    "excluded_purchased_trial_count",
    "training_rows",
    "training_trials",
    "available_label_seconds",
    "purchased_label_seconds",
    "fit_seconds",
    "predict_seconds",
    "diagnostics",
    "trial_coverage",
    "train_mask_sha256",
    "training_mask_sha256",
    "secondary_mask_sha256",
)

TRAINING_FIELDS = (
    "normalization_start_bin",
    "normalization_stop_bin",
    "label_start_bin",
    "label_stop_bin",
    "last_training_bin",
    "training_rows",
    "training_trials",
    "training_mask_sha256",
    "training_weights_sha256",
    "training_labels_sha256",
    "weight_sum",
    "effective_sample_size",
    "effective_trials",
    "direction_octant_rows",
    "direction_octants_with_at_least50_rows",
    "fallback_zero",
    "feature_penalty",
    "free_intercept",
    "multiscale_design_divisor",
    "export_prediction_max_error",
)


def aggregate(manifest, audit):
    if manifest.get("study") != "V8" or manifest.get("status") != "complete":
        raise ValueError("complete V8 study required")
    if audit.get("status") != "PASS":
        raise ValueError("independent audit must pass")
    result = {
        "schema_version": 1,
        "study": "V8",
        "phase": manifest["phase"],
        "plan_sha256": manifest["plan"]["sha256"],
        "index_sha256": manifest["index"]["sha256"],
        "scientific_code_sha256": manifest["code_sha256"],
        "primary_method": manifest.get("primary_method"),
        "configurations": manifest.get("configurations"),
        "summary": manifest.get("summary"),
        "selection": manifest.get("selection"),
        "rows": [{k: r[k] for k in ROW_FIELDS if k in r} for r in manifest["rows"]],
        "evidence_role": (
            "Previously exposed T2 sessions; source development only"
            if manifest["phase"] == "development"
            else "Fresh fixed archive ranks from known participants; T2-selected recipe"
        ),
    }
    for row, original in zip(result["rows"], manifest["rows"], strict=True):
        row["training"] = {
            k: original["training"][k] for k in TRAINING_FIELDS if k in original.get("training", {})
        }
    return result


def publish(manifest_path, audit_path, output):
    manifest = json.loads(Path(manifest_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("audit must bind this exact manifest")
    result = aggregate(manifest, audit)
    result["private_manifest_sha256"] = digest(manifest_path)
    result["audit_sha256"] = digest(audit_path)
    write_json(output, result)
    return result


def seal_selection(manifest_path, audit_path, output):
    """Bind audited source choices before committing them for fresh execution."""
    manifest = json.loads(Path(manifest_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if manifest.get("phase") != "development":
        raise ValueError("only development produces a selection")
    aggregate(manifest, audit)
    if audit.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("selection audit must bind this exact manifest")
    selection = dict(manifest["selection"])
    selection.update(
        source_audit_status="PASS",
        source_manifest_sha256=digest(manifest_path),
        source_audit_sha256=digest(audit_path),
    )
    write_json(output, selection)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()
    publish(args.manifest, args.audit, args.output)
    if args.selection_output:
        seal_selection(args.manifest, args.audit, args.selection_output)
