"""Allowlisted aggregate publication; never publish arrays, labels or coefficients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_experiments.v3.public import safe_write
from neural_experiments.v3.study import digest

ROW_FIELDS = {
    "participant",
    "session_id",
    "post_implant_day",
    "domain",
    "kind",
    "method",
    "candidate_id",
    "budget_seconds",
    "normalization_bins",
    "training_rows",
    "history",
    "fallback",
    "penalty",
    "bias",
    "metrics",
    "unavailable",
    "model_sha256",
    "trace_sha256",
    "eval_mask_sha256",
}
AUDIT_FIELDS = {
    "status",
    "selection_sha256",
    "assessment_sha256",
    "auditor_source_sha256",
    "plan_sha256",
    "development_index_sha256",
    "cases_checked",
    "models_checked",
    "scores_checked",
    "predictions_checked",
    "fit_checks",
    "choices_checked",
    "targeted_choices_checked",
    "max_prediction_error",
    "max_fit_residual",
    "elapsed_seconds",
    "fresh_labels_opened",
}


def publish(artifact_path, audit_path, output):
    artifact = json.loads(Path(artifact_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    stage = artifact["stage"]
    identity = "selection_sha256" if stage == "source" else "assessment_sha256"
    if audit.get("status") != "PASS" or audit.get(identity) != digest(artifact_path):
        raise ValueError("aggregate publication requires an exact passed independent audit")
    result = {
        "schema_version": 1,
        "stage": stage,
        "status": "complete",
        identity: digest(artifact_path),
        "audit_sha256": digest(audit_path),
        "independent_audit": {k: v for k, v in audit.items() if k in AUDIT_FIELDS},
        "cases": [{k: v for k, v in row.items() if k in ROW_FIELDS} for row in artifact["cases"]],
        "data_policy": (
            "Aggregate metrics and artifact hashes only; "
            "raw data, labels, models and predictions stay local."
        ),
    }
    if stage == "source":
        result.update(
            {
                k: artifact[k]
                for k in (
                    "plan",
                    "code_sha256",
                    "candidates",
                    "targeted_candidates",
                    "choices",
                    "targeted_choices",
                    "fresh_labels_opened",
                    "elapsed_seconds",
                )
            }
        )
        result["development_index_sha256"] = artifact["development_index"]["sha256"]
    elif stage == "assessment" and artifact["status"] == "assessment_complete":
        result.update(
            selection_sha256=artifact["selection_sha256"],
            source_audit_sha256=artifact["source_audit_sha256"],
            index_sha256={k: v["sha256"] for k, v in artifact["indices"].items()},
            summary=artifact["summary"],
        )
    else:
        raise ValueError("incomplete or unknown artifact")
    safe_write(output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    publish(args.artifact, args.audit, args.output)


if __name__ == "__main__":
    main()
