"""Audit-bound allowlisted aggregates; no neural arrays or model coefficients."""

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
    "method",
    "label_seconds",
    "available_label_seconds",
    "action",
    "candidate_id",
    "policy_id",
    "decision",
    "label_requests",
    "model_sha256",
    "trace_sha256",
    "eval_mask_sha256",
    "metrics",
    "daily120_r2",
}
AUDIT_FIELDS = {
    "stage",
    "source_cases_checked",
    "source_choices_checked",
    "full_traces_checked",
    "code_sha256",
    "assessment_cases_checked",
    "summary_checked",
    "status",
    "selection_sha256",
    "assessment_sha256",
    "manifest_sha256",
    "source_audit_sha256",
    "assessment_indices",
    "unavailable_scores",
    "auditor_source_sha256",
    "plan_sha256",
    "development_index_sha256",
    "cases_checked",
    "models_checked",
    "fits_checked",
    "history_probes_checked",
    "stages_checked",
    "decisions_checked",
    "choices_checked",
    "scores_checked",
    "max_prediction_error",
    "max_fit_residual",
    "max_prediction_absolute_error",
    "max_normal_equation_relative_residual",
    "source_code_sha256",
    "scope",
    "elapsed_seconds",
    "data_indices_checked",
}


def publish(artifact_path, audit_path, output):
    artifact = json.loads(Path(artifact_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    stage = artifact["stage"]
    binding = "selection_sha256" if stage == "source" else "assessment_sha256"
    if audit.get("status") != "PASS" or audit.get(binding) != digest(artifact_path):
        raise ValueError("Exact passed independent audit required for publication")
    expected = "complete" if stage == "source" else "assessment_complete"
    if stage not in ("source", "assessment") or artifact["status"] != expected:
        raise ValueError("Only complete studies can publish")
    result = {
        "schema_version": 1,
        "stage": stage,
        "status": "complete",
        binding: digest(artifact_path),
        "audit_sha256": digest(audit_path),
        "independent_audit": {k: v for k, v in audit.items() if k in AUDIT_FIELDS},
        "cases": [{k: v for k, v in row.items() if k in ROW_FIELDS} for row in artifact["cases"]],
        "data_policy": (
            "Aggregate metrics, decisions and artifact hashes only. "
            "Raw data, labels, coefficients, models and sample predictions stay local."
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
                    "policies",
                    "choices",
                    "fresh_labels_opened",
                    "elapsed_seconds",
                )
            }
        )
        result["development_index_sha256"] = artifact["development_index"]["sha256"]
        result["source_evidence_counts"] = {
            "fits": len(artifact["fits"]),
            "history_probes": len(artifact["history_probes"]),
            "sessions": len(artifact["evidence"]),
            "stages": sum(len(r["stages"]) for r in artifact["evidence"]),
        }
    else:
        result.update(
            selection_sha256=artifact["selection_sha256"],
            source_audit_sha256=artifact["source_audit_sha256"],
            index_sha256={k: v["sha256"] for k, v in artifact["indices"].items()},
            summary=artifact["summary"],
            data_dispositions=artifact["data_dispositions"],
        )
        result["calibration_compute"] = [
            {k: r[k] for k in ("domain", "participant", "session_id", "timing")}
            for r in artifact["evidence"]
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
