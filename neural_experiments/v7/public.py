"""Publish allowlisted aggregate V7 evidence without samples or model arrays."""

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
    "n_bins",
    "metrics",
    "score_mask_sha256",
    "unique_requested_label_seconds",
)
WINDOW_FIELDS = (
    "participant",
    "session_id",
    "post_implant_day",
    "method",
    "start_bin",
    "stop_bin",
    "disposition",
    "metrics",
    "valid_rows",
    "score_mask_sha256",
    "model_sha256",
    "trace_sha256",
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
    "method",
    "update_bin",
    "label_start_bin",
    "label_stop_bin",
    "normalization",
    "embargo_seconds",
    "selected_interval_seconds",
    "available_label_seconds",
)


def aggregate(manifest):
    if manifest.get("status") != "complete" or manifest.get("phase") != "exposed_recoverability":
        raise ValueError("completed exposed recovery study required")
    windows = []
    for row in manifest["window_rows"]:
        window = {k: row[k] for k in WINDOW_FIELDS if k in row}
        fit = row.get("fit")
        window["fit"] = {k: fit[k] for k in FIT_FIELDS if k in fit} if fit else None
        windows.append(window)
    return {
        "schema_version": 1,
        "phase": "exposed_recoverability",
        "evidence_role": "development diagnostic; no fresh outcomes in this artifact",
        "plan_sha256": manifest["plan"]["sha256"],
        "index_sha256": manifest["index"]["sha256"],
        "scientific_code_sha256": manifest["code_sha256"],
        "rows": [{k: r[k] for k in ROW_FIELDS} for r in manifest["rows"]],
        "window_rows": windows,
        "summary": manifest["summary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(json.loads(args.manifest.read_text()))
    result["private_manifest_sha256"] = digest(args.manifest)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
