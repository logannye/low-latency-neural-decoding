"""Publish every exposed-pilot fit's aggregate metrics and provenance, no arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from neural_experiments.v6.diagnosis_summary import summarize

ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "n",
    "r2",
    "mse",
    "bias_mse",
    "centered_mse",
    "zero_mse",
    "training_mean_mse",
    "gain_vs_zero",
    "gain_vs_training_mean",
    "direction_sector_counts",
    "start_bin",
    "stop_bin",
    "mask_sha256",
)
FIT_FIELDS = (
    "candidate_id",
    "head",
    "normalization",
    "normalization_stop_bin",
    "label_start_bin",
    "label_stop_bin",
    "requested_label_seconds",
    "training_rows",
    "training_mask_sha256",
    "fallback",
    "fallback_output",
    "feature_penalty",
    "intercept_penalty",
    "free_intercept",
    "constrained_zero_intercept",
    "prior_sha256",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keep(value, keys):
    return {key: value[key] for key in keys if key in value}


def metrics(value):
    return keep(value, METRICS)


def confirm(value):
    return {key: metrics(value[key]) for key in ("pooled", "first_half", "last_half")}


def export(run):
    run = Path(run)
    pilot = json.loads((run / "diagnosis.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    summary = summarize(run / "diagnosis.json")
    if (
        pilot["plan_sha256"] != digest(run / "plan.json")
        or plan["source_sha256"] != pilot["source_sha256"]
    ):
        raise ValueError("pilot plan/source binding differs")
    for name, expected in pilot["source_sha256"].items():
        if digest(run / "source" / name) != expected:
            raise ValueError("retained pilot source differs")
    fits = []
    for row in pilot["fits"]:
        expected = (
            run
            / "models"
            / f"{row['session_id']}-{row['budget']}-{row['candidate_id']}-{row['phase']}.npz"
        )
        if digest(expected) != row["model_sha256"]:
            raise ValueError("retained numeric fit differs")
        value = keep(
            row,
            (
                "session_id",
                "post_implant_day",
                "budget",
                "phase",
                "candidate_id",
                "model_sha256",
                "chosen_before_confirmation",
            ),
        )
        value["fit"] = keep(row["fit"], FIT_FIELDS)
        if row["phase"] == "inner":
            value["inner_validation"] = metrics(row["inner_validation"])
        else:
            value.update(confirmation=confirm(row["confirmation"]), common=metrics(row["common"]))
        fits.append(value)
    prior = []
    for row in pilot["prior0"]:
        if digest(run / "models" / f"{row['session_id']}-prior0.npz") != row["model_sha256"]:
            raise ValueError("retained prior differs")
        prior.append(
            {
                **keep(
                    row,
                    (
                        "session_id",
                        "post_implant_day",
                        "history_ids",
                        "model_sha256",
                        "label_seconds",
                    ),
                ),
                "confirmation": confirm(row["confirmation"]),
                "common": metrics(row["common"]),
            }
        )
    original = []
    for row in pilot["original_v5"]["rows"]:
        value = keep(
            row,
            (
                "session_id",
                "method",
                "candidate_id",
                "model_sha256",
                "trace_sha256",
                "trace_max_absolute_error",
                "original_inner_model_sha256",
            ),
        )
        value.update(
            old_endpoint=metrics(row["old_endpoint"]),
            common_endpoint=metrics(row["common_endpoint"]),
        )
        if "artifact_comparison" in row:
            value["artifact_comparison"] = {
                name: {arm: metrics(v) for arm, v in arms.items()}
                for name, arms in row["artifact_comparison"].items()
            }
        original.append(value)
    result = {
        "status": "exposed_exploratory_pilot_complete",
        "scope": summary["scope"],
        "pilot_sha256": summary["pilot_sha256"],
        "plan_sha256": pilot["plan_sha256"],
        "source_sha256": pilot["source_sha256"],
        "exporter_source_sha256": digest(__file__),
        "summary_source_sha256": summary["summary_source_sha256"],
        "plan": plan,
        "inputs": [
            keep(
                row,
                (
                    "session_id",
                    "post_implant_day",
                    "n_bins",
                    "index_sha256",
                    "feature_sha256",
                    "target_sha256",
                ),
            )
            for row in pilot["inputs"]
        ],
        "fit_count": len(fits),
        "prior_count": len(prior),
        "elapsed_seconds": pilot["elapsed_seconds"],
        "fits": fits,
        "prior0": prior,
        "stages": [
            {
                **keep(
                    row,
                    (
                        "session_id",
                        "post_implant_day",
                        "budget",
                        "candidate_id",
                        "model_sha256",
                        "selection_model_sha256",
                        "label_seconds",
                    ),
                ),
                "selection_metric": metrics(row["selection_metric"]),
            }
            for row in pilot["stages"]
        ],
        "sequential": [
            {
                **keep(
                    row,
                    (
                        "session_id",
                        "post_implant_day",
                        "margin",
                        "require_halves",
                        "action",
                        "candidate_id",
                        "label_seconds",
                        "training_budget",
                        "accepted_worse_than_zero",
                        "accepted_negative_r2",
                    ),
                ),
                "common": metrics(row["common"]),
                "zero_common": metrics(row["zero_common"]),
            }
            for row in pilot["sequential"]
        ],
        "fixed_aggregates": summary["fixed_aggregates"],
        "strictly_earlier_histories": summary["strictly_earlier_histories"],
        "original_v5": {
            "manifest_sha256": pilot["original_v5"]["manifest_sha256"],
            "rows": original,
        },
        "interpretation": summary["interpretation"],
        "data_policy": (
            "All fit metrics and failed choices retained; numeric models, "
            "per-bin samples, raw arrays and private paths excluded."
        ),
    }
    encoded = json.dumps(result, allow_nan=False)
    if any(token in encoded for token in ("/Users/", "/home/", "X-Amz-", "PRIVATE KEY")):
        raise ValueError("private content in aggregate")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "neural_experiments/results/v6-diagnosis.json"
    )
    args = parser.parse_args()
    result = export(args.run)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
