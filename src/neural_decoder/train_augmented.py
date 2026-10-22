"""One prespecified TCN channel-dropout comparison, scored over every seed/day."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import sha256, write_json
from .train_temporal import run_temporal


def compare_summaries(actual, baseline, decision):
    seeds, days = decision["all_seeds_required"], decision["all_days_required"]
    if [row["seed"] for row in actual["seeds"]] != seeds or [
        row["seed"] for row in baseline["seeds"]
    ] != seeds:
        raise ValueError("comparison requires every prespecified seed in the same order")
    if actual["sealed_days_scored"] or baseline["sealed_days_scored"]:
        raise ValueError("augmentation comparison is development-only")
    per_seed = []
    paired_days = {day: [] for day in days}
    for new, old in zip(actual["seeds"], baseline["seeds"], strict=True):
        a, b = new["scores"], old["scores"]
        if set(a["per_day"]) != set(days) or set(b["per_day"]) != set(days):
            raise ValueError("all forward development days are required")
        differences = {
            day: a["per_day"][day]["r2_variance_weighted"]
            - b["per_day"][day]["r2_variance_weighted"]
            for day in days
        }
        for day in days:
            paired_days[day].append(differences[day])
        per_seed.append(
            {
                "seed": new["seed"],
                "baseline_mean_r2": b["r2_mean"],
                "augmented_mean_r2": a["r2_mean"],
                "mean_r2_difference": a["r2_mean"] - b["r2_mean"],
                "per_day_r2_difference": differences,
            }
        )
    day_differences = {day: float(np.mean(values)) for day, values in paired_days.items()}
    gain = float(np.mean([row["mean_r2_difference"] for row in per_seed]))
    worst_harm = max(0.0, -min(day_differences.values()))
    criteria = {
        "mean_gain": gain >= decision["minimum_seed_average_mean_r2_gain"],
        "day_harm": worst_harm <= decision["maximum_seed_average_day_harm"],
    }
    return {
        "per_seed": per_seed,
        "seed_average_mean_r2_gain": gain,
        "seed_average_day_r2_differences": day_differences,
        "seed_average_worst_day_harm": worst_harm,
        "seed_average_days_harmed": [day for day, value in day_differences.items() if value < 0],
        "criteria": criteria,
        "retain_channel_dropout": all(criteria.values()),
        "selected_recipe": "tcn-dropout-v1" if all(criteria.values()) else "tcn-v1",
        "scope": "development selection; every seed/day retained; no best-seed selection",
    }


def run_augmented(index_path, baseline_run, config_path, output, *, root):
    config = json.loads(Path(config_path).read_text())
    if config.get("channel_dropout") != 0.1 or config.get("hypothesis_number") != 2:
        raise ValueError("only the second, fixed p=0.1 hypothesis is authorized here")
    baseline_run, output = Path(baseline_run), Path(output)
    baseline_manifest = json.loads((baseline_run / "manifest.json").read_text())
    baseline_hash = sha256(baseline_run / "manifest.json")
    if baseline_manifest["status"] != "complete" or baseline_manifest["index_sha256"] != sha256(
        index_path
    ):
        raise ValueError("baseline must be complete and use identical development inputs")
    if sha256(baseline_run / "summary.json") != baseline_manifest["summary_sha256"]:
        raise ValueError("baseline summary identity changed")
    baseline_summary = json.loads((baseline_run / "summary.json").read_text())
    original_config = baseline_manifest["configuration"]
    for key in (
        "architecture",
        "seeds",
        "epochs",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "chunk_scored_bins",
        "burn_in_bins",
        "batch_chunks",
        "threads",
        "checkpoint",
    ):
        if config[key] != original_config[key]:
            raise ValueError(f"ablation must keep the original recipe's {key}")
    run_temporal(index_path, config_path, output, root=root)
    manifest = json.loads((output / "manifest.json").read_text())
    try:
        if sha256(baseline_run / "manifest.json") != baseline_hash:
            raise ValueError("baseline changed during experiment")
        actual = json.loads((output / "summary.json").read_text())
        comparison = compare_summaries(actual, baseline_summary, config["decision"])
        write_json(output / "comparison.json", comparison)
        manifest.update(
            {
                "baseline_manifest_sha256": baseline_hash,
                "baseline_summary_sha256": baseline_manifest["summary_sha256"],
                "comparison_sha256": sha256(output / "comparison.json"),
            }
        )
    except BaseException as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_augmented(
        args.index,
        args.baseline_run,
        args.config,
        args.output,
        root=Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    main()
