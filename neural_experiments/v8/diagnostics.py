"""Descriptive target/coverage/feature diagnostics; no model or sample export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v3.study import digest, write_json
from neural_experiments.v8 import data


def _coverage(y, trial_id, mask):
    ids, counts = np.unique(trial_id[mask], return_counts=True)
    if np.any(ids < 0):
        raise ValueError("coverage mask contains non-trial observations")
    sector = np.floor((np.arctan2(y[mask, 1], y[mask, 0]) + np.pi) / (np.pi / 4)).astype(int) % 8
    hist = np.bincount(sector, minlength=8)
    probability = hist / max(1, hist.sum())
    positive = probability[probability > 0]
    return {
        "rows": int(mask.sum()),
        "distinct_trials": len(ids),
        "effective_equal_duration_trials": float(counts.sum() ** 2 / np.square(counts).sum())
        if len(ids)
        else 0.0,
        "octant_counts": hist.tolist(),
        "occupied_octants": int(np.count_nonzero(hist)),
        "normalized_octant_entropy": float(-(positive * np.log(positive)).sum() / np.log(8)),
        "mean_direction_norm": float(np.linalg.norm(y[mask].mean(0))) if mask.any() else None,
    }


def describe(case):
    item, context, y = case["item"], case["context"], case["y"].astype(float)
    n = len(y)
    valid = case["eval_mask"].copy()
    for a, b in case["segments"]:
        valid[a : min(a + 62, b)] = False
    trial = context["trial_id"]
    positions = np.arange(n)
    purchased = (positions >= 1500) & (positions < 7500)
    train = valid & purchased
    overlapping = np.unique(trial[purchased & (trial >= 0)])
    test = valid & (positions >= 7600) & ~np.isin(trial, overlapping)
    threshold = data.FAR_DISTANCE_BY_SESSION.get(
        item["session_id"], data.FAR_DISTANCE_BY_PARTICIPANT[item["participant_id"]]
    )
    far = context["reference40_available"] & (context["reference40_distance"] > threshold)
    phase = (context["trial_age_seconds"] >= 0.2) & (context["trial_age_seconds"] < 1.0) & far
    output = {
        "participant": item["participant_id"],
        "session_id": item["session_id"],
        "post_implant_day": item["post_implant_day"],
        "n_bins": n,
        "train": _coverage(y, trial, train),
        "test": _coverage(y, trial, test),
        "clean_train": _coverage(y, trial, train & phase & (context["trial_onset_seconds"] >= 30)),
        "clean_test": _coverage(y, trial, test & phase),
        "reference_far_distance": threshold,
    }
    a, b = np.array(output["train"]["octant_counts"]), np.array(output["test"]["octant_counts"])
    output["train_test_octant_total_variation"] = (
        float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum()) if a.sum() and b.sum() else None
    )
    output["target_geometry"] = {}
    for name, mask in [("train", train), ("test", test), ("clean_test", test & phase)]:
        use = mask & context["reference40_available"]
        angles = np.degrees(
            np.arccos(
                np.clip(np.sum(y[use] * context["reference40_direction"][use], axis=1), -1, 1)
            )
        )
        output["target_geometry"][name] = {
            "rows": int(mask.sum()),
            "far_reference_fraction": float(far[mask].mean()) if mask.any() else None,
            "current_near_target_fraction": float((context["distance"][mask] <= threshold).mean())
            if mask.any()
            else None,
            "reference_angle_median_degrees": float(np.median(angles)) if len(angles) else None,
            "reference_angle_p90_degrees": float(np.quantile(angles, 0.9)) if len(angles) else None,
            "reference_angle_over90_fraction": float((angles > 90).mean()) if len(angles) else None,
        }
    output["neural"] = {}
    for name in ("x_tx3", "x_tx45"):
        value = case["features"][name]
        output["neural"][name] = {
            "calibration_nonzero_fraction": float((value[purchased] > 0).mean())
            if purchased.any()
            else None,
            "calibration_active_channels": int(np.count_nonzero(value[purchased].sum(0)))
            if purchased.any()
            else 0,
            "channels": value.shape[1],
        }
    return output


def run(index_path, output):
    index_path = Path(index_path)
    index = data.load_index(index_path, verify_hashes=True)
    rows = [
        describe(data.load_case(index_path, item, labels=True, context=True))
        for item in index["sessions"]
    ]
    result = {
        "schema_version": 1,
        "study": "V8",
        "role": index["role"],
        "index_sha256": digest(index_path),
        "code_sha256": digest(__file__),
        "interpretation": (
            "Descriptive diagnostics only; no task-type, assistance "
            "or intention-ground-truth inference."
        ),
        "rows": rows,
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.index, args.output)
    print(json.dumps({"sessions": len(result["rows"])}))
