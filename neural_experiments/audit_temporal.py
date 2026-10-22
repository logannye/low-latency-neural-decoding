"""Independent whole-file inference and scoring audit for the fixed H1 TCN study.

No production TCN/model/metric functions are imported and no model is trained.
Saved float32 numeric weights drive functional Torch convolutions over complete
files. Inputs enter inference without targets, masks, or trial boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from neural_decoder.data import load_index

FOLDS = ((["S0", "S1"], "S2"), (["S0", "S1", "S2"], "S3"), (["S0", "S1", "S2", "S3"], "S4"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Changed artifact: {path}")


def read_json(path):
    return json.loads(Path(path).read_text())


def whole_file(counts, weights):
    """Feature-only calculation with standardized zero padding at each layer."""
    values = (counts.astype(np.float32) - weights["input_mean"]) / weights["input_scale"]
    hidden = torch.from_numpy(values.T.copy())[None]
    with torch.inference_mode():
        for layer, dilation in enumerate((1, 2, 4), 1):
            hidden = F.relu(
                F.conv1d(
                    F.pad(hidden, (2 * dilation, 0)),
                    torch.from_numpy(weights[f"conv{layer}_weight"]),
                    torch.from_numpy(weights[f"conv{layer}_bias"]),
                    dilation=dilation,
                )
            )
        output = F.linear(
            hidden.transpose(1, 2),
            torch.from_numpy(weights["head_weight"]),
            torch.from_numpy(weights["head_bias"]),
        )[0].numpy()
    return output * weights["target_scale"] + weights["target_mean"]


def audit(run, index_path):
    run, index_path = Path(run).resolve(), Path(index_path).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete" or manifest["pilot"]:
        raise ValueError("Requires a completed non-pilot study")
    config = manifest["configuration"]
    if config["seeds"] != [0, 1, 2] or config["regime"] != "frozen_zero_new_day_calibration":
        raise ValueError("Unexpected recipe/seeds")
    index = load_index(index_path)
    verify(index_path, manifest["index_sha256"])
    verify(run / "input_index.json", manifest["index_sha256"])
    verify(run / "summary.json", manifest["summary_sha256"])
    verify(run / "source.diff", manifest["source"]["diff_sha256"])
    for relative, expected in manifest["source"]["files"].items():
        verify(run / "source" / relative, expected)
    config_path = run / "source/neural_experiments/configs/tcn-v1.json"
    verify(config_path, manifest["config_sha256"])
    if config != read_json(config_path):
        raise ValueError("Configuration value does not match its hashed bytes")
    files = {}
    for record in index["files"]:
        if record["day"] not in {"S0", "S1", "S2", "S3", "S4"}:
            raise ValueError("Refusing sealed sample arrays")
        arrays = {}
        for kind in ("features", "targets"):
            path = index_path.parent / record[kind]
            verify(path, record[f"{kind}_sha256"])
            with np.load(path, allow_pickle=False) as bundle:
                arrays[kind] = {name: bundle[name] for name in bundle.files}
        files[record["file_id"]] = (record, arrays)
    transforms = {}
    for training_days, day in FOLDS:
        training = [arrays for record, arrays in files.values() if record["day"] in training_days]
        x = np.concatenate([a["features"]["counts"] for a in training]).astype(np.float64)
        y = np.concatenate(
            [a["targets"]["targets"][a["targets"]["eval_mask"]] for a in training]
        ).astype(np.float64)
        transforms[day] = {
            "input_mean": x.mean(0).astype(np.float32),
            "input_scale": np.maximum(x.std(0), 1e-6).astype(np.float32),
            "target_mean": y.mean(0).astype(np.float32),
            "target_scale": np.maximum(y.std(0), 1e-6).astype(np.float32),
        }
    summary = read_json(run / "summary.json")
    if summary["sealed_days_scored"] or summary["selection_role"] != "development_only":
        raise ValueError("Incorrect assessment designation")
    if [item["seed"] for item in summary["seeds"]] != config["seeds"]:
        raise ValueError("Seed inventory changed")
    seed_reports, max_error, checked_traces = [], 0.0, 0
    for seed in config["seeds"]:
        day_reports = []
        for training_days, day in FOLDS:
            folder = run / f"seed-{seed}" / day
            identity = read_json(folder / "identity.json")
            if identity["train_days"] != training_days or identity["evaluation_day"] != day:
                raise ValueError("Fold role mismatch")
            verify(folder / "model.npz", identity["model_sha256"])
            with np.load(folder / "model.npz", allow_pickle=False) as bundle:
                weights = {name: bundle[name] for name in bundle.files}
            shapes = {"head_weight": (7, 32), "head_bias": (7,)}
            for layer, width in ((1, 176), (2, 32), (3, 32)):
                shapes[f"conv{layer}_weight"] = (32, width, 3)
                shapes[f"conv{layer}_bias"] = (32,)
            for name, shape in shapes.items():
                array = weights[name]
                if (
                    array.shape != shape
                    or array.dtype != np.float32
                    or not np.isfinite(array).all()
                ):
                    raise ValueError("Model weights violate the fixed float32 architecture")
            parameters = sum(weights[name].size for name in shapes)
            if parameters != identity["parameter_count"]:
                raise ValueError("Wrong parameter count")
            for name, expected in transforms[day].items():
                np.testing.assert_array_equal(weights[name], expected)
            evaluation = [(r, a) for r, a in files.values() if r["day"] == day]
            if set(identity["prediction_sha256"]) != {r["file_id"] for r, _ in evaluation}:
                raise ValueError("Prediction inventory changed")
            target_rows, prediction_rows, reference_rows, trace_differences = [], [], [], {}
            for record, arrays in evaluation:
                path = folder / f"{record['file_id']}.npy"
                verify(path, identity["prediction_sha256"][record["file_id"]])
                saved = np.load(path, allow_pickle=False)
                reference = whole_file(arrays["features"]["counts"], weights)
                np.testing.assert_allclose(saved, reference, atol=3e-5, rtol=3e-5)
                error = float(np.max(np.abs(saved - reference)))
                trace_differences[record["file_id"]] = error
                max_error = max(max_error, error)
                checked_traces += 1
                # Only the evaluator joins behavior/masks after full inference.
                mask = arrays["targets"]["eval_mask"]
                target_rows.append(arrays["targets"]["targets"][mask])
                prediction_rows.append(saved[mask])
                reference_rows.append(reference[mask])
            target = np.concatenate(target_rows).astype(np.float64)
            predicted = np.concatenate(prediction_rows).astype(np.float64)
            reference = np.concatenate(reference_rows).astype(np.float64)
            score = float(r2_score(target, predicted, multioutput="variance_weighted"))
            per_output = r2_score(target, predicted, multioutput="raw_values")
            reference_score = float(r2_score(target, reference, multioutput="variance_weighted"))
            stored = read_json(folder / "scores.json")
            np.testing.assert_allclose(stored["r2_mean"], score, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                stored["per_day"][day]["r2_per_output"],
                per_output,
                atol=1e-12,
                rtol=0,
            )
            day_reports.append(
                {
                    "day": day,
                    "r2": score,
                    "r2_per_output": per_output.tolist(),
                    "full_file_torch_r2": reference_score,
                    "r2_trace_difference": score - reference_score,
                    "scored_bins": len(target),
                    "parameter_count": parameters,
                    "trace_max_absolute_difference": trace_differences,
                }
            )
        values = [row["r2"] for row in day_reports]
        seed_summary = next(item for item in summary["seeds"] if item["seed"] == seed)["scores"]
        np.testing.assert_allclose(seed_summary["r2_mean"], np.mean(values), atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            seed_summary["r2_population_std"], np.std(values), atol=1e-12, rtol=0
        )
        seed_reports.append(
            {
                "seed": seed,
                "day_r2_mean": float(np.mean(values)),
                "day_r2_population_std": float(np.std(values)),
                "days": day_reports,
            }
        )
    return {
        "status": "passed",
        "audited_utc": datetime.now(UTC).isoformat(),
        "run_manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
        "audit_source_sha256": digest(__file__),
        "source_snapshot_files_checked": len(manifest["source"]["files"]),
        "prepared_development_files_checked": len(files),
        "models_checked": 9,
        "prediction_traces_checked": checked_traces,
        "max_absolute_prediction_difference": max_error,
        "reference": "independent functional whole-file Torch convolutions from numeric NPZ",
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "scoring_dtype": "float64, retaining original float32 predictions",
        "seeds": seed_reports,
        "sealed_arrays_opened": False,
        "limitations": [
            "Verifies inference arithmetic and recorded scores; does not rerun model training.",
            "Normalization independently reproduced from authorized training files only.",
            "Development days informed this study; no untouched final assessment is implied.",
            "No kernel speedup, step latency, tail latency, memory, or G4 claim is validated.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite prior audit evidence")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(1):
            result = audit(args.run, args.index)
    finally:
        torch.set_num_threads(previous)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "models_checked",
                    "prediction_traces_checked",
                    "max_absolute_prediction_difference",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
