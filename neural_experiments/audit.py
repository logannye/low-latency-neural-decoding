"""Independent numerical audit of development-only linear H1 result bundles.

Uses NumPy convolution, explicit lag stacking, and direct R2 arithmetic rather
than the production feature/model/scoring functions. Only source-authorized
S0-S4 prepared arrays are opened. Fails before opening arrays if identities drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_decoder.data import load_index


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Changed artifact: {path}")


def read_json(path):
    return json.loads(Path(path).read_text())


def filtered_counts(counts, weights):
    return np.column_stack(
        [
            np.convolve(counts[:, channel], weights, mode="full")[: len(counts)]
            for channel in range(counts.shape[1])
        ]
    )


def independent_design(filtered, mean, scale, history):
    standardized = (filtered - mean) / scale
    lagged = []
    for lag in range(history):
        values = np.zeros_like(standardized)
        values[lag:] = standardized[: len(standardized) - lag]
        lagged.append(values)
    return np.concatenate(lagged, axis=1)


def direct_scores(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = target - prediction
    sse = np.einsum("ij,ij->j", error, error)
    centered = target - np.mean(target, axis=0)
    sst = np.einsum("ij,ij->j", centered, centered)
    per_output = np.ones(target.shape[1])
    varying = sst != 0
    per_output[varying] -= sse[varying] / sst[varying]
    per_output[~varying & (sse != 0)] = 0
    weighted = float(np.average(per_output, weights=sst) if varying.any() else per_output.mean())
    return weighted, per_output


def audit(run, index_path):
    run, index_path = Path(run).resolve(), Path(index_path).resolve()
    index = load_index(index_path)  # reconstruct source authority before opening sample arrays
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only complete runs can receive a successful independent audit")
    check_hash(index_path, manifest["index_sha256"])
    check_hash(run / "input_index.json", manifest["index_sha256"])
    check_hash(run / "summary.json", manifest["summary_sha256"])
    check_hash(run / "source.diff", manifest["source"]["diff_sha256"])
    for relative, expected in manifest["source"]["files"].items():
        check_hash(run / "source" / relative, expected)
    configuration = manifest["configuration"]
    config_path = run / "source/neural_experiments/configs/ridge-v1.json"
    check_hash(config_path, manifest["config_sha256"])
    if read_json(config_path) != configuration:
        raise ValueError("Configuration value disagrees with hashed configuration")
    folds = [(["S0", "S1"], "S2"), (["S0", "S1", "S2"], "S3"), (["S0", "S1", "S2", "S3"], "S4")]
    if configuration["folds"] != [{"train": days, "evaluate": day} for days, day in folds]:
        raise ValueError("Unexpected fold roles")
    if configuration["regime"] != "frozen_zero_new_day_calibration":
        raise ValueError("Unsupported regime")
    raw_manifest = read_json(index_path.parent.parent / "manifest.json")
    assets = {asset["asset_id"]: asset for asset in raw_manifest["assets"]}
    weights = np.exp(-np.arange(12, dtype=float) / 12)
    weights /= weights.sum()
    files = {}
    for record in index["files"]:
        if record["day"] not in {"S0", "S1", "S2", "S3", "S4"}:
            raise ValueError("Refusing sample-level access to sealed data")
        raw_path = index_path.parent.parent / assets[record["asset_id"]]["local_path"]
        check_hash(raw_path, record["raw_sha256"])
        arrays = {}
        for kind in ("features", "targets"):
            path = index_path.parent / record[kind]
            check_hash(path, record[f"{kind}_sha256"])
            with np.load(path, allow_pickle=False) as bundle:
                arrays[kind] = {key: bundle[key] for key in bundle.files}
        x = arrays["features"]["counts"].astype(np.float64)
        arrays["filtered"] = filtered_counts(x, weights)
        files[record["file_id"]] = (record, arrays)
    summary = read_json(run / "summary.json")
    if summary["sealed_days_scored"] != [] or summary["selection_role"] != "development_only":
        raise ValueError("Invalid declared assessment role")
    reports, diagnostics = [], []
    max_prediction_error = 0.0
    expected_candidates = {"constant-h1-lambda0.01"} | {
        f"ridge-h{history}-lambda{lam:g}"
        for history in configuration["history_bins"]
        for lam in configuration["regularization"]
    }
    if {c["candidate"] for c in summary["candidates"]} != expected_candidates:
        raise ValueError("Candidate inventory disagrees with configuration")
    for candidate in summary["candidates"]:
        name, day_scores = candidate["candidate"], []
        for training_days, evaluation_day in folds:
            destination = run / name / evaluation_day
            identity = read_json(destination / "identity.json")
            if (
                identity["train_days"] != training_days
                or identity["evaluation_day"] != evaluation_day
            ):
                raise ValueError("Fold identity disagrees with authorized forward split")
            check_hash(destination / "model.npz", identity["model_sha256"])
            with np.load(destination / "model.npz", allow_pickle=False) as bundle:
                model = {key: bundle[key] for key in bundle.files}
            assert model["history"].item() == candidate["history"]
            assert model["kind"].item() == candidate["kind"]
            np.testing.assert_allclose(model["weights"], weights, rtol=0, atol=1e-15)
            training = [a for r, a in files.values() if r["day"] in training_days]
            training_filtered = np.concatenate([a["filtered"] for a in training])
            np.testing.assert_allclose(model["mean"], training_filtered.mean(0), atol=1e-13)
            np.testing.assert_allclose(
                model["scale"],
                np.maximum(training_filtered.std(0), 1e-6),
                atol=1e-13,
            )
            evaluation = [(r, a) for r, a in files.values() if r["day"] == evaluation_day]
            assert set(identity["prediction_sha256"]) == {r["file_id"] for r, _ in evaluation}
            targets, predictions, evaluation_filtered = [], [], []
            for record, arrays in evaluation:
                prediction_path = destination / f"{record['file_id']}.npy"
                check_hash(prediction_path, identity["prediction_sha256"][record["file_id"]])
                actual = np.load(prediction_path, allow_pickle=False)
                design = independent_design(
                    arrays["filtered"],
                    model["mean"],
                    model["scale"],
                    int(model["history"]),
                )
                expected = design @ model["coefficient"] + model["intercept"]
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
                max_prediction_error = max(
                    max_prediction_error, float(np.max(abs(actual - expected)))
                )
                mask = arrays["targets"]["eval_mask"]
                targets.append(arrays["targets"]["targets"][mask])
                predictions.append(actual[mask])
                evaluation_filtered.append(arrays["filtered"])
            target = np.concatenate(targets).astype(np.float64)
            prediction = np.concatenate(predictions).astype(np.float64)
            score, per_output = direct_scores(target, prediction)
            stored = read_json(destination / "scores.json")
            np.testing.assert_allclose(stored["r2_mean"], score, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                stored["per_day"][evaluation_day]["r2_per_output"],
                per_output,
                rtol=0,
                atol=1e-12,
            )
            day_scores.append(score)
            if name == summary["selected_ridge"]:
                xs, ys = [], []
                for arrays in training:
                    design = independent_design(
                        arrays["filtered"],
                        model["mean"],
                        model["scale"],
                        int(model["history"]),
                    )
                    mask = arrays["targets"]["eval_mask"]
                    xs.append(design[mask])
                    ys.append(arrays["targets"]["targets"][mask])
                train_x, train_y = np.concatenate(xs), np.concatenate(ys)
                x_mean, y_mean = train_x.mean(0), train_y.mean(0)
                centered_x, centered_y = train_x - x_mean, train_y - y_mean
                gram = centered_x.T @ centered_x / len(train_x)
                coefficient = np.linalg.solve(
                    gram + candidate["regularization"] * np.eye(gram.shape[0]),
                    centered_x.T @ centered_y / len(train_x),
                )
                np.testing.assert_allclose(model["coefficient"], coefficient, atol=1e-11)
                np.testing.assert_allclose(
                    model["intercept"], y_mean - x_mean @ coefficient, atol=1e-11
                )
                residual = target - prediction
                bias = residual.mean(0)
                bias_sse = len(target) * bias**2
                centered_sse = ((residual - bias) ** 2).sum(0)
                total_sse = (residual**2).sum(0)
                np.testing.assert_allclose(bias_sse + centered_sse, total_sse, atol=1e-10)
                shift = (np.concatenate(evaluation_filtered).mean(0) - model["mean"]) / model[
                    "scale"
                ]
                diagnostics.append(
                    {
                        "day": evaluation_day,
                        "r2": score,
                        "r2_per_output": per_output.tolist(),
                        "scored_bins": len(target),
                        "mean_target_minus_prediction": bias.tolist(),
                        "bias_sse_per_output": bias_sse.tolist(),
                        "centered_residual_sse_per_output": centered_sse.tolist(),
                        "bias_fraction_total_sse": float(bias_sse.sum() / total_sse.sum()),
                        "bias_fraction_per_output": (bias_sse / total_sse).tolist(),
                        "normalizer_mean_shift_rms_training_sd": float(np.sqrt(np.mean(shift**2))),
                        "normalizer_mean_shift_max_abs_training_sd": float(np.max(np.abs(shift))),
                        "normalizer_mean_shift_per_channel_training_sd": shift.tolist(),
                        "diagnostic_scope": (
                            "retrospective development diagnostics; no causal attribution"
                        ),
                    }
                )
        np.testing.assert_allclose(candidate["scores"]["r2_mean"], np.mean(day_scores), atol=1e-12)
        np.testing.assert_allclose(
            candidate["scores"]["r2_population_std"],
            np.std(day_scores),
            atol=1e-12,
        )
        reports.append(
            {"candidate": name, "day_r2": day_scores, "mean_r2": float(np.mean(day_scores))}
        )
    selected = max(
        (candidate for candidate in summary["candidates"] if candidate["kind"] == "ridge"),
        key=lambda item: (item["scores"]["r2_mean"], -item["history"], item["regularization"]),
    )["candidate"]
    assert selected == summary["selected_ridge"]
    return {
        "status": "passed",
        "audited_utc": datetime.now(UTC).isoformat(),
        "run_manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
        "audit_source_sha256": digest(__file__),
        "source_snapshot_files_checked": len(manifest["source"]["files"]),
        "development_files_checked": len(files),
        "models_checked": len(reports) * len(folds),
        "selected_models_independently_refit": len(folds),
        "max_absolute_prediction_difference": max_prediction_error,
        "selected_ridge": selected,
        "candidates": reports,
        "selected_development_diagnostics": diagnostics,
        "sealed_arrays_opened": False,
        "limitations": [
            "Exploratory result source snapshot; final comparisons require committed source.",
            "Hashes bind bytes and manifests; data.load_index reconstructs source day authority.",
            "Day diagnostics were used for development and are not untouched final evidence.",
            "No runtime performance claim is validated by this numerical audit.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite earlier audit evidence")
    with threadpool_limits(1):
        result = audit(args.run, args.index)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "models_checked",
                    "selected_ridge",
                    "max_absolute_prediction_difference",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
