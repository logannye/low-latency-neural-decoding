"""Independent final inference, day scoring and supervised-prefix audit.

No production model, feature, calibration, or scoring functions are imported.
The assessment gateway is reused only to establish authorized artifact identities.
Run this only after the coordinator releases the independently pinned freeze.
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

from neural_decoder.assessment_data import load_assessment_index, validate_freeze

try:
    from .audit import filtered_counts, independent_design
    from .audit_temporal import whole_file
except ImportError:  # Direct script invocation from the repository root.
    from audit import filtered_counts, independent_design
    from audit_temporal import whole_file


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Changed artifact: {path}")


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_snapshot(run, manifest, freeze):
    source = manifest["source"]
    verify(run / "source.diff", source["diff_sha256"])
    for relative, checksum in source["files"].items():
        verify(run / "source" / relative, checksum)
    for relative, checksum in freeze["source"]["files_sha256"].items():
        if source["files"].get(relative) != checksum:
            raise ValueError("Evaluation source differs from frozen implementation/configuration")


def independent_prediction(counts, model, kind):
    if kind == "linear":
        filtered = filtered_counts(counts.astype(np.float64), model["weights"])
        design = independent_design(
            filtered, model["mean"], model["scale"], int(model["history"].item())
        )
        return design @ model["coefficient"] + model["intercept"]
    if kind == "tcn":
        return whole_file(counts, model)
    raise ValueError("Unknown independently audited model kind")


def independent_scores(records):
    """records: (file_id, day, full prediction, full target, boolean mask)."""
    days = {}
    for file_id, day, prediction, target, mask in sorted(records, key=lambda r: (r[1], r[0])):
        if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 7:
            raise ValueError("Malformed prediction/target matrix")
        if mask.dtype != bool or mask.shape != (len(target),):
            raise ValueError("Malformed evaluation mask")
        days.setdefault(day, []).append((file_id, prediction, target, mask))
    reports = {}
    for day, files in days.items():
        y = np.concatenate([target[mask] for _, _, target, mask in files]).astype(np.float64)
        p = np.concatenate([pred[mask] for _, pred, _, mask in files]).astype(np.float64)
        if len(y) < 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("Insufficient or invalid scored samples")
        reports[day] = {
            "r2_variance_weighted": float(r2_score(y, p, multioutput="variance_weighted")),
            "r2_per_output": r2_score(y, p, multioutput="raw_values").tolist(),
            "num_scored_bins": len(y),
            "num_bins": sum(len(p) for _, p, _, _ in files),
            "file_ids": [file_id for file_id, *_ in files],
        }
    if not reports:
        raise ValueError("No assessment day")
    scores = np.array([row["r2_variance_weighted"] for row in reports.values()])
    return {
        "r2_mean": float(scores.mean()),
        "r2_population_std": float(scores.std(ddof=0)),
        "r2_worst_day": float(scores.min()),
        "per_day": reports,
        "canonical_day_order": list(reports),
    }


def check_scores(stored, actual):
    for key in ("r2_mean", "r2_population_std", "r2_worst_day"):
        np.testing.assert_allclose(stored[key], actual[key], atol=1e-12, rtol=0)
    if stored["canonical_day_order"] != actual["canonical_day_order"]:
        raise ValueError("Canonical day order differs")
    if set(stored["per_day"]) != set(actual["per_day"]):
        raise ValueError("Scored day inventory differs")
    for day, row in actual["per_day"].items():
        other = stored["per_day"][day]
        for key in ("r2_variance_weighted", "r2_per_output"):
            np.testing.assert_allclose(other[key], row[key], atol=1e-12, rtol=0)
        for key in ("num_scored_bins", "num_bins", "file_ids"):
            if other[key] != row[key]:
                raise ValueError(f"Scoring sample/file identity differs: {key}")


def audit_primary(run, index_path, freeze_path, expected_sha256, *, root, cache):
    root, run, index_path = Path(root).resolve(), Path(run).resolve(), Path(index_path).resolve()
    freeze = validate_freeze(freeze_path, expected_sha256, root=root, cache=cache)
    index = load_assessment_index(index_path, freeze_path, expected_sha256, root=root, cache=cache)
    manifest, summary = read_json(run / "manifest.json"), read_json(run / "summary.json")
    if (
        manifest["status"] != "complete"
        or manifest["freeze_sha256"] != expected_sha256
        or manifest["index_sha256"] != digest(index_path)
    ):
        raise ValueError("Requires the exact completed primary assessment")
    verify(run / "summary.json", manifest["summary_sha256"])
    verify_snapshot(run, manifest, freeze)
    if (
        summary["stage"] != freeze["stage"]
        or summary["models"] != manifest["models"]
        or summary["freeze_sha256"] != expected_sha256
        or summary["index_sha256"] != digest(index_path)
        or [entry["model"] for entry in manifest["models"]] != freeze["models"]
    ):
        raise ValueError("Primary summary/model/freeze identity mismatch")
    reports, retained, max_error, trace_count = [], {}, 0.0, 0
    streams = {row["file_id"] for row in index["files"]}
    for entry in manifest["models"]:
        model_record = entry["model"]
        name, kind = model_record["name"], model_record["kind"]
        model_path = root / model_record["path"]
        verify(model_path, model_record["sha256"])
        with np.load(model_path, allow_pickle=False) as arrays:
            model = {key: arrays[key] for key in arrays.files}
        if set(entry["prediction_sha256"]) != streams:
            raise ValueError("Prediction stream inventory differs")
        records, trace_errors = [], {}
        for row in index["files"]:
            feature_path = index_path.parent / row["features"]
            verify(feature_path, row["features_sha256"])
            with np.load(feature_path, allow_pickle=False) as features:
                if set(features.files) != {"counts", "timestamps", "sequence_id", "channel_ids"}:
                    raise ValueError("Feature-only boundary violated")
                counts = features["counts"]
                if (
                    counts.shape != (row["n_bins"], 176)
                    or not np.array_equal(features["sequence_id"], np.arange(len(counts)))
                    or not np.array_equal(features["channel_ids"], np.arange(176))
                ):
                    raise ValueError("Invalid feature schema")
            # Whole-file inference accepts only counts and frozen numeric weights.
            reference = independent_prediction(counts, model, kind)
            prediction_path = run / name / f"{row['file_id']}.npy"
            verify(prediction_path, entry["prediction_sha256"][row["file_id"]])
            saved = np.load(prediction_path, allow_pickle=False)
            tolerance = 3e-5 if kind == "tcn" else 1e-11
            np.testing.assert_allclose(saved, reference, atol=tolerance, rtol=tolerance)
            error = float(np.max(np.abs(saved - reference)))
            trace_errors[row["file_id"]] = error
            max_error, trace_count = max(max_error, error), trace_count + 1
            # Labels and masks join only after the independent prediction is complete.
            target_path = index_path.parent / row["targets"]
            verify(target_path, row["targets_sha256"])
            with np.load(target_path, allow_pickle=False) as targets:
                y, mask = targets["targets"], targets["eval_mask"]
            records.append((row["file_id"], row["day"], saved, y, mask))
            retained[(name, row["file_id"])] = (row, saved, y, mask)
        scores = independent_scores(records)
        check_scores(entry["scores"], scores)
        score_file = read_json(run / name / "scores.json")
        if score_file != entry["scores"]:
            raise ValueError("Per-model score file and summary disagree")
        reports.append(
            {
                "model": name,
                "kind": kind,
                "scores": scores,
                "trace_max_absolute_difference": trace_errors,
            }
        )
    result = {
        "status": "passed",
        "stage": freeze["stage"],
        "models_checked": len(reports),
        "prediction_traces_checked": trace_count,
        "max_absolute_prediction_difference": max_error,
        "models": reports,
        "freeze_sha256": expected_sha256,
        "index_sha256": digest(index_path),
        "primary_manifest_sha256": digest(run / "manifest.json"),
        "primary_summary_sha256": digest(run / "summary.json"),
    }
    return result, retained, freeze


def audit_calibration(run, primary_run, primary_result, retained, freeze):
    run, primary_run = Path(run).resolve(), Path(primary_run).resolve()
    manifest, summary = read_json(run / "manifest.json"), read_json(run / "summary.json")
    if manifest.get("status") != "complete" or freeze["stage"] != "later":
        raise ValueError("Calibration audit requires a complete later-day study")
    verify(run / "summary.json", manifest["summary_sha256"])
    verify_snapshot(run, manifest, freeze)
    for record in (manifest, summary):
        for key in ("freeze_sha256", "index_sha256"):
            if record[key] != primary_result[key]:
                raise ValueError("Calibration differs from primary identity")
        if record["primary_summary_sha256"] != digest(primary_run / "summary.json"):
            raise ValueError("Calibration points to another primary summary")
    names = [row["name"] for row in freeze["models"] if row["name"] != "constant"]
    expected_arms = {(name, seconds) for name in names for seconds in (0, 5, 15, 30)}
    actual_arms = {(row["model"], row["prefix_seconds_per_file"]) for row in summary["results"]}
    if actual_arms != expected_arms or len(summary["results"]) != len(expected_arms):
        raise ValueError("Calibration budget/model inventory differs")
    reports, checked_paths, max_error = [], set(), 0.0
    for arm in summary["results"]:
        name, seconds = arm["model"], arm["prefix_seconds_per_file"]
        budget = int(seconds * 50)
        available = sorted((key[1], value) for key, value in retained.items() if key[0] == name)
        accounting = {row["file_id"]: row for row in arm["files"]}
        if set(accounting) != {file_id for file_id, _ in available}:
            raise ValueError("Calibration file inventory differs")
        if arm["nominal_label_seconds_per_day"] != 2 * seconds:
            raise ValueError("Per-file budget was incorrectly labelled per-day")
        records = []
        for file_id, (row, prediction, target, mask) in available:
            if len(target) <= 1500:
                raise ValueError("No common suffix available")
            offset = np.zeros(7)
            if budget:
                valid = mask[:budget]
                if not valid.any():
                    raise ValueError("Calibration prefix has no available scored labels")
                residual = target[:budget][valid].astype(np.float64) - prediction[:budget][valid]
                offset = residual.mean(axis=0)
            expected = prediction[1500:] + offset
            relative = f"{name}/prefix-{budget}/{file_id}.npy"
            verify(run / relative, summary["prediction_sha256"][relative])
            saved = np.load(run / relative, allow_pickle=False)
            np.testing.assert_array_equal(saved, expected)
            max_error = max(max_error, float(np.max(np.abs(saved - expected))))
            checked_paths.add(relative)
            facts = accounting[file_id]
            for key, value in {
                "day": row["day"],
                "nominal_label_seconds": seconds,
                "scored_prefix_bins": int(mask[:budget].sum()),
                "scored_label_seconds": int(mask[:budget].sum()) * 0.02,
                "suffix_start_bin": 1500,
                "suffix_scored_bins": int(mask[1500:].sum()),
            }.items():
                if facts[key] != value:
                    raise ValueError(f"Calibration accounting differs: {key}")
            np.testing.assert_array_equal(facts["offset"], offset)
            records.append((file_id, row["day"], saved, target[1500:], mask[1500:]))
        scores = independent_scores(records)
        check_scores(arm["scores"], scores)
        reports.append({"model": name, "prefix_seconds_per_file": seconds, "scores": scores})
    if checked_paths != set(summary["prediction_sha256"]):
        raise ValueError("Unexpected or missing calibration prediction artifact")
    return {
        "status": "passed",
        "arms_checked": len(reports),
        "prediction_traces_checked": len(checked_paths),
        "max_absolute_prediction_difference": max_error,
        "arms": reports,
        "manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
    }


def audit(run, index_path, freeze_path, expected_sha256, *, root, cache, calibration_run=None):
    result, retained, freeze = audit_primary(
        run, index_path, freeze_path, expected_sha256, root=root, cache=cache
    )
    if calibration_run is not None:
        result["calibration"] = audit_calibration(calibration_run, run, result, retained, freeze)
    result.update(
        audited_utc=datetime.now(UTC).isoformat(),
        audit_source_sha256=digest(__file__),
        torch_version=torch.__version__,
        sklearn_version=sklearn.__version__,
        inference_reference="independent whole-file Torch and NumPy FIR/lag calculation",
        scoring_reference="sklearn float64 per-day variance-weighted R2",
        limitations=[
            "Does not rerun training or establish biological/clinical validity.",
            "Authorizes only the supplied local public-recording assessment.",
            "Does not validate latency, energy, or hidden FALCON evaluation.",
        ],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "index", "freeze", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, default=Path("data/neural_h1"))
    parser.add_argument("--calibration-run", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Prior audit evidence must not be overwritten")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(1):
            result = audit(
                args.run,
                args.index,
                args.freeze,
                args.sha256,
                root=args.root,
                cache=args.cache,
                calibration_run=args.calibration_run,
            )
    except BaseException as error:
        args.output.write_text(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        torch.set_num_threads(previous)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "stage", "models_checked", "prediction_traces_checked")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
