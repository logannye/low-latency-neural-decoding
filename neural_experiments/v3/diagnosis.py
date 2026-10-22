"""Post-hoc V2 failure decomposition using only already-exposed 600 s windows.

This module fits nothing and performs no decoder inference. It consumes saved
120-second-calibration predictions, while retaining the full original inventory
and unavailable cases. Error-mean subtraction is a diagnostic decomposition,
not an evaluated deployable correction. Unit-ball projection is fixed, not fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAP = 30000
PREFIX = 6000
SUFFIX = 15000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"artifact hash mismatch: {Path(path).name}")
    return actual


def artifact(path):
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be repository-relative")
    return ROOT / relative


def project_unit_ball(prediction):
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("prediction must be finite N by 2")
    norms = np.hypot(values[:, 0], values[:, 1])
    return values / np.maximum(norms[:, None], 1.0)


def decompose(y, prediction, mask, segments):
    """Return sample-weighted squared-error components within one score window.

    Segment offsets refer to the full arrays. No fitted correction is returned.
    Undefined R²/correlation use None, including constant target/predictor cases.
    """
    y, prediction, mask = np.asarray(y), np.asarray(prediction), np.asarray(mask)
    if y.ndim != 2 or y.shape[1] != 2 or prediction.shape != y.shape:
        raise ValueError("target and prediction must be aligned N by 2")
    if mask.dtype != bool or mask.shape != (len(y),):
        raise ValueError("score mask must be an aligned boolean vector")
    if not mask.any():
        return None
    a, b = y[mask].astype(float), prediction[mask].astype(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("scored values must be finite")
    if np.any(np.hypot(a[:, 0], a[:, 1]) > 1 + 1e-6):
        raise ValueError("unit-ball diagnostic requires bounded direction targets")
    ac, bc = a - a.mean(0), b - b.mean(0)
    sst, pss = float(np.square(ac).sum()), float(np.square(bc).sum())
    error = b - a
    sse = float(np.square(error).sum())
    bias = len(a) * float(np.square(error.mean(0)).sum())
    centered_sse = float(np.square(error - error.mean(0)).sum())
    bnorm = np.hypot(b[:, 0], b[:, 1])
    projected_sse = float(np.square(a - project_unit_ball(b)).sum())
    block_bias, covered = 0.0, np.zeros(len(y), dtype=bool)
    for start, stop in segments:
        if not 0 <= start <= stop <= len(y) or covered[start:stop].any():
            raise ValueError("invalid or overlapping segments")
        covered[start:stop] = True
        eligible = mask[start:stop]
        if eligible.any():
            block_error = (prediction[start:stop].astype(float) - y[start:stop])[eligible]
            block_bias += len(block_error) * float(np.square(block_error.mean(0)).sum())
    if not covered[mask].all():
        raise ValueError("segments must cover every scored observation")

    def r2(value):
        return 1 - value / sst if sst > 1e-12 else None

    return {
        "n": len(a),
        "r2": r2(sse),
        "mse": sse / (2 * len(a)),
        "sse": sse,
        "sst": sst,
        "mean_bias_sse": bias,
        "centered_error_sse": centered_sse,
        "bias_fraction_sse": bias / sse if sse else 0.0,
        "block_bias_fraction_sse": block_bias / sse if sse else 0.0,
        "centered_r2_diagnostic": r2(centered_sse),
        "mean_prediction_norm": float(bnorm.mean()),
        "p99_prediction_norm": float(np.quantile(bnorm, 0.99)),
        "fraction_prediction_norm_over_one": float(np.mean(bnorm > 1)),
        "target_mean_norm": float(np.linalg.norm(a.mean(0))),
        "prediction_mean_norm": float(np.linalg.norm(b.mean(0))),
        "prediction_variance_over_target": pss / sst if sst > 1e-12 else None,
        "centered_xy_correlation": float((ac * bc).sum() / np.sqrt(sst * pss))
        if sst * pss > 1e-24
        else None,
        "zero_predictor_r2": r2(float(np.square(a).sum())),
        "unit_ball_projected_r2_diagnostic": r2(projected_sse),
    }


def summarize_rows(rows):
    """Average days/seeds within people, then people, never pool sample counts."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["partition"])].append(row)
    output = []
    fields = [
        k
        for k, v in rows[0].items()
        if isinstance(v, float) and k not in {"sse", "sst", "mean_bias_sse", "centered_error_sse"}
    ]
    for (method, partition), values in sorted(groups.items()):
        people = defaultdict(list)
        for row in values:
            people[row["participant"]].append(row)
        means = {}
        for field in fields:
            if any(r[field] is None for r in values):
                means[field] = None
            else:
                means[field] = float(
                    np.mean([np.mean([r[field] for r in pp]) for pp in people.values()])
                )
        output.append(
            {
                "method": method,
                "partition": partition,
                "people": len(people),
                "rows": len(values),
                **means,
            }
        )
    return output


def diagnose(index_path, run_path, control_path):
    index_path, run_path, control_path = map(Path, (index_path, run_path, control_path))
    index = read(index_path)
    index_hash = sha256(index_path)
    manifests, identities, source_hashes, inventory = [], [], {}, []
    unavailable, identity_pins = [], []
    for name, folder in (("original_transfer", run_path), ("random_core_control", control_path)):
        manifest = read(folder / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError("diagnosis requires completed source runs")
        if name == "original_transfer" and manifest["index_sha256"] != index_hash:
            raise ValueError("index differs from exposed original run")
        for source, expected in manifest["source"].items():
            checked(artifact(manifest["source_snapshots"][source]), expected)
            if source in source_hashes and source_hashes[source] != expected:
                raise ValueError("source run implementations disagree")
            source_hashes[source] = expected
        manifests.append({"role": name, "sha256": sha256(folder / "manifest.json")})
        counts = Counter()
        for entry in manifest["models"]:
            checked(artifact(entry["identity_path"]), entry["identity_sha256"])
            ident = read(artifact(entry["identity_path"]))
            counts[(ident["method"], ident["budget_seconds"])] += 1
            identity_pins.append(
                {"model_id": ident["model_id"], "sha256": entry["identity_sha256"]}
            )
            if ident["budget_seconds"] == 120:
                checked(artifact(ident["model_path"]), ident["model_sha256"])
                identities.append(ident)
        inventory.extend(
            {"run": name, "method": method, "budget_seconds": budget, "completed_models": count}
            for (method, budget), count in sorted(counts.items())
        )
        unavailable.extend(manifest.get("unavailable", []))

    by_person = defaultdict(list)
    for item in index["sessions"]:
        if item["bin_seconds"] != 0.02:
            raise ValueError("expected fixed 20 ms observations")
        by_person[item["participant_id"]].append(item)
    for values in by_person.values():
        values.sort(key=lambda r: r["post_implant_day"])
    rows, shifts, training, data_pins, trace_pins = [], [], [], [], []
    for person, records in by_person.items():
        models = [d for d in identities if d["target_participant"] == person]
        canonical = next(
            d for d in models if d["method"] == "pretrained_frozen_core" and d["seed"] == 0
        )
        with np.load(artifact(canonical["model_path"]), allow_pickle=False) as arrays:
            mean, scale = arrays["input_mean"].copy(), arrays["input_scale"].copy()
        calibration_id = records[0]["session_id"]
        for ident in models:
            history_path = artifact(ident["model_path"]).parent / "training.json"
            history = read(history_path)
            if isinstance(history, dict):
                history = history.get("history", history.get("training_history", []))
            if history:
                training.append(
                    {
                        "participant": person,
                        "method": ident["method"],
                        "seed": ident["seed"],
                        "history_sha256": sha256(history_path),
                        "first_epoch_mse": history[0]["mse"],
                        "final_epoch_mse": history[-1]["mse"],
                    }
                )
        for item in records:
            sid = item["session_id"]
            target_path, feature_path = (
                index_path.parent / item[k] for k in ("target_path", "feature_path")
            )
            checked(target_path, item["target_sha256"])
            checked(feature_path, item["feature_sha256"])
            data_pins.append(
                {
                    "session_id": sid,
                    "target_sha256": item["target_sha256"],
                    "feature_sha256": item["feature_sha256"],
                }
            )
            # NPZ members can be longer; every subsequent operation uses only 0:CAP.
            with np.load(target_path, allow_pickle=False) as arrays:
                y, mask = arrays["y"][:CAP].copy(), arrays["eval_mask"][:CAP].copy()
            with np.load(feature_path, allow_pickle=False) as arrays:
                x, channel_ids = arrays["x"][:CAP].copy(), arrays["channel_ids"].copy()
            segments = [
                (s["start"], min(s["end"], len(y))) for s in item["segments"] if s["start"] < len(y)
            ]
            z = (np.sqrt(x) - mean) / scale
            channel_mean = z.mean(0)
            shifts.append(
                {
                    "participant": person,
                    "session": sid,
                    "partition": "calibration_day" if sid == calibration_id else "future_day",
                    "n": len(z),
                    "n_channels": len(mean),
                    "rms_channel_mean_z": float(np.sqrt(np.mean(channel_mean**2))),
                    "fraction_channels_mean_abs_over_one": float(np.mean(np.abs(channel_mean) > 1)),
                    "fraction_values_abs_z_over_five": float(np.mean(np.abs(z) > 5)),
                    "maximum_abs_z": float(np.max(np.abs(z))),
                    "calibration_scale_floor_fraction": float(np.mean(scale <= 0.100001)),
                    "block_mean_z_rms": [
                        float(np.sqrt(np.mean(z[a:b].mean(0) ** 2))) for a, b in segments
                    ],
                    "channel_ids_sha256": hashlib.sha256(channel_ids.tobytes()).hexdigest(),
                }
            )
            del x, z
            for ident in models:
                trace = next(t for t in ident["traces"] if t["session_id"] == sid)
                checked(artifact(trace["prediction_path"]), trace["prediction_sha256"])
                prediction = np.load(artifact(trace["prediction_path"]), allow_pickle=False)
                if len(prediction) != len(y) or len(y) > CAP:
                    raise ValueError("trace does not match exposed 600-second cap")
                trace_pins.append(
                    {
                        "model_id": ident["model_id"],
                        "session_id": sid,
                        "sha256": trace["prediction_sha256"],
                    }
                )
                score_mask = mask.copy()
                score_mask[: trace["start_bin"]] = False
                if (
                    hashlib.sha256(
                        np.ascontiguousarray(score_mask, dtype=bool).tobytes()
                    ).hexdigest()
                    != trace["score_mask_sha256"]
                ):
                    raise ValueError("mask differs from retained scored window")
                tasks = []
                if sid == calibration_id:
                    fit_mask = mask.copy()
                    fit_mask[PREFIX:] = False
                    for start, stop in segments:
                        fit_mask[start : min(start + 62, stop)] = False
                    tasks.append(("calibration_fit", fit_mask))
                    if trace["start_bin"] != min(SUFFIX, len(y)):
                        raise ValueError("changed common suffix")
                elif trace["start_bin"] != 0:
                    raise ValueError("future-day window no longer starts at zero")
                if trace["metrics"]["r2"] is not None:
                    tasks.append((trace["partition"], score_mask))
                for partition, selected in tasks:
                    result = decompose(y, prediction, selected, segments)
                    if result is not None:
                        if partition != "calibration_fit" and not np.isclose(
                            result["r2"], trace["metrics"]["r2"], rtol=0, atol=1e-10
                        ):
                            raise ValueError(
                                "retained trace score differs from direct decomposition"
                            )
                        rows.append(
                            {
                                "participant": person,
                                "session": sid,
                                "method": ident["method"],
                                "seed": ident["seed"],
                                "partition": partition,
                                **result,
                            }
                        )

    sources = []
    for pretrain in read(run_path / "manifest.json")["pretraining"]:
        checked(artifact(pretrain["receipt_path"]), pretrain["receipt_sha256"])
        receipt = read(artifact(pretrain["receipt_path"]))
        sources.append(
            {
                "target": pretrain["target_participant"],
                "seed": pretrain["seed"],
                "receipt_sha256": pretrain["receipt_sha256"],
                "selected_step": receipt["selected_step"],
                "checkpoints": receipt["checkpoints"],
                "first_80_step_mse": float(np.mean([x["loss"] for x in receipt["losses"][:80]])),
                "last_80_step_mse": float(np.mean([x["loss"] for x in receipt["losses"][-80:]])),
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "exploratory": True,
        "scope": (
            "Post-hoc diagnosis of exposed V2 windows; no fit, decoder inference, "
            "new participant, or new-window score"
        ),
        "window": {
            "start_bin": 0,
            "maximum_stop_bin": CAP,
            "bin_seconds": 0.02,
            "calibration_budget_seconds": 120,
            "common_suffix_start_bin": SUFFIX,
            "training_burn_bins_per_block": 62,
        },
        "aggregation": (
            "Equal days and seeds within each person, then equal people; "
            "bias fractions are means of per-trace fractions"
        ),
        "diagnostic_limits": {
            "error_mean_subtraction_uses_evaluation_labels": True,
            "error_mean_subtraction_is_deployable": False,
            "unit_ball_projection_is_fixed_unfitted": True,
            "zero_vector_is_not_useful_direction_control": True,
        },
        "diagnosis_source_sha256": sha256(__file__),
        "index_sha256": index_hash,
        "source_manifests": manifests,
        "executed_v2_source_sha256": source_hashes,
        "all_original_model_inventory": inventory,
        "unavailable_original_cases": unavailable,
        "model_identity_pins": identity_pins,
        "data_pins": data_pins,
        "trace_pins": trace_pins,
        "diagnostic_models": len(identities),
        "diagnostic_traces": len(trace_pins),
        "diagnostic_score_windows": len(rows),
        "per_person": [
            {"participant": person, **group}
            for person in sorted(by_person)
            for group in summarize_rows([r for r in rows if r["participant"] == person])
        ],
        "aggregate": summarize_rows(rows),
        "feature_shift": shifts,
        "adaptation_training": training,
        "source_training": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.index, args.run, args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": result["diagnostic_models"],
                "traces": result["diagnostic_traces"],
                "source": result["diagnosis_source_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
