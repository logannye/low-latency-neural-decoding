"""Prespecified G3 development experiment using immutable, already fitted ridges."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .adaptation import AdaptiveRidge
from .artifacts import environment, sha256, source_identity, write_json
from .metrics import FileEvaluation, score_files
from .models import load_model
from .train import FOLDS, load_development_files


def select_mixture_window(targets, window_bins=3000, target_index=6):
    """Evaluator-only contiguous selection; never call from predictor code."""
    if targets.ndim != 2 or len(targets) < window_bins:
        raise ValueError("each diagnostic file must contain a complete 60-second window")
    values = np.asarray(targets[:, target_index], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("mixture diagnostic requires finite target selection values")
    cumulative = np.r_[0.0, np.cumsum(values)]
    means = (cumulative[window_bins:] - cumulative[:-window_bins]) / window_bins
    start = int(np.argmax(means))
    return start, start + window_bins, float(means[start])


def score_and_bias(records):
    scores = score_files(records)
    bias = {}
    for day in scores["canonical_day_order"]:
        selected = [r for r in records if r.day == day]
        residual = np.concatenate(
            [
                np.asarray(r.targets[r.eval_mask], dtype=np.float64) - r.predictions[r.eval_mask]
                for r in selected
            ]
        )
        average = residual.mean(0)
        mean_sse = len(residual) * average**2
        centered_sse = ((residual - average) ** 2).sum(0)
        total = (residual**2).sum(0)
        np.testing.assert_allclose(mean_sse + centered_sse, total, atol=1e-10)
        bias[day] = {
            "mean_target_minus_prediction": average.tolist(),
            "bias_sse_per_output": mean_sse.tolist(),
            "centered_residual_sse_per_output": centered_sse.tolist(),
            "total_sse_per_output": total.tolist(),
            "bias_fraction_total_sse": float(mean_sse.sum() / total.sum()) if total.sum() else 0.0,
        }
    return {"scores": scores, "bias": bias}


def paired_comparison(actual, frozen):
    differences = {
        day: value["r2_variance_weighted"] - frozen["per_day"][day]["r2_variance_weighted"]
        for day, value in actual["per_day"].items()
    }
    harmed = [day for day, difference in differences.items() if difference < 0]
    return {
        "mean_r2_difference": actual["r2_mean"] - frozen["r2_mean"],
        "per_day_r2_difference": differences,
        "days_harmed": harmed,
        "number_days_harmed": len(harmed),
        "fraction_days_harmed": len(harmed) / len(differences),
        "worst_day_harm": max(0.0, -min(differences.values())),
        "worst_day_difference": min(differences.values()),
    }


def apply_decision(results, config):
    rules = config["decision"]
    ordinary_mixture_harm = results["movement_mixture"]["ema"]["comparison"]["worst_day_harm"]
    candidates = []
    for cap in config["caps_training_sd"]:
        name = f"bounded-{cap:g}"
        real = results["real"][name]["comparison"]
        offset = results["observation_offset"][name]["comparison"]
        mixture = results["movement_mixture"][name]["comparison"]
        criteria = {
            "real_mean_gain": real["mean_r2_difference"] >= rules["real_minimum_mean_r2_gain"],
            "real_worst_day_harm": real["worst_day_harm"] <= rules["real_maximum_worst_day_harm"],
            "offset_benefit": offset["mean_r2_difference"] > 0,
            "less_mixture_harm_than_ordinary": mixture["worst_day_harm"] < ordinary_mixture_harm,
        }
        candidates.append(
            {
                "method": name,
                "cap_training_sd": cap,
                "criteria": criteria,
                "qualifies": all(criteria.values()),
                "real": real,
                "offset": offset,
                "mixture": mixture,
            }
        )
    qualifying = [candidate for candidate in candidates if candidate["qualifies"]]
    selected = (
        max(qualifying, key=lambda c: (c["real"]["mean_r2_difference"], -c["cap_training_sd"]))
        if qualifying
        else None
    )
    return {
        "selected_method": selected["method"] if selected else "frozen",
        "retain_bounded_correction": bool(qualifying),
        "candidates": candidates,
        "ordinary_mixture_worst_day_harm": ordinary_mixture_harm,
        "action": "retain qualifying bounded correction for later freeze review"
        if qualifying
        else "retain frozen normalization; abandon this bounded mean-correction hypothesis",
        "scope": "development decision only; no sealed or biological mechanism claim",
    }


def _validate_config(config):
    if (
        config["candidate"] != "ridge-h1-lambda0.1"
        or config["caps_training_sd"] != [0.25, 0.5, 1.0]
        or config["tau_seconds"] != 30.0
        or config["bin_seconds"] != 0.02
        or config["conditions"]["observation_offset"]["magnitude_training_filtered_sd"] != 0.5
        or config["conditions"]["movement_mixture"]["window_bins"] != 3000
        or config["conditions"]["movement_mixture"]["target_index"] != 6
        or config["folds"] != [{"train": train, "evaluate": day} for train, day in FOLDS]
    ):
        raise ValueError("Only the prespecified G3 recipe is supported")


def run(index_path, baseline_run, config_path, output, *, root):
    index_path, baseline_run = Path(index_path), Path(baseline_run)
    config = json.loads(Path(config_path).read_text())
    _validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "configuration": config,
        "config_sha256": sha256(config_path),
        "source": source_identity(root, output),
        "environment": environment(),
        "index_sha256": sha256(index_path),
        "baseline_manifest_sha256": sha256(baseline_run / "manifest.json"),
        "regime": config["regime"],
        "sealed_days_opened": [],
        "new_models_fitted": 0,
    }
    write_json(output / "manifest.json", manifest)
    try:
        baseline = json.loads((baseline_run / "manifest.json").read_text())
        if baseline["status"] != "complete" or baseline["index_sha256"] != manifest["index_sha256"]:
            raise ValueError("Baseline must be complete and use this exact development input index")
        if sha256(baseline_run / "summary.json") != baseline["summary_sha256"]:
            raise ValueError("Baseline summary identity changed")
        index, all_files = load_development_files(index_path)
        write_json(output / "input_index.json", index)
        models, identities, intervals = {}, {}, {}
        evaluation_files = [
            record for record in all_files if record[0]["day"] in {"S2", "S3", "S4"}
        ]
        for train_days, day in FOLDS:
            folder = baseline_run / config["candidate"] / day
            identity = json.loads((folder / "identity.json").read_text())
            if identity["train_days"] != train_days or identity["evaluation_day"] != day:
                raise ValueError("Baseline model training roles disagree with development fold")
            if sha256(folder / "model.npz") != identity["model_sha256"]:
                raise ValueError("Baseline model changed")
            models[day] = load_model(folder / "model.npz")
            identities[day] = identity
        for item, _, targets, _ in evaluation_files:
            start, stop, selection_value = select_mixture_window(targets)
            intervals[item["file_id"]] = {
                "day": item["day"],
                "file_bins": item["n_bins"],
                "start_inclusive": start,
                "stop_exclusive": stop,
                "target_g3_mean": selection_value,
                "selection_role": "evaluator_only",
                "reset_at_window_start": True,
            }
        write_json(output / "diagnostic_windows.json", intervals)
        write_json(output / "baseline_model_identities.json", identities)
        methods = [("frozen", "frozen", None), ("ema", "ema", None)] + [
            (f"bounded-{cap:g}", "bounded", cap) for cap in config["caps_training_sd"]
        ]
        results, traces = {}, []
        expected_bytes = 5 * 7 * 8 * (2 * sum(len(x) for _, x, _, _ in evaluation_files) + 18000)
        manifest["estimated_prediction_bytes"] = expected_bytes
        write_json(output / "manifest.json", manifest)
        with threadpool_limits(config["threads"]):
            for condition in ("real", "observation_offset", "movement_mixture"):
                results[condition] = {}
                for name, mode, cap in methods:
                    records = []
                    folder = output / condition / name
                    folder.mkdir(parents=True)
                    for item, original_counts, original_targets, original_mask in evaluation_files:
                        model = models[item["day"]]
                        counts, targets, mask = original_counts, original_targets, original_mask
                        interval = [0, len(counts)]
                        if condition == "observation_offset":
                            counts = counts.astype(np.float64) + 0.5 * model.scale
                        elif condition == "movement_mixture":
                            window = intervals[item["file_id"]]
                            start, stop = window["start_inclusive"], window["stop_exclusive"]
                            counts, targets, mask = (
                                counts[start:stop],
                                targets[start:stop],
                                mask[start:stop],
                            )
                            interval = [start, stop]
                        decoder = AdaptiveRidge(
                            model,
                            mode=mode,
                            cap_sd=cap,
                            tau_seconds=config["tau_seconds"],
                            bin_seconds=config["bin_seconds"],
                        )
                        prediction = decoder.predict_chunk(counts)
                        prediction_path = folder / f"{item['file_id']}.npy"
                        np.save(prediction_path, prediction, allow_pickle=False)
                        state_path = folder / f"{item['file_id']}-state.npz"
                        np.savez(state_path, **decoder.state_snapshot())
                        if condition == "real" and mode == "frozen":
                            original_path = (
                                baseline_run
                                / config["candidate"]
                                / item["day"]
                                / prediction_path.name
                            )
                            if (
                                sha256(original_path)
                                != identities[item["day"]]["prediction_sha256"][item["file_id"]]
                            ):
                                raise ValueError("Retained baseline prediction changed")
                            np.testing.assert_array_equal(
                                prediction, np.load(original_path, allow_pickle=False)
                            )
                        records.append(
                            FileEvaluation(item["file_id"], item["day"], prediction, targets, mask)
                        )
                        traces.append(
                            {
                                "condition": condition,
                                "method": name,
                                "file_id": item["file_id"],
                                "day": item["day"],
                                "input_interval": interval,
                                "source_features_sha256": item["features_sha256"],
                                "input_array_dtype": str(counts.dtype),
                                "input_array_shape": list(counts.shape),
                                "input_array_sha256": hashlib.sha256(
                                    np.ascontiguousarray(counts).tobytes()
                                ).hexdigest(),
                                "prediction_path": str(prediction_path.relative_to(output)),
                                "prediction_sha256": sha256(prediction_path),
                                "state_path": str(state_path.relative_to(output)),
                                "state_sha256": sha256(state_path),
                                "persistent_numeric_state_bytes": decoder.state_nbytes,
                            }
                        )
                    result = score_and_bias(records)
                    write_json(folder / "scores_and_bias.json", result)
                    results[condition][name] = result
                    print(
                        json.dumps(
                            {
                                "condition": condition,
                                "method": name,
                                "r2_mean": result["scores"]["r2_mean"],
                            }
                        ),
                        flush=True,
                    )
                frozen = results[condition]["frozen"]["scores"]
                for result in results[condition].values():
                    result["comparison"] = paired_comparison(result["scores"], frozen)
        decision = apply_decision(results, config)
        summary = {
            "conditions": results,
            "decision": decision,
            "selection_role": "development_only",
            "sealed_days_scored": [],
            "synthetic_conditions": ["observation_offset", "movement_mixture"],
        }
        write_json(output / "trace_manifest.json", traces)
        write_json(output / "summary.json", summary)
        manifest.update(
            {
                "status": "complete",
                "summary_sha256": sha256(output / "summary.json"),
                "trace_manifest_sha256": sha256(output / "trace_manifest.json"),
                "diagnostic_windows_sha256": sha256(output / "diagnostic_windows.json"),
                "baseline_model_identities_sha256": sha256(
                    output / "baseline_model_identities.json"
                ),
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
    run(
        args.index,
        args.baseline_run,
        args.config,
        args.output,
        root=Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    main()
