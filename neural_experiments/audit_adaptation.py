"""Independent fixed-array adaptation audit with explicit artifact locations.

Reconstructs the retained study with SciPy's delayed EMA and batch affine
arithmetic. It imports no adaptive predictor, feature implementation, or scorer.
The data-index authority check is reused before opening development arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from neural_decoder.data import load_index


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Changed artifact: {path}")


def read_json(path):
    return json.loads(Path(path).read_text())


def numeric(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes its declared root")
    return path


def audit(run, index_path, baseline_run):
    run, index_path, baseline_run = map(
        lambda p: Path(p).resolve(), (run, index_path, baseline_run)
    )
    index = load_index(index_path)
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only a complete adaptation run can be audited")
    verify(index_path, manifest["index_sha256"])
    verify(run / "input_index.json", manifest["index_sha256"])
    for filename, field in (
        ("summary.json", "summary_sha256"),
        ("trace_manifest.json", "trace_manifest_sha256"),
        ("diagnostic_windows.json", "diagnostic_windows_sha256"),
        ("baseline_model_identities.json", "baseline_model_identities_sha256"),
    ):
        verify(run / filename, manifest[field])
    verify(run / "source.diff", manifest["source"]["diff_sha256"])
    for relative, expected in manifest["source"]["files"].items():
        verify(contained(run / "source", relative), expected)
    config_path = run / "source/neural_experiments/configs/adaptation-v1.json"
    verify(config_path, manifest["config_sha256"])
    config = read_json(config_path)
    if config != manifest["configuration"]:
        raise ValueError("Manifest configuration differs from hashed source config")
    if (config["bin_seconds"], config["tau_seconds"], config["caps_training_sd"]) != (
        0.02,
        30.0,
        [0.25, 0.5, 1.0],
    ):
        raise ValueError("Unsupported adaptation settings; this audits the locked study")
    verify(baseline_run / "manifest.json", manifest["baseline_manifest_sha256"])
    baseline_manifest = read_json(baseline_run / "manifest.json")
    if baseline_manifest["status"] != "complete" or baseline_manifest["index_sha256"] != sha(
        index_path
    ):
        raise ValueError("Baseline completion/index identity mismatch")
    identities = read_json(run / "baseline_model_identities.json")
    windows = read_json(run / "diagnostic_windows.json")
    summary = read_json(run / "summary.json")
    traces = read_json(run / "trace_manifest.json")
    rows = {row["file_id"]: row for row in index["files"] if row["day"] in {"S2", "S3", "S4"}}
    methods = {"frozen", "ema", *(f"bounded-{cap:g}" for cap in config["caps_training_sd"])}
    conditions = {"real", "observation_offset", "movement_mixture"}
    expected = {
        (condition, method, file_id)
        for condition in conditions
        for method in methods
        for file_id in rows
    }
    actual = [(trace["condition"], trace["method"], trace["file_id"]) for trace in traces]
    if len(actual) != len(set(actual)) or set(actual) != expected or len(actual) != 90:
        raise ValueError("Missing, duplicated, or unexpected trace inventory")
    arrays, models = {}, {}
    for file_id, row in rows.items():
        for kind in ("features", "targets"):
            verify(contained(index_path.parent, row[kind]), row[f"{kind}_sha256"])
        features = numeric(contained(index_path.parent, row["features"]))
        labels = numeric(contained(index_path.parent, row["targets"]))
        arrays[file_id] = (features["counts"], labels["targets"], labels["eval_mask"])
    for fold in config["folds"]:
        day = fold["evaluate"]
        if day not in {"S2", "S3", "S4"}:
            raise ValueError("Sealed/nondevelopment model requested")
        model_path = contained(baseline_run, f"{config['candidate']}/{day}/model.npz")
        identity = read_json(model_path.with_name("identity.json"))
        if identity != identities[day] or identity["train_days"] != fold["train"]:
            raise ValueError("Baseline identity or training role mismatch")
        verify(model_path, identities[day]["model_sha256"])
        model = numeric(model_path)
        if model["history"].item() != 1 or model["kind"].item() != "ridge":
            raise ValueError("Unsupported baseline model")
        models[day] = model
    alpha = -np.expm1(-config["bin_seconds"] / config["tau_seconds"])
    grouped, maximum, state_maximum = {}, 0.0, 0.0
    for trace in traces:
        row = rows[trace["file_id"]]
        if trace["day"] != row["day"] or trace["source_features_sha256"] != row["features_sha256"]:
            raise ValueError("Trace source identity mismatch")
        counts, targets, mask = arrays[trace["file_id"]]
        model = models[row["day"]]
        start, stop = trace["input_interval"]
        if trace["condition"] == "movement_mixture":
            window = windows[trace["file_id"]]
            interval = [window["start_inclusive"], window["stop_exclusive"]]
            if [start, stop] != interval or stop - start != config["conditions"][
                "movement_mixture"
            ]["window_bins"]:
                raise ValueError("Mixture diagnostic interval mismatch")
        elif [start, stop] != [0, len(counts)]:
            raise ValueError("Whole-file condition was cropped")
        if not 0 <= start < stop <= len(counts):
            raise ValueError("Invalid diagnostic interval")
        counts, targets, mask = counts[start:stop], targets[start:stop], mask[start:stop]
        if trace["condition"] == "observation_offset":
            offset = config["conditions"]["observation_offset"]["magnitude_training_filtered_sd"]
            counts = counts.astype(float) + offset * model["scale"]
        input_sha = hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest()
        if (
            input_sha != trace["input_array_sha256"]
            or str(counts.dtype) != trace["input_array_dtype"]
            or list(counts.shape) != trace["input_array_shape"]
        ):
            raise ValueError("Condition input array differs from recorded identity")
        filtered = lfilter(model["weights"], [1], counts, axis=0)
        deviation = filtered - model["mean"]
        previous = lfilter([0, alpha], [1, -(1 - alpha)], deviation, axis=0)
        correction = previous
        method = trace["method"]
        if method == "frozen":
            correction = np.zeros_like(previous)
        elif method.startswith("bounded-"):
            cap = float(method.split("-")[1])
            correction = np.clip(previous, -cap * model["scale"], cap * model["scale"])
        expected_prediction = ((deviation - correction) / model["scale"]) @ model[
            "coefficient"
        ] + model["intercept"]
        path = contained(run, trace["prediction_path"])
        verify(path, trace["prediction_sha256"])
        actual_prediction = np.load(path, allow_pickle=False)
        np.testing.assert_allclose(actual_prediction, expected_prediction, atol=1e-11, rtol=1e-11)
        maximum = max(maximum, float(np.max(abs(actual_prediction - expected_prediction))))
        state_path = contained(run, trace["state_path"])
        verify(state_path, trace["state_sha256"])
        state = numeric(state_path)
        expected_state = (
            np.zeros_like(model["mean"])
            if method == "frozen"
            else (lfilter([alpha], [1, -(1 - alpha)], deviation, axis=0)[-1])
        )
        np.testing.assert_allclose(state["ema_deviation"], expected_state, atol=1e-11, rtol=1e-11)
        state_maximum = max(
            state_maximum, float(np.max(abs(state["ema_deviation"] - expected_state)))
        )
        grouped.setdefault((trace["condition"], method, row["day"]), []).append(
            (targets[mask], actual_prediction[mask])
        )
    means = {}
    for (condition, method, day), parts in grouped.items():
        target = np.concatenate([part[0] for part in parts]).astype(float)
        pred = np.concatenate([part[1] for part in parts]).astype(float)
        expected_score = r2_score(target, pred, multioutput="variance_weighted")
        actual_score = summary["conditions"][condition][method]["scores"]["per_day"][day]
        np.testing.assert_allclose(
            actual_score["r2_variance_weighted"], expected_score, atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            actual_score["r2_per_output"],
            r2_score(target, pred, multioutput="raw_values"),
            atol=1e-12,
            rtol=0,
        )
        means.setdefault((condition, method), []).append(expected_score)
    for (condition, method), scores in means.items():
        recorded = summary["conditions"][condition][method]["scores"]
        np.testing.assert_allclose(
            [recorded["r2_mean"], recorded["r2_population_std"]],
            [np.mean(scores), np.std(scores)],
            atol=1e-12,
            rtol=0,
        )
    return {
        "status": "passed",
        "traces_checked": len(traces),
        "day_scores_checked": len(grouped),
        "per_output_day_scores_checked": len(grouped) * 7,
        "condition_method_aggregates_checked": len(means),
        "max_absolute_prediction_difference": maximum,
        "max_absolute_ema_state_difference": state_maximum,
        "summary_sha256": sha(run / "summary.json"),
        "run_manifest_sha256": sha(run / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "index_sha256": sha(index_path),
        "baseline_manifest_sha256": sha(baseline_run / "manifest.json"),
        "baseline_model_sha256": {
            day: identity["model_sha256"] for day, identity in identities.items()
        },
        "trace_manifest_sha256": sha(run / "trace_manifest.json"),
        "audit_source_sha256": sha(__file__),
        "sealed_arrays_opened": False,
        "reference": (
            "scipy lfilter delayed EMA and batch affine predictions; sklearn float64 scoring"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source = output.with_name(output.stem + "-source.py")
    if output.exists() or source.exists():
        raise FileExistsError("Audit output and source snapshot must be fresh paths")
    with threadpool_limits(1):
        result = audit(args.run, args.index, args.baseline_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
