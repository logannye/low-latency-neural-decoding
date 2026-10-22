"""Independent reconstruction of fixed M2 transfer artifacts; no production imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Changed artifact: {path}")


def read_json(path):
    return json.loads(Path(path).read_text())


def numeric(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def convolve(counts, weights):
    return np.column_stack(
        [
            np.convolve(counts[:, channel], weights, mode="full")[: len(counts)]
            for channel in range(counts.shape[1])
        ]
    )


def check_raw(path, prepared):
    """Read NWB datasets directly with HDF5; neither PyNWB nor runner loader."""
    with h5py.File(path, "r") as h5:
        units = h5["units"]
        np.testing.assert_array_equal(units["id"][:], np.arange(96))
        times = h5["acquisition/finger_vel/index/timestamps"][:]
        np.testing.assert_array_equal(times, h5["acquisition/finger_vel/mrs/timestamps"][:])
        np.testing.assert_array_equal(times, h5["acquisition/eval_mask/timestamps"][:])
        target = np.column_stack(
            [h5[f"acquisition/finger_vel/{label}/data"][:] for label in ("index", "mrs")]
        )
        mask = h5["acquisition/eval_mask/data"][:].astype(bool)
        starts = h5["intervals/trials/start_time"][:]
        stops = h5["intervals/trials/stop_time"][:]
        spikes = units["spike_times"][:]
        indices = units["spike_times_index"][:]
    edges = np.r_[times[0] + 0.02 - 0.02, times + 0.02]
    count_columns = []
    offset = 0
    for end in indices:
        channel_spikes = spikes[offset:end]
        counts, _ = np.histogram(channel_spikes, bins=edges)
        count_columns.append(counts)
        offset = end
    counts = np.column_stack(count_columns)
    np.testing.assert_array_equal(counts, prepared["counts"])
    np.testing.assert_array_equal(target, prepared["targets"])
    np.testing.assert_array_equal(mask, prepared["eval_mask"])
    np.testing.assert_array_equal(times, prepared["timestamps"])
    np.testing.assert_array_equal(times + 0.02, prepared["release_times"])
    np.testing.assert_array_equal(starts, prepared["trial_starts"])
    np.testing.assert_array_equal(stops, prepared["trial_stops"])
    trial_change = np.zeros(len(times), dtype=bool)
    trial_change[np.searchsorted(times, starts)] = True
    np.testing.assert_array_equal(trial_change, prepared["trial_change"])
    if len(starts) != 2 or counts.shape[1] != 96 or target.shape[1] != 2:
        raise ValueError("M2 schema changed")
    return {
        "bins": len(times),
        "scored_bins": int(mask.sum()),
        "trials": len(starts),
        "direct_hdf5_preparation_exact": True,
        "release_offset_seconds": 0.02,
    }


def audit(run):
    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs can be audited")
    for relative, expected in manifest["files"].items():
        verify(run / relative, expected)
    verify(run / "source.diff", manifest["source"]["diff_sha256"])
    for relative, expected in manifest["source"]["files"].items():
        verify(run / "source" / relative, expected)
    verify(run / "config.json", manifest["config_sha256"])
    verify(run / "summary.json", manifest["summary_sha256"])
    verify(run / "input_index.json", manifest["input_index_sha256"])
    config = read_json(run / "config.json")
    if config["source_pin"] != "770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f":
        raise ValueError("Unexpected FALCON revision")
    for relative, expected in config["source_sha256"].items():
        verify(run / "upstream" / relative, expected)
    inputs = read_json(run / "input_index.json")
    expected_ids = [
        "2020-10-19_Run1",
        "2020-10-19_Run2",
        "2020-10-20_Run1",
        "2020-10-20_Run2",
        "2020-10-27_Run1",
        "2020-10-27_Run2",
        "2020-10-28_Run1",
    ]
    if [item["file_id"] for item in inputs] != expected_ids:
        raise ValueError("Unexpected source file inventory")
    arrays, raw_checks = {}, {}
    for item, asset in zip(inputs, config["assets"], strict=True):
        file_id = item["file_id"]
        expected_role = "training" if file_id.startswith("2020-10-19_") else "evaluation"
        if item["role"] != expected_role or asset["role"] != expected_role:
            raise ValueError("Unauthorized fitting role")
        if item["asset_id"] != asset["asset_id"] or item["raw_sha256"] != asset["sha256"]:
            raise ValueError("Pinned asset identity mismatch")
        verify(item["raw_path"], asset["sha256"])
        if Path(item["raw_path"]).stat().st_size != asset["content_size"]:
            raise ValueError("Raw size mismatch")
        verify(run / "metadata" / f"{asset['asset_id']}.json", item["metadata_sha256"])
        metadata = read_json(run / "metadata" / f"{asset['asset_id']}.json")
        if metadata["digest"]["dandi:sha2-256"] != asset["sha256"]:
            raise ValueError("Provider metadata digest mismatch")
        prepared_path = run / "prepared" / f"{file_id}.npz"
        verify(prepared_path, item["prepared_sha256"])
        arrays[file_id] = numeric(prepared_path)
        raw_checks[file_id] = check_raw(item["raw_path"], arrays[file_id])
    train_ids = expected_ids[:2]
    eval_ids = expected_ids[2:]
    weights = np.exp(-np.arange(12, dtype=np.float64) / 12.0)
    weights /= weights.sum()
    filtered_train = [convolve(arrays[name]["counts"], weights) for name in train_ids]
    concatenated = np.concatenate(filtered_train)
    mean = concatenated.mean(axis=0)
    scale = np.maximum(concatenated.std(axis=0), 1e-6)
    x = np.concatenate(
        [
            ((values - mean) / scale)[arrays[name]["eval_mask"]]
            for name, values in zip(train_ids, filtered_train, strict=True)
        ]
    )
    y = np.concatenate([arrays[name]["targets"][arrays[name]["eval_mask"]] for name in train_ids])
    # Algebraic verification only, not an extra candidate or a model-selection fit.
    reference_fit = Ridge(alpha=0.1 * len(x), fit_intercept=True, solver="cholesky").fit(x, y)
    summary = read_json(run / "summary.json")
    if summary["training_file_ids"] != train_ids or summary["training_trials"] != 4:
        raise ValueError("Reported fitting roles/counts mismatch")
    result = {
        "status": "passed",
        "run_manifest_sha256": digest(run / "manifest.json"),
        "source_head": manifest["source"]["head"],
        "config_sha256": manifest["config_sha256"],
        "independent_methods": [
            "direct HDF5 arrays and histogram binning",
            "NumPy full-file convolution",
            "sklearn Ridge normal-equation parity",
            "sklearn per-day R2",
        ],
        "raw_checks": raw_checks,
        "training_bins": len(x),
        "training_seconds": len(x) * 0.02,
        "evaluation_bins": sum(len(arrays[name]["counts"]) for name in eval_ids),
        "models": {},
    }
    for kind in ("constant", "ridge"):
        model_path = run / "models" / f"{kind}.npz"
        verify(model_path, summary["models"][kind]["model_sha256"])
        model = numeric(model_path)
        if model["history"].item() != 1 or model["kind"].item() != kind:
            raise ValueError("Changed model contract")
        np.testing.assert_allclose(model["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(model["scale"], scale, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(model["weights"], weights)
        expected_coefficient = reference_fit.coef_.T if kind == "ridge" else np.zeros((96, 2))
        expected_intercept = reference_fit.intercept_ if kind == "ridge" else y.mean(axis=0)
        np.testing.assert_allclose(model["coefficient"], expected_coefficient, atol=1e-12, rtol=0)
        np.testing.assert_allclose(model["intercept"], expected_intercept, atol=1e-12, rtol=0)
        predictions, differences = {}, []
        for file_id in eval_ids:
            record = arrays[file_id]
            path = run / summary["models"][kind]["predictions"][file_id]["path"]
            verify(path, summary["models"][kind]["predictions"][file_id]["sha256"])
            prediction = np.load(path, allow_pickle=False)
            values = (convolve(record["counts"], weights) - mean) / scale
            reconstructed = values @ expected_coefficient + expected_intercept
            np.testing.assert_allclose(prediction, reconstructed, atol=1e-10, rtol=0)
            differences.append(float(np.max(np.abs(prediction - reconstructed))))
            predictions[file_id] = prediction
        per_day = {}
        for day in ("2020-10-20", "2020-10-27", "2020-10-28"):
            files = [name for name in eval_ids if name.startswith(day + "_")]
            target = np.concatenate(
                [arrays[name]["targets"][arrays[name]["eval_mask"]] for name in files]
            )
            prediction = np.concatenate(
                [predictions[name][arrays[name]["eval_mask"]] for name in files]
            )
            score = float(r2_score(target, prediction, multioutput="variance_weighted"))
            outputs = r2_score(target, prediction, multioutput="raw_values")
            stored = summary["models"][kind]["scores"]["per_day"][day]
            np.testing.assert_allclose(score, stored["r2_variance_weighted"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(outputs, stored["r2_per_output"], atol=1e-12, rtol=0)
            per_day[day] = {
                "r2_variance_weighted": score,
                "r2_per_output": outputs.tolist(),
                "scored_bins": len(target),
            }
        scores = np.array([item["r2_variance_weighted"] for item in per_day.values()])
        np.testing.assert_allclose(
            scores.mean(), summary["models"][kind]["scores"]["r2_mean"], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            scores.std(), summary["models"][kind]["scores"]["r2_population_std"], atol=1e-12, rtol=0
        )
        result["models"][kind] = {
            "per_day": per_day,
            "r2_mean": float(scores.mean()),
            "r2_population_std": float(scores.std()),
            "max_prediction_abs_error": max(differences),
            "independent_fit_atol": 1e-12,
            "coefficient_max_abs_error": float(
                np.max(np.abs(model["coefficient"] - expected_coefficient))
            ),
            "intercept_max_abs_error": float(
                np.max(np.abs(model["intercept"] - expected_intercept))
            ),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        report = audit(arguments.run)
    source = arguments.run / "independent-audit-source.py"
    shutil.copyfile(__file__, source)
    report["audit_source_sha256"] = digest(source)
    (arguments.run / "independent-audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
