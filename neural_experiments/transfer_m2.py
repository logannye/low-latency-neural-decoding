"""Bounded M2 minival transfer; run only after the primary-work release.

No H1 arrays, fitted H1 weights, or decoder source files are changed. The fixed
config admits seven exact public assets. All output is local and failure-safe.
This command performs no latency benchmark and submits nothing to EvalAI.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import traceback
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from neural_decoder.artifacts import environment, sha256, source_identity, write_json
from neural_decoder.metrics import regression_metrics
from neural_decoder.models import LinearDecoder, fit_linear, load_model

PIN = "770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f"
TRAIN_DAY = "2020-10-19"
EVALUATION_DAYS = ["2020-10-20", "2020-10-27", "2020-10-28"]


def checked_config(path):
    config = json.loads(Path(path).read_text())
    fixed = {
        "source_pin": PIN,
        "channels": 96,
        "output_names": ["index", "mrs"],
        "bin_seconds": 0.02,
        "history": 1,
        "regularization": 0.1,
        "fir_taps": 12,
        "fir_tau_bins": 12,
        "threads": 1,
        "train_days": [TRAIN_DAY],
        "evaluation_days": EVALUATION_DAYS,
        "raw_bytes": 124175536,
        "total_added_bytes_cap": 384 * 1024**2,
    }
    for key, value in fixed.items():
        if config[key] != value:
            raise ValueError(f"Locked transfer setting changed: {key}")
    assets = config["assets"]
    expected_sessions = [
        "2020-10-19_Run1",
        "2020-10-19_Run2",
        "2020-10-20_Run1",
        "2020-10-20_Run2",
        "2020-10-27_Run1",
        "2020-10-27_Run2",
        "2020-10-28_Run1",
    ]
    if [a["session"]["identifier"] for a in assets] != expected_sessions:
        raise ValueError("Changed asset/day inventory")
    if len({a["asset_id"] for a in assets}) != 7:
        raise ValueError("Duplicated asset")
    if sum(a["content_size"] for a in assets) != config["raw_bytes"]:
        raise ValueError("Asset size sum mismatch")
    for i, asset in enumerate(assets):
        if asset["role"] != ("training" if i < 2 else "evaluation"):
            raise ValueError("Changed training/evaluation role")
        if asset["download_url"] != (
            f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
        ):
            raise ValueError("Unexpected download provider")
        if Path(asset["path"]).name != (
            "sub-MonkeyN-held-in-minival_ses-"
            + asset["session"]["identifier"].replace("_", "-")
            + "_behavior+ecephys.nwb"
        ):
            raise ValueError("Unexpected provider path")
    return config


def reference_functions(root, config, out):
    """Execute only hash-checked pinned definitions, with their real dependencies."""
    from pynwb import NWBHDF5IO

    upstream = root / "data/neural_h1/upstream"
    for relative, expected in config["source_sha256"].items():
        path = upstream / relative
        if sha256(path) != expected:
            raise ValueError(f"Pinned upstream source changed: {relative}")
        target = out / "upstream" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(upstream / "LICENSE", out / "upstream/LICENSE")
    tasks = SimpleNamespace(**{name: name for name in ("h1", "h2", "m1", "m2", "b1")})
    namespace = {
        "np": np,
        "Path": Path,
        "NWBHDF5IO": NWBHDF5IO,
        "FalconTask": tasks,
        "r2_score": r2_score,
    }
    for filename, names in (
        ("dataloaders.py", {"bin_units", "load_nwb"}),
        ("evaluator.py", {"compute_metrics_regression"}),
    ):
        source = upstream / "falcon_challenge" / filename
        definitions = [
            node
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        if {node.name for node in definitions} != names:
            raise ValueError("Pinned reference definitions missing")
        for node in definitions:
            node.decorator_list = []
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
        exec(compile(module, str(source), "exec"), namespace)
    return namespace["load_nwb"], namespace["compute_metrics_regression"]


def acquire(asset, raw_dir, receipts):
    """Complete NWB download only; exact expected size/digest, no range reads."""
    target = raw_dir / Path(asset["path"]).name
    # Re-fetch public metadata before downloading, preserving complete response.
    request = urllib.request.Request(asset["metadata_url"], headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("Unexpected oversized metadata")
    metadata = json.loads(payload)
    for key, expected in (
        ("identifier", asset["asset_id"]),
        ("path", asset["path"]),
        ("contentSize", asset["content_size"]),
    ):
        if metadata[key] != expected:
            raise ValueError(f"Provider metadata identity changed: {key}")
    if metadata["digest"]["dandi:sha2-256"] != asset["sha256"]:
        raise ValueError("Provider digest changed after lock")
    receipt = receipts / f"{asset['asset_id']}.json"
    receipt.write_bytes(payload)
    if target.exists():
        if target.stat().st_size != asset["content_size"] or sha256(target) != asset["sha256"]:
            raise ValueError(f"Existing raw cache failed digest: {target.name}")
        downloaded = False
    else:
        temporary = target.with_suffix(".nwb.partial")
        if temporary.exists():
            raise FileExistsError(f"Retained incomplete download: {temporary}")
        size = 0
        try:
            with urllib.request.urlopen(asset["download_url"], timeout=60) as response:
                with temporary.open("xb") as output:
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > asset["content_size"]:
                            raise ValueError("Download exceeds pinned asset size")
                        output.write(block)
            if size != asset["content_size"] or sha256(temporary) != asset["sha256"]:
                raise ValueError("Downloaded asset size/digest mismatch")
            temporary.replace(target)
        except BaseException:
            # Preserve failure metadata in the result, but remove unusable partial bytes.
            temporary.unlink(missing_ok=True)
            raise
        downloaded = True
    return target, {
        "asset_id": asset["asset_id"],
        "raw_sha256": sha256(target),
        "metadata_sha256": sha256(receipt),
        "downloaded": downloaded,
        "raw_bytes": target.stat().st_size,
        "raw_path": str(target),
    }


def independent_bins(spikes, timestamps, bin_seconds):
    """Searchsorted reference for the official regular-bin histogram contract."""
    if len(timestamps) < 2 or not np.allclose(np.diff(timestamps), bin_seconds):
        raise ValueError("Expected regularly spaced M2 timestamps; do not silently resample")
    ends = timestamps + bin_seconds
    edges = np.r_[ends[0] - bin_seconds, ends]
    counts = np.empty((len(timestamps), len(spikes)), dtype=np.int64)
    for channel, times in enumerate(spikes):
        times = np.sort(np.asarray(times, dtype=np.float64))
        if not np.isfinite(times).all():
            raise ValueError("Nonfinite spike time")
        positions = np.searchsorted(times, edges, side="left")
        positions[-1] = np.searchsorted(times, edges[-1], side="right")
        counts[:, channel] = np.diff(positions)
    if (counts > 255).any():
        raise ValueError("Pinned uint8 loader would overflow; stop rather than reproduce overflow")
    return counts


def prepare(path, asset, config, reference_loader):
    """Offline ingestion: behavior and masks stay in scorer/trainer records."""
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        ids = units.index.to_numpy()
        if not np.array_equal(ids, np.arange(config["channels"])):
            raise ValueError("Unexpected unit identities/order")
        container = nwb.acquisition["finger_vel"]
        names = list(container.time_series)
        if names != config["output_names"]:
            raise ValueError(f"Unexpected velocity output order: {names}")
        series = [container.get_timeseries(name) for name in names]
        times = np.asarray(series[0].timestamps[:], dtype=np.float64)
        if any(not np.array_equal(times, item.timestamps[:]) for item in series[1:]):
            raise ValueError("Output timestamps disagree")
        targets = np.column_stack([item.data[:] for item in series]).astype(np.float64)
        target_units = [str(item.unit) for item in series]
        if target_units != ["AU", "AU"]:
            raise ValueError(f"Unexpected target units: {target_units}")
        mask_series = nwb.acquisition["eval_mask"]
        if not np.array_equal(times, mask_series.timestamps[:]):
            raise ValueError("Mask timestamps disagree")
        raw_mask = np.asarray(mask_series.data[:])
        if not np.isin(raw_mask, [0, 1]).all():
            raise ValueError("Nonboolean provider mask")
        mask = raw_mask.astype(bool)
        trial_table = nwb.trials.to_dataframe()
        starts = trial_table["start_time"].to_numpy()
        stops = trial_table["stop_time"].to_numpy()
        if len(starts) != config["expected_trials_per_file"]:
            raise ValueError("Provider minival trial count changed")
        if (np.diff(starts) <= 0).any() or (stops < starts).any():
            raise ValueError("Invalid trial intervals")
        counts = independent_bins(units.spike_times, times, config["bin_seconds"])
    if targets.shape != (len(times), 2) or mask.shape != (len(times),):
        raise ValueError("Misaligned targets/mask")
    if not np.isfinite(targets).all() or not np.isfinite(times).all() or mask.sum() < 2:
        raise ValueError("Invalid target/time/scored rows")
    reference_counts, reference_targets, reference_trials, reference_mask = reference_loader(
        path, "m2"
    )
    np.testing.assert_array_equal(counts, reference_counts)
    np.testing.assert_array_equal(targets, reference_targets)
    np.testing.assert_array_equal(mask, reference_mask)
    trial_change = np.zeros(len(times), dtype=bool)
    trial_change[np.searchsorted(times, starts)] = True
    np.testing.assert_array_equal(trial_change, reference_trials)
    day = asset["session"]["identifier"][:10]
    return {
        "file_id": asset["session"]["identifier"],
        "day": day,
        "role": asset["role"],
        "counts": counts,
        "targets": targets,
        "eval_mask": mask,
        "timestamps": times,
        "release_times": times + 0.02,
        "trial_change": trial_change,
        "trial_starts": starts,
        "trial_stops": stops,
        "metadata": {
            "num_bins": len(times),
            "num_scored_bins": int(mask.sum()),
            "num_trials": len(starts),
            "output_names": names,
            "target_units": target_units,
            "unit_ids": ids.tolist(),
            "nominal_seconds": float(len(times) * 0.02),
            "elapsed_timestamp_seconds": float(times[-1] - times[0] + 0.02),
            "trial_interval_seconds": float((stops - starts).sum()),
            "official_loader_exact_parity": True,
        },
    }


def independent_prediction(model, counts):
    """Whole-file convolution; does not reuse incremental filter/history state."""
    filtered = np.column_stack(
        [
            np.convolve(counts[:, c], model.weights, mode="full")[: len(counts)]
            for c in range(counts.shape[1])
        ]
    )
    values = (filtered - model.mean) / model.scale
    columns = []
    for lag in range(model.history):
        delayed = np.zeros_like(values)
        delayed[lag:] = values[: len(values) - lag]
        columns.append(delayed)
    return np.concatenate(columns, axis=1) @ model.coefficient + model.intercept


def predict_counts(model, counts):
    """The complete predictor boundary: no targets, masks, day or trial metadata."""
    return LinearDecoder(model).predict_chunk(counts)


def verify_stream(model, counts, prediction, atol):
    batch = independent_prediction(model, counts)
    np.testing.assert_allclose(prediction, batch, atol=atol, rtol=0)
    cut = len(counts) // 2
    changed = counts.copy()
    changed[cut:] += 7
    future = predict_counts(model, changed)
    np.testing.assert_array_equal(prediction[:cut], future[:cut])
    decoder = LinearDecoder(model)
    chunks = []
    offset = 0
    for width in (1, 7, 31, 2, 64, len(counts)):
        end = min(offset + width, len(counts))
        chunks.append(decoder.predict_chunk(counts[offset:end]))
        offset = end
    chunked = np.concatenate(chunks)
    np.testing.assert_array_equal(prediction, chunked)
    decoder.reset()
    reset = decoder.predict_chunk(counts)
    np.testing.assert_array_equal(prediction, reset)
    # Reject attempts to extend the predictor API with evaluation-only inputs.
    for forbidden in ("targets", "eval_mask", "trial_change"):
        try:
            predict_counts(model, counts, **{forbidden: np.zeros(len(counts))})
        except TypeError:
            pass
        else:
            raise AssertionError("Evaluation metadata entered predictor API")
    return {
        "max_batch_stream_abs_error": float(np.max(np.abs(batch - prediction))),
        "future_perturbation_exact_prefix": True,
        "chunk_partition_exact": True,
        "file_reset_exact": True,
        "evaluation_metadata_arguments_rejected": True,
    }


def score_calendar_days(records, predictions, official):
    per_day = {}
    ordered = sorted(records, key=lambda item: (item["day"], item["file_id"]))
    lengths = {}
    for day in EVALUATION_DAYS:
        files = [r for r in ordered if r["day"] == day]
        if not files:
            raise ValueError(f"Missing evaluation day: {day}")
        target = np.concatenate([r["targets"][r["eval_mask"]] for r in files])
        prediction = np.concatenate([predictions[r["file_id"]][r["eval_mask"]] for r in files])
        scores = regression_metrics(target, prediction)
        sklearn_score = float(r2_score(target, prediction, multioutput="variance_weighted"))
        np.testing.assert_allclose(
            scores["r2_variance_weighted"], sklearn_score, atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            scores["r2_per_output"],
            r2_score(target, prediction, multioutput="raw_values"),
            atol=1e-12,
            rtol=0,
        )
        scores.update(
            {
                "sklearn_variance_weighted_r2": sklearn_score,
                "file_ids": [r["file_id"] for r in files],
                "num_files": len(files),
                "num_bins": sum(len(r["counts"]) for r in files),
            }
        )
        per_day[day] = scores
        lengths[day] = [len(r["counts"]) for r in files]
    values = np.array([r["r2_variance_weighted"] for r in per_day.values()])
    pinned = official(
        np.concatenate([predictions[r["file_id"]] for r in ordered]),
        np.concatenate([r["targets"][r["eval_mask"]] for r in ordered]),
        np.concatenate([r["eval_mask"] for r in ordered]),
        lengths,
    )
    np.testing.assert_allclose(
        [values.mean(), values.std(ddof=0)],
        [pinned["R2 Mean"], pinned["R2 Std."]],
        atol=1e-12,
        rtol=0,
    )
    return {
        "evaluation_kind": "local_minival_transfer",
        "metric": "mean_day_variance_weighted_r2",
        "per_day": per_day,
        "r2_mean": float(values.mean()),
        "r2_population_std": float(values.std(ddof=0)),
        "num_days": len(per_day),
        "num_files": len(ordered),
        "official_helper_parity": True,
        "independent_sklearn_parity": True,
    }


def run(root, config_path, out):
    root, config_path, out = Path(root).resolve(), Path(config_path).resolve(), Path(out).resolve()
    config = checked_config(config_path)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "started",
        "experiment": config["name"],
        "environment": environment(),
        "config_sha256": sha256(config_path),
        "scope": config["scope"],
        "files": {},
    }
    write_json(out / "manifest.json", manifest)
    try:
        # PyNWB creates a schema cache on import. Reuse the existing local cache;
        # module import and --help must not write into the user's home directory.
        os.environ.setdefault("XDG_CACHE_HOME", str(root / "data/neural_h1/library_cache"))
        shutil.copyfile(config_path, out / "config.json")
        manifest["source"] = source_identity(root, out)
        loader, official = reference_functions(root, config, out)
        raw_dir = root / "data/neural_m2/raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in ("metadata", "prepared", "models", "predictions"):
            (out / name).mkdir()
        if shutil.disk_usage(out).free < config["total_added_bytes_cap"]:
            raise ValueError("Insufficient free disk for bounded transfer")
        records, receipts = [], []
        for asset in config["assets"]:
            path, receipt = acquire(asset, raw_dir, out / "metadata")
            record = prepare(path, asset, config, loader)
            prepared = out / "prepared" / f"{record['file_id']}.npz"
            np.savez_compressed(
                prepared,
                **{
                    key: record[key]
                    for key in (
                        "counts",
                        "targets",
                        "eval_mask",
                        "timestamps",
                        "release_times",
                        "trial_change",
                        "trial_starts",
                        "trial_stops",
                    )
                },
            )
            receipt.update(
                {
                    "prepared_sha256": sha256(prepared),
                    "role": record["role"],
                    "file_id": record["file_id"],
                    "day": record["day"],
                    "schema": record["metadata"],
                }
            )
            receipts.append(receipt)
            records.append(record)
            write_json(out / "input_index.json", receipts)
        training = [r for r in records if r["role"] == "training"]
        evaluation = [r for r in records if r["role"] == "evaluation"]
        if {r["day"] for r in training} != {TRAIN_DAY}:
            raise ValueError("Training role violation")
        if any(r["day"] <= TRAIN_DAY for r in evaluation):
            raise ValueError("Evaluation chronology violation")
        train_tuples = [(r["counts"], r["targets"], r["eval_mask"]) for r in training]
        result = {
            "scope": config["scope"],
            "training_file_ids": [r["file_id"] for r in training],
            "training_nominal_seconds": sum(r["metadata"]["nominal_seconds"] for r in training),
            "training_scored_seconds": float(sum(r["eval_mask"].sum() for r in training) * 0.02),
            "training_trials": sum(r["metadata"]["num_trials"] for r in training),
            "models": {},
            "timing_claim": "none; numerical transfer only",
        }
        for kind in ("constant", "ridge"):
            model = fit_linear(
                train_tuples,
                history=config["history"],
                regularization=config["regularization"],
                constant=kind == "constant",
            )
            expected_fir = np.exp(-np.arange(12) / 12)
            expected_fir /= expected_fir.sum()
            np.testing.assert_array_equal(model.weights, expected_fir)
            model_path = out / "models" / f"{kind}.npz"
            model.save(model_path)
            model = load_model(model_path)
            predictions, checks, paths = {}, {}, {}
            for record in evaluation:
                file_id = record["file_id"]
                prediction = predict_counts(model, record["counts"])
                prediction_path = out / "predictions" / f"{kind}-{file_id}.npy"
                np.save(prediction_path, prediction, allow_pickle=False)
                predictions[file_id] = prediction
                checks[file_id] = verify_stream(
                    model, record["counts"], prediction, config["parity_atol"]
                )
                paths[file_id] = {
                    "path": str(prediction_path.relative_to(out)),
                    "sha256": sha256(prediction_path),
                }
            result["models"][kind] = {
                "scores": score_calendar_days(evaluation, predictions, official),
                "checks": checks,
                "predictions": paths,
                "model_sha256": sha256(model_path),
            }
            write_json(out / "summary.json", result)
        ridge = result["models"]["ridge"]["scores"]
        baseline = result["models"]["constant"]["scores"]
        result["ridge_minus_constant"] = {
            "mean_r2": ridge["r2_mean"] - baseline["r2_mean"],
            "per_day": {
                day: ridge["per_day"][day]["r2_variance_weighted"]
                - baseline["per_day"][day]["r2_variance_weighted"]
                for day in EVALUATION_DAYS
            },
        }
        write_json(out / "summary.json", result)
        added = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) + config["raw_bytes"]
        if added > config["total_added_bytes_cap"]:
            raise ValueError("Added-storage cap exceeded; retain evidence and stop")
        manifest.update(
            {
                "status": "complete",
                "summary_sha256": sha256(out / "summary.json"),
                "input_index_sha256": sha256(out / "input_index.json"),
                "total_added_bytes": added,
            }
        )
        manifest["files"] = {
            str(p.relative_to(out)): sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(out / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "run": str(out),
                    "ridge_r2_mean": ridge["r2_mean"],
                    "constant_r2_mean": baseline["r2_mean"],
                }
            )
        )
    except BaseException as error:
        manifest.update(
            {"status": "failed", "failure": repr(error), "traceback": traceback.format_exc()}
        )
        write_json(out / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--released-after-primary-work",
        action="store_true",
        help="Explicit orchestration release; this command downloads and computes",
    )
    args = parser.parse_args()
    if not args.released_after_primary_work:
        parser.error("Download/compute remains gated until required primary work and timing finish")
    config = args.config or args.root / "neural_experiments/configs/m2-minival-transfer-v1.json"
    output = args.output or args.root / "results/neural-m2/minival-transfer-v1"
    with threadpool_limits(limits=1):
        run(args.root, config, output)


if __name__ == "__main__":
    sys.exit(main())
