"""Forward-day development only. Sealed assessment requires a later frozen recipe."""

import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .artifacts import environment, sha256, source_identity, write_json
from .data import load_index
from .metrics import FileEvaluation, score_files
from .models import LinearDecoder, fit_linear

DEV_DAYS = {f"S{i}" for i in range(5)}
FOLDS = [(["S0", "S1"], "S2"), (["S0", "S1", "S2"], "S3"), (["S0", "S1", "S2", "S3"], "S4")]


def checked_development_index(path):
    """Reject unauthorized roles before opening any sample-level feature or target file."""
    path = Path(path)
    index = load_index(path)
    if index.get("schema_version") != 1 or index.get("role") != "development":
        raise ValueError("requires schema v1 development-only index")
    files = index["files"]
    if not files or any(item["day"] not in DEV_DAYS for item in files):
        raise ValueError("sealed or unknown day in development index")
    if any(item["provider_split"] != "held-in-calib" for item in files):
        raise ValueError("minival and held-out calibration are not development inputs")
    if len({item["file_id"] for item in files}) != len(files):
        raise ValueError("duplicate file identity")
    if {item["day"] for item in files} != DEV_DAYS:
        raise ValueError("all S0-S4 development days must be represented")
    for item in files:
        for key in ("features", "targets"):
            resolved = (path.parent / item[key]).resolve()
            if not resolved.is_relative_to(path.parent.resolve()):
                raise ValueError("prepared input escapes cache")
            if sha256(resolved) != item[f"{key}_sha256"]:
                raise ValueError(f"changed {key} for {item['file_id']}")
    return index


def load_development_files(path):
    index = checked_development_index(path)
    files = []
    for item in sorted(index["files"], key=lambda f: (int(f["day"][1:]), f["file_id"])):
        with np.load(Path(path).parent / item["features"], allow_pickle=False) as features:
            if set(features.files) != {"counts", "timestamps", "sequence_id", "channel_ids"}:
                raise ValueError("feature payload contains unexpected information")
            x = features["counts"]
            if x.shape != (item["n_bins"], 176) or not np.isfinite(x).all() or (x < 0).any():
                raise ValueError("invalid H1 neural counts")
            if not np.array_equal(features["channel_ids"], np.arange(176)):
                raise ValueError("unexpected H1 channel order")
            if not np.array_equal(features["sequence_id"], np.arange(len(x))):
                raise ValueError("invalid sequence IDs")
        with np.load(Path(path).parent / item["targets"], allow_pickle=False) as targets:
            y, mask = targets["targets"], targets["eval_mask"]
            if y.shape != (len(x), 7) or mask.shape != (len(x),) or mask.dtype != np.bool_:
                raise ValueError("invalid training/scoring arrays")
        files.append((item, x, y, mask))
    return index, files


def evaluate_linear(model, files, destination):
    records = []
    for item, x, y, mask in files:
        decoder = LinearDecoder(model)
        predictions = decoder.predict_chunk(x)
        np.save(destination / f"{item['file_id']}.npy", predictions, allow_pickle=False)
        records.append(FileEvaluation(item["file_id"], item["day"], predictions, y, mask))
    return records


def run_baselines(index_path, config_path, output, *, root):
    config = json.loads(Path(config_path).read_text())
    expected = [{"train": train, "evaluate": day} for train, day in FOLDS]
    if config["folds"] != expected or config["fir_taps"] != 12 or config["fir_tau_bins"] != 12:
        raise ValueError("unsupported development fold/filter contract")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "configuration": config,
        "environment": environment(),
        "source": source_identity(root, output),
        "index_sha256": sha256(index_path),
        "config_sha256": sha256(config_path),
        "regime": config["regime"],
    }
    write_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        index, files = load_development_files(index_path)
        write_json(output / "input_index.json", index)
        candidates = [("constant", 1, 0.01)] + [
            ("ridge", history, lam)
            for history in config["history_bins"]
            for lam in config["regularization"]
        ]
        summary = []
        with threadpool_limits(limits=config["threads"]):
            manifest["threadpools"] = threadpool_info()
            for kind, history, lam in candidates:
                name = f"{kind}-h{history}-lambda{lam:g}"
                combined, fold_times = [], []
                for train_days, eval_day in FOLDS:
                    fold_started = time.perf_counter()
                    destination = output / name / eval_day
                    destination.mkdir(parents=True)
                    training = [
                        (x, y, mask) for item, x, y, mask in files if item["day"] in train_days
                    ]
                    evaluation = [record for record in files if record[0]["day"] == eval_day]
                    model = fit_linear(
                        training, history=history, regularization=lam, constant=kind == "constant"
                    )
                    model.save(destination / "model.npz")
                    records = evaluate_linear(model, evaluation, destination)
                    scores = score_files(records)
                    write_json(destination / "scores.json", scores)
                    write_json(
                        destination / "identity.json",
                        {
                            "train_days": train_days,
                            "evaluation_day": eval_day,
                            "model_sha256": sha256(destination / "model.npz"),
                            "prediction_sha256": {
                                r.file_id: sha256(destination / f"{r.file_id}.npy") for r in records
                            },
                        },
                    )
                    combined.extend(records)
                    fold_times.append(time.perf_counter() - fold_started)
                    print(
                        json.dumps(
                            {
                                "candidate": name,
                                "day": eval_day,
                                "r2": scores["r2_mean"],
                                "seconds": fold_times[-1],
                            }
                        ),
                        flush=True,
                    )
                scores = score_files(combined)
                summary.append(
                    {
                        "candidate": name,
                        "kind": kind,
                        "history": history,
                        "regularization": lam,
                        "scores": scores,
                        "fold_wall_seconds": fold_times,
                    }
                )
                write_json(output / "summary.partial.json", summary)
        ridge = [item for item in summary if item["kind"] == "ridge"]
        selected = max(
            ridge,
            key=lambda item: (item["scores"]["r2_mean"], -item["history"], item["regularization"]),
        )
        write_json(
            output / "summary.json",
            {
                "candidates": summary,
                "selected_ridge": selected["candidate"],
                "selection_role": "development_only",
                "sealed_days_scored": [],
            },
        )
        manifest.update(
            {
                "status": "complete",
                "wall_seconds": time.perf_counter() - started,
                "summary_sha256": sha256(output / "summary.json"),
            }
        )
    except BaseException as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output
