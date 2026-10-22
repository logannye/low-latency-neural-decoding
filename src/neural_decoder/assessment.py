"""Fixed-recipe refit, two-stage sealed evaluation and separate prefix calibration.

Training and evaluation are distinct commands. Sealed arrays require an external
freeze digest; predictions are completed before the evaluator joins targets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .artifacts import environment, sha256, source_identity, write_json
from .assessment_data import (
    PRIMARY_METRICS,
    STAGES,
    load_assessment_files,
    validate_freeze,
)
from .data import UPSTREAM_PIN
from .metrics import FileEvaluation, score_files
from .models import LinearDecoder, fit_linear, load_model
from .temporal import TCNStream, export_tcn, load_tcn
from .train import load_development_files
from .train_temporal import fit_temporal


def read_json(path):
    return json.loads(Path(path).read_text())


def assessment_files(index_path, freeze_path, expected_sha256, *, root, cache):
    return load_assessment_files(index_path, freeze_path, expected_sha256, root=root, cache=cache)


def fit_stage(
    index_path,
    recipe_path,
    output,
    *,
    stage,
    root,
    cache,
    early_index=None,
    early_freeze=None,
    early_sha256=None,
):
    """Later fitting adds S5 only, with the identical pre-S5 recipe."""
    recipe = read_json(recipe_path)
    _, files = load_development_files(index_path)
    index_hashes = {"development": sha256(index_path)}
    if stage == "later":
        if not all((early_index, early_freeze, early_sha256)):
            raise ValueError("Later refit requires the early freeze and released S5 index")
        parent = validate_freeze(early_freeze, early_sha256, root=root, cache=cache)
        if parent["stage"] != "early" or parent["recipe"]["sha256"] != sha256(recipe_path):
            raise ValueError("Later refit changed the frozen early recipe")
        _, extra = assessment_files(early_index, early_freeze, early_sha256, root=root, cache=cache)
        files += extra
        index_hashes["early"] = sha256(early_index)
    if (
        sorted({r["day"] for r, *_ in files}, key=lambda d: int(d[1:]))
        != STAGES[stage]["training_days"]
    ):
        raise ValueError("Training days disagree with stage")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "stage": stage,
        "recipe_sha256": sha256(recipe_path),
        "index_sha256": index_hashes,
        "environment": environment(),
        "source": source_identity(root, output),
        "models": [],
    }
    write_json(output / "manifest.json", manifest)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        training = [(x, y, m) for _, x, y, m in files]
        with threadpool_limits(limits=1):
            for name, constant in [("constant", True), ("ridge", False)]:
                model = fit_linear(
                    training,
                    history=recipe["ridge"]["history"],
                    regularization=recipe["ridge"]["regularization"],
                    constant=constant,
                )
                path = output / f"{name}.npz"
                model.save(path)
                manifest["models"].append(
                    {
                        "name": name,
                        "kind": "linear",
                        "path": str(path.resolve().relative_to(Path(root).resolve())),
                        "sha256": sha256(path),
                        "training_days": STAGES[stage]["training_days"],
                        "regime": "frozen_zero_calibration",
                    }
                )
            config = recipe["temporal"]
            for seed in config["seeds"]:
                model, transforms, losses = fit_temporal(training, config, seed, config["epochs"])
                path = output / f"tcn-seed-{seed}.npz"
                export_tcn(model, path, **transforms)
                write_json(output / f"tcn-seed-{seed}-losses.json", losses)
                manifest["models"].append(
                    {
                        "name": f"tcn-seed-{seed}",
                        "kind": "tcn",
                        "seed": seed,
                        "path": str(path.resolve().relative_to(Path(root).resolve())),
                        "sha256": sha256(path),
                        "training_days": STAGES[stage]["training_days"],
                        "regime": "frozen_zero_calibration",
                    }
                )
                print(json.dumps({"stage": stage, "fitted_seed": seed}), flush=True)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.set_num_threads(previous)
        write_json(output / "manifest.json", manifest)
    return output


def freeze_stage(
    run, recipe_path, destination, *, root, cache, parent_freeze=None, parent_sha256=None
):
    """Bind fitted artifacts to committed source before the release decision."""
    root, cache, destination = Path(root).resolve(), Path(cache), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    manifest = read_json(Path(run) / "manifest.json")
    if manifest["status"] != "complete" or manifest["recipe_sha256"] != sha256(recipe_path):
        raise ValueError("Refit is incomplete or recipe changed")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    paths = sorted((root / "src/neural_decoder").rglob("*.py")) + [Path(recipe_path).resolve()]
    hashes = {}
    import hashlib

    for path in paths:
        relative = str(path.relative_to(root))
        committed = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root)
        hashes[relative] = sha256(path)
        if hashlib.sha256(committed).hexdigest() != hashes[relative]:
            raise ValueError(f"Uncommitted final source: {relative}")
    # Fitting must use the same implementation bytes too.
    for relative, digest in hashes.items():
        if relative.startswith("src/") and manifest["source"]["files"].get(relative) != digest:
            raise ValueError("Training source differs from final source")
    stage = manifest["stage"]
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "stage": stage,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {"git_revision": revision, "files_sha256": hashes},
        "recipe": {
            "path": str(Path(recipe_path).resolve().relative_to(root)),
            "sha256": sha256(recipe_path),
        },
        "models": manifest["models"],
        "data": {"upstream_pin": UPSTREAM_PIN},
        "evaluation_days": STAGES[stage]["evaluation_days"],
        "provider_split": STAGES[stage]["provider_split"],
        "primary_metrics": PRIMARY_METRICS,
    }
    for filename, key in [
        ("manifest.json", "manifest_sha256"),
        ("schema_audit.json", "schema_audit_sha256"),
        ("split_manifest.json", "split_manifest_sha256"),
    ]:
        freeze["data"][key] = sha256(cache / filename)
    if stage == "later":
        if not parent_freeze or not parent_sha256:
            raise ValueError("Later freeze requires early parent")
        freeze["parent_freeze"] = {
            "path": str(Path(parent_freeze).resolve().relative_to(root)),
            "sha256": parent_sha256,
        }
    write_json(destination, freeze)
    digest = sha256(destination)
    validate_freeze(destination, digest, root=root, cache=cache)
    return digest


def make_decoder(model_record, root):
    path = Path(root) / model_record["path"]
    if sha256(path) != model_record["sha256"]:
        raise ValueError("Model changed after freeze")
    if model_record["kind"] == "linear":
        return LinearDecoder(load_model(path))
    if model_record["kind"] == "tcn":
        return TCNStream(load_tcn(path), implementation="incremental")
    raise ValueError("Unknown frozen decoder kind")


def evaluate_stage(index_path, freeze_path, expected_sha256, output, *, root, cache):
    freeze = validate_freeze(freeze_path, expected_sha256, root=root, cache=cache)
    index, files = assessment_files(
        index_path, freeze_path, expected_sha256, root=root, cache=cache
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "freeze_sha256": expected_sha256,
        "index_sha256": sha256(index_path),
        "environment": environment(),
        "source": source_identity(root, output),
        "models": [],
    }
    write_json(output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            for model in freeze["models"]:
                folder = output / model["name"]
                folder.mkdir()
                records, hashes = [], {}
                for item, x, y, mask in files:
                    decoder = make_decoder(model, root)
                    prediction = decoder.predict_chunk(x)
                    path = folder / f"{item['file_id']}.npy"
                    np.save(path, prediction, allow_pickle=False)
                    hashes[item["file_id"]] = sha256(path)
                    records.append(
                        FileEvaluation(item["file_id"], item["day"], prediction, y, mask)
                    )
                scores = score_files(records)
                write_json(folder / "scores.json", scores)
                manifest["models"].append(
                    {"model": model, "scores": scores, "prediction_sha256": hashes}
                )
                print(
                    json.dumps(
                        {"stage": index["stage"], "model": model["name"], "r2": scores["r2_mean"]}
                    ),
                    flush=True,
                )
        summary = {
            "stage": index["stage"],
            "designation": "local public calibration recording assessment",
            "regime": "frozen_zero_new_day_calibration",
            "models": manifest["models"],
            "freeze_sha256": expected_sha256,
            "index_sha256": sha256(index_path),
        }
        write_json(output / "summary.json", summary)
        manifest.update(status="complete", summary_sha256=sha256(output / "summary.json"))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output


def prefix_offset(predictions, targets, mask, budget_bins):
    """Separate calibration job: only explicitly passed preceding rows are usable."""
    predictions, targets, mask = np.asarray(predictions), np.asarray(targets), np.asarray(mask)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 7:
        raise ValueError("Calibration arrays must align")
    if mask.dtype != np.bool_ or mask.shape != (len(targets),):
        raise ValueError("Calibration mask must align")
    if budget_bins not in (0, 250, 750, 1500) or len(targets) != budget_bins:
        raise ValueError("Calibration job must receive exactly its declared prefix")
    if not budget_bins:
        return np.zeros(7)
    if not mask.any() or not np.isfinite(targets[mask]).all() or not np.isfinite(predictions).all():
        raise ValueError("Calibration prefix needs finite scored rows")
    return (targets[mask].astype(np.float64) - predictions[mask]).mean(0)


def checked_primary_run(primary_run, freeze, expected_sha256, index_path):
    """Bind calibration to the completed primary run and exact input/model set."""
    primary_run = Path(primary_run)
    manifest = read_json(primary_run / "manifest.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("freeze_sha256") != expected_sha256
        or manifest.get("index_sha256") != sha256(index_path)
    ):
        raise ValueError("Primary assessment is incomplete or differs from frozen inputs")
    summary_path = primary_run / "summary.json"
    if sha256(summary_path) != manifest.get("summary_sha256"):
        raise ValueError("Primary summary identity changed")
    summary = read_json(summary_path)
    if (
        summary.get("freeze_sha256") != expected_sha256
        or summary.get("index_sha256") != sha256(index_path)
        or summary.get("stage") != freeze["stage"]
        or summary.get("models") != manifest.get("models")
    ):
        raise ValueError("Primary summary and completion manifest disagree")
    if [entry["model"] for entry in manifest["models"]] != freeze["models"]:
        raise ValueError("Primary model inventory differs from frozen artifacts")
    return manifest


def calibration_study(
    index_path, freeze_path, expected_sha256, primary_run, output, *, root, cache
):
    """0/5/15/30s per-file intercept-only calibration; all score the same suffix."""
    freeze = validate_freeze(freeze_path, expected_sha256, root=root, cache=cache)
    if freeze["stage"] != "later":
        raise ValueError("The separate calibration study is prespecified for later days only")
    primary = checked_primary_run(primary_run, freeze, expected_sha256, index_path)
    _, files = assessment_files(index_path, freeze_path, expected_sha256, root=root, cache=cache)
    expected_streams = {item["file_id"] for item, *_ in files}
    for entry in primary["models"]:
        if set(entry["prediction_sha256"]) != expected_streams:
            raise ValueError("Primary predictions do not cover exact assessment streams")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "regime": "explicit_supervised_prefix_calibration",
        "freeze_sha256": expected_sha256,
        "index_sha256": sha256(index_path),
        "primary_summary_sha256": sha256(Path(primary_run) / "summary.json"),
        "source": source_identity(root, output),
        "environment": environment(),
    }
    write_json(output / "manifest.json", manifest)
    try:
        results, identities = [], {}
        for entry in primary["models"]:
            name = entry["model"]["name"]
            if name == "constant":
                continue
            for budget in (0, 250, 750, 1500):
                folder = output / name / f"prefix-{budget}"
                folder.mkdir(parents=True)
                records, accounting = [], []
                for item, _, y, mask in files:
                    path = Path(primary_run) / name / f"{item['file_id']}.npy"
                    if sha256(path) != entry["prediction_sha256"][item["file_id"]]:
                        raise ValueError("Primary prediction trace changed")
                    prediction = np.load(path, allow_pickle=False)
                    if prediction.shape != y.shape or not np.isfinite(prediction).all():
                        raise ValueError("Primary prediction trace has invalid shape or values")
                    if len(y) <= 1500:
                        raise ValueError("File too short for common suffix")
                    offset = prefix_offset(prediction[:budget], y[:budget], mask[:budget], budget)
                    adjusted = prediction[1500:] + offset
                    saved = folder / f"{item['file_id']}.npy"
                    np.save(saved, adjusted, allow_pickle=False)
                    identities[str(saved.relative_to(output))] = sha256(saved)
                    records.append(
                        FileEvaluation(
                            item["file_id"], item["day"], adjusted, y[1500:], mask[1500:]
                        )
                    )
                    accounting.append(
                        {
                            "file_id": item["file_id"],
                            "day": item["day"],
                            "nominal_label_seconds": budget * 0.02,
                            "scored_prefix_bins": int(mask[:budget].sum()),
                            "scored_label_seconds": int(mask[:budget].sum()) * 0.02,
                            "suffix_start_bin": 1500,
                            "suffix_scored_bins": int(mask[1500:].sum()),
                            "offset": offset.tolist(),
                        }
                    )
                results.append(
                    {
                        "model": name,
                        "prefix_seconds_per_file": budget * 0.02,
                        "nominal_label_seconds_per_day": budget * 0.04,
                        "scores": score_files(records),
                        "files": accounting,
                    }
                )
        summary = {
            "regime": manifest["regime"],
            "method": "unregularized mean prediction residual per output on scored prefix rows",
            "selection": "prespecified fixed budgets; no selection on suffix scores",
            "state": "all arms ingest preceding neural bins identically; reset each file",
            "source": manifest["source"],
            "environment": manifest["environment"],
            "freeze_sha256": expected_sha256,
            "index_sha256": manifest["index_sha256"],
            "primary_summary_sha256": manifest["primary_summary_sha256"],
            "results": results,
            "prediction_sha256": identities,
        }
        write_json(output / "summary.json", summary)
        manifest.update(status="complete", summary_sha256=sha256(output / "summary.json"))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "freeze", "evaluate", "calibration"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, default=Path("data/neural_h1"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["early", "later"])
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--early-index", type=Path)
    parser.add_argument("--early-freeze", type=Path)
    parser.add_argument("--early-sha256")
    args = parser.parse_args()
    common = {"root": args.root, "cache": args.cache}
    if args.command == "fit":
        result = fit_stage(
            args.index,
            args.recipe,
            args.output,
            stage=args.stage,
            **common,
            early_index=args.early_index,
            early_freeze=args.early_freeze,
            early_sha256=args.early_sha256,
        )
    elif args.command == "freeze":
        result = freeze_stage(
            args.run,
            args.recipe,
            args.output,
            **common,
            parent_freeze=args.early_freeze,
            parent_sha256=args.early_sha256,
        )
    elif args.command == "evaluate":
        result = evaluate_stage(args.index, args.freeze, args.sha256, args.output, **common)
    else:
        result = calibration_study(
            args.index, args.freeze, args.sha256, args.run, args.output, **common
        )
    print(result)


if __name__ == "__main__":
    main()
