"""Final H1 artifact inventory and fixed five-path conversion quality evidence.

The experiment layer preserves every historical source/configuration byte. Sample
features enter prediction without labels; saved predictions join masks/targets
only in the subsequent scoring pass. Every invocation owns a create-new folder.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from neural_decoder.artifacts import environment, sha256, source_identity, write_json
from neural_decoder.assessment_data import _artifact, _read_json, load_assessment_index
from neural_decoder.data import load_feature_file, load_target_file
from neural_decoder.metrics import FileEvaluation, score_files
from neural_decoder.optimized import ONNXTCNStream, model_fingerprint
from neural_decoder.temporal import TCNStream, load_tcn
from neural_decoder.train import checked_development_index

ARMS = ("A", "B", "C", "D", "E")
GRAPH_NAMES = {"C": "float32.onnx", "D": "quantization-input.onnx", "E": "dynamic-int8.onnx"}
TOLERANCES = {"rtol": 3e-5, "atol": 1e-7, "day_r2_atol": 1e-5, "int8_day_r2_min_delta": -0.005}
VERSIONS = {"onnx": "1.19.1", "onnxruntime": "1.23.2"}
MODEL_HASHES = (
    "95f6a0b3131e1971f146298e23c95a42f7da28467d8acc24f37f916d9dc049ec",
    "b33b6ed1b38bad2d82a3605f6be4ae0707faae77361596245f1dd484f621579b",
    "12aed331e7bac3c9f42f67d75ff56dac32f7dcbffde0c942d96215dac8263627",
)
PINNED_INPUTS = {
    "freeze": "fd458c649b2af3c78e90ab2298821b2e28a1dc38011c7cc5ee3b29ef19dcea1c",
    "parent_freeze": "471c01b3fdc030c432b61c8188641c110b56f97aeb25ea64f5455312849a1fff",
    "assessment_index": "dd27945ae0f2b9aaab9e2ae6d6f35e77ae811f2cc0b95e19286c75eb7f2dc717",
    "development_index": "a87f1e638114ec41c5938366876c1789ceea2150b1c3a144e88f023b05d562c2",
}

read_json = _read_json
artifact = _artifact


def load_config(path):
    config = read_json(Path(path))
    if (
        config.get("schema_version") != 1
        or config.get("seeds") != [0, 1, 2]
        or config.get("timing_seed") != 0
        or config.get("execution_paths") != list(ARMS)
        or config.get("tolerances") != TOLERANCES
        or config.get("runtime_versions") != VERSIONS
    ):
        raise ValueError("Final closure configuration changes the fixed protocol")
    for key, digest in PINNED_INPUTS.items():
        if config["inputs"][key]["sha256"] != digest:
            raise ValueError(f"Final closure changed the pinned {key}")
    models = config["models"]
    if [m["seed"] for m in models] != [0, 1, 2] or [m["sha256"] for m in models] != list(
        MODEL_HASHES
    ):
        raise ValueError("Final closure requires all three predetermined final models")
    return config


def require_committed(root, paths):
    """Require current tracked bytes equal HEAD; ignore unrelated working changes."""
    import subprocess

    root = Path(root).resolve()
    for relative in sorted(set(paths)):
        path = root / relative
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root)
        if not path.is_file() or path.read_bytes() != committed:
            raise ValueError(f"Study dependency must be committed before measurement: {relative}")


def authenticate(config, root):
    """Authenticate manifests, caches and predictions without inference/export."""
    root = Path(root).resolve()
    paths = {}
    for name, record in config["inputs"].items():
        if name == "development_index":
            # This historical development cache is shared through a symlink.
            # Its frozen loader authenticates all arrays against the resolved
            # cache directory. Assessment/model containment remains unchanged.
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Development index must have a repository-relative path")
            path = root / relative
            if sha256(path) != record["sha256"]:
                raise ValueError("Development index identity changed")
            paths[name] = path
        else:
            paths[name] = artifact(root, record)
    cache = root / config["cache"]
    freeze = read_json(paths["freeze"])
    index = load_assessment_index(
        paths["assessment_index"],
        paths["freeze"],
        config["inputs"]["freeze"]["sha256"],
        root=root,
        cache=cache,
    )
    development = checked_development_index(paths["development_index"])
    if len(development["files"]) != 11:
        raise ValueError("Timing workload must retain all 11 development files")
    if (
        len(index["files"]),
        sum(r["n_bins"] for r in index["files"]),
        sum(r["scored_bins"] for r in index["files"]),
    ) != (14, 34490, 33613):
        raise ValueError("Later assessment sample inventory changed")
    historical = read_json(paths["historical_manifest"])
    summary = read_json(paths["historical_summary"])
    public = read_json(paths["public_final"])
    if (
        historical.get("status") != "complete"
        or historical["freeze_sha256"] != config["inputs"]["freeze"]["sha256"]
        or historical["index_sha256"] != config["inputs"]["assessment_index"]["sha256"]
        or historical["summary_sha256"] != sha256(paths["historical_summary"])
        or summary != public
        or summary["models"] != historical["models"]
        or [entry["model"] for entry in historical["models"]] != freeze["models"]
    ):
        raise ValueError("Historical final assessment identity is inconsistent")
    historical_root = paths["historical_manifest"].parent
    source = historical["source"]
    if sha256(historical_root / "source.diff") != source["diff_sha256"]:
        raise ValueError("Historical source diff changed")
    for relative, digest in source["files"].items():
        if sha256(historical_root / "source" / relative) != digest:
            raise ValueError(f"Historical source snapshot changed: {relative}")
    for relative, digest in freeze["source"]["files_sha256"].items():
        if source["files"].get(relative) != digest:
            raise ValueError("Historical evaluation source differs from the freeze")
    entries = {entry["model"]["name"]: entry for entry in historical["models"]}
    expected_files = {r["file_id"] for r in index["files"]}
    models = []
    for requested in config["models"]:
        model_path = artifact(root, requested)
        entry = entries[requested["name"]]
        if any(entry["model"].get(k) != requested[k] for k in ("name", "path", "sha256")):
            raise ValueError("Requested final model differs from historical assessment")
        if entry["model"]["kind"] != "tcn" or set(entry["prediction_sha256"]) != expected_files:
            raise ValueError("Historical model kind or full-trace inventory changed")
        if read_json(historical_root / requested["name"] / "scores.json") != entry["scores"]:
            raise ValueError("Historical standalone score differs from manifest")
        predictions = {}
        for file_id, digest in entry["prediction_sha256"].items():
            record = {
                "path": str(
                    (historical_root / requested["name"] / f"{file_id}.npy").relative_to(root)
                ),
                "sha256": digest,
            }
            artifact(root, record)
            predictions[file_id] = record
        models.append(
            {
                "seed": requested["seed"],
                "model": entry["model"],
                "model_bytes": model_path.stat().st_size,
                "historical_predictions": predictions,
                "historical_scores": entry["scores"],
            }
        )
    return {
        "inputs": config["inputs"],
        "resolved_input_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "cache": config["cache"],
        "models": models,
        "assessment_files": index["files"],
        "development_files": development["files"],
        "exposure": (
            "S6-S12 previously assessed; fixed conversion validation, not new generalization"
        ),
        "byte_integrity_only": True,
    }


def whole_file(frozen, counts):
    """Frozen causal convolution over one full feature file in original units."""
    a = frozen.arrays
    x = (np.asarray(counts, dtype=np.float32) - a["input_mean"]) / a["input_scale"]
    with torch.inference_mode():
        normalized = frozen.to_torch()(torch.from_numpy(x)[None])[0].numpy()
    return normalized * a["target_scale"] + a["target_mean"]


def make_decoder(arm, frozen, folder):
    if arm in ("A", "B"):
        return TCNStream(frozen, implementation="window" if arm == "A" else "incremental")
    if arm not in GRAPH_NAMES:
        raise ValueError("Unknown execution arm")
    return ONNXTCNStream(
        Path(folder) / GRAPH_NAMES[arm],
        expected_model_identity=model_fingerprint(frozen),
        precision="dynamic_int8" if arm == "E" else "float32",
    )


def trace_check(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if (
        actual.shape != reference.shape
        or actual.dtype != np.float32
        or not np.isfinite(actual).all()
    ):
        raise ValueError("Malformed or nonfinite full-file prediction")
    return {
        "allclose": bool(
            np.allclose(actual, reference, rtol=TOLERANCES["rtol"], atol=TOLERANCES["atol"])
        ),
        "max_absolute_error": float(np.max(np.abs(actual - reference))),
        "num_bins": len(actual),
    }


def score_predictions(root, inventory, predictions):
    """Only this post-save scoring phase opens target and mask payloads."""
    base = Path(root) / inventory["inputs"]["assessment_index"]["path"]
    records = []
    for row in inventory["assessment_files"]:
        target = load_target_file(base.parent / row["targets"], row["targets_sha256"])
        prediction = np.load(
            artifact(Path(root).resolve(), predictions[row["file_id"]]), allow_pickle=False
        )
        records.append(
            FileEvaluation(
                row["file_id"], row["day"], prediction, target["targets"], target["eval_mask"]
            )
        )
    return score_files(records)


def score_deltas(scores, reference):
    if set(scores["per_day"]) != set(reference["per_day"]):
        raise ValueError("Score day inventories differ")
    return {
        day: {
            "r2_variance_weighted": values["r2_variance_weighted"]
            - reference["per_day"][day]["r2_variance_weighted"],
            "r2_per_output": (
                np.array(values["r2_per_output"])
                - np.array(reference["per_day"][day]["r2_per_output"])
            ).tolist(),
        }
        for day, values in scores["per_day"].items()
    }


def quality_gate(arm, checks, deltas):
    if not checks or not deltas:
        raise ValueError("Cannot qualify incomplete quality evidence")
    trace_gate = None if arm == "E" else all(c["allclose"] for c in checks.values())
    differences = [d["r2_variance_weighted"] for d in deltas.values()]
    score_gate = (
        min(differences) >= TOLERANCES["int8_day_r2_min_delta"]
        if arm == "E"
        else max(abs(d) for d in differences) <= TOLERANCES["day_r2_atol"]
    )
    return {
        "trace_gate": trace_gate,
        "score_gate": bool(score_gate),
        "qualified": bool(score_gate and trace_gate is not False),
    }


def _save_prediction(path, prediction, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, prediction, allow_pickle=False)
    return {"path": str(path.relative_to(output)), "sha256": sha256(path)}


def _root_predictions(output, predictions, root):
    return {
        key: {"path": str((output / record["path"]).relative_to(root)), "sha256": record["sha256"]}
        for key, record in predictions.items()
    }


def _graph_record(path, output, identity, precision):
    import onnx

    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    metadata = {item.key: item.value for item in graph.metadata_props}
    ops = [n.op_type for n in graph.graph.node]
    if metadata.get("precision") != precision or metadata.get("frozen_fingerprint") != identity:
        raise ValueError("Graph precision or model fingerprint changed")
    if [op.version for op in graph.opset_import if op.domain in ("", "ai.onnx")] != [17]:
        raise ValueError("Graph no longer uses opset 17")
    expected_op = "MatMulInteger" if precision == "dynamic_int8" else "MatMul"
    if ops.count(expected_op) != 4:
        raise ValueError("Graph requires exactly four matrix operators at declared precision")
    return {
        "path": str(path.relative_to(output)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "operations": {op: ops.count(op) for op in sorted(set(ops))},
        "metadata": metadata,
    }


def _quality(output, config, root, manifest):
    import onnx
    import onnxruntime

    from neural_experiments.compression import convert

    if {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__} != VERSIONS:
        raise ValueError("Conversion runtime versions differ from the prespecified versions")
    inventory = authenticate(config, root)
    write_json(output / "inventory.json", inventory)
    manifest["inventory_sha256"] = sha256(output / "inventory.json")
    require_committed(root, [*manifest["source"]["files"], manifest["config"]["path"]])
    torch.set_num_threads(1)
    index_base = root / config["inputs"]["assessment_index"]["path"]
    models = []
    # Complete all historical reconstruction checks before any graph is exported.
    for item in inventory["models"]:
        seed, model = item["seed"], item["model"]
        frozen = load_tcn(artifact(root, model))
        destination = output / f"seed-{seed}"
        destination.mkdir()
        reference = {"predictions": {}, "historical_trace": {}}
        historical_predictions = {}
        for row in inventory["assessment_files"]:
            fid = row["file_id"]
            features = load_feature_file(
                index_base.parent / row["features"], row["features_sha256"]
            )
            actual = whole_file(frozen, features["counts"])
            reference["predictions"][fid] = _save_prediction(
                destination / "reference" / f"{fid}.npy", actual, output
            )
            historical_path = artifact(root, item["historical_predictions"][fid])
            historical = np.load(historical_path, allow_pickle=False)
            reference["historical_trace"][fid] = trace_check(actual, historical)
            historical_predictions[fid] = item["historical_predictions"][fid]
        reference["scores"] = score_predictions(
            root, inventory, _root_predictions(output, reference["predictions"], root)
        )
        historical_scores = score_predictions(root, inventory, historical_predictions)
        if historical_scores != item["historical_scores"]:
            raise ValueError(f"Historical score reconstruction changed for seed {seed}")
        reference["historical_scores_match"] = True
        reference["historical_delta_per_day"] = score_deltas(reference["scores"], historical_scores)
        reference["historical_gate"] = quality_gate(
            "A", reference["historical_trace"], reference["historical_delta_per_day"]
        )
        write_json(destination / "reference.json", reference)
        if not reference["historical_gate"]["qualified"]:
            raise ValueError(f"Whole-file historical reconstruction failed for seed {seed}")
        models.append(
            {
                "seed": seed,
                "model": model,
                "fingerprint": model_fingerprint(frozen),
                "parameter_count": frozen.parameter_count,
                "reference": reference,
                "arms": {},
            }
        )
        print(f"Reconstructed historical full traces: seed {seed}", flush=True)
    for entry in models:
        seed, folder = entry["seed"], output / f"seed-{entry['seed']}"
        frozen = load_tcn(artifact(root, entry["model"]))
        started = time.perf_counter_ns()
        conversion_error = None
        try:
            convert(frozen, folder)
        except Exception as error:
            conversion_error = f"{type(error).__name__}: {error}"
        entry["export_wall_seconds"] = (time.perf_counter_ns() - started) / 1e9
        entry["conversion_error"] = conversion_error
        for arm in ARMS:
            result = {
                "status": "running",
                "precision": "dynamic_int8" if arm == "E" else "float32",
                "graph": None,
                "predictions": {},
                "trace_checks": {},
            }
            entry["arms"][arm] = result
            try:
                if arm in GRAPH_NAMES:
                    result["graph"] = _graph_record(
                        folder / GRAPH_NAMES[arm], output, entry["fingerprint"], result["precision"]
                    )
                for row in inventory["assessment_files"]:
                    fid = row["file_id"]
                    features = load_feature_file(
                        index_base.parent / row["features"], row["features_sha256"]
                    )
                    decoder = make_decoder(arm, frozen, folder)
                    result["numeric_state_bytes"] = decoder.state_nbytes
                    prediction = decoder.predict_chunk(features["counts"])
                    result["predictions"][fid] = _save_prediction(
                        folder / arm / f"{fid}.npy", prediction, output
                    )
                    reference = np.load(
                        output / entry["reference"]["predictions"][fid]["path"], allow_pickle=False
                    )
                    result["trace_checks"][fid] = trace_check(prediction, reference)
                result["scores"] = score_predictions(
                    root, inventory, _root_predictions(output, result["predictions"], root)
                )
                result["delta_per_day"] = score_deltas(
                    result["scores"], entry["reference"]["scores"]
                )
                result.update(quality_gate(arm, result["trace_checks"], result["delta_per_day"]))
                result["status"] = "complete"
            except Exception as error:
                result.update(
                    status="failed",
                    qualified=False,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            write_json(folder / f"{arm}-result.json", result)
            print(
                f"Completed seed {seed}, path {arm}: {result['status']}, "
                f"qualified={result['qualified']}",
                flush=True,
            )
    gates = {
        "historical_reference": True,
        "float32_equivalence": {
            arm: all(m["arms"][arm]["qualified"] for m in models) for arm in ARMS[:-1]
        },
        "compression_quality": all(m["arms"]["E"]["qualified"] for m in models),
    }
    rows = [
        {"seed": m["seed"], "arm": arm, "day": day, **delta}
        for m in models
        for arm, result in m["arms"].items()
        for day, delta in result.get("delta_per_day", {}).items()
    ]
    int8 = [row["r2_variance_weighted"] for row in rows if row["arm"] == "E"]
    return {
        "schema_version": 1,
        "models": models,
        "gates": gates,
        "seed_day_rows": rows,
        "mean_delta_r2": float(np.mean(int8)) if int8 else None,
        "worst_seed_day_delta": min(int8) if int8 else None,
        "completed_seed_day_cells": len(int8),
        "timing_seed": 0,
        "exposure": inventory["exposure"],
    }


def run(command, config_path, output, *, root):
    """Retain start, final, source and failure evidence even for preflight errors."""
    root, output, config_path = (
        Path(root).resolve(),
        Path(output).resolve(),
        Path(config_path).resolve(),
    )
    if not output.is_relative_to(root / "results/neural-h1"):
        raise ValueError("Private evidence must remain under results/neural-h1")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "kind": f"h1_final_closure_{command}",
        "status": "running",
        "run": str(output),
        "started_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "statuses": {
            "implementation_execution": "running",
            "artifact_completion": "incomplete",
            "independent_verification": "not_run",
            "float32_equivalence": "not_run",
            "compression_quality": "not_run",
            "runtime_goals": "not_run",
        },
    }
    write_json(output / "start.json", manifest)
    write_json(output / "manifest.json", manifest)
    try:
        manifest.update(
            config={"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
            environment=environment(),
            source=source_identity(root, output),
        )
        (output / "configuration.json").write_bytes(config_path.read_bytes())
        config = load_config(config_path)
        manifest.update(
            configuration=config,
            inputs=config["inputs"],
        )
        with threadpool_limits(limits=1):
            summary = (
                authenticate(config, root)
                if command == "inventory"
                else _quality(output, config, root, manifest)
            )
        write_json(output / "summary.json", summary)
        manifest.update(summary_sha256=sha256(output / "summary.json"), status="complete")
        manifest["statuses"].update(
            implementation_execution="complete", artifact_completion="complete"
        )
        if command == "quality":
            execution_complete = all(
                a["status"] == "complete" for m in summary["models"] for a in m["arms"].values()
            )
            manifest["statuses"].update(
                implementation_execution="complete" if execution_complete else "partial_failure",
                float32_equivalence="passed"
                if all(summary["gates"]["float32_equivalence"].values())
                else "failed",
                compression_quality="passed"
                if summary["gates"]["compression_quality"]
                else "failed",
            )
        manifest["artifacts"] = {
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
    except BaseException as error:
        manifest.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        manifest["statuses"].update(
            implementation_execution="failed", artifact_completion="incomplete"
        )
        raise
    finally:
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "quality"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.command, args.config, args.output, root=args.root)


if __name__ == "__main__":
    main()
