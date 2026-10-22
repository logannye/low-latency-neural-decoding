"""Fold one retained H1 prefix calibration into a numeric model; never fit labels.

Full-file replay of the frozen artifact is runtime-only. Scientific quality
evaluation begins after the original 30-second calibration prefix.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from neural_decoder.artifacts import sha256, write_json

SPEC = importlib.util.spec_from_file_location(
    "h1_v2_accuracy_for_composition", Path(__file__).with_name("accuracy.py")
)
accuracy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accuracy)


def compose_head(model, transforms, latent_mean, latent_scale, coefficient, intercept):
    """Copy the model and fold an already-fit source-unit residual into its head."""
    width, outputs = model.width, model.output_channels
    values = [
        np.asarray(value, dtype=np.float64)
        for value in (latent_mean, latent_scale, coefficient, intercept)
    ]
    shapes = [(width,), (width,), (width, outputs), (outputs,)]
    if any(
        value.shape != shape or not np.isfinite(value).all()
        for value, shape in zip(values, shapes, strict=True)
    ):
        raise ValueError("invalid retained calibration arrays")
    mean, scale, coefficients, intercept = values
    target_scale = np.asarray(transforms["target_scale"], dtype=np.float64)
    if (scale <= 0).any() or (target_scale <= 0).any():
        raise ValueError("positive latent and target scales required")
    result = copy.deepcopy(model).eval()
    weight = model.head.weight.detach().numpy().astype(np.float64)
    bias = model.head.bias.detach().numpy().astype(np.float64)
    weight += (coefficients / scale[:, None] / target_scale[None, :]).T
    bias += (intercept - (mean / scale) @ coefficients) / target_scale
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("nonfinite composed head")
    with torch.no_grad():
        result.head.weight.copy_(torch.from_numpy(weight.astype(np.float32)))
        result.head.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    return result


def run(run_path, index_path, output):
    run_path, index_path, output = map(Path, (run_path, index_path, output))
    summary = json.loads((run_path / "summary.json").read_text())
    manifest = json.loads((run_path / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or sha256(run_path / "summary.json") != manifest["summary_sha256"]
    ):
        raise ValueError("completed parent run with matching summary required")
    if sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("parent input index changed")
    index = json.loads(index_path.read_text())
    item = sorted((r for r in index["files"] if r["day"] == "S4"), key=lambda r: r["file_id"])[0]
    file_id = item["file_id"]
    parent = next(
        r
        for r in summary["main"]
        if r["arm"] == "rf15_standardized" and r["seed"] == 0 and r["day"] == "S4"
    )
    calibration = next(
        r
        for r in summary["calibration"]
        if r["arm"] == parent["arm"]
        and r["seed"] == 0
        and r["day"] == "S4"
        and r["method"] == "ridge_delta_head"
        and r["budget_seconds"] == 30
    )
    prediction_evidence = calibration["predictions"][file_id]
    parent_path = run_path / parent["model"]["path"]
    prediction_path = run_path / prediction_evidence["path"]
    coefficient_path = prediction_path.with_suffix(".calibration.npz")
    latent_path = parent_path.parent / "calibration/latent-normalizer.npz"
    inventory = json.loads((run_path / "artifact-hashes.json").read_text())
    if sha256(run_path / "artifact-hashes.json") != manifest["artifacts_sha256"]:
        raise ValueError("parent artifact inventory changed")
    inputs = {
        "parent_model": parent_path,
        "calibration": coefficient_path,
        "latent_normalizer": latent_path,
        "retained_prediction": prediction_path,
    }
    for path in inputs.values():
        if sha256(path) != inventory[str(path.relative_to(run_path))]:
            raise ValueError("retained parent artifact changed")
    model, transforms, _ = accuracy.load_model(parent_path)
    with np.load(coefficient_path, allow_pickle=False) as data:
        coefficient, intercept = data["coefficient"], data["intercept"]
    with np.load(latent_path, allow_pickle=False) as data:
        latent_mean, latent_scale = data["latent_mean"], data["latent_scale"]
    composed = compose_head(model, transforms, latent_mean, latent_scale, coefficient, intercept)
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "model.npz"
    accuracy.save_model(model_path, composed, transforms)
    # The artifact is saved before any target payload is opened. Labels below
    # serve only independent post-save scoring, never composition or selection.
    features_path = index_path.parent / item["features"]
    targets_path = index_path.parent / item["targets"]
    if (
        sha256(features_path) != item["features_sha256"]
        or sha256(targets_path) != item["targets_sha256"]
    ):
        raise ValueError("canonical evaluation input changed")
    with np.load(features_path, allow_pickle=False) as data:
        counts = data["counts"]
        np.testing.assert_array_equal(data["sequence_id"], np.arange(len(counts)))
    loaded, affine, _ = accuracy.load_model(model_path)
    prediction = accuracy.predict(loaded, affine, counts)
    retained = np.load(prediction_path, allow_pickle=False)
    boundary = calibration["common_suffix_start_bin"]
    np.testing.assert_allclose(prediction[boundary:], retained[boundary:], atol=1e-7, rtol=3e-5)
    np.save(output / "full-runtime-replay.npy", prediction, allow_pickle=False)
    with np.load(targets_path, allow_pickle=False) as data:
        target = data["targets"].astype(np.float64)
        suffix = data["eval_mask"] & (np.arange(len(target)) >= boundary)
    source_score = r2_score(target[suffix], retained[suffix], multioutput="variance_weighted")
    composed_score = r2_score(
        target[suffix], prediction[suffix].astype(np.float64), multioutput="variance_weighted"
    )
    if abs(composed_score - source_score) > 1e-6:
        raise ValueError("composed score exceeds numerical tolerance")
    for before, after in zip(model.convs, loaded.convs, strict=True):
        np.testing.assert_array_equal(before.weight.detach().numpy(), after.weight.detach().numpy())
        np.testing.assert_array_equal(before.bias.detach().numpy(), after.bias.detach().numpy())
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "source_sha256": sha256(Path(__file__)),
        "accuracy_source_sha256": sha256(Path(accuracy.__file__)),
        "parent_summary_sha256": manifest["summary_sha256"],
        "parent_artifact_inventory_sha256": manifest["artifacts_sha256"],
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()
        },
        "model": {"path": str(model_path), "sha256": sha256(model_path)},
        "canonical_file_id": file_id,
        "day": "S4",
        "seed": 0,
        "arm": parent["arm"],
        "features": {"path": str(features_path), "sha256": sha256(features_path)},
        "targets_used_only_after_save_for_scoring_sha256": sha256(targets_path),
        "budget_seconds": 30,
        "prefix_scored_labels": prediction_evidence["prefix_scored_labels"],
        "common_suffix_start_bin": boundary,
        "suffix_scored_bins": int(suffix.sum()),
        "receptive_field": loaded.receptive_field,
        "parameter_count": sum(p.numel() for p in loaded.parameters()),
        "suffix_max_absolute_prediction_error": float(
            np.max(np.abs(prediction[boundary:] - retained[boundary:]))
        ),
        "retained_suffix_r2": float(source_score),
        "composed_suffix_r2": float(composed_score),
        "suffix_r2_difference": float(composed_score - source_score),
        "runtime_replay_sha256": sha256(output / "full-runtime-replay.npy"),
        "quality_scope": (
            "after first 30 nominal seconds of S4_set_1 only; "
            "full-file calibrated replay is runtime-only"
        ),
        "fit_performed": False,
        "suffix_labels_used_for_fitting_or_selection": False,
    }
    write_json(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/neural-v2/accuracy-v1"))
    parser.add_argument("--index", type=Path, default=Path("data/neural_h1/prepared/index.json"))
    parser.add_argument("--output", type=Path, default=Path("results/neural-v2/composed-head-v1"))
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        print(json.dumps(run(arguments.run, arguments.index, arguments.output), indent=2))
