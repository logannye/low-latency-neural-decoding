"""Algebraic error decomposition of saved predictions; no decoder fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_experiments.v3.study import digest, write_json


def decompose(y, prediction, mask):
    y, p = np.asarray(y, float)[mask], np.asarray(prediction, float)[mask]
    if len(y) < 50:
        return None
    residual = p - y
    yc, pc = y - y.mean(0), p - p.mean(0)
    variance = float(np.square(yc).sum(1).mean())
    covariance = float((yc * pc).sum(1).mean())
    p_variance = float(np.square(pc).sum(1).mean())
    mse = float(np.square(residual).sum(1).mean())
    mean_error = float(np.square(residual.mean(0)).sum())
    result = {
        "rows": len(y),
        "summed_mse": mse,
        "target_squared_norm": float(np.square(y).sum(1).mean()),
        "prediction_squared_norm": float(np.square(p).sum(1).mean()),
        "target_prediction_dot": float((y * p).sum(1).mean()),
        "mean_residual_squared_norm": mean_error,
        "centered_residual_error": float(np.square(residual - residual.mean(0)).sum(1).mean()),
        "target_mean_norm": float(np.linalg.norm(y.mean(0))),
        "target_total_variance": variance,
        "centered_prediction_target_correlation": covariance / np.sqrt(variance * p_variance)
        if variance * p_variance > 0
        else None,
        "r2": 1 - mse / variance if variance > 1e-12 else None,
    }
    reconstructed = (
        result["target_squared_norm"]
        + result["prediction_squared_norm"]
        - 2 * result["target_prediction_dot"]
    )
    if not np.isclose(reconstructed, mse, rtol=0, atol=1e-12) or not np.isclose(
        mean_error + result["centered_residual_error"], mse, rtol=0, atol=1e-12
    ):
        raise AssertionError("error identities failed")
    return result


def run(manifest_path, output):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("study") != "V8":
        raise ValueError("complete V8 prediction manifest required")
    result = {
        "study": "V8",
        "manifest_sha256": digest(manifest_path),
        "code_sha256": digest(__file__),
        "phase": manifest["phase"],
        "rows": [],
    }
    for row in manifest["rows"]:
        path = Path(row["trace"]["trace_path"])
        if digest(path) != row["trace"]["trace_sha256"]:
            raise ValueError("prediction trace identity changed")
        with np.load(path, allow_pickle=False) as trace:
            prediction = trace[row["method"] + "__prediction"]
            value = {k: row[k] for k in ("participant", "session_id", "method")}
            value["primary"] = decompose(trace["y"], prediction, trace["primary_mask"])
            value["secondary_instantaneous"] = decompose(
                trace["y"], prediction, trace["secondary_mask"]
            )
            value["secondary_onset"] = decompose(
                trace["onset_y"], prediction, trace["secondary_mask"]
            )
            result["rows"].append(value)
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.output)
    print(json.dumps({"rows": len(result["rows"])}))
