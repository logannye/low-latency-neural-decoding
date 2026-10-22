"""Post-hoc sparse-count scaling pilot; capped exposed T3 prepared files only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_experiments.v3 import model
from neural_experiments.v4.diagnosis import EXPOSED_INDEXES, metric, read_exposed_index, sha

CONFIGS = [
    {"name": "reference", "floor": 0.1, "penalty": 1.0, "bias": 1.0},
    {"name": "floor01_lambda01", "floor": 0.01, "penalty": 0.1, "bias": 1.0},
    {"name": "floor01_lambda1", "floor": 0.01, "penalty": 1.0, "bias": 1.0},
    {"name": "floor001_lambda01", "floor": 0.001, "penalty": 0.1, "bias": 1.0},
    {"name": "floor001_lambda1", "floor": 0.001, "penalty": 1.0, "bias": 1.0},
    {"name": "floor01_lambda1_bias025", "floor": 0.01, "penalty": 1.0, "bias": 0.25},
    {"name": "floor001_lambda1_bias0", "floor": 0.001, "penalty": 1.0, "bias": 0.0},
]


def run(paths, field, output, development_index, assessment_index):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    allowed = {}
    original_indexes = []
    for role in ("development", "assessment"):
        p = development_index if role == "development" else assessment_index
        d = read_exposed_index(p, f"v3_{role}")
        original_indexes.append(sha(p))
        for item in d["sessions"]:
            if item["participant_id"] == "T3":
                allowed[item["session_id"]] = role
    result = {
        "status": "post_hoc_exposed_only",
        "feature_field": field,
        "tau_seconds": 0.4,
        "budgets_seconds": [30, 120],
        "scoring_seconds": [302, 600],
        "configs": CONFIGS,
        "frozen_v3_model_sha256": sha(model.__file__),
        "metric_helper_sha256": sha(Path(__file__).with_name("diagnosis.py")),
        "indexes": [],
        "inputs": [],
        "rows": [],
    }
    seen = set()
    for path in paths:
        role = next((role for role, digest in EXPOSED_INDEXES.items() if sha(path) == digest), None)
        d = read_exposed_index(path, role)
        assert sha(path) in original_indexes or d["role"] == "development"
        result["indexes"].append({"sha256": sha(path), "role": d["role"]})
        for item in d["sessions"]:
            if item["participant_id"] != "T3":
                continue
            sid = item["session_id"]
            assert sid in allowed and sid not in seen and item["n_bins"] == 30000
            seen.add(sid)
            fp, tp = path.parent / item["feature_path"], path.parent / item["target_path"]
            assert sha(fp) == item["feature_sha256"] and sha(tp) == item["target_sha256"]
            with np.load(fp, allow_pickle=False) as z:
                tx, q = z[field].astype(float), z["sequence_id"]
            with np.load(tp, allow_pickle=False) as z:
                y, mask = z["y"], z["eval_mask"]
            assert len(tx) == len(y) == len(q) == 30000
            bounds = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)]
            segments = list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
            raw = model.exponential(np.sqrt(tx), segments, 0.4)
            result["inputs"].append(
                {
                    "session_id": sid,
                    "formerly": allowed[sid],
                    "feature_sha256": sha(fp),
                    "target_sha256": sha(tp),
                    "n_bins": len(tx),
                    "count_nonzero_fraction": float((tx > 0).mean()),
                    "mean_rate_hz": float(tx.mean() / 0.02),
                }
            )
            for budget in [30, 120]:
                n = budget * 50
                pre_segments = [(a, min(b, n)) for a, b in segments if a < n]
                train = model.training_mask(mask[:n], pre_segments)
                mean, std = raw[:n].mean(0), raw[:n].std(0)
                for cfg in CONFIGS:
                    scale = np.maximum(std, cfg["floor"])
                    z = np.clip((raw - mean) / scale, -8, 8)
                    x, yy = z[:n][train], y[:n][train].astype(float)
                    xm, ym = x.mean(0), yy.mean(0)
                    xc, yc = x - xm, yy - ym
                    gram = xc.T @ xc / len(x)
                    gram.flat[:: len(gram) + 1] += cfg["penalty"]
                    coef = np.linalg.solve(gram, xc.T @ yc / len(x))
                    intercept = (ym - xm @ coef) * cfg["bias"]
                    pred = model.project_direction(z @ coef + intercept)
                    scored = mask & (np.arange(len(mask)) >= 15100)
                    result["rows"].append(
                        {
                            "session_id": sid,
                            "formerly": allowed[sid],
                            "budget_seconds": budget,
                            "config": cfg["name"],
                            "fraction_channels_below_floor": float((std < cfg["floor"]).mean()),
                            "training": metric(y[:n], pred[:n], train),
                            "common_suffix": metric(y, pred, scored),
                        }
                    )
            print(sid, "done", flush=True)
    assert seen == set(allowed)
    result["summary"] = []
    for cohort in ["all_exposed", "development", "assessment"]:
        for budget in [30, 120]:
            for cfg in CONFIGS:
                rows = [
                    r
                    for r in result["rows"]
                    if r["budget_seconds"] == budget
                    and r["config"] == cfg["name"]
                    and (cohort == "all_exposed" or r["formerly"] == cohort)
                ]
                result["summary"].append(
                    {
                        "cohort": cohort,
                        "budget_seconds": budget,
                        "config": cfg["name"],
                        "n_sessions": len(rows),
                        "mean_training_r2": float(np.mean([r["training"]["r2"] for r in rows])),
                        "mean_suffix_r2": float(np.mean([r["common_suffix"]["r2"] for r in rows])),
                        "mean_suffix_angular_degrees": float(
                            np.mean([r["common_suffix"]["angular_degrees"] for r in rows])
                        ),
                    }
                )
    result.update(source_sha256=sha(__file__), elapsed_seconds=time.monotonic() - started)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output, result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--indices", type=Path, nargs="+", required=True)
    p.add_argument("--field", choices=["x_tx", "x_tx3", "x_tx45"], default="x_tx")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--development-index", type=Path, required=True)
    p.add_argument("--assessment-index", type=Path, required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        run(args.indices, args.field, args.output, args.development_index, args.assessment_index)
