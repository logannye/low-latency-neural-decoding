"""Locked diagnostic of whether recent labels restore future T2 decoding.

This is exposed-data method development, not a fresh assessment. Every model
predicts the same disjoint future windows after a two-second training embargo.
No claim of independent trials is made; uncertainty resamples whole sessions.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_experiments.v3.study import digest, mask_digest, relative, score, write_json
from neural_experiments.v4 import model as core
from neural_experiments.v7 import model

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA256 = "28ead298081c41b6028bffa0cf9aa5ee0eeb7811917400863aff350525aa7671"
UPDATES_SECONDS = (272, 332, 392, 452, 512, 572)
METHODS = ("zero", *model.METHODS)
ELIGIBLE = ("rolling60", "rolling120", "rolling240", "exponential60", "rolling120_local")


def metric(y, prediction, mask):
    values = score(y, prediction, mask)
    values["summed_mse"] = (
        float(np.square(y[mask] - prediction[mask]).sum(1).mean())
        if int(mask.sum()) >= 50
        else None
    )
    return values


def evaluate_session(case, *, save_artifact=None):
    """Run seven prespecified methods and zero on one exposed session.

    The optional artifact callback is invoked after fitting and receives the
    exact model, window prediction and common mask for independent auditing.
    """
    item, features = case["item"], case["features"]
    segments, y, mask = case["segments"], case["y"], case["eval_mask"]
    raw = model.representation(features, segments, model.CONFIG)
    n = len(raw)
    if len(y) != n or n > 30000:
        raise ValueError("aligned exposure prefix of at most 600 seconds required")
    use = core.training_mask(mask, segments)
    use[:13600] = False
    pooled_use = use.copy()
    for seconds in UPDATES_SECONDS:
        start, stop = 50 * seconds, min(n, 50 * (seconds + 60), 30000)
        if start < stop and use[start:stop].sum() < 50:
            pooled_use[start:stop] = False
    rows, window_rows = [], []
    for method in METHODS:
        prediction = np.zeros((n, 2), dtype=np.float64)
        windows = []
        fits = []
        label_intervals = []
        for seconds in UPDATES_SECONDS:
            start, stop = 50 * seconds, min(n, 50 * (seconds + 60), 30000)
            if start >= n:
                windows.append(
                    {
                        "start_bin": start,
                        "stop_bin": start,
                        "disposition": "recording_ended_before_forecast_window",
                        "metrics": None,
                        "valid_rows": 0,
                        "score_mask_sha256": mask_digest(np.zeros(0, bool)),
                        "fit": None,
                    }
                )
                continue
            fitted, receipt = None, None
            if method != "zero":
                fit_stop = 7500 if method == "frozen120" else start - model.GAP_BINS
                fitted, receipt = model.fit(
                    raw,
                    y[:fit_stop],
                    mask[:fit_stop],
                    segments,
                    item["n_channels"],
                    method,
                    start,
                )
                prediction[start:stop] = model.predict_raw(fitted, raw[start:stop])
                fits.append(receipt)
                label_intervals.append((receipt["label_start_bin"], receipt["label_stop_bin"]))
            window = {
                "start_bin": start,
                "stop_bin": stop,
                "disposition": "scored" if use[start:stop].sum() >= 50 else "insufficient_labels",
                "fit": receipt,
                "metrics": metric(y[start:stop], prediction[start:stop], use[start:stop]),
                "score_mask_sha256": mask_digest(use[start:stop]),
                "valid_rows": int(use[start:stop].sum()),
            }
            if save_artifact is not None:
                window.update(
                    save_artifact(
                        method, start, stop, fitted, prediction[start:stop], use[start:stop]
                    )
                )
            windows.append(window)
        labels_requested = np.zeros(n, bool)
        for a, b in label_intervals:
            labels_requested[a:b] = True
        label_seconds = float(labels_requested.sum() * model.DT)
        identity = {
            "participant": item["participant_id"],
            "session_id": item["session_id"],
            "post_implant_day": item["post_implant_day"],
            "method": method,
        }
        for window in windows:
            window_rows.append({**identity, **window})
        rows.append(
            {
                "participant": item["participant_id"],
                "session_id": item["session_id"],
                "post_implant_day": item["post_implant_day"],
                "method": method,
                "n_bins": n,
                "metrics": metric(y, prediction, pooled_use),
                "score_mask_sha256": mask_digest(pooled_use),
                "unique_requested_label_seconds": label_seconds,
            }
        )
    return rows, window_rows


def run(index_path, output, plan_path, plan_sha256):
    from neural_experiments.v7 import data, metrics

    if plan_sha256 != PLAN_SHA256 or digest(plan_path) != plan_sha256:
        raise ValueError("locked recoverability plan identity mismatch")
    index = data.load_index(index_path, expected_role="development", verify_hashes=True)
    plan = json.loads(Path(plan_path).read_text())
    if (
        len(index["sessions"]) != plan["scope"]["sessions"]
        or index["parents"]["development"]["sha256"] != plan["parents"]["v6_development_sha256"]
        or index["parents"]["new_sessions"]["sha256"] != plan["parents"]["v6_new_sessions_sha256"]
    ):
        raise ValueError("recoverability development cohort differs from locked plan")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "traces").mkdir()
    source_paths = tuple(
        f"neural_experiments/{version}/{name}.py"
        for version, names in (
            ("v7", ("model", "recoverability", "data", "metrics")),
            ("v6", ("data",)),
            ("v5", ("data",)),
            ("v4", ("model", "data")),
            ("v3", ("study", "model", "data")),
            ("v2", ("data",)),
        )
        for name in names
    )
    for source in source_paths:
        destination = output / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    manifest = {
        "schema_version": 1,
        "phase": "exposed_recoverability",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "threadpool_limit": 1,
        },
        "plan": {"path": relative(plan_path), "sha256": plan_sha256},
        "index": {"path": relative(index_path), "sha256": digest(index_path)},
        "code_sha256": {path: digest(ROOT / path) for path in source_paths},
        "rows": [],
        "window_rows": [],
    }
    write_json(output / "manifest.json", manifest)
    with threadpool_limits(limits=1):
        for item in index["sessions"]:
            case = data.load_case(index_path, item, labels=True)
            raw = model.representation(case["features"], case["segments"], model.CONFIG)

            def save_artifact(method, start, stop, fitted, prediction, mask):
                key = f"{item['session_id']}-{method}-{start}"
                model_path = output / "models" / f"{key}.npz"
                trace_path = output / "traces" / f"{key}.npz"
                if fitted is not None:
                    fitted.save(model_path)
                    exported = model.Decoder.load(model_path)
                    np.testing.assert_array_equal(exported.weights, fitted.weights)
                    np.testing.assert_array_equal(exported.mean, fitted.mean)
                    np.testing.assert_array_equal(exported.scale, fitted.scale)
                    if exported.config != fitted.config:
                        raise ValueError("exported decoder configuration differs")
                    np.testing.assert_array_equal(
                        model.predict_raw(exported, raw[start:stop]), prediction
                    )
                np.savez_compressed(
                    trace_path,
                    prediction=prediction,
                    eval_mask=mask,
                    start_bin=np.asarray(start, np.int64),
                    stop_bin=np.asarray(stop, np.int64),
                )
                return {
                    "model_path": relative(model_path) if fitted is not None else None,
                    "model_sha256": digest(model_path) if fitted is not None else None,
                    "trace_path": relative(trace_path),
                    "trace_sha256": digest(trace_path),
                }

            rows, window_rows = evaluate_session(case, save_artifact=save_artifact)
            manifest["rows"].extend(rows)
            manifest["window_rows"].extend(window_rows)
            write_json(output / "manifest.json", manifest)
            print(json.dumps({"recoverability_session": item["session_id"]}), flush=True)
    manifest["summary"] = metrics.summarize(manifest["rows"], manifest["window_rows"])
    if any(digest(ROOT / path) != expected for path, expected in manifest["code_sha256"].items()):
        raise ValueError("scientific source changed during recoverability execution")
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    result = run(args.index, args.output, args.plan, args.plan_sha256)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
