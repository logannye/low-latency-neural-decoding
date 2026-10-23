"""Bounded source-only timing of a predetermined accumulating-label decoder.

The exported model uses labels through540s. Replaying it over earlier features
checks numerical parity and compute only; these are never causal efficacy scores.
The scientific V7A/V7B execution files and choices are unchanged by timing.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from neural_experiments.v4 import data as arrays
from neural_experiments.v4.runtime import (
    digest,
    filter_response,
    load_workload,
    measure,
    timestamp_summary,
    write_json,
)
from neural_experiments.v7 import model

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA256 = "4f88a6cf98dd7147691f4eaf64143784736863e1cfa4fd5fbd7b3fadd1eaea23"
SOURCE_FILES = (
    "neural_experiments/v7/runtime_continual.py",
    "neural_experiments/v7/model.py",
    "neural_experiments/v4/runtime.py",
    "neural_experiments/v4/model.py",
    "neural_experiments/v4/data.py",
    "neural_experiments/v2/data.py",
)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def checked(root, name, expected):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or digest(path) != expected:
        raise ValueError("runtime artifact path or identity mismatch")
    return path


def bind(source_path, index_path, audit_path, *, root=ROOT):
    """Latest numeric source T2 day, periodic240 threshold0.1, last complete fit."""
    root = Path(root).resolve()
    source, index, audit = [
        json.loads(Path(p).read_text()) for p in (source_path, index_path, audit_path)
    ]
    if (
        source.get("status") != "complete"
        or source.get("phase") != "source"
        or source.get("study") != "V7B"
        or source["plan"]["sha256"] != PLAN_SHA256
        or audit.get("status") != "PASS"
        or audit.get("stage") != "source"
        or audit.get("manifest_sha256") != digest(source_path)
        or audit.get("index_sha256") != digest(index_path)
        or audit.get("plan_sha256") != PLAN_SHA256
        or audit.get("code_sha256") != source["code_sha256"]
        or index.get("role") != "development"
    ):
        raise ValueError("exact completed and independently audited V7B source run required")
    checked(root, source["plan"]["path"], PLAN_SHA256)
    if (
        checked(root, source["index"]["path"], source["index"]["sha256"])
        != Path(index_path).resolve()
    ):
        raise ValueError("source index path differs")
    for path, checksum in source["code_sha256"].items():
        checked(root, path, checksum)
    candidates = [s for s in index["sessions"] if s["participant_id"] == "T2"]
    if not candidates:
        raise ValueError("predetermined source T2 workload unavailable")
    item = max(candidates, key=lambda s: (s["post_implant_day"], s["session_id"]))
    if item["post_implant_day"] != 250 or item["n_bins"] != 30000:
        raise ValueError("declared T2day250 full600s representative unavailable")
    fits = [
        e
        for e in source["events"]
        if e["session_id"] == item["session_id"]
        and e["method"] == "periodic240"
        and e["threshold"] == 0.1
        and e["event"] == "model_fitted"
    ]
    if len(fits) != 5:
        raise ValueError("all five predetermined periodic fits required")
    fit = max(fits, key=lambda e: e["event_bin"])
    if fit["event_bin"] != 27100 or fit["fit"]["fit_stop_bin"] != 27000:
        raise ValueError("last periodic fit must use labels through540s and deploy542s")
    rows = [
        r
        for r in source["rows"]
        if r["session_id"] == item["session_id"]
        and r["method"] == "periodic240"
        and r["threshold"] == 0.1
    ]
    if len(rows) != 1:
        raise ValueError("unique periodic trace required")
    paths = {
        "model": checked(root, fit["model_path"], fit["model_sha256"]),
        "trace": checked(root, rows[0]["trace_path"], rows[0]["trace_sha256"]),
        "features": checked(Path(index_path).parent, item["feature_path"], item["feature_sha256"]),
        "targets": checked(Path(index_path).parent, item["target_path"], item["target_sha256"]),
    }
    return item, fit, paths


def refit_measurements(
    raw, y, valid, purchased, segments, channels, event, expected, output, *, clock=time
):
    """Ten fits plus export/reload, with causal representations already cached."""
    stop = event["fit"]["fit_stop_bin"]
    if purchased.dtype != bool or purchased.shape != valid.shape or len(purchased) != len(raw):
        raise ValueError("exact paid-label mask geometry required")
    expected_paid = np.zeros(len(raw), bool)
    for start, end in event["fit"]["purchased_intervals"]:
        if not 1500 <= start < end <= stop:
            raise ValueError("future or invalid purchased interval")
        expected_paid[start:end] = True
    np.testing.assert_array_equal(expected_paid[:stop], purchased[:stop])
    result = []
    for repeat in range(10):
        start = clock.perf_counter_ns()
        selected = purchased[:stop] & valid[:stop]
        fitted, receipt = model.fit_mask(raw, y[:stop], selected, segments, channels, stop)
        solved = clock.perf_counter_ns()
        path = Path(output) / f"refit-{repeat}.npz"
        fitted.save(path)
        restored = model.Decoder.load(path)
        end = clock.perf_counter_ns()
        if (
            restored.config != expected.config
            or receipt["training_rows"] != event["fit"]["training_rows"]
        ):
            raise ValueError("timed refit differs from original fit geometry")
        maximum = 0.0
        for key in ("mean", "scale", "weights"):
            a, b = getattr(restored, key), getattr(expected, key)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
            maximum = max(maximum, float(np.max(np.abs(a - b))))
        result.append(
            {
                "repeat": repeat,
                "fit_ns": solved - start,
                "export_reload_ns": end - solved,
                "total_ns": end - start,
                "numeric_max_absolute_error": maximum,
                "model_path": relative(path),
                "model_sha256": digest(path),
                "training_rows": receipt["training_rows"],
            }
        )
    return result


def run(source_path, index_path, audit_path, output):
    item, event, paths = bind(source_path, index_path, audit_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "kind": "V7B_source_runtime",
        "started_at": datetime.now(UTC).isoformat(),
        "source_manifest_path": relative(source_path),
        "source_manifest_sha256": digest(source_path),
        "source_audit_path": relative(audit_path),
        "source_audit_sha256": digest(audit_path),
        "index_path": relative(index_path),
        "index_sha256": digest(index_path),
        "plan_sha256": PLAN_SHA256,
        "representative": {
            "participant": "T2",
            "session_id": item["session_id"],
            "post_implant_day": 250,
            "method": "periodic240",
            "threshold_context": 0.1,
            "fit_stop_seconds": 540,
            "deployment_seconds": 542,
            "training_rows": event["fit"]["training_rows"],
            "selection_rule": (
                "greatest numeric exposed T2 day; last complete periodic fit; no outcome selection"
            ),
        },
        "artifacts": {
            key: {"path": relative(path), "sha256": digest(path)} for key, path in paths.items()
        },
        "protocol": {
            "fast_repeats": 3,
            "paced_repeats": 2,
            "paced_seconds_each": 60,
            "refit_repeats": 10,
            "period_ms": 20,
            "warmup_bins": 500,
            "threads": 1,
            "parity_atol": 1e-10,
            "parity_rtol": 1e-10,
            "refit_allowance_seconds": 2,
            "timing_scope": (
                "feature-ready steps; cached causal features and loaded labels for refit, "
                "including mask, normalizer, ridge, export/reload"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": {},
        "source_snapshots": {},
        "blocks": [],
        "refits": [],
        "interpretation": (
            "Source-only numerical and compute replay, not fresh efficacy. The final model uses "
            "labels through540s, so earlier replay predictions are not causal scientific scores. "
            "Finite scheduling measurements do not guarantee hard realtime or acquisition latency. "
            "Cached representation and feature loading are outside refit timing."
        ),
    }
    for name in SOURCE_FILES:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        manifest["source_sha256"][name] = digest(ROOT / name)
        manifest["source_snapshots"][name] = relative(destination)
    write_json(output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            decoder, features, boundaries, expected, error, load_ms = load_workload(
                paths["model"], paths["features"]
            )
            manifest["environment"]["threadpools"] = [
                {k: p[k] for k in ("user_api", "internal_api", "num_threads", "version") if k in p}
                for p in threadpool_info()
            ]
            manifest["numpy_incremental_vs_batch_max_error"] = error
            manifest["model_load_ms"] = load_ms
            manifest["numeric_model_array_bytes"] = sum(
                getattr(decoder, k).nbytes for k in ("mean", "scale", "weights")
            )
            manifest["filter_response"] = filter_response(decoder.config)
            with np.load(paths["trace"], allow_pickle=False) as saved:
                paid = saved["purchased_mask"].copy()
                assessed = saved["prediction"]
                np.testing.assert_allclose(
                    expected[event["event_bin"] :],
                    assessed[event["event_bin"] :],
                    rtol=1e-10,
                    atol=1e-10,
                )
                manifest["post_deployment_vs_source_trace_max_error"] = float(
                    np.max(np.abs(expected[event["event_bin"] :] - assessed[event["event_bin"] :]))
                )
            targets = arrays.load_targets(paths["targets"], item["n_bins"])
            starts = sorted(boundaries)
            segments = list(zip(starts, starts[1:] + [len(expected)], strict=True))
            raw = model.representation(features, segments, decoder.config)
            manifest["refits"] = refit_measurements(
                raw,
                targets["y"],
                targets["eval_mask"],
                paid,
                segments,
                item["n_channels"],
                event,
                decoder,
                output,
            )
            durations = np.array([r["total_ns"] for r in manifest["refits"]]) / 1e6
            manifest["refit_summary"] = {
                "n": 10,
                "p50_ms": float(np.quantile(durations, 0.5)),
                "p95_ms": float(np.quantile(durations, 0.95)),
                "max_ms": float(durations.max()),
                "exceeded_two_second_allowance": int(np.sum(durations > 2000)),
                "cached_representation_bytes": raw.nbytes,
            }
            del raw, targets, paid
            write_json(output / "manifest.json", manifest)
            for mode, repeat in [("fast", i) for i in range(3)] + [("paced", i) for i in range(2)]:
                count = len(expected) if mode == "fast" else 3000
                stamps, prediction, details = measure(
                    decoder, features, boundaries, paced=mode == "paced", count=count
                )
                np.testing.assert_allclose(prediction, expected[:count], rtol=1e-10, atol=1e-10)
                path = output / f"{mode}-{repeat}.npz"
                np.savez(path, timestamps_ns=stamps, predictions=prediction)
                manifest["blocks"].append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "path": relative(path),
                        "sha256": digest(path),
                        **timestamp_summary(stamps),
                        **details,
                    }
                )
                write_json(output / "manifest.json", manifest)
                print(
                    json.dumps(
                        {
                            "runtime_block": mode,
                            "repeat": repeat,
                            "service_p99_ms": manifest["blocks"][-1]["service_p99_ms"],
                        }
                    ),
                    flush=True,
                )
        for name, checksum in manifest["source_sha256"].items():
            if digest(ROOT / name) != checksum:
                raise ValueError("timing source changed during measurement")
        manifest["status"] = "complete"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        write_json(output / "manifest.json", manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(output / "manifest.json", manifest)
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.index, args.source_audit, args.output)
