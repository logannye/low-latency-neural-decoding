"""Source-only bounded-state calibration equivalence and update/solve timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from neural_experiments.v4 import data
from neural_experiments.v7 import model
from neural_experiments.v7.incremental import MAX_CHUNK_ROWS, CenteredRidge
from neural_experiments.v7.runtime_continual import ROOT, bind, checked, digest, relative

SOURCES = ("neural_experiments/v7/incremental.py", "neural_experiments/v7/incremental_runtime.py")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def duration_summary(stamps, first, last):
    values = (stamps[:, last] - stamps[:, first]) / 1e6
    return {
        "n": len(values),
        "p50_ms": float(np.quantile(values, 0.5)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "max_ms": float(values.max()),
        "total_ms": float(values.sum()),
    }


def run(source_path, index_path, audit_path, output):
    item, _, paths = bind(source_path, index_path, audit_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads(Path(source_path).read_text())
    events = sorted(
        (
            e
            for e in source["events"]
            if e["session_id"] == item["session_id"]
            and e["method"] == "periodic240"
            and e["threshold"] == 0.1
            and e["event"] == "model_fitted"
        ),
        key=lambda e: e["event_bin"],
    )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "kind": "V7B_source_incremental_ridge",
        "started_at": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": digest(source_path),
        "source_audit_sha256": digest(audit_path),
        "index_sha256": digest(index_path),
        "source_manifest_path": relative(source_path),
        "source_audit_path": relative(audit_path),
        "index_path": relative(index_path),
        "representative": {
            "participant": "T2",
            "post_implant_day": 250,
            "method": "periodic240",
            "threshold_context": 0.1,
        },
        "protocol": {
            "max_chunk_rows": MAX_CHUNK_ROWS,
            "parity_atol": 1e-10,
            "parity_rtol": 1e-10,
            "feature_penalty": 1,
            "free_intercept": True,
            "label_policy_changed": False,
            "normalization_changed": False,
            "solve_count": 5,
            "replay_count": 1,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source_sha256": {},
        "fits": [],
        "scope": (
            "Source-only arithmetic/compute proof; no efficacy result. Timing excludes "
            "feature acquisition, causal feature generation and input loading. Memory counts "
            "persistent numeric components, not process RSS, Python objects or solver scratch."
        ),
    }
    for name in SOURCES:
        dest = output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        manifest["source_sha256"][name] = digest(ROOT / name)
    write(output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=1):
            manifest["environment"]["threadpools"] = [
                {k: p[k] for k in ("user_api", "internal_api", "num_threads", "version") if k in p}
                for p in threadpool_info()
            ]
            features = data.load_features(paths["features"])
            targets = data.load_targets(paths["targets"], item["n_bins"])
            n, d = item["n_bins"], 2 * item["n_channels"]
            sequence = features["sequence_id"]
            boundaries = np.r_[0, np.flatnonzero(np.diff(sequence)) + 1, n]
            start_for_bin = np.repeat(boundaries[:-1], np.diff(boundaries))
            segments = [
                (int(a), int(min(b, 1500)))
                for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
                if a < 1500
            ]
            prefix = model.representation(
                {k: features[k][:1500] for k in model.core.ARRAY_KEYS}, segments, model.CONFIG
            )
            floor = np.r_[np.full(d // 2, 0.1), np.full(d // 2, 0.01)]
            state = CenteredRidge(prefix.mean(0), np.maximum(prefix.std(0), floor))
            normalizer_buffer_bytes = prefix.nbytes
            del prefix
            first_model = model.Decoder.load(
                checked(ROOT, events[0]["model_path"], events[0]["model_sha256"])
            )
            np.testing.assert_array_equal(state.mean, first_model.mean)
            np.testing.assert_array_equal(state.scale, first_model.scale)
            stream = model.StreamingDecoder(
                model.Decoder(model.CONFIG, state.mean, state.scale, np.zeros((d + 1, 2)))
            )
            paid = np.zeros(n, bool)
            for start, stop in events[-1]["fit"]["purchased_intervals"]:
                paid[start:stop] = True
            expected_eligible = paid & targets["eval_mask"] & (np.arange(n) >= 1500)
            expected_eligible &= np.arange(n) - start_for_bin >= 62
            selected_ledger = np.zeros(n, bool)
            raw_buffer = np.empty((MAX_CHUNK_ROWS, d))
            target_buffer = np.empty((MAX_CHUNK_ROWS, 2))
            update_stamps, solve_stamps, updates = [], [], []
            chunk_start, used, next_event = 1500, 0, 0
            last_stop = events[-1]["fit"]["fit_stop_bin"]
            for i in range(last_stop):
                if i == start_for_bin[i]:
                    stream.reset()
                stream.step({k: features[k][i] for k in model.core.ARRAY_KEYS})
                if i < 1500:
                    continue
                raw_buffer[used] = stream.fast
                target_buffer[used] = targets["y"][i] if expected_eligible[i] else np.nan
                used += 1
                at_fit = i + 1 == events[next_event]["fit"]["fit_stop_bin"]
                if used == MAX_CHUNK_ROWS or at_fit:
                    stop = i + 1
                    before = state.n
                    t0, c0 = time.perf_counter_ns(), time.process_time_ns()
                    selected = state.add(
                        raw_buffer[:used],
                        target_buffer[:used],
                        np.arange(chunk_start, stop),
                        paid[chunk_start:stop],
                        targets["eval_mask"][chunk_start:stop],
                        start_for_bin[chunk_start:stop],
                        observed_stop=stop,
                    )
                    c1, t1 = time.process_time_ns(), time.perf_counter_ns()
                    update_stamps.append((t0, t1, c0, c1))
                    selected_ledger[chunk_start:stop] = selected
                    updates.append((chunk_start, stop, before, state.n))
                    used, chunk_start = 0, stop
                if at_fit:
                    event = events[next_event]
                    expected = model.Decoder.load(
                        checked(ROOT, event["model_path"], event["model_sha256"])
                    )
                    t0, c0 = time.perf_counter_ns(), time.process_time_ns()
                    fitted = state.decoder()
                    c1, t1 = time.process_time_ns(), time.perf_counter_ns()
                    path = output / f"fit-{next_event}.npz"
                    fitted.save(path)
                    fitted = model.Decoder.load(path)
                    c2, t2 = time.process_time_ns(), time.perf_counter_ns()
                    solve_stamps.append((t0, t1, t2, c0, c1, c2))
                    np.testing.assert_allclose(
                        fitted.weights, expected.weights, rtol=1e-10, atol=1e-10
                    )
                    if state.n != event["fit"]["training_rows"]:
                        raise ValueError("accumulator and audited paid-label count differ")
                    np.testing.assert_array_equal(
                        selected_ledger[: i + 1], expected_eligible[: i + 1]
                    )
                    stats_path = output / f"moments-{next_event}.npz"
                    np.savez(stats_path, **state.snapshot())
                    manifest["fits"].append(
                        {
                            "ordinal": next_event,
                            "fit_stop_bin": i + 1,
                            "deployment_bin": event["event_bin"],
                            "training_rows": state.n,
                            "source_model_sha256": event["model_sha256"],
                            "model_path": relative(path),
                            "model_sha256": digest(path),
                            "moments_path": relative(stats_path),
                            "moments_sha256": digest(stats_path),
                            "selected_mask_sha256": hashlib.sha256(
                                selected_ledger[: i + 1].tobytes()
                            ).hexdigest(),
                            "coefficient_max_error": float(
                                np.max(np.abs(fitted.weights - expected.weights))
                            ),
                            "moment_bytes": state.moment_nbytes,
                            "normalizer_bytes": state.normalizer_nbytes,
                        }
                    )
                    next_event += 1
            if next_event != 5:
                raise ValueError("all five chronological source fits required")
            update_stamps = np.asarray(update_stamps, dtype=np.int64)
            solve_stamps = np.asarray(solve_stamps, dtype=np.int64)
            clocks = output / "clocks.npz"
            np.savez(
                clocks,
                update_timestamps_ns=update_stamps,
                solve_timestamps_ns=solve_stamps,
                update_geometry=np.asarray(updates, dtype=np.int64),
            )
            manifest["clocks"] = {"path": relative(clocks), "sha256": digest(clocks)}
            manifest["memory"] = {
                "moment_bytes": state.moment_nbytes,
                "normalizer_bytes": state.normalizer_nbytes,
                "persistent_calibration_bytes": state.moment_nbytes + state.normalizer_nbytes,
                "raw_and_target_chunk_buffers_bytes": raw_buffer.nbytes + target_buffer.nbytes,
                "normalization_startup_buffer_bytes": normalizer_buffer_bytes,
                "causal_filter_state_bytes": stream.state_nbytes,
                "batch_full_prefix_representation_bytes": n * d * 8,
                "feature_dimension": d,
                "scope": (
                    "numeric calibration state and explicit buffers only; validation loaded "
                    "source inputs, masks and references are excluded; arithmetic scratch "
                    "and Python overhead excluded"
                ),
            }
            manifest["timing"] = {
                "update_wall": duration_summary(update_stamps, 0, 1),
                "update_cpu": duration_summary(update_stamps, 2, 3),
                "solve_wall": duration_summary(solve_stamps, 0, 1),
                "solve_cpu": duration_summary(solve_stamps, 3, 4),
                "solve_export_reload_wall": duration_summary(solve_stamps, 0, 2),
                "solve_export_reload_cpu": duration_summary(solve_stamps, 3, 5),
            }
        for name, expected_hash in manifest["source_sha256"].items():
            if digest(ROOT / name) != expected_hash:
                raise ValueError("engineering source changed during measurement")
        manifest["status"] = "complete"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        write(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", str(exc)
        write(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "index", "source-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.index, args.source_audit, args.output)
    print(
        json.dumps(
            {"status": result["status"], "memory": result["memory"], "timing": result["timing"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
