"""Same-artifact full-step fast and paced replay for either V2 decoder family."""

from __future__ import annotations

import argparse
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from neural_experiments.v2.transfer import digest, relative, write_json


class WindowDecoder:
    """Independent PyTorch execution of the exact observed receptive-field window."""

    def __init__(self, model, transforms, kind):
        self.model, self.transforms, self.kind = model, transforms, kind
        self.receptive_field = model.receptive_field
        self.reset()

    @property
    def state_nbytes(self):
        return sum(row.nbytes for row in self.rows)

    def reset(self):
        self.rows = []

    @torch.inference_mode()
    def step(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        channels = (
            len(self.transforms[0])
            if self.kind == "transfer"
            else len(self.transforms["input_mean"])
        )
        if raw.shape != (channels,) or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("invalid neural observations")
        self.rows.append(raw.copy())
        if len(self.rows) > self.receptive_field:
            self.rows.pop(0)
        x = np.asarray(self.rows, dtype=np.float32)
        if self.kind == "transfer":
            x = (np.sqrt(x) - self.transforms[0]) / self.transforms[1]
        else:
            x = (x - self.transforms["input_mean"]) / self.transforms["input_scale"]
        prediction = self.model(torch.from_numpy(x)[None])[0, -1].numpy()
        if self.kind == "h1":
            prediction = (
                prediction * self.transforms["target_scale"] + self.transforms["target_mean"]
            )
        return prediction


def load_workload(kind, model_path, feature_path):
    if kind == "h1":
        from neural_decoder.data import load_feature_file
        from neural_experiments.v2.accuracy import CausalStream, load_model, predict

        model, transforms, _ = load_model(model_path)
        x = load_feature_file(feature_path)["counts"][:30000].astype(np.float32)
        segments = [(0, len(x))]
        expected = predict(model, transforms, x)

        def factory():
            return CausalStream(model, transforms)
    else:
        from neural_experiments.v2.data import load_features
        from neural_experiments.v2.transfer import IncrementalDecoder, load_model, predict

        model, transforms = load_model(model_path)
        features = load_features(feature_path)
        x = features["x"][:30000]
        boundaries = np.r_[
            0, np.flatnonzero(np.diff(features["sequence_id"][: len(x)])) + 1, len(x)
        ]
        segments = list(zip(boundaries[:-1], boundaries[1:], strict=True))
        expected = predict(model, x, segments, transforms)

        def factory():
            return IncrementalDecoder(model, transforms)

    return (
        model,
        transforms,
        x,
        segments,
        expected,
        {
            "numpy_incremental": factory,
            "torch_window": lambda: WindowDecoder(model, transforms, kind),
        },
    )


def summarize_timestamps(stamps):
    # Columns: nominal release, step start, step completion, absolute deadline.
    service = (stamps[:, 2] - stamps[:, 1]) / 1e6
    queue = (stamps[:, 1] - stamps[:, 0]) / 1e6
    end_to_end = (stamps[:, 2] - stamps[:, 0]) / 1e6
    return {
        "n": len(stamps),
        "service_p50_ms": float(np.quantile(service, 0.5)),
        "service_p99_ms": float(np.quantile(service, 0.99)),
        "service_max_ms": float(service.max()),
        "queue_p99_ms": float(np.quantile(queue, 0.99)),
        "release_to_completion_p99_ms": float(np.quantile(end_to_end, 0.99)),
        "deadline_misses": int(np.sum(stamps[:, 2] > stamps[:, 3])),
        "negative_queue_count": int(np.sum(stamps[:, 1] < stamps[:, 0])),
    }


def retained_numeric_bytes(arrays):
    """Count owning allocations, including any extra row retained by a NumPy view."""
    owners = {}
    for array in arrays:
        owner = array
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        owners[id(owner)] = owner.nbytes
    return sum(owners.values())


def measure(factory, x, boundaries, *, paced, count, warmup=500, period_ns=20_000_000):
    started = time.perf_counter_ns()
    decoder = factory()
    initialization_ns = time.perf_counter_ns() - started
    for i in range(warmup):
        if i % len(x) in boundaries:
            decoder.reset()
        decoder.step(x[i % len(x)])
    decoder.reset()
    stamps = np.empty((count, 4), dtype=np.int64)
    predictions = []
    origin = time.perf_counter_ns() + (20_000_000 if paced else 0)
    for i in range(count):
        release = origin + i * period_ns if paced else time.perf_counter_ns()
        if paced:
            remaining = release - time.perf_counter_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)
        start = time.perf_counter_ns()
        if i % len(x) in boundaries:
            decoder.reset()
        result = decoder.step(x[i % len(x)])
        completed = time.perf_counter_ns()
        stamps[i] = release, start, completed, release + period_ns
        predictions.append(result)
    arrays = getattr(decoder, "rings", getattr(decoder, "histories", getattr(decoder, "rows", [])))
    return (
        stamps,
        np.asarray(predictions, dtype=np.float32),
        {
            "initialization_ms": initialization_ns / 1e6,
            "persistent_numeric_state_bytes": retained_numeric_bytes(arrays),
            "state_measurement": "numeric caches only, excluding weights, Python and scratch",
        },
    )


def run(kind, model_path, feature_path, output, *, paced_seconds=60, fast_repeats=3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        model, transforms, x, segments, expected, factories = load_workload(
            kind, model_path, feature_path
        )
        if not len(x):
            raise ValueError("empty timing workload")
        manifest = {
            "kind": kind,
            "model_path": relative(model_path),
            "model_sha256": digest(model_path),
            "feature_path": relative(feature_path),
            "feature_sha256": digest(feature_path),
            "runtime_source_sha256": digest(__file__),
            "status": "running",
            "protocol": {
                "period_ms": 20,
                "warmup_bins": 500,
                "fast_repeats": fast_repeats,
                "feature_prefix_cap_bins": 30000,
                "workload_bins": len(x),
                "paced_blocks_per_implementation": 2,
                "paced_seconds": paced_seconds,
                "service_boundary": "validation, normalization, model, state, boundary reset",
                "exclude": "neural acquisition, upstream feature extraction, transport",
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "threadpools": threadpool_info(),
            },
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "numeric_weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "full_trace_parity": {},
            "blocks": [],
        }
        consumed = [Path(__file__), Path(__file__).with_name("transfer.py")]
        if kind == "h1":
            from neural_decoder import data as feature_loader

            consumed.extend(
                [Path(__file__).with_name("accuracy.py"), Path(feature_loader.__file__)]
            )
        else:
            consumed.append(Path(__file__).with_name("data.py"))
        manifest["source"] = {}
        manifest["source_snapshots"] = {}
        for source in consumed:
            original = relative(source)
            snapshot = output / "source" / original
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
            manifest["source"][original] = digest(source)
            manifest["source_snapshots"][original] = relative(snapshot)
        atol = 1e-7 if kind == "h1" else 3e-5
        manifest["protocol"]["parity_atol"] = atol
        manifest["protocol"]["parity_rtol"] = 3e-5
        write_json(output / "manifest.json", manifest)
        boundaries = {int(start) for start, _ in segments}
        # Full trace parity precedes timing. These are the same serialized weights.
        for name, factory in factories.items():
            decoder = factory()
            actual = np.empty_like(expected)
            for i, row in enumerate(x):
                if i in boundaries:
                    decoder.reset()
                actual[i] = decoder.step(row)
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=atol)
            manifest["full_trace_parity"][name] = float(np.max(np.abs(actual - expected)))
        names = list(factories)
        schedules = [
            ("fast", repeat, names if repeat % 2 == 0 else names[::-1])
            for repeat in range(fast_repeats)
        ]
        schedules += [
            ("paced", repeat, names if repeat == 0 else names[::-1]) for repeat in range(2)
        ]
        for mode, repeat, order in schedules:
            for name in order:
                count = round(paced_seconds / 0.02) if mode == "paced" else len(x)
                stamps, predictions, allocation = measure(
                    factories[name], x, boundaries, paced=mode == "paced", count=count
                )
                expected_slice = expected[np.arange(count) % len(x)]
                np.testing.assert_allclose(predictions, expected_slice, rtol=3e-5, atol=atol)
                stem = f"{mode}-{repeat}-{name}"
                np.savez(output / f"{stem}.npz", timestamps=stamps, predictions=predictions)
                block = {
                    "mode": mode,
                    "repeat": repeat,
                    "implementation": name,
                    "artifact_path": relative(output / f"{stem}.npz"),
                    "artifact_sha256": digest(output / f"{stem}.npz"),
                    **summarize_timestamps(stamps),
                    **allocation,
                }
                manifest["blocks"].append(block)
                write_json(output / "manifest.json", manifest)
                print(
                    f"{stem}: p99={block['service_p99_ms']:.6f} ms; "
                    f"misses={block['deadline_misses']}",
                    flush=True,
                )
        manifest["status"] = "complete"
        write_json(output / "manifest.json", manifest)
        summary = {**manifest, "manifest_sha256": digest(output / "manifest.json")}
        write_json(output / "summary.json", summary)


def audit(output):
    """Reconstruct all reported timing summaries from saved monotonic timestamps."""
    import json

    from neural_experiments.v2.transfer import resolve

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("timing run incomplete")
    for source, expected_hash in manifest["source"].items():
        if digest(resolve(manifest["source_snapshots"][source])) != expected_hash:
            raise ValueError("changed timing source snapshot")
    for block in manifest["blocks"]:
        path = resolve(block["artifact_path"])
        if digest(path) != block["artifact_sha256"]:
            raise ValueError("changed timing artifact")
        with np.load(path, allow_pickle=False) as arrays:
            stamps = arrays["timestamps"]
        if np.any(stamps[:, 2] < stamps[:, 1]) or np.any(np.diff(stamps[:, 0]) < 0):
            raise ValueError("invalid monotonic clock ordering")
        derived = summarize_timestamps(stamps)
        for key, value in derived.items():
            if not np.isclose(value, block[key], rtol=0, atol=1e-12):
                raise ValueError(f"timing reconstruction differs: {key}")
    receipt = {
        "status": "passed",
        "blocks_checked": len(manifest["blocks"]),
        "manifest_sha256": digest(output / "manifest.json"),
        "audit_source_sha256": digest(__file__),
    }
    write_json(output / "timestamp-audit.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("h1", "transfer"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.kind, args.model, args.features, args.output)
    audit(args.output)


if __name__ == "__main__":
    main()
