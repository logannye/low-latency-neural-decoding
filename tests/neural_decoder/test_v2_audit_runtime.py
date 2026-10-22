"""Independent timestamp arithmetic and tampering checks without clocks/decoders."""

import json

import numpy as np
import pytest

from neural_experiments.v2.audit_runtime import audit, digest, reconstruct


def stamps(count, *, paced):
    result = np.empty((count, 4), dtype=np.int64)
    previous = 0
    for i in range(count):
        release = 1_000_000_000 + i * (20_000_000 if paced else 2_000_000)
        start = max(release, previous)
        completed = start + (25_000_000 if paced and i == 600 else 1_000_000)
        result[i] = release, start, completed, release + 20_000_000
        previous = completed
    return result


@pytest.fixture
def retained_run(tmp_path):
    run = tmp_path / "results/runtime"
    run.mkdir(parents=True)
    model = tmp_path / "model.npz"
    features = tmp_path / "features.npz"
    np.savez(model, **{"state__head.weight": np.zeros((7, 2), dtype=np.float32)})
    np.savez(features, counts=np.zeros((6, 176), dtype=np.float32))
    sources = (
        "neural_experiments/v2/runtime.py",
        "neural_experiments/v2/transfer.py",
        "neural_experiments/v2/accuracy.py",
        "src/neural_decoder/data.py",
    )
    manifest = {
        "status": "complete",
        "kind": "h1",
        "model_path": "model.npz",
        "model_sha256": digest(model),
        "feature_path": "features.npz",
        "feature_sha256": digest(features),
        "parameter_count": 14,
        "numeric_weight_bytes": 56,
        "source": {},
        "source_snapshots": {},
        "protocol": {
            "period_ms": 20,
            "warmup_bins": 500,
            "parity_atol": 1e-7,
            "parity_rtol": 3e-5,
            "fast_repeats": 3,
            "paced_blocks_per_implementation": 2,
            "paced_seconds": 60,
            "workload_bins": 6,
            "feature_prefix_cap_bins": 30000,
        },
        "full_trace_parity": {"numpy_incremental": 0.0, "torch_window": 0.0},
        "blocks": [],
    }
    for name in sources:
        snapshot = run / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(f"# retained fixture for {name}\n")
        manifest["source"][name] = digest(snapshot)
        manifest["source_snapshots"][name] = str(snapshot.relative_to(tmp_path))
    manifest["runtime_source_sha256"] = manifest["source"][sources[0]]
    for mode, repeats in (("fast", 3), ("paced", 2)):
        for repeat in range(repeats):
            names = ["numpy_incremental", "torch_window"]
            if repeat % 2:
                names.reverse()
            for name in names:
                count = 3000 if mode == "paced" else 6
                raw = stamps(count, paced=mode == "paced")
                path = run / f"{mode}-{repeat}-{name}.npz"
                np.savez(path, timestamps=raw, predictions=np.zeros((count, 7), np.float32))
                manifest["blocks"].append(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "implementation": name,
                        "artifact_path": str(path.relative_to(tmp_path)),
                        "artifact_sha256": digest(path),
                        # Known fixture values, not production or auditor summary arithmetic.
                        "n": count,
                        "service_p50_ms": 1.0,
                        "service_p99_ms": 1.0,
                        "service_max_ms": 25.0 if mode == "paced" else 1.0,
                        "queue_p99_ms": 0.0,
                        "release_to_completion_p99_ms": 1.0,
                        "deadline_misses": 1 if mode == "paced" else 0,
                        "negative_queue_count": 0,
                    }
                )

    def save():
        (run / "manifest.json").write_text(json.dumps(manifest, allow_nan=False))
        summary = {**manifest, "manifest_sha256": digest(run / "manifest.json")}
        (run / "summary.json").write_text(json.dumps(summary, allow_nan=False))

    save()
    return tmp_path, run, manifest, save


def test_reconstruct_known_deadline_miss_and_linear_percentile():
    raw = np.array(
        [[0, 1_000_000, 3_000_000, 20_000_000], [20_000_000, 22_000_000, 45_000_000, 40_000_000]],
        dtype=np.int64,
    )
    result = reconstruct(raw, paced=True, period_ns=20_000_000)
    assert result["service_p50_ms"] == 12.5
    assert result["service_p99_ms"] == 22.79
    assert result["release_to_completion_p99_ms"] == 24.78
    assert result["deadline_misses"] == 1


def test_valid_run_hashes_arithmetic_and_explicit_prediction_scope(retained_run):
    root, run, _, _ = retained_run
    receipt = audit(run, root=root)
    assert receipt["blocks_checked"] == 10
    assert receipt["source_snapshots_checked"] == 4
    assert receipt["summary_sha256"] == digest(run / "summary.json")
    assert not receipt["prediction_correctness_independently_verified"]
    assert sum(row["deadline_misses"] for row in receipt["blocks"]) == 4


@pytest.mark.parametrize("changed", ["model", "feature", "source"])
def test_changed_bound_artifact_is_rejected(retained_run, changed):
    root, run, manifest, _ = retained_run
    relative = (
        next(iter(manifest["source_snapshots"].values()))
        if changed == "source"
        else manifest[f"{changed}_path"]
    )
    path = root / relative
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match=f"changed {changed}"):
        audit(run, root=root)


@pytest.mark.parametrize(
    "defect", ["negative_queue", "paced_release", "deadline", "reported_p99", "block_count"]
)
def test_resigned_incorrect_evidence_is_rejected(retained_run, defect):
    root, run, manifest, save = retained_run
    if defect == "block_count":
        manifest["blocks"].pop()
    elif defect == "reported_p99":
        manifest["blocks"][0]["service_p99_ms"] += 0.1
    else:
        block = next(row for row in manifest["blocks"] if row["mode"] == "paced")
        path = root / block["artifact_path"]
        with np.load(path, allow_pickle=False) as data:
            raw, predictions = data["timestamps"], data["predictions"]
        if defect == "negative_queue":
            raw[0, 1] = raw[0, 0] - 1
        elif defect == "paced_release":
            raw[100] += 1
        else:
            raw[0, 3] += 1
        np.savez(path, timestamps=raw, predictions=predictions)
        block["artifact_sha256"] = digest(path)
    save()  # Internally consistent hashes must not hide invalid measurement evidence.
    with pytest.raises(ValueError):
        audit(run, root=root)
