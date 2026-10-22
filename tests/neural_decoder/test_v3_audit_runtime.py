"""Independent timing audit rejects schedule tampering with updated file hashes."""

import json

import numpy as np
import pytest

from neural_experiments.v3 import audit_runtime as audit


def fixture_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = {
        "kind": "regularized",
        "family": "tx",
        "tau_seconds": 0.2,
        "baseline_tau_seconds": None,
        "ridge_lambda": 1.0,
    }
    np.savez(
        tmp_path / "model.npz",
        mean=np.zeros(2),
        scale=np.ones(2),
        coefficients=np.zeros((2, 2)),
        intercept=np.zeros(2),
        config=np.array(json.dumps(config)),
    )
    np.savez(tmp_path / "feature.npz", x_tx=np.ones((5, 2)))
    (tmp_path / "plan.json").write_text("{}")
    manifest = {
        "status": "complete",
        "kind": "v3_regularized_ridge",
        "source": {},
        "source_snapshots": {},
        "config": config,
        "parameter_count": 6,
        "numeric_weight_bytes": 48,
        "numeric_transform_bytes": 32,
        "numeric_model_array_bytes": 80,
        "full_trace_parity": {"numpy_incremental_vs_batch": 0.0},
        "blocks": [],
    }
    for name in ("plan", "model", "feature"):
        path = tmp_path / f"{name}.{'json' if name == 'plan' else 'npz'}"
        manifest[name + "_path"] = path.name
        manifest[name + "_sha256"] = audit.digest(path)
    for name in (
        "neural_experiments/v3/runtime.py",
        "neural_experiments/v3/model.py",
        "neural_experiments/v3/data.py",
        "neural_experiments/v2/data.py",
    ):
        path = run / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source")
        manifest["source"][name] = audit.digest(path)
        manifest["source_snapshots"][name] = str(path.relative_to(tmp_path))
    manifest["runtime_source_sha256"] = manifest["source"]["neural_experiments/v3/runtime.py"]
    manifest["protocol"] = {
        "period_ms": 20,
        "warmup_bins": 500,
        "fast_repeats": 3,
        "paced_blocks_per_implementation": 2,
        "paced_seconds": 60,
        "implementation": "numpy_incremental",
        "feature_prefix_cap_bins": 30000,
        "parity_atol": 1e-10,
        "parity_rtol": 1e-10,
        "workload_bins": 5,
    }
    for mode, repeat in [("fast", i) for i in range(3)] + [("paced", i) for i in range(2)]:
        n = 3000 if mode == "paced" else 5
        release = np.arange(n, dtype=np.int64) * 20_000_000
        stamps = np.column_stack((release, release + 30, release + 1000, release + 20_000_000))
        path = run / f"{mode}-{repeat}.npz"
        np.savez(path, timestamps_ns=stamps, predictions=np.zeros((n, 2)))
        manifest["blocks"].append(
            {
                "mode": mode,
                "repeat": repeat,
                "implementation": "numpy_incremental",
                "artifact_path": str(path.relative_to(tmp_path)),
                "artifact_sha256": audit.digest(path),
                "persistent_numeric_state_bytes": 32,
                **audit.reconstruct(stamps, paced=mode == "paced", period_ns=20_000_000),
            }
        )
    save_manifest(run, manifest)
    return run, manifest


def save_manifest(run, manifest):
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.json").write_text(
        json.dumps({**manifest, "manifest_sha256": audit.digest(run / "manifest.json")})
    )


def test_five_block_inventory_and_percentiles_verified(tmp_path):
    run, _ = fixture_run(tmp_path)
    receipt = audit.audit(run, tmp_path)
    assert receipt["status"] == "passed"
    assert receipt["blocks_checked"] == 5
    assert receipt["blocks"][0]["service_p99_ms"] == pytest.approx(0.00097)


def test_tampered_deadline_rejected_even_with_rehashed_artifacts(tmp_path):
    run, manifest = fixture_run(tmp_path)
    block = manifest["blocks"][3]
    path = tmp_path / block["artifact_path"]
    with np.load(path) as z:
        stamps, predictions = z["timestamps_ns"], z["predictions"]
    stamps[5, 3] += 1
    np.savez(path, timestamps_ns=stamps, predictions=predictions)
    block["artifact_sha256"] = audit.digest(path)
    save_manifest(run, manifest)
    with pytest.raises(ValueError, match="deadline"):
        audit.audit(run, tmp_path)


def test_missing_repeat_rejected(tmp_path):
    run, manifest = fixture_run(tmp_path)
    manifest["blocks"].pop()
    save_manifest(run, manifest)
    with pytest.raises(ValueError, match="inventory"):
        audit.audit(run, tmp_path)
