"""Saved-clock audit must reject tampering even after artifact hashes are refreshed."""

import json
import shutil

import numpy as np
import pytest

from neural_experiments.v4 import runtime, runtime_public
from neural_experiments.v4 import runtime_audit as audit


def save_manifest(run, manifest):
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.json").write_text(
        json.dumps({**manifest, "manifest_sha256": audit.digest(run / "manifest.json")})
    )


def fixture_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = {
        "family": "tx45",
        "tau_seconds": 0.4,
        "baseline_tau_seconds": None,
        "adapt_gamma": 1.0,
    }
    np.savez(
        tmp_path / "model.npz",
        format_version=np.asarray(4, dtype=np.int64),
        mean=np.zeros(96),
        scale=np.ones(96),
        weights=np.zeros((97, 2)),
        config=np.asarray(json.dumps(config)),
    )
    np.savez(
        tmp_path / "feature.npz",
        **{key: np.ones((5, 96), dtype=np.float32) for key in ("x_tx3", "x_tx45", "x_sbp")},
        timestamps=np.arange(5, dtype=np.float64) * 0.02,
        sequence_id=np.zeros(5, dtype=np.int64),
        channel_ids=np.arange(96, dtype=np.int64),
    )
    shutil.copyfile(audit.ROOT / "neural_experiments/v4/plan.json", tmp_path / "plan.json")
    manifest = {
        "status": "complete",
        "kind": "v4_history_adapted_ridge",
        "representative_rule": audit.read(tmp_path / "plan.json")["runtime"],
        "source": {},
        "source_snapshots": {},
        "config": config,
        "parameter_count": 194,
        "numeric_weight_bytes": 1552,
        "numeric_transform_bytes": 1536,
        "numeric_model_array_bytes": 3088,
        "full_trace_parity": {"numpy_incremental_vs_batch": 0.0},
        "environment": {"threads": 1, "threadpools": [{"num_threads": 1}]},
        "blocks": [],
    }
    for name in ("plan", "model", "feature"):
        path = tmp_path / f"{name}.{'json' if name == 'plan' else 'npz'}"
        manifest[f"{name}_path"] = path.name
        manifest[f"{name}_sha256"] = audit.digest(path)
    for name in (
        "neural_experiments/v4/runtime.py",
        "neural_experiments/v4/model.py",
        "neural_experiments/v4/data.py",
        "neural_experiments/v2/data.py",
    ):
        path = run / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source")
        manifest["source"][name] = audit.digest(path)
        manifest["source_snapshots"][name] = str(path.relative_to(tmp_path))
    manifest["runtime_source_sha256"] = manifest["source"]["neural_experiments/v4/runtime.py"]
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
        service = (np.arange(n, dtype=np.int64) % 5 + 1) * 1000
        stamps = np.column_stack(
            (release, release + 30, release + 30 + service, release + 20_000_000)
        )
        path = run / f"{mode}-{repeat}.npz"
        np.savez(path, timestamps_ns=stamps, predictions=np.zeros((n, 2)))
        manifest["blocks"].append(
            {
                "mode": mode,
                "repeat": repeat,
                "implementation": "numpy_incremental",
                "artifact_path": str(path.relative_to(tmp_path)),
                "artifact_sha256": audit.digest(path),
                "persistent_numeric_state_bytes": 1536,
                **audit.reconstruct(stamps, paced=mode == "paced", period_ns=20_000_000),
            }
        )
    save_manifest(run, manifest)
    return run, manifest


def test_exact_inventory_and_independent_order_statistic(tmp_path):
    run, _ = fixture_run(tmp_path)
    result = audit.audit(run, tmp_path)
    assert result["status"] == "passed" and result["blocks_checked"] == 5
    assert result["blocks"][0]["service_p99_ms"] == pytest.approx(0.00496)
    assert result["blocks"][3]["service_p99_ms"] == pytest.approx(0.005)
    assert result["blocks"][3]["release_to_completion_p99_ms"] == pytest.approx(0.00503)
    assert all(b["deadline_misses"] == 0 for b in result["blocks"])


@pytest.mark.parametrize("change", ["deadline", "queue", "release", "overlap"])
def test_rehashed_invalid_clocks_rejected(tmp_path, change):
    run, manifest = fixture_run(tmp_path)
    block = manifest["blocks"][3]
    path = tmp_path / block["artifact_path"]
    with np.load(path) as z:
        stamps, predictions = z["timestamps_ns"], z["predictions"]
    if change == "deadline":
        stamps[5, 3] += 1
    elif change == "queue":
        stamps[5, 1] = stamps[5, 0] - 1
    elif change == "release":
        stamps[5] += 1
    else:
        stamps[5, 2] = stamps[6, 1] + 1
    np.savez(path, timestamps_ns=stamps, predictions=predictions)
    block["artifact_sha256"] = audit.digest(path)
    save_manifest(run, manifest)
    with pytest.raises(ValueError):
        audit.audit(run, tmp_path)


@pytest.mark.parametrize("identity", ["model", "feature", "source"])
def test_changed_bound_artifact_rejected(tmp_path, identity):
    run, manifest = fixture_run(tmp_path)
    path = (
        tmp_path / manifest["source_snapshots"]["neural_experiments/v4/model.py"]
        if identity == "source"
        else tmp_path / manifest[f"{identity}_path"]
    )
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        audit.audit(run, tmp_path)


@pytest.mark.parametrize("change", ["missing_repeat", "false_metric", "wrong_state", "bad_parity"])
def test_rehashed_manifest_declarations_rejected(tmp_path, change):
    run, manifest = fixture_run(tmp_path)
    if change == "missing_repeat":
        manifest["blocks"].pop()
    elif change == "false_metric":
        manifest["blocks"][0]["service_p99_ms"] = 0.0
    elif change == "wrong_state":
        manifest["blocks"][0]["persistent_numeric_state_bytes"] = 0
    else:
        manifest["full_trace_parity"]["numpy_incremental_vs_batch"] = 1e-3
    save_manifest(run, manifest)
    with pytest.raises(ValueError):
        audit.audit(run, tmp_path)


def fixture_publication(tmp_path):
    run, manifest = fixture_run(tmp_path)
    manifest.update(
        schema_version=1,
        filter_response=runtime.filter_response(manifest["config"]),
        model_load_ms=0.1,
        scope="feature-ready replay",
        private_notes="/Users/private/should-not-publish",
    )
    manifest["protocol"].update(
        service_boundary="complete step",
        exclude="file loading",
        scheduler="sleep",
        private_config_path="/Users/private/protocol",
    )
    manifest["environment"].update(
        python="3.12", platform="synthetic", numpy="2", scipy="1", private_path="/Users/private"
    )
    manifest["environment"]["threadpools"][0]["filepath"] = "/Users/private/library.dylib"
    for block in manifest["blocks"]:
        block.update(
            initialization_ms=0.02,
            initialization_scope="constructor before warmup",
            state_measurement="numeric arrays",
            process_cpu_seconds=0.01,
            block_wall_seconds=60.0,
            process_cpu_fraction_of_block_wall=0.01 / 60,
            process_peak_rss_bytes=4096,
            rss_scope="process high-water",
            raw_samples=[[1, 2]],
        )
    save_manifest(run, manifest)
    receipt = audit.audit(run, tmp_path)
    # Independent interpolation and NumPy quantiles may differ by a rounding ulp.
    receipt["blocks"][0]["service_p99_ms"] += 1e-16
    receipt["private_path"] = "/Users/private/audit"
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(receipt))
    return run, path, manifest


def test_publication_is_allowlisted_and_keeps_both_bound_metric_records(tmp_path):
    run, path, manifest = fixture_publication(tmp_path)
    result = runtime_public.export(run, path, tmp_path)
    text = json.dumps(result)
    for forbidden in ("/Users/", "_path", "source_snapshots", "raw_samples", "neural_experiments/"):
        assert forbidden not in text
    assert set(result["source_sha256"]) == set(runtime_public.COMPONENTS)
    assert result["blocks"][0]["service_p99_ms"] == manifest["blocks"][0]["service_p99_ms"]
    assert (
        result["independent_audit"]["blocks"][0]["service_p99_ms"]
        != (result["blocks"][0]["service_p99_ms"])
    )
    assert result["model_sha256"] == manifest["model_sha256"]


@pytest.mark.parametrize("change", ["failed", "stale_manifest", "wrong_source"])
def test_publication_requires_exact_passed_audit(tmp_path, change):
    run, path, _ = fixture_publication(tmp_path)
    receipt = audit.read(path)
    key = {
        "failed": "status",
        "stale_manifest": "manifest_sha256",
        "wrong_source": "auditor_source_sha256",
    }[change]
    receipt[key] = "wrong"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="audit"):
        runtime_public.export(run, path, tmp_path)
