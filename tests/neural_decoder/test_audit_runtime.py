"""Independent timing arithmetic and evidence mutation checks, without wall-clock waits."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def auditor():
    path = Path(__file__).resolve().parents[2] / "neural_experiments/audit_runtime.py"
    spec = importlib.util.spec_from_file_location("independent_runtime_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def test_reproduction_config_is_resolved_by_hash_not_original_filename(auditor, tmp_path):
    old_name = "neural_experiments/configs/runtime-v1.json"
    fresh_name = "neural_experiments/configs/runtime-reproduction-20261022.json"
    old = {"primary_model_sha256": "original-model"}
    fresh = {"primary_model_sha256": "new-model"}
    for name, value in ((old_name, old), (fresh_name, fresh)):
        write_json(tmp_path / "source" / name, value)
    manifest = {
        "configuration": fresh,
        "config_sha256": auditor.digest(tmp_path / "source" / fresh_name),
        "source": {
            "files": {
                name: auditor.digest(tmp_path / "source" / name) for name in (old_name, fresh_name)
            }
        },
    }
    assert auditor.retained_configuration(tmp_path, manifest) == fresh
    manifest["config_sha256"] = "unretained-hash"
    with pytest.raises(ValueError, match="no retained configuration"):
        auditor.retained_configuration(tmp_path, manifest)


def test_retained_config_rejects_changed_bytes_and_summary(auditor, tmp_path):
    name = "neural_experiments/configs/fresh-runtime.json"
    path = tmp_path / "source" / name
    write_json(path, {"threads": 1})
    checksum = auditor.digest(path)
    manifest = {
        "config_sha256": checksum,
        "configuration": {"threads": 2},
        "source": {"files": {name: checksum}},
    }
    with pytest.raises(ValueError, match="differs from its retained"):
        auditor.retained_configuration(tmp_path, manifest)
    manifest["configuration"] = {"threads": 1}
    write_json(path, {"threads": 2})
    with pytest.raises(ValueError, match="changed evidence"):
        auditor.retained_configuration(tmp_path, manifest)


def test_known_queue_service_deadline_and_quantile_arithmetic(auditor):
    stamps = (
        np.array([[0, 0, 10, 20, 20], [20, 25, 50, 65, 40], [40, 65, 70, 75, 60]], dtype=np.int64)
        * 1_000_000
    )
    actual = auditor.independently_summarize(stamps, 20_000_000)
    assert actual["step_service"] == {
        "p50_ms": 20.0,
        "p95_ms": 38.0,
        "p99_ms": 39.6,
        "max_ms": 40.0,
    }
    assert actual["queue_delay"]["p50_ms"] == 5
    assert actual["release_to_completion"]["max_ms"] == 45
    assert actual["missed_deadlines"] == 2
    assert actual["max_backlog_bins"] == 1
    assert actual["first_start_to_last_completion_seconds"] == 0.075
    invalid = stamps.copy()
    invalid[2, 1] = stamps[1, 3] - 1
    with pytest.raises(ValueError, match="event order"):
        auditor.independently_summarize(invalid, 20_000_000)
    changed_summary = dict(actual, missed_deadlines=1)
    with pytest.raises(ValueError, match="missed_deadlines"):
        auditor.compare_summary(actual, changed_summary)


def test_independent_reference_has_exact_causal_delay_and_transforms(auditor):
    weights = {
        "input_mean": np.ones(176, dtype=np.float32),
        "input_scale": np.full(176, 2, dtype=np.float32),
        "target_mean": np.full(7, 7, dtype=np.float32),
        "target_scale": np.full(7, 3, dtype=np.float32),
        "head_weight": np.zeros((7, 32), dtype=np.float32),
        "head_bias": np.zeros(7, dtype=np.float32),
    }
    weights["head_weight"][:, 0] = 1
    for layer, channels in enumerate((176, 32, 32), 1):
        weights[f"conv{layer}_weight"] = np.zeros((32, channels, 3), dtype=np.float32)
        weights[f"conv{layer}_weight"][0, 0, 0] = 1
        weights[f"conv{layer}_bias"] = np.zeros(32, dtype=np.float32)
    counts = np.ones((20, 176), dtype=np.uint32)
    counts[:, 0] = np.arange(1, 21)
    expected = np.full((20, 7), 7, dtype=np.float32)
    expected[14:] += (np.arange(6, dtype=np.float32) * 1.5)[:, None]
    np.testing.assert_array_equal(auditor.reference_trace(counts, weights), expected)


@pytest.fixture
def evidence(tmp_path, auditor):
    """A full counterbalanced compression inventory with synthetic timestamps."""
    run, quality, cache = (tmp_path / name for name in ("run", "quality", "cache"))
    run.mkdir()
    cache.mkdir()
    (run / "source.diff").write_bytes(b"")
    model = tmp_path / "model.npz"
    np.savez(model, unused=np.array(1))
    files = []
    for i, day in enumerate((0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4)):
        file_id = f"S{day}_set_{i + 1}"
        bins = 3001 if i == 0 else 3
        path = cache / f"{file_id}.npz"
        np.savez(path, counts=np.zeros((bins, 176), dtype=np.uint8))
        files.append(
            {
                "file_id": file_id,
                "day": f"S{day}",
                "n_bins": bins,
                "provider_split": "held-in-calib",
                "features": path.name,
                "features_sha256": auditor.digest(path),
            }
        )
    index = cache / "index.json"
    write_json(index, {"schema_version": 1, "role": "development", "files": files})
    write_json(quality / "summary.json", {"quality_gate": True})
    graph_hashes = {}
    for precision, filename in (("float32", "float32.onnx"), ("dynamic_int8", "dynamic-int8.onnx")):
        path = quality / "seed-0/S4" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(precision.encode())
        graph_hashes[precision] = auditor.digest(path)
    rows = []
    for mode in ("fast", "paced"):
        for repeat, order in enumerate((("float32", "dynamic_int8"), ("dynamic_int8", "float32"))):
            for precision in order:
                folder = run / f"{mode}-{repeat}-{precision}"
                folder.mkdir()
                selected = files if mode == "fast" else files[:1]
                traces, all_stamps, offset = [], [], 1_000_000_000
                for item in selected:
                    bins = item["n_bins"]
                    release = offset + np.arange(bins, dtype=np.int64) * (
                        20_000_000 if mode == "paced" else 1000
                    )
                    stamps = np.column_stack(
                        (release, release, release + 100, release + 200, release + 20_000_000)
                    )
                    path = folder / f"{item['file_id']}.npz"
                    np.savez(
                        path,
                        stamps=stamps,
                        sequence_id=np.arange(bins),
                        predictions=np.zeros((bins, 7), dtype=np.float32),
                    )
                    traces.append(
                        {
                            "file_id": item["file_id"],
                            "bins": bins,
                            "trace_sha256": auditor.digest(path),
                        }
                    )
                    all_stamps.append(stamps)
                    offset = int(stamps[-1, 3]) + 1000
                row = {
                    "mode": mode,
                    "repeat": repeat,
                    "precision": precision,
                    "files": traces,
                    "summary": auditor.independently_summarize(
                        np.concatenate(all_stamps), 20_000_000
                    ),
                }
                write_json(folder / "summary.json", row)
                rows.append(row)
    write_json(run / "summary.json", {"runs": rows})
    manifest = {
        "status": "complete",
        "summary_sha256": auditor.digest(run / "summary.json"),
        "index_sha256": auditor.digest(index),
        "model_sha256": auditor.digest(model),
        "source": {
            "files": {},
            "head": "synthetic-test",
            "diff_sha256": auditor.digest(run / "source.diff"),
        },
        "comparison": "synthetic compression audit test",
        "quality_summary_sha256": auditor.digest(quality / "summary.json"),
        "graph_sha256": graph_hashes,
        "period_ns": 20_000_000,
    }
    write_json(run / "manifest.json", manifest)
    return run, index, model, quality


def test_complete_inventory_hashes_and_raw_summaries(auditor, evidence):
    run, index, model, quality = evidence
    report = auditor.audit(run, index, model, quality_run=quality)
    assert report["status"] == "passed" and len(report["runs"]) == 8
    assert report["max_primary_trace_absolute_difference"] is None
    trace = next((run / "fast-0-float32").glob("*.npz"))
    trace.write_bytes(trace.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed evidence"):
        auditor.audit(run, index, model, quality_run=quality)


def test_rehashed_summary_cannot_override_raw_timestamps(auditor, evidence):
    run, index, model, quality = evidence
    summary = auditor.read_json(run / "summary.json")
    row = summary["runs"][0]
    row["summary"]["step_service"]["p99_ms"] = 99
    write_json(run / "fast-0-float32/summary.json", row)
    write_json(run / "summary.json", summary)
    manifest = auditor.read_json(run / "manifest.json")
    manifest["summary_sha256"] = auditor.digest(run / "summary.json")
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="p99_ms"):
        auditor.audit(run, index, model, quality_run=quality)


def test_rehashed_workload_cannot_drop_a_file(auditor, evidence):
    run, index, model, quality = evidence
    summary = auditor.read_json(run / "summary.json")
    row = summary["runs"][0]
    row["files"].pop()
    write_json(run / "fast-0-float32/summary.json", row)
    write_json(run / "summary.json", summary)
    manifest = auditor.read_json(run / "manifest.json")
    manifest["summary_sha256"] = auditor.digest(run / "summary.json")
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="file inventory"):
        auditor.audit(run, index, model, quality_run=quality)
