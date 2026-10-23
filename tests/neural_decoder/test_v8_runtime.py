import copy
import json

import numpy as np
import pytest

from neural_experiments.v8 import model, runtime


class FakeClock:
    def __init__(self):
        self.now = 0
        self.cpu = 0

    def perf_counter_ns(self):
        self.now += 1000
        return self.now

    def process_time_ns(self):
        self.cpu += 17000
        return self.cpu

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


def test_paced_raw_clocks_prediction_parity_reset_and_complete_updates():
    rng = np.random.default_rng(881)
    features = {key: rng.random((25, 2)) for key in ("x_tx3", "x_tx45", "x_sbp")}
    decoder = model.Decoder(
        {**model.BASELINE, "taus_seconds": [0.05, 0.5]},
        np.zeros(8),
        np.ones(8),
        rng.normal(0, 0.1, (9, 2)),
    )
    segments = [(0, 14), (14, 25)]
    clocks, predictions, resource = runtime.measure(
        decoder, features, segments, count=25, warmup=10, clock=FakeClock()
    )
    np.testing.assert_allclose(predictions, decoder.predict(features, segments), atol=1e-14)
    summary = runtime.timestamp_summary(clocks)
    assert summary["n"] == 25 and summary["deadline_misses"] == 0
    assert summary["service_p50_ms"] == 0.001
    assert summary["queue_p50_ms"] == 0.002
    assert resource["numeric_filter_state_bytes"] == 8 * 8 + 2 * 8
    assert resource["process_cpu_seconds"] == 17000 / 1e9
    assert (
        resource["block_wall_seconds"]
        == (resource["wall_end_ns"] - resource["wall_start_ns"]) / 1e9
    )


def test_clock_summary_rejects_early_or_inconsistent_release_and_counts_misses():
    stamps = np.array(
        [[0, 1, 20_000_001, 20_000_000], [20_000_000, 20_000_002, 20_000_003, 40_000_000]],
        dtype=np.int64,
    )
    assert runtime.timestamp_summary(stamps)["deadline_misses"] == 1
    for column, value in ((1, -1), (2, 0), (3, 19_000_000)):
        bad = stamps.copy()
        bad[0, column] = value
        with pytest.raises(ValueError, match="monotonic"):
            runtime.timestamp_summary(bad)
    with pytest.raises(ValueError, match="monotonic"):
        runtime.timestamp_summary(stamps.astype(float))


def binding(tmp_path):
    (tmp_path / "features.npz").write_bytes(b"private neural")
    (tmp_path / "model.npz").write_bytes(b"private coefficients")
    items = [
        {
            "participant_id": "T2",
            "session_id": f"s{i}",
            "post_implant_day": 1000 - i,
            "source": {"archive_ordinal": i},
            "n_bins": 30000,
            "feature_path": "features.npz",
            "feature_sha256": runtime.digest(tmp_path / "features.npz"),
        }
        for i in range(1, 22)
    ]
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"role": "development", "sessions": items}))
    source = {
        "status": "complete",
        "study": "V8",
        "phase": "development",
        "index": {"sha256": runtime.digest(index_path)},
        "code_sha256": {},
        "plan": {"sha256": "a" * 64},
        "selection": {"primary_method": "baseline", "configurations": {"baseline": model.BASELINE}},
        "factorial_rows": [
            {
                "session_id": "s16",
                "method": "baseline",
                "config": model.BASELINE,
                "model": {
                    "model_path": "model.npz",
                    "model_sha256": runtime.digest(tmp_path / "model.npz"),
                },
            }
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "phase": "development",
                "plan_sha256": "a" * 64,
                "manifest_sha256": runtime.digest(source_path),
                "index_sha256": runtime.digest(index_path),
            }
        )
    )
    return source_path, index_path, audit_path


def test_representative_uses_archive_rank_not_numeric_day_and_exact_audit(tmp_path):
    source, index, audit = binding(tmp_path)
    item, row, paths = runtime.bind(source, index, audit, root=tmp_path)
    assert item["session_id"] == "s16" and item["post_implant_day"] == 984
    assert row["method"] == "baseline" and paths["model"] == tmp_path / "model.npz"
    changed = json.loads(audit.read_text())
    changed["manifest_sha256"] = "0" * 64
    audit.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="audited"):
        runtime.bind(source, index, audit, root=tmp_path)


def test_binding_rejects_changed_model_and_path_escape(tmp_path):
    source, index, audit = binding(tmp_path)
    (tmp_path / "model.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        runtime.bind(source, index, audit, root=tmp_path)
    outside = tmp_path.parent / "outside-v8-runtime.txt"
    outside.write_text("not under binding root")
    with pytest.raises(ValueError, match="path"):
        runtime.checked(tmp_path, outside, runtime.digest(outside))


def test_public_summary_omits_private_arrays_paths_rawclocks_and_unrecognized_fields():
    stamp = np.array([[0, 1, 2, 20_000_000]], dtype=np.int64)
    block = {
        **runtime.timestamp_summary(stamp),
        "n": 3000,
        "block_wall_seconds": 0.02,
        "process_cpu_seconds": 0.001,
        "numeric_filter_state_bytes": 1536,
        "prediction_max_absolute_error": 1e-15,
        "artifact_path": "/private/neural",
        "wall_start_ns": 12345,
        "prediction": [[0.1, 0.2]],
    }
    manifest = {
        "status": "complete",
        "kind": "source_runtime",
        "source_manifest_sha256": "a" * 64,
        "source_audit_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
        "representative": {
            "participant": "T2",
            "archive_rank": 16,
            "post_implant_day": 237,
            "config": model.BASELINE,
            "private": "/private/secret",
        },
        "artifacts": {"model": {"sha256": "d" * 64}, "features": {"sha256": "e" * 64}},
        "environment": {
            "python": "3.12",
            "numpy": "2.0",
            "platform": "macOS",
            "private": "/private",
        },
        "memory": {
            "model_numeric_bytes": 6160,
            "model_artifact_bytes": 5000,
            "feature_channels": 96,
            "filtered_dimensions": 192,
            "workload_rows": 3000,
        },
        "blocks": [{"repeat": i, **block} for i in range(2)],
    }
    audit = {"status": "PASS", "manifest_sha256": "f" * 64, "blocks_checked": 2}
    result = runtime.public_summary(
        manifest, audit, manifest_sha256="f" * 64, audit_sha256="0" * 64
    )
    encoded = json.dumps(result)
    for secret in ("/private", "wall_start_ns", "artifact_path", "12345", '"prediction"'):
        assert secret not in encoded
    changed = copy.deepcopy(audit)
    changed["status"] = "FAIL"
    with pytest.raises(ValueError, match="audited"):
        runtime.public_summary(manifest, changed, manifest_sha256="f" * 64, audit_sha256="0" * 64)
