"""Synthetic timing and mutation checks; no real recordings or performance assertions."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neural_experiments import audit_h1_final_closure_runtime as audit
from neural_experiments import h1_final_closure_runtime as runtime


class Clock:
    def __init__(self):
        self.now = 1_000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class Decoder:
    def __init__(self, clock):
        self.clock = clock
        self.position = 0

    def reset(self):
        self.clock.now += 25
        self.position = 0

    def step(self, row):
        self.clock.now += 3
        self.position += 1
        return np.full(7, self.position, dtype=np.float32)


@pytest.fixture
def trace(tmp_path):
    clock = Clock()
    files = [{"file_id": "first", "n_bins": 3}, {"file_id": "second", "n_bins": 2}]
    arrays = [
        (np.ones((row["n_bins"], 176), dtype=np.uint8), np.arange(row["n_bins"]) * 0.02)
        for row in files
    ]
    writer = runtime.BoundedWriter(tmp_path / "records.npy", 8)
    resets, rss = io.StringIO(), io.StringIO()
    result = runtime.replay(
        Decoder(clock),
        files,
        arrays,
        mode="soak",
        total_bins=8,
        period_ns=10,
        writer=writer,
        resets=resets,
        rss_log=rss,
        clock=clock,
        sleep=clock.sleep,
        process=SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1234)),
    )
    records = np.array(writer.mapping)
    metrics = runtime.timing_summary(
        np.column_stack([records[name] for name in runtime.STAMP_FIELDS]), period_ns=10
    )
    summary = dict(
        metrics,
        replay=result,
        warmup_bins=500,
        warmup_complete_ns=999,
        deadline_miss_rate=metrics["missed_deadlines"] / 8,
        service_goal_passed=True,
        logger=writer.stats(),
    )
    return (
        records,
        summary,
        [json.loads(line) for line in resets.getvalue().splitlines()],
        files,
        arrays,
    )


def run_audit(trace):
    records, summary, resets, files, arrays = trace
    references = [
        np.repeat(np.arange(1, row["n_bins"] + 1, dtype=np.float32)[:, None], 7, axis=1)
        for row in files
    ]
    return audit.audit_records(
        records,
        summary,
        resets,
        files,
        [pair[1] for pair in arrays],
        references,
        mode="soak",
        arm="B",
        period_ns=10,
        expected_bins=8,
    )


def test_global_release_origin_survives_resets_cycles_and_stalls(trace):
    records, summary, resets, _, _ = trace
    np.testing.assert_array_equal(records["release_ns"], 1000 + np.arange(8) * 10)
    assert [row["global_id"] for row in resets] == [0, 3, 5]
    assert [row["cycle"] for row in resets] == [0, 0, 1]
    assert records["sequence_id"].tolist() == [0, 1, 2, 0, 1, 0, 1, 2]
    assert summary["missed_deadlines"] > 0
    assert summary["max_backlog_bins"] >= 2
    metrics, error = run_audit(trace)
    assert error == 0 and metrics["bins"] == 8


@pytest.mark.parametrize(
    "mutation", ["prediction", "timestamp", "release_origin", "sequence", "reset", "warmup"]
)
def test_independent_auditor_rejects_plausible_mutations(trace, mutation):
    records, summary, resets, _, _ = trace
    if mutation == "prediction":
        records["prediction"][4, 2] += 0.1
    elif mutation == "timestamp":
        records["complete_ns"][1] = records["start_ns"][1] - 1
    elif mutation == "release_origin":
        records["release_ns"][3:] += 10
        records["deadline_ns"][3:] += 10
    elif mutation == "sequence":
        records["sequence_id"][3] = 2
    elif mutation == "reset":
        resets[1]["cycle"] = 1
    else:
        summary["warmup_bins"] = 499
    with pytest.raises((ValueError, AssertionError)):
        run_audit(trace)


def test_bounded_writer_flushes_without_dropping_or_growing(tmp_path):
    writer = runtime.BoundedWriter(tmp_path / "bounded.npy", 9001)
    record = np.zeros((), dtype=runtime.RECORD_DTYPE)
    for i in range(9001):
        record["global_id"] = i
        writer.append(record)
    writer.flush()
    assert writer.stats()["high_water_records"] == 4096
    assert writer.stats()["written_records"] == 9001
    assert writer.stats()["flushes"] == 3
    assert len(writer.buffer) == 4096
    np.testing.assert_array_equal(writer.mapping["global_id"], np.arange(9001))
    with pytest.raises(ValueError, match="extent"):
        writer.append(record)
    with pytest.raises(FileExistsError):
        runtime.BoundedWriter(tmp_path / "bounded.npy", 2)


def test_counterbalanced_fallback_and_real_file_boundaries():
    assert runtime.cyclic_orders("ABCD") == [list("ABCD"), list("BCDA"), list("CDAB"), list("DABC")]
    assert runtime.cyclic_orders("B") == [["B"]]
    with pytest.raises(ValueError):
        runtime.cyclic_orders([])
    files = [{"n_bins": 3}, {"n_bins": 4}]
    assert list(runtime.file_segments(files, "soak", 9)) == [(0, 0, 3), (0, 1, 4), (1, 0, 2)]
    with pytest.raises(ValueError, match="short paced"):
        list(runtime.file_segments(files, "paced", 4))
    with pytest.raises(ValueError, match="one complete"):
        list(runtime.file_segments(files, "fast", 8))


def test_fixed_runtime_protocol_cannot_be_weakened():
    path = (
        Path(__file__).resolve().parents[2] / "neural_experiments/configs/h1-final-closure-v1.json"
    )
    config = json.loads(path.read_text())
    runtime.validate_runtime_config(config)
    config["runtime"]["soak_bins"] = 89_999
    with pytest.raises(ValueError, match="prespecified"):
        runtime.validate_runtime_config(config)


def test_failed_admission_leaves_an_immutable_attempt(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}")

    def fail(*args, **kwargs):
        raise ValueError("quality receipt mismatch")

    monkeypatch.setattr(runtime, "admit_quality", fail)
    output = tmp_path / "results/neural-h1/attempt"
    with pytest.raises(ValueError, match="receipt"):
        runtime.run_campaign(
            config, tmp_path / "quality", tmp_path / "audit", output, root=tmp_path
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["statuses"]["independent_verification"] == "not_run"
    with pytest.raises(FileExistsError):
        runtime.run_campaign(
            config, tmp_path / "quality", tmp_path / "audit", output, root=tmp_path
        )


def test_admission_authenticates_predetermined_model_bytes(tmp_path, monkeypatch):
    from neural_experiments import audit_h1_final_closure as quality_audit

    path = (
        Path(__file__).resolve().parents[2] / "neural_experiments/configs/h1-final-closure-v1.json"
    )
    config = json.loads(path.read_text())
    model_path = tmp_path / "model.npz"
    model_path.write_bytes(b"synthetic pinned model")
    config["models"][0].update(path="model.npz", sha256=runtime.sha256(model_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    quality = tmp_path / "quality"
    quality.mkdir()
    (quality / "summary.json").write_text(
        json.dumps({"models": [{"seed": 0, "model": config["models"][0]}]})
    )
    receipt = {
        "config_sha256": runtime.sha256(config_path),
        "qualified_arms": dict.fromkeys("ABCDE", True),
    }
    monkeypatch.setattr(quality_audit, "validate_audit_receipt", lambda *args: receipt)
    runtime.admit_quality(config_path, quality, tmp_path / "audit.json", tmp_path)
    model_path.write_bytes(b"finite-looking replacement weights")
    with pytest.raises(ValueError, match="changed evidence"):
        runtime.admit_quality(config_path, quality, tmp_path / "audit.json", tmp_path)
