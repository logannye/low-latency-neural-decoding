"""Deterministic timing-accounting fixtures; no wall-clock performance assertions."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_decoder.optimized import export_step_onnx
from neural_decoder.runtime import _validate_config, replay_timed, timing_summary, validate_runtime
from neural_decoder.temporal import CausalTCN, freeze_tcn


class FakeClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class TimedFixture:
    def __init__(self, clock):
        self.clock, self.index = clock, 0

    def step(self, row):
        self.clock.now += (3, 12, 3, 3)[self.index]
        self.index += 1
        return row[:7].astype(np.float32)


def test_pacing_accounts_for_queue_deadlines_backlog_without_dropping_bins():
    clock = FakeClock()
    counts = np.arange(4 * 176).reshape(4, 176)
    predictions, stamps = replay_timed(
        TimedFixture(clock), counts, paced=True, period_ns=10, clock=clock, sleep=clock.sleep
    )
    np.testing.assert_array_equal(
        stamps, [[0, 0, 3, 3, 10], [10, 10, 22, 22, 20], [20, 22, 25, 25, 30], [30, 30, 33, 33, 40]]
    )
    np.testing.assert_array_equal(predictions, counts[:, :7])
    result = timing_summary(stamps, period_ns=10)
    assert result["missed_deadlines"] == 1
    assert result["queue_delay"]["max_ms"] == 2e-6
    assert result["step_service"]["max_ms"] == 12e-6
    assert result["max_backlog_bins"] == 0
    # A fully queued bin differs from merely a nonzero scheduling delay.
    delayed = stamps.copy()
    delayed[2, 1:4] += 10
    delayed[3, 1:4] += 5
    assert timing_summary(delayed, period_ns=10)["max_backlog_bins"] == 1


def test_fast_replay_releases_at_processing_start_and_retains_every_output():
    clock = FakeClock()
    counts = np.ones((4, 176))
    prediction, stamps = replay_timed(
        TimedFixture(clock), counts, paced=False, period_ns=10, clock=clock, sleep=clock.sleep
    )
    np.testing.assert_array_equal(stamps[:, 0], stamps[:, 1])
    np.testing.assert_array_equal(prediction, counts[:, :7])
    summary = timing_summary(stamps, period_ns=10)
    assert summary["queue_delay"]["max_ms"] == 0
    assert summary["missed_deadlines"] == 1


def test_impossible_timing_and_invalid_traces_are_rejected():
    with pytest.raises(ValueError, match="order"):
        timing_summary(np.array([[0, 3, 2, 4, 10]]), period_ns=10)
    with pytest.raises(ValueError, match="deadline"):
        timing_summary(np.array([[0, 1, 2, 3, 9]]), period_ns=10)
    with pytest.raises(ValueError, match="sequential"):
        timing_summary(np.array([[0, 0, 15, 15, 10], [10, 10, 12, 12, 20]]), period_ns=10)
    with pytest.raises(ValueError):
        replay_timed(None, np.zeros((0, 176)), paced=False)
    with pytest.raises(ValueError):
        replay_timed(None, np.full((5, 176), -1), paced=False)


def test_prespecified_configuration_and_counterbalance():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "neural_experiments/configs/runtime-v1.json").read_text())
    _validate_config(config)
    assert config["paced_seconds_per_block"] * len(config["paced_orders"]) >= 120
    assert config["paced_orders"][0] == config["paced_orders"][1][::-1]
    for position in range(3):
        assert len({order[position] for order in config["fast_orders"]}) == 3
    for key, value in (("threads", 2), ("warmup_bins", 499), ("paced_seconds_per_block", 59)):
        with pytest.raises(ValueError):
            _validate_config({**config, key: value})


def test_full_trace_validation_includes_unscored_and_file_start_bins(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            frozen = freeze_tcn(CausalTCN(), target_scale=np.full(7, 0.01))
        onnx = tmp_path / "step.onnx"
        export_step_onnx(frozen, onnx)
        rng = np.random.default_rng(92)
        files = []
        for day, length in (("S0", 31), ("S1", 43)):
            counts = rng.poisson(3, (length, 176))
            targets = rng.normal(0, 0.01, (length, 7))
            mask = np.arange(length) % 3 != 0
            files.append(({"file_id": f"{day}_set_1", "day": day}, counts, targets, mask))
        result = validate_runtime(
            frozen,
            onnx,
            files,
            tmp_path / "validation",
            {"parity_rtol": 3e-5, "parity_atol": 1e-7, "quality_numerical_tolerance": 1e-5},
        )
        assert result["status"] == "passed"
        assert result["scores"]["onnx_incremental"]["num_bins"] == 74
        for item, counts, _, _ in files:
            saved = np.load(tmp_path / "validation" / f"{item['file_id']}.onnx_incremental.npy")
            assert saved.shape == (len(counts), 7)
    finally:
        torch.set_num_threads(old_threads)
