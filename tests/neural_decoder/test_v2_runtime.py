"""Timing arithmetic and full-window/incremental equivalence, without sleeping."""

import numpy as np
import torch

from neural_experiments.v2.accuracy import CausalStream, TemporalCore
from neural_experiments.v2.runtime import (
    WindowDecoder,
    measure,
    retained_numeric_bytes,
    summarize_timestamps,
)
from neural_experiments.v2.transfer import AdapterTCN, IncrementalDecoder


def test_window_matches_incremental_transfer_startup_and_steady_state():
    torch.manual_seed(5)
    model = AdapterTCN(8, 4, (1, 2, 4))
    stats = np.zeros(8, np.float32), np.ones(8, np.float32)
    window = WindowDecoder(model, stats, "transfer")
    incremental = IncrementalDecoder(model, stats)
    for row in np.random.default_rng(5).poisson(1.0, (90, 8)).astype(np.float32):
        np.testing.assert_allclose(window.step(row), incremental.step(row), atol=1e-6, rtol=1e-5)
    assert len(window.rows) == 15
    assert window.state_nbytes == 15 * 8 * 4


def test_window_matches_incremental_h1_and_real_prefix_startup():
    torch.manual_seed(5)
    model = TemporalCore(8, 3, (1, 2, 4), 4)
    stats = {
        "input_mean": np.ones(8, np.float32),
        "input_scale": np.ones(8, np.float32),
        "target_mean": np.ones(3, np.float32),
        "target_scale": np.ones(3, np.float32),
    }
    window = WindowDecoder(model, stats, "h1")
    incremental = CausalStream(model, stats)
    for row in np.random.default_rng(5).poisson(1.0, (90, 8)).astype(np.float32):
        np.testing.assert_allclose(window.step(row), incremental.step(row), atol=1e-6, rtol=1e-5)


def test_timing_definitions_separate_queue_service_and_completion():
    stamps = np.array(
        [[0, 1_000_000, 3_000_000, 20_000_000], [20_000_000, 22_000_000, 45_000_000, 40_000_000]],
        dtype=np.int64,
    )
    metrics = summarize_timestamps(stamps)
    assert metrics["deadline_misses"] == 1
    assert metrics["service_max_ms"] == 23
    assert metrics["negative_queue_count"] == 0
    assert metrics["service_p99_ms"] < metrics["release_to_completion_p99_ms"]


def test_fast_measure_resets_at_every_source_boundary_and_wrap():
    class Accumulator:
        def __init__(self):
            self.reset()

        def reset(self):
            self.value = 0.0

        def step(self, value):
            self.value += value[0]
            return np.array([self.value])

    stamps, values, _ = measure(
        Accumulator, np.ones((4, 1)), {0, 2}, paced=False, count=10, warmup=3
    )
    np.testing.assert_array_equal(values.ravel(), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    assert np.all(stamps[:, 2] >= stamps[:, 1])


def test_persistent_state_counts_owned_storage_behind_views_once():
    owner = np.zeros((3, 7), dtype=np.float32)
    assert retained_numeric_bytes([owner[1:], owner[2:]]) == owner.nbytes
