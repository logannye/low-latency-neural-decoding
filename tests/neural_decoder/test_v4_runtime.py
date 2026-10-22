import numpy as np
import pytest

from neural_experiments.v4.model import Decoder
from neural_experiments.v4.runtime import filter_response, measure, timestamp_summary


class FakeClock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def perf_counter_ns(self):
        self.now += 1000
        return self.now

    def process_time_ns(self):
        return self.now // 100

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += round(seconds * 1e9)


def fixture_model():
    return Decoder(
        {
            "family": "tx45",
            "tau_seconds": 0.2,
            "baseline_tau_seconds": None,
            "adapt_gamma": 1.0,
        },
        np.zeros(2),
        np.ones(2),
        np.array([[0.1, 0.2], [0.2, -0.1], [0.0, 0.0]]),
    )


def test_paced_timestamp_semantics_warmup_exclusion_and_wrap_reset():
    model = fixture_model()
    tx = np.array([[0.0, 1.0], [4.0, 9.0], [2.0, 0.0], [1.0, 1.0]])
    features = {"x_tx3": tx, "x_tx45": tx, "x_sbp": np.zeros_like(tx)}
    clock = FakeClock()
    stamps, predictions, details = measure(
        model, features, {0, 2}, paced=True, count=9, warmup=7, clock=clock
    )
    np.testing.assert_array_equal(np.diff(stamps[:, 0]), 20_000_000)
    np.testing.assert_array_equal(stamps[:, 3] - stamps[:, 0], 20_000_000)
    assert len(clock.sleeps) == 9
    assert np.all(stamps[:, 1] >= stamps[:, 0])
    assert np.all(stamps[:, 2] >= stamps[:, 1])
    expected = model.predict(features, [(0, 2), (2, 4)])
    np.testing.assert_allclose(predictions, expected[np.arange(9) % 4], rtol=1e-12, atol=1e-12)
    assert details["persistent_numeric_state_bytes"] == 32
    assert timestamp_summary(stamps)["deadline_misses"] == 0


def test_summary_uses_release_completion_and_strict_deadline_not_sum_quantiles():
    stamps = np.array([[0, 10, 20, 30], [30, 40, 65, 60], [60, 70, 90, 90]], dtype=np.int64)
    summary = timestamp_summary(stamps)
    assert summary["deadline_misses"] == 1
    assert summary["service_max_ms"] == 25 / 1e6
    assert summary["release_to_completion_p99_ms"] == pytest.approx(34.9 / 1e6)
    assert summary["queue_p99_ms"] == 10 / 1e6


def test_response_descriptor_distinguishes_temporal_support_from_compute():
    response = filter_response(fixture_model().config)
    decay = np.exp(-0.02 / 0.2)
    assert response["fast_steady_state_mean_sample_age_seconds"] == pytest.approx(
        0.02 * decay / (1 - decay)
    )
    assert response["fast_95_percent_step_response_bins"] == 30
    assert "infinite" in response["temporal_support"]
