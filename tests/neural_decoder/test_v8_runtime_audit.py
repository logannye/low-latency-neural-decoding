"""Mutation tests for independent raw timing and numerical replay verification."""

import ast
import inspect

import numpy as np
import pytest

from neural_experiments.v8 import audit_runtime


def raw_block():
    n = audit_runtime.COUNT
    release = np.arange(n, dtype=np.int64) * audit_runtime.PERIOD + 1_000_000_000
    queue = (np.arange(n) % 17) * 1000
    service = (np.arange(n) % 31) * 1000 + 100_000
    stamps = np.c_[
        release, release + queue, release + queue + service, release + audit_runtime.PERIOD
    ]
    prediction = np.zeros((n, 2), np.float64)
    row = audit_runtime.clock_reference(stamps)
    row.update(
        wall_start_ns=int(release[0] - audit_runtime.PERIOD),
        wall_end_ns=int(stamps[-1, 2] + 100),
        cpu_start_ns=200,
        cpu_end_ns=500000200,
        numeric_filter_state_bytes=1544,
        prediction_max_absolute_error=0.0,
    )
    row["block_wall_seconds"] = (row["wall_end_ns"] - row["wall_start_ns"]) / 1e9
    row["process_cpu_seconds"] = (row["cpu_end_ns"] - row["cpu_start_ns"]) / 1e9
    return row, {"clocks": stamps, "prediction": prediction}


def test_no_timing_or_model_producer_imports():
    for node in ast.walk(ast.parse(inspect.getsource(audit_runtime))):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in (
                "neural_experiments.v8.runtime",
                "neural_experiments.v8.model",
            )


def test_service_and_scheduler_queue_are_separate_exact_observations():
    row, arrays = raw_block()
    result = audit_runtime.block_reference(row, arrays, arrays["prediction"], 1544)
    assert result["n"] == 3000 and result["deadline_misses"] == 0
    assert result["release_to_completion_p99_ms"] > result["service_p99_ms"]
    assert result["process_cpu_seconds"] == 0.5
    assert result["independent_prediction_max_absolute_error"] == 0


@pytest.mark.parametrize(
    "tamper", ["floatclock", "earlystart", "negative", "serialoverlap", "cadence", "deadline"]
)
def test_corrupt_raw_clock_geometry_fails(tamper):
    _, arrays = raw_block()
    x = arrays["clocks"]
    if tamper == "floatclock":
        x = x.astype(float)
    elif tamper == "earlystart":
        x[0, 1] = x[0, 0] - 1
    elif tamper == "negative":
        x[0, 0] = -1
    elif tamper == "serialoverlap":
        x[0, 2] = x[1, 1] + 1
    elif tamper == "cadence":
        x[1, 0] += 1
    elif tamper == "deadline":
        x[0, 3] += 1
    with pytest.raises(ValueError, match="raw clocks"):
        audit_runtime.clock_reference(x)


@pytest.mark.parametrize(
    "key",
    [
        "service_p95_ms",
        "queue_max_ms",
        "deadline_misses",
        "process_cpu_seconds",
        "numeric_filter_state_bytes",
    ],
)
def test_summary_and_memory_tampering_fails(key):
    row, arrays = raw_block()
    row[key] += 1
    with pytest.raises(ValueError, match="runtime"):
        audit_runtime.block_reference(row, arrays, arrays["prediction"], 1544)


def test_rehashed_prediction_and_block_clock_tampering_fails():
    row, arrays = raw_block()
    expected = arrays["prediction"].copy()
    arrays["prediction"][12, 0] = 0.01
    with pytest.raises(ValueError, match="prediction replay"):
        audit_runtime.block_reference(row, arrays, expected, 1544)
    arrays["prediction"] = expected
    row["wall_end_ns"] = row["wall_start_ns"]
    with pytest.raises(ValueError, match="containment"):
        audit_runtime.block_reference(row, arrays, expected, 1544)
