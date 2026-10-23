import ast
import inspect

import numpy as np
import pytest

from neural_experiments.v7 import audit_incremental as audit
from neural_experiments.v7.incremental import CenteredRidge


def moments():
    rng = np.random.default_rng(81)
    raw = rng.normal(size=(2000, 4))
    y = rng.normal(size=(2000, 2))
    mean, scale = raw[:1500].mean(0), raw[:1500].std(0)
    selected = np.zeros(2000, bool)
    selected[1500:] = rng.random(500) > 0.3
    state = CenteredRidge(mean, scale)
    for a in range(1500, 2000, 64):
        b = min(a + 64, 2000)
        state.add(
            raw[a:b],
            y[a:b],
            np.arange(a, b),
            selected[a:b],
            np.ones(b - a, bool),
            np.zeros(b - a, np.int64),
            observed_stop=b,
        )
    expected = audit.moments_reference(raw, y, selected, 2000, mean, scale)
    return state, expected


def test_actual_chunk_merges_match_independent_batch_moments():
    state, expected = moments()
    assert audit.validate_moments(state.snapshot(), expected) < 1e-10
    assert state.moment_nbytes == sum(
        v.nbytes for k, v in expected.items() if not k.startswith("normalizer")
    )


@pytest.mark.parametrize(
    "key", ["centered_gram", "centered_rhs", "count", "last_bin", "normalizer_mean"]
)
def test_altered_numeric_state_rejected(key):
    state, expected = moments()
    changed = {k: v.copy() for k, v in state.snapshot().items()}
    changed[key].flat[0] += 1
    with pytest.raises(ValueError):
        audit.validate_moments(changed, expected)


def geometry():
    stops = [1564, 1628, 1692, 1756, 1820]
    updates = np.array(
        [[i * 10000, i * 10000 + 1000, i * 5000, i * 5000 + 500] for i in range(5)], np.int64
    )
    solves = np.array(
        [
            [
                i * 10000 + 1500,
                i * 10000 + 2500,
                i * 10000 + 3500,
                i * 5000 + 1000,
                i * 5000 + 2000,
                i * 5000 + 3000,
            ]
            for i in range(5)
        ],
        np.int64,
    )
    paid = np.zeros(1820, bool)
    paid[1500:] = True
    rows = np.array(
        [[1500 + 64 * i, 1564 + 64 * i, 64 * i, 64 * (i + 1)] for i in range(5)], np.int64
    )
    return rows, updates, solves, paid, paid.copy(), stops


def test_raw_update_solve_order_and_clock_summary():
    audit.update_geometry_reference(*geometry())
    summary = audit.clock_summary(geometry()[1], 0, 1)
    assert summary == {"n": 5, "p50_ms": 0.001, "p99_ms": 0.001, "max_ms": 0.001, "total_ms": 0.005}


@pytest.mark.parametrize("tamper", ["duplicate", "future_count", "duration", "overlap", "unpaid"])
def test_clock_or_paid_row_mutations_rejected(tamper):
    rows, updates, solves, paid, eligible, stops = geometry()
    if tamper == "duplicate":
        rows[1, 0] -= 1
    elif tamper == "future_count":
        rows[1, 3] += 1
    elif tamper == "duration":
        updates[1, 1] = updates[1, 0] - 1
    elif tamper == "overlap":
        solves[1, 0] = updates[1, 1] - 1
    else:
        paid[1500] = False
    with pytest.raises(ValueError):
        audit.update_geometry_reference(rows, updates, solves, paid, eligible, stops)


def test_independent_engineering_auditor_imports_no_producer():
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "neural_experiments"
        ):
            assert "audit" in node.module.split(".")[-1]
