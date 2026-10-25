"""Synthetic metadata, cadence, checkpoint and poisoned-state contracts."""

import json

import numpy as np
import pytest
import torch

from neural_decoder.optimized import ONNXTCNStream, export_step_onnx, model_fingerprint
from neural_decoder.temporal import CausalTCN, freeze_tcn
from neural_experiments.h1_final_closure_recovery import (
    CadencedONNXStream,
    channel_signature,
    check_numpy,
    check_onnx_lifecycle,
)


@pytest.fixture
def model(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(624)
        frozen = freeze_tcn(CausalTCN())
    path = tmp_path / "model.onnx"
    export_step_onnx(frozen, path)
    return frozen, lambda: ONNXTCNStream(path, expected_model_identity=model_fingerprint(frozen))


def test_full_lifecycle_contract_and_prior_snapshot_preserved(model, tmp_path):
    frozen, factory = model
    counts = np.random.default_rng(7).poisson(2, (512, 176))
    report = check_onnx_lifecycle(factory, counts, np.arange(176), tmp_path / "lifecycle")
    assert report["status"] == "passed"
    assert len(report["checks"]) >= 15
    assert all(report["checks"].values())
    assert check_numpy(frozen, counts)["state_exact"]


def test_cadence_restore_validation_is_atomic(model, tmp_path):
    _, factory = model
    channels = np.arange(176)
    stream = CadencedONNXStream(factory(), stream_id="run/0/file", channel_ids=channels)
    stream.step(np.ones(176), sequence_id=0, timestamp_ns=0, channel_ids=channels)
    path = tmp_path / "state.npz"
    stream.snapshot(path)
    before = stream.decoder.state_snapshot()
    sidecar = path.with_suffix(".npz.json")
    metadata = json.loads(sidecar.read_text())
    metadata["period_ns"] += 1
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata"):
        stream.restore(
            path, expected_stream_id="run/0/file", expected_next_sequence_id=1, channel_ids=channels
        )
    for a, b in zip(before, stream.decoder.state_snapshot(), strict=True):
        np.testing.assert_array_equal(a, b)
    assert stream.wrapper.next_sequence_id == 1


def test_channel_map_requires_unique_integer_identity():
    with pytest.raises(ValueError):
        channel_signature(np.zeros(176, dtype=np.int64))
    with pytest.raises(ValueError):
        channel_signature(np.arange(176, dtype=float))
    assert channel_signature(np.arange(176)) != channel_signature(np.arange(176)[::-1])


def test_independent_saved_evidence_rejects_finite_prediction_and_clock_mutations(model, tmp_path):
    from neural_experiments.h1_final_closure_recovery import (
        audit_saved_traces,
        overhead,
        stall_check,
    )

    _, factory = model
    counts = np.random.default_rng(85).poisson(1, (3001, 176))
    channels = np.arange(176)
    folder = tmp_path / "independent"
    lifecycle = check_onnx_lifecycle(factory, counts, channels, folder)
    timing = overhead(factory, counts, channels, folder)
    stall = stall_check(factory(), counts, folder)
    graph = tmp_path / "model.onnx"
    assert audit_saved_traces(folder, lifecycle, timing, stall, counts, graph)["status"] == "passed"
    path = folder / "uninterrupted.npy"
    original = np.load(path)
    changed = original.copy()
    changed[3, 2] += 0.001
    np.save(path, changed)
    with pytest.raises(AssertionError):
        audit_saved_traces(folder, lifecycle, timing, stall, counts, graph)
    np.save(path, original)
    path = folder / "injected-stall.npz"
    with np.load(path) as saved:
        timestamps, predictions = saved["timestamps"], saved["predictions"]
    timestamps[5, 0] += 1
    np.savez(path, timestamps=timestamps, predictions=predictions)
    with pytest.raises(ValueError, match="schedule"):
        audit_saved_traces(folder, lifecycle, timing, stall, counts, graph)
