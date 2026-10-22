"""Portable explicit-state pause/resume and corrupt-checkpoint rejection."""

import numpy as np
import pytest
import torch

from neural_decoder.optimized import ONNXTCNStream, export_step_onnx, model_fingerprint
from neural_decoder.snapshot import CheckpointedONNXStream
from neural_decoder.temporal import CausalTCN, freeze_tcn


@pytest.fixture
def factory(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(79)
        frozen = freeze_tcn(CausalTCN())
    path = tmp_path / "model.onnx"
    export_step_onnx(frozen, path)
    return lambda: CheckpointedONNXStream(
        ONNXTCNStream(path, expected_model_identity=model_fingerprint(frozen)), stream_id="file-one"
    )


def test_pause_restored_suffix_matches_uninterrupted_exactly(factory, tmp_path):
    counts = np.random.default_rng(32).poisson(3, (61, 176))
    continuous = factory()
    expected = np.array(
        [
            continuous.step(row, sequence_id=i, timestamp_ns=i * 20_000_000)
            for i, row in enumerate(counts)
        ]
    )
    partial = factory()
    for i, row in enumerate(counts[:23]):
        partial.step(row, sequence_id=i, timestamp_ns=i * 20_000_000)
    path = tmp_path / "state.npz"
    partial.snapshot(path)
    restored = factory()
    restored.restore(path, expected_stream_id="file-one", expected_next_sequence_id=23)
    actual = np.array(
        [
            restored.step(counts[i], sequence_id=i, timestamp_ns=i * 20_000_000)
            for i in range(23, len(counts))
        ]
    )
    np.testing.assert_array_equal(actual, expected[23:])
    assert restored.next_sequence_id == len(counts)
    assert restored.last_timestamp_ns == 60 * 20_000_000
    for a, b in zip(
        restored.decoder.state_snapshot(), continuous.decoder.state_snapshot(), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(FileExistsError):
        partial.snapshot(path)


@pytest.mark.parametrize(
    "field",
    [
        "model_identity",
        "graph_sha256",
        "stream_id",
        "precision",
        "history0",
        "history1",
        "last_sequence_id",
        "last_timestamp_ns",
    ],
)
def test_corrupt_state_does_not_mutate_active_stream(factory, tmp_path, field):
    original = factory()
    original.step(np.ones(176), sequence_id=0, timestamp_ns=0)
    source = tmp_path / "state.npz"
    original.snapshot(source)
    with np.load(source, allow_pickle=False) as contents:
        arrays = {name: contents[name] for name in contents.files}
    if field in ("model_identity", "graph_sha256", "stream_id", "precision"):
        arrays[field] = np.frombuffer(b"wrong-identity", dtype=np.uint8)
    elif field == "history0":
        arrays[field] = np.zeros((3, 176), dtype=np.float32)
    elif field == "history1":
        arrays[field][0, 0] = np.nan
    else:
        arrays[field] = np.asarray(-2, dtype=np.int64)
    broken = tmp_path / "broken.npz"
    np.savez(broken, **arrays)
    restored = factory()
    with pytest.raises(ValueError):
        restored.restore(broken, expected_stream_id="file-one", expected_next_sequence_id=1)
    assert restored.next_sequence_id == 0
    assert all(not np.any(a) for a in restored.decoder.state_snapshot())


def test_restore_requires_exact_next_sequence_and_monotonic_time(factory, tmp_path):
    stream = factory()
    stream.step(np.ones(176), sequence_id=0, timestamp_ns=100)
    checkpoint = tmp_path / "state.npz"
    stream.snapshot(checkpoint)
    restored = factory()
    with pytest.raises(ValueError, match="sequence"):
        restored.restore(checkpoint, expected_stream_id="file-one", expected_next_sequence_id=2)
    restored.restore(checkpoint, expected_stream_id="file-one", expected_next_sequence_id=1)
    for sequence, timestamp in ((0, 120), (2, 120), (1, 100), (1, 99)):
        with pytest.raises(ValueError):
            restored.step(np.ones(176), sequence_id=sequence, timestamp_ns=timestamp)
    assert restored.next_sequence_id == 1
    # A real nonuniform gap is retained; there is no invented zero-count bin.
    assert np.isfinite(restored.step(np.ones(176), sequence_id=1, timestamp_ns=1000)).all()


def test_empty_stream_can_resume_before_first_bin(factory, tmp_path):
    original = factory()
    path = tmp_path / "empty.npz"
    original.snapshot(path)
    resumed = factory()
    resumed.restore(path, expected_stream_id="file-one", expected_next_sequence_id=0)
    np.testing.assert_array_equal(
        original.step(np.zeros(176), sequence_id=0, timestamp_ns=0),
        resumed.step(np.zeros(176), sequence_id=0, timestamp_ns=0),
    )
