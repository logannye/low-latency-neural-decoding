"""Optional portable ONNX history checkpoint with stream/sequence identity.

This is separate from inference weights and training checkpoints. Only feature
history and consumed input position are stored; labels and masks cannot enter.
"""

from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from .optimized import STATE_SHAPES, ONNXTCNStream

_FIELDS = {
    "schema_version",
    "model_identity",
    "graph_sha256",
    "precision",
    "stream_id",
    "next_sequence_id",
    "last_sequence_id",
    "last_timestamp_ns",
    "history0",
    "history1",
    "history2",
}


def _text(value):
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode(value):
    if value.dtype != np.uint8 or value.ndim != 1 or len(value) > 256:
        raise ValueError("invalid snapshot text field")
    try:
        return value.tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("invalid UTF-8 snapshot field") from error


def _integer(value):
    if value.dtype != np.int64 or value.shape != ():
        raise ValueError("snapshot position must be scalar int64")
    return value.item()


class CheckpointedONNXStream:
    """Add strict incoming sequence/time checks and explicit checkpoint/restore.

    Sequence IDs begin at zero for each file and increase by one. Timestamps are
    integer nanoseconds and strictly increase; recorded nonuniform gaps are kept.
    Snapshot I/O is explicit, outside ordinary prediction. The numeric runtime
    benchmark excludes this optional deployment metadata wrapper.
    """

    def __init__(self, decoder: ONNXTCNStream, *, stream_id: str):
        if not isinstance(decoder, ONNXTCNStream):
            raise TypeError("expected ONNXTCNStream")
        self.decoder = decoder
        self.reset(stream_id=stream_id)

    def reset(self, *, stream_id):
        if not isinstance(stream_id, str) or not stream_id or len(stream_id.encode()) > 256:
            raise ValueError("stream identity must be nonempty UTF-8 text up to256 bytes")
        self.decoder.reset()
        self.stream_id = stream_id
        self.next_sequence_id = 0
        self.last_timestamp_ns = -1

    def step(self, counts, *, sequence_id, timestamp_ns):
        for value in (sequence_id, timestamp_ns):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("sequence and timestamp must be integers")
            if value < 0 or value >= np.iinfo(np.int64).max:
                raise ValueError("sequence/timestamp outside supported int64 range")
        if sequence_id != self.next_sequence_id:
            raise ValueError("missing, duplicate or out-of-order input sequence")
        if timestamp_ns <= self.last_timestamp_ns:
            raise ValueError("timestamps must strictly increase")
        counts = np.asarray(counts)
        if counts.dtype.kind not in "fiu" or (counts < 0).any():
            raise ValueError("counts must be numeric and nonnegative")
        prediction = self.decoder.step(counts)
        self.next_sequence_id += 1
        self.last_timestamp_ns = int(timestamp_ns)
        return prediction

    def snapshot(self, path):
        if self.decoder._failed:
            raise ValueError("cannot checkpoint an arithmetic-failed stream")
        arrays = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "model_identity": _text(self.decoder.model_identity),
            "graph_sha256": _text(self.decoder.graph_sha256),
            "precision": _text(self.decoder.precision),
            "stream_id": _text(self.stream_id),
            "next_sequence_id": np.asarray(self.next_sequence_id, dtype=np.int64),
            "last_sequence_id": np.asarray(self.next_sequence_id - 1, dtype=np.int64),
            "last_timestamp_ns": np.asarray(self.last_timestamp_ns, dtype=np.int64),
            **{f"history{i}": value for i, value in enumerate(self.decoder.state_snapshot())},
        }
        with Path(path).open("xb") as handle:
            np.savez(handle, **arrays)

    def restore(self, path, *, expected_stream_id, expected_next_sequence_id):
        """Validate the entire artifact before replacing any active stream state."""
        try:
            with ZipFile(path) as archive:
                entries = archive.infolist()
                if (
                    len(entries) != len(_FIELDS)
                    or {v.filename for v in entries} != {f"{name}.npy" for name in _FIELDS}
                    or sum(v.file_size for v in entries) > 32_768
                ):
                    raise ValueError("invalid snapshot archive schema or size")
            with np.load(path, allow_pickle=False, max_header_size=4096) as archive:
                arrays = {name: archive[name] for name in archive.files}
        except (BadZipFile, EOFError, OSError) as error:
            raise ValueError("unreadable stream snapshot") from error
        if _integer(arrays["schema_version"]) != 1:
            raise ValueError("unsupported snapshot version")
        expected = {
            "model_identity": self.decoder.model_identity,
            "graph_sha256": self.decoder.graph_sha256,
            "precision": self.decoder.precision,
            "stream_id": expected_stream_id,
        }
        for field, value in expected.items():
            if _decode(arrays[field]) != value:
                raise ValueError(f"snapshot {field} identity mismatch")
        if not isinstance(expected_stream_id, str) or not expected_stream_id:
            raise ValueError("expected stream identity required")
        sequence = _integer(arrays["next_sequence_id"])
        last_sequence = _integer(arrays["last_sequence_id"])
        timestamp = _integer(arrays["last_timestamp_ns"])
        if sequence < 0 or sequence != expected_next_sequence_id or last_sequence != sequence - 1:
            raise ValueError("snapshot next sequence disagrees with expected input position")
        if (sequence == 0 and timestamp != -1) or (sequence > 0 and timestamp < 0):
            raise ValueError("inconsistent snapshot timestamp")
        state = []
        for index, shape in enumerate(STATE_SHAPES):
            value = arrays[f"history{index}"]
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError("invalid snapshot history shape/dtype/values")
            if sequence == 0 and np.any(value):
                raise ValueError("empty stream must have zero history")
            state.append(value.copy())
        self.decoder._state = tuple(state)
        self.decoder._failed = False
        self.stream_id = expected_stream_id
        self.next_sequence_id = sequence
        self.last_timestamp_ns = timestamp
