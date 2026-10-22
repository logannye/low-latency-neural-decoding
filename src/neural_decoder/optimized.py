"""Same-weight, explicit-history TCN inference in a standard optimized runtime.

No compilation or provider download occurs here. Export requires the installed
onnx package; execution requires an installed ONNX Runtime CPU wheel.
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .temporal import FrozenTCN

STATE_SHAPES = ((2, 176), (4, 32), (8, 32))
INPUT_NAMES = ("counts", "history0", "history1", "history2")
OUTPUT_NAMES = ("prediction", "next_history0", "next_history1", "next_history2")


def model_fingerprint(frozen: FrozenTCN) -> str:
    """Hash exact named numeric contents, independent of NPZ ZIP timestamps."""
    digest = hashlib.sha256()
    for name, value in sorted(frozen.arrays.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


class ExplicitTCNStep(nn.Module):
    """One bin plus chronological layer histories in; prediction and histories out.

    Missing latent history is zero, not bias-induced synthetic prehistory. Each
    matrix uses kernel-time-major columns so its vector concatenates old/mid/new
    taps. This is algebraically the existing cross-correlation convolution.
    """

    def __init__(self, frozen: FrozenTCN):
        super().__init__()
        for name, value in frozen.arrays.items():
            if name == "format_version":
                continue
            tensor = torch.from_numpy(value.copy())
            if name.startswith("conv") and name.endswith("weight"):
                tensor = tensor.permute(0, 2, 1).reshape(32, -1).contiguous()
            self.register_buffer(name, tensor)

    def forward(self, counts, history0, history1, history2):
        current = (counts - self.input_mean) / self.input_scale
        sequence0 = torch.cat((history0, current.unsqueeze(0)), dim=0)
        hidden0 = torch.relu(self.conv1_weight @ sequence0.flatten() + self.conv1_bias)
        sequence1 = torch.cat((history1, hidden0.unsqueeze(0)), dim=0)
        taps1 = torch.stack((sequence1[0], sequence1[2], sequence1[4]))
        hidden1 = torch.relu(self.conv2_weight @ taps1.flatten() + self.conv2_bias)
        sequence2 = torch.cat((history2, hidden1.unsqueeze(0)), dim=0)
        taps2 = torch.stack((sequence2[0], sequence2[4], sequence2[8]))
        hidden2 = torch.relu(self.conv3_weight @ taps2.flatten() + self.conv3_bias)
        prediction = self.head_weight @ hidden2 + self.head_bias
        prediction = prediction * self.target_scale + self.target_mean
        return prediction, sequence0[1:], sequence1[1:], sequence2[1:]


def export_step_onnx(frozen: FrozenTCN, path: str | Path) -> dict:
    """Export opset-17 float32 inference without pickle or an implicit native build."""
    started = time.perf_counter_ns()
    import onnx

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    model = ExplicitTCNStep(frozen).eval()
    inputs = (torch.zeros(176), *(torch.zeros(shape) for shape in STATE_SHAPES))
    buffer = io.BytesIO()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            buffer,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    graph = onnx.load_model_from_string(buffer.getvalue())
    metadata = {
        "neural_decoder_format": "explicit_tcn_step_v1",
        "frozen_fingerprint": model_fingerprint(frozen),
        "precision": "float32",
        "state_contract": "chronological_input_histories_2x176_4x32_8x32",
    }
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph, full_check=True)
    serialized = graph.SerializeToString()
    with path.open("xb") as handle:
        handle.write(serialized)
    return {
        **metadata,
        "onnx_sha256": hashlib.sha256(serialized).hexdigest(),
        "onnx_bytes": len(serialized),
        "opset": 17,
        "export_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


class ONNXTCNStream:
    """CPU-only, batch-one stateful execution with fixed shapes and one thread.

    All affine transforms and layer/state updates are inside the exported graph.
    Host conversion, shape/finite checks, state replacement and output validation
    are part of step. Python/ORT allocations remain; no allocation-free claim.
    """

    def __init__(
        self, path: str | Path, *, expected_model_identity: str, precision: str = "float32"
    ):
        import onnxruntime as ort

        if precision not in {"float32", "dynamic_int8"}:
            raise ValueError("unsupported ONNX precision contract")
        path = Path(path)
        if path.stat().st_size > 2_000_000:
            raise ValueError("explicit TCN graph exceeds its storage bound")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # A steady single-user stream should not dedicate a core to spinning between bins.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("precision") != precision:
            raise ValueError("ONNX precision differs from explicitly requested precision")
        if metadata.get("neural_decoder_format") != "explicit_tcn_step_v1":
            raise ValueError("unsupported explicit TCN format")
        if metadata.get("frozen_fingerprint") != expected_model_identity:
            raise ValueError("ONNX weights do not match the expected frozen model identity")
        expected_inputs = ((176,), *STATE_SHAPES)
        expected_outputs = ((7,), *STATE_SHAPES)
        for nodes, names, shapes in (
            (self.session.get_inputs(), INPUT_NAMES, expected_inputs),
            (self.session.get_outputs(), OUTPUT_NAMES, expected_outputs),
        ):
            if len(nodes) != len(names):
                raise ValueError("unexpected ONNX interface")
            for node, name, shape in zip(nodes, names, shapes, strict=True):
                if node.name != name or tuple(node.shape) != shape or node.type != "tensor(float)":
                    raise ValueError("incompatible ONNX input/output schema")
        self.model_identity = expected_model_identity
        self.precision = precision
        self.graph_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.reset()

    def reset(self):
        self._state = tuple(np.zeros(shape, dtype=np.float32) for shape in STATE_SHAPES)
        self._failed = False

    @property
    def state_nbytes(self):
        return sum(value.nbytes for value in self._state)

    def state_snapshot(self):
        """Copies for correctness inspection, not a portable stream checkpoint."""
        return tuple(value.copy() for value in self._state)

    def step(self, counts):
        if self._failed:
            raise RuntimeError("ONNX stream had an arithmetic failure; reset before reuse")
        counts = np.asarray(counts)
        if counts.shape != (176,) or counts.dtype.kind not in "fiu":
            raise ValueError("expected a numeric feature vector with shape (176,)")
        if not np.isfinite(counts).all():
            raise ValueError("features must be finite")
        try:
            with np.errstate(over="raise", invalid="raise"):
                current = counts.astype(np.float32)
            feed = dict(zip(INPUT_NAMES, (current, *self._state), strict=True))
            outputs = self.session.run(list(OUTPUT_NAMES), feed)
            if any(not np.isfinite(value).all() for value in outputs):
                raise FloatingPointError("nonfinite ONNX prediction or state")
            self._state = tuple(outputs[1:])
            return outputs[0].copy()
        except (FloatingPointError, RuntimeError) as error:
            self._failed = True
            raise FloatingPointError("ONNX arithmetic failed; reset before reuse") from error

    def predict_chunk(self, counts):
        counts = np.asarray(counts)
        if counts.ndim != 2 or counts.shape[1] != 176 or counts.dtype.kind not in "fiu":
            raise ValueError("expected a numeric feature chunk [time,176]")
        if not np.isfinite(counts).all():
            raise ValueError("features must be finite")
        if not len(counts):
            return np.empty((0, 7), dtype=np.float32)
        return np.stack([self.step(row) for row in counts])
