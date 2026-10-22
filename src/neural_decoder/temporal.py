"""A fixed causal TCN and two feature-only, same-weight streaming implementations.

Training uses standardized features/targets. Exported streaming predictors include
the frozen affine transforms and consume raw features. File boundaries, clocks,
channel identity and scorer-only data belong to the outer replay contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUT_CHANNELS = 176
HIDDEN_CHANNELS = 32
OUTPUT_CHANNELS = 7
DILATIONS = (1, 2, 4)
KERNEL_SIZE = 3
RECEPTIVE_FIELD = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)
FORMAT_VERSION = 1

_SHAPES = {
    "conv1_weight": (32, 176, 3),
    "conv1_bias": (32,),
    "conv2_weight": (32, 32, 3),
    "conv2_bias": (32,),
    "conv3_weight": (32, 32, 3),
    "conv3_bias": (32,),
    "head_weight": (7, 32),
    "head_bias": (7,),
    "input_mean": (176,),
    "input_scale": (176,),
    "target_mean": (7,),
    "target_scale": (7,),
}
_STATE_NAMES = {
    **{f"conv{i + 1}_{kind}": f"convs.{i}.{kind}" for i in range(3) for kind in ("weight", "bias")},
    "head_weight": "head.weight",
    "head_bias": "head.bias",
}


class CausalTCN(nn.Module):
    """Three single-convolution ReLU layers; [batch, time, channels] layout.

    There is no dropout, batch normalization, lookahead or implicit stream state.
    Zero padding is in standardized feature/activation space at each layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(cin, HIDDEN_CHANNELS, KERNEL_SIZE, dilation=dilation)
                for cin, dilation in zip((INPUT_CHANNELS, 32, 32), DILATIONS, strict=True)
            ]
        )
        self.head = nn.Linear(HIDDEN_CHANNELS, OUTPUT_CHANNELS)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[2] != INPUT_CHANNELS:
            raise ValueError("expected [batch, time, 176] standardized features")
        if features.shape[1] < 1:
            raise ValueError("a forward pass requires at least one bin")
        hidden = features.transpose(1, 2)
        for conv, dilation in zip(self.convs, DILATIONS, strict=True):
            hidden = F.relu(conv(F.pad(hidden, ((KERNEL_SIZE - 1) * dilation, 0))))
        return self.head(hidden.transpose(1, 2))


@dataclass(frozen=True)
class FrozenTCN:
    """Validated immutable float32 inference weights and train-fitted transforms."""

    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        expected = set(_SHAPES) | {"format_version"}
        if set(self.arrays) != expected:
            raise ValueError("TCN artifact has missing or unexpected arrays")
        version = np.asarray(self.arrays["format_version"])
        if version.shape != () or version.dtype != np.dtype("int64"):
            raise ValueError("format_version must be a scalar int64")
        if version.item() != FORMAT_VERSION:
            raise ValueError("unsupported TCN artifact version")
        checked = {}
        for name, shape in _SHAPES.items():
            array = np.asarray(self.arrays[name])
            if array.shape != shape or array.dtype != np.dtype("float32"):
                raise ValueError(f"{name} must have float32 dtype and shape {shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains nonfinite values")
            if name.endswith("_scale") and not (array > 0).all():
                raise ValueError(f"{name} must be strictly positive")
            checked[name] = np.array(array, copy=True, order="C")
        checked["format_version"] = version.copy()
        for array in checked.values():
            array.setflags(write=False)
        object.__setattr__(self, "arrays", MappingProxyType(checked))

    def to_torch(self) -> CausalTCN:
        # Reconstructing inference weights must not perturb an ongoing training RNG.
        with torch.random.fork_rng(devices=[]):
            model = CausalTCN()
        model.load_state_dict(
            {
                state_name: torch.from_numpy(self.arrays[name].copy())
                for name, state_name in _STATE_NAMES.items()
            }
        )
        return model.eval()

    @property
    def parameter_count(self) -> int:
        return sum(self.arrays[name].size for name in _STATE_NAMES)


def freeze_tcn(
    model: CausalTCN,
    *,
    input_mean: np.ndarray | None = None,
    input_scale: np.ndarray | None = None,
    target_mean: np.ndarray | None = None,
    target_scale: np.ndarray | None = None,
) -> FrozenTCN:
    """Copy an inference artifact; omitted affine terms give identity transforms."""
    if not isinstance(model, CausalTCN):
        raise TypeError("expected CausalTCN")
    state = model.state_dict()
    arrays = {
        name: state[state_name].detach().cpu().numpy().copy()
        for name, state_name in _STATE_NAMES.items()
    }
    for name, value, width, default in (
        ("input_mean", input_mean, INPUT_CHANNELS, 0),
        ("input_scale", input_scale, INPUT_CHANNELS, 1),
        ("target_mean", target_mean, OUTPUT_CHANNELS, 0),
        ("target_scale", target_scale, OUTPUT_CHANNELS, 1),
    ):
        arrays[name] = (
            np.full(width, default, dtype=np.float32)
            if value is None
            else np.asarray(value, dtype=np.float32)
        )
    arrays["format_version"] = np.asarray(FORMAT_VERSION, dtype=np.int64)
    return FrozenTCN(arrays)


def export_tcn(model: CausalTCN, path: str | Path, **transforms: np.ndarray) -> FrozenTCN:
    """Write a non-pickle NPZ to a new path, never replace earlier evidence."""
    frozen = freeze_tcn(model, **transforms)
    with Path(path).open("xb") as handle:
        np.savez(handle, **frozen.arrays)
    return frozen


def load_tcn(path: str | Path) -> FrozenTCN:
    """Reject malformed schemas, objects, nonfinite weights and oversized archives."""
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            expected = {f"{name}.npy" for name in (*_SHAPES, "format_version")}
            if len(entries) != len(expected) or {entry.filename for entry in entries} != expected:
                raise ValueError("TCN archive has missing, duplicate or unexpected entries")
            if sum(entry.file_size for entry in entries) > 1_000_000:
                raise ValueError("TCN archive exceeds its fixed format storage bound")
        with np.load(path, allow_pickle=False, max_header_size=4096) as archive:
            return FrozenTCN({name: archive[name] for name in archive.files})
    except (BadZipFile, EOFError, OSError) as error:
        raise ValueError("unreadable TCN archive") from error


class TCNStream:
    """One independent feature stream with a reset at each new recording file.

    ``window`` repeats a PyTorch forward over up to 15 observed bins. During
    startup it uses the actual prefix length, so bias-induced hidden activations
    are never invented before the first bin. ``incremental`` caches each layer's
    inputs in rings initialized to zero, including missing latent history.

    Trial boundaries and score masks do not change this state. This is a numeric
    inference engine; the caller must validate channel identities/times/counts.
    The Python implementation allocates scratch arrays and makes no allocation-
    free or real-time deadline guarantee.
    """

    def __init__(
        self,
        frozen: FrozenTCN,
        implementation: Literal["incremental", "window"] = "incremental",
    ) -> None:
        if not isinstance(frozen, FrozenTCN):
            raise TypeError("expected validated FrozenTCN")
        if implementation not in ("incremental", "window"):
            raise ValueError("implementation must be incremental or window")
        self.frozen = frozen
        self.implementation = implementation
        self._model = frozen.to_torch() if implementation == "window" else None
        self._window = np.zeros((RECEPTIVE_FIELD, INPUT_CHANNELS), dtype=np.float32)
        self._rings = [
            np.zeros((1 + 2 * dilation, width), dtype=np.float32)
            for dilation, width in zip(DILATIONS, (INPUT_CHANNELS, 32, 32), strict=True)
        ]
        self.reset()

    def reset(self) -> None:
        """Start a new file; do not call this for trials, chunks or unscored bins."""
        self._window.fill(0)
        for ring in self._rings:
            ring.fill(0)
        self._cursors = [0, 0, 0]
        self._warm_bins = 0
        self._failed = False

    @property
    def state_nbytes(self) -> int:
        """Persistent numeric array bytes; excludes weights, scratch and Python overhead."""
        return self._window.nbytes + sum(ring.nbytes for ring in self._rings)

    def state_snapshot(self) -> tuple:
        """Copies for causality/isolation audits, not a portable restore artifact."""
        return (
            self._warm_bins,
            tuple(self._cursors),
            self._window.copy(),
            tuple(ring.copy() for ring in self._rings),
        )

    def step(self, features: np.ndarray) -> np.ndarray:
        """Consume exactly one bin and return seven values in original target units."""
        if self._failed:
            raise RuntimeError("TCN stream had an arithmetic failure; reset before reuse")
        raw = np.asarray(features)
        if raw.shape != (INPUT_CHANNELS,) or raw.dtype.kind not in "fiu":
            raise ValueError("expected a numeric feature vector with shape (176,)")
        if not np.isfinite(raw).all():
            raise ValueError("features must be finite")
        a = self.frozen.arrays
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                current = (raw.astype(np.float32) - a["input_mean"]) / a["input_scale"]
                normalized = self._evaluate(current)
                output = normalized * a["target_scale"] + a["target_mean"]
            if not np.isfinite(output).all():
                raise FloatingPointError("nonfinite TCN output")
        except (FloatingPointError, RuntimeError) as error:
            self._failed = True
            raise FloatingPointError(
                "TCN arithmetic failed; reset before reusing stream"
            ) from error
        return output.copy()

    def _evaluate(self, current: np.ndarray) -> np.ndarray:
        a = self.frozen.arrays
        if self.implementation == "window":
            self._window[:-1] = self._window[1:]
            self._window[-1] = current
            self._warm_bins = min(RECEPTIVE_FIELD, self._warm_bins + 1)
            observed = self._window[-self._warm_bins :].copy()
            with torch.inference_mode():
                return self._model(torch.from_numpy(observed)[None])[0, -1].numpy()
        for i, (ring, dilation) in enumerate(zip(self._rings, DILATIONS, strict=True)):
            cursor = self._cursors[i]
            ring[cursor] = current
            hidden = a[f"conv{i + 1}_bias"].copy()
            weight = a[f"conv{i + 1}_weight"]
            for tap in range(KERNEL_SIZE):
                lag = (KERNEL_SIZE - 1 - tap) * dilation
                hidden += weight[:, :, tap] @ ring[(cursor - lag) % len(ring)]
            current = np.maximum(hidden, np.float32(0))
            self._cursors[i] = (cursor + 1) % len(ring)
        self._warm_bins = min(RECEPTIVE_FIELD, self._warm_bins + 1)
        return a["head_weight"] @ current + a["head_bias"]

    def predict_chunk(self, features: np.ndarray) -> np.ndarray:
        """Replay an ordered host chunk without resetting at its boundary."""
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != INPUT_CHANNELS:
            raise ValueError("expected a feature chunk with shape [time, 176]")
        if features.dtype.kind not in "fiu" or not np.isfinite(features).all():
            raise ValueError("feature chunks must be numeric and finite")
        if len(features) == 0:
            return np.empty((0, OUTPUT_CHANNELS), dtype=np.float32)
        return np.stack([self.step(row) for row in features])
