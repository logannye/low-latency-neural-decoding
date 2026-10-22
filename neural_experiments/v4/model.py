"""Causal V4 features, equal-session history ridge, and prior-centered adaptation.

The caller owns chronological history selection and calibration boundaries.
Prediction receives features and acquisition resets only. All fits use mean
summed two-output squared error per labeled bin, not per output coordinate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

DT = 0.02
MIN_LABELS = 50
BURN_BINS = 62
VERSION = 4
CONFIG_KEYS = {"family", "tau_seconds", "baseline_tau_seconds", "adapt_gamma"}
ARRAY_KEYS = {"x_tx3", "x_tx45", "x_sbp"}
FEATURE_KEYS = ARRAY_KEYS | {"timestamps", "sequence_id", "channel_ids"}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("exact V4 configuration fields required")
    if config["family"] not in ("tx45", "tx3", "tx45_sbp", "tx3_sbp"):
        raise ValueError("unknown feature family")
    for key, values in (
        ("tau_seconds", (0.2, 0.4)),
        ("baseline_tau_seconds", (None, 5.0)),
        ("adapt_gamma", (1.0, 10.0)),
    ):
        if isinstance(config[key], (bool, np.bool_)) or config[key] not in values:
            raise ValueError(f"unplanned {key}")


def _transformed(features, config, ndim):
    if not isinstance(features, dict) or not ARRAY_KEYS <= features.keys():
        raise ValueError("pure V4 feature dictionary required")
    if set(features) - FEATURE_KEYS:
        raise ValueError("labels or unknown fields are not feature inputs")
    arrays = {k: np.asarray(features[k], dtype=np.float64) for k in ARRAY_KEYS}
    shape = arrays["x_tx3"].shape
    if (
        len(shape) != ndim
        or not 1 <= shape[-1] <= 512
        or (ndim == 2 and shape[0] == 0)
        or any(
            a.shape != shape or not np.isfinite(a).all() or np.any(a < 0) for a in arrays.values()
        )
    ):
        raise ValueError("finite nonnegative matching feature geometry required")
    family = config["family"]
    tx = arrays["x_tx3" if family.startswith("tx3") else "x_tx45"]
    value = np.sqrt(tx)
    if family.endswith("_sbp"):
        value = np.concatenate((value, np.log1p(arrays["x_sbp"])), axis=-1)
    return value


def _segments(segments, n):
    if (
        not isinstance(segments, (list, tuple))
        or not segments
        or any(
            len(s) != 2
            or any(
                not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in s
            )
            for s in segments
        )
        or segments[0][0] != 0
        or segments[-1][1] != n
        or any(a < 0 or a >= b for a, b in segments)
        or any(a[1] != b[0] for a, b in zip(segments, segments[1:]))
    ):
        raise ValueError("ordered nonempty segments must exactly partition observations")


def _exponential(value, segments, tau):
    decay = np.exp(-DT / tau)
    result = np.empty_like(value)
    for a, b in segments:
        result[a:b] = lfilter(
            [1 - decay], [1, -decay], value[a:b], axis=0, zi=decay * value[a][None]
        )[0]
    return result


def representation(features, segments, config):
    """Current feature transform, optional past-only recenter, then fast EWMA.

    The slow baseline initializes to the first transformed observation at each
    block; the first centered observation is zero. Later observations subtract
    the PREVIOUS baseline before that baseline incorporates the current value.
    Both feature families, including counts, follow the same declared operation.
    """
    validate_config(config)
    value = _transformed(features, config, 2)
    _segments(segments, len(value))
    if config["baseline_tau_seconds"] is not None:
        slow = _exponential(value, segments, config["baseline_tau_seconds"])
        past = np.empty_like(value)
        for a, b in segments:
            past[a] = value[a]
            past[a + 1 : b] = slow[a : b - 1]
        value = value - past
    return _exponential(value, segments, config["tau_seconds"])


def design(raw, prefix, channels):
    """Return full augmented design, mean, scale fitted on exact prefix only."""
    raw = np.asarray(raw, dtype=np.float64)
    if (
        not isinstance(channels, (int, np.integer))
        or isinstance(channels, (bool, np.bool_))
        or not 1 <= channels <= 512
        or raw.ndim != 2
        or raw.shape[1] not in (channels, 2 * channels)
        or not np.isfinite(raw).all()
        or not isinstance(prefix, (int, np.integer))
        or isinstance(prefix, (bool, np.bool_))
        or not 1 <= prefix <= len(raw)
    ):
        raise ValueError("finite design and valid positive normalization prefix required")
    floor = np.full(raw.shape[1], 0.1)
    floor[channels:] = 0.01
    mean, scale = raw[:prefix].mean(0), np.maximum(raw[:prefix].std(0), floor)
    return np.column_stack((np.clip((raw - mean) / scale, -8, 8), np.ones(len(raw)))), mean, scale


def training_mask(mask, segments, stop=None):
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("boolean training mask required")
    _segments(segments, len(mask))
    stop = len(mask) if stop is None else stop
    if (
        not isinstance(stop, (int, np.integer))
        or isinstance(stop, (bool, np.bool_))
        or not 0 <= stop <= len(mask)
    ):
        raise ValueError("invalid training prefix")
    result = mask[:stop].copy()
    for a, b in segments:
        if a >= stop:
            break
        result[a : min(a + BURN_BINS, b, stop)] = False
    return result


def sufficient(x, y, mask):
    """Mean-loss augmented Gram/RHS; unselected labels never enter arithmetic."""
    x, y, mask = np.asarray(x, dtype=np.float64), np.asarray(y), np.asarray(mask)
    if (
        x.ndim != 2
        or not 2 <= x.shape[1] <= 1025
        or y.shape != (len(x), 2)
        or mask.shape != (len(x),)
        or mask.dtype != np.bool_
        or not np.isfinite(x).all()
        or not np.all(x[:, -1] == 1)
    ):
        raise ValueError("invalid augmented design/target/mask geometry")
    xx, yy = x[mask], np.asarray(y[mask], dtype=np.float64)
    if not np.isfinite(yy).all():
        raise ValueError("selected labels must be finite")
    n = len(xx)
    return {
        "gram": xx.T @ xx / n if n else np.zeros((x.shape[1], x.shape[1])),
        "rhs": xx.T @ yy / n if n else np.zeros((x.shape[1], 2)),
        "n": n,
    }


def _stats(stats):
    if not isinstance(stats, dict) or set(stats) != {"gram", "rhs", "n"}:
        raise ValueError("exact sufficient statistic fields required")
    gram, rhs, n = np.asarray(stats["gram"]), np.asarray(stats["rhs"]), stats["n"]
    if (
        gram.ndim != 2
        or not 2 <= len(gram) <= 1025
        or gram.shape != (len(gram), len(gram))
        or rhs.shape != (len(gram), 2)
        or not np.isfinite(gram).all()
        or not np.isfinite(rhs).all()
        or not np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12)
        or not isinstance(n, (int, np.integer))
        or isinstance(n, (bool, np.bool_))
        or n < 0
        or (n > 0 and not np.isclose(gram[-1, -1], 1, rtol=0, atol=1e-12))
        or (n == 0 and (np.any(gram) or np.any(rhs)))
    ):
        raise ValueError("invalid sufficient statistics")
    return np.asarray(gram, dtype=np.float64), np.asarray(rhs, dtype=np.float64), n


def _weights(value):
    value = np.asarray(value, dtype=np.float64)
    if (
        value.ndim != 2
        or not 2 <= len(value) <= 1025
        or value.shape[1] != 2
        or not np.isfinite(value).all()
    ):
        raise ValueError("finite augmented two-output weights required")
    return value


def solve_history(liststats, dimension=None):
    """Equal-session mean Gram/RHS, ridge1 and free intercept; no sample pooling."""
    if not isinstance(liststats, (list, tuple)):
        raise ValueError("historical statistics list required")
    if not liststats:
        if (
            not isinstance(dimension, (int, np.integer))
            or isinstance(dimension, (bool, np.bool_))
            or not 2 <= dimension <= 1025
        ):
            raise ValueError("empty history requires augmented dimension")
        return np.zeros((dimension, 2), dtype=np.float64)
    values = [_stats(s) for s in liststats]
    if any(n < MIN_LABELS for _, _, n in values):
        raise ValueError("history entries require at least50 effective labels")
    size = len(values[0][0])
    if any(len(g) != size for g, _, _ in values) or (dimension is not None and dimension != size):
        raise ValueError("history geometry mismatch")
    gram = sum(v[0] for v in values) / len(values)
    rhs = sum(v[1] for v in values) / len(values)
    gram = gram.copy()
    gram[np.arange(size - 1), np.arange(size - 1)] += 1
    return np.linalg.solve(gram, rhs)


def adapt(stats, prior, gamma):
    """Target loss + gamma||W-W0||² + .1gamma||b-b0||²."""
    prior = _weights(prior)
    if isinstance(gamma, (bool, np.bool_)) or gamma not in (1.0, 10.0):
        raise ValueError("adaptation gamma must be1 or10")
    if stats is None:
        return prior.copy()
    gram, rhs, n = _stats(stats)
    if rhs.shape != prior.shape:
        raise ValueError("prior/target geometry mismatch")
    if n < MIN_LABELS:
        return prior.copy()
    diagonal = np.full(len(prior), gamma, dtype=np.float64)
    diagonal[-1] *= 0.1
    return np.linalg.solve(gram + np.diag(diagonal), rhs + diagonal[:, None] * prior)


def solve_daily(stats, penalty=1.0, bias=1):
    """Daily ridge; bias0 is an exactly constrained zero-intercept solution."""
    if (
        isinstance(penalty, (bool, np.bool_))
        or penalty not in (1.0, 10.0)
        or type(bias) is not int
        or bias not in (0, 1)
    ):
        raise ValueError("daily penalty1/10 and integer bias0/1 required")
    gram, rhs, n = _stats(stats)
    if n < MIN_LABELS:
        return np.zeros_like(rhs)
    if bias == 0:
        weights = np.zeros_like(rhs)
        weights[:-1] = np.linalg.solve(gram[:-1, :-1] + penalty * np.eye(len(gram) - 1), rhs[:-1])
        return weights
    diagonal = np.full(len(gram), penalty)
    diagonal[-1] = 0
    return np.linalg.solve(gram + np.diag(diagonal), rhs)


def project_direction(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim not in (1, 2) or value.shape[-1] != 2 or not np.isfinite(value).all():
        raise ValueError("finite two-output predictions required")
    return value / np.maximum(1.0, np.hypot(value[..., 0], value[..., 1]))[..., None]


@dataclass
class Decoder:
    config: dict
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        validate_config(self.config)
        arrays = (self.mean, self.scale, self.weights)
        if any(
            not isinstance(a, np.ndarray) or a.dtype != np.float64 or not np.isfinite(a).all()
            for a in arrays
        ):
            raise ValueError("finite float64 decoder arrays required")
        if (
            self.mean.ndim != 1
            or not 1 <= len(self.mean) <= 1024
            or self.scale.shape != self.mean.shape
            or np.any(self.scale <= 0)
            or self.weights.shape != (len(self.mean) + 1, 2)
            or (self.config["family"].endswith("_sbp") and len(self.mean) % 2)
            or not 1 <= self.channels <= 512
        ):
            raise ValueError("invalid decoder geometry/normalizer")
        self.config = dict(self.config)
        self.mean, self.scale, self.weights = (a.copy() for a in arrays)

    @property
    def channels(self):
        return len(self.mean) // (2 if self.config["family"].endswith("_sbp") else 1)

    def predict(self, features, segments):
        raw = representation(features, segments, self.config)
        if raw.shape[1] != len(self.mean):
            raise ValueError("decoder/feature geometry mismatch")
        z = np.clip((raw - self.mean) / self.scale, -8, 8)
        return project_direction(z @ self.weights[:-1] + self.weights[-1])

    def save(self, path):
        self.__post_init__()
        with Path(path).open("wb") as f:
            np.savez_compressed(
                f,
                format_version=np.asarray(VERSION, dtype=np.int64),
                config=np.asarray(json.dumps(self.config, sort_keys=True, allow_nan=False)),
                mean=self.mean,
                scale=self.scale,
                weights=self.weights,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"format_version", "config", "mean", "scale", "weights"}:
                raise ValueError("unknown decoder export fields")
            version, config = z["format_version"], z["config"]
            if version.shape != () or version.dtype != np.int64 or version.item() != VERSION:
                raise ValueError("unsupported decoder format version")
            if config.shape != () or config.dtype.kind != "U":
                raise ValueError("scalar JSON configuration required")
            return cls(json.loads(config.item()), z["mean"], z["scale"], z["weights"])


class StreamingDecoder:
    """Fixed O(channels) state; no labels, masks, or future samples."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.fast = np.zeros_like(decoder.mean)
        self.slow = np.zeros_like(decoder.mean)
        self.fast_decay = np.exp(-DT / decoder.config["tau_seconds"])
        tau = decoder.config["baseline_tau_seconds"]
        self.slow_decay = None if tau is None else np.exp(-DT / tau)
        self.reset()

    def reset(self):
        self.fast.fill(0)
        self.slow.fill(0)
        self.initialized = False

    @property
    def state_nbytes(self):
        return self.fast.nbytes + self.slow.nbytes

    def step(self, features):
        value = _transformed(features, self.decoder.config, 1)
        if value.shape != self.fast.shape:
            raise ValueError("decoder/feature geometry mismatch")
        if not self.initialized:
            self.slow[:] = value
            self.fast[:] = value if self.slow_decay is None else 0
            self.initialized = True
        else:
            centered = value if self.slow_decay is None else value - self.slow
            self.fast[:] = self.fast_decay * self.fast + (1 - self.fast_decay) * centered
            if self.slow_decay is not None:
                self.slow[:] = self.slow_decay * self.slow + (1 - self.slow_decay) * value
        z = np.clip((self.fast - self.decoder.mean) / self.decoder.scale, -8, 8)
        return project_direction(z @ self.decoder.weights[:-1] + self.decoder.weights[-1])
