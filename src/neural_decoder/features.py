"""Finite causal filtering and history; no scoring metadata enters this module."""

import numpy as np
from scipy.signal import lfilter


def fir_weights(taps=12, tau_bins=12.0):
    if taps < 1 or tau_bins <= 0:
        raise ValueError("positive taps and time constant required")
    weights = np.exp(-np.arange(taps, dtype=np.float64) / tau_bins)
    return weights / weights.sum()


def validate_counts(counts, channels):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape[-1:] != (channels,) or counts.ndim not in (1, 2):
        raise ValueError("invalid count dimensions")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    return counts


def filter_file(counts, weights):
    """Left zero padding, reset once per file, output for every observed bin."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("invalid counts")
    return lfilter(weights, [1.0], counts, axis=0)


def history_matrix(filtered, mean, scale, history):
    """Current bin first; missing standardized history is zero at file start."""
    if history < 1:
        raise ValueError("history must be positive")
    values = (filtered - mean) / scale
    result = np.zeros((len(values), history, values.shape[1]), dtype=np.float64)
    for lag in range(min(history, len(values))):
        result[lag:, lag] = values[: len(values) - lag]
    return result.reshape(len(values), -1)


class CausalFeatures:
    def __init__(self, mean, scale, history, weights):
        self.mean = np.asarray(mean, dtype=np.float64).copy()
        self.scale = np.asarray(scale, dtype=np.float64).copy()
        self.weights = np.asarray(weights, dtype=np.float64).copy()
        self.history = int(history)
        if (
            self.mean.ndim != 1
            or self.scale.shape != self.mean.shape
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or (self.scale <= 0).any()
            or self.history < 1
            or self.weights.ndim != 1
            or not len(self.weights)
            or not np.isfinite(self.weights).all()
        ):
            raise ValueError("invalid frozen feature parameters")
        self.reset()

    def reset(self):
        self.raw = np.zeros((len(self.weights), len(self.mean)))
        self.past = np.zeros((self.history, len(self.mean)))
        self.next_sequence_id = 0

    def step(self, counts, sequence_id=None):
        counts = validate_counts(counts, len(self.mean))
        if counts.ndim != 1:
            raise ValueError("one-bin step requires one count vector")
        if sequence_id is not None and sequence_id != self.next_sequence_id:
            raise ValueError("missing, duplicate or out-of-order sequence ID")
        self.raw[1:] = self.raw[:-1]
        self.raw[0] = counts
        filtered = self.weights @ self.raw
        self.past[1:] = self.past[:-1]
        self.past[0] = (filtered - self.mean) / self.scale
        self.next_sequence_id += 1
        return self.past.ravel().copy()
