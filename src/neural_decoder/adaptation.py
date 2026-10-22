"""Feature-only causal mean correction for a fixed history-one ridge decoder.

The applied correction is anchored at the training mean. For each bin, predict
using the previous EMA state and then update that state from the current filtered
neural observation. Variance, model weights and intercept never change.
"""

from __future__ import annotations

import numpy as np

from .features import CausalFeatures, validate_counts


class AdaptiveRidge:
    """One stream, reset per provider file; trial boundaries are not inputs.

    ``ema`` subtracts the previous unconstrained EMA deviation. ``bounded`` clips
    that applied correction channelwise to ±cap_sd times training SD; it retains
    the same internal EMA as ``ema``. ``frozen`` applies zero correction. State
    memory is fixed in size, but Python step scratch allocation is not eliminated.
    """

    def __init__(
        self,
        model,
        *,
        mode="frozen",
        cap_sd=None,
        tau_seconds=30.0,
        bin_seconds=0.02,
    ):
        self.model = model.validate()
        if model.kind != "ridge" or model.history != 1:
            raise ValueError("adaptation requires a fixed history-one ridge model")
        if mode not in {"frozen", "ema", "bounded"}:
            raise ValueError("unknown mean correction mode")
        if not np.isfinite(tau_seconds) or tau_seconds <= 0:
            raise ValueError("tau_seconds must be positive and finite")
        if not np.isfinite(bin_seconds) or bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive and finite")
        if mode == "bounded":
            if cap_sd is None or not np.isfinite(cap_sd) or cap_sd <= 0:
                raise ValueError("bounded correction requires a positive finite cap")
        elif cap_sd is not None:
            raise ValueError("cap_sd applies only to bounded correction")
        self.mode, self.cap_sd = mode, cap_sd
        self.alpha = float(-np.expm1(-bin_seconds / tau_seconds))
        self.features = CausalFeatures(model.mean, model.scale, model.history, model.weights)
        self.reset()

    def reset(self):
        self.features.reset()
        self.ema_deviation = np.zeros_like(self.model.mean, dtype=np.float64)
        self.last_correction = np.zeros_like(self.ema_deviation)
        self.failed = False

    @property
    def state_nbytes(self):
        """Persistent numeric state only; excludes model and Python overhead."""
        return sum(
            array.nbytes
            for array in (
                self.features.raw,
                self.features.past,
                self.ema_deviation,
                self.last_correction,
            )
        )

    def state_snapshot(self):
        """Copies for causality testing, not a portable restore format."""
        return {
            "raw": self.features.raw.copy(),
            "past": self.features.past.copy(),
            "ema_deviation": self.ema_deviation.copy(),
            "last_correction": self.last_correction.copy(),
            "next_sequence_id": self.features.next_sequence_id,
            "failed": self.failed,
        }

    def step(self, counts, sequence_id=None):
        """Return prediction after all work for the bin, including the EMA update."""
        if self.failed:
            raise RuntimeError("reset required after an arithmetic failure")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                values = self.features.step(counts, sequence_id)
                if self.mode == "frozen":
                    correction = np.zeros_like(self.ema_deviation)
                elif self.mode == "ema":
                    correction = self.ema_deviation.copy()
                else:
                    bound = self.cap_sd * self.model.scale
                    correction = np.clip(self.ema_deviation, -bound, bound)
                prediction = (
                    values - correction / self.model.scale
                ) @ self.model.coefficient + self.model.intercept
                if self.mode != "frozen":
                    filtered = self.features.weights @ self.features.raw
                    self.ema_deviation = (1 - self.alpha) * self.ema_deviation + self.alpha * (
                        filtered - self.model.mean
                    )
                self.last_correction[:] = correction
                if not np.isfinite(prediction).all() or not np.isfinite(self.ema_deviation).all():
                    raise FloatingPointError("nonfinite mean adaptation state or prediction")
        except FloatingPointError:
            self.failed = True
            raise
        return prediction.copy()

    def predict_chunk(self, counts):
        counts = validate_counts(counts, len(self.model.mean))
        if counts.ndim != 2:
            raise ValueError("chunk must be time by channel")
        if not len(counts):
            return np.empty((0, len(self.model.intercept)), dtype=np.float64)
        return np.array([self.step(row) for row in counts])
