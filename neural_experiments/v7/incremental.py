"""Bounded accumulating-label ridge state, separate from the frozen V7 producer."""

from __future__ import annotations

import numpy as np

from neural_experiments.v7.model import CONFIG, Decoder

MAX_CHUNK_ROWS = 64


class CenteredRidge:
    """Merge centered moments; solve mean squared loss + ridge1, free intercept.

    The caller supplies causal representation rows and explicit label-purchase
    receipts. Only selected past targets enter arithmetic. Incoming blocks are
    bounded and strictly chronological; labels cannot be counted twice.
    """

    def __init__(self, mean, scale):
        mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
        if (
            mean.ndim != 1
            or not 2 <= len(mean) <= 1024
            or len(mean) % 2
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("finite fixed combined-feature normalizer required")
        self.mean, self.scale = mean.copy(), scale.copy()
        d = len(mean)
        self.x_mean, self.y_mean = np.zeros(d), np.zeros(2)
        self.centered_gram, self.centered_rhs = np.zeros((d, d)), np.zeros((d, 2))
        self.n, self.last_bin = 0, 1499

    @property
    def moment_nbytes(self):
        return (
            sum(a.nbytes for a in (self.x_mean, self.y_mean, self.centered_gram, self.centered_rhs))
            + 16
        )

    @property
    def normalizer_nbytes(self):
        return self.mean.nbytes + self.scale.nbytes

    def add(self, raw, y, indices, purchased, valid_go, segment_starts, *, observed_stop):
        raw, y, indices = np.asarray(raw), np.asarray(y), np.asarray(indices)
        purchased, valid_go = np.asarray(purchased), np.asarray(valid_go)
        starts = np.asarray(segment_starts)
        rows = len(raw)
        if (
            raw.ndim != 2
            or raw.shape != (rows, len(self.mean))
            or not 1 <= rows <= MAX_CHUNK_ROWS
            or y.shape != (rows, 2)
            or indices.shape != (rows,)
            or indices.dtype.kind not in "iu"
            or purchased.shape != (rows,)
            or purchased.dtype != bool
            or valid_go.shape != (rows,)
            or valid_go.dtype != bool
            or starts.shape != (rows,)
            or starts.dtype.kind not in "iu"
            or type(observed_stop) is not int
            or indices[0] <= self.last_bin
            or np.any(indices[1:] <= indices[:-1])
            or indices[-1] >= observed_stop
            or np.any(starts > indices)
            or np.any(starts < 0)
            or not np.isfinite(raw).all()
        ):
            raise ValueError("bounded chronological observed feature/label block required")
        selected = purchased & valid_go & (indices - starts >= 62)
        target = np.asarray(y[selected], dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("purchased eligible labels must be finite")
        x = np.clip((raw[selected] - self.mean) / self.scale, -8, 8)
        m = len(x)
        if m:
            bx, by = x.mean(0), target.mean(0)
            dx, dy = bx - self.x_mean, by - self.y_mean
            centered_x, centered_y = x - bx, target - by
            total = self.n + m
            correction = self.n * m / total
            self.centered_gram += centered_x.T @ centered_x + correction * np.outer(dx, dx)
            self.centered_rhs += centered_x.T @ centered_y + correction * np.outer(dx, dy)
            self.x_mean += dx * (m / total)
            self.y_mean += dy * (m / total)
            self.n = total
        self.last_bin = int(indices[-1])
        return selected

    def decoder(self):
        """Centered equations eliminate the unpenalized intercept exactly."""
        d = len(self.mean)
        weights = np.zeros((d + 1, 2))
        if self.n >= 50:
            gram = self.centered_gram / self.n
            weights[:-1] = np.linalg.solve(gram + np.eye(d), self.centered_rhs / self.n)
            weights[-1] = self.y_mean - self.x_mean @ weights[:-1]
        return Decoder(CONFIG, self.mean.copy(), self.scale.copy(), weights)

    def snapshot(self):
        """Fixed-size sufficient state; no retained sample rows or labels."""
        return {
            "normalizer_mean": self.mean,
            "normalizer_scale": self.scale,
            "x_mean": self.x_mean,
            "y_mean": self.y_mean,
            "centered_gram": self.centered_gram,
            "centered_rhs": self.centered_rhs,
            "count": np.array(self.n, dtype=np.int64),
            "last_bin": np.array(self.last_bin, dtype=np.int64),
        }
