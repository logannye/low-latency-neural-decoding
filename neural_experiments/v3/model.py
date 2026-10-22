"""Causal sufficient-state filtering and regularized, bounded direction decoding.

Fitting accepts an already clipped calibration prefix. Prediction accepts neural
features and acquisition boundaries only. No target or evaluation mask enters
either batch or incremental inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

DT = 0.02


def validate_config(config):
    if not isinstance(config, dict) or config.get("kind") not in ("regularized", "v2_ridge"):
        raise ValueError("unknown decoder configuration")
    keys = {"kind", "family"}
    if config["kind"] == "regularized":
        keys |= {"tau_seconds", "baseline_tau_seconds", "ridge_lambda"}
    if set(config) != keys or config["family"] not in ("tx", "tx_sbp"):
        raise ValueError("unexpected decoder configuration fields")
    if config["kind"] == "v2_ridge":
        if config["family"] != "tx":
            raise ValueError("V2 ridge requires count features")
        return
    for key, permitted in (
        ("tau_seconds", (0.1, 0.2, 0.4)),
        ("baseline_tau_seconds", (None, 30.0)),
        ("ridge_lambda", (0.1, 1.0, 10.0)),
    ):
        if isinstance(config[key], bool) or config[key] not in permitted:
            raise ValueError(f"unplanned {key}")


def validate_features(tx, sbp):
    tx, sbp = np.asarray(tx, dtype=np.float64), np.asarray(sbp, dtype=np.float64)
    if tx.shape != sbp.shape or tx.ndim not in (1, 2) or not 1 <= tx.shape[-1] <= 512:
        raise ValueError("feature geometry mismatch")
    if not np.isfinite(tx).all() or not np.isfinite(sbp).all() or np.any(tx < 0) or np.any(sbp < 0):
        raise ValueError("finite nonnegative features required")
    return tx, sbp


def features(tx, sbp, family):
    tx, sbp = validate_features(tx, sbp)
    if family == "tx":
        return np.sqrt(tx)
    if family == "tx_sbp":
        return np.concatenate((np.sqrt(tx), np.log1p(sbp)), axis=-1)
    raise ValueError("unknown feature family")


def check_segments(segments, n):
    if not isinstance(segments, (list, tuple)) or any(
        len(segment) != 2 or any(not isinstance(value, (int, np.integer)) for value in segment)
        for segment in segments
    ):
        raise ValueError("integer segment bounds required")
    if not segments or segments[0][0] != 0 or segments[-1][1] != n:
        raise ValueError("segments must cover recording")
    if any(a >= b for a, b in segments) or any(
        a[1] != b[0] for a, b in zip(segments, segments[1:])
    ):
        raise ValueError("segments must be ordered, nonempty, and contiguous")


def exponential(x, segments, tau):
    check_segments(segments, len(x))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("positive filter time constant required")
    decay = np.exp(-DT / tau)
    result = np.empty_like(x, dtype=np.float64)
    for start, end in segments:
        result[start:end] = lfilter(
            [1 - decay], [1, -decay], x[start:end], axis=0, zi=decay * x[start][None]
        )[0]
    return result


def representation(tx, sbp, segments, config):
    x = features(tx, sbp, config["family"])
    fast = exponential(x, segments, config["tau_seconds"])
    baseline = config["baseline_tau_seconds"]
    return fast if baseline is None else fast - exponential(x, segments, baseline)


def training_mask(mask, segments, burn=62):
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != bool:
        raise ValueError("training mask must be a boolean vector")
    if not isinstance(burn, (int, np.integer)) or isinstance(burn, bool) or burn < 0:
        raise ValueError("nonnegative integer burn required")
    result = mask.copy()
    check_segments(segments, len(result))
    for start, end in segments:
        result[start : min(start + burn, end)] = False
    return result


def project_direction(prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim not in (1, 2) or prediction.shape[-1] != 2:
        raise ValueError("direction prediction must have two outputs")
    if not np.isfinite(prediction).all():
        raise ValueError("finite direction prediction required")
    norm = np.hypot(prediction[..., 0], prediction[..., 1])
    return prediction / np.maximum(1.0, norm)[..., None]


def prepare_system(raw, y, mask, segments, channels):
    """Cache one prefix's sufficient statistics for a fixed regularization grid."""
    raw = np.asarray(raw, dtype=np.float64)
    if (
        not isinstance(channels, (int, np.integer))
        or not 1 <= channels <= 512
        or raw.ndim != 2
        or raw.shape[1] not in (channels, 2 * channels)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("invalid filtered feature geometry or values")
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (len(raw), 2) or np.asarray(mask).shape != (len(raw),):
        raise ValueError("target geometry mismatch")
    usable = training_mask(mask, segments)
    if usable.sum() < 50 or not np.isfinite(y[usable]).all():
        raise ValueError("insufficient or invalid calibration targets")
    floor = np.full(raw.shape[1], 0.1)
    floor[channels:] = 0.01
    mean, scale = raw.mean(0), np.maximum(raw.std(0), floor)
    x = np.clip((raw - mean) / scale, -8.0, 8.0)[usable]
    target = y[usable]
    xm, ym = x.mean(0), target.mean(0)
    centered = x - xm
    n = int(usable.sum())
    return {
        "mean": mean,
        "scale": scale,
        "gram": centered.T @ centered / n,
        "rhs": centered.T @ (target - ym) / n,
        "xmean": xm,
        "ymean": ym,
        "counts": {
            "n_prefix_bins": len(raw),
            "n_available_labels": int(np.asarray(mask).sum()),
            "n_training_labels": n,
        },
    }


def solve_system(system, config):
    validate_config(config)
    if config["kind"] != "regularized":
        raise ValueError("normalized sufficient system requires regularized model")
    gram = system["gram"].copy()
    gram.flat[:: len(gram) + 1] += config["ridge_lambda"]
    coefficients = np.linalg.solve(gram, system["rhs"])
    model = Decoder(
        dict(config),
        system["mean"],
        system["scale"],
        coefficients,
        system["ymean"] - system["xmean"] @ coefficients,
    )
    receipt = dict(system["counts"])
    receipt["sum_loss_ridge_penalty"] = config["ridge_lambda"] * receipt["n_training_labels"]
    return model, receipt


@dataclass
class Decoder:
    config: dict
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray

    def __post_init__(self):
        validate_config(self.config)
        arrays = (self.mean, self.scale, self.coefficients, self.intercept)
        if (
            any(not isinstance(a, np.ndarray) for a in arrays)
            or self.mean.ndim != 1
            or not 1 <= len(self.mean) <= 1024
            or self.scale.shape != self.mean.shape
            or self.coefficients.shape != (len(self.mean), 2)
            or self.intercept.shape != (2,)
            or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in arrays)
            or np.any(self.scale <= 0)
            or (self.config["family"] == "tx_sbp" and len(self.mean) % 2)
            or not 1 <= self.channels <= 512
        ):
            raise ValueError("invalid saved model geometry or numeric values")

    @property
    def channels(self):
        return len(self.mean) // (2 if self.config["family"] == "tx_sbp" else 1)

    def predict(self, tx, sbp, segments):
        tx, sbp = validate_features(tx, sbp)
        if tx.ndim != 2 or tx.shape[-1] != self.channels:
            raise ValueError("model/feature channel mismatch")
        check_segments(segments, len(tx))
        if self.config.get("kind") == "v2_ridge":
            z = (features(tx, sbp, "tx") - self.mean) / self.scale
            weights = np.exp(-np.arange(12) / 12)
            weights /= weights.sum()
            x = np.empty_like(z)
            for start, stop in segments:
                x[start:stop] = lfilter(weights, [1.0], z[start:stop], axis=0)
        else:
            x = (representation(tx, sbp, segments, self.config) - self.mean) / self.scale
            x = np.clip(x, -8.0, 8.0)
        pred = x @ self.coefficients + self.intercept
        return pred if self.config.get("kind") == "v2_ridge" else project_direction(pred)

    def save(self, path):
        import json

        with Path(path).open("xb") as stream:
            np.savez(
                stream,
                format_version=np.asarray(3, dtype=np.int64),
                config=np.asarray(json.dumps(self.config, sort_keys=True)),
                mean=self.mean,
                scale=self.scale,
                coefficients=self.coefficients,
                intercept=self.intercept,
            )

    @classmethod
    def load(cls, path):
        import json

        with np.load(path, allow_pickle=False) as z:
            required = {"format_version", "config", "mean", "scale", "coefficients", "intercept"}
            if set(z.files) != required or z["format_version"].shape != ():
                raise ValueError("unexpected model fields")
            if z["format_version"].dtype != np.int64 or z["format_version"].item() != 3:
                raise ValueError("unknown model format")
            if z["config"].shape != () or z["config"].dtype.kind != "U":
                raise ValueError("configuration must be scalar JSON text")
            config = json.loads(z["config"].item())
            arrays = [z[k] for k in ("mean", "scale", "coefficients", "intercept")]
        mean, scale, coef, intercept = arrays
        return cls(config, mean, scale, coef, intercept)


def fit(tx, sbp, y, mask, segments, config):
    """Fit from prefix-only arrays. Caller controls and records the prefix boundary."""
    validate_config(config)
    tx, sbp = validate_features(tx, sbp)
    if tx.ndim != 2:
        raise ValueError("fit requires a sequence of observations")
    y = np.asarray(y, dtype=np.float64)
    if config.get("kind") != "v2_ridge":
        raw = representation(tx, sbp, segments, config)
        return solve_system(prepare_system(raw, y, mask, segments, tx.shape[1]), config)
    if y.shape != (len(tx), 2) or np.asarray(mask).shape != (len(tx),):
        raise ValueError("target geometry mismatch")
    usable = training_mask(mask, segments)
    if usable.sum() < 50 or not np.isfinite(y[usable]).all():
        raise ValueError("insufficient or invalid calibration targets")
    if config.get("kind") == "v2_ridge":
        raw = features(tx, sbp, "tx")
        mean, scale = raw.mean(0), np.maximum(raw.std(0), 0.1)
        weights = np.exp(-np.arange(12) / 12)
        weights /= weights.sum()
        x = np.empty_like(raw)
        for start, stop in segments:
            x[start:stop] = lfilter(weights, [1.0], (raw[start:stop] - mean) / scale, axis=0)
        penalty = 10.0
    xx, yy = x[usable], y[usable]
    xm, ym = xx.mean(0), yy.mean(0)
    centered = xx - xm
    gram = centered.T @ centered
    gram.flat[:: len(gram) + 1] += penalty
    coefficients = np.linalg.solve(gram, centered.T @ (yy - ym))
    model = Decoder(dict(config), mean, scale, coefficients, ym - xm @ coefficients)
    return model, {
        "n_prefix_bins": len(tx),
        "n_available_labels": int(np.asarray(mask).sum()),
        "n_training_labels": int(usable.sum()),
        "sum_loss_ridge_penalty": float(penalty),
    }


class StreamingDecoder:
    """O(channels) sufficient state; no observations after the current step."""

    def __init__(self, model):
        if model.config["kind"] != "regularized":
            raise ValueError("incremental V3 runtime requires regularized model")
        self.model = model
        self.fast = np.zeros_like(model.mean)
        self.slow = np.zeros_like(model.mean)
        self.fast_decay = np.exp(-DT / model.config["tau_seconds"])
        tau = model.config["baseline_tau_seconds"]
        self.slow_decay = None if tau is None else np.exp(-DT / tau)
        self.reset()

    def reset(self):
        self.fast.fill(0)
        self.slow.fill(0)
        self.initialized = False

    @property
    def state_nbytes(self):
        return self.fast.nbytes + self.slow.nbytes

    def step(self, tx, sbp):
        if np.asarray(tx).shape != (self.model.channels,):
            raise ValueError("model/feature channel mismatch")
        x = features(tx, sbp, self.model.config["family"])
        if not self.initialized:
            self.fast[:] = x
            self.slow[:] = x
            self.initialized = True
        else:
            self.fast[:] = self.fast_decay * self.fast + (1 - self.fast_decay) * x
            if self.slow_decay is not None:
                self.slow[:] = self.slow_decay * self.slow + (1 - self.slow_decay) * x
        value = self.fast if self.slow_decay is None else self.fast - self.slow
        z = np.clip((value - self.model.mean) / self.model.scale, -8.0, 8.0)
        return project_direction(z @ self.model.coefficients + self.model.intercept)
