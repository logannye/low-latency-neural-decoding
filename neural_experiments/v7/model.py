"""Past-only ridge updates with fixed normalization and explicit label accounting.

The representation and portable decoder are the unchanged V4 implementations.
Every solve minimizes weighted mean two-output squared loss plus ridge1 on the
feature coefficients, with a free intercept. Prediction does not consume labels.
"""

from __future__ import annotations

import hashlib

import numpy as np

from neural_experiments.v4 import model as core

DT = 0.02
WARMUP_BINS = 1500
GAP_BINS = 100
CONFIG = {
    "family": "tx45_sbp",
    "tau_seconds": 0.2,
    "baseline_tau_seconds": None,
    "adapt_gamma": 10.0,
}
METHODS = (
    "frozen120",
    "expanding",
    "rolling60",
    "rolling120",
    "rolling240",
    "exponential60",
    "rolling120_local",
)
Decoder = core.Decoder
StreamingDecoder = core.StreamingDecoder
representation = core.representation


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fit_mask(
    raw,
    y,
    mask,
    segments,
    channels,
    stop,
    *,
    normalization_start=0,
    normalization_stop=WARMUP_BINS,
    half_life_seconds=None,
):
    """Fit exclusively from masked labels before stop, never reading later labels.

    Masked-out targets may be nonfinite. Normalization uses the declared past
    feature interval, including unlabeled rows. Array geometry may describe the
    whole replay, but all numeric fitting operations are bounded by ``stop``.
    The caller is responsible for proving that selected labels were purchased.
    """
    raw, y, mask = np.asarray(raw), np.asarray(y), np.asarray(mask)
    if (
        raw.ndim != 2
        or type(channels) is not int
        or not 1 <= channels <= 512
        or raw.shape[1] != 2 * channels
        or type(stop) is not int
        or not WARMUP_BINS <= stop <= len(raw)
        or y.ndim != 2
        or y.shape[1] != 2
        or not stop <= len(y) <= len(raw)
        or mask.dtype != bool
        or mask.shape != (len(y),)
        or type(normalization_start) is not int
        or type(normalization_stop) is not int
        or not 0 <= normalization_start < normalization_stop <= stop
        or half_life_seconds not in (None, 60.0)
        or isinstance(half_life_seconds, (bool, np.bool_))
    ):
        raise ValueError("past-only label prefix and valid V7 fitting geometry required")
    core._segments(segments, len(raw))
    prefix = np.asarray(raw[:stop], dtype=np.float64)
    if not np.isfinite(prefix).all():
        raise ValueError("observed feature prefix must be finite")
    normal = prefix[normalization_start:normalization_stop]
    floor = np.concatenate((np.full(channels, 0.1), np.full(channels, 0.01)))
    mean, scale = normal.mean(0), np.maximum(normal.std(0), floor)
    selected = core.training_mask(mask[:stop], _prefix_segments(segments, stop))
    selected[:WARMUP_BINS] = False
    indices = np.flatnonzero(selected)
    labels = np.asarray(y[:stop][selected], dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("selected past labels must be finite")
    x = np.column_stack((np.clip((prefix[selected] - mean) / scale, -8, 8), np.ones(len(indices))))
    sample_weights = (
        np.ones(len(indices), dtype=np.float64)
        if half_life_seconds is None
        else np.exp2(-((stop - 1 - indices) * DT) / half_life_seconds)
    )
    n = len(indices)
    total = float(sample_weights.sum())
    weights = np.zeros((raw.shape[1] + 1, 2), dtype=np.float64)
    if n >= core.MIN_LABELS:
        gram = (x.T * sample_weights) @ x / total
        rhs = (x.T * sample_weights) @ labels / total
        penalty = np.ones(len(gram))
        penalty[-1] = 0
        weights = np.linalg.solve(gram + np.diag(penalty), rhs)
    return Decoder(CONFIG, mean, scale, weights), {
        "fit_stop_bin": stop,
        "normalization_start_bin": normalization_start,
        "normalization_stop_bin": normalization_stop,
        "half_life_seconds": half_life_seconds,
        "training_rows": n,
        "training_mask_sha256": digest_array(selected),
        "training_weights_sha256": digest_array(sample_weights),
        "weight_sum": total,
        "effective_sample_size": total**2 / float(np.square(sample_weights).sum()) if n else 0.0,
        "first_training_bin": int(indices[0]) if n else None,
        "last_training_bin": int(indices[-1]) if n else None,
        "feature_penalty": 1.0,
        "free_intercept": True,
        "fallback_zero": n < core.MIN_LABELS,
    }


def _prefix_segments(segments, stop):
    return [(a, min(b, stop)) for a, b in segments if a < stop]


def fit(raw, y, mask, segments, channels, method, update_bin):
    """Fit one prespecified recoverability method with a 2-second embargo."""
    if method not in METHODS or type(update_bin) is not int:
        raise ValueError("known V7 method and integer update boundary required")
    stop = update_bin - GAP_BINS
    if stop < 13500 or update_bin > len(raw):
        raise ValueError("recoverability requires a complete 272-second update boundary")
    start = WARMUP_BINS
    if method == "frozen120":
        stop = WARMUP_BINS + 6000
    elif method.startswith("rolling"):
        seconds = int(method.removeprefix("rolling").removesuffix("_local"))
        start = stop - 50 * seconds
    selected = np.asarray(mask).copy()
    selected[:start] = False
    selected[stop:] = False
    norm_start, norm_stop = (start, stop) if method.endswith("_local") else (0, WARMUP_BINS)
    fitted, receipt = fit_mask(
        raw,
        y,
        selected,
        segments,
        channels,
        stop,
        normalization_start=norm_start,
        normalization_stop=norm_stop,
        half_life_seconds=60.0 if method == "exponential60" else None,
    )
    return fitted, {
        **receipt,
        "method": method,
        "update_bin": update_bin,
        "label_start_bin": start,
        "label_stop_bin": stop,
        "normalization": "local_training_interval" if method.endswith("_local") else "first30",
        "embargo_seconds": (update_bin - stop) * DT,
        "selected_interval_seconds": (stop - start) * DT,
        "available_label_seconds": (stop - WARMUP_BINS) * DT,
    }


def predict_raw(fitted, raw):
    """Predict from the cached causal representation, preserving disk projection."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(fitted.mean) or not np.isfinite(raw).all():
        raise ValueError("finite matching causal representation required")
    x = np.clip((raw - fitted.mean) / fitted.scale, -8, 8)
    return core.project_direction(x @ fitted.weights[:-1] + fitted.weights[-1])
