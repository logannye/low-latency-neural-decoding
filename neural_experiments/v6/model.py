"""V6 numeric foundation: fit once, freeze normalization, confirm exact export.

This wraps the frozen V4 causal representation and numeric export without altering
older studies. The caller owns chronological data admission and label requests.
"""

from __future__ import annotations

import hashlib

import numpy as np

from neural_experiments.v4 import model as core

DT = 0.02
WARMUP_BINS = 1500
Decoder = core.Decoder
StreamingDecoder = core.StreamingDecoder
representation = core.representation


def candidate_grid():
    return [
        {
            "candidate_id": f"{family}_{head}_{normalization}",
            "config": {
                "family": family,
                "tau_seconds": tau,
                "baseline_tau_seconds": None,
                "adapt_gamma": 10.0,
            },
            "head": head,
            "normalization": normalization,
        }
        for family, tau in (("tx45", 0.4), ("tx45_sbp", 0.2))
        for head in ("prior10", "scratch1", "scratch10_zero")
        for normalization in ("first30", "fit_available")
    ]


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fit(raw, y, mask, segments, channels, stop, candidate, prior=None, start=WARMUP_BINS):
    """Fit labeled [start,stop), normalizing no later than stop; no suffix access.

    raw is a cached causal representation over all observed features. y/mask may
    be just the broker's prefix or longer; only the declared interval is read.
    Arbitrary integer stops support inner splits and extended calibration budgets.
    Short/incomplete requested intervals are explicit errors, never silently fit.
    """
    raw, y, mask = np.asarray(raw), np.asarray(y), np.asarray(mask)
    if candidate not in candidate_grid():
        raise ValueError("unplanned V6 candidate")
    if (
        type(start) is not int
        or type(stop) is not int
        or start != WARMUP_BINS
        or not start < stop <= len(raw)
        or y.ndim != 2
        or y.shape[1] != 2
        or mask.dtype != bool
        or mask.shape != (len(y),)
        or not stop <= len(y) <= len(raw)
    ):
        raise ValueError("complete prospective label prefix and integer bounds required")
    core._segments(segments, len(raw))
    normalization_stop = WARMUP_BINS if candidate["normalization"] == "first30" else stop
    z, mean, scale = core.design(raw, normalization_stop, channels)
    selected = mask[:stop].copy()
    selected[:start] = False
    for a, b in segments:
        if a < stop:
            selected[a : min(a + core.BURN_BINS, b, stop)] = False
    stats = core.sufficient(z[:stop], y[:stop], selected)
    if prior is None:
        prior = np.zeros_like(stats["rhs"])
    prior = np.asarray(prior)
    if (
        prior.shape != stats["rhs"].shape
        or prior.dtype != np.float64
        or not np.isfinite(prior).all()
    ):
        raise ValueError("finite matching float64 prior required")
    head = candidate["head"]
    weights = (
        core.adapt(stats, prior, 10.0)
        if head == "prior10"
        else core.solve_daily(
            stats, 1.0 if head == "scratch1" else 10.0, 1 if head == "scratch1" else 0
        )
    )
    decoder = Decoder(candidate["config"], mean, scale, weights)
    return decoder, {
        "candidate_id": candidate["candidate_id"],
        "head": head,
        "normalization": candidate["normalization"],
        "normalization_stop_bin": normalization_stop,
        "label_start_bin": start,
        "label_stop_bin": stop,
        "requested_label_seconds": (stop - start) * DT,
        "training_rows": stats["n"],
        "training_mask_sha256": digest_array(selected),
        "fallback": stats["n"] < core.MIN_LABELS,
        "fallback_output": "prior" if head == "prior10" else "zero",
        "feature_penalty": 1.0 if head == "scratch1" else 10.0,
        "intercept_penalty": 1.0 if head == "prior10" else None,
        "free_intercept": head == "scratch1",
        "constrained_zero_intercept": head == "scratch10_zero",
        "prior_sha256": digest_array(prior) if head == "prior10" else None,
    }


def historical_stats(raw, y, mask, segments, channels):
    if not 1 <= len(raw) <= 30000:
        raise ValueError("history must be an authorized <=600s prefix")
    z, _, _ = core.design(raw, len(raw), channels)
    selected = core.training_mask(mask, segments) & (np.arange(len(raw)) % 5 == 0)
    return core.sufficient(z, y, selected)


history_stats = historical_stats


def fit_candidate(raw, y, mask, segments, stop, candidate, prior, n_channels):
    """Runner-facing spelling with explicit channel count after the candidate."""
    return fit(raw, y, mask, segments, n_channels, stop, candidate, prior)


def prior_decoder(raw, channels, config, prior):
    """No query labels: fixed first30 normalizer and previously fitted coefficients."""
    _, mean, scale = core.design(raw, WARMUP_BINS, channels)
    return Decoder(config, mean, scale, np.asarray(prior, dtype=np.float64))
