"""Six V5 candidate heads with prospective labels after a fixed30s warmup.

The frozen V4 numeric core supplies causal transforms, convex solves and strict
version4 exports. This module controls V5 candidate identity and calibration
boundaries. Historical selection and policy decisions belong to the runner.
"""

from __future__ import annotations

import hashlib

import numpy as np

from neural_experiments.v4 import model as core

DT = 0.02
WARMUP_BINS = 1500
HISTORY_CAP_BINS = 30000
LABEL_BUDGET_SECONDS = (0, 30, 60, 120, 240)
DECISION_STOPS = (1500, 3000, 4500, 7500, 13500)
# 2250 is the half-label fit for the30s stage; other half fits are full-stage stops.
FIT_STOPS = (1500, 2250, 3000, 4500, 7500, 13500)
MIN_LABELS = core.MIN_LABELS
Decoder = core.Decoder
StreamingDecoder = core.StreamingDecoder
solve_history = core.solve_history
project_direction = core.project_direction


def candidate_grid():
    """Fixed order: counts/combined, then history/free-scratch/zero-bias-scratch."""
    result = []
    for family, tau in (("tx45", 0.4), ("tx45_sbp", 0.2)):
        for head in ("prior10", "scratch1", "scratch10_zero"):
            result.append(
                {
                    "candidate_id": f"{family}_{head}",
                    "config": {
                        "family": family,
                        "tau_seconds": tau,
                        "baseline_tau_seconds": None,
                        # The numeric export shares the core schema; head identity is separate.
                        "adapt_gamma": 10.0,
                    },
                    "head": head,
                }
            )
    return result


def validate_config(config):
    core.validate_config(config)
    expected = {"tx45": 0.4, "tx45_sbp": 0.2}
    if (
        config["family"] not in expected
        or config["tau_seconds"] != expected[config["family"]]
        or config["baseline_tau_seconds"] is not None
        or config["adapt_gamma"] != 10.0
    ):
        raise ValueError("unplanned V5 representation")


def validate_candidate(candidate):
    if not isinstance(candidate, dict) or set(candidate) != {"candidate_id", "config", "head"}:
        raise ValueError("exact V5 candidate fields required")
    validate_config(candidate["config"])
    if candidate not in candidate_grid():
        raise ValueError("unplanned V5 candidate identity")


def representation(features, segments, config):
    validate_config(config)
    return core.representation(features, segments, config)


def design(raw, channels):
    """All query candidates share the first1500 unlabeled feature observations."""
    return core.design(raw, WARMUP_BINS, channels)


def historical_stats(raw, y, mask, segments, channels):
    """Earlier full-recording normalization, burned go rows and global stride5.

    The caller must establish that this is a strictly earlier permitted session.
    This helper cannot infer chronological authorization from numeric arrays.
    """
    if not 1 <= len(raw) <= HISTORY_CAP_BINS:
        raise ValueError("history must be a nonempty first600s capped recording")
    z, _, _ = core.design(raw, len(raw), channels)
    eligible = core.training_mask(mask, segments)
    eligible &= np.arange(len(eligible)) % 5 == 0
    return core.sufficient(z, y, eligible)


# Short alias for the runner's per-representation history-statistic cache.
history_stats = historical_stats


def daily_control_design(raw, channels, stop):
    """Strong fixed daily controls normalize all features observed by150/270s.

    Adaptive candidates use design() instead. The supervised interval is still
    [30s,stop); this helper changes only the control's permitted normalizer.
    """
    if (
        not isinstance(stop, (int, np.integer))
        or isinstance(stop, (bool, np.bool_))
        or stop not in (7500, 13500)
    ):
        raise ValueError("daily120/240 control normalization stop required")
    if len(raw) < WARMUP_BINS:
        raise ValueError("at least30s unlabeled warmup required")
    return core.design(raw, min(stop, len(raw)), channels)


def calibration_mask(mask, segments, stop, start=WARMUP_BINS):
    """Only observed labels in[start,stop), retaining per-acquisition burn."""
    if (
        not isinstance(start, (int, np.integer))
        or isinstance(start, (bool, np.bool_))
        or start != WARMUP_BINS
        or not isinstance(stop, (int, np.integer))
        or isinstance(stop, (bool, np.bool_))
        or stop not in FIT_STOPS
    ):
        raise ValueError("fixed30s start and planned full/half stage stop required")
    if len(mask) < WARMUP_BINS:
        raise ValueError("at least30s unlabeled warmup required")
    eligible = core.training_mask(mask, segments, min(stop, len(mask)))
    eligible[:start] = False
    return eligible


def fit_stage(
    z,
    y,
    mask,
    segments,
    stop,
    candidate,
    prior=None,
    start=WARMUP_BINS,
    *,
    normalization_bins=WARMUP_BINS,
):
    """Return augmented weights and an exact label-window/fit receipt.

    z is a cached design produced by design(), whose normalizer is fixed before
    any calibration labels arrive. Unselected labels may be nonfinite; only the
    exact eligible[start,stop) rows enter sufficient-statistics arithmetic.
    """
    validate_candidate(candidate)
    z, y, mask = np.asarray(z), np.asarray(y), np.asarray(mask)
    if z.ndim != 2 or y.shape != (len(z), 2) or mask.shape != (len(z),):
        raise ValueError("matching full query design/target/mask geometry required")
    eligible = calibration_mask(mask, segments, stop, start)
    observed_stop = len(eligible)
    if (
        not isinstance(normalization_bins, (int, np.integer))
        or isinstance(normalization_bins, (bool, np.bool_))
        or (
            normalization_bins != WARMUP_BINS
            and not (
                candidate["candidate_id"] == "tx45_sbp_scratch1"
                and stop in (7500, 13500)
                and normalization_bins == observed_stop
            )
        )
    ):
        raise ValueError("normalization must match fixed warmup or declared daily control")
    stats = core.sufficient(z[:observed_stop], y[:observed_stop], eligible)
    if prior is None:
        prior = np.zeros_like(stats["rhs"])
    prior = np.asarray(prior)
    if (
        prior.shape != stats["rhs"].shape
        or prior.dtype != np.float64
        or not np.isfinite(prior).all()
    ):
        raise ValueError("finite float64 prior with matching augmented geometry required")
    head = candidate["head"]
    if head == "prior10":
        weights = core.adapt(stats, prior, 10.0)
        penalty, intercept_penalty, free_intercept = 10.0, 1.0, False
    else:
        penalty, bias = (1.0, 1) if head == "scratch1" else (10.0, 0)
        weights = core.solve_daily(stats, penalty, bias)
        intercept_penalty, free_intercept = None, bool(bias)
    return weights, {
        "candidate_id": candidate["candidate_id"],
        "head": head,
        "normalization_bins": int(normalization_bins),
        "label_start_bin": int(start),
        "label_stop_bin": observed_stop,
        "requested_label_stop_bin": int(stop),
        "observed_label_stop_bin": observed_stop,
        "label_seconds": (stop - start) * DT,
        "available_label_seconds": (observed_stop - start) * DT,
        "complete_requested_interval": observed_stop == stop,
        "training_rows": stats["n"],
        "training_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(eligible).tobytes()
        ).hexdigest(),
        "training_mask_length": len(eligible),
        "fallback": stats["n"] < MIN_LABELS,
        "fallback_output": "prior" if head == "prior10" else "zero",
        "prior_centered": head == "prior10",
        "feature_penalty": penalty,
        "intercept_penalty": intercept_penalty,
        "free_intercept": free_intercept,
        "constrained_zero_intercept": head == "scratch10_zero",
    }
