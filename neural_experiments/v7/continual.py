"""Chronological, label-budgeted updates retaining every purchased past label.

This separate V7B engine leaves the failed V7A recency gate intact. Neural-only
drift can trigger a paid label burst; it never exposes its future labels. All
calibration and fixed fitting delays yield zero output in the common score.
"""

from __future__ import annotations

import numpy as np

from neural_experiments.v3.study import mask_digest, score
from neural_experiments.v4 import model as core
from neural_experiments.v7 import model

METHODS = ("zero", "frozen120", "fixed240", "periodic240", "triggered240")
THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
SLOTS_SECONDS = (210, 270, 330, 390, 450, 510)
PERIODIC_SECONDS = (210, 330, 450, 510)
INITIAL_START = 1500
INITIAL_STOP = 7500
INITIAL_DEPLOY = 7600
PROBE_BINS = 1500
FIT_DELAY_BINS = 100
MAX_PROBES = 4
MAX_GAP_BINS = 9000


def drift_evidence(raw, channels, start_bin):
    """Diagonal KL(current past60 || reference30..150), using neural inputs only.

    Features follow the exact causal decoder representation and first30
    normalization/clipping. Variances, not standard deviations, have floor .05.
    No label, target sector, trial identity, or future neural observation enters.
    """
    raw = np.asarray(raw)
    if (
        raw.ndim != 2
        or type(channels) is not int
        or raw.shape[1] != 2 * channels
        or type(start_bin) is not int
        or start_bin < 10500
        or start_bin > len(raw)
        or not np.isfinite(raw[:start_bin]).all()
    ):
        raise ValueError("complete causal neural prefix and valid candidate slot required")
    normal = np.asarray(raw[:1500], np.float64)
    floor = np.r_[np.full(channels, 0.1), np.full(channels, 0.01)]
    mean, scale = normal.mean(0), np.maximum(normal.std(0), floor)
    reference = np.clip((raw[1500:7500] - mean) / scale, -8, 8)
    recent = np.clip((raw[start_bin - 3000 : start_bin] - mean) / scale, -8, 8)
    reference_mean, recent_mean = reference.mean(0), recent.mean(0)
    reference_variance = np.maximum(reference.var(0), 0.05)
    recent_variance = np.maximum(recent.var(0), 0.05)
    per_feature = 0.5 * (
        np.log(reference_variance / recent_variance)
        + (recent_variance + np.square(recent_mean - reference_mean)) / reference_variance
        - 1
    )
    return {
        "kl_mean": float(np.mean(per_feature)),
        "current_start_bin": start_bin - 3000,
        "current_stop_bin": start_bin,
        "reference_start_bin": 1500,
        "reference_stop_bin": 7500,
        "variance_floor": 0.05,
        "kl_direction": "current_to_reference",
    }


def choose_probe(method, slot_index, used_probes, last_probe_end_bin, kl, threshold):
    """A pure decision on observed inputs; future recording length is not an input."""
    if (
        method not in ("periodic240", "triggered240")
        or type(slot_index) is not int
        or not 0 <= slot_index < len(SLOTS_SECONDS)
        or type(used_probes) is not int
        or not 0 <= used_probes <= MAX_PROBES
        or type(last_probe_end_bin) is not int
        or last_probe_end_bin < INITIAL_STOP
        or threshold not in THRESHOLDS
        or isinstance(threshold, (bool, np.bool_))
        or not np.isfinite(kl)
        or kl < -1e-12
    ):
        raise ValueError("valid chronological probe-decision inputs required")
    seconds = SLOTS_SECONDS[slot_index]
    remaining_probes = MAX_PROBES - used_probes
    remaining_slots = len(SLOTS_SECONDS) - slot_index
    gap = seconds * 50 - last_probe_end_bin
    if gap < 0 or remaining_probes > remaining_slots:
        raise ValueError("impossible remaining budget or probe chronology")
    forced_quota = remaining_probes > 0 and remaining_probes == remaining_slots
    forced_gap = remaining_probes > 0 and gap >= MAX_GAP_BINS
    if remaining_probes == 0:
        request, reason = False, "quota_exhausted"
    elif method == "periodic240":
        request = seconds in PERIODIC_SECONDS
        reason = "periodic_schedule" if request else "not_a_periodic_slot"
    elif forced_quota:
        request, reason = True, "remaining_slots_equal_remaining_probes"
    elif forced_gap:
        request, reason = True, "scheduled_maximum_gap"
    elif kl >= threshold:
        request, reason = True, "neural_drift_threshold"
    else:
        request, reason = False, "below_threshold"
    return {
        "request_probe": request,
        "reason": reason,
        "remaining_probes_before": remaining_probes,
        "remaining_slots_including_current": remaining_slots,
        "gap_since_last_probe_end_seconds": gap * model.DT,
        "forced_quota": forced_quota,
        "forced_gap": forced_gap,
        "threshold": threshold if method == "triggered240" else None,
    }


def _metric(y, prediction, use):
    values = score(y, prediction, use)
    values["summed_mse"] = (
        float(np.square(y[use] - prediction[use]).sum(1).mean()) if use.sum() >= 50 else None
    )
    return values


def evaluate_session(case, threshold, save_artifact=None):
    """Return all five policy rows and their complete chronological event ledgers.

    Model callback: ``save_artifact('model', method, ordinal, payload)`` receives
    ``decoder`` and ``fit``; it returns ``decoder`` (the reloaded exact artifact)
    and ``record`` (JSON model references). Trace callback receives prediction,
    eval_mask and interruption_mask and returns JSON artifact references. With
    no callback, the same immutable numeric decoder is used in synthetic tests.
    """
    if threshold not in THRESHOLDS or isinstance(threshold, (bool, np.bool_)):
        raise ValueError("one locked V7B threshold required")
    item, features = case["item"], case["features"]
    segments, y, label_valid = case["segments"], case["y"], case["eval_mask"]
    raw = model.representation(features, segments, model.CONFIG)
    n = len(raw)
    if len(y) != n or n > 30000:
        raise ValueError("aligned exposure prefix of at most600 seconds required")
    score_mask = core.training_mask(label_valid, segments)
    score_mask[:INITIAL_DEPLOY] = False
    rows, events = [], []
    for method in METHODS:
        identity = {
            "participant": item["participant_id"],
            "session_id": item["session_id"],
            "post_implant_day": item["post_implant_day"],
            "method": method,
        }
        prediction = np.zeros((n, 2), np.float64)
        interruptions = np.zeros(n, bool)
        paid = np.zeros(n, bool)
        purchases, fits = [], []
        requested = 0.0
        method_events = []

        def purchase(start, stop, reason):
            nonlocal requested
            requested += (stop - start) * model.DT
            paid[min(start, n) : min(stop, n)] = True
            purchases.append([start, stop])
            method_events.append(
                {
                    **identity,
                    "event": "labels_requested",
                    "type": "probe",
                    "event_bin": start,
                    "label_start_bin": start,
                    "label_stop_bin": stop,
                    "requested_label_seconds": (stop - start) * model.DT,
                    "available_label_seconds": max(0, min(stop, n) - start) * model.DT,
                    "reason": reason,
                }
            )

        def fit(stop, deploy):
            if deploy >= n:
                method_events.append(
                    {
                        **identity,
                        "event": "fit_not_observed",
                        "type": "fit",
                        "event_bin": min(deploy, n),
                        "scheduled_deployment_bin": deploy,
                        "fit_stop_bin": stop,
                    }
                )
                return None
            selected = paid[:stop] & label_valid[:stop]
            decoder, receipt = model.fit_mask(
                raw, y[:stop], selected, segments, item["n_channels"], stop
            )
            receipt = {
                **receipt,
                "deployment_bin": deploy,
                "purchased_intervals": [list(interval) for interval in purchases],
                "purchased_mask_sha256": mask_digest(paid[:stop]),
            }
            references = {}
            if save_artifact is not None:
                result = save_artifact(
                    "model", method, len(fits), {"decoder": decoder, "fit": receipt}
                )
                exported = result["decoder"]
                if exported.config != decoder.config:
                    raise ValueError("exported decoder configuration differs")
                for key in ("mean", "scale", "weights"):
                    np.testing.assert_array_equal(getattr(exported, key), getattr(decoder, key))
                decoder, references = exported, result["record"]
            fits.append(receipt)
            method_events.append(
                {
                    **identity,
                    "event": "model_fitted",
                    "type": "fit",
                    "event_bin": deploy,
                    "fit_ordinal": len(fits) - 1,
                    "fit": receipt,
                    **references,
                }
            )
            return decoder

        if method == "zero" or n <= INITIAL_START:
            decoder = None
        elif method == "fixed240":
            purchase(INITIAL_START, 13500, "fixed_startup240")
            interruptions[INITIAL_DEPLOY : min(13600, n)] = True
            decoder = fit(13500, 13600)
            if decoder is not None:
                prediction[13600:] = model.predict_raw(decoder, raw[13600:])
        else:
            purchase(INITIAL_START, INITIAL_STOP, "initial120")
            decoder = fit(INITIAL_STOP, INITIAL_DEPLOY)
            cursor = INITIAL_DEPLOY
            if method == "frozen120":
                if decoder is not None:
                    prediction[cursor:] = model.predict_raw(decoder, raw[cursor:])
            else:
                used_probes, last_probe_end = 0, INITIAL_STOP
                for slot_index, seconds in enumerate(SLOTS_SECONDS):
                    start = seconds * 50
                    if start >= n:
                        break
                    if decoder is not None:
                        prediction[cursor:start] = model.predict_raw(decoder, raw[cursor:start])
                    evidence = drift_evidence(raw, item["n_channels"], start)
                    decision = choose_probe(
                        method,
                        slot_index,
                        used_probes,
                        last_probe_end,
                        evidence["kl_mean"],
                        threshold,
                    )
                    method_events.append(
                        {
                            **identity,
                            "event": "probe_decision",
                            "type": "decision",
                            "event_bin": start,
                            "slot_index": slot_index,
                            "evidence": evidence,
                            **decision,
                        }
                    )
                    cursor = start
                    if decision["request_probe"]:
                        end = start + PROBE_BINS
                        deploy = end + FIT_DELAY_BINS
                        purchase(start, end, decision["reason"])
                        interruptions[start : min(deploy, n)] = True
                        decoder = fit(end, deploy)
                        used_probes += 1
                        last_probe_end = end
                        cursor = min(deploy, n)
                if decoder is not None and cursor < n:
                    prediction[cursor:] = model.predict_raw(decoder, raw[cursor:])
        method_events.sort(key=lambda event: event["event_bin"])
        events.extend(method_events)
        windows = []
        for start in range(INITIAL_DEPLOY, 30000, 1500):
            stop = min(start + 1500, n, 30000)
            if start >= n:
                windows.append(
                    {
                        "start_bin": start,
                        "stop_bin": start,
                        "valid_rows": 0,
                        "metrics": None,
                        "score_mask_sha256": mask_digest(np.zeros(0, bool)),
                    }
                )
            else:
                windows.append(
                    {
                        "start_bin": start,
                        "stop_bin": stop,
                        "valid_rows": int(score_mask[start:stop].sum()),
                        "metrics": _metric(
                            y[start:stop], prediction[start:stop], score_mask[start:stop]
                        ),
                        "score_mask_sha256": mask_digest(score_mask[start:stop]),
                    }
                )
        row = {
            **identity,
            "threshold": threshold if method == "triggered240" else None,
            "n_bins": n,
            "metrics": _metric(y, prediction, score_mask),
            "score_mask_sha256": mask_digest(score_mask),
            "planned_label_seconds": 0
            if method == "zero"
            else (120 if method == "frozen120" else 240),
            "unique_requested_label_seconds": requested,
            "available_label_seconds": float(paid.sum() * model.DT),
            "interruption_seconds": float(interruptions.sum() * model.DT),
            "interrupted_valid_seconds": float((interruptions & score_mask).sum() * model.DT),
            "interruption_mask_sha256": mask_digest(interruptions),
            "purchased_mask_sha256": mask_digest(paid),
            "completed_fits": len(fits),
            "diagnostic_windows": windows,
            "requested_probes": len(purchases) - 1
            if method.endswith("240") and method != "fixed240" and purchases
            else 0,
        }
        if save_artifact is not None:
            row.update(
                save_artifact(
                    "trace",
                    method,
                    0,
                    {
                        "prediction": prediction,
                        "eval_mask": score_mask,
                        "interruption_mask": interruptions,
                        "purchased_mask": paid,
                    },
                )
            )
        rows.append(row)
    return rows, events
