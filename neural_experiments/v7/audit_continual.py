"""Separate-code audit for the explicitly amended V7B sparse-feedback study.

No producer/controller/scorer/importer is imported. The immutable V7A and prior
independent numeric references supply filtering, export and sklearn primitives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neural_experiments.v7.audit import (
    CONFIG,
    ROOT,
    Inputs,
    average,
    burned_mask,
    compare_structure,
    contained,
    mask_hash,
    numeric,
    read,
    reference_representation,
    replay,
    score_reference,
    sha256,
    unique,
    validate_normalization,
    verify,
)

PLAN_SHA = "4f88a6cf98dd7147691f4eaf64143784736863e1cfa4fd5fbd7b3fadd1eaea23"
THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
METHODS = ("zero", "frozen120", "fixed240", "periodic240", "triggered240")
SLOTS = (10500, 13500, 16500, 19500, 22500, 25500)
PERIODIC = (10500, 16500, 22500, 25500)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def requested_mask(n, intervals):
    """Complete purchase geometry; no task mask influences what was requested."""
    mask = np.zeros(n, bool)
    last = 1500
    for a, b in intervals:
        if type(a) is not int or type(b) is not int or not last <= a < b <= n:
            raise ValueError("Label intervals overlap, regress, or include future labels")
        mask[a:b] = True
        last = b
    return mask


def fit_reference(arrays, receipt, raw, labels, segments, channels, stop, intervals):
    """Reconstruct purchased-label mean-loss ridge with fixed first30 statistics."""
    if not 1500 <= stop <= len(raw):
        raise ValueError("Fit must follow observed normalization warmup")
    purchase = requested_mask(stop, intervals)
    selected = burned_mask(labels["eval_mask"], segments)[:stop] & purchase
    selected[:1500] = False
    index = np.flatnonzero(selected)
    n = len(index)
    expected = {
        "fit_stop_bin": stop,
        "normalization_start_bin": 0,
        "normalization_stop_bin": 1500,
        "half_life_seconds": None,
        "training_rows": n,
        "training_mask_sha256": mask_hash(selected),
        "training_weights_sha256": array_hash(np.ones(n, np.float64)),
        "weight_sum": float(n),
        "effective_sample_size": float(n),
        "first_training_bin": int(index[0]) if n else None,
        "last_training_bin": int(index[-1]) if n else None,
        "feature_penalty": 1.0,
        "free_intercept": True,
        "fallback_zero": n < 50,
    }
    compare_structure(receipt, expected, "cumulative purchased fit receipt")
    validate_normalization(arrays, raw[:1500], 1500, channels, CONFIG)
    weights = arrays["weights"]
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite fitted coefficients")
    if n < 50:
        if np.any(weights != 0):
            raise ValueError("Sparse fit must export exact zero fallback")
        return 0.0
    x = np.c_[
        np.clip((raw[:stop][selected] - arrays["mean"]) / arrays["scale"], -8, 8),
        np.ones(n),
    ]
    y = labels["y"][:stop][selected].astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite purchased labels")
    gram, rhs = x.T @ x / n, x.T @ y / n
    diagonal = np.r_[np.ones(len(weights) - 1), 0.0]
    residual = float(
        np.linalg.norm((gram + np.diag(diagonal)) @ weights - rhs) / max(1.0, np.linalg.norm(rhs))
    )
    if residual > 1e-8:
        raise ValueError("Not the declared cumulative purchased-label ridge optimum")
    return residual


def drift_reference(raw, channels, at):
    """Current-to-reference diagonal Gaussian KL, computed without labels/masks."""
    if at not in SLOTS or len(raw) < at:
        raise ValueError("Unobserved or unplanned trigger boundary")
    floor = np.r_[np.full(channels, 0.1), np.full(channels, 0.01)]
    mean = raw[:1500].mean(0)
    scale = np.maximum(raw[:1500].std(0), floor)
    z = np.clip((raw[:at] - mean) / scale, -8, 8)
    ref, current = z[1500:7500], z[at - 3000 : at]
    rm, cm = ref.mean(0), current.mean(0)
    rv, cv = np.maximum(ref.var(0), 0.05), np.maximum(current.var(0), 0.05)
    kl = 0.5 * (np.log(rv / cv) + (cv + np.square(cm - rm)) / rv - 1)
    return {
        "kl_mean": float(kl.mean()),
        "current_start_bin": at - 3000,
        "current_stop_bin": at,
        "reference_start_bin": 1500,
        "reference_stop_bin": 7500,
        "variance_floor": 0.05,
        "kl_direction": "current_to_reference",
    }


def decision_reference(method, slot_index, used, last_end, kl, threshold):
    remaining, slots_left = 4 - used, 6 - slot_index
    gap = SLOTS[slot_index] - last_end
    force_quota = remaining > 0 and remaining == slots_left
    force_gap = remaining > 0 and gap >= 9000
    if remaining == 0:
        request, reason = False, "quota_exhausted"
    elif method == "periodic240":
        request = SLOTS[slot_index] in PERIODIC
        reason = "periodic_schedule" if request else "not_a_periodic_slot"
    elif force_quota:
        request, reason = True, "remaining_slots_equal_remaining_probes"
    elif force_gap:
        request, reason = True, "scheduled_maximum_gap"
    elif kl >= threshold:
        request, reason = True, "neural_drift_threshold"
    else:
        request, reason = False, "below_threshold"
    return {
        "request_probe": request,
        "reason": reason,
        "remaining_probes_before": remaining,
        "remaining_slots_including_current": slots_left,
        "gap_since_last_probe_end_seconds": gap * 0.02,
        "forced_quota": force_quota,
        "forced_gap": force_gap,
        "threshold": threshold if method == "triggered240" else None,
    }


def check_case(item, labels, raw, method, threshold, row, events, root):
    """Reconstruct all paid requests before using their labels in each fit."""
    n = item["n_bins"]
    identity = {
        "participant": item["participant_id"],
        "session_id": item["session_id"],
        "post_implant_day": item["post_implant_day"],
        "method": method,
        "threshold": threshold,
    }
    actual_fits = unique(
        [e for e in events if e["event"] == "model_fitted"], lambda e: e["fit_ordinal"]
    )
    expected_events, purchases, deployments, residuals = [], [], [], []
    paid, interruption = np.zeros(n, bool), np.zeros(n, bool)
    requested = 0.0

    def purchase(a, b, reason):
        nonlocal requested
        requested += (b - a) * 0.02
        paid[a : min(n, b)] = True
        purchases.append([a, b])
        expected_events.append(
            {
                **identity,
                "event": "labels_requested",
                "type": "probe",
                "event_bin": a,
                "label_start_bin": a,
                "label_stop_bin": b,
                "requested_label_seconds": (b - a) * 0.02,
                "available_label_seconds": max(0, min(n, b) - a) * 0.02,
                "reason": reason,
            }
        )

    def fit(stop, deploy):
        if deploy >= n:
            expected_events.append(
                {
                    **identity,
                    "event": "fit_not_observed",
                    "type": "fit",
                    "event_bin": min(n, deploy),
                    "scheduled_deployment_bin": deploy,
                    "fit_stop_bin": stop,
                }
            )
            return
        ordinal = len(deployments)
        actual = actual_fits[ordinal]
        path = contained(root, actual["model_path"])
        verify(path, actual["model_sha256"])
        arrays = numeric(path)
        receipt = actual["fit"]
        extra = {
            "deployment_bin": deploy,
            "purchased_intervals": purchases.copy(),
            "purchased_mask_sha256": mask_hash(paid[:stop]),
        }
        for key, value in extra.items():
            compare_structure(receipt[key], value, "purchased fit " + key)
        residuals.append(
            fit_reference(
                arrays,
                {k: v for k, v in receipt.items() if k not in extra},
                raw,
                labels,
                item["segments"],
                item["n_channels"],
                stop,
                purchases,
            )
        )
        deployments.append((deploy, arrays))
        expected_events.append(
            {
                **identity,
                "event": "model_fitted",
                "type": "fit",
                "event_bin": deploy,
                "fit_ordinal": ordinal,
                "fit": receipt,
                "model_path": actual["model_path"],
                "model_sha256": actual["model_sha256"],
            }
        )

    if method != "zero" and n > 1500:
        if method == "fixed240":
            purchase(1500, 13500, "fixed_startup240")
            interruption[7600 : min(n, 13600)] = True
            fit(13500, 13600)
        else:
            purchase(1500, 7500, "initial120")
            fit(7500, 7600)
            if method in ("periodic240", "triggered240"):
                used, last_end = 0, 7500
                for slot_index, at in enumerate(SLOTS):
                    if at >= n:
                        break
                    evidence = drift_reference(raw, item["n_channels"], at)
                    decision = decision_reference(
                        method, slot_index, used, last_end, evidence["kl_mean"], threshold
                    )
                    expected_events.append(
                        {
                            **identity,
                            "event": "probe_decision",
                            "type": "decision",
                            "event_bin": at,
                            "slot_index": slot_index,
                            "evidence": evidence,
                            **decision,
                            "threshold": threshold,
                        }
                    )
                    if decision["request_probe"]:
                        purchase(at, at + 1500, decision["reason"])
                        interruption[at : min(n, at + 1600)] = True
                        fit(at + 1500, at + 1600)
                        used += 1
                        last_end = at + 1500
    expected_events.sort(key=lambda e: e["event_bin"])
    compare_structure(events, expected_events, "chronological neural-only policy ledger")
    prediction = np.zeros((n, 2))
    for i, (start, arrays) in enumerate(deployments):
        end = deployments[i + 1][0] if i + 1 < len(deployments) else n
        prediction[start:end] = replay(arrays, raw[start:end])
    prediction[interruption] = 0
    use = burned_mask(labels["eval_mask"], item["segments"])
    use[:7600] = False
    trace_path = contained(root, row["trace_path"])
    verify(trace_path, row["trace_sha256"])
    trace = numeric(trace_path)
    if set(trace) != {"prediction", "eval_mask", "interruption_mask", "purchased_mask"}:
        raise ValueError("Unexpected full continual trace fields")
    for key, expected in (
        ("eval_mask", use),
        ("interruption_mask", interruption),
        ("purchased_mask", paid),
    ):
        if trace[key].dtype != bool or not np.array_equal(trace[key], expected):
            raise ValueError("Full continual purchase/interruption/score mask mismatch")
    saved = trace["prediction"]
    if saved.dtype != np.float64 or saved.shape != (n, 2) or not np.isfinite(saved).all():
        raise ValueError("Invalid full prediction geometry")
    error = float(np.max(np.abs(saved - prediction))) if n else 0.0
    if not np.allclose(saved, prediction, atol=1e-9, rtol=1e-9):
        raise ValueError("Independent chronological full prediction mismatch")
    windows = []
    for start in range(7600, 30000, 1500):
        if start >= n:
            windows.append(
                {
                    "start_bin": start,
                    "stop_bin": start,
                    "valid_rows": 0,
                    "metrics": None,
                    "score_mask_sha256": mask_hash(np.zeros(0, bool)),
                }
            )
        else:
            stop = min(start + 1500, n, 30000)
            windows.append(
                {
                    "start_bin": start,
                    "stop_bin": stop,
                    "valid_rows": int(use[start:stop].sum()),
                    "metrics": score_reference(
                        labels["y"][start:stop], saved[start:stop], use[start:stop]
                    ),
                    "score_mask_sha256": mask_hash(use[start:stop]),
                }
            )
    expected = {
        **identity,
        "n_bins": n,
        "metrics": score_reference(labels["y"], saved, use),
        "score_mask_sha256": mask_hash(use),
        "planned_label_seconds": 0 if method == "zero" else (120 if method == "frozen120" else 240),
        "unique_requested_label_seconds": requested,
        "available_label_seconds": float(paid.sum() * 0.02),
        "interruption_seconds": float(interruption.sum() * 0.02),
        "interrupted_valid_seconds": float((interruption & use).sum() * 0.02),
        "interruption_mask_sha256": mask_hash(interruption),
        "purchased_mask_sha256": mask_hash(paid),
        "completed_fits": len(deployments),
        "diagnostic_windows": windows,
        "requested_probes": max(0, len(purchases) - 1)
        if method in ("periodic240", "triggered240")
        else 0,
        "trace_path": row["trace_path"],
        "trace_sha256": row["trace_sha256"],
    }
    compare_structure(row, expected, "full-interval continual metrics and costs")
    return expected, error, residuals


def interval(values):
    if not len(values):
        return [None, None]
    a = np.asarray(values, float)
    indexes = np.random.default_rng(20261027).integers(0, len(a), size=(10000, len(a)))
    return np.quantile(a[indexes].mean(1), [0.025, 0.975]).tolist()


def selection_reference(rows):
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    choices = []
    for heldout in people:
        candidates = []
        for threshold in THRESHOLDS:
            units = []
            for person in people:
                if person == heldout:
                    continue
                values = [
                    r["metrics"]["r2"]
                    for r in rows
                    if r["participant"] == person
                    and r["method"] == "triggered240"
                    and r["threshold"] == threshold
                    and r["metrics"]["r2"] is not None
                ]
                if values:
                    units.append(
                        {"participant": person, "mean_r2": average(values), "n": len(values)}
                    )
            candidates.append(
                {
                    "threshold": threshold,
                    "source_units": units,
                    "mean_r2": average([u["mean_r2"] for u in units]),
                }
            )
        if any(c["mean_r2"] is None for c in candidates):
            raise ValueError("A threshold has no eligible source people")
        winner = max(candidates, key=lambda c: c["mean_r2"])
        choices.append(
            {
                "participant": heldout,
                "threshold": winner["threshold"],
                "source_mean_r2": winner["mean_r2"],
                "candidates": candidates,
            }
        )
    return {"choices": choices, "evidence_role": "exposed whole-person-excluded threshold ranking"}


def summary_reference(rows):
    lookup = unique(rows, lambda r: (r["session_id"], r["method"]))
    ids = sorted({r["session_id"] for r in rows})
    if set(lookup) != {(s, m) for s in ids for m in METHODS}:
        raise ValueError("Incomplete continual method/session grid")
    for sid in ids:
        paired = [lookup[sid, m] for m in METHODS]
        if (
            len({r["score_mask_sha256"] for r in paired}) != 1
            or len({r["metrics"]["r2"] is None for r in paired}) != 1
        ):
            raise ValueError("Unpaired full-interval masks/availability")
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    methods = {}
    for method in METHODS:
        units = []
        for person in people:
            requested = [r for r in rows if r["participant"] == person and r["method"] == method]
            scored = [r for r in requested if r["metrics"]["r2"] is not None]
            total, bad = 0, 0
            for row in requested:
                zero = lookup[row["session_id"], "zero"]
                for w, z in zip(row["diagnostic_windows"], zero["diagnostic_windows"], strict=True):
                    if w["metrics"] is not None and w["metrics"]["summed_mse"] is not None:
                        total += w["valid_rows"]
                        if w["metrics"]["summed_mse"] > z["metrics"]["summed_mse"] + 1e-12:
                            bad += w["valid_rows"]
            units.append(
                {
                    "participant": person,
                    "selected_sessions": len(requested),
                    "scored_sessions": len(scored),
                    "mean_r2": average([r["metrics"]["r2"] for r in scored]),
                    "positive_sessions": sum(r["metrics"]["r2"] > 0 for r in scored),
                    "mean_summed_mse": average([r["metrics"]["summed_mse"] for r in scored]),
                    "mean_angular_error_degrees": average(
                        [r["metrics"]["angular_error_degrees"] for r in scored]
                    ),
                    "mean_planned_label_seconds": average(
                        [r["planned_label_seconds"] for r in requested]
                    ),
                    "mean_available_label_seconds": average(
                        [r["available_label_seconds"] for r in requested]
                    ),
                    "mean_interruption_seconds": average(
                        [r["interruption_seconds"] for r in requested]
                    ),
                    "harmful_window_valid_time_fraction": bad / total if total else None,
                    "harmful_sessions": [
                        r["session_id"]
                        for r in scored
                        if r["metrics"]["summed_mse"]
                        > lookup[r["session_id"], "zero"]["metrics"]["summed_mse"] + 1e-12
                    ],
                }
            )
        scored_people = [u for u in units if u["mean_r2"] is not None]
        values = [u["mean_r2"] for u in scored_people]
        methods[method] = {
            "participants": units,
            "mean_r2": average(values),
            "participant_bootstrap_95": interval(values),
            "positive_participants": sum(v > 0 for v in values),
            "n_participants": len(scored_people),
            "scored_sessions": sum(u["scored_sessions"] for u in units),
            "mean_available_label_seconds": average(
                [u["mean_available_label_seconds"] for u in units]
            ),
            "mean_interruption_seconds": average([u["mean_interruption_seconds"] for u in units]),
        }
    contrasts = {}
    for a, b in (
        ("periodic240", "frozen120"),
        ("periodic240", "fixed240"),
        ("triggered240", "periodic240"),
        ("triggered240", "frozen120"),
    ):
        units, t2 = [], []
        for person in people:
            gains = [
                lookup[s, a]["metrics"]["r2"] - lookup[s, b]["metrics"]["r2"]
                for s in ids
                if lookup[s, a]["participant"] == person
                and lookup[s, a]["metrics"]["r2"] is not None
            ]
            if gains:
                units.append({"participant": person, "mean_gain": average(gains)})
            if person == "T2":
                t2 = gains
        values = [u["mean_gain"] for u in units]
        contrasts[f"{a}_minus_{b}"] = {
            "participants": units,
            "mean_gain": average(values),
            "paired_participant_95": interval(values),
            "T2_mean_gain": average(t2),
            "T2_paired_session_95": interval(t2),
        }
    primary = {}
    for method in METHODS:
        requested = [r for r in rows if r["participant"] == "T2" and r["method"] == method]
        values = [r["metrics"]["r2"] for r in requested if r["metrics"]["r2"] is not None]
        primary[method] = {
            "selected_sessions": len(requested),
            "scored_sessions": len(values),
            "mean_r2": average(values),
            "session_bootstrap_95": interval(values),
            "positive_sessions": sum(v > 0 for v in values),
        }
    t = primary["periodic240"]
    gain = contrasts["periodic240_minus_frozen120"]
    timing = contrasts["triggered240_minus_periodic240"]
    full = [s for s in ids if lookup[s, "periodic240"]["n_bins"] == 30000]
    equal = bool(full) and all(
        abs(
            lookup[s, "periodic240"]["available_label_seconds"]
            - lookup[s, "triggered240"]["available_label_seconds"]
        )
        < 1e-12
        for s in full
    )
    gates = {
        "periodic_T2_recovery": bool(
            t["selected_sessions"] == t["scored_sessions"] == 3
            and t["mean_r2"] >= 0.05
            and t["positive_sessions"] >= 2
            and gain["T2_mean_gain"] >= 0.05
            and gain["T2_paired_session_95"][0] > 0
        ),
        "periodic_cohort_gain": bool(
            methods["periodic240"]["n_participants"] == 8
            and gain["mean_gain"] >= 0.02
            and gain["paired_participant_95"][0] > 0
        ),
        "triggered_timing_value": bool(
            equal
            and timing["mean_gain"] is not None
            and timing["mean_gain"] >= 0.01
            and timing["paired_participant_95"][0] > 0
            and primary["triggered240"]["mean_r2"] is not None
            and t["mean_r2"] is not None
            and primary["triggered240"]["mean_r2"] >= t["mean_r2"]
        ),
    }
    return {
        "selected_sessions": len(ids),
        "methods": methods,
        "T2": primary,
        "contrasts": contrasts,
        "complete_sessions_matched_feedback_budget": equal,
        "gates": gates,
    }


def audit_run(path, root=ROOT):
    root, path = Path(root).resolve(), Path(path).resolve()
    manifest = read(path)
    if (
        manifest["schema_version"] != 1
        or manifest["study"] != "V7B"
        or manifest["status"] != "complete"
        or manifest["phase"] not in ("source", "assessment")
    ):
        raise ValueError("Incomplete or wrong continual experiment")
    if manifest["plan"]["sha256"] != PLAN_SHA:
        raise ValueError("Wrong amended continual plan")
    plan_path = contained(root, manifest["plan"]["path"])
    verify(plan_path, PLAN_SHA)
    plan = read(plan_path)
    sources = {
        f"neural_experiments/{v}/{name}.py"
        for v, names in (
            (
                "v7",
                (
                    "continual",
                    "continual_metrics",
                    "study_continual",
                    "model",
                    "data",
                    "fresh_data",
                ),
            ),
            ("v6", ("data",)),
            ("v5", ("data",)),
            ("v4", ("model", "data")),
            ("v3", ("study", "model", "data")),
            ("v2", ("data",)),
        )
        for name in names
    }
    if set(manifest["code_sha256"]) != sources:
        raise ValueError("Executed scientific source inventory differs")
    for name, digest in manifest["code_sha256"].items():
        verify(contained(root, name), digest)
        verify(contained(path.parent / "source_snapshot", name), digest)
    index_path = contained(root, manifest["index"]["path"])
    verify(index_path, manifest["index"]["sha256"])
    if manifest["phase"] != "source":
        raise ValueError("Assessment binding requires completed source publication")
    if (
        manifest["index"]["sha256"] != plan["development_index_sha256"]
        or manifest["source_selection"] is not None
    ):
        raise ValueError("Source role/index mismatch")
    inputs = Inputs(index_path)
    rows = unique(manifest["rows"], lambda r: (r["session_id"], r["threshold"], r["method"]))
    expected_keys = {(s, t, m) for s in inputs.items for t in THRESHOLDS for m in METHODS}
    if set(rows) != expected_keys:
        raise ValueError("Incomplete continual case grid")
    grouped = {key: [] for key in expected_keys}
    for event in manifest["events"]:
        grouped[event["session_id"], event["threshold"], event["method"]].append(event)
    checked, errors, residuals = [], [], []
    for sid, item in inputs.items.items():
        features, labels = inputs.arrays(item)
        raw = reference_representation(features, item["segments"], CONFIG)
        for threshold in THRESHOLDS:
            for method in METHODS:
                key = sid, threshold, method
                row, error, fitted = check_case(
                    item, labels, raw, method, threshold, rows[key], grouped[key], root
                )
                checked.append(row)
                errors.append(error)
                residuals.extend(fitted)
        print(f"independently checked continual {sid}", flush=True)
    selection = selection_reference(checked)
    compare_structure(manifest["selection"], selection, "whole-person source choice")
    choices = {r["participant"]: r["threshold"] for r in selection["choices"]}
    selected = [r for r in checked if r["threshold"] == choices[r["participant"]]]
    summary = summary_reference(selected)
    compare_structure(manifest["summary"], summary, "independent full-interval summaries")
    return {
        "status": "PASS",
        "stage": manifest["phase"],
        "created_at": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": sha256(__file__),
        "manifest_sha256": sha256(path),
        "plan_sha256": PLAN_SHA,
        "index_sha256": sha256(index_path),
        "code_sha256": manifest["code_sha256"],
        "sessions_checked": len(inputs.items),
        "cases_checked": len(checked),
        "models_checked": len(residuals),
        "full_traces_checked": len(errors),
        "events_checked": len(manifest["events"]),
        "max_prediction_absolute_error": max(errors, default=0.0),
        "max_normal_equation_relative_residual": max(residuals, default=0.0),
        "selection": selection,
        "summary": summary,
        "scope": (
            "Separate-code neural-only KL decisions, cumulative paid-label ridge, full deployment "
            "replay including zero interruptions, sklearn scoring, equal-person source selection "
            "and bootstrap. Reuses independent V7A/V2-V4 audit primitives; no producer imports "
            "and no external replication claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit receipts cannot be overwritten")
    path = args.run / "manifest.json" if args.run.is_dir() else args.run
    result = audit_run(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("summary", "selection", "code_sha256")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
