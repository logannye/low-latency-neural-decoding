"""Separate-code verification of the locked V7 exposed recoverability screen.

The producer, model, scorer and importer are never imported. The established
independent NumPy recurrence and sklearn score references are reused. This is
arithmetic/protocol verification, not external scientific replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit import compare_structure, score_arrays
from neural_experiments.v3.audit import (
    contained,
    mask_hash,
    numeric,
    read,
    segment_ranges,
    sha256,
    verify,
)
from neural_experiments.v4.audit import Inputs as ArrayInputs
from neural_experiments.v4.audit import (
    reference_representation,
    replay,
    validate_normalization,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = "28ead298081c41b6028bffa0cf9aa5ee0eeb7811917400863aff350525aa7671"
PARENT_PINS = {
    "development": "357f8b917abec17f6354dac3cd93b4d87d24df0db4bf60afeee8bdbefbf84678",
    "new_sessions": "dfe5ed95bf7f43b0bb51f3a11d163505e99641a4affc86daf38591fbdc5af534",
}
CONFIG = {
    "family": "tx45_sbp",
    "tau_seconds": 0.2,
    "baseline_tau_seconds": None,
    "adapt_gamma": 10.0,
}
METHODS = (
    "zero",
    "frozen120",
    "expanding",
    "rolling60",
    "rolling120",
    "rolling240",
    "exponential60",
    "rolling120_local",
)
ELIGIBLE = ("rolling60", "rolling120", "rolling240", "exponential60", "rolling120_local")
UPDATES = (13600, 16600, 19600, 22600, 25600, 28600)


def burned_mask(mask, segments):
    """Acquisition burn uses physical segment starts, not fit-window starts."""
    value = np.asarray(mask)
    if value.dtype != bool or value.ndim != 1:
        raise ValueError("Boolean scoring/training mask required")
    result = value.copy()
    for a, b in segment_ranges(segments, len(result)):
        result[a : min(a + 62, b)] = False
    return result


def fit_geometry(method, update):
    if method not in METHODS[1:] or update not in UPDATES:
        raise ValueError("Unplanned method or update boundary")
    stop = 7500 if method == "frozen120" else update - 100
    start = 1500
    if method.startswith("rolling"):
        start = stop - int(method[7:].removesuffix("_local")) * 50
    normal = (start, stop) if method == "rolling120_local" else (0, 1500)
    return start, stop, normal


def unique_label_seconds(intervals):
    """Union length charges each requested label once, including invalid-go bins."""
    total, right = 0, -1
    for a, b in sorted(intervals):
        if type(a) is not int or type(b) is not int or not 1500 <= a < b <= 30000:
            raise ValueError("Invalid purchased-label interval")
        total += max(0, b - max(a, right))
        right = max(right, b)
    return total * 0.02


def fit_reference(raw, y, mask, segments, channels, method, update):
    """Direct weighted normal equations over the declared past-only interval."""
    start, stop, (na, nb) = fit_geometry(method, update)
    raw, y, mask = np.asarray(raw), np.asarray(y), np.asarray(mask)
    if (
        raw.ndim != 2
        or raw.shape[1] != 2 * channels
        or len(raw) < update
        or y.shape != (len(raw), 2)
        or mask.shape != (len(raw),)
        or mask.dtype != bool
        or not np.isfinite(raw[:stop]).all()
    ):
        raise ValueError("Invalid full exposed-array geometry")
    selected = burned_mask(mask, segments)[:stop]
    selected[:start] = False
    indices = np.flatnonzero(selected)
    if not np.isfinite(y[:stop][selected]).all():
        raise ValueError("Nonfinite selected training labels")
    floor = np.r_[np.full(channels, 0.1), np.full(channels, 0.01)]
    mean = np.mean(raw[na:nb], axis=0)
    scale = np.maximum(np.std(raw[na:nb], axis=0), floor)
    x = np.c_[np.clip((raw[:stop][selected] - mean) / scale, -8, 8), np.ones(len(indices))]
    yy = np.asarray(y[:stop][selected], np.float64)
    weights = np.ones(len(indices), np.float64)
    half = 60.0 if method == "exponential60" else None
    if half is not None:
        weights = np.exp2(-(stop - 1 - indices) * 0.02 / half)
    mass = float(weights.sum())
    d = raw.shape[1] + 1
    if len(indices):
        root_weight = np.sqrt(weights / mass)
        xx = x * root_weight[:, None]
        gram, rhs = xx.T @ xx, xx.T @ (yy * root_weight[:, None])
    else:
        gram, rhs = np.zeros((d, d)), np.zeros((d, 2))
    receipt = {
        "fit_stop_bin": stop,
        "normalization_start_bin": na,
        "normalization_stop_bin": nb,
        "half_life_seconds": half,
        "training_rows": len(indices),
        "training_mask_sha256": mask_hash(selected),
        "training_weights_sha256": hashlib.sha256(
            np.ascontiguousarray(weights).tobytes()
        ).hexdigest(),
        "weight_sum": mass,
        "effective_sample_size": mass**2 / float(weights @ weights) if len(indices) else 0.0,
        "first_training_bin": int(indices[0]) if len(indices) else None,
        "last_training_bin": int(indices[-1]) if len(indices) else None,
        "feature_penalty": 1.0,
        "free_intercept": True,
        "fallback_zero": len(indices) < 50,
        "method": method,
        "update_bin": update,
        "label_start_bin": start,
        "label_stop_bin": stop,
        "normalization": "local_training_interval" if method.endswith("_local") else "first30",
        "embargo_seconds": (update - stop) * 0.02,
        "selected_interval_seconds": (stop - start) * 0.02,
        "available_label_seconds": (stop - 1500) * 0.02,
    }
    return receipt, mean, scale, gram, rhs


def validate_fit(arrays, recorded, raw, labels, segments, channels, method, update):
    receipt, mean, scale, gram, rhs = fit_reference(
        raw, labels["y"], labels["eval_mask"], segments, channels, method, update
    )
    compare_structure(recorded, receipt, "fit receipt")
    a, b = receipt["normalization_start_bin"], receipt["normalization_stop_bin"]
    validate_normalization(arrays, raw[a:b], b - a, channels, CONFIG)
    for name, expected in (("mean", mean), ("scale", scale)):
        if not np.allclose(arrays[name], expected, atol=1e-10, rtol=1e-10):
            raise ValueError("Wrong past-only normalization")
    coefficient = arrays["weights"]
    if not np.isfinite(coefficient).all():
        raise ValueError("Nonfinite exported coefficients")
    if receipt["fallback_zero"]:
        if np.any(coefficient != 0):
            raise ValueError("Insufficient-label fallback must be exactly zero")
        return 0.0
    penalty = np.diag(np.r_[np.ones(len(coefficient) - 1), 0.0])
    residual = float(
        np.linalg.norm((gram + penalty) @ coefficient - rhs) / max(1.0, np.linalg.norm(rhs))
    )
    if residual > 1e-8:
        raise ValueError("Model is not the declared weighted ridge optimum")
    return residual


def score_reference(y, prediction, mask):
    result = score_arrays(y, prediction, mask)
    n = result["n"]
    if n < 50:
        for key in ("r2", "angular_error_degrees"):
            result[key] = None
        result["per_output_r2"] = [None, None]
        result["summed_mse"] = None
        return result
    yy, pp = np.asarray(y, float)[mask], np.asarray(prediction, float)[mask]
    result.update(
        {
            "sse": np.square(yy - pp).sum(0).tolist(),
            "sst": np.square(yy - yy.mean(0)).sum(0).tolist(),
            "prediction_norm_rms": float(np.sqrt(np.square(pp).sum(1).mean())),
            "summed_mse": float(np.square(yy - pp).sum(1).mean()),
        }
    )
    return result


def interval(values):
    if not len(values):
        return [None, None]
    a = np.asarray(values, np.float64)
    draws = np.random.default_rng(20261026).integers(0, len(a), size=(10000, len(a)))
    return np.quantile(a[draws].mean(1), [0.025, 0.975]).tolist()


def average(values):
    return float(np.mean(values)) if len(values) else None


def unique(rows, key):
    result = {key(r): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate case/artifact identity")
    return result


class Inputs(ArrayInputs):
    def __init__(self, path):
        self.path, self.index, self.checked = Path(path).resolve(), read(path), set()
        index = self.index
        people = ["T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11"]
        if (
            index["schema_version"] != 1
            or index["role"] != "development"
            or index["participants"] != people
            or index["provider"] != {"doi": "10.5061/dryad.x0k6djj1h", "version_id": 462702}
            or index["exposure"]
            != {
                "rule": "exact_union_of_exposed_v6_development_and_new_session_prefixes",
                "sessions": 152,
                "maximum_bins_per_session": 30000,
                "continuation_tails_included": False,
                "fresh_data_included": False,
            }
            or set(index["parents"]) != set(PARENT_PINS)
        ):
            raise ValueError("Wrong exposed cohort role/provider/extent")
        parent_rows = {}
        for role, pin in PARENT_PINS.items():
            ref = index["parents"][role]
            if ref["sha256"] != pin:
                raise ValueError("Wrong authenticated V6 parent")
            parent_path = Path(ref["path"]).resolve()
            verify(parent_path, pin)
            parent = read(parent_path)
            if (
                parent["role"] != role
                or parent["feature_definition"] != index["feature_definition"]
            ):
                raise ValueError("Parent role/features changed")
            for item in parent["sessions"]:
                if item["session_id"] in parent_rows:
                    raise ValueError("Parent sessions overlap")
                parent_rows[item["session_id"]] = (item, parent_path, role, pin)
        self.items = unique(index["sessions"], lambda r: r["session_id"])
        if set(self.items) != set(parent_rows) or len(self.items) != 152:
            raise ValueError("Missing or additional exposed session")
        for sid, item in self.items.items():
            parent, parent_path, role, pin = parent_rows[sid]
            if (
                set(item) != set(parent) | {"v7_parent", "reused_parent_artifacts"}
                or item["role"] != "development"
                or not 0 <= item["n_bins"] <= 30000
                or item["bin_seconds"] != 0.02
            ):
                raise ValueError("Prepared session fields/extent changed")
            for field in parent.keys() - {
                "role",
                "feature_path",
                "target_path",
                "reused_parent_artifacts",
            }:
                compare_structure(item[field], parent[field], "exposed parent " + field)
            compare_structure(
                item["v7_parent"],
                {
                    "index_path": str(parent_path),
                    "index_sha256": pin,
                    "role": role,
                    "session_id": sid,
                },
                "parent binding",
            )
            if set(item["reused_parent_artifacts"]) != {"feature", "target"}:
                raise ValueError("Incomplete parent artifact identities")
            for kind in ("feature", "target"):
                compare_structure(
                    item["reused_parent_artifacts"][kind],
                    {
                        "path": str(contained(parent_path.parent, parent[kind + "_path"])),
                        "sha256": parent[kind + "_sha256"],
                        "mode": "immutable_hard_link",
                    },
                    "parent artifact",
                )
            segment_ranges(item["segments"], item["n_bins"])
        if any(
            sum(r["participant_id"] == p for r in self.items.values()) != (8 if p == "T3" else 18)
            for p in people
        ):
            raise ValueError("Exposed participant count changed")


def summarize_reference(rows, window_rows):
    lookup = unique(rows, lambda r: (r["session_id"], r["method"]))
    window = unique(window_rows, lambda r: (r["session_id"], r["method"], r["start_bin"]))
    sessions = sorted({sid for sid, _ in lookup})
    if not sessions or set(lookup) != {(s, m) for s in sessions for m in METHODS}:
        raise ValueError("Incomplete declared session/method inventory")
    if set(window) != {(s, m, t) for s in sessions for m in METHODS for t in UPDATES}:
        raise ValueError("Incomplete declared window inventory")
    for sid in sessions:
        group = [lookup[sid, m] for m in METHODS]
        if (
            len({r["score_mask_sha256"] for r in group}) != 1
            or len({r["metrics"]["r2"] is None for r in group}) != 1
            or len({(r["participant"], r["post_implant_day"]) for r in group}) != 1
        ):
            raise ValueError("Unpaired session masks/availability/identity")
        for t in UPDATES:
            if len({window[sid, m, t]["score_mask_sha256"] for m in METHODS}) != 1:
                raise ValueError("Unpaired window masks")
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    primary_ids = [
        r["session_id"]
        for r in sorted(
            (lookup[s, "zero"] for s in sessions if lookup[s, "zero"]["participant"] == "T2"),
            key=lambda r: (r["post_implant_day"], r["session_id"]),
        )[-6:]
    ]
    summaries, primary = {}, {}
    for method in METHODS:
        participant_rows = []
        for person in people:
            requested = [
                lookup[s, method] for s in sessions if lookup[s, method]["participant"] == person
            ]
            scored = [r for r in requested if r["metrics"]["r2"] is not None]
            gains = [
                r["metrics"]["r2"] - lookup[r["session_id"], "frozen120"]["metrics"]["r2"]
                for r in scored
            ]
            total, bad = 0, 0
            for r in requested:
                for t in UPDATES:
                    w = window[r["session_id"], method, t]
                    if w["metrics"] is None or w["metrics"]["summed_mse"] is None:
                        continue
                    total += w["valid_rows"]
                    zero = window[r["session_id"], "zero", t]
                    if w["metrics"]["summed_mse"] > zero["metrics"]["summed_mse"] + 1e-12:
                        bad += w["valid_rows"]
            participant_rows.append(
                {
                    "participant": person,
                    "selected_sessions": len(requested),
                    "scored_sessions": len(scored),
                    "mean_r2": average([r["metrics"]["r2"] for r in scored]),
                    "gain_over_frozen120": average(gains),
                    "gain_over_zero": average(
                        [
                            r["metrics"]["r2"] - lookup[r["session_id"], "zero"]["metrics"]["r2"]
                            for r in scored
                        ]
                    ),
                    "mean_summed_mse": average([r["metrics"]["summed_mse"] for r in scored]),
                    "mean_angular_error_degrees": average(
                        [r["metrics"]["angular_error_degrees"] for r in scored]
                    ),
                    "positive_sessions": sum(r["metrics"]["r2"] > 0 for r in scored),
                    "mean_unique_requested_label_seconds": average(
                        [r["unique_requested_label_seconds"] for r in requested]
                    ),
                    "fraction_valid_time_in_windows_worse_than_zero": bad / total
                    if total
                    else None,
                }
            )
        scored_people = [p for p in participant_rows if p["mean_r2"] is not None]
        values = [p["mean_r2"] for p in scored_people]
        gains = [p["gain_over_frozen120"] for p in scored_people]
        summaries[method] = {
            "participants": participant_rows,
            "mean_r2": average(values),
            "participant_bootstrap_95": interval(values),
            "gain_over_frozen120": average(gains),
            "gain_over_zero": average([p["gain_over_zero"] for p in scored_people]),
            "paired_participant_gain_bootstrap_95": interval(gains),
            "mean_unique_requested_label_seconds": average(
                [p["mean_unique_requested_label_seconds"] for p in participant_rows]
            ),
            "scored_sessions": sum(p["scored_sessions"] for p in participant_rows),
        }
        requested = [lookup[s, method] for s in primary_ids]
        scored = [r for r in requested if r["metrics"]["r2"] is not None]
        values = [r["metrics"]["r2"] for r in scored]
        gains = [
            r["metrics"]["r2"] - lookup[r["session_id"], "frozen120"]["metrics"]["r2"]
            for r in scored
        ]
        ci = interval(gains)
        conditions = {
            "all_six_scorable": len(primary_ids) == len(scored) == 6,
            "mean_r2_at_least_005": bool(values and average(values) >= 0.05),
            "gain_at_least_005": bool(gains and average(gains) >= 0.05),
            "four_positive_sessions": sum(v > 0 for v in values) >= 4,
            "paired_interval_lower_positive": ci[0] is not None and ci[0] > 0,
        }
        primary[method] = {
            "selected_sessions": len(requested),
            "scored_sessions": len(scored),
            "mean_r2": average(values),
            "session_bootstrap_95": interval(values),
            "positive_sessions": sum(v > 0 for v in values),
            "mean_gain_over_frozen120": average(gains),
            "mean_gain_over_zero": average(
                [
                    r["metrics"]["r2"] - lookup[r["session_id"], "zero"]["metrics"]["r2"]
                    for r in scored
                ]
            ),
            "paired_gain_bootstrap_95": ci,
            "conditions": conditions,
            "eligible": method in ELIGIBLE,
            "gate_passed": method in ELIGIBLE and all(conditions.values()),
            "sessions": [
                {
                    "session_id": r["session_id"],
                    "post_implant_day": r["post_implant_day"],
                    "r2": r["metrics"]["r2"],
                }
                for r in requested
            ],
        }
    passing = [m for m in ELIGIBLE if primary[m]["gate_passed"]]
    chosen = max(passing, key=lambda m: primary[m]["mean_r2"]) if passing else None
    return {
        "selected_sessions": len(sessions),
        "methods": summaries,
        "primary_T2_session_ids": primary_ids,
        "primary_T2": primary,
        "gate": {
            "passed": bool(passing),
            "passing_methods": passing,
            "selected_method": chosen,
            "next_action": "freeze_continual_policy_then_fresh_assessment"
            if passing
            else "stop_no_fresh_acquisition",
            "evidence_role": (
                "exposed development selection; not fresh validation or "
                "multiplicity-adjusted inference"
            ),
        },
    }


def check_session(item, labels, raw, rows, windows, root):
    """Bind all saved interval exports to their declared fit and pooled forecast."""
    n = item["n_bins"]
    clean = burned_mask(labels["eval_mask"], item["segments"])
    clean[:13600] = False
    pooled = clean.copy()
    for start in UPDATES:
        stop = min(n, start + 3000)
        if start < stop and clean[start:stop].sum() < 50:
            pooled[start:stop] = False
    checked, checked_windows, errors, residuals = [], [], [], []
    for method in METHODS:
        identity = {
            "participant": item["participant_id"],
            "session_id": item["session_id"],
            "post_implant_day": item["post_implant_day"],
            "method": method,
        }
        prediction = np.zeros((n, 2))
        purchases = []
        for start in UPDATES:
            w = windows[item["session_id"], method, start]
            if start >= n:
                expected = {
                    **identity,
                    "start_bin": start,
                    "stop_bin": start,
                    "disposition": "recording_ended_before_forecast_window",
                    "metrics": None,
                    "valid_rows": 0,
                    "score_mask_sha256": mask_hash(np.zeros(0, bool)),
                    "fit": None,
                }
                compare_structure(w, expected, "unavailable window")
                checked_windows.append(expected)
                continue
            stop = min(n, start + 3000)
            trace_path = contained(root, w["trace_path"])
            verify(trace_path, w["trace_sha256"])
            trace = numeric(trace_path)
            if set(trace) != {"prediction", "eval_mask", "start_bin", "stop_bin"}:
                raise ValueError("Unexpected trace fields")
            for key, value in (("start_bin", start), ("stop_bin", stop)):
                if (
                    trace[key].shape != ()
                    or trace[key].dtype != np.int64
                    or trace[key].item() != value
                ):
                    raise ValueError("Trace time identity mismatch")
            saved = trace["prediction"]
            mask = clean[start:stop]
            if (
                saved.dtype != np.float64
                or saved.shape != (stop - start, 2)
                or not np.isfinite(saved).all()
                or trace["eval_mask"].dtype != bool
                or not np.array_equal(trace["eval_mask"], mask)
            ):
                raise ValueError("Trace numeric shape/mask mismatch")
            if method == "zero":
                if (
                    w["model_path"] is not None
                    or w["model_sha256"] is not None
                    or w["fit"] is not None
                ):
                    raise ValueError("Zero control unexpectedly fitted a model")
                expected_prediction = np.zeros_like(saved)
            else:
                path = contained(root, w["model_path"])
                verify(path, w["model_sha256"])
                arrays = numeric(path)
                residuals.append(
                    validate_fit(
                        arrays,
                        w["fit"],
                        raw,
                        labels,
                        item["segments"],
                        item["n_channels"],
                        method,
                        start,
                    )
                )
                a, b, _ = fit_geometry(method, start)
                purchases.append((a, b))
                expected_prediction = replay(arrays, raw[start:stop])
            error = float(np.max(np.abs(expected_prediction - saved)))
            if not np.allclose(saved, expected_prediction, rtol=1e-9, atol=1e-9):
                raise ValueError("Full independent window replay mismatch")
            errors.append(error)
            prediction[start:stop] = saved
            values = score_reference(labels["y"][start:stop], saved, mask)
            expected = {
                **identity,
                "start_bin": start,
                "stop_bin": stop,
                "disposition": "scored" if int(mask.sum()) >= 50 else "insufficient_labels",
                "fit": w["fit"],
                "metrics": values,
                "score_mask_sha256": mask_hash(mask),
                "valid_rows": int(mask.sum()),
                **{k: w[k] for k in ("model_path", "model_sha256", "trace_path", "trace_sha256")},
            }
            compare_structure(w, expected, "window identity and score")
            checked_windows.append(expected)
        expected = {
            **identity,
            "n_bins": n,
            "metrics": score_reference(labels["y"], prediction, pooled),
            "score_mask_sha256": mask_hash(pooled),
            "unique_requested_label_seconds": unique_label_seconds(purchases),
        }
        compare_structure(rows[item["session_id"], method], expected, "pooled session score")
        checked.append(expected)
    return checked, checked_windows, errors, residuals


def audit_run(path, root=ROOT):
    root, path = Path(root).resolve(), Path(path).resolve()
    manifest = read(path)
    if (
        manifest["schema_version"] != 1
        or manifest["phase"] != "exposed_recoverability"
        or manifest["status"] != "complete"
    ):
        raise ValueError("Incomplete or wrong experiment")
    if manifest["plan"]["sha256"] != PLAN_SHA:
        raise ValueError("Wrong locked plan")
    verify(contained(root, manifest["plan"]["path"]), PLAN_SHA)
    source_paths = {
        f"neural_experiments/{v}/{name}.py"
        for v, names in (
            ("v7", ("model", "recoverability", "data", "metrics")),
            ("v6", ("data",)),
            ("v5", ("data",)),
            ("v4", ("model", "data")),
            ("v3", ("study", "model", "data")),
            ("v2", ("data",)),
        )
        for name in names
    }
    if set(manifest["code_sha256"]) != source_paths:
        raise ValueError("Executed source inventory differs")
    for relative, digest in manifest["code_sha256"].items():
        verify(contained(root, relative), digest)
        verify(contained(path.parent / "source_snapshot", relative), digest)
    index_path = contained(root, manifest["index"]["path"])
    verify(index_path, manifest["index"]["sha256"])
    inputs = Inputs(index_path)
    rows = unique(manifest["rows"], lambda r: (r["session_id"], r["method"]))
    windows = unique(
        manifest["window_rows"], lambda r: (r["session_id"], r["method"], r["start_bin"])
    )
    if set(rows) != {(s, m) for s in inputs.items for m in METHODS} or set(windows) != {
        (s, m, t) for s in inputs.items for m in METHODS for t in UPDATES
    }:
        raise ValueError("Missing/additional session/method/window cases")
    all_rows, all_windows, all_errors, all_residuals = [], [], [], []
    for item in inputs.items.values():
        features, labels = inputs.arrays(item)
        raw = reference_representation(features, item["segments"], CONFIG)
        rr, ww, errors, residuals = check_session(item, labels, raw, rows, windows, root)
        all_rows.extend(rr)
        all_windows.extend(ww)
        all_errors.extend(errors)
        all_residuals.extend(residuals)
        print(f"independently checked {item['session_id']}", flush=True)
    summary = summarize_reference(all_rows, all_windows)
    compare_structure(manifest["summary"], summary, "locked gate and summaries")
    return {
        "status": "PASS",
        "stage": "exposed_recoverability",
        "created_at": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": sha256(__file__),
        "manifest_sha256": sha256(path),
        "plan_sha256": PLAN_SHA,
        "index_sha256": sha256(index_path),
        "code_sha256": manifest["code_sha256"],
        "sessions_checked": len(inputs.items),
        "session_method_cases_checked": len(all_rows),
        "method_window_cases_checked": len(all_windows),
        "models_checked": len(all_residuals),
        "full_window_traces_checked": len(all_errors),
        "unavailable_windows_checked": sum(w["metrics"] is None for w in all_windows),
        "sparse_windows_checked": sum(
            w["disposition"] == "insufficient_labels" for w in all_windows
        ),
        "max_prediction_absolute_error": max(all_errors, default=0.0),
        "max_normal_equation_relative_residual": max(all_residuals, default=0.0),
        "summary": summary,
        "scope": (
            "Independent NumPy recurrence, weighted ridge optimality, prefix/window "
            "normalization, full saved window replay, sklearn score arithmetic, label-union "
            "accounting, case inventory, source/index binding and descriptive session/participant "
            "bootstrap. No V7 producer/model/scorer/importer imports; not external replication "
            "or fresh inference."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory or manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit receipts are immutable; choose a fresh output path")
    path = args.run / "manifest.json" if args.run.is_dir() else args.run
    receipt = audit_run(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("summary", "code_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
