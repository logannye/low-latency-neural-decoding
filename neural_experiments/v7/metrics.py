"""Paired development summaries and the locked conditional recovery gate."""

from __future__ import annotations

import numpy as np

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


def mean(values):
    return float(np.mean(values)) if values else None


def interval(values):
    if not values:
        return [None, None]
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite bootstrap values required")
    rng = np.random.default_rng(20261026)
    indices = rng.integers(len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist()


def _metric(row, key="r2"):
    value = row.get("metrics")
    return None if value is None else value[key]


def summarize(rows, window_rows):
    lookup = {(r["session_id"], r["method"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate session-method")
    sessions = sorted({r["session_id"] for r in rows})
    if not sessions:
        raise ValueError("empty study")
    for sid in sessions:
        if {r["method"] for r in rows if r["session_id"] == sid} != set(METHODS):
            raise ValueError("complete declared method grid required")
        matched = [lookup[sid, m] for m in METHODS]
        if (
            len({r["score_mask_sha256"] for r in matched}) != 1
            or len({_metric(r) is None for r in matched}) != 1
            or len({(r["participant"], r["post_implant_day"]) for r in matched}) != 1
        ):
            raise ValueError("matched session geometry and scoring masks required")
    windows = {(r["session_id"], r["method"], r["start_bin"]): r for r in window_rows}
    if len(windows) != len(window_rows):
        raise ValueError("duplicate method-window")
    for sid, _, start in windows:
        matched = [windows.get((sid, m, start)) for m in METHODS]
        if any(r is None for r in matched):
            raise ValueError("incomplete window method grid")
        if len({r["score_mask_sha256"] for r in matched}) != 1:
            raise ValueError("window score masks differ")
    t2 = sorted(
        [lookup[sid, "zero"] for sid in sessions if lookup[sid, "zero"]["participant"] == "T2"],
        key=lambda r: (r["post_implant_day"], r["session_id"]),
    )
    primary_ids = [r["session_id"] for r in t2[-6:]]
    methods = {}
    for method in METHODS:
        participant_units = []
        rr = [lookup[sid, method] for sid in sessions]
        for person in sorted({r["participant"] for r in rr}, key=lambda p: int(p[1:])):
            pp = [r for r in rr if r["participant"] == person]
            scored = [r for r in pp if _metric(r) is not None]
            gains = [_metric(r) - _metric(lookup[r["session_id"], "frozen120"]) for r in scored]
            zero_gains = [_metric(r) - _metric(lookup[r["session_id"], "zero"]) for r in scored]
            bad_rows = total_rows = 0
            for w in window_rows:
                if (
                    w["participant"] != person
                    or w["method"] != method
                    or _metric(w, "summed_mse") is None
                ):
                    continue
                control = windows[w["session_id"], "zero", w["start_bin"]]
                total_rows += w["valid_rows"]
                if _metric(w, "summed_mse") > _metric(control, "summed_mse") + 1e-12:
                    bad_rows += w["valid_rows"]
            participant_units.append(
                {
                    "participant": person,
                    "selected_sessions": len(pp),
                    "scored_sessions": len(scored),
                    "mean_r2": mean([_metric(r) for r in scored]),
                    "gain_over_frozen120": mean(gains),
                    "gain_over_zero": mean(zero_gains),
                    "mean_summed_mse": mean([_metric(r, "summed_mse") for r in scored]),
                    "mean_angular_error_degrees": mean(
                        [_metric(r, "angular_error_degrees") for r in scored]
                    ),
                    "positive_sessions": sum(_metric(r) > 0 for r in scored),
                    "mean_unique_requested_label_seconds": mean(
                        [r["unique_requested_label_seconds"] for r in pp]
                    ),
                    "fraction_valid_time_in_windows_worse_than_zero": bad_rows / total_rows
                    if total_rows
                    else None,
                }
            )
        units = [p for p in participant_units if p["mean_r2"] is not None]
        methods[method] = {
            "participants": participant_units,
            "mean_r2": mean([p["mean_r2"] for p in units]),
            "participant_bootstrap_95": interval([p["mean_r2"] for p in units]),
            "gain_over_frozen120": mean([p["gain_over_frozen120"] for p in units]),
            "gain_over_zero": mean([p["gain_over_zero"] for p in units]),
            "paired_participant_gain_bootstrap_95": interval(
                [p["gain_over_frozen120"] for p in units]
            ),
            "mean_unique_requested_label_seconds": mean(
                [p["mean_unique_requested_label_seconds"] for p in participant_units]
            ),
            "scored_sessions": sum(p["scored_sessions"] for p in participant_units),
        }
    primary = {}
    for method in METHODS:
        pp = [lookup[sid, method] for sid in primary_ids]
        scored = [r for r in pp if _metric(r) is not None]
        values = [_metric(r) for r in scored]
        gains = [_metric(r) - _metric(lookup[r["session_id"], "frozen120"]) for r in scored]
        zero_gains = [_metric(r) - _metric(lookup[r["session_id"], "zero"]) for r in scored]
        ci = interval(gains)
        conditions = {
            "all_six_scorable": len(primary_ids) == len(scored) == 6,
            "mean_r2_at_least_005": bool(values and mean(values) >= 0.05),
            "gain_at_least_005": bool(gains and mean(gains) >= 0.05),
            "four_positive_sessions": sum(v > 0 for v in values) >= 4,
            "paired_interval_lower_positive": ci[0] is not None and ci[0] > 0,
        }
        primary[method] = {
            "selected_sessions": len(pp),
            "scored_sessions": len(scored),
            "mean_r2": mean(values),
            "session_bootstrap_95": interval(values),
            "positive_sessions": sum(v > 0 for v in values),
            "mean_gain_over_frozen120": mean(gains),
            "mean_gain_over_zero": mean(zero_gains),
            "paired_gain_bootstrap_95": ci,
            "conditions": conditions,
            "eligible": method in ELIGIBLE,
            "gate_passed": method in ELIGIBLE and all(conditions.values()),
            "sessions": [
                {
                    "session_id": r["session_id"],
                    "post_implant_day": r["post_implant_day"],
                    "r2": _metric(r),
                }
                for r in pp
            ],
        }
    passing = [m for m in ELIGIBLE if primary[m]["gate_passed"]]
    chosen = max(passing, key=lambda m: primary[m]["mean_r2"]) if passing else None
    return {
        "selected_sessions": len(sessions),
        "methods": methods,
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
