"""Matched full-interval outcomes and exposed threshold selection for V7B."""

from __future__ import annotations

import numpy as np

METHODS = ("zero", "frozen120", "fixed240", "periodic240", "triggered240")
THRESHOLDS = (0.1, 0.25, 0.5, 1.0)


def mean(values):
    return float(np.mean(values)) if values else None


def interval(values):
    if not values:
        return [None, None]
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite bootstrap inputs required")
    rng = np.random.default_rng(20261027)
    indexes = rng.integers(len(values), size=(10000, len(values)))
    return np.quantile(values[indexes].mean(1), [0.025, 0.975]).tolist()


def select_source(rows):
    keys = {(r["session_id"], r["method"], r["threshold"]) for r in rows}
    sessions = {r["session_id"] for r in rows}
    if len(keys) != len(rows) or keys != {
        (sid, method, threshold)
        for sid in sessions
        for method in METHODS
        for threshold in THRESHOLDS
    }:
        raise ValueError("complete unique source threshold-method grid required")
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    choices = []
    for excluded in people:
        candidates = []
        for threshold in THRESHOLDS:
            units = []
            for person in people:
                if person == excluded:
                    continue
                values = [
                    r["metrics"]["r2"]
                    for r in rows
                    if r["method"] == "triggered240"
                    and r["threshold"] == threshold
                    and r["participant"] == person
                    and r["metrics"]["r2"] is not None
                ]
                if values:
                    units.append({"participant": person, "mean_r2": mean(values), "n": len(values)})
            candidates.append(
                {
                    "threshold": threshold,
                    "source_units": units,
                    "mean_r2": mean([u["mean_r2"] for u in units]),
                }
            )
        if any(c["mean_r2"] is None for c in candidates):
            raise ValueError("every threshold requires scored source participants")
        selected = max(candidates, key=lambda c: c["mean_r2"])
        choices.append(
            {
                "participant": excluded,
                "threshold": selected["threshold"],
                "source_mean_r2": selected["mean_r2"],
                "candidates": candidates,
            }
        )
    return {"choices": choices, "evidence_role": "exposed whole-person-excluded threshold ranking"}


def summarize(rows):
    lookup = {(r["session_id"], r["method"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate method-session")
    ids = sorted({r["session_id"] for r in rows})
    for sid in ids:
        rr = [lookup.get((sid, m)) for m in METHODS]
        if any(r is None for r in rr):
            raise ValueError("incomplete method grid")
        if (
            len({r["score_mask_sha256"] for r in rr}) != 1
            or len({r["metrics"]["r2"] is None for r in rr}) != 1
        ):
            raise ValueError("common full-interval mask and score availability required")
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    methods = {}
    for method in METHODS:
        units = []
        for person in people:
            rr = [r for r in rows if r["participant"] == person and r["method"] == method]
            valid = [r for r in rr if r["metrics"]["r2"] is not None]
            harmful, total = 0, 0
            for row in rr:
                zero = lookup[row["session_id"], "zero"]
                for w, z in zip(row["diagnostic_windows"], zero["diagnostic_windows"], strict=True):
                    if w["metrics"] is None or w["metrics"]["summed_mse"] is None:
                        continue
                    total += w["valid_rows"]
                    if w["metrics"]["summed_mse"] > z["metrics"]["summed_mse"] + 1e-12:
                        harmful += w["valid_rows"]
            units.append(
                {
                    "participant": person,
                    "selected_sessions": len(rr),
                    "scored_sessions": len(valid),
                    "mean_r2": mean([r["metrics"]["r2"] for r in valid]),
                    "positive_sessions": sum(r["metrics"]["r2"] > 0 for r in valid),
                    "mean_summed_mse": mean([r["metrics"]["summed_mse"] for r in valid]),
                    "mean_angular_error_degrees": mean(
                        [r["metrics"]["angular_error_degrees"] for r in valid]
                    ),
                    "mean_planned_label_seconds": mean([r["planned_label_seconds"] for r in rr]),
                    "mean_available_label_seconds": mean(
                        [r["available_label_seconds"] for r in rr]
                    ),
                    "mean_interruption_seconds": mean([r["interruption_seconds"] for r in rr]),
                    "harmful_window_valid_time_fraction": harmful / total if total else None,
                    "harmful_sessions": [
                        r["session_id"]
                        for r in valid
                        if r["metrics"]["summed_mse"]
                        > lookup[r["session_id"], "zero"]["metrics"]["summed_mse"] + 1e-12
                    ],
                }
            )
        scored = [u for u in units if u["mean_r2"] is not None]
        methods[method] = {
            "participants": units,
            "mean_r2": mean([u["mean_r2"] for u in scored]),
            "participant_bootstrap_95": interval([u["mean_r2"] for u in scored]),
            "positive_participants": sum(u["mean_r2"] > 0 for u in scored),
            "n_participants": len(scored),
            "scored_sessions": sum(u["scored_sessions"] for u in units),
            "mean_available_label_seconds": mean(
                [u["mean_available_label_seconds"] for u in units]
            ),
            "mean_interruption_seconds": mean([u["mean_interruption_seconds"] for u in units]),
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
            gains = []
            for sid in ids:
                row = lookup[sid, a]
                if row["participant"] == person and row["metrics"]["r2"] is not None:
                    gains.append(row["metrics"]["r2"] - lookup[sid, b]["metrics"]["r2"])
            if gains:
                units.append({"participant": person, "mean_gain": mean(gains)})
            if person == "T2":
                t2 = gains
        contrasts[f"{a}_minus_{b}"] = {
            "participants": units,
            "mean_gain": mean([u["mean_gain"] for u in units]),
            "paired_participant_95": interval([u["mean_gain"] for u in units]),
            "T2_mean_gain": mean(t2),
            "T2_paired_session_95": interval(t2),
        }
    t2 = {}
    for method in METHODS:
        rr = [r for r in rows if r["participant"] == "T2" and r["method"] == method]
        values = [r["metrics"]["r2"] for r in rr if r["metrics"]["r2"] is not None]
        t2[method] = {
            "selected_sessions": len(rr),
            "scored_sessions": len(values),
            "mean_r2": mean(values),
            "session_bootstrap_95": interval(values),
            "positive_sessions": sum(v > 0 for v in values),
        }
    t = t2["periodic240"]
    gain = contrasts["periodic240_minus_frozen120"]
    timing = contrasts["triggered240_minus_periodic240"]
    full = [sid for sid in ids if lookup[sid, "periodic240"]["n_bins"] == 30000]
    matched = bool(full) and all(
        abs(
            lookup[sid, "periodic240"]["available_label_seconds"]
            - lookup[sid, "triggered240"]["available_label_seconds"]
        )
        < 1e-12
        for sid in full
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
            matched
            and timing["mean_gain"] is not None
            and timing["mean_gain"] >= 0.01
            and timing["paired_participant_95"][0] > 0
            and t2["triggered240"]["mean_r2"] is not None
            and t["mean_r2"] is not None
            and t2["triggered240"]["mean_r2"] >= t["mean_r2"]
        ),
    }
    return {
        "selected_sessions": len(ids),
        "methods": methods,
        "T2": t2,
        "contrasts": contrasts,
        "complete_sessions_matched_feedback_budget": matched,
        "gates": gates,
    }
