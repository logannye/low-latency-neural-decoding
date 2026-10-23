"""Matched accuracy, accepted failures, label costs and later-label opportunities."""

from __future__ import annotations

import numpy as np

METHODS = ("zero", "history0", "daily120", "daily240", "daily360", "v5_adaptive", "confirmed")
PEOPLE = ("T2", "T5", "T6", "T7", "T8", "T9", "T10", "T11")


def interval(values):
    if not values:
        return [None, None]
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261025)
    indices = rng.integers(len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist()


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(rows):
    lookup = {(r["session_id"], r["method"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate session-method result")
    sessions = sorted({r["session_id"] for r in rows})
    for sid in sessions:
        rr = [lookup[sid, m] for m in METHODS]
        if len({r["eval_mask_sha256"] for r in rr}) != 1:
            raise ValueError("All methods must use identical evaluation masks")
        if len({r["metrics"]["r2"] is None for r in rr}) != 1:
            raise ValueError("All methods must retain the same unavailable scores")
    methods = {}
    for method in METHODS:
        requested = [r for r in rows if r["method"] == method]
        scored = [r for r in requested if r["metrics"]["r2"] is not None]
        units, costs = [], []
        for person in sorted({r["participant"] for r in requested}):
            rr = [r for r in requested if r["participant"] == person]
            valid = [r for r in scored if r["participant"] == person]
            costs.append(
                {
                    "participant": person,
                    "requested_label_seconds": mean([r["label_seconds"] for r in rr]),
                    "available_label_seconds": mean([r["available_label_seconds"] for r in rr]),
                }
            )
            if valid:
                units.append(
                    {
                        "participant": person,
                        "n_sessions": len(valid),
                        "r2": mean([r["metrics"]["r2"] for r in valid]),
                        "angular_error_degrees": mean(
                            [r["metrics"]["angular_error_degrees"] for r in valid]
                        ),
                    }
                )
        accepted = [r for r in scored if r["action"] == "accept"]
        nonpositive = [r["session_id"] for r in accepted if r["metrics"]["r2"] <= 0]
        harmful = [
            r["session_id"]
            for r in accepted
            if r["metrics"]["summed_mse"]
            > lookup[r["session_id"], "zero"]["metrics"]["summed_mse"] + 1e-12
        ]
        values = [r["r2"] for r in units]
        risk_units = []
        for person in sorted({r["participant"] for r in scored}):
            rr = [r for r in scored if r["participant"] == person]
            ids = {r["session_id"] for r in rr}
            risk_units.append(
                {
                    "participant": person,
                    "nonpositive_burden": len(ids.intersection(nonpositive)) / len(rr),
                    "harmful_burden": len(ids.intersection(harmful)) / len(rr),
                    "coverage": sum(r["action"] == "accept" for r in rr) / len(rr),
                }
            )
        methods[method] = {
            "units": units,
            "cost_units": costs,
            "requested_sessions": len(requested),
            "scored_sessions": len(scored),
            "unavailable_sessions": len(requested) - len(scored),
            "n_participants": len(units),
            "mean_r2": mean(values),
            "r2_bootstrap_95": interval(values),
            "positive_participants": sum(v > 0 for v in values),
            "mean_angular_error_degrees": mean([u["angular_error_degrees"] for u in units]),
            "mean_requested_label_seconds": mean([c["requested_label_seconds"] for c in costs]),
            "mean_available_label_seconds": mean([c["available_label_seconds"] for c in costs]),
            "accepted_scored_sessions": len(accepted),
            "acceptance_coverage": len(accepted) / len(scored) if scored else None,
            "abstained_selected_sessions": sum(r["action"] == "abstain" for r in requested),
            "nonpositive_accepted_sessions": nonpositive,
            "harmful_vs_zero_accepted_sessions": harmful,
            "nonpositive_acceptance_fraction": len(nonpositive) / len(accepted)
            if accepted
            else None,
            "harmful_acceptance_fraction": len(harmful) / len(accepted) if accepted else None,
            "risk_units": risk_units,
            "mean_nonpositive_burden": mean([r["nonpositive_burden"] for r in risk_units]),
            "mean_harmful_burden": mean([r["harmful_burden"] for r in risk_units]),
        }
    contrasts = []
    for control in METHODS[:-1]:
        gains = []
        for sid in sessions:
            a, b = lookup[sid, "confirmed"], lookup[sid, control]
            if a["metrics"]["r2"] is not None:
                gains.append(
                    {
                        "participant": a["participant"],
                        "session_id": sid,
                        "gain": a["metrics"]["r2"] - b["metrics"]["r2"],
                    }
                )
        values = [
            mean([r["gain"] for r in gains if r["participant"] == p])
            for p in sorted({r["participant"] for r in gains})
        ]
        contrasts.append(
            {
                "control": control,
                "n_participants": len(values),
                "mean_gain": mean(values),
                "gain_bootstrap_95": interval(values),
            }
        )
    opportunities = []
    for sid in sessions:
        a = lookup[sid, "confirmed"]
        if a["metrics"]["r2"] is None:
            continue
        base = lookup[sid, "daily120"]["metrics"]["r2"]
        zero = lookup[sid, "zero"]["metrics"]["r2"]
        late = max(lookup[sid, m]["metrics"]["r2"] for m in ("daily240", "daily360"))
        if late - base >= 0.05 and late - zero >= 0.05:
            gain = a["metrics"]["r2"] - base
            escalated = a.get("training_label_seconds", 0) >= 240
            opportunities.append(
                {
                    "participant": a["participant"],
                    "session_id": sid,
                    "later_fixed_gain_over_daily120": late - base,
                    "confirmed_gain_over_daily120": gain,
                    "escalated_to_at_least240": escalated,
                    "captured": a["action"] == "accept" and escalated and gain >= 0.025,
                }
            )
    t2rows = [
        lookup[sid, "confirmed"]
        for sid in sessions
        if lookup[sid, "confirmed"]["participant"] == "T2"
    ]
    t2scored = [r for r in t2rows if r["metrics"]["r2"] is not None]
    t2mean = mean([r["metrics"]["r2"] for r in t2scored])
    t2positive = sum(r["action"] == "accept" and r["metrics"]["r2"] > 0 for r in t2scored)
    t2harmful = [
        r["session_id"]
        for r in t2scored
        if r["action"] == "accept"
        and r["metrics"]["summed_mse"]
        > lookup[r["session_id"], "zero"]["metrics"]["summed_mse"] + 1e-12
    ]
    t2gains = [
        r["metrics"]["r2"] - lookup[r["session_id"], "v5_adaptive"]["metrics"]["r2"]
        for r in t2scored
    ]
    c, old = methods["confirmed"], methods["v5_adaptive"]
    risk_contrasts = {}
    for field in ("nonpositive_burden", "harmful_burden", "coverage"):
        old_units = {r["participant"]: r for r in old["risk_units"]}
        deltas = [r[field] - old_units[r["participant"]][field] for r in c["risk_units"]]
        risk_contrasts[field] = {
            "mean_difference": mean(deltas),
            "participant_bootstrap_95": interval(deltas),
        }
    useful = bool(
        c["n_participants"] == 8
        and c["mean_r2"] >= 0.1
        and c["r2_bootstrap_95"][0] > 0
        and c["positive_participants"] >= 6
    )
    captured = sum(r["captured"] for r in opportunities)
    return {
        "methods": methods,
        "contrasts": contrasts,
        "risk_contrasts_vs_v5": risk_contrasts,
        "T2": {
            "requested_sessions": len(t2rows),
            "scored_sessions": len(t2scored),
            "mean_r2": t2mean,
            "r2_session_bootstrap_95": interval([r["metrics"]["r2"] for r in t2scored]),
            "positive_accepted_sessions": t2positive,
            "harmful_accepted_sessions": t2harmful,
            "gain_over_v5": mean(t2gains),
            "gain_session_bootstrap_95": interval(t2gains),
        },
        "later_label_opportunities": {
            "definition": (
                "Prespecified descriptive counterfactual: max(daily240,daily360) exceeds "
                "daily120 and zero by at least0.05 R2; no scorer outcome enters policy."
            ),
            "cases": opportunities,
            "n": len(opportunities),
            "captured": captured,
            "capture_fraction": captured / len(opportunities) if opportunities else None,
        },
        "gates": {
            "useful_cohort_accuracy": useful,
            "fewer_false_acceptances_with_useful_coverage": bool(
                useful
                and c["acceptance_coverage"] >= 0.5
                and len(c["nonpositive_accepted_sessions"])
                < len(old["nonpositive_accepted_sessions"])
                and len(c["harmful_vs_zero_accepted_sessions"])
                <= len(old["harmful_vs_zero_accepted_sessions"])
            ),
            "T2_recovered_without_harmful_acceptance": bool(
                len(t2rows) == len(t2scored) == 3
                and t2mean >= 0.05
                and t2positive >= 2
                and not t2harmful
                and mean(t2gains) >= 0.03
            ),
            "captured_later_label_benefit": (
                captured / len(opportunities) >= 0.5 if opportunities else None
            ),
        },
        "limitations": (
            "Participant bootstrap for cohort; T2 session intervals are descriptive small-N. "
            "Observed acceptance fractions are not population safety bounds. Opportunity "
            "analysis is a prespecified post-assessment comparison, not an oracle available "
            "to the controller."
        ),
    }
