"""Equal-person accuracy, calibration cost, coverage and paired engineering gates."""

from __future__ import annotations

import numpy as np

METHODS = ("zero", "history0", "history30", "daily120", "daily240", "adaptive")


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(20261024)
    samples = rng.integers(0, len(values), (10000, len(values)))
    return np.quantile(values[samples].mean(1), [0.025, 0.975]).tolist()


def _average(rows, field):
    return float(np.mean([r[field] for r in rows])) if rows else None


def summarize(rows):
    domains = {}
    for domain in ("new_sessions", "t3_blocks"):
        allrows = [r for r in rows if r["domain"] == domain]
        unit_key = "participant" if domain == "new_sessions" else "session_id"
        methods = {}
        for method in METHODS:
            requested = [r for r in allrows if r["method"] == method]
            eligible = [r for r in requested if r["metrics"]["r2"] is not None]
            units = []
            for unit in sorted({r[unit_key] for r in eligible}):
                rr = [r for r in eligible if r[unit_key] == unit]
                units.append(
                    {
                        "unit": unit,
                        "n_sessions": len(rr),
                        "r2": float(np.mean([r["metrics"]["r2"] for r in rr])),
                        "angular_error_degrees": float(
                            np.mean([r["metrics"]["angular_error_degrees"] for r in rr])
                        ),
                    }
                )
            cost_units = []
            for unit in sorted({r[unit_key] for r in requested}):
                rr = [r for r in requested if r[unit_key] == unit]
                cost_units.append(
                    {
                        "unit": unit,
                        "label_seconds": _average(rr, "label_seconds"),
                        "available_label_seconds": _average(rr, "available_label_seconds"),
                    }
                )
            values = [u["r2"] for u in units]
            accepted = [r for r in eligible if r["action"] != "abstain"]
            methods[method] = {
                "units": units,
                "n_units": len(units),
                "requested_sessions": len(requested),
                "scored_sessions": len(eligible),
                "unavailable_sessions": len(requested) - len(eligible),
                "mean_r2": float(np.mean(values)) if values else None,
                "r2_bootstrap_95": interval(values),
                "positive_units": sum(v > 0 for v in values),
                "mean_angular_error_degrees": _average(units, "angular_error_degrees"),
                "cost_units": cost_units,
                "mean_requested_label_seconds": _average(cost_units, "label_seconds"),
                "mean_available_label_seconds": _average(cost_units, "available_label_seconds"),
                "zero_label_sessions": sum(r["label_seconds"] == 0 for r in requested),
                "abstained_sessions": sum(r["action"] == "abstain" for r in requested),
                "accepted_scored_sessions": len(accepted),
                "acceptance_coverage": len(accepted) / len(eligible) if eligible else None,
                "negative_accepted_sessions": sum(r["metrics"]["r2"] < 0 for r in accepted),
            }
        contrasts = []
        for control in METHODS[:-1]:
            left = {
                r["session_id"]: r
                for r in allrows
                if r["method"] == "adaptive" and r["metrics"]["r2"] is not None
            }
            right = {
                r["session_id"]: r
                for r in allrows
                if r["method"] == control and r["metrics"]["r2"] is not None
            }
            pairs = []
            for sid in sorted(left.keys() & right.keys()):
                a, b = left[sid], right[sid]
                if a["eval_mask_sha256"] != b["eval_mask_sha256"]:
                    raise ValueError("Paired evaluation masks differ")
                pairs.append(
                    {
                        "unit": a[unit_key],
                        "participant": a["participant"],
                        "session_id": sid,
                        "gain": a["metrics"]["r2"] - b["metrics"]["r2"],
                    }
                )
            values = [
                float(np.mean([r["gain"] for r in pairs if r["unit"] == u]))
                for u in sorted({r["unit"] for r in pairs})
            ]
            contrasts.append(
                {
                    "arm": "adaptive",
                    "control": control,
                    "n_units": len(values),
                    "mean_gain": float(np.mean(values)) if values else None,
                    "gain_bootstrap_95": interval(values),
                }
            )
        domains[domain] = {"methods": methods, "contrasts": contrasts}
    new = domains["new_sessions"]
    adaptive = new["methods"]["adaptive"]
    contrasts = {r["control"]: r for r in new["contrasts"]}
    useful = bool(
        adaptive["n_units"] == 8
        and adaptive["mean_r2"] >= 0.1
        and adaptive["r2_bootstrap_95"][0] > 0
        and adaptive["positive_units"] >= 6
        and adaptive["mean_angular_error_degrees"] <= 80
        and contrasts["zero"]["mean_gain"] >= 0.05
    )
    t2rows = [
        r
        for r in rows
        if r["domain"] == "new_sessions"
        and r["participant"] == "T2"
        and r["metrics"]["r2"] is not None
    ]
    t2 = {m: [r for r in t2rows if r["method"] == m] for m in METHODS}
    gains = []
    for r in t2["adaptive"]:
        prior = next(c for c in t2["history0"] if c["session_id"] == r["session_id"])
        gains.append(r["metrics"]["r2"] - prior["metrics"]["r2"])
    t2mean = (
        float(np.mean([r["metrics"]["r2"] for r in t2["adaptive"]])) if t2["adaptive"] else None
    )
    zeromean = float(np.mean([r["metrics"]["r2"] for r in t2["zero"]])) if t2["zero"] else None
    t2positive = sum(r["metrics"]["r2"] > 0 and r["action"] != "abstain" for r in t2["adaptive"])
    t2ci = interval(gains)
    return {
        "domains": domains,
        "T2": {
            "mean_r2": t2mean,
            "positive_accepted_sessions": t2positive,
            "gain_over_history0": float(np.mean(gains)) if gains else None,
            "gain_bootstrap_95": t2ci,
        },
        "gates": {
            "adaptive_useful": useful,
            "reduced_calibration_noninferiority": bool(
                useful
                and contrasts["daily120"]["gain_bootstrap_95"][0] > -0.05
                and adaptive["mean_requested_label_seconds"] <= 90
            ),
            "T2_recovered": bool(
                len(t2["adaptive"]) == 3
                and t2mean >= 0.05
                and t2positive >= 2
                and t2mean - zeromean >= 0.05
                and t2ci[0] > 0
            ),
        },
    }
