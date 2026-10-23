"""Fixed V9 comparisons with session-level pairing and explicit recovery gates."""

from __future__ import annotations

import numpy as np

PRIMARY = "quota_sector_mean"
METHODS = (
    "baseline",
    "duration",
    "quota_continuous",
    "trial_startup_sector_mean",
    "trial_duration_sector_mean",
    "quota_uniform_endpoint",
    "quota_sector_endpoint",
    "quota_uniform_mean",
    PRIMARY,
    "zero",
    "constant",
    "trialshuffle",
)
CONTRASTS = {
    "primary_minus_baseline": (PRIMARY, "baseline"),
    "primary_minus_zero": (PRIMARY, "zero"),
    "primary_minus_duration": (PRIMARY, "duration"),
    "duration_minus_baseline": ("duration", "baseline"),
    "primary_minus_quota_continuous": (PRIMARY, "quota_continuous"),
    "sector_weighting": (PRIMARY, "quota_uniform_mean"),
    "window_target": (PRIMARY, "quota_sector_endpoint"),
    "full_budget_minus_quota": ("trial_duration_sector_mean", PRIMARY),
    "quota_minus_startup_trial": (PRIMARY, "trial_startup_sector_mean"),
}


def mean(values):
    return float(np.mean(values)) if values else None


def interval(values):
    if not values:
        return [None, None]
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite bootstrap units required")
    rng = np.random.default_rng(20261023)
    indices = rng.integers(len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def summarize(rows, primary_method=PRIMARY):
    if primary_method != PRIMARY or not rows:
        raise ValueError("fixed V9 primary and nonempty rows required")
    if {r["method"] for r in rows} != set(METHODS):
        raise ValueError("all twelve fixed V9 methods required")
    ids = sorted({r["session_id"] for r in rows})
    lookup = {(r["session_id"], r["method"]): r for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(ids) * len(METHODS):
        raise ValueError("complete unique method-session grid required")
    for sid in ids:
        group = [lookup[sid, m] for m in METHODS]
        if len({r["participant"] for r in group}) != 1:
            raise ValueError("session participant mismatch")
        if len({r["score_mask_sha256"] for r in group}) != 1:
            raise ValueError("methods must share primary scoring mask")
        if len({r["metrics"]["r2"] is None for r in group}) != 1:
            raise ValueError("method-dependent score availability is forbidden")
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    methods = {}
    for method in METHODS:
        units = []
        for p in people:
            selected = [r for r in rows if r["participant"] == p and r["method"] == method]
            scored = [r for r in selected if r["metrics"]["r2"] is not None]
            values = [r["metrics"]["r2"] for r in scored]
            units.append(
                {
                    "participant": p,
                    "selected_sessions": len(selected),
                    "scored_sessions": len(scored),
                    "mean_r2": mean(values),
                    "positive_sessions": sum(v > 0 for v in values),
                    "session_bootstrap_95": interval(values),
                    "mean_summed_mse": mean([r["metrics"]["summed_mse"] for r in scored]),
                    "sessions": [
                        {k: r[k] for k in ("session_id", "post_implant_day", "metrics")}
                        for r in selected
                    ],
                }
            )
        valid = [u for u in units if u["mean_r2"] is not None]
        methods[method] = {
            "participants": units,
            "mean_r2": mean([u["mean_r2"] for u in valid]),
            "participant_bootstrap_95": interval([u["mean_r2"] for u in valid]),
            "scored_sessions": sum(u["scored_sessions"] for u in units),
            "n_participants": len(valid),
            "non_T2_mean_r2": mean([u["mean_r2"] for u in valid if u["participant"] != "T2"]),
        }
    contrasts = {}
    for name, (first, second) in CONTRASTS.items():
        units = []
        for p in people:
            pairs = []
            for sid in ids:
                a, b = lookup[sid, first], lookup[sid, second]
                if a["participant"] == p and a["metrics"]["r2"] is not None:
                    pairs.append(
                        {"session_id": sid, "gain": a["metrics"]["r2"] - b["metrics"]["r2"]}
                    )
            values = [pair["gain"] for pair in pairs]
            units.append(
                {
                    "participant": p,
                    "paired_sessions": len(pairs),
                    "mean_gain": mean(values),
                    "session_bootstrap_95": interval(values),
                    "session_gains": pairs,
                }
            )
        valid = [u for u in units if u["mean_gain"] is not None]
        contrasts[name] = {
            "first": first,
            "second": second,
            "participants": units,
            "mean_gain": mean([u["mean_gain"] for u in valid]),
            "paired_participant_95": interval([u["mean_gain"] for u in valid]),
        }
    t2 = next((u for u in methods[PRIMARY]["participants"] if u["participant"] == "T2"), None)

    def positive_gain(name):
        u = next((u for u in contrasts[name]["participants"] if u["participant"] == "T2"), None)
        return bool(
            u
            and u["paired_sessions"] == 6
            and u["mean_gain"] is not None
            and u["mean_gain"] > 0
            and u["session_bootstrap_95"][0] > 0
        )

    recovery = bool(
        t2
        and t2["selected_sessions"] == t2["scored_sessions"] == 6
        and t2["mean_r2"] >= 0.05
        and t2["positive_sessions"] >= 4
        and positive_gain("primary_minus_baseline")
        and positive_gain("primary_minus_zero")
    )
    return {
        "selected_sessions": len(ids),
        "primary_method": PRIMARY,
        "methods": methods,
        "contrasts": contrasts,
        "gates": {
            "useful_T2_recovery": recovery,
            "incremental_benefit_over_more_labels": recovery
            and positive_gain("primary_minus_duration"),
        },
        "interval_limit": (
            "T2 intervals resample available sessions from one person; "
            "descriptive, not independent participant replication"
        ),
    }
