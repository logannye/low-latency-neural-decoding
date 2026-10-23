"""Matched V8 summaries: sessions within people, then people equally.

T2 session intervals are descriptive with three fresh sessions, not independent
participant replication. A changed secondary target cannot satisfy primary gates.
"""

from __future__ import annotations

import numpy as np


def mean(values):
    return float(np.mean(values)) if values else None


def interval(values):
    if not values:
        return [None, None]
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite bootstrap units required")
    rng = np.random.default_rng(20261023)
    samples = rng.integers(len(values), size=(10000, len(values)))
    return np.quantile(values[samples].mean(1), [0.025, 0.975]).tolist()


def factorial_effects(rows):
    """Descriptive paired factorial main effects and interactions; never selection."""
    lookup = {(r["session_id"], r["method"]): r for r in rows}

    def method(r, p, w):
        if (r, p, w) == (0, 0, 0):
            return "baseline"
        return (
            f"{('baseline', 'selected')[r]}_{('all', 'clean')[p]}_{('uniform', 'equal_trial')[w]}"
        )

    cells = [(r, p, w) for r in (0, 1) for p in (0, 1) for w in (0, 1)]
    names = [method(*c) for c in cells]
    if not set(names) <= {r["method"] for r in rows}:
        return {}
    output = {}
    for label, axes in (
        ("representation", (0,)),
        ("clean_phase", (1,)),
        ("equal_trial", (2,)),
        ("representation_x_phase", (0, 1)),
        ("representation_x_weighting", (0, 2)),
        ("phase_x_weighting", (1, 2)),
        ("three_way", (0, 1, 2)),
    ):
        session_units = []
        for sid in sorted({r["session_id"] for r in rows}):
            group = [lookup[sid, name] for name in names]
            if any(r["metrics"]["r2"] is None for r in group):
                continue
            delta = sum(
                np.prod([2 * c[a] - 1 for a in axes]) * r["metrics"]["r2"]
                for c, r in zip(cells, group, strict=True)
            ) / (2 ** (3 - len(axes)))
            session_units.append(
                {"session_id": sid, "participant": group[0]["participant"], "effect": float(delta)}
            )
        people = sorted({u["participant"] for u in session_units}, key=lambda p: int(p[1:]))
        units = [
            {
                "participant": p,
                "mean_effect": mean([u["effect"] for u in session_units if u["participant"] == p]),
            }
            for p in people
        ]
        t2 = [u["effect"] for u in session_units if u["participant"] == "T2"]
        output[label] = {
            "participants": units,
            "sessions": session_units,
            "mean_effect": mean([u["mean_effect"] for u in units]),
            "paired_participant_95": interval([u["mean_effect"] for u in units]),
            "T2_mean_effect": mean(t2),
            "T2_session_95": interval(t2),
            "role": "Prespecified descriptive attribution; no multiplicity-adjusted claim",
        }
    return output


def summarize(rows, primary_method):
    if not rows:
        raise ValueError("nonempty rows required")
    methods = sorted({r["method"] for r in rows})
    if not {primary_method, "baseline", "zero"} <= set(methods):
        raise ValueError("primary, baseline and zero required")
    lookup = {(r["session_id"], r["method"]): r for r in rows}
    ids = sorted({r["session_id"] for r in rows})
    if len(lookup) != len(rows) or len(rows) != len(ids) * len(methods):
        raise ValueError("complete unique method-session grid required")
    for sid in ids:
        group = [lookup[sid, m] for m in methods]
        if len({r["score_mask_sha256"] for r in group}) != 1:
            raise ValueError("all methods must share primary scoring mask")
        if len({r["metrics"]["r2"] is None for r in group}) != 1:
            raise ValueError("methods must share primary score availability")
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    summaries = {}
    for method in methods:
        units = []
        for person in people:
            selected = [r for r in rows if r["participant"] == person and r["method"] == method]
            scored = [r for r in selected if r["metrics"]["r2"] is not None]
            values = [r["metrics"]["r2"] for r in scored]
            units.append(
                {
                    "participant": person,
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
        scored_units = [u for u in units if u["mean_r2"] is not None]
        summaries[method] = {
            "participants": units,
            "mean_r2": mean([u["mean_r2"] for u in scored_units]),
            "participant_bootstrap_95": interval([u["mean_r2"] for u in scored_units]),
            "scored_sessions": sum(u["scored_sessions"] for u in units),
            "n_participants": len(scored_units),
            "non_T2_mean_r2": mean(
                [u["mean_r2"] for u in scored_units if u["participant"] != "T2"]
            ),
            "non_T2_participant_bootstrap_95": interval(
                [u["mean_r2"] for u in scored_units if u["participant"] != "T2"]
            ),
        }
    contrasts = {}
    for comparator in ("baseline", "zero"):
        units = []
        for person in people:
            values = []
            for sid in ids:
                row = lookup[sid, primary_method]
                if row["participant"] == person and row["metrics"]["r2"] is not None:
                    values.append(row["metrics"]["r2"] - lookup[sid, comparator]["metrics"]["r2"])
            units.append(
                {
                    "participant": person,
                    "mean_gain": mean(values),
                    "session_gains": values,
                    "session_bootstrap_95": interval(values),
                }
            )
        scored = [u for u in units if u["mean_gain"] is not None]
        contrasts[f"primary_minus_{comparator}"] = {
            "participants": units,
            "mean_gain": mean([u["mean_gain"] for u in scored]),
            "paired_participant_95": interval([u["mean_gain"] for u in scored]),
        }
    t2 = next(
        (u for u in summaries[primary_method]["participants"] if u["participant"] == "T2"), None
    )
    comparisons = [
        next((u for u in c["participants"] if u["participant"] == "T2"), None)
        for c in contrasts.values()
    ]
    gate = bool(
        t2
        and t2["selected_sessions"] == t2["scored_sessions"] == 3
        and t2["mean_r2"] >= 0.05
        and t2["positive_sessions"] >= 2
        and all(
            c
            and c["mean_gain"] is not None
            and c["mean_gain"] > 0
            and c["session_bootstrap_95"][0] > 0
            for c in comparisons
        )
    )
    return {
        "selected_sessions": len(ids),
        "primary_method": primary_method,
        "methods": summaries,
        "contrasts": contrasts,
        "factorial_effects": factorial_effects(rows),
        "gates": {"useful_T2_primary_recovery": gate},
        "interval_limit": (
            "Three T2 sessions give descriptive session uncertainty; cohort intervals resample "
            "people. Neither establishes untouched-participant transfer."
        ),
    }
