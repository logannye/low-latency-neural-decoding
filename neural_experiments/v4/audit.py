"""Independent V4 protocol and numerical references.

This module does not import the V4 predictor, fitter, scorer or importer.
References use direct NumPy recurrences and ridge normal equations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit import compare_structure
from neural_experiments.v3.audit import (
    contained,
    mask_hash,
    numeric,
    read,
    score_reference,
    segment_ranges,
    sha256,
    validate_feature_metadata,
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA256 = "f00419baf098ab5e42d6757714bd43cc5107579edf2a7c915c24a4f6ff7d8aba"


def reference_representation(features, segments, config):
    """Previous slow state subtraction, then current fast update; reset each block."""
    required = {"x_tx3", "x_tx45", "x_sbp"}
    allowed = required | {"timestamps", "sequence_id", "channel_ids"}
    if not required <= set(features) or set(features) - allowed:
        raise ValueError("Only neural features and recording metadata enter replay")
    arrays = {k: np.asarray(features[k], dtype=np.float64) for k in required}
    tx = arrays["x_tx3" if config["family"].startswith("tx3") else "x_tx45"]
    if tx.ndim != 2 or any(
        v.shape != tx.shape or not np.isfinite(v).all() or np.any(v < 0) for v in arrays.values()
    ):
        raise ValueError("Invalid matching nonnegative feature arrays")
    raw = np.sqrt(tx)
    if config["family"].endswith("_sbp"):
        raw = np.concatenate((raw, np.log1p(arrays["x_sbp"])), axis=1)
    result = np.empty_like(raw)
    fast_decay = np.exp(-0.02 / config["tau_seconds"])
    tau = config["baseline_tau_seconds"]
    slow_decay = None if tau is None else np.exp(-0.02 / tau)
    for a, b in segment_ranges(segments, len(raw)):
        slow = raw[a].copy()
        fast = raw[a].copy() if tau is None else np.zeros(raw.shape[1])
        result[a] = fast
        for t in range(a + 1, b):
            centered = raw[t] if tau is None else raw[t] - slow
            fast = fast_decay * fast + (1 - fast_decay) * centered
            if tau is not None:
                slow = slow_decay * slow + (1 - slow_decay) * raw[t]
            result[t] = fast
    return result


def reference_design(raw, prefix, channels):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not 0 < prefix <= len(raw) or raw.shape[1] not in (channels, channels * 2):
        raise ValueError("Invalid normalization prefix or dimensions")
    floor = np.full(raw.shape[1], 0.1)
    floor[channels:] = 0.01
    mean, scale = raw[:prefix].mean(0), np.maximum(raw[:prefix].std(0), floor)
    z = np.clip((raw - mean) / scale, -8, 8)
    return np.column_stack((z, np.ones(len(raw)))), mean, scale


def reference_mask(mask, segments, stop=None, stride=1):
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.ndim != 1:
        raise ValueError("Training mask must be a boolean vector")
    stop = len(mask) if stop is None else stop
    if not 0 <= stop <= len(mask) or stride not in (1, 5):
        raise ValueError("Unplanned training prefix or stride")
    value = mask.copy()
    for a, b in segment_ranges(segments, len(mask)):
        value[a : min(a + 62, b)] = False
    value &= np.arange(len(value)) % stride == 0
    return value[:stop]


def reference_statistics(x, y, mask):
    x, y, mask = np.asarray(x, float), np.asarray(y), np.asarray(mask)
    if x.ndim != 2 or y.shape != (len(x), 2) or mask.shape != (len(x),) or mask.dtype != bool:
        raise ValueError("Invalid sufficient-statistics geometry")
    if not np.isfinite(x).all() or not np.isfinite(y[mask]).all() or not np.all(x[:, -1] == 1):
        raise ValueError("Invalid augmented design or selected labels")
    xx, yy = x[mask], np.asarray(y[mask], float)
    n, d = len(xx), x.shape[1]
    return {
        "gram": xx.T @ xx / n if n else np.zeros((d, d)),
        "rhs": xx.T @ yy / n if n else np.zeros((d, 2)),
        "n": n,
    }


def reference_history(stats, dimension):
    if not stats:
        return np.zeros((dimension, 2))
    if any(s["n"] < 50 for s in stats):
        raise ValueError("Insufficient history must be explicitly omitted before fitting")
    gram = sum(s["gram"] for s in stats) / len(stats)
    rhs = sum(s["rhs"] for s in stats) / len(stats)
    penalty = np.ones(dimension)
    penalty[-1] = 0
    return np.linalg.solve(gram + np.diag(penalty), rhs)


def validate_weights(weights, stats, prior=None, gamma=None, daily_penalty=None, free_bias=True):
    """Check the declared convex optimum and no-label fallback without model.fit."""
    weights = np.asarray(weights)
    if (
        weights.dtype != np.float64
        or weights.shape != stats["rhs"].shape
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Saved weight geometry/dtype mismatch")
    history_mode = prior is not None
    if stats["n"] < 50:
        expected = prior if history_mode else np.zeros_like(weights)
        if not np.allclose(weights, expected, rtol=1e-10, atol=1e-10):
            raise ValueError("Low-label fallback differs from exact prior/zero")
        return {"fallback": True, "normal_equation_relative_residual": None}
    gram, rhs = stats["gram"], stats["rhs"]
    if history_mode:
        diagonal = np.full(len(weights), gamma)
        diagonal[-1] *= 0.1
        rhs = rhs + diagonal[:, None] * prior
    else:
        diagonal = np.full(len(weights), daily_penalty)
        diagonal[-1] = 0
        if not free_bias:
            if not np.array_equal(weights[-1], np.zeros(2)):
                raise ValueError("Constrained zero-intercept model has nonzero intercept")
            gram, rhs, diagonal, weights = gram[:-1, :-1], rhs[:-1], diagonal[:-1], weights[:-1]
    residual = (gram + np.diag(diagonal)) @ weights - rhs
    relative = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(rhs)))
    if relative > 1e-8:
        raise ValueError("Weights are not the declared prefix/history ridge optimum")
    return {"fallback": False, "normal_equation_relative_residual": relative}


def validate_normalization(arrays, raw, prefix, channels, config):
    if set(arrays) != {"format_version", "config", "mean", "scale", "weights"}:
        raise ValueError("Unexpected model export fields")
    if (
        arrays["format_version"].shape != ()
        or arrays["format_version"].dtype != np.int64
        or arrays["format_version"].item() != 4
    ):
        raise ValueError("Unsupported model format")
    if arrays["config"].shape != () or arrays["config"].dtype.kind != "U":
        raise ValueError("Scalar Unicode model configuration required")
    if json.loads(arrays["config"].item()) != config:
        raise ValueError("Exported model configuration mismatch")
    x, mean, scale = reference_design(raw, prefix, channels)
    for key, expected in (("mean", mean), ("scale", scale)):
        value = arrays[key]
        if (
            value.dtype != np.float64
            or value.shape != expected.shape
            or not np.allclose(value, expected, rtol=1e-10, atol=1e-10)
        ):
            raise ValueError("Model normalization uses wrong prefix/feature transform")
    if arrays["weights"].shape != (raw.shape[1] + 1, 2) or arrays["weights"].dtype != np.float64:
        raise ValueError("Invalid model weights")
    return x


def replay(arrays, raw):
    z = np.clip((raw - arrays["mean"]) / arrays["scale"], -8, 8)
    prediction = z @ arrays["weights"][:-1] + arrays["weights"][-1]
    return prediction / np.maximum(1, np.linalg.norm(prediction, axis=1))[:, None]


def candidate_grids(plan):
    """Enumerate the locked order independently; exact ties keep the first entry."""
    global_grid, targeted = [], []
    g, t = plan["global_grid"], plan["T3_targeted_grid"]
    for family, baseline, gamma in itertools.product(
        g["families"], g["baseline_tau_seconds"], g["adapt_gamma"]
    ):
        global_grid.append(
            {
                "candidate_id": f"g{len(global_grid):02d}",
                "config": {
                    "family": family,
                    "tau_seconds": g["tau_seconds_by_family"][family],
                    "baseline_tau_seconds": baseline,
                    "adapt_gamma": gamma,
                },
            }
        )
    for family, baseline, penalty, bias in itertools.product(
        t["families"], t["baseline_tau_seconds"], t["ridge_penalty"], t["free_intercept"]
    ):
        targeted.append(
            {
                "candidate_id": f"t{len(targeted):02d}",
                "config": {
                    "family": family,
                    "tau_seconds": t["tau_seconds"],
                    "baseline_tau_seconds": baseline,
                    "adapt_gamma": 1.0,
                },
                "penalty": penalty,
                "bias": int(bias),
            }
        )
    return global_grid, targeted


def selected_histories(query, items, layouts):
    eligible = [
        s
        for s in items
        if s["participant_id"] == query["participant_id"]
        and s["post_implant_day"] < query["post_implant_day"]
        and layouts[s["session_id"]] == layouts[query["session_id"]]
    ]
    return sorted(eligible, key=lambda s: (s["post_implant_day"], s["session_id"]))[-3:]


def checked_prior(query, items, layouts, stats, dimension):
    receipt, selected = [], []
    for item in selected_histories(query, items, layouts):
        st = stats[item["session_id"]]
        good = st["n"] >= 50
        receipt.append(
            {
                "session_id": item["session_id"],
                "post_implant_day": item["post_implant_day"],
                "n": st["n"],
                "disposition": "used" if good else "insufficient_labels",
            }
        )
        if good:
            selected.append(st)
    return reference_history(selected, dimension), receipt


def reconstruct_choices(rows, candidates, targeted, items, people):
    def ranking(source, grid, count_people=False):
        ranked = []
        for candidate in grid:
            eligible = [
                r
                for r in source
                if r["candidate_id"] == candidate["candidate_id"] and r["metrics"]["r2"] is not None
            ]
            grouped = {}
            for row in eligible:
                grouped.setdefault(row["participant"], []).append(row["metrics"]["r2"])
            if not grouped:
                raise ValueError("No source selection evidence")
            value = {
                "candidate_id": candidate["candidate_id"],
                "objective": float(np.mean([np.mean(v) for v in grouped.values()])),
            }
            if count_people:
                value["eligible_source_people"] = len(grouped)
            ranked.append(value)
        return ranked

    choices, targeted_choices = [], []
    for person in people:
        source = [r for r in rows if r["kind"] == "global" and r["participant"] != person]
        ranked = ranking(source, candidates, True)
        choices.append(
            {
                "target_participant": person,
                "ranking": ranked,
                "candidate_id": max(ranked, key=lambda r: r["objective"])["candidate_id"],
            }
        )
    for query in items:
        if query["participant_id"] != "T3":
            continue
        source = [
            r
            for r in rows
            if r["kind"] == "targeted"
            and r["participant"] == "T3"
            and r["post_implant_day"] < query["post_implant_day"]
            and r["metrics"]["r2"] is not None
        ]
        scope = "earlier_T3" if source else "other_people"
        if not source:
            source = [
                r
                for r in rows
                if r["kind"] == "targeted"
                and r["participant"] != "T3"
                and r["metrics"]["r2"] is not None
            ]
        ranked = ranking(source, targeted)
        targeted_choices.append(
            {
                "target_session_id": query["session_id"],
                "post_implant_day": query["post_implant_day"],
                "selection_scope": scope,
                "source_session_ids": sorted({r["session_id"] for r in source}),
                "candidate_id": max(ranked, key=lambda r: r["objective"])["candidate_id"],
                "ranking": ranked,
            }
        )
    return choices, targeted_choices


class Inputs:
    """Hash-bound prepared inputs, without importing the experiment's data loader."""

    def __init__(self, path, role, plan):
        self.path, self.index, self.checked = Path(path), read(path), set()
        index = self.index
        if (
            index["schema_version"] != 1
            or index["role"] != role
            or index["acquisition_plan_sha256"] != plan["acquisition_plan_sha256"]
            or index["provider"] != {"doi": "10.5061/dryad.x0k6djj1h", "version_id": 462702}
        ):
            raise ValueError("Wrong prepared input role or provider")
        items = index["sessions"]
        self.items = {s["session_id"]: s for s in items}
        if len(self.items) != len(items) or any(s["role"] != role for s in items):
            raise ValueError("Duplicate session or mixed input roles")
        if role == "development":
            verify(path, plan["development_index_sha256"])
            if len(items) != 80 or set(index["participants"]) != set(plan["participants"]):
                raise ValueError("Development cohort differs from the lock")
        else:
            if index["study_lock"]["sha256"] != PLAN_SHA256:
                raise ValueError("Fresh input was not prepared under the study lock")
        for item in items:
            if item["bin_seconds"] != 0.02 or item["n_bins"] <= 0:
                raise ValueError("Invalid recording geometry")
            if role != "t3_blocks" and item["n_bins"] > 30000:
                raise ValueError("Unauthorized post-600-second data")
            segment_ranges(item["segments"], item["n_bins"])

    def arrays(self, item):
        arrays = []
        for kind in ("feature", "target"):
            path = contained(self.path.parent, item[f"{kind}_path"])
            if path not in self.checked:
                verify(path, item[f"{kind}_sha256"])
                self.checked.add(path)
            arrays.append(numeric(path))
        features, target = arrays
        if set(features) != {
            "x_tx3",
            "x_tx45",
            "x_sbp",
            "timestamps",
            "sequence_id",
            "channel_ids",
        }:
            raise ValueError("Predictor feature container includes undeclared fields")
        validate_feature_metadata(features, item)
        for name in ("x_tx3", "x_tx45", "x_sbp"):
            values = features[name]
            if (
                values.dtype != np.float32
                or values.shape != (item["n_bins"], item["n_channels"])
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError("Invalid feature shape/dtype/value")
        if (
            set(target) != {"y", "eval_mask"}
            or target["y"].dtype != np.float32
            or target["y"].shape != (item["n_bins"], 2)
            or target["eval_mask"].dtype != bool
            or target["eval_mask"].shape != (item["n_bins"],)
        ):
            raise ValueError("Invalid target container")
        valid = target["y"][target["eval_mask"]]
        if (
            not np.isfinite(target["y"]).all()
            or not np.allclose(np.linalg.norm(valid, axis=1), 1, atol=1e-6, rtol=1e-6)
            or int(target["eval_mask"].sum()) != item["n_eval_bins"]
        ):
            raise ValueError("Invalid direction labels or eligible count")
        return features, target


def check_source_identity(selection, root):
    if (
        selection["schema_version"] != 1
        or selection["stage"] != "source"
        or selection["fresh_labels_opened"] is not False
    ):
        raise ValueError("Not a complete source-only selection")
    plan_path = contained(root, selection["plan"]["path"])
    verify(plan_path, PLAN_SHA256)
    if selection["plan"]["sha256"] != PLAN_SHA256:
        raise ValueError("Unrecognized study plan")
    plan = read(plan_path)
    expected_files = {
        "neural_experiments/v4/model.py",
        "neural_experiments/v4/data.py",
        "neural_experiments/v4/study.py",
        "neural_experiments/v3/study.py",
        "neural_experiments/v2/data.py",
    }
    if set(selection["code_sha256"]) != expected_files:
        raise ValueError("Missing executed source identities")
    for path, digest in selection["code_sha256"].items():
        verify(contained(root, path), digest)
    dev_path = contained(root, selection["development_index"]["path"])
    verify(dev_path, selection["development_index"]["sha256"])
    return plan, Inputs(dev_path, "development", plan)


def check_artifacts(row, raw, item, config, prefix, target, mask, root):
    model_path, trace_path = contained(root, row["model_path"]), contained(root, row["trace_path"])
    verify(model_path, row["model_sha256"])
    verify(trace_path, row["trace_sha256"])
    model, trace = numeric(model_path), numeric(trace_path)
    design = validate_normalization(model, raw, prefix, item["n_channels"], config)
    if (
        set(trace) != {"prediction", "eval_mask"}
        or trace["prediction"].dtype != np.float64
        or trace["prediction"].shape != (item["n_bins"], 2)
        or not np.isfinite(trace["prediction"]).all()
        or trace["eval_mask"].dtype != bool
        or not np.array_equal(trace["eval_mask"], mask)
        or row["eval_mask_sha256"] != mask_hash(mask)
    ):
        raise ValueError("Trace geometry or scoring mask mismatch")
    predicted = replay(model, raw)
    error = float(np.max(np.abs(predicted - trace["prediction"])))
    if not np.allclose(predicted, trace["prediction"], rtol=1e-9, atol=1e-9):
        raise ValueError("Independent full trace replay failed")
    metrics, _ = score_reference(target["y"], trace["prediction"], mask)
    metrics["mse"] = (
        float(np.square(target["y"][mask] - trace["prediction"][mask]).mean())
        if mask.any()
        else None
    )
    compare_structure(row["metrics"], metrics, "independent metrics")
    return design, model, metrics, error


def audit_selection(selection_path, root=ROOT):
    selection = read(selection_path)
    plan, inputs = check_source_identity(selection, root)
    candidates, targeted = candidate_grids(plan)
    compare_structure(selection["candidates"], candidates, "candidate grid")
    compare_structure(selection["targeted_candidates"], targeted, "targeted grid")
    rows = {(r["session_id"], r["candidate_id"]): r for r in selection["cases"]}
    expected_keys = {
        (sid, c["candidate_id"]) for sid in inputs.items for c in candidates + targeted
    }
    if set(rows) != expected_keys or len(rows) != len(selection["cases"]):
        raise ValueError("Incomplete or duplicate source case inventory")
    checked, residuals, errors, unavailable = [], [], [], 0
    items = list(inputs.items.values())
    for start in range(0, len(candidates), 2):
        group = candidates[start : start + 2]
        config = group[0]["config"]
        daily = [
            c
            for c in targeted
            if c["config"]["family"] == config["family"]
            and c["config"]["baseline_tau_seconds"] == config["baseline_tau_seconds"]
        ]
        stats, layouts = {}, {}
        for item in items:
            features, labels = inputs.arrays(item)
            raw = reference_representation(features, item["segments"], config)
            z, _, _ = reference_design(raw, len(raw), item["n_channels"])
            mask = reference_mask(labels["eval_mask"], item["segments"], stride=5)
            stats[item["session_id"]] = reference_statistics(z, labels["y"], mask)
            layouts[item["session_id"]] = tuple(features["channel_ids"])
        for item in items:
            features, labels = inputs.arrays(item)
            raw = reference_representation(features, item["segments"], config)
            prior, histories = checked_prior(item, items, layouts, stats, raw.shape[1] + 1)
            for candidate in group + daily:
                row = rows[item["session_id"], candidate["candidate_id"]]
                kind = "global" if candidate in group else "targeted"
                identity = {
                    "participant": item["participant_id"],
                    "session_id": item["session_id"],
                    "post_implant_day": item["post_implant_day"],
                    "kind": kind,
                    "candidate_id": candidate["candidate_id"],
                }
                for key, value in identity.items():
                    if row[key] != value:
                        raise ValueError("Source row identity mismatch")
                if kind == "targeted" and item["n_bins"] < 6000:
                    compare_structure(
                        row,
                        {**identity, "unavailable": "short_120s_prefix", "metrics": {"r2": None}},
                    )
                    checked.append(row)
                    unavailable += 1
                    continue
                stop = min(1500, len(raw)) if kind == "global" else 6000
                mask = labels["eval_mask"].copy()
                mask[: 6100 if kind == "global" else 15100] = False
                z, model, metrics, error = check_artifacts(
                    row, raw, item, candidate["config"], stop, labels, mask, root
                )
                training = reference_mask(labels["eval_mask"], item["segments"], stop)
                target = reference_statistics(z[:stop], labels["y"][:stop], training)
                for key, expected in {
                    "normalization_bins": stop,
                    "training_rows": target["n"],
                    "fallback": target["n"] < 50,
                    "history": histories if kind == "global" else [],
                }.items():
                    compare_structure(row[key], expected, key)
                optimum = validate_weights(
                    model["weights"],
                    target,
                    prior=prior if kind == "global" else None,
                    gamma=candidate["config"]["adapt_gamma"],
                    daily_penalty=candidate.get("penalty"),
                    free_bias=bool(candidate.get("bias", 1)),
                )
                if optimum["normal_equation_relative_residual"] is not None:
                    residuals.append(optimum["normal_equation_relative_residual"])
                errors.append(error)
                checked.append({**row, "metrics": metrics})
        print(
            f"independent source {group[0]['candidate_id']}..{group[-1]['candidate_id']} checked",
            flush=True,
        )
    # Preserve the runner's case order in the public-verifiable score ledger.
    by_key = {(r["session_id"], r["candidate_id"]): r for r in checked}
    checked = [by_key[r["session_id"], r["candidate_id"]] for r in selection["cases"]]
    choices, targeted_choices = reconstruct_choices(
        checked, candidates, targeted, items, plan["participants"]
    )
    compare_structure(selection["choices"], choices, "global whole-person exclusion")
    compare_structure(
        selection["targeted_choices"], targeted_choices, "targeted chronological selection"
    )
    return {
        "status": "PASS",
        "stage": "source",
        "created_at": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": sha256(__file__),
        "selection_sha256": sha256(selection_path),
        "plan_sha256": PLAN_SHA256,
        "development_index_sha256": sha256(inputs.path),
        "code_sha256": selection["code_sha256"],
        "cases_checked": len(checked),
        "models_and_full_traces_checked": len(errors),
        "unavailable_cases_checked": unavailable,
        "global_choices_checked": len(choices),
        "targeted_choices_checked": len(targeted_choices),
        "max_prediction_absolute_error": max(errors, default=0),
        "max_normal_equation_relative_residual": max(residuals, default=0),
        "independent_scores": [
            {k: r[k] for k in ("participant", "session_id", "candidate_id", "kind", "metrics")}
            for r in checked
        ],
        "scope": (
            "Independent NumPy recurrence, prefix statistics, ridge optimality, sklearn score "
            "reference, chronological/LOPO selection; no V4 model/study/data imports."
        ),
    }


def interval(values):
    """Fixed descriptive resampling of complete participant/session means."""
    a = np.asarray(values, dtype=float)
    if not len(a):
        return [None, None]
    draw = np.random.default_rng(20261023).integers(0, len(a), size=(10000, len(a)))
    return np.quantile(a[draw].mean(axis=1), [0.025, 0.975]).tolist()


def reconstruct_summary(rows, plan):
    domains = {}
    for domain in ("new_sessions", "t3_blocks"):
        methods, contrasts = {}, []
        unit_field = "participant" if domain == "new_sessions" else "session_id"
        usable = [r for r in rows if r["domain"] == domain and r["metrics"]["r2"] is not None]
        for method in plan["assessment"][domain]["methods"]:
            group = [r for r in usable if r["method"] == method]
            units = []
            for unit in sorted({r[unit_field] for r in group}):
                scores = [r["metrics"] for r in group if r[unit_field] == unit]
                units.append(
                    {
                        "unit": unit,
                        "r2": float(np.mean([s["r2"] for s in scores])),
                        "angular_error_degrees": float(
                            np.mean([s["angular_error_degrees"] for s in scores])
                        ),
                        "n_sessions": len(scores),
                    }
                )
            values = [u["r2"] for u in units]
            methods[method] = {
                "units": units,
                "n_units": len(units),
                "n_sessions": len(group),
                "mean_r2": float(np.mean(values)) if values else None,
                "r2_bootstrap_95": interval(values),
                "positive_units": sum(v > 0 for v in values),
                "mean_angular_error_degrees": float(
                    np.mean([u["angular_error_degrees"] for u in units])
                )
                if units
                else None,
            }
        for arm in (
            "selected_prior0",
            "selected_prior30",
            "selected_prior120",
            "targeted_daily120",
        ):
            if arm not in methods:
                continue
            for control in ("zero", "prefix_mean30", "v3_daily120", "selected_scratch30"):
                left = {r["session_id"]: r for r in usable if r["method"] == arm}
                right = {r["session_id"]: r for r in usable if r["method"] == control}
                gains = {}
                for sid in sorted(left.keys() & right.keys()):
                    a, b = left[sid], right[sid]
                    if a["eval_mask_sha256"] != b["eval_mask_sha256"]:
                        raise ValueError("Paired comparison uses different scoring masks")
                    gains.setdefault(a[unit_field], []).append(
                        a["metrics"]["r2"] - b["metrics"]["r2"]
                    )
                values = [float(np.mean(gains[u])) for u in sorted(gains)]
                contrasts.append(
                    {
                        "arm": arm,
                        "control": control,
                        "n_units": len(values),
                        "mean_gain": float(np.mean(values)) if values else None,
                        "gain_bootstrap_95": interval(values),
                    }
                )
        domains[domain] = {"methods": methods, "contrasts": contrasts}

    def useful(arm):
        m = domains["new_sessions"]["methods"][arm]
        pairs = [
            c
            for c in domains["new_sessions"]["contrasts"]
            if c["arm"] == arm and c["control"] in ("zero", "prefix_mean30")
        ]
        return bool(
            m["n_units"] == 8
            and m["mean_r2"] >= 0.1
            and m["r2_bootstrap_95"][0] > 0
            and m["positive_units"] >= 6
            and m["mean_angular_error_degrees"] <= 80
            and len(pairs) == 2
            and all(c["n_units"] == 8 and c["mean_gain"] >= 0.05 for c in pairs)
        )

    ni = next(
        c
        for c in domains["new_sessions"]["contrasts"]
        if c["arm"] == "selected_prior30" and c["control"] == "v3_daily120"
    )
    t = domains["t3_blocks"]["methods"]["targeted_daily120"]
    tc = {
        c["control"]: c
        for c in domains["t3_blocks"]["contrasts"]
        if c["arm"] == "targeted_daily120"
    }
    recovered = bool(
        t["n_units"] == 8
        and t["mean_r2"] >= 0.05
        and t["r2_bootstrap_95"][0] > 0
        and t["positive_units"] >= 6
        and t["mean_angular_error_degrees"] <= 80
        and tc["zero"]["n_units"] == 8
        and tc["zero"]["mean_gain"] >= 0.05
        and tc["v3_daily120"]["n_units"] == 8
        and tc["v3_daily120"]["gain_bootstrap_95"][0] > 0
    )
    return {
        "domains": domains,
        "gates": {
            "reduced_label_useful": useful("selected_prior30"),
            "zero_fresh_label_useful": useful("selected_prior0"),
            "T3_recovered": recovered,
            "calibration_noninferiority": bool(
                useful("selected_prior30")
                and ni["n_units"] == 8
                and ni["gain_bootstrap_95"][0] > -0.05
            ),
        },
    }


def validate_fresh_provenance(inputs, development, plan):
    """Authenticate archive order and known-prefix bindings without the V4 importer."""
    index, role = inputs.index, inputs.index["role"]
    expected_people = plan["new_session_participants"] if role == "new_sessions" else ["T3"]
    if index["participants"] != expected_people:
        raise ValueError("Fresh domain participant inventory mismatch")
    for key in ("v2_index_sha256", "v3_index_sha256"):
        if index[key] != development.index[key]:
            raise ValueError("Fresh input parent cohort changed")
    expected = {}
    if role == "new_sessions":
        if len(index["acquisition_sources"]) != len(expected_people):
            raise ValueError("Missing acquisition provenance")
        for person in expected_people:
            references = [
                r for r in index["acquisition_sources"] if r["path"] == f"acquisition/{person}.json"
            ]
            if len(references) != 1:
                raise ValueError("Duplicate or missing acquisition receipt")
            ref = references[0]
            path = contained(inputs.path.parent, ref["path"])
            verify(path, ref["sha256"])
            receipt = read(path)
            prior = sorted(
                [s["source"] for s in development.items.values() if s["participant_id"] == person],
                key=lambda s: s["archive_ordinal"],
            )
            if (
                receipt["status"] != "complete"
                or receipt["participant_id"] != person
                or receipt["plan_sha256"] != plan["acquisition_plan_sha256"]
                or receipt["v2_index_sha256"] != index["v2_index_sha256"]
                or receipt["v3_index_sha256"] != index["v3_index_sha256"]
                or receipt["selection"]["regular_session_ranks_one_based"] != [10, 11, 12]
                or receipt["selection"]["scores_or_duration_used_for_selection"] is not False
                or len(prior) != 9
            ):
                raise ValueError("Invalid fixed archive acquisition role")
            archive = receipt["archive"]
            if (
                archive["file_id"] != prior[0]["archive_file_id"]
                or archive["provider_sha256"] != prior[0]["archive_sha256_published"]
            ):
                raise ValueError("Fresh acquisition changed provider archive")
            pattern = rf"(?:\./)?decoding/{person}/{person}_day_(\d+)_decoding\.mat"
            seen = [
                r
                for r in receipt["seen_members"]
                if r["regular_file"] and re.fullmatch(pattern, r["name"])
            ]
            selected = receipt["selected"]
            authenticated = receipt["verified_prior_members"]
            if (
                len(authenticated) != 9
                or len(seen) != 9 + len(selected)
                or not 0 <= len(selected) <= 3
                or (len(selected) < 3 and not receipt["tar_end_observed"])
                or receipt["missing_requested_ranks"] != list(range(10 + len(selected), 13))
            ):
                raise ValueError("Archive rank or shortage accounting mismatch")
            for rank, (member, observed) in enumerate(
                zip(authenticated + selected, seen, strict=True), 1
            ):
                match = re.fullmatch(pattern, member["member_name"])
                if (
                    match is None
                    or member["regular_session_rank"] != rank
                    or member["participant_id"] != person
                    or member["post_implant_day"] != int(match.group(1))
                    or member["session_id"] != Path(member["member_name"]).stem
                    or member["member_name"] != observed["name"]
                    or member["member_size"] != observed["size"]
                    or member["archive_ordinal"] != observed["ordinal"]
                ):
                    raise ValueError("Archive walk does not bind selected identity")
                if rank <= 9:
                    for key in (
                        "member_name",
                        "member_size",
                        "member_sha256",
                        "archive_ordinal",
                        "session_id",
                    ):
                        if member[key] != prior[rank - 1][key]:
                            raise ValueError("Previously exposed archive prefix changed")
                else:
                    expected[member["session_id"]] = {
                        **member,
                        "archive_file_id": archive["file_id"],
                        "archive_sha256_published": archive["provider_sha256"],
                        "full_archive_sha256_verified": False,
                    }
    else:
        if (
            index["development_index_sha256"] != sha256(development.path)
            or Path(index["development_index_path"]).resolve() != development.path.resolve()
        ):
            raise ValueError("Known-prefix source index changed")
        expected = {
            sid: s["source"] for sid, s in development.items.items() if s["participant_id"] == "T3"
        }
    if set(inputs.items) != set(expected):
        raise ValueError("Fresh session inventory omitted or substituted an identity")
    for sid, item in inputs.items.items():
        if item["source"] != expected[sid]:
            raise ValueError("Prepared input source differs from authenticated selection")
        if item["post_implant_day"] != expected[sid]["post_implant_day"]:
            raise ValueError("Prepared chronology differs from source identity")
        raw_path = contained(item["raw_source_root"], item["source"]["raw_path"])
        verify(raw_path, item["source"]["member_sha256"])
        if role == "new_sessions":
            if item["n_bins"] != min(item["full_prepared_n_bins"], 30000):
                raise ValueError("New recording does not preserve the fixed 600-second cap")
        else:
            known = development.items[sid]
            expected_path = contained(development.path.parent, known["target_path"])
            if (
                Path(item["known_prefix_target_path"]).resolve() != expected_path
                or item["known_prefix_target_sha256"] != known["target_sha256"]
                or item["known_prefix_n_bins"] != known["n_bins"]
            ):
                raise ValueError("T3 calibration points outside its fixed known-prefix file")
            eligible = [
                s for s in item["segments"] if s["start"] >= 30000 and s["end"] > s["start"]
            ]
            selected = [s for s in item["segments"] if s["selected_for_evaluation"]]
            if (
                selected != eligible[:2]
                or not selected
                or len(selected) > 2
                or item["n_bins"] != selected[-1]["end"]
                or item["selected_evaluation_block_ids"] != [s["block_id"] for s in selected]
                or item["unavailable_continuation_blocks"] != 2 - len(selected)
            ):
                raise ValueError("T3 continuation block selection differs from fixed metadata rule")
            # Read only acquisition block metadata from the independently hashed raw member.
            from scipy.io import loadmat

            blocks = loadmat(raw_path, simplify_cells=True, variable_names=["blocks"])["blocks"]
            ids, starts, ends = [
                np.asarray(blocks[k]).reshape(-1)
                for k in ("block_id", "sample_start_index", "sample_end_index")
            ]
            if (
                len(ids) != len(starts)
                or len(ids) != len(ends)
                or np.any(ends <= starts)
                or np.any(starts < 0)
                or np.any(starts[1:] < ends[:-1])
            ):
                raise ValueError("Malformed original acquisition blocks")
            lengths = ((ends - starts) // 2).astype(np.int64)
            offsets = np.r_[0, np.cumsum(lengths)]
            chosen = [i for i in range(len(ids)) if offsets[i] >= 30000 and lengths[i] > 0][:2]
            expected_segments = [
                {
                    "start": int(offsets[i]),
                    "end": int(offsets[i + 1]),
                    "block_id": int(ids[i]),
                    "original_block_index": i,
                    "selected_for_evaluation": i in chosen,
                }
                for i in range(chosen[-1] + 1)
            ]
            actual_segments = [{k: s[k] for k in expected_segments[0]} for s in item["segments"]]
            compare_structure(actual_segments, expected_segments, "whole original T3 blocks")


def audit_assessment(assessment_path, root=ROOT):
    assessment = read(assessment_path)
    if (
        assessment["schema_version"] != 1
        or assessment["stage"] != "assessment"
        or assessment["status"] != "assessment_complete"
    ):
        raise ValueError("Assessment is incomplete")
    selection_path = contained(root, assessment["selection_path"])
    verify(selection_path, assessment["selection_sha256"])
    selection = read(selection_path)
    plan, development = check_source_identity(selection, root)
    source_audit_path = contained(root, assessment["source_audit_path"])
    verify(source_audit_path, assessment["source_audit_sha256"])
    source_audit = read(source_audit_path)
    if (
        source_audit["status"] != "PASS"
        or source_audit["stage"] != "source"
        or source_audit["selection_sha256"] != sha256(selection_path)
        or source_audit["plan_sha256"] != PLAN_SHA256
        or source_audit["development_index_sha256"] != sha256(development.path)
        or source_audit["code_sha256"] != selection["code_sha256"]
    ):
        raise ValueError("Source selection lacks a matching passed audit")
    source_scores = [
        {**r, "post_implant_day": development.items[r["session_id"]]["post_implant_day"]}
        for r in source_audit["independent_scores"]
    ]
    choices, targeted_choices = reconstruct_choices(
        source_scores,
        selection["candidates"],
        selection["targeted_candidates"],
        list(development.items.values()),
        plan["participants"],
    )
    compare_structure(selection["choices"], choices)
    compare_structure(selection["targeted_choices"], targeted_choices)
    if set(assessment["indices"]) != {"new_sessions", "t3_blocks"}:
        raise ValueError("Separate assessment domains missing")
    inputs_by_domain = {}
    for role, ref in assessment["indices"].items():
        path = contained(root, ref["path"])
        verify(path, ref["sha256"])
        inputs = Inputs(path, role, plan)
        validate_fresh_provenance(inputs, development, plan)
        inputs_by_domain[role] = inputs
    rows = {(r["domain"], r["session_id"], r["method"]): r for r in assessment["cases"]}
    expected = {
        (role, sid, method)
        for role, inputs in inputs_by_domain.items()
        for sid in inputs.items
        for method in plan["assessment"][role]["methods"]
    }
    if set(rows) != expected or len(rows) != len(assessment["cases"]):
        raise ValueError("Assessment case inventory omitted, duplicated or added cases")
    layouts, stats_cache = {}, {}
    for sid, item in development.items.items():
        features, _ = development.arrays(item)
        layouts[sid] = tuple(features["channel_ids"])

    def prior_for(item, config, dimension):
        config_key = tuple(config[k] for k in ("family", "tau_seconds", "baseline_tau_seconds"))
        selected = selected_histories(item, list(development.items.values()), layouts)
        stats = {}
        for history in selected:
            sid = history["session_id"]
            key = (sid, config_key)
            if key not in stats_cache:
                features, target = development.arrays(history)
                raw = reference_representation(features, history["segments"], config)
                x, _, _ = reference_design(raw, len(raw), history["n_channels"])
                train = reference_mask(target["eval_mask"], history["segments"], stride=5)
                stats_cache[key] = reference_statistics(x, target["y"], train)
            stats[sid] = stats_cache[key]
        return checked_prior(item, list(development.items.values()), layouts, stats, dimension)

    checked, errors, residuals, unavailable, fallbacks = [], [], [], 0, 0
    for domain, inputs in inputs_by_domain.items():
        for sid, item in inputs.items.items():
            features, labels = inputs.arrays(item)
            layouts[sid] = tuple(features["channel_ids"])
            methods = plan["assessment"][domain]["methods"]
            if domain == "new_sessions" and item["n_bins"] < 6100:
                for method in methods:
                    row = rows[domain, sid, method]
                    expected_row = {
                        "domain": domain,
                        "method": method,
                        "participant": item["participant_id"],
                        "session_id": sid,
                        "unavailable": "duration_before_common_evaluation",
                        "metrics": {"r2": None},
                    }
                    compare_structure(row, expected_row)
                    checked.append(row)
                    unavailable += 1
                continue
            mask = labels["eval_mask"].copy()
            fit_labels = labels
            if domain == "new_sessions":
                mask[:6100] = False
            else:
                allowed = np.zeros(item["n_bins"], bool)
                for segment in item["segments"]:
                    if segment["selected_for_evaluation"]:
                        allowed[segment["start"] : segment["end"]] = True
                if np.any(mask & ~allowed) or np.any(labels["y"][~allowed]):
                    raise ValueError("T3 scorer labels escape the selected fresh blocks")
                known_features, known = development.arrays(development.items[sid])
                for name in features:
                    before = known_features[name]
                    current = (
                        features[name] if name == "channel_ids" else features[name][: len(before)]
                    )
                    if not np.array_equal(current, before):
                        raise ValueError("T3 causal feature prefix differs from its known session")
                fit_labels = {"y": np.zeros_like(labels["y"]), "eval_mask": np.zeros_like(mask)}
                fit_labels["y"][:6000] = known["y"][:6000]
                fit_labels["eval_mask"][:6000] = known["eval_mask"][:6000]
            selected_id = next(
                c["candidate_id"]
                for c in choices
                if c["target_participant"] == item["participant_id"]
            )
            selected = next(c for c in selection["candidates"] if c["candidate_id"] == selected_id)
            raw_cache = {}
            for method in methods:
                row = rows[domain, sid, method]
                config = selected["config"]
                candidate_id = selected_id
                budget = (
                    0
                    if method in ("zero", "selected_prior0")
                    else 120
                    if method.endswith("120")
                    else 30
                )
                normalizer, penalty, bias = 30, config["adapt_gamma"], 1
                daily = method.startswith(("v3_daily", "targeted_daily"))
                if method.startswith("v3_daily"):
                    config = {
                        "family": "tx45_sbp",
                        "tau_seconds": 0.2,
                        "baseline_tau_seconds": None,
                        "adapt_gamma": 1.0,
                    }
                    candidate_id, normalizer, penalty = "v3_fixed", budget, 1.0
                elif method.startswith("targeted_daily"):
                    candidate_id = next(
                        c["candidate_id"] for c in targeted_choices if c["target_session_id"] == sid
                    )
                    candidate = next(
                        c
                        for c in selection["targeted_candidates"]
                        if c["candidate_id"] == candidate_id
                    )
                    config, penalty, bias = (
                        candidate["config"],
                        candidate["penalty"],
                        candidate["bias"],
                    )
                    normalizer = budget
                raw_key = tuple(
                    config[k] for k in ("family", "tau_seconds", "baseline_tau_seconds")
                )
                if raw_key not in raw_cache:
                    raw_cache[raw_key] = reference_representation(
                        features, item["segments"], config
                    )
                raw = raw_cache[raw_key]
                x, fitted, metrics, error = check_artifacts(
                    row, raw, item, config, normalizer * 50, labels, mask, root
                )
                stop = budget * 50
                train = reference_mask(fit_labels["eval_mask"], item["segments"], stop)
                st = reference_statistics(x[:stop], fit_labels["y"][:stop], train)
                prior, histories = np.zeros_like(fitted["weights"]), []
                if method.startswith("selected_prior"):
                    prior, histories = prior_for(item, config, len(prior))
                metadata = {
                    "domain": domain,
                    "method": method,
                    "participant": item["participant_id"],
                    "session_id": sid,
                    "post_implant_day": item["post_implant_day"],
                    "kind": "assessment",
                    "candidate_id": candidate_id,
                    "budget_seconds": budget,
                    "normalization_bins": normalizer * 50,
                    "training_rows": st["n"],
                    "history": histories,
                    "fallback": budget > 0 and st["n"] < 50,
                    "penalty": penalty,
                    "bias": bias,
                }
                for key, expected_value in metadata.items():
                    compare_structure(row[key], expected_value, key)
                if method in ("zero", "prefix_mean30"):
                    expected_weights = np.zeros_like(fitted["weights"])
                    if method == "prefix_mean30" and st["n"] >= 50:
                        expected_weights[-1] = st["rhs"][-1]
                    if not np.allclose(fitted["weights"], expected_weights, atol=1e-12, rtol=1e-12):
                        raise ValueError(
                            "Constant control used different labels or nonzero feature weights"
                        )
                else:
                    optimum = validate_weights(
                        fitted["weights"],
                        st,
                        prior=None if daily else prior,
                        gamma=config["adapt_gamma"],
                        daily_penalty=penalty,
                        free_bias=bool(bias),
                    )
                    if optimum["normal_equation_relative_residual"] is not None:
                        residuals.append(optimum["normal_equation_relative_residual"])
                fallbacks += int(metadata["fallback"])
                errors.append(error)
                checked.append({**row, "metrics": metrics})
            print(f"independent assessment {domain} {sid} checked", flush=True)
    by_key = {(r["domain"], r["session_id"], r["method"]): r for r in checked}
    checked = [by_key[r["domain"], r["session_id"], r["method"]] for r in assessment["cases"]]
    summary = reconstruct_summary(checked, plan)
    compare_structure(assessment["summary"], summary, "independent assessment summary")
    return {
        "status": "PASS",
        "stage": "assessment",
        "created_at": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": sha256(__file__),
        "assessment_sha256": sha256(assessment_path),
        "selection_sha256": sha256(selection_path),
        "source_audit_sha256": sha256(source_audit_path),
        "plan_sha256": PLAN_SHA256,
        "development_index_sha256": sha256(development.path),
        "indices": assessment["indices"],
        "cases_checked": len(checked),
        "models_and_full_traces_checked": len(errors),
        "unavailable_cases_checked": unavailable,
        "insufficient_label_fallbacks_checked": fallbacks,
        "max_prediction_absolute_error": max(errors, default=0),
        "max_normal_equation_relative_residual": max(residuals, default=0),
        "summary": summary,
        "independent_scores": checked,
        "scope": (
            "Independent saved-model optimum, recurrence replay, prefix/history boundaries, "
            "score arithmetic, case inventory, archive-order receipts, paired summaries and "
            "fixed gates. No V4 implementation imports."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selection", type=Path)
    group.add_argument("--assessment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an audit receipt")
    result = (
        audit_selection(args.selection) if args.selection else audit_assessment(args.assessment)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
