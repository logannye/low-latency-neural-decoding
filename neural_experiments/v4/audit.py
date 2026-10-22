"""Independent V4 protocol and numerical references.

This module does not import the V4 predictor, fitter, scorer or importer.
References use direct NumPy recurrences and ridge normal equations.
"""

from __future__ import annotations

import argparse
import itertools
import json
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
    if arrays["format_version"].shape != () or arrays["format_version"].item() != 4:
        raise ValueError("Unsupported model format")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an audit receipt")
    result = audit_selection(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
