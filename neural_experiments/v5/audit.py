"""Independent V5 fit, trace, prefix-decision and selection reconstruction.

No V5 producer module is imported. Existing independent NumPy ridge/recurrence
and sklearn score references are reused. The policy author also wrote this
separate implementation; this is arithmetic verification, not an external lab.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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
    verify,
)
from neural_experiments.v4.audit import (
    Inputs as PreviousInputs,
)
from neural_experiments.v4.audit import (
    reference_design,
    reference_history,
    reference_mask,
    reference_representation,
    reference_statistics,
    replay,
    selected_histories,
    validate_normalization,
    validate_weights,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = "87c417f915259e7f77b7999217d3a2461884fffd175c676188e20c346472390e"
BUDGETS = (30, 60, 120, 240)
FAMILIES = ("tx45", "tx45_sbp")
METHODS = ("zero", "history0", "history30", "daily120", "daily240", "adaptive")
SOURCE_FILES = {
    f"neural_experiments/v5/{name}.py" for name in ("model", "policy", "study", "data", "metrics")
} | {
    "neural_experiments/v4/model.py",
    "neural_experiments/v4/study.py",
    "neural_experiments/v4/data.py",
    "neural_experiments/v3/study.py",
    "neural_experiments/v2/data.py",
}


def grids():
    candidates = []
    for family, tau in zip(FAMILIES, (0.4, 0.2), strict=True):
        for head in ("prior10", "scratch1", "scratch10_zero"):
            candidates.append(
                {
                    "candidate_id": family + "_" + head,
                    "head": head,
                    "config": {
                        "family": family,
                        "tau_seconds": tau,
                        "baseline_tau_seconds": None,
                        "adapt_gamma": 10.0,
                    },
                }
            )
    policies = [
        {"policy_id": f"p{i}", "history_mean_mse_ceiling": h, "validation_margin": m}
        for i, (h, m) in enumerate(itertools.product((0.4, 0.45), (0.025, 0.05)))
    ]
    return candidates, policies


def unique(rows, key, expected=None):
    result = {key(row): row for row in rows}
    if len(result) != len(rows) or expected is not None and set(result) != set(expected):
        raise ValueError("Duplicate, missing or unexpected artifact inventory")
    return result


class Inputs(PreviousInputs):
    """Reuse strict independent array validation, with V5 role and ancestry binding."""

    def __init__(self, path, role, plan):
        self.path, self.index, self.checked = Path(path), read(path), set()
        index = self.index
        if (
            index["schema_version"] != 1
            or index["role"] != role
            or index["acquisition_plan_sha256"] != plan["acquisition_plan_sha256"]
            or index["provider"] != {"doi": "10.5061/dryad.x0k6djj1h", "version_id": 462702}
        ):
            raise ValueError("Input role/provider differs from locked V5 protocol")
        self.items = unique(index["sessions"], lambda s: s["session_id"])
        for item in self.items.values():
            if (
                item["role"] != role
                or item["bin_seconds"] != 0.02
                or item["n_bins"] <= 0
                or role != "t3_blocks"
                and item["n_bins"] > 30000
            ):
                raise ValueError("Invalid role or authorized recording extent")
            segment_ranges(item["segments"], item["n_bins"])
        if role == "development":
            verify(path, plan["development_index_sha256"])
            if len(self.items) != 104 or set(index["participants"]) != set(plan["participants"]):
                raise ValueError("Development cohort changed")
            parents = []
            for key, pin in (
                (
                    "v4_development",
                    "aa6bf303c746edde0fb4d0a3a937a0fb07afb849270f2c79093fc021a9c2c500",
                ),
                ("v4_new", "76293562fd6e586e7d6f5506c41ebda97241c971e18984c22489b69ac29558cc"),
            ):
                if index[key + "_index_sha256"] != pin:
                    raise ValueError("Unrecognized exposed parent")
                verify(index[key + "_index_path"], pin)
                parents.extend(read(index[key + "_index_path"])["sessions"])
            by_id = unique(parents, lambda s: s["session_id"], self.items)
            for sid, item in self.items.items():
                for field in (
                    "participant_id",
                    "post_implant_day",
                    "source",
                    "feature_sha256",
                    "target_sha256",
                    "n_bins",
                    "n_channels",
                    "segments",
                ):
                    compare_structure(item[field], by_id[sid][field], "exposed parent " + field)
        elif index["study_lock"]["sha256"] != PLAN_SHA:
            raise ValueError("Fresh input not bound to prespecified plan")


def check_identity(selection, root):
    if (
        selection["schema_version"] != 1
        or selection["stage"] != "source"
        or selection["status"] != "complete"
        or selection["fresh_labels_opened"] is not False
    ):
        raise ValueError("Source selection is incomplete or fresh outcomes were opened")
    plan_path = contained(root, selection["plan"]["path"])
    verify(plan_path, PLAN_SHA)
    if selection["plan"]["sha256"] != PLAN_SHA or set(selection["code_sha256"]) != SOURCE_FILES:
        raise ValueError("Plan/executed-source inventory mismatch")
    for path, digest in selection["code_sha256"].items():
        verify(contained(root, path), digest)
    path = contained(root, selection["development_index"]["path"])
    verify(path, selection["development_index"]["sha256"])
    plan = read(plan_path)
    return plan, Inputs(path, "development", plan)


def prior_for(query, items, layouts, stats, dimension):
    receipt, usable = [], []
    for old in selected_histories(query, items, layouts):
        values = stats[old["session_id"]]
        good = values["n"] >= 50
        receipt.append(
            {
                "session_id": old["session_id"],
                "post_implant_day": old["post_implant_day"],
                "training_rows": values["n"],
                "disposition": "used" if good else "insufficient_labels",
            }
        )
        if good:
            usable.append(values)
    return reference_history(usable, dimension), receipt


def label_mask(mask, segments, stop):
    value = reference_mask(mask, segments, min(stop, len(mask)))
    value[:1500] = False
    return value


def sectors(y, mask):
    # The label broker exposes float64; preserve that boundary arithmetic exactly.
    selected = np.asarray(y[mask], dtype=np.float64)
    angle = np.arctan2(selected[:, 1], selected[:, 0])
    index = (np.floor((angle + np.pi) / (np.pi / 4)).astype(int)) % 8
    return np.bincount(index, minlength=8).tolist()


def stage_reference(y, mask, segments, budget, predictions):
    """Separate grouped validation calculation, with no producer helper calls."""
    stop, cut = 1500 + 50 * budget, 1500 + 25 * budget
    if budget not in BUDGETS or len(y) != stop or mask.shape != (stop,):
        raise ValueError("Only the requested label prefix may enter stage reconstruction")
    if set(predictions) != {c["candidate_id"] for c in grids()[0]}:
        raise ValueError("Missing validation candidate")
    if any(p.shape != (stop, 2) for p in predictions.values()):
        raise ValueError("Future/full-session prediction supplied to prefix evidence")
    ranges = [
        (a, min(b, stop))
        for a, b in segment_ranges(
            segments, segments[-1]["end"] if isinstance(segments[-1], dict) else segments[-1][1]
        )
        if a < stop
    ]
    clean = mask.copy()
    for a, b in ranges:
        clean[a : min(a + 62, b)] = False
    training = clean.copy()
    training[:1500], training[cut:] = False, False
    eligible = clean.copy()
    eligible[: cut + 100] = False
    retained = np.zeros(stop, bool)
    mean = y[training].astype(float).mean(0) if training.sum() >= 50 else np.zeros(2)
    groups = []
    for a in range(cut + 100, stop - 99, 100):
        b = a + 100
        if not any(left <= a and b <= right for left, right in ranges):
            continue
        selected = eligible[a:b]
        n = int(selected.sum())
        if n < 25:
            continue
        retained[a:b] = selected
        yy = np.asarray(y[a:b][selected], float)
        groups.append(
            {
                "start": a,
                "stop": b,
                "n": n,
                "zero_mse": float(np.mean(yy**2)),
                "prefix_mean_mse": float(np.mean((yy - mean) ** 2)),
                "candidate_mse": {
                    k: float(np.mean((yy - p[a:b][selected]) ** 2)) for k, p in predictions.items()
                },
            }
        )
    return {
        "budget_seconds": budget,
        "train_stop": cut,
        "validation_start": cut + 100,
        "stop": stop,
        "training_rows": int(training.sum()),
        "training_sector_counts": sectors(y, training),
        "validation_eligible_rows": int(eligible.sum()),
        "validation_rows": int(retained.sum()),
        "validation_sector_counts": sectors(y, retained),
        "training_mask_sha256": mask_hash(training),
        "validation_mask_sha256": mask_hash(retained),
        "groups": groups,
    }


def decision_reference(history, stages, config):
    """Reconstruct operational stopping using only history and available prefix evidence."""
    events, ready, requested = [], [], []
    for entry in history["prior_candidates"]:
        values = [p["mse"] for p in entry["probes"]]
        avg = sum(values) / len(values) if values else None
        worst = max(values) if values else None
        passed = bool(
            history["warmup_available"]
            and entry["prior_available"]
            and len(values) >= 2
            and avg <= config["history_mean_mse_ceiling"]
            and worst <= 0.5
        )
        events.append(
            {
                "budget_seconds": 0,
                "candidate_id": entry["candidate_id"],
                "n_probes": len(values),
                "mean_mse": avg,
                "worst_mse": worst,
                "passed": passed,
            }
        )
        if passed:
            ready.append((avg, entry["candidate_id"]))

    def result(action, candidate, budget, reason, available=None):
        available = budget if available is None else available
        return {
            "action": action,
            "candidate_id": candidate,
            "budget_seconds": budget,
            "requested_label_seconds": budget,
            "available_label_seconds": available,
            "decision_time_seconds": 30 + available,
            "reason": reason,
            "events": events,
        }, requested

    if ready:
        return result(
            "accept", min(ready, key=lambda x: x[0])[1], 0, "historical_reliability_passed"
        )
    if not history["warmup_available"]:
        return result("abstain", "zero", 0, "unavailable_30s_warmup", 0)
    for budget in BUDGETS:
        requested.append(budget)
        stage = stages[budget]
        if "unavailable" in stage:
            events.append(stage)
            return result(
                "abstain",
                "zero",
                budget,
                "checkpoint_unavailable",
                stage["available_label_seconds"],
            )
        groups, total = stage["groups"], stage["validation_rows"]
        geometry = (
            stage["training_rows"] >= 100
            and total >= 100
            and len(groups) >= 3
            and sum(n >= 10 for n in stage["training_sector_counts"]) >= 4
            and sum(n >= 10 for n in stage["validation_sector_counts"]) >= 4
        )
        null = {
            k: sum(g["n"] * g[k] for g in groups) / total if total else None
            for k in ("zero_mse", "prefix_mean_mse")
        }
        checked = []
        for candidate in grids()[0]:
            cid = candidate["candidate_id"]
            loss = sum(g["n"] * g["candidate_mse"][cid] for g in groups) / total if total else None
            wins = sum(
                g["candidate_mse"][cid] < min(g["zero_mse"], g["prefix_mean_mse"]) for g in groups
            )
            passed = bool(
                geometry
                and loss <= min(null.values()) - config["validation_margin"]
                and wins >= math.ceil(2 * len(groups) / 3)
            )
            checked.append(
                {"candidate_id": cid, "mse": loss, "winning_groups": wins, "passed": passed}
            )
        good = [c for c in checked if c["passed"]]
        chosen = min(good, key=lambda c: c["mse"])["candidate_id"] if good else None
        events.append(
            {
                "budget_seconds": budget,
                "geometry_passed": bool(geometry),
                "training_rows": stage["training_rows"],
                "validation_rows": total,
                "training_sector_counts": stage["training_sector_counts"],
                "validation_sector_counts": stage["validation_sector_counts"],
                "validation_groups": len(groups),
                "null_mse": null,
                "candidates": checked,
                "selected_candidate": chosen,
            }
        )
        if chosen is not None:
            return result("accept", chosen, budget, "held_out_prefix_evidence_passed")
    return result("abstain", "zero", 240, "no_candidate_passed_at_cap")


def selection_reference(rows, people):
    choices = []
    for target in people:
        ranking = []
        for policy in grids()[1]:
            people_scores, people_costs, people_daily = [], [], []
            for person in people:
                if person == target:
                    continue
                allrows = [
                    r
                    for r in rows
                    if r["participant"] == person and r["policy_id"] == policy["policy_id"]
                ]
                valid = [r for r in allrows if r["metrics"]["r2"] is not None]
                if valid:
                    people_scores.append(np.mean([r["metrics"]["r2"] for r in valid]))
                    people_costs.append(np.mean([r["label_seconds"] for r in allrows]))
                    people_daily.append(np.mean([r["daily120_r2"] for r in valid]))
            mean, cost, daily = map(
                float, map(np.mean, (people_scores, people_costs, people_daily))
            )
            ranking.append(
                {
                    "policy_id": policy["policy_id"],
                    "mean_r2": mean,
                    "mean_label_seconds": cost,
                    "daily120_mean_r2": daily,
                    "feasible": mean >= daily - 0.03,
                    "objective": mean - 0.05 * cost / 240,
                    "eligible_source_people": len(people_scores),
                }
            )
        feasible = [r for r in ranking if r["feasible"]]
        best = (
            min(feasible, key=lambda r: (r["mean_label_seconds"], -r["mean_r2"]))
            if feasible
            else max(ranking, key=lambda r: r["objective"])
        )
        choices.append(
            {
                "target_participant": target,
                "policy_id": best["policy_id"],
                "selection_rule": "minimum_cost_within_0.03_daily120"
                if feasible
                else "fallback_r2_minus_0.05_cost_fraction",
                "ranking": ranking,
            }
        )
    return choices


def fit_key(row):
    return (
        row["session_id"],
        row["candidate_id"],
        row["budget_seconds"],
        row["phase"],
        row["daily_control"],
    )


class Verifier:
    def __init__(self, inputs, root):
        self.inputs, self.root = inputs, root
        self.items = list(inputs.items.values())
        self.layouts, self.stats = {}, {f: {} for f in FAMILIES}
        self.errors, self.residuals = [], []
        self.models_checked, self.traces_checked, self.fits_checked = 0, 0, 0
        for item in self.items:
            features, target = inputs.arrays(item)
            self.layouts[item["session_id"]] = tuple(features["channel_ids"])
            for candidate in grids()[0][::3]:
                family = candidate["config"]["family"]
                raw = reference_representation(features, item["segments"], candidate["config"])
                z, _, _ = reference_design(raw, len(raw), item["n_channels"])
                mask = reference_mask(target["eval_mask"], item["segments"], stride=5)
                self.stats[family][item["session_id"]] = reference_statistics(z, target["y"], mask)
        print("independent historical sufficient statistics complete", flush=True)

    def query(self, inputs, item):
        features, targets = inputs.arrays(item)
        self.layouts[item["session_id"]] = tuple(features["channel_ids"])
        raw, prior, history = {}, {}, {}
        for candidate in grids()[0][::3]:
            family = candidate["config"]["family"]
            raw[family] = reference_representation(features, item["segments"], candidate["config"])
            prior[family], history[family] = prior_for(
                item, self.items, self.layouts, self.stats[family], raw[family].shape[1] + 1
            )
        return raw, prior, history, targets

    def model(self, row, item, raw, config, prefix):
        path = contained(self.root, row["model_path"])
        verify(path, row["model_sha256"])
        arrays = numeric(path)
        z = validate_normalization(arrays, raw, prefix, item["n_channels"], config)
        if not np.isfinite(arrays["weights"]).all():
            raise ValueError("Nonfinite model weights")
        self.models_checked += 1
        return arrays, z

    def fit(self, row, item, raw, prior, history, labels):
        candidate = next(c for c in grids()[0] if c["candidate_id"] == row["candidate_id"])
        family, head = candidate["config"]["family"], candidate["head"]
        budget, phase, control = row["budget_seconds"], row["phase"], row["daily_control"]
        if (
            budget not in BUDGETS
            or phase not in ("validation", "full")
            or control
            and (
                phase != "full"
                or budget not in (120, 240)
                or candidate["candidate_id"] != "tx45_sbp_scratch1"
            )
        ):
            raise ValueError("Unplanned fitted model role")
        requested = 1500 + budget * (25 if phase == "validation" else 50)
        stop = min(requested, item["n_bins"], len(labels["y"]))
        prefix = stop if control else 1500
        arrays, z = self.model(row, item, raw[family], candidate["config"], prefix)
        # Continuation labels are padded; only the known original prefix is eligible.
        mask = np.zeros(item["n_bins"], bool)
        y = np.zeros((item["n_bins"], 2), float)
        mask[: len(labels["eval_mask"])] = labels["eval_mask"]
        y[: len(labels["y"])] = labels["y"]
        chosen = label_mask(mask, item["segments"], requested)
        stats = reference_statistics(z[: len(chosen)], y[: len(chosen)], chosen)
        expected = {
            "candidate_id": candidate["candidate_id"],
            "head": head,
            "normalization_bins": prefix,
            "label_start_bin": 1500,
            "label_stop_bin": len(chosen),
            "requested_label_stop_bin": requested,
            "observed_label_stop_bin": len(chosen),
            "label_seconds": (requested - 1500) * 0.02,
            "available_label_seconds": (len(chosen) - 1500) * 0.02,
            "complete_requested_interval": len(chosen) == requested,
            "training_rows": stats["n"],
            "training_mask_sha256": mask_hash(chosen),
            "training_mask_length": len(chosen),
            "fallback": stats["n"] < 50,
            "fallback_output": "prior" if head == "prior10" else "zero",
            "prior_centered": head == "prior10",
            "feature_penalty": 1.0 if head == "scratch1" else 10.0,
            "intercept_penalty": 1.0 if head == "prior10" else None,
            "free_intercept": head == "scratch1",
            "constrained_zero_intercept": head == "scratch10_zero",
        }
        compare_structure(row["fit"], expected, "fit label boundaries/normalizer")
        compare_structure(row["history"], history[family], "strict chronological history")
        if (
            row["family"] != family
            or row["participant"] != item["participant_id"]
            or row["post_implant_day"] != item["post_implant_day"]
        ):
            raise ValueError("Fit identity mismatch")
        residual = validate_weights(
            arrays["weights"],
            stats,
            prior=prior[family] if head == "prior10" else None,
            gamma=10.0,
            daily_penalty=1.0 if head == "scratch1" else 10.0,
            free_bias=head == "scratch1",
        )["normal_equation_relative_residual"]
        if residual is not None:
            self.residuals.append(residual)
        self.fits_checked += 1
        return arrays

    def score(self, row, item, raw, targets, expected_weights, config, prefix, domain):
        arrays, _ = self.model(row, item, raw, config, prefix)
        if not np.allclose(arrays["weights"], expected_weights, rtol=1e-10, atol=1e-10):
            raise ValueError("Scored model differs from its independently checked fit/prior")
        path = contained(self.root, row["trace_path"])
        verify(path, row["trace_sha256"])
        trace = numeric(path)
        mask = targets["eval_mask"].copy()
        if domain == "new_sessions":
            mask[:13600], mask[30000:] = False, False
        if (
            set(trace) != {"prediction", "eval_mask"}
            or trace["prediction"].shape != (item["n_bins"], 2)
            or trace["prediction"].dtype != np.float64
            or not np.isfinite(trace["prediction"]).all()
            or trace["eval_mask"].dtype != bool
            or not np.array_equal(trace["eval_mask"], mask)
            or row["eval_mask_sha256"] != mask_hash(mask)
        ):
            raise ValueError("Saved trace/scoring mask mismatch")
        predicted = replay(arrays, raw)
        error = float(np.max(np.abs(predicted - trace["prediction"])))
        if not np.allclose(predicted, trace["prediction"], atol=1e-9, rtol=1e-9):
            raise ValueError("Independent full-session recurrence/model replay failed")
        self.errors.append(error)
        metrics, _ = score_reference(targets["y"], trace["prediction"], mask)
        metrics["summed_mse"] = (
            float(np.mean(np.sum((targets["y"][mask] - trace["prediction"][mask]) ** 2, axis=1)))
            if mask.sum() >= 50
            else None
        )
        compare_structure(row["metrics"], metrics, "independent sklearn/angle/MSE")
        self.traces_checked += 1
        return metrics


def history_evidence(item, dev_items, layouts, histories, probes):
    priors = []
    for family in FAMILIES:
        eligible = []
        for old in selected_histories(item, dev_items, layouts):
            row = probes[old["session_id"], family]
            if row["metrics"]["summed_mse"] is not None and any(
                h["disposition"] == "used" for h in row["history"]
            ):
                eligible.append(
                    {
                        "session_id": old["session_id"],
                        "post_implant_day": old["post_implant_day"],
                        "n": row["metrics"]["n"],
                        "mse": row["metrics"]["summed_mse"] / 2,
                    }
                )
        priors.append(
            {
                "candidate_id": family + "_prior10",
                "prior_available": any(h["disposition"] == "used" for h in histories[family]),
                "probes": eligible,
            }
        )
    return {
        "current_session_id": item["session_id"],
        "current_post_implant_day": item["post_implant_day"],
        "warmup_available": item["n_bins"] >= 1500,
        "prior_candidates": priors,
    }


def check_case(
    verifier, item, raw, prior, history, targets, evidence, fits, rows, probes, source=True
):
    candidates, policies = grids()
    candidate_map = {c["candidate_id"]: c for c in candidates}
    identity = {
        "participant": item["participant_id"],
        "session_id": item["session_id"],
        "post_implant_day": item["post_implant_day"],
    }
    for row in rows + list(evidence.get("controls", [])):
        for key, value in identity.items():
            if row[key] != value:
                raise ValueError("Scored row identity mismatch")
    checked_fits = {
        fit_key(row): verifier.fit(row, item, raw, prior, history, targets) for row in fits
    }
    expected_history = history_evidence(item, verifier.items, verifier.layouts, history, probes)
    compare_structure(
        evidence["history_evidence"], expected_history, "chronological reliability probes"
    )
    stages = {}
    for stage in evidence["stages"]:
        budget = stage["budget_seconds"]
        if budget not in BUDGETS or budget in stages:
            raise ValueError("Duplicate or unplanned prefix stage")
        stop = 1500 + 50 * budget
        observed = min(len(targets["y"]), item["n_bins"])
        if observed < stop:
            expected = {
                "budget_seconds": budget,
                "unavailable": "requested_checkpoint_not_observed",
                "available_label_seconds": max(0, (observed - 1500) * 0.02),
            }
        else:
            yy, mask = targets["y"][:stop].copy(), targets["eval_mask"][:stop].copy()
            yy[:1500], mask[:1500] = 0, False
            predictions = {
                c["candidate_id"]: replay(
                    checked_fits[
                        (item["session_id"], c["candidate_id"], budget, "validation", False)
                    ],
                    raw[c["config"]["family"]],
                )[:stop]
                for c in candidates
            }
            expected = stage_reference(yy, mask, item["segments"], budget, predictions)
        compare_structure(stage, expected, "held-out prefix validation evidence")
        stages[budget] = expected
    expected_fits = {
        (item["session_id"], "tx45_sbp_prior10", 30, "full", False),
        (item["session_id"], "tx45_sbp_scratch1", 120, "full", True),
        (item["session_id"], "tx45_sbp_scratch1", 240, "full", True),
    }
    requested_union = set()
    for row in rows:
        cfg = next(p for p in policies if p["policy_id"] == row["policy_id"])
        decision, requested = decision_reference(expected_history, stages, cfg)
        compare_structure(row["decision"], decision, "independent sequential decision")
        requested_union.update(requested)
        if not source:
            compare_structure(row["label_requests"], requested, "actual lazy label requests")
        budget, cid = decision["budget_seconds"], decision["candidate_id"]
        for key, value in {
            "method": "adaptive",
            "action": decision["action"],
            "label_seconds": budget,
            "available_label_seconds": max(0, min(budget, item["n_bins"] * 0.02 - 30)),
            "candidate_id": cid,
        }.items():
            compare_structure(row[key], value, "policy row " + key)
        family = "tx45_sbp" if cid == "zero" else candidate_map[cid]["config"]["family"]
        config = next(c["config"] for c in candidates if c["config"]["family"] == family)
        if cid == "zero":
            weights = np.zeros_like(prior[family])
        elif budget == 0:
            weights = prior[family]
        else:
            key = item["session_id"], cid, budget, "full", False
            expected_fits.add(key)
            weights = checked_fits[key]["weights"]
        verifier.score(row, item, raw[family], targets, weights, config, 1500, "new_sessions")
    if set(stages) != requested_union:
        raise ValueError("Unused future-stage evidence or missing requested stage")
    for budget in requested_union:
        if "unavailable" not in stages[budget]:
            expected_fits.update(
                (item["session_id"], c["candidate_id"], budget, "validation", False)
                for c in candidates
            )
    if set(checked_fits) != expected_fits:
        raise ValueError("Unexpected/unbound fit inventory")
    controls = unique(evidence["controls"], lambda r: r["method"], METHODS[:-1])
    for name, row in controls.items():
        budget = {"zero": 0, "history0": 0, "history30": 30, "daily120": 120, "daily240": 240}[name]
        prefix = min(1500 + 50 * budget, item["n_bins"]) if name.startswith("daily") else 1500
        if name == "zero":
            weights = np.zeros_like(prior["tx45_sbp"])
        elif name == "history0":
            weights = prior["tx45_sbp"]
        else:
            key = (
                item["session_id"],
                "tx45_sbp_prior10" if name == "history30" else "tx45_sbp_scratch1",
                budget,
                "full",
                name.startswith("daily"),
            )
            weights = checked_fits[key]["weights"]
        for key, value in {
            "label_seconds": budget,
            "available_label_seconds": max(0, min(budget, item["n_bins"] * 0.02 - 30)),
            "action": "abstain" if name == "zero" else "accept",
            "candidate_id": None,
        }.items():
            compare_structure(row[key], value, "control row " + key)
        verifier.score(
            row,
            item,
            raw["tx45_sbp"],
            targets,
            weights,
            candidates[3]["config"],
            prefix,
            "new_sessions",
        )
    for row in rows:
        compare_structure(
            row["daily120_r2"], controls["daily120"]["metrics"]["r2"], "matched source comparator"
        )


def audit_selection(path, root=ROOT):
    selection = read(path)
    plan, inputs = check_identity(selection, root)
    candidates, policies = grids()
    compare_structure(selection["candidates"], candidates, "candidate inventory")
    compare_structure(selection["policies"], policies, "policy grid")
    rows = unique(
        selection["cases"],
        lambda r: (r["session_id"], r["policy_id"]),
        itertools.product(inputs.items, [p["policy_id"] for p in policies]),
    )
    evidence = unique(selection["evidence"], lambda r: r["session_id"], inputs.items)
    fits = unique(selection["fits"], fit_key)
    if any(k[0] not in inputs.items for k in fits):
        raise ValueError("Fit refers to a session outside the exposed source cohort")
    probes = unique(
        selection["history_probes"],
        lambda r: (r["session_id"], r["family"]),
        itertools.product(inputs.items, FAMILIES),
    )
    verifier = Verifier(inputs, root)
    for item in verifier.items:
        raw, prior, history, labels = verifier.query(inputs, item)
        for family in FAMILIES:
            row = probes[item["session_id"], family]
            for key, value in {
                "participant": item["participant_id"],
                "post_implant_day": item["post_implant_day"],
                "phase": "history_probe",
                "normalization_bins": 1500,
                "history": history[family],
            }.items():
                compare_structure(row[key], value, "history probe " + key)
            config = next(c["config"] for c in candidates if c["config"]["family"] == family)
            verifier.score(
                row, item, raw[family], labels, prior[family], config, 1500, "new_sessions"
            )
    print("all chronological held-out history probes verified", flush=True)
    for item in verifier.items:
        sid = item["session_id"]
        raw, prior, history, labels = verifier.query(inputs, item)
        check_case(
            verifier,
            item,
            raw,
            prior,
            history,
            labels,
            evidence[sid],
            [r for k, r in fits.items() if k[0] == sid],
            [rows[sid, p["policy_id"]] for p in policies],
            probes,
        )
        print(f"independent prefix fits, decisions and traces checked: {sid}", flush=True)
    compare_structure(
        selection["choices"],
        selection_reference(selection["cases"], plan["participants"]),
        "whole-person-excluded policy ranking",
    )
    return {
        "status": "PASS",
        "stage": "source",
        "created_at": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": sha256(__file__),
        "selection_sha256": sha256(path),
        "plan_sha256": PLAN_SHA,
        "development_index_sha256": sha256(inputs.path),
        "code_sha256": selection["code_sha256"],
        "source_cases_checked": len(rows),
        "history_probes_checked": len(probes),
        "fits_checked": verifier.fits_checked,
        "models_checked": verifier.models_checked,
        "full_traces_checked": verifier.traces_checked,
        "source_choices_checked": len(selection["choices"]),
        "max_prediction_absolute_error": max(verifier.errors, default=0),
        "max_normal_equation_relative_residual": max(verifier.residuals, default=0),
        "scope": (
            "No V5 producer imports. Independent NumPy model optimality, full feature-only "
            "recurrence replay, sklearn scores, grouped prefix evidence, sequential routing, "
            "history chronology and whole-person-excluded selection. Policy and verifier share "
            "an author; this is separate-code arithmetic verification, not external replication."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous audit attempts; output must be fresh")
    receipt = audit_selection(args.selection)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
