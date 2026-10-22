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
    verifier,
    item,
    raw,
    prior,
    history,
    targets,
    evidence,
    fits,
    rows,
    probes,
    source=True,
    calibration_labels=None,
    domain="new_sessions",
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
    labels = targets if calibration_labels is None else calibration_labels
    checked_fits = {
        fit_key(row): verifier.fit(row, item, raw, prior, history, labels) for row in fits
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
        observed = min(len(labels["y"]), item["n_bins"])
        if observed < stop:
            expected = {
                "budget_seconds": budget,
                "unavailable": "requested_checkpoint_not_observed",
                "available_label_seconds": max(0, (observed - 1500) * 0.02),
            }
        else:
            yy, mask = labels["y"][:stop].copy(), labels["eval_mask"][:stop].copy()
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
            "domain": domain,
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
        verifier.score(row, item, raw[family], targets, weights, config, 1500, domain)
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
            "domain": domain,
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
            domain,
        )
    if source:
        for row in rows:
            compare_structure(
                row["daily120_r2"],
                controls["daily120"]["metrics"]["r2"],
                "matched source comparator",
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


def validate_fresh_provenance(inputs, development, plan):
    """Bind ranks13–15/prior12 and the third/fourth eligible original T3 blocks."""
    index, role = inputs.index, inputs.index["role"]
    people = plan["new_session_participants"] if role == "new_sessions" else ["T3"]
    if index["participants"] != people:
        raise ValueError("Fresh participant inventory changed")
    for key in ("v4_development_index_sha256", "v4_new_index_sha256", "v4_t3_index_sha256"):
        if index[key] != development.index[key]:
            raise ValueError("Fresh input ancestry changed")
    expected, missing = {}, []
    if role == "new_sessions":
        refs = unique(
            index["acquisition_sources"],
            lambda r: r["path"],
            [f"acquisition/{p}.json" for p in people],
        )
        for person in people:
            ref = refs[f"acquisition/{person}.json"]
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
                or len(prior) != 12
                or receipt["selection"]["regular_session_ranks_one_based"] != [13, 14, 15]
                or receipt["selection"]["scores_or_duration_used_for_selection"] is not False
            ):
                raise ValueError("Fresh archive acquisition policy changed")
            for key in ("v4_development_index_sha256", "v4_new_index_sha256"):
                if receipt[key] != development.index[key]:
                    raise ValueError("Acquisition authenticated the wrong exposed prefix")
            archive = receipt["archive"]
            if (
                archive["file_id"] != prior[0]["archive_file_id"]
                or archive["provider_sha256"] != prior[0]["archive_sha256_published"]
            ):
                raise ValueError("Provider archive identity changed")
            pattern = rf"(?:\./)?decoding/{person}/{person}_day_(\d+)_decoding\.mat"
            seen = [
                s
                for s in receipt["seen_members"]
                if s["regular_file"] and re.fullmatch(pattern, s["name"])
            ]
            authenticated, selected = receipt["verified_prior_members"], receipt["selected"]
            if (
                len(authenticated) != 12
                or len(seen) != 12 + len(selected)
                or len(selected) > 3
                or len(selected) < 3
                and not receipt["tar_end_observed"]
                or receipt["missing_requested_ranks"] != list(range(13 + len(selected), 16))
            ):
                raise ValueError("Archive walk or shortage inventory mismatch")
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
                    raise ValueError("Selected source member does not match archive ordinal")
                if rank <= 12:
                    for key in (
                        "member_name",
                        "member_size",
                        "member_sha256",
                        "archive_ordinal",
                        "session_id",
                    ):
                        if member[key] != prior[rank - 1][key]:
                            raise ValueError("Prior twelve authenticated members changed")
                else:
                    expected[member["session_id"]] = {
                        **member,
                        "archive_file_id": archive["file_id"],
                        "archive_sha256_published": archive["provider_sha256"],
                        "full_archive_sha256_verified": False,
                    }
            if receipt["missing_requested_ranks"]:
                missing.append(
                    {
                        "participant_id": person,
                        "missing_requested_ranks": receipt["missing_requested_ranks"],
                    }
                )
        if missing:
            raise ValueError("Unobserved fixed new-session ranks require explicit audit review")
        if index["participant_dispositions"]:
            raise ValueError("Unexpected new-session disposition")
    else:
        if (
            index["development_index_sha256"] != sha256(development.path)
            or Path(index["development_index_path"]).resolve() != development.path.resolve()
        ):
            raise ValueError("T3 known-prefix cohort changed")
        parent = read(index["v4_t3_index_path"])
        verify(index["v4_t3_index_path"], development.index["v4_t3_index_sha256"])
        parent_items = unique(parent["sessions"], lambda s: s["session_id"])
        from scipy.io import loadmat

        for sid, old in development.items.items():
            if old["participant_id"] != "T3":
                continue
            rawpath = contained(old["raw_source_root"], old["source"]["raw_path"])
            verify(rawpath, old["source"]["member_sha256"])
            blocks = loadmat(rawpath, simplify_cells=True, variable_names=["blocks"])["blocks"]
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
                raise ValueError("Malformed original acquisition metadata")
            lengths = ((ends - starts) // 2).astype(np.int64)
            offsets = np.r_[0, np.cumsum(lengths)]
            eligible = [i for i in range(len(ids)) if offsets[i] >= 30000 and lengths[i] > 0]
            if [int(ids[i]) for i in eligible[:2]] != parent_items[sid][
                "selected_evaluation_block_ids"
            ]:
                raise ValueError("Previously exposed V4 continuation blocks changed")
            chosen = eligible[2:4]
            if not chosen:
                missing.append(
                    {
                        "participant_id": "T3",
                        "session_id": sid,
                        "reason": "no_remaining_complete_block_after_V4_selection",
                        "requested_continuation_blocks": 2,
                        "available_continuation_blocks": 0,
                    }
                )
                continue
            expected[sid] = old["source"]
            item = inputs.items[sid]
            fields = ("start", "end", "block_id", "original_block_index", "selected_for_evaluation")
            geometry = [
                {
                    "start": int(offsets[i]),
                    "end": int(offsets[i + 1]),
                    "block_id": int(ids[i]),
                    "original_block_index": i,
                    "selected_for_evaluation": i in chosen,
                }
                for i in range(chosen[-1] + 1)
            ]
            compare_structure(
                [{k: s[k] for k in fields} for s in item["segments"]],
                geometry,
                "fresh original T3 blocks after prior selections",
            )
            if (
                item["n_bins"] != int(offsets[chosen[-1] + 1])
                or item["selected_evaluation_block_ids"] != [int(ids[i]) for i in chosen]
                or item["unavailable_continuation_blocks"] != 2 - len(chosen)
            ):
                raise ValueError("T3 block endpoint/shortage changed")
            known = contained(development.path.parent, old["target_path"])
            if (
                Path(item["known_prefix_target_path"]).resolve() != known
                or item["known_prefix_target_sha256"] != old["target_sha256"]
                or item["known_prefix_n_bins"] != old["n_bins"]
            ):
                raise ValueError("Continuation calibration includes unauthorized labels")
        compare_structure(
            index["participant_dispositions"], missing, "unavailable original T3 sessions"
        )
    if set(expected) != set(inputs.items):
        raise ValueError("Fresh session selected inventory changed")
    for sid, item in inputs.items.items():
        compare_structure(item["source"], expected[sid], "receipt-bound prepared source")
        if item["post_implant_day"] != expected[sid]["post_implant_day"]:
            raise ValueError("Assessment chronology changed")
        if role == "new_sessions":
            verify(
                contained(item["raw_source_root"], item["source"]["raw_path"]),
                item["source"]["member_sha256"],
            )
            if item["n_bins"] != min(item["full_prepared_n_bins"], 30000):
                raise ValueError("New session cap/truncation differs from fixed selection")


def interval(values):
    if not values:
        return [None, None]
    rng = np.random.default_rng(20261024)
    draws = rng.integers(len(values), size=(10000, len(values)))
    means = np.asarray(values, float)[draws].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def average(values):
    return float(np.mean(values)) if values else None


def summary_reference(rows):
    """Equal-person accuracy/cost, identical-mask contrasts and fixed engineering gates."""
    domains = {}
    for domain in ("new_sessions", "t3_blocks"):
        allrows = [r for r in rows if r["domain"] == domain]
        unit_key = "participant" if domain == "new_sessions" else "session_id"
        methods, contrasts = {}, []
        for method in METHODS:
            requested = [r for r in allrows if r["method"] == method]
            scored = [r for r in requested if r["metrics"]["r2"] is not None]
            units = []
            for unit in sorted({r[unit_key] for r in scored}):
                group = [r for r in scored if r[unit_key] == unit]
                units.append(
                    {
                        "unit": unit,
                        "n_sessions": len(group),
                        "r2": average([r["metrics"]["r2"] for r in group]),
                        "angular_error_degrees": average(
                            [r["metrics"]["angular_error_degrees"] for r in group]
                        ),
                    }
                )
            costs = []
            for unit in sorted({r[unit_key] for r in requested}):
                group = [r for r in requested if r[unit_key] == unit]
                costs.append(
                    {
                        "unit": unit,
                        "label_seconds": average([r["label_seconds"] for r in group]),
                        "available_label_seconds": average(
                            [r["available_label_seconds"] for r in group]
                        ),
                    }
                )
            values = [u["r2"] for u in units]
            accepted = [r for r in scored if r["action"] != "abstain"]
            methods[method] = {
                "units": units,
                "n_units": len(units),
                "requested_sessions": len(requested),
                "scored_sessions": len(scored),
                "unavailable_sessions": len(requested) - len(scored),
                "mean_r2": average(values),
                "r2_bootstrap_95": interval(values),
                "positive_units": sum(v > 0 for v in values),
                "mean_angular_error_degrees": average([u["angular_error_degrees"] for u in units]),
                "cost_units": costs,
                "mean_requested_label_seconds": average([c["label_seconds"] for c in costs]),
                "mean_available_label_seconds": average(
                    [c["available_label_seconds"] for c in costs]
                ),
                "zero_label_sessions": sum(r["label_seconds"] == 0 for r in requested),
                "abstained_sessions": sum(r["action"] == "abstain" for r in requested),
                "accepted_scored_sessions": len(accepted),
                "acceptance_coverage": len(accepted) / len(scored) if scored else None,
                "negative_accepted_sessions": sum(r["metrics"]["r2"] < 0 for r in accepted),
            }
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
            gains = {}
            for sid in sorted(left.keys() & right.keys()):
                a, b = left[sid], right[sid]
                if a["eval_mask_sha256"] != b["eval_mask_sha256"]:
                    raise ValueError("Paired masks differ")
                gains.setdefault(a[unit_key], []).append(a["metrics"]["r2"] - b["metrics"]["r2"])
            values = [average(gains[k]) for k in sorted(gains)]
            contrasts.append(
                {
                    "arm": "adaptive",
                    "control": control,
                    "n_units": len(values),
                    "mean_gain": average(values),
                    "gain_bootstrap_95": interval(values),
                }
            )
        domains[domain] = {"methods": methods, "contrasts": contrasts}
    adaptive = domains["new_sessions"]["methods"]["adaptive"]
    comparisons = {c["control"]: c for c in domains["new_sessions"]["contrasts"]}
    useful = bool(
        adaptive["n_units"] == 8
        and adaptive["mean_r2"] >= 0.1
        and adaptive["r2_bootstrap_95"][0] > 0
        and adaptive["positive_units"] >= 6
        and adaptive["mean_angular_error_degrees"] <= 80
        and comparisons["zero"]["mean_gain"] >= 0.05
    )
    t2 = [
        r
        for r in rows
        if r["domain"] == "new_sessions"
        and r["participant"] == "T2"
        and r["metrics"]["r2"] is not None
    ]
    t2adaptive = [r for r in t2 if r["method"] == "adaptive"]
    history = {r["session_id"]: r["metrics"]["r2"] for r in t2 if r["method"] == "history0"}
    gains = [r["metrics"]["r2"] - history[r["session_id"]] for r in t2adaptive]
    mean = average([r["metrics"]["r2"] for r in t2adaptive])
    zero = average([r["metrics"]["r2"] for r in t2 if r["method"] == "zero"])
    positive = sum(r["metrics"]["r2"] > 0 and r["action"] != "abstain" for r in t2adaptive)
    ci = interval(gains)
    return {
        "domains": domains,
        "T2": {
            "mean_r2": mean,
            "positive_accepted_sessions": positive,
            "gain_over_history0": average(gains),
            "gain_bootstrap_95": ci,
        },
        "gates": {
            "adaptive_useful": useful,
            "reduced_calibration_noninferiority": bool(
                useful
                and comparisons["daily120"]["gain_bootstrap_95"][0] > -0.05
                and adaptive["mean_requested_label_seconds"] <= 90
            ),
            "T2_recovered": bool(
                len(t2adaptive) == 3
                and mean >= 0.05
                and positive >= 2
                and mean - zero >= 0.05
                and ci[0] > 0
            ),
        },
    }


def audit_assessment(path, root=ROOT):
    assessment = read(path)
    if (
        assessment["schema_version"] != 1
        or assessment["stage"] != "assessment"
        or assessment["status"] != "assessment_complete"
    ):
        raise ValueError("Assessment is incomplete")
    manifest_path = Path(path).with_name("manifest.json")
    compare_structure(read(manifest_path), assessment, "completed assessment manifest")
    selection_path = contained(root, assessment["selection_path"])
    verify(selection_path, assessment["selection_sha256"])
    selection = read(selection_path)
    plan, development = check_identity(selection, root)
    source_audit_path = contained(root, assessment["source_audit_path"])
    verify(source_audit_path, assessment["source_audit_sha256"])
    source_audit = read(source_audit_path)
    for field, expected in {
        "status": "PASS",
        "stage": "source",
        "selection_sha256": sha256(selection_path),
        "plan_sha256": PLAN_SHA,
        "development_index_sha256": sha256(development.path),
        "code_sha256": selection["code_sha256"],
    }.items():
        compare_structure(source_audit[field], expected, "bound passed source audit " + field)
    compare_structure(
        selection["choices"],
        selection_reference(selection["cases"], plan["participants"]),
        "locked source policy choices",
    )
    if set(assessment["indices"]) != {"new_sessions", "t3_blocks"}:
        raise ValueError("Separate assessment domains missing")
    inputs_by_domain = {}
    for domain, ref in assessment["indices"].items():
        indexpath = contained(root, ref["path"])
        verify(indexpath, ref["sha256"])
        inputs = Inputs(indexpath, domain, plan)
        validate_fresh_provenance(inputs, development, plan)
        inputs_by_domain[domain] = inputs
        compare_structure(
            assessment["data_dispositions"][domain],
            inputs.index["participant_dispositions"],
            "retained acquisition shortage",
        )
    expected = {
        (domain, sid, method)
        for domain, inputs in inputs_by_domain.items()
        for sid in inputs.items
        for method in METHODS
    }
    rows = unique(
        assessment["cases"], lambda r: (r["domain"], r["session_id"], r["method"]), expected
    )
    cases = {(domain, sid) for domain, inputs in inputs_by_domain.items() for sid in inputs.items}
    evidence = unique(assessment["evidence"], lambda r: (r["domain"], r["session_id"]), cases)
    fits = unique(assessment["fits"], fit_key)
    if any(k[0] not in {sid for _, sid in cases} for k in fits):
        raise ValueError("Assessment fit outside selected cases")
    probes = unique(selection["history_probes"], lambda r: (r["session_id"], r["family"]))
    verifier = Verifier(development, root)
    for domain, inputs in inputs_by_domain.items():
        for sid, item in inputs.items.items():
            raw, prior, history, targets = verifier.query(inputs, item)
            labels = targets
            if domain == "t3_blocks":
                verify(item["known_prefix_target_path"], item["known_prefix_target_sha256"])
                labels = numeric(item["known_prefix_target_path"])
                selected = np.zeros(item["n_bins"], bool)
                for seg in item["segments"]:
                    if seg["selected_for_evaluation"]:
                        selected[seg["start"] : seg["end"]] = True
                if np.any(targets["eval_mask"] & ~selected) or np.any(targets["y"][~selected] != 0):
                    raise ValueError("Continuation targets leaked outside fresh selected blocks")
            adaptive = rows[domain, sid, "adaptive"]
            expected_policy = next(
                c["policy_id"]
                for c in selection["choices"]
                if c["target_participant"] == item["participant_id"]
            )
            if adaptive["policy_id"] != expected_policy:
                raise ValueError("Assessment did not use frozen target-excluded policy")
            observed_evidence = {
                **evidence[domain, sid],
                "controls": [rows[domain, sid, method] for method in METHODS[:-1]],
            }
            check_case(
                verifier,
                item,
                raw,
                prior,
                history,
                targets,
                observed_evidence,
                [r for k, r in fits.items() if k[0] == sid],
                [adaptive],
                probes,
                source=False,
                calibration_labels=labels,
                domain=domain,
            )
            print(f"independent assessment decisions, fits and traces: {domain} {sid}", flush=True)
    summary = summary_reference(assessment["cases"])
    compare_structure(assessment["summary"], summary, "equal-person costs/accuracy/paired gates")
    return {
        "status": "PASS",
        "stage": "assessment",
        "created_at": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": sha256(__file__),
        "assessment_sha256": sha256(path),
        "manifest_sha256": sha256(manifest_path),
        "selection_sha256": sha256(selection_path),
        "source_audit_sha256": sha256(source_audit_path),
        "plan_sha256": PLAN_SHA,
        "development_index_sha256": sha256(development.path),
        "assessment_indices": {k: sha256(v.path) for k, v in inputs_by_domain.items()},
        "code_sha256": selection["code_sha256"],
        "cases_checked": len(rows),
        "fits_checked": verifier.fits_checked,
        "models_checked": verifier.models_checked,
        "full_traces_checked": verifier.traces_checked,
        "decisions_checked": len(evidence),
        "unavailable_scores": sum(r["metrics"]["r2"] is None for r in rows.values()),
        "acquisition_dispositions": assessment["data_dispositions"],
        "max_prediction_absolute_error": max(verifier.errors, default=0),
        "max_normal_equation_relative_residual": max(verifier.residuals, default=0),
        "independent_summary": summary,
        "scope": (
            "Separate NumPy/sklearn reference, no V5 producer imports; exact source selection, "
            "archive order, original T3 block geometry, calibration-only labels, model optimum, "
            "full traces, operational decisions and equal-person paired uncertainty. "
            "Policy/verifier share an author; not external replication."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--selection", type=Path)
    source.add_argument("--assessment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous audit attempts; output must be fresh")
    receipt = (
        audit_selection(args.selection) if args.selection else audit_assessment(args.assessment)
    )
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
