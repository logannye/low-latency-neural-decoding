"""Separate-code V6 arithmetic and protocol verification.

No producer fitter, controller, metric or importer is imported. Independent
V3--V5 NumPy/sklearn references are reused; the policy and verifier share an
author, so this is arithmetic verification rather than external replication.
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
    read,
    segment_ranges,
    sha256,
    verify,
)
from neural_experiments.v3.audit import (
    numeric as numeric,
)
from neural_experiments.v4.audit import Inputs as ArrayInputs
from neural_experiments.v4.audit import (
    reference_statistics,
    replay,
    selected_histories,
    validate_weights,
)
from neural_experiments.v5 import audit as previous

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = "f8c075f99c51b6da1bfe07a57861746fff158ec2fd518ec21ecf45f10b170b7a"
DEVELOPMENT_SHA = "357f8b917abec17f6354dac3cd93b4d87d24df0db4bf60afeee8bdbefbf84678"
ACQUISITION_SHA = "0849f2188f13b0df565809db1302ae673af938d4bf55e5f31b2c890960ac33c5"
OLD_SELECTION_SHA = "c9bc13df6eda97efd1e26cbd050993101aca2589d5da5d4d029e522ffcd7a044"
BUDGETS = (0, 60, 120, 240, 360)
FAMILIES = ("tx45", "tx45_sbp")
FIXED = "tx45_sbp_scratch1_fit_available"
SOURCE_FILES = {
    f"neural_experiments/{version}/{name}.py"
    for version, names in (
        ("v6", ("model", "policy", "study", "data", "metrics")),
        ("v5", ("model", "policy", "study", "data", "metrics")),
        ("v4", ("model", "study", "data")),
        ("v3", ("study",)),
        ("v2", ("data",)),
    )
    for name in names
}


def grids():
    candidates = [
        {**c, "candidate_id": c["candidate_id"] + "_" + norm, "normalization": norm}
        for c in previous.grids()[0]
        for norm in ("first30", "fit_available")
    ]
    policies = [
        {"policy_id": f"c{len_before}", "gain_margin": margin, "pooled_r2_min": threshold}
        for len_before, (margin, threshold) in enumerate(
            itertools.product((0.0, 0.025), (0.0, 0.05))
        )
    ]
    return candidates, policies


class Inputs(ArrayInputs):
    def __init__(self, path, role, plan):
        self.path, self.index, self.checked = Path(path), read(path), set()
        index = self.index
        if (
            index["schema_version"] != 1
            or index["role"] != role
            or index["acquisition_plan_sha256"] != ACQUISITION_SHA
            or index["provider"] != {"doi": "10.5061/dryad.x0k6djj1h", "version_id": 462702}
        ):
            raise ValueError("Incorrect V6 role, source provider or acquisition lock")
        self.items = previous.unique(index["sessions"], lambda s: s["session_id"])
        for item in self.items.values():
            if (
                item["role"] != role
                or item["bin_seconds"] != 0.02
                or not 0 < item["n_bins"] <= 30000
            ):
                raise ValueError("Unauthorized input extent or mixed role")
            segment_ranges(item["segments"], item["n_bins"])
        if role == "development":
            verify(path, DEVELOPMENT_SHA)
            if (
                plan["development_index_sha256"] != DEVELOPMENT_SHA
                or len(self.items) != 128
                or set(index["participants"]) != set(plan["participants"])
            ):
                raise ValueError("Development cohort changed")
            parents = []
            for key, pin in (
                (
                    "v5_development",
                    "1fb2d9b3167e2c671e279dd89fe05ebb6f38633d4f3e6f3b56078a0c61621fb1",
                ),
                ("v5_new", "6ac718377da28dcb43ca127c8da6e6a78b40703ffc4b81358b06bb56ada4c5a4"),
            ):
                if index[key + "_index_sha256"] != pin:
                    raise ValueError("Wrong exposed parent index")
                verify(index[key + "_index_path"], pin)
                parents.extend(read(index[key + "_index_path"])["sessions"])
            parent = previous.unique(parents, lambda s: s["session_id"], self.items)
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
                    compare_structure(item[field], parent[sid][field], "exposed V5 parent " + field)
        elif index["study_lock"]["sha256"] != PLAN_SHA:
            raise ValueError("Fresh data does not bind the study lock")


def check_identity(selection, root):
    if (
        selection["schema_version"] != 1
        or selection["stage"] != "source"
        or selection["status"] != "complete"
        or selection["fresh_labels_opened"] is not False
    ):
        raise ValueError("Source execution is incomplete or fresh outcomes were opened")
    plan_path = contained(root, selection["plan"]["path"])
    verify(plan_path, PLAN_SHA)
    if selection["plan"]["sha256"] != PLAN_SHA or set(selection["code_sha256"]) != SOURCE_FILES:
        raise ValueError("Wrong frozen plan/executed source inventory")
    for path, digest in selection["code_sha256"].items():
        verify(contained(root, path), digest)
    old_path = contained(root, selection["old_selection"]["path"])
    if selection["old_selection"]["sha256"] != OLD_SELECTION_SHA:
        raise ValueError("Frozen V5 thresholds changed")
    verify(old_path, OLD_SELECTION_SHA)
    dev_path = contained(root, selection["development_index"]["path"])
    verify(dev_path, selection["development_index"]["sha256"])
    plan = read(plan_path)
    return plan, Inputs(dev_path, "development", plan), read(old_path)


def confirmation_reference(y, mask, segments, start, stop, prediction, training_mean):
    """Direct sums for one frozen artifact; targets beyond stop cannot enter."""
    if y.shape != (stop, 2) or mask.shape != (stop,) or prediction.shape != (stop, 2):
        raise ValueError("Confirmation supplied future/full arrays")
    yy = np.asarray(y, np.float64)
    clean = mask.copy()
    ranges = [
        (a, min(b, stop))
        for a, b in segment_ranges(
            segments, segments[-1]["end"] if isinstance(segments[-1], dict) else segments[-1][1]
        )
        if a < stop
    ]
    for a, b in ranges:
        clean[a : min(a + 62, b)] = False

    def window(a, b):
        use = np.zeros(stop, bool)
        use[a:b] = clean[a:b]
        n = int(use.sum())
        target, pred = yy[use], prediction[use]
        error = float(np.square(target - pred).sum()) if n else None
        sst = float(np.square(target - target.mean(0)).sum()) if n else 0
        mse = error / (2 * n) if n else None
        zero = float(np.square(target).sum() / (2 * n)) if n else None
        mean = float(np.square(target - training_mean).sum() / (2 * n)) if n else None
        return {
            "start_bin": a,
            "stop_bin": b,
            "n": n,
            "direction_sector_counts": previous.sectors(yy, use),
            "mse": mse,
            "r2": 1 - error / sst if sst > 0 else None,
            "zero_mse": zero,
            "training_mean_mse": mean,
            "gain_vs_zero": zero - mse if n else None,
            "gain_vs_training_mean": mean - mse if n else None,
            "mask_sha256": mask_hash(use),
        }

    mid = (start + stop) // 2
    groups = []
    for a in range(start, stop - 99, 100):
        if any(left <= a and a + 100 <= right for left, right in ranges):
            row = window(a, a + 100)
            if row["n"] >= 25:
                groups.append(row)
    return {
        "start_bin": start,
        "stop_bin": stop,
        "pooled": window(start, stop),
        "first_half": window(start, mid),
        "last_half": window(mid, stop),
        "groups": groups,
    }


def floor_reference(history):
    curves = history["past_learning_curves"]
    if (
        len(curves) > 3
        or len({c["session_id"] for c in curves}) != len(curves)
        or any(
            c["session_id"] == history["current_session_id"]
            or c["post_implant_day"] >= history["current_post_implant_day"]
            for c in curves
        )
    ):
        raise ValueError("Future/current or duplicate history in budget routing")
    valid = [c for c in curves if c["budgets"]["120"]["r2"] is not None]
    means = {
        b: sum(c["budgets"][str(b)]["r2"] for c in valid) / len(valid) if valid else None
        for b in (120, 240, 360)
    }
    comparisons, floor = [], 0
    for a, b in ((120, 240), (240, 360)):
        gains = [c["budgets"][str(b)]["r2"] - c["budgets"][str(a)]["r2"] for c in valid]
        gain = sum(gains) / len(gains) if gains else None
        positive = sum(v > 0 for v in gains)
        trigger = bool(
            len(valid) >= 2
            and means[120] <= 0.05
            and means[b] > 0
            and gain >= 0.03
            and positive >= np.ceil(2 * len(valid) / 3)
        )
        comparisons.append(
            {
                "earlier_training_seconds": a,
                "later_training_seconds": b,
                "paired_cases": len(valid),
                "source_session_ids": [c["session_id"] for c in valid],
                "mean_120_r2": means[120],
                "mean_later_r2": means[b],
                "mean_paired_gain": gain,
                "positive_paired_gains": positive,
                "triggered": trigger,
            }
        )
        if trigger:
            floor = b
    return {
        "training_seconds": floor,
        "reason": "strict_past_later_label_benefit" if floor else "no_supported_budget_floor",
        "source_session_ids": [c["session_id"] for c in curves],
        "paired_comparisons": comparisons,
    }


def decision_reference(history, stages, config):
    floor, events, visited = floor_reference(history), [], []
    action, final_reason = "abstain", "no_confirmed_artifact_at_cap"
    for budget in BUDGETS:
        if budget < floor["training_seconds"]:
            continue
        visited.append(budget)
        stage = stages[budget]
        if "unavailable" in stage:
            events.append(
                {"training_seconds": budget, "accepted": False, "reason": stage["unavailable"]}
            )
            final_reason = "incomplete_requested_confirmation"
            break
        values = stage["confirmation"]
        geometry = all(
            w["n"] >= 100 and sum(v >= 10 for v in w["direction_sector_counts"]) >= 4
            for w in (values["pooled"], values["first_half"], values["last_half"])
        )
        r2, threshold, margin = (
            values["pooled"]["r2"],
            config["pooled_r2_min"],
            config["gain_margin"],
        )
        r2_pass = r2 is not None and (r2 > 0 if threshold == 0 else r2 >= threshold)
        gains = [
            values[h][k]
            for h in ("first_half", "last_half")
            for k in ("gain_vs_zero", "gain_vs_training_mean")
        ]
        gain_pass = all(g is not None and (g > 0 if margin == 0 else g >= margin) for g in gains)
        passed = geometry and r2_pass and gain_pass
        reason = (
            "insufficient_confirmation_geometry"
            if not geometry
            else "pooled_r2_below_threshold"
            if not r2_pass
            else "temporal_half_gain_below_threshold"
            if not gain_pass
            else "frozen_artifact_confirmed"
        )
        events.append(
            {
                "training_seconds": budget,
                "model_sha256": stage["model_sha256"],
                "candidate_id": stage["candidate_id"],
                "accepted": passed,
                "reason": reason,
            }
        )
        if passed:
            action, final_reason = "accept", reason
            break
    return {
        "action": action,
        "training_seconds": budget,
        "requested_label_seconds": stage["requested_label_seconds"],
        "available_label_seconds": stage["available_label_seconds"],
        "decision_time_seconds": stage["decision_time_seconds"],
        "candidate_id": stage.get("candidate_id") if action == "accept" else None,
        "model_sha256": stage.get("model_sha256") if action == "accept" else None,
        "reason": final_reason,
        "events": events,
        "floor": floor,
    }, visited


def selection_reference(rows, people):
    choices = []
    for target in people:
        sources = [p for p in people if p != target]
        ranking = []
        for config in grids()[1]:
            measures = []
            costs = []
            for person in sources:
                cases = [
                    r
                    for r in rows
                    if r["participant"] == person and r["policy_id"] == config["policy_id"]
                ]
                costs.append(sum(r["label_seconds"] for r in cases) / len(cases))
                available = [r for r in cases if r["metrics"]["r2"] is not None]
                if not available:
                    continue
                accepted = [r for r in available if r["action"] == "accept"]
                measures.append(
                    (
                        np.mean([r["metrics"]["r2"] for r in available]),
                        np.mean([r["daily240_r2"] for r in available]),
                        len(accepted) / len(available),
                        sum(
                            r["metrics"]["summed_mse"] / 2 > r["zero_mse"] + 1e-12 for r in accepted
                        )
                        / len(available),
                        sum(r["metrics"]["r2"] <= 0 for r in accepted) / len(available),
                    )
                )
            accuracy, baseline, coverage, harm, fail = np.asarray(measures).mean(0).tolist()
            cost = float(np.mean(costs))
            ranking.append(
                {
                    "policy_id": config["policy_id"],
                    "mean_r2": accuracy,
                    "daily240_mean_r2": baseline,
                    "mean_coverage": coverage,
                    "mean_harmful_burden": harm,
                    "mean_nonpositive_burden": fail,
                    "mean_label_seconds": cost,
                    "eligible_source_people": len(measures),
                    "source_people_with_cost": len(costs),
                    "feasible": coverage >= 0.6 and accuracy >= baseline - 0.05,
                    "objective": accuracy - 0.1 * harm - 0.05 * fail - 0.02 * cost / 420,
                }
            )
        possible = [r for r in ranking if r["feasible"]]
        chosen = (
            sorted(
                possible,
                key=lambda r: (
                    r["mean_harmful_burden"],
                    r["mean_nonpositive_burden"],
                    -r["mean_r2"],
                    r["mean_label_seconds"],
                ),
            )[0]
            if possible
            else sorted(ranking, key=lambda r: -r["objective"])[0]
        )
        choices.append(
            {
                "target_participant": target,
                "policy_id": chosen["policy_id"],
                "source_participants": sources,
                "selection_rule": "feasible_risk_accuracy_cost"
                if possible
                else "fallback_fixed_utility",
                "ranking": ranking,
            }
        )
    return choices


class Verifier(previous.Verifier):
    def fit_v6(self, row, item, raw, prior, history, labels):
        phase, budget = row["phase"], row["training_seconds"]
        if row["participant"] != item["participant_id"] or row["session_id"] != item["session_id"]:
            raise ValueError("Fit recording identity mismatch")
        if phase == "prior_confirmation":
            if budget != 0 or row["family"] not in FAMILIES:
                raise ValueError("Invalid historical confirmation fit")
            family = row["family"]
            config = next(c["config"] for c in grids()[0] if c["config"]["family"] == family)
            arrays, _ = self.model(row, item, raw[family], config, 1500)
            if not np.allclose(arrays["weights"], prior[family], atol=1e-10, rtol=1e-10):
                raise ValueError("Prior confirmation weights changed")
        else:
            if (
                phase not in ("frozen_full", "control")
                or budget not in (60, 120, 240, 360)
                or phase == "control"
                and budget == 60
            ):
                raise ValueError("Unexpected V6 fit role")
            candidate = next(c for c in grids()[0] if c["candidate_id"] == FIXED)
            compare_structure(row["candidate"], candidate, "fixed recipe")
            family = "tx45_sbp"
            stop = min(1500 + budget * 50, item["n_bins"])
            if phase == "frozen_full" and stop != 1500 + budget * 50:
                raise ValueError("Incomplete prefix presented as frozen confirmed fit")
            arrays, z = self.model(row, item, raw[family], candidate["config"], stop)
            use = previous.label_mask(labels["eval_mask"], item["segments"], stop)
            stats = reference_statistics(z[:stop], labels["y"][:stop], use)
            expected = {
                "candidate_id": FIXED,
                "head": "scratch1",
                "normalization": "fit_available",
                "normalization_stop_bin": stop,
                "label_start_bin": 1500,
                "label_stop_bin": stop,
                "requested_label_seconds": (stop - 1500) * 0.02,
                "training_rows": stats["n"],
                "training_mask_sha256": mask_hash(use),
                "fallback": stats["n"] < 50,
                "fallback_output": "zero",
                "feature_penalty": 1.0,
                "intercept_penalty": None,
                "free_intercept": True,
                "constrained_zero_intercept": False,
                "prior_sha256": None,
            }
            compare_structure(row["fit"], expected, "frozen fit normalizer/label bounds")
            residual = validate_weights(
                arrays["weights"], stats, daily_penalty=1.0, free_bias=True
            )["normal_equation_relative_residual"]
            if residual is not None:
                self.residuals.append(residual)
        compare_structure(row["history"], history[family], "same-person strict-past fit history")
        self.fits_checked += 1
        return arrays

    def score_v6(self, row, item, raw, targets, expected_weights, config, prefix):
        # The independent V5 scorer is reused on a copied target mask with the V6
        # common suffix; continuation mode avoids adding the earlier V5 cutoff.
        targets = {"y": targets["y"], "eval_mask": targets["eval_mask"].copy()}
        targets["eval_mask"][:22600] = False
        targets["eval_mask"][30000:] = False
        return self.score(
            row, item, raw, targets, expected_weights, config, prefix, "v6_common_suffix"
        )


def check_v5(verifier, item, raw, prior, history, labels, evidence, probes, old_selection):
    """Rebuild the immutable V5 decision on the new, common historical pool."""
    if item["n_bins"] < 1500:
        # The frozen V6 wrapper explicitly abstains before constructing the V5
        # replay, whose normalizer requires the complete30s warmup.
        expected = {
            "history_evidence": {"warmup_available": False},
            "stages": [],
            "fits": [],
            "decision": {
                "action": "abstain",
                "budget_seconds": 0,
                "reason": "unlabeled_warmup_unavailable",
            },
            "policy_id": None,
            "label_requests": [],
        }
        compare_structure(evidence, expected, "V5 incomplete-warmup guard")
        return expected["decision"], "tx45_sbp", np.zeros_like(prior["tx45_sbp"])
    candidates = previous.grids()[0]
    checked = {
        previous.fit_key(r): previous.Verifier.fit(verifier, r, item, raw, prior, history, labels)
        for r in evidence["fits"]
    }
    if len(checked) != len(evidence["fits"]):
        raise ValueError("Duplicate V5 comparison fit")
    expected_history = previous.history_evidence(
        item, verifier.items, verifier.layouts, history, probes
    )
    compare_structure(evidence["history_evidence"], expected_history, "matched-history V5 evidence")
    stages = {}
    expected_fits = set()
    for stage in evidence["stages"]:
        b = stage["budget_seconds"]
        stop = 1500 + b * 50
        if b in stages or b not in previous.BUDGETS:
            raise ValueError("Unexpected V5 stage")
        if item["n_bins"] < stop:
            expected = {
                "budget_seconds": b,
                "unavailable": "requested_checkpoint_not_observed",
                "available_label_seconds": max(0, (item["n_bins"] - 1500) * 0.02),
            }
        else:
            y, mask = labels["y"][:stop].astype(float), labels["eval_mask"][:stop].copy()
            y[:1500], mask[:1500] = 0, False
            predictions = {}
            for c in candidates:
                key = item["session_id"], c["candidate_id"], b, "validation", False
                expected_fits.add(key)
                predictions[c["candidate_id"]] = replay(checked[key], raw[c["config"]["family"]])[
                    :stop
                ]
            expected = previous.stage_reference(y, mask, item["segments"], b, predictions)
        compare_structure(stage, expected, "V5 prefix validation")
        stages[b] = expected
    selected = next(
        c["policy_id"]
        for c in old_selection["choices"]
        if c["target_participant"] == item["participant_id"]
    )
    if evidence["policy_id"] != selected:
        raise ValueError("V5 frozen target-person threshold changed")
    config = next(c for c in previous.grids()[1] if c["policy_id"] == selected)
    decision, requested = previous.decision_reference(expected_history, stages, config)
    compare_structure(evidence["decision"], decision, "V5 sequential comparison decision")
    compare_structure(evidence["label_requests"], requested, "V5 label requests")
    if set(stages) != set(requested):
        raise ValueError("Unrequested V5 future stage")
    cid, budget = decision["candidate_id"], decision["budget_seconds"]
    family = (
        "tx45_sbp"
        if cid == "zero"
        else next(c["config"]["family"] for c in candidates if c["candidate_id"] == cid)
    )
    if cid == "zero":
        weights = np.zeros_like(prior[family])
    elif budget == 0:
        weights = prior[family]
    else:
        key = item["session_id"], cid, budget, "full", False
        expected_fits.add(key)
        weights = checked[key]["weights"]
    if set(checked) != expected_fits:
        raise ValueError("Unbound V5 comparison fit")
    return decision, family, weights


def prior_selection_reference(item, verifier, histories, probes):
    result = []
    for family in FAMILIES:
        cases = []
        for old in selected_histories(item, verifier.items, verifier.layouts):
            row = probes[old["session_id"], family]
            if row["metrics"]["summed_mse"] is not None and any(
                h["disposition"] == "used" for h in row["history"]
            ):
                cases.append(
                    {
                        "session_id": old["session_id"],
                        "post_implant_day": old["post_implant_day"],
                        "n": row["metrics"]["n"],
                        "summed_mse": row["metrics"]["summed_mse"],
                    }
                )
        result.append({"family": family, "cases": cases})
    eligible = [
        r
        for r in result
        if r["cases"] and any(h["disposition"] == "used" for h in histories[r["family"]])
    ]
    family = (
        min(eligible, key=lambda r: sum(c["summed_mse"] for c in r["cases"]) / len(r["cases"]))[
            "family"
        ]
        if eligible
        else "tx45_sbp"
    )
    return {"family": family, "historical_probes": result}


def learning_curves_reference(item, verifier, controls):
    rows = []
    for old in selected_histories(item, verifier.items, verifier.layouts):
        budgets = {}
        for b in (120, 240, 360):
            row = controls[old["session_id"], f"daily{b}"]
            m = row["metrics"]
            budgets[str(b)] = {
                "r2": m["r2"],
                "n": m["n"],
                "coordinate_mse": m["summed_mse"] / 2 if m["summed_mse"] is not None else None,
                "model_sha256": row["model_sha256"],
                "trace_sha256": row["trace_sha256"],
            }
        rows.append(
            {
                "session_id": old["session_id"],
                "post_implant_day": old["post_implant_day"],
                "budgets": budgets,
            }
        )
    return {
        "current_session_id": item["session_id"],
        "current_post_implant_day": item["post_implant_day"],
        "past_learning_curves": rows,
    }


def row_identity(row, item, method, requested, action, training):
    for key, value in {
        "participant": item["participant_id"],
        "session_id": item["session_id"],
        "post_implant_day": item["post_implant_day"],
        "domain": "new_sessions",
        "method": method,
        "label_seconds": requested,
        "available_label_seconds": max(0, min(requested, item["n_bins"] * 0.02 - 30)),
        "action": action,
        "training_label_seconds": training,
    }.items():
        compare_structure(row[key], value, "scored identity/cost " + key)


def check_case(
    verifier,
    inputs,
    item,
    evidence,
    fits,
    confirmed,
    controls,
    all_controls,
    probes,
    old_selection,
    source,
):
    raw, prior, history, labels = verifier.query(inputs, item)
    if (
        evidence["participant"] != item["participant_id"]
        or evidence["session_id"] != item["session_id"]
    ):
        raise ValueError("Evidence identity mismatch")
    fit_rows = previous.unique(fits, lambda r: (r["phase"], r["training_seconds"]))
    checked = {
        key: verifier.fit_v6(row, item, raw, prior, history, labels)
        for key, row in fit_rows.items()
    }
    expected_keys = {("control", b) for b in (120, 240, 360)} if item["n_bins"] > 1500 else set()
    expected_prior_selection = prior_selection_reference(item, verifier, history, probes)
    stages = {}
    for stage in evidence["stages"]:
        b = stage["training_seconds"]
        if b not in BUDGETS or b in stages:
            raise ValueError("Unexpected confirmation stage")
        stop, end = 1500 + b * 50, 4500 + b * 50
        expected = {
            "training_seconds": b,
            "requested_label_seconds": b + 60,
            "available_label_seconds": max(0, min(b + 60, item["n_bins"] * 0.02 - 30)),
            "decision_time_seconds": min(90 + b, item["n_bins"] * 0.02),
        }
        if item["n_bins"] < end:
            expected["unavailable"] = "complete_fit_and_confirmation_not_observed"
        else:
            key = ("prior_confirmation" if b == 0 else "frozen_full", b)
            expected_keys.add(key)
            arrays, fitted = checked[key], fit_rows[key]
            if b == 0:
                compare_structure(
                    fitted["prior_selection"],
                    expected_prior_selection,
                    "historical prior family selection",
                )
                family = expected_prior_selection["family"]
                if fitted["family"] != family:
                    raise ValueError("Prior family differs from strictly past selection")
                expected.update(
                    candidate_id=family + "_prior0",
                    inner_selection={"kind": "strict_past_only", **expected_prior_selection},
                )
                mean = np.zeros(2)
            else:
                family = "tx45_sbp"
                train = previous.label_mask(labels["eval_mask"], item["segments"], stop)
                mean = (
                    labels["y"][:stop][train].astype(float).mean(0) if train.any() else np.zeros(2)
                )
                expected.update(
                    candidate_id=FIXED,
                    selection_rule="fixed_before_fresh_assessment",
                    fit=fitted["fit"],
                )
            prediction = replay(arrays, raw[family])[:end]
            y, mask = labels["y"][:end].astype(float), labels["eval_mask"][:end].copy()
            y[:1500], mask[:1500] = 0, False
            expected.update(
                model_sha256=fitted["model_sha256"],
                model_path=fitted["model_path"],
                fit_stop_bin=stop,
                confirmation_start_bin=stop + 100,
                confirmation_stop_bin=end,
                confirmation=confirmation_reference(
                    y, mask, item["segments"], stop + 100, end, prediction, mean
                ),
            )
        compare_structure(stage, expected, "independent exact-artifact confirmation")
        stages[b] = expected
    if set(checked) != expected_keys:
        raise ValueError("Extra, absent or unbound V6 fit export")
    curves = learning_curves_reference(item, verifier, all_controls)
    compare_structure(
        evidence["history_evidence"], curves, "strict-past fixed-recipe learning curves"
    )
    used = set()
    config_map = {p["policy_id"]: p for p in grids()[1]}
    for row in confirmed:
        decision, visited = decision_reference(curves, stages, config_map[row["policy_id"]])
        used.update(visited)
        compare_structure(row["decision"], decision, "independent confirmed stopping decision")
        if not source:
            requests = []
            for b in visited:
                asks = [b + 60] if b == 0 or "unavailable" in stages[b] else [b, b + 60]
                for requested in asks:
                    if not requests or requested > requests[-1]:
                        requests.append(requested)
            compare_structure(row["label_requests"], requests, "actual lazy label requests")
        b, action = decision["training_seconds"], decision["action"]
        row_identity(row, item, "confirmed", decision["requested_label_seconds"], action, b)
        if action == "abstain":
            family, prefix, weights = (
                "tx45_sbp",
                min(1500, item["n_bins"]),
                np.zeros_like(prior["tx45_sbp"]),
            )
        else:
            if row["model_sha256"] != decision["model_sha256"]:
                raise ValueError("Deployment is not the exact confirmed model bytes")
            key = ("prior_confirmation" if b == 0 else "frozen_full", b)
            family = expected_prior_selection["family"] if b == 0 else "tx45_sbp"
            prefix, weights = 1500 + 50 * b, checked[key]["weights"]
        config = next(c["config"] for c in grids()[0] if c["config"]["family"] == family)
        verifier.score_v6(row, item, raw[family], labels, weights, config, prefix)
    if set(stages) != (set(BUDGETS) if source else used):
        raise ValueError("Unrequested future confirmation evidence")
    old_decision, old_family, old_weights = check_v5(
        verifier,
        item,
        raw,
        prior,
        history,
        labels,
        evidence["v5_comparator"],
        probes,
        old_selection,
    )
    controls = previous.unique(
        controls,
        lambda r: r["method"],
        ("zero", "history0", "daily120", "daily240", "daily360", "v5_adaptive"),
    )
    for name, row in controls.items():
        family = old_family if name == "v5_adaptive" else "tx45_sbp"
        budget = (
            old_decision["budget_seconds"]
            if name == "v5_adaptive"
            else int(name[5:])
            if name.startswith("daily")
            else 0
        )
        prefix = min(1500 + 50 * budget if name.startswith("daily") else 1500, item["n_bins"])
        action = (
            old_decision["action"]
            if name == "v5_adaptive"
            else "abstain"
            if name == "zero"
            else "accept"
        )
        row_identity(row, item, name, budget, action, budget)
        weights = (
            old_weights
            if name == "v5_adaptive"
            else np.zeros_like(prior[family])
            if name == "zero"
            or item["n_bins"] < 1500
            or name.startswith("daily")
            and item["n_bins"] == 1500
            else prior[family]
            if name == "history0"
            else checked["control", budget]["weights"]
        )
        config = next(c["config"] for c in grids()[0] if c["config"]["family"] == family)
        verifier.score_v6(row, item, raw[family], labels, weights, config, prefix)
    if source:
        if evidence["zero_model_path"] != controls["zero"]["model_path"]:
            raise ValueError("Source abstention export not bound to zero control")
        for row in confirmed:
            compare_structure(
                row["daily240_r2"], controls["daily240"]["metrics"]["r2"], "paired source daily240"
            )
            zero = controls["zero"]["metrics"]["summed_mse"]
            compare_structure(
                row["zero_mse"],
                zero / 2 if zero is not None else None,
                "paired source zero loss units",
            )


def audit_selection(path, root=ROOT):
    selection = read(path)
    plan, inputs, old_selection = check_identity(selection, root)
    candidates, policies = grids()
    compare_structure(selection["candidates"], candidates, "diagnostic candidate inventory")
    compare_structure(selection["policies"], policies, "locked controller grid")
    manifest = read(Path(path).parent / "manifest.json")
    expected_manifest = {
        k: selection[k]
        for k in (
            "schema_version",
            "plan",
            "development_index",
            "old_selection",
            "code_sha256",
            "stage",
            "status",
            "fresh_labels_opened",
        )
    }
    expected_manifest["selection_sha256"] = sha256(path)
    compare_structure(manifest, expected_manifest, "complete source manifest")
    rows = previous.unique(
        selection["cases"],
        lambda r: (r["session_id"], r["policy_id"]),
        itertools.product(inputs.items, [c["policy_id"] for c in policies]),
    )
    evidence = previous.unique(selection["evidence"], lambda r: r["session_id"], inputs.items)
    fits = previous.unique(
        selection["fits"], lambda r: (r["session_id"], r["phase"], r["training_seconds"])
    )
    if any(k[0] not in inputs.items for k in fits):
        raise ValueError("Fit uses undeclared session")
    probes = previous.unique(
        selection["history_probes"],
        lambda r: (r["session_id"], r["family"]),
        itertools.product(inputs.items, FAMILIES),
    )
    controls = previous.unique(
        [r for e in evidence.values() for r in e["controls"]],
        lambda r: (r["session_id"], r["method"]),
        itertools.product(
            inputs.items, ("zero", "history0", "daily120", "daily240", "daily360", "v5_adaptive")
        ),
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
                compare_structure(row[key], value, "V5-window historical probe " + key)
            config = next(c["config"] for c in candidates if c["config"]["family"] == family)
            verifier.score(
                row, item, raw[family], labels, prior[family], config, 1500, "new_sessions"
            )
    print("all strictly chronological prior probes checked", flush=True)
    for item in verifier.items:
        sid = item["session_id"]
        check_case(
            verifier,
            inputs,
            item,
            evidence[sid],
            [r for k, r in fits.items() if k[0] == sid],
            [rows[sid, c["policy_id"]] for c in policies],
            evidence[sid]["controls"],
            controls,
            probes,
            old_selection,
            True,
        )
        print(f"independent frozen fits, confirmation and decisions checked: {sid}", flush=True)
    compare_structure(
        selection["choices"],
        selection_reference(selection["cases"], plan["participants"]),
        "whole-person-excluded source ranking",
    )
    return {
        "status": "PASS",
        "stage": "source",
        "created_at": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": sha256(__file__),
        "selection_sha256": sha256(path),
        "manifest_sha256": sha256(Path(path).parent / "manifest.json"),
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
            "No producer imports. Separate NumPy ridge optimality, feature-only recurrence "
            "replay, sklearn/angle/MSE, exact-artifact confirmation, strict-past budget floors, "
            "lazy decisions, matched V5 and whole-person source ranking. Policy and verifier "
            "share an author; not external replication."
        ),
    }


def validate_fresh_provenance(inputs, development, plan):
    """Authenticate prior fifteen archive members and fixed ranks16--18, no substitutions."""
    index = inputs.index
    people = plan["new_session_participants"]
    if index["participants"] != people or "T3" in people:
        raise ValueError("Fresh participant inventory differs from locked domain")
    for key in ("v5_development_index_sha256", "v5_new_index_sha256"):
        if index[key] != development.index[key]:
            raise ValueError("Fresh source ancestry changed")
    sources = previous.unique(
        index["acquisition_sources"], lambda r: r["path"], [f"acquisition/{p}.json" for p in people]
    )
    expected, receipts = {}, []
    for person in people:
        ref = sources[f"acquisition/{person}.json"]
        path = contained(inputs.path.parent, ref["path"])
        verify(path, ref["sha256"])
        receipt = read(path)
        receipts.append(receipt)
        prior = sorted(
            [s["source"] for s in development.items.values() if s["participant_id"] == person],
            key=lambda s: s["archive_ordinal"],
        )
        if (
            receipt["status"] != "complete"
            or receipt["participant_id"] != person
            or receipt["plan_sha256"] != ACQUISITION_SHA
            or len(prior) != 15
            or receipt["selection"]["regular_session_ranks_one_based"] != [16, 17, 18]
            or receipt["selection"]["scores_or_duration_used_for_selection"] is not False
        ):
            raise ValueError("Fresh archive selection or authenticated-prefix inventory changed")
        for key in ("v5_development_index_sha256", "v5_new_index_sha256"):
            if receipt[key] != development.index[key]:
                raise ValueError("Acquisition authenticated wrong exposed parent")
        archive = receipt["archive"]
        if (
            archive["file_id"] != prior[0]["archive_file_id"]
            or archive["provider_sha256"] != prior[0]["archive_sha256_published"]
            or receipt["full_archive_sha256_verified"] is not False
        ):
            raise ValueError("Pinned provider archive identity or hash claim changed")
        pattern = rf"(?:\./)?decoding/{person}/{person}_day_(\d+)_decoding\.mat"
        seen = [
            s
            for s in receipt["seen_members"]
            if s["regular_file"] and re.fullmatch(pattern, s["name"])
        ]
        authenticated, selected = receipt["verified_prior_members"], receipt["selected"]
        if (
            len(authenticated) != 15
            or len(seen) != 15 + len(selected)
            or len(selected) > 3
            or len(selected) < 3
            and not receipt["tar_end_observed"]
            or receipt["missing_requested_ranks"] != list(range(16 + len(selected), 19))
        ):
            raise ValueError("Archive walk/shortage identity mismatch")
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
                or not re.fullmatch("[0-9a-f]{64}", member["member_sha256"])
            ):
                raise ValueError("Member does not bind exact chronological archive walk")
            if rank <= 15:
                for key in (
                    "member_name",
                    "member_size",
                    "member_sha256",
                    "archive_ordinal",
                    "session_id",
                ):
                    if member[key] != prior[rank - 1][key]:
                        raise ValueError("Authenticated prior fifteen changed")
            else:
                if (
                    member["session_id"] in expected
                    or member["raw_path"] != f"raw/{person}/{member['session_id']}.mat"
                ):
                    raise ValueError("Duplicate or unsafe selected member")
                expected[member["session_id"]] = {
                    **member,
                    "archive_file_id": archive["file_id"],
                    "archive_sha256_published": archive["provider_sha256"],
                    "full_archive_sha256_verified": False,
                }
    previous.validate_acquisition_dispositions(index["participant_dispositions"], receipts)
    if set(expected) != set(inputs.items):
        raise ValueError("Prepared fresh session inventory differs from selected archive members")
    for sid, item in inputs.items.items():
        compare_structure(item["source"], expected[sid], "exact receipt-bound fresh source")
        if item["post_implant_day"] != expected[sid]["post_implant_day"] or item["n_bins"] != min(
            item["full_prepared_n_bins"], 30000
        ):
            raise ValueError("Fresh chronology or fixed cap changed")
        verify(
            contained(item["raw_source_root"], item["source"]["raw_path"]),
            item["source"]["member_sha256"],
        )


METHODS = ("zero", "history0", "daily120", "daily240", "daily360", "v5_adaptive", "confirmed")


def average(values):
    return sum(values) / len(values) if values else None


def interval(values):
    if not values:
        return [None, None]
    rng = np.random.default_rng(20261025)
    draws = rng.integers(0, len(values), (10000, len(values)))
    distribution = np.asarray(values, float)[draws].sum(1) / len(values)
    return np.quantile(distribution, (0.025, 0.975)).tolist()


def summary_reference(rows):
    sessions = sorted({r["session_id"] for r in rows})
    lookup = previous.unique(
        rows, lambda r: (r["session_id"], r["method"]), itertools.product(sessions, METHODS)
    )
    for sid in sessions:
        same = [lookup[sid, m] for m in METHODS]
        if (
            len({r["eval_mask_sha256"] for r in same}) != 1
            or len({r["metrics"]["r2"] is None for r in same}) != 1
        ):
            raise ValueError("Methods do not share paired scoring geometry")
    methods = {}
    for name in METHODS:
        all_rows = [r for r in rows if r["method"] == name]
        scored = [r for r in all_rows if r["metrics"]["r2"] is not None]
        units, costs, risk = [], [], []
        for person in sorted({r["participant"] for r in all_rows}):
            requested = [r for r in all_rows if r["participant"] == person]
            valid = [r for r in scored if r["participant"] == person]
            costs.append(
                {
                    "participant": person,
                    "requested_label_seconds": average([r["label_seconds"] for r in requested]),
                    "available_label_seconds": average(
                        [r["available_label_seconds"] for r in requested]
                    ),
                }
            )
            if valid:
                units.append(
                    {
                        "participant": person,
                        "n_sessions": len(valid),
                        "r2": average([r["metrics"]["r2"] for r in valid]),
                        "angular_error_degrees": average(
                            [r["metrics"]["angular_error_degrees"] for r in valid]
                        ),
                    }
                )
                risk.append(
                    {
                        "participant": person,
                        "nonpositive_burden": sum(
                            r["action"] == "accept" and r["metrics"]["r2"] <= 0 for r in valid
                        )
                        / len(valid),
                        "harmful_burden": sum(
                            r["action"] == "accept"
                            and r["metrics"]["summed_mse"]
                            > lookup[r["session_id"], "zero"]["metrics"]["summed_mse"] + 1e-12
                            for r in valid
                        )
                        / len(valid),
                        "coverage": sum(r["action"] == "accept" for r in valid) / len(valid),
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
        scores = [u["r2"] for u in units]
        methods[name] = {
            "units": units,
            "cost_units": costs,
            "requested_sessions": len(all_rows),
            "scored_sessions": len(scored),
            "unavailable_sessions": len(all_rows) - len(scored),
            "n_participants": len(units),
            "mean_r2": average(scores),
            "r2_bootstrap_95": interval(scores),
            "positive_participants": sum(v > 0 for v in scores),
            "mean_angular_error_degrees": average([u["angular_error_degrees"] for u in units]),
            "mean_requested_label_seconds": average([u["requested_label_seconds"] for u in costs]),
            "mean_available_label_seconds": average([u["available_label_seconds"] for u in costs]),
            "accepted_scored_sessions": len(accepted),
            "acceptance_coverage": len(accepted) / len(scored) if scored else None,
            "abstained_selected_sessions": sum(r["action"] == "abstain" for r in all_rows),
            "nonpositive_accepted_sessions": nonpositive,
            "harmful_vs_zero_accepted_sessions": harmful,
            "nonpositive_acceptance_fraction": len(nonpositive) / len(accepted)
            if accepted
            else None,
            "harmful_acceptance_fraction": len(harmful) / len(accepted) if accepted else None,
            "risk_units": risk,
            "mean_nonpositive_burden": average([u["nonpositive_burden"] for u in risk]),
            "mean_harmful_burden": average([u["harmful_burden"] for u in risk]),
        }
    contrasts = []
    for method in METHODS[:-1]:
        people = sorted({r["participant"] for r in rows if r["metrics"]["r2"] is not None})
        paired = []
        for person in people:
            rr = [
                r
                for r in rows
                if r["method"] == "confirmed"
                and r["participant"] == person
                and r["metrics"]["r2"] is not None
            ]
            paired.append(
                average(
                    [
                        r["metrics"]["r2"] - lookup[r["session_id"], method]["metrics"]["r2"]
                        for r in rr
                    ]
                )
            )
        contrasts.append(
            {
                "control": method,
                "n_participants": len(paired),
                "mean_gain": average(paired),
                "gain_bootstrap_95": interval(paired),
            }
        )
    opportunities = []
    for sid in sessions:
        row = lookup[sid, "confirmed"]
        if row["metrics"]["r2"] is None:
            continue
        early, zero = lookup[sid, "daily120"]["metrics"]["r2"], lookup[sid, "zero"]["metrics"]["r2"]
        late = max(lookup[sid, m]["metrics"]["r2"] for m in ("daily240", "daily360"))
        if late - early >= 0.05 and late - zero >= 0.05:
            gain, escalation = row["metrics"]["r2"] - early, row["training_label_seconds"] >= 240
            opportunities.append(
                {
                    "participant": row["participant"],
                    "session_id": sid,
                    "later_fixed_gain_over_daily120": late - early,
                    "confirmed_gain_over_daily120": gain,
                    "escalated_to_at_least240": escalation,
                    "captured": row["action"] == "accept" and escalation and gain >= 0.025,
                }
            )
    t2 = [lookup[s, "confirmed"] for s in sessions if lookup[s, "confirmed"]["participant"] == "T2"]
    valid = [r for r in t2 if r["metrics"]["r2"] is not None]
    t2_scores = [r["metrics"]["r2"] for r in valid]
    t2_gains = [
        r["metrics"]["r2"] - lookup[r["session_id"], "v5_adaptive"]["metrics"]["r2"] for r in valid
    ]
    t2_harm = [
        r["session_id"]
        for r in valid
        if r["action"] == "accept"
        and r["metrics"]["summed_mse"]
        > lookup[r["session_id"], "zero"]["metrics"]["summed_mse"] + 1e-12
    ]
    positive = sum(r["action"] == "accept" and r["metrics"]["r2"] > 0 for r in valid)
    confirmed, old = methods["confirmed"], methods["v5_adaptive"]
    old_risk = {r["participant"]: r for r in old["risk_units"]}
    risk_contrasts = {}
    for key in ("nonpositive_burden", "harmful_burden", "coverage"):
        deltas = [r[key] - old_risk[r["participant"]][key] for r in confirmed["risk_units"]]
        risk_contrasts[key] = {
            "mean_difference": average(deltas),
            "participant_bootstrap_95": interval(deltas),
        }
    useful = bool(
        confirmed["n_participants"] == 8
        and confirmed["mean_r2"] >= 0.1
        and confirmed["r2_bootstrap_95"][0] > 0
        and confirmed["positive_participants"] >= 6
    )
    captured = sum(r["captured"] for r in opportunities)
    return {
        "methods": methods,
        "contrasts": contrasts,
        "risk_contrasts_vs_v5": risk_contrasts,
        "T2": {
            "requested_sessions": len(t2),
            "scored_sessions": len(valid),
            "mean_r2": average(t2_scores),
            "r2_session_bootstrap_95": interval(t2_scores),
            "positive_accepted_sessions": positive,
            "harmful_accepted_sessions": t2_harm,
            "gain_over_v5": average(t2_gains),
            "gain_session_bootstrap_95": interval(t2_gains),
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
                and confirmed["acceptance_coverage"] >= 0.5
                and len(confirmed["nonpositive_accepted_sessions"])
                < len(old["nonpositive_accepted_sessions"])
                and len(confirmed["harmful_vs_zero_accepted_sessions"])
                <= len(old["harmful_vs_zero_accepted_sessions"])
            ),
            "T2_recovered_without_harmful_acceptance": bool(
                len(t2) == len(valid) == 3
                and average(t2_scores) >= 0.05
                and positive >= 2
                and not t2_harm
                and average(t2_gains) >= 0.03
            ),
            "captured_later_label_benefit": captured / len(opportunities) >= 0.5
            if opportunities
            else None,
        },
        "limitations": (
            "Participant bootstrap for cohort; T2 session intervals are descriptive small-N. "
            "Observed acceptance fractions are not population safety bounds. Opportunity "
            "analysis is a prespecified post-assessment comparison, not an oracle available "
            "to the controller."
        ),
    }


def audit_assessment(path, root=ROOT):
    assessment = read(path)
    if (
        assessment["schema_version"] != 1
        or assessment["stage"] != "assessment"
        or assessment["status"] != "assessment_complete"
        or assessment["fresh_labels_opened"] is not True
    ):
        raise ValueError("Assessment is incomplete")
    compare_structure(
        read(Path(path).parent / "manifest.json"), assessment, "final manifest/results identity"
    )
    selection_path = contained(root, assessment["selection_path"])
    verify(selection_path, assessment["selection_sha256"])
    selection = read(selection_path)
    plan, development, old_selection = check_identity(selection, root)
    receipt_path = contained(root, assessment["source_audit_path"])
    verify(receipt_path, assessment["source_audit_sha256"])
    receipt = read(receipt_path)
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stage") != "source"
        or receipt["selection_sha256"] != sha256(selection_path)
        or receipt["plan_sha256"] != PLAN_SHA
        or receipt["development_index_sha256"] != DEVELOPMENT_SHA
    ):
        raise ValueError("Assessment lacks passed exact source-selection audit")
    index_path = contained(root, assessment["index"]["path"])
    verify(index_path, assessment["index"]["sha256"])
    inputs = Inputs(index_path, "new_sessions", plan)
    validate_fresh_provenance(inputs, development, plan)
    compare_structure(
        assessment["data_dispositions"],
        inputs.index["participant_dispositions"],
        "all acquisition shortages retained",
    )
    rows = previous.unique(
        assessment["cases"],
        lambda r: (r["session_id"], r["method"]),
        itertools.product(inputs.items, METHODS),
    )
    evidence = previous.unique(assessment["evidence"], lambda r: r["session_id"], inputs.items)
    fits = previous.unique(
        assessment["fits"], lambda r: (r["session_id"], r["phase"], r["training_seconds"])
    )
    if any(k[0] not in inputs.items for k in fits):
        raise ValueError("Fit refers to an unselected fresh session")
    controls = previous.unique(
        [r for e in selection["evidence"] for r in e["controls"]],
        lambda r: (r["session_id"], r["method"]),
    )
    probes = previous.unique(selection["history_probes"], lambda r: (r["session_id"], r["family"]))
    choices = previous.unique(
        selection["choices"], lambda r: r["target_participant"], plan["participants"]
    )
    verifier = Verifier(development, root)
    for item in inputs.items.values():
        sid = item["session_id"]
        row = rows[sid, "confirmed"]
        if row["policy_id"] != choices[item["participant_id"]]["policy_id"]:
            raise ValueError("Assessment changed source-selected threshold")
        check_case(
            verifier,
            inputs,
            item,
            evidence[sid],
            [r for k, r in fits.items() if k[0] == sid],
            [row],
            [rows[sid, m] for m in METHODS[:-1]],
            controls,
            probes,
            old_selection,
            False,
        )
        print(f"independent fresh case verified: {sid}", flush=True)
    summary = summary_reference(assessment["cases"])
    compare_structure(
        assessment["summary"], summary, "independent matched participant aggregate/gates"
    )
    return {
        "status": "PASS",
        "stage": "assessment",
        "created_at": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": sha256(__file__),
        "assessment_sha256": sha256(path),
        "manifest_sha256": sha256(Path(path).parent / "manifest.json"),
        "selection_sha256": sha256(selection_path),
        "source_audit_sha256": sha256(receipt_path),
        "plan_sha256": PLAN_SHA,
        "development_index_sha256": sha256(development.path),
        "index_sha256": sha256(index_path),
        "code_sha256": selection["code_sha256"],
        "assessment_cases_checked": len(rows),
        "decisions_checked": len(evidence),
        "fits_checked": verifier.fits_checked,
        "models_checked": verifier.models_checked,
        "full_traces_checked": verifier.traces_checked,
        "unavailable_scores": sum(r["metrics"]["r2"] is None for r in rows.values()),
        "max_prediction_absolute_error": max(verifier.errors, default=0),
        "max_normal_equation_relative_residual": max(verifier.residuals, default=0),
        "summary": summary,
        "scope": (
            "No producer imports. Independently bound selected ranks, input hashes, ridge "
            "optima, exact confirmed exports, complete recurrence traces, masks, sklearn "
            "metrics, matched V5 comparator, lazy routing and participant summaries/gates. "
            "Reuses previously passed source receipt; separate-code same-author arithmetic "
            "verification."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selection", type=Path)
    mode.add_argument("--assessment", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Retain audit attempts; output must be fresh")
    result = (
        audit_selection(args.selection) if args.selection else audit_assessment(args.assessment)
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
