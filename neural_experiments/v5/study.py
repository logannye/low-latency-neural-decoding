"""Frozen-source policy selection and a one-shot prospective calibration replay.

The controller receives only historical diagnostics and requested prefix evidence.
Scorer labels and counterfactual controls are kept outside its decision interface.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_experiments.v3.study import digest, mask_digest, relative, score, write_json
from neural_experiments.v4 import model as core
from neural_experiments.v4.study import histories
from neural_experiments.v5 import data, metrics, model, policy

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "neural_experiments/v5/plan.json"
PLAN_SHA = "87c417f915259e7f77b7999217d3a2461884fffd175c676188e20c346472390e"
SOURCE_PATHS = (
    "neural_experiments/v5/model.py",
    "neural_experiments/v5/policy.py",
    "neural_experiments/v5/study.py",
    "neural_experiments/v5/data.py",
    "neural_experiments/v5/metrics.py",
    "neural_experiments/v4/model.py",
    "neural_experiments/v4/study.py",
    "neural_experiments/v4/data.py",
    "neural_experiments/v3/study.py",
    "neural_experiments/v2/data.py",
)


def identity(plan_path, index_path):
    if digest(plan_path) != PLAN_SHA:
        raise ValueError("Unrecognized locked V5 study plan")
    return {
        "schema_version": 1,
        "plan": {"path": relative(plan_path), "sha256": digest(plan_path)},
        "development_index": {"path": relative(index_path), "sha256": digest(index_path)},
        "code_sha256": {p: digest(ROOT / p) for p in SOURCE_PATHS},
    }


def load_case(index_path, item, labels=False):
    base = Path(index_path).resolve().parent
    feature_path = base / item["feature_path"]
    if digest(feature_path) != item["feature_sha256"]:
        raise ValueError("Feature identity mismatch")
    case = {
        "item": item,
        "features": data.load_features(feature_path),
        "segments": [(s["start"], s["end"]) for s in item["segments"] if s["end"] > s["start"]],
    }
    if labels:
        path = base / item["target_path"]
        if digest(path) != item["target_sha256"]:
            raise ValueError("Target identity mismatch")
        case.update(data.load_targets(path, item["n_bins"]))
    return case


class PrefixLabels:
    """A narrow label broker: returns only requested prospective label intervals."""

    def __init__(self, index_path, item):
        self.item = item
        self.path = (
            Path(item["known_prefix_target_path"])
            if "known_prefix_target_path" in item
            else Path(index_path).resolve().parent / item["target_path"]
        )
        self.expected = item.get("known_prefix_target_sha256", item["target_sha256"])
        self.n = item.get("known_prefix_n_bins", item["n_bins"])
        self.requests = []

    def request(self, budget):
        if budget not in (30, 60, 120, 240):
            raise ValueError("Unplanned label request")
        if self.requests and budget <= self.requests[-1]:
            raise ValueError("Label requests must increase")
        if digest(self.path) != self.expected:
            raise ValueError("Prefix target identity mismatch")
        values = data.load_targets(self.path, self.n)
        stop = min(1500 + 50 * budget, self.n, self.item["n_bins"])
        y = np.zeros((stop, 2), dtype=np.float64)
        mask = np.zeros(stop, dtype=bool)
        y[1500:] = values["y"][1500:stop]
        mask[1500:] = values["eval_mask"][1500:stop]
        self.requests.append(budget)
        return y, mask


def init_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path / "models").mkdir()
    (path / "traces").mkdir()
    return path


def evaluation_mask(case, domain):
    mask = case["eval_mask"].copy()
    if domain == "new_sessions":
        mask[:13600] = False
        mask[30000:] = False
    return mask


def save_fit(output, key, fitted, metadata):
    path = output / "models" / f"{key}.npz"
    fitted.save(path)
    return {**metadata, "model_path": relative(path), "model_sha256": digest(path)}


def save_score(output, key, fitted, case, domain, metadata):
    row = save_fit(output, key, fitted, metadata)
    # Reload the actual serialized numeric model before feature-only prediction.
    prediction = core.Decoder.load(ROOT / row["model_path"]).predict(
        case["features"], case["segments"]
    )
    mask = evaluation_mask(case, domain)
    path = output / "traces" / f"{key}.npz"
    np.savez_compressed(path, prediction=prediction, eval_mask=mask)
    values = score(case["y"], prediction, mask)
    values["summed_mse"] = (
        float(np.square(case["y"][mask] - prediction[mask]).sum(1).mean())
        if mask.sum() >= 50
        else None
    )
    return {
        **row,
        "metrics": values,
        "trace_path": relative(path),
        "trace_sha256": digest(path),
        "eval_mask_sha256": mask_digest(mask),
    }


def history_statistics(index_path, items, config):
    cache = {}
    for item in items:
        case = load_case(index_path, item, True)
        raw = core.representation(case["features"], case["segments"], config)
        cache[item["session_id"]] = model.historical_stats(
            raw, case["y"], case["eval_mask"], case["segments"], item["n_channels"]
        )
    return cache


def prior_for(item, items, layouts, statistics, dimension):
    selected = histories(item, items, layouts)
    valid, receipt = [], []
    for old in selected:
        st = statistics[old["session_id"]]
        ok = st["n"] >= 50
        receipt.append(
            {
                "session_id": old["session_id"],
                "post_implant_day": old["post_implant_day"],
                "training_rows": st["n"],
                "disposition": "used" if ok else "insufficient_labels",
            }
        )
        if ok:
            valid.append(st)
    return core.solve_history(valid, dimension), receipt


def historical_evidence(item, items, layouts, prior_rows):
    """Past out-of-session predictions, never history-training fit accuracy."""
    selected = histories(item, items, layouts)
    result = []
    for family in ("tx45", "tx45_sbp"):
        rows = []
        for old in selected:
            r = prior_rows[(old["session_id"], family)]
            if r["metrics"]["summed_mse"] is not None and any(
                h["disposition"] == "used" for h in r["history"]
            ):
                rows.append(
                    {
                        "session_id": old["session_id"],
                        "post_implant_day": old["post_implant_day"],
                        "n": r["metrics"]["n"],
                        "summed_mse": r["metrics"]["summed_mse"],
                    }
                )
        result.append({"family": family, "cases": rows})
    return result


def choose_policy(rows, configurations, people):
    """Whole target-person exclusion; source score then predetermined cost tradeoff."""
    choices = []
    for target in people:
        ranking = []
        for config in configurations:
            by_person = []
            for source in people:
                if source == target:
                    continue
                rr = [
                    r
                    for r in rows
                    if r["participant"] == source and r["policy_id"] == config["policy_id"]
                ]
                eligible = [r for r in rr if r["metrics"]["r2"] is not None]
                if eligible:
                    by_person.append(
                        {
                            "participant": source,
                            "r2": float(np.mean([r["metrics"]["r2"] for r in eligible])),
                            "label_seconds": float(np.mean([r["label_seconds"] for r in rr])),
                            "daily120_r2": float(np.mean([r["daily120_r2"] for r in eligible])),
                        }
                    )
            r2 = float(np.mean([r["r2"] for r in by_person]))
            cost = float(np.mean([r["label_seconds"] for r in by_person]))
            baseline = float(np.mean([r["daily120_r2"] for r in by_person]))
            feasible = r2 >= baseline - 0.03
            ranking.append(
                {
                    "policy_id": config["policy_id"],
                    "mean_r2": r2,
                    "mean_label_seconds": cost,
                    "daily120_mean_r2": baseline,
                    "feasible": feasible,
                    "objective": r2 - 0.05 * cost / 240,
                    "eligible_source_people": len(by_person),
                }
            )
        feasible = [r for r in ranking if r["feasible"]]
        selected = (
            min(feasible, key=lambda r: (r["mean_label_seconds"], -r["mean_r2"]))
            if feasible
            else max(ranking, key=lambda r: r["objective"])
        )
        choices.append(
            {
                "target_participant": target,
                "policy_id": selected["policy_id"],
                "selection_rule": "minimum_cost_within_0.03_daily120"
                if feasible
                else "fallback_r2_minus_0.05_cost_fraction",
                "ranking": ranking,
            }
        )
    return choices


class CaseReplay:
    """A query's feature cache and narrow, sequential label broker."""

    def __init__(self, index_path, item, dev_items, layouts, statistics, output, domain):
        self.case = load_case(index_path, item)
        self.item = item
        self.index_path = index_path
        self.domain = domain
        self.output = output
        self.broker = PrefixLabels(index_path, item)
        self.candidates = model.candidate_grid()
        self.raw = {}
        self.designs = {}
        self.priors = {}
        self.history = {}
        self.stage_fits = {}
        self.stages = {}
        self.saved_fits = []
        self.timing = []
        for c in self.candidates:
            family = c["config"]["family"]
            if family in self.raw:
                continue
            raw = core.representation(self.case["features"], self.case["segments"], c["config"])
            self.raw[family] = raw
            if len(raw) < 1500:
                continue
            self.designs[family] = model.design(raw, item["n_channels"])
            prior, receipt = prior_for(
                item, dev_items, layouts, statistics[family], raw.shape[1] + 1
            )
            self.priors[family] = prior
            self.history[family] = receipt

    def fit(self, candidate, budget, y, mask, phase="full", daily_control=False):
        family = candidate["config"]["family"]
        requested = 1500 + 50 * budget
        stop = 1500 + 25 * budget if phase == "validation" else requested
        n = self.item["n_bins"]
        padded_y = np.zeros((n, 2), dtype=np.float64)
        padded_mask = np.zeros(n, dtype=bool)
        padded_y[: len(y)], padded_mask[: len(mask)] = y, mask
        z, mean, scale = (
            model.daily_control_design(self.raw[family], self.item["n_channels"], stop)
            if daily_control
            else self.designs[family]
        )
        normalizer = min(stop, n) if daily_control else 1500
        weights, receipt = model.fit_stage(
            z,
            padded_y,
            padded_mask,
            self.case["segments"],
            stop,
            candidate,
            self.priors[family],
            normalization_bins=normalizer,
        )
        fitted = core.Decoder(candidate["config"], mean, scale, weights)
        key = f"{self.item['session_id']}-{candidate['candidate_id']}-b{budget}-{phase}" + (
            "-control" if daily_control else ""
        )
        row = save_fit(
            self.output,
            key,
            fitted,
            {
                "participant": self.item["participant_id"],
                "session_id": self.item["session_id"],
                "post_implant_day": self.item["post_implant_day"],
                "family": family,
                "candidate_id": candidate["candidate_id"],
                "phase": phase,
                "budget_seconds": budget,
                "daily_control": daily_control,
                "fit": receipt,
                "history": self.history[family],
            },
        )
        self.saved_fits.append(row)
        return fitted, row

    def get_stage(self, budget):
        if budget in self.stages:
            return self.stages[budget]
        began = time.perf_counter()
        y, mask = self.broker.request(budget)
        stop = 1500 + 50 * budget
        if len(y) < stop:
            evidence = {
                "budget_seconds": budget,
                "unavailable": "requested_checkpoint_not_observed",
                "available_label_seconds": max(0, (len(y) - 1500) * 0.02),
            }
        else:
            predictions = {}
            for c in self.candidates:
                fitted, row = self.fit(c, budget, y, mask, "validation")
                z = self.designs[c["config"]["family"]][0][:stop]
                predictions[c["candidate_id"]] = core.project_direction(z @ fitted.weights)
                self.stage_fits[(budget, c["candidate_id"], "validation")] = (fitted, row)
            evidence = policy.stage_evidence(
                y,
                mask,
                self.case["segments"],
                1500 + 25 * budget,
                1600 + 25 * budget,
                stop,
                predictions,
            )
        self.stages[budget] = evidence
        self.timing.append(
            {"budget_seconds": budget, "validation_build_seconds": time.perf_counter() - began}
        )
        return evidence

    def full_fit(self, candidate_id, budget):
        key = (budget, candidate_id, "full")
        if key in self.stage_fits:
            return self.stage_fits[key]
        # Separate replay broker for final refit; only the same allowed interval.
        y, mask = PrefixLabels(self.index_path, self.item).request(budget)
        candidate = next(c for c in self.candidates if c["candidate_id"] == candidate_id)
        value = self.fit(candidate, budget, y, mask)
        self.stage_fits[key] = value
        return value

    def prior_decoder(self, family):
        c = next(c for c in self.candidates if c["config"]["family"] == family)
        _, mean, scale = self.designs[family]
        return core.Decoder(c["config"], mean, scale, self.priors[family])

    def zero_decoder(self):
        c = self.candidates[3]
        _, mean, scale = self.designs["tx45_sbp"]
        return core.Decoder(c["config"], mean, scale, np.zeros_like(self.priors["tx45_sbp"]))

    def decoder_for(self, decision):
        if decision["action"] == "abstain":
            return self.zero_decoder()
        if decision["budget_seconds"] == 0:
            return self.prior_decoder(decision["candidate_id"].removesuffix("_prior10"))
        return self.full_fit(decision["candidate_id"], decision["budget_seconds"])[0]

    def controls(self):
        controls = {"zero": self.zero_decoder(), "history0": self.prior_decoder("tx45_sbp")}
        controls["history30"] = self.full_fit("tx45_sbp_prior10", 30)[0]
        c = next(c for c in self.candidates if c["candidate_id"] == "tx45_sbp_scratch1")
        for budget in (120, 240):
            y, mask = PrefixLabels(self.index_path, self.item).request(budget)
            controls[f"daily{budget}"] = self.fit(c, budget, y, mask, daily_control=True)[0]
        return controls


def quality_evidence(item, items, layouts, prior_rows, replay):
    probes = historical_evidence(item, items, layouts, prior_rows)
    return {
        "current_session_id": item["session_id"],
        "current_post_implant_day": item["post_implant_day"],
        "warmup_available": item["n_bins"] >= 1500,
        "prior_candidates": [
            {
                "candidate_id": p["family"] + "_prior10",
                "prior_available": any(
                    h["disposition"] == "used" for h in replay.history[p["family"]]
                ),
                "probes": [
                    {
                        "session_id": r["session_id"],
                        "post_implant_day": r["post_implant_day"],
                        "n": r["n"],
                        "mse": r["summed_mse"] / 2,
                    }
                    for r in p["cases"]
                ],
            }
            for p in probes
        ],
    }


def row_metadata(item, domain, method, budget, action, candidate_id=None):
    return {
        "participant": item["participant_id"],
        "session_id": item["session_id"],
        "post_implant_day": item["post_implant_day"],
        "domain": domain,
        "method": method,
        "label_seconds": budget,
        "available_label_seconds": max(0, min(budget, item["n_bins"] * 0.02 - 30)),
        "action": action,
        "candidate_id": candidate_id,
    }


def source_prior_rows(index_path, index, layouts, statistics, output):
    rows = {}
    for item in index["sessions"]:
        replay = CaseReplay(
            index_path, item, index["sessions"], layouts, statistics, output, "new_sessions"
        )
        scored = load_case(index_path, item, True)
        for family in ("tx45", "tx45_sbp"):
            fitted = replay.prior_decoder(family)
            row = save_score(
                output,
                f"probe-{item['session_id']}-{family}",
                fitted,
                scored,
                "new_sessions",
                {
                    "participant": item["participant_id"],
                    "session_id": item["session_id"],
                    "post_implant_day": item["post_implant_day"],
                    "family": family,
                    "phase": "history_probe",
                    "history": replay.history[family],
                    "normalization_bins": 1500,
                },
            )
            rows[(item["session_id"], family)] = row
    return rows


def select(plan_path, index_path, output):
    began = time.perf_counter()
    plan = json.loads(Path(plan_path).read_text())
    identity_value = identity(plan_path, index_path)
    if identity_value["development_index"]["sha256"] != plan["development_index_sha256"]:
        raise ValueError("Development index differs from locked plan")
    index = data.load_index(index_path, expected_role="development")
    output = init_output(output)
    write_json(
        output / "manifest.json",
        {**identity_value, "stage": "source", "status": "running", "fresh_labels_opened": False},
    )
    layouts = {
        s["session_id"]: tuple(load_case(index_path, s)["features"]["channel_ids"])
        for s in index["sessions"]
    }
    statistics = {
        c["config"]["family"]: history_statistics(index_path, index["sessions"], c["config"])
        for c in model.candidate_grid()[::3]
    }
    prior_rows = source_prior_rows(index_path, index, layouts, statistics, output)
    configs = policy.policy_grid()
    rows, case_evidence, fits = [], [], []
    for item in index["sessions"]:
        replay = CaseReplay(
            index_path, item, index["sessions"], layouts, statistics, output, "new_sessions"
        )
        evidence = quality_evidence(item, index["sessions"], layouts, prior_rows, replay)
        scored = load_case(index_path, item, True)
        controls = replay.controls()
        control_rows = {
            name: save_score(
                output,
                f"source-{item['session_id']}-{name}",
                fitted,
                scored,
                "new_sessions",
                row_metadata(
                    item,
                    "new_sessions",
                    name,
                    {"zero": 0, "history0": 0, "history30": 30, "daily120": 120, "daily240": 240}[
                        name
                    ],
                    "abstain" if name == "zero" else "accept",
                ),
            )
            for name, fitted in controls.items()
        }
        for config in configs:
            decision = policy.decide(evidence, replay.get_stage, config)
            fitted = replay.decoder_for(decision)
            metadata = {
                **row_metadata(
                    item,
                    "new_sessions",
                    "adaptive",
                    decision["budget_seconds"],
                    decision["action"],
                    decision["candidate_id"],
                ),
                "policy_id": config["policy_id"],
                "decision": decision,
                "daily120_r2": control_rows["daily120"]["metrics"]["r2"],
            }
            rows.append(
                save_score(
                    output,
                    f"source-{item['session_id']}-{config['policy_id']}",
                    fitted,
                    scored,
                    "new_sessions",
                    metadata,
                )
            )
        case_evidence.append(
            {
                "participant": item["participant_id"],
                "session_id": item["session_id"],
                "history_evidence": evidence,
                "stages": list(replay.stages.values()),
                "controls": list(control_rows.values()),
                "timing": replay.timing,
            }
        )
        fits.extend(replay.saved_fits)
        print(f"source {item['session_id']} complete", flush=True)
    artifact = {
        **identity_value,
        "stage": "source",
        "status": "complete",
        "fresh_labels_opened": False,
        "candidates": model.candidate_grid(),
        "policies": configs,
        "cases": rows,
        "evidence": case_evidence,
        "fits": fits,
        "history_probes": list(prior_rows.values()),
        "choices": choose_policy(rows, configs, plan["participants"]),
        "elapsed_seconds": time.perf_counter() - began,
    }
    write_json(output / "selection.json", artifact)
    write_json(
        output / "manifest.json",
        {
            **identity_value,
            "stage": "source",
            "status": "complete",
            "selection_sha256": digest(output / "selection.json"),
            "fresh_labels_opened": False,
        },
    )
    return artifact


def assess(selection_path, audit_path, indices, output):
    selection = json.loads(Path(selection_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("status") != "PASS" or audit.get("selection_sha256") != digest(selection_path):
        raise ValueError("Passed exact source audit required")
    for path, sha in selection["code_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Frozen source changed: {path}")
    dev_path = ROOT / selection["development_index"]["path"]
    if identity(ROOT / selection["plan"]["path"], dev_path) != {
        k: selection[k] for k in identity(ROOT / selection["plan"]["path"], dev_path)
    }:
        raise ValueError("Source provenance changed")
    dev = data.load_index(dev_path, expected_role="development")
    output = init_output(output)
    artifact = {
        "schema_version": 1,
        "stage": "assessment",
        "status": "running",
        "fresh_labels_opened": False,
        "selection_path": relative(selection_path),
        "selection_sha256": digest(selection_path),
        "source_audit_path": relative(audit_path),
        "source_audit_sha256": digest(audit_path),
        "indices": {},
        "cases": [],
        "evidence": [],
        "fits": [],
        "data_dispositions": {},
    }
    write_json(output / "manifest.json", artifact)
    layouts = {
        s["session_id"]: tuple(load_case(dev_path, s)["features"]["channel_ids"])
        for s in dev["sessions"]
    }
    statistics = {
        c["config"]["family"]: history_statistics(dev_path, dev["sessions"], c["config"])
        for c in model.candidate_grid()[::3]
    }
    prior_rows = {(r["session_id"], r["family"]): r for r in selection["history_probes"]}
    try:
        for domain, index_path in zip(("new_sessions", "t3_blocks"), indices, strict=True):
            index = data.load_index(index_path, expected_role=domain)
            artifact["indices"][domain] = {
                "path": relative(index_path),
                "sha256": digest(index_path),
            }
            artifact["data_dispositions"][domain] = index["participant_dispositions"]
            for item in index["sessions"]:
                # Acquisition-unavailable T3 entries are stored in index dispositions.
                layouts[item["session_id"]] = tuple(
                    load_case(index_path, item)["features"]["channel_ids"]
                )
                replay = CaseReplay(
                    index_path, item, dev["sessions"], layouts, statistics, output, domain
                )
                evidence = quality_evidence(item, dev["sessions"], layouts, prior_rows, replay)
                selected = next(
                    c["policy_id"]
                    for c in selection["choices"]
                    if c["target_participant"] == item["participant_id"]
                )
                config = next(c for c in selection["policies"] if c["policy_id"] == selected)
                artifact["fresh_labels_opened"] = True
                write_json(output / "manifest.json", artifact)
                decision = policy.decide(evidence, replay.get_stage, config)
                fitted = replay.decoder_for(decision)
                # The policy has stopped before scorer targets/counterfactual controls load.
                adaptive_requests = list(replay.broker.requests)
                scored = load_case(index_path, item, True)
                metadata = {
                    **row_metadata(
                        item,
                        domain,
                        "adaptive",
                        decision["budget_seconds"],
                        decision["action"],
                        decision["candidate_id"],
                    ),
                    "policy_id": selected,
                    "decision": decision,
                    "label_requests": adaptive_requests,
                }
                artifact["cases"].append(
                    save_score(
                        output,
                        f"{domain}-{item['session_id']}-adaptive",
                        fitted,
                        scored,
                        domain,
                        metadata,
                    )
                )
                for name, control in replay.controls().items():
                    budget = {
                        "zero": 0,
                        "history0": 0,
                        "history30": 30,
                        "daily120": 120,
                        "daily240": 240,
                    }[name]
                    metadata = row_metadata(
                        item, domain, name, budget, "abstain" if name == "zero" else "accept"
                    )
                    artifact["cases"].append(
                        save_score(
                            output,
                            f"{domain}-{item['session_id']}-{name}",
                            control,
                            scored,
                            domain,
                            metadata,
                        )
                    )
                artifact["evidence"].append(
                    {
                        "domain": domain,
                        "participant": item["participant_id"],
                        "session_id": item["session_id"],
                        "history_evidence": evidence,
                        "stages": list(replay.stages.values()),
                        "timing": replay.timing,
                    }
                )
                artifact["fits"].extend(replay.saved_fits)
                write_json(output / "manifest.json", artifact)
                print(f"assessment {domain} {item['session_id']} complete", flush=True)
        artifact["summary"] = metrics.summarize(artifact["cases"])
        artifact["status"] = "assessment_complete"
        write_json(output / "assessment.json", artifact)
        write_json(output / "manifest.json", artifact)
        return artifact
    except BaseException as error:
        artifact.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", artifact)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["select", "assess"])
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument(
        "--development-index", type=Path, default=ROOT / "data/neural_v5/development/index.json"
    )
    p.add_argument("--selection", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--new-index", type=Path)
    p.add_argument("--t3-index", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.command == "select":
            select(a.plan, a.development_index, a.output)
        else:
            assess(a.selection, a.audit, [a.new_index, a.t3_index], a.output)


if __name__ == "__main__":
    main()
