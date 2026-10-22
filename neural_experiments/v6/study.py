"""Prospective exact-model confirmation, strict-past budget routing and fresh audit."""

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
from neural_experiments.v5 import policy as old_policy
from neural_experiments.v5 import study as old
from neural_experiments.v6 import data, metrics, model, policy

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "neural_experiments/v6/plan.json"
PLAN_SHA = "f8c075f99c51b6da1bfe07a57861746fff158ec2fd518ec21ecf45f10b170b7a"
SOURCE_PATHS = tuple(
    f"neural_experiments/{version}/{name}.py"
    for version, names in (
        ("v6", ("model", "policy", "study", "data", "metrics")),
        ("v5", ("model", "policy", "study", "data", "metrics")),
        ("v4", ("model", "study", "data")),
        ("v3", ("study",)),
        ("v2", ("data",)),
    )
    for name in names
)
BUDGETS = (0, 60, 120, 240, 360)
REQUESTS = (60, 120, 180, 240, 300, 360, 420)
OLD_SELECTION = ROOT / "results/neural_v5/source/selection.json"
OLD_SELECTION_SHA = "c9bc13df6eda97efd1e26cbd050993101aca2589d5da5d4d029e522ffcd7a044"


def identity(plan_path, index_path):
    if digest(plan_path) != PLAN_SHA:
        raise ValueError("Unrecognized locked V6 study plan")
    if digest(OLD_SELECTION) != OLD_SELECTION_SHA:
        raise ValueError("Frozen V5 threshold selection changed")
    return {
        "schema_version": 1,
        "plan": {"path": relative(plan_path), "sha256": digest(plan_path)},
        "development_index": {"path": relative(index_path), "sha256": digest(index_path)},
        "old_selection": {"path": relative(OLD_SELECTION), "sha256": OLD_SELECTION_SHA},
        "code_sha256": {p: digest(ROOT / p) for p in SOURCE_PATHS},
    }


def load_case(index_path, item, labels=False):
    base = Path(index_path).resolve().parent
    path = base / item["feature_path"]
    if digest(path) != item["feature_sha256"]:
        raise ValueError("Feature identity mismatch")
    result = {
        "item": item,
        "features": data.load_features(path),
        "segments": [(s["start"], s["end"]) for s in item["segments"] if s["end"] > s["start"]],
    }
    if labels:
        path = base / item["target_path"]
        if digest(path) != item["target_sha256"]:
            raise ValueError("Target identity mismatch")
        result.update(data.load_targets(path, item["n_bins"]))
    return result


class PrefixLabels:
    """Return bounded views; newly observed requests grow monotonically."""

    def __init__(self, index_path, item):
        self.item = item
        self.path = Path(index_path).resolve().parent / item["target_path"]
        self.requests = []

    def request(self, seconds):
        if seconds not in REQUESTS:
            raise ValueError("Unplanned label request")
        if digest(self.path) != self.item["target_sha256"]:
            raise ValueError("Target identity mismatch")
        if not self.requests or seconds > self.requests[-1]:
            self.requests.append(seconds)
        values = data.load_targets(self.path, self.item["n_bins"])
        stop = min(1500 + 50 * seconds, self.item["n_bins"])
        y = np.zeros((stop, 2), dtype=np.float64)
        mask = np.zeros(stop, dtype=bool)
        y[1500:] = values["y"][1500:stop]
        mask[1500:] = values["eval_mask"][1500:stop]
        return y, mask


def init_output(path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "traces").mkdir()
    return output


def save_fit(output, key, fitted, metadata):
    path = output / "models" / f"{key}.npz"
    fitted.save(path)
    return {**metadata, "model_path": relative(path), "model_sha256": digest(path)}


def save_score(output, key, fitted, case, metadata):
    row = save_fit(output, key, fitted, metadata)
    prediction = core.Decoder.load(ROOT / row["model_path"]).predict(
        case["features"], case["segments"]
    )
    use = case["eval_mask"].copy()
    use[:22600] = False
    use[30000:] = False
    path = output / "traces" / f"{key}.npz"
    np.savez_compressed(path, prediction=prediction, eval_mask=use)
    values = score(case["y"], prediction, use)
    values["summed_mse"] = (
        float(np.square(case["y"][use] - prediction[use]).sum(1).mean())
        if use.sum() >= 50
        else None
    )
    return {
        **row,
        "metrics": values,
        "trace_path": relative(path),
        "trace_sha256": digest(path),
        "eval_mask_sha256": mask_digest(use),
    }


def row_metadata(item, method, requested, action, training=0):
    return {
        "participant": item["participant_id"],
        "session_id": item["session_id"],
        "post_implant_day": item["post_implant_day"],
        "domain": "new_sessions",
        "method": method,
        "action": action,
        "training_label_seconds": training,
        "label_seconds": requested,
        "available_label_seconds": max(0, min(requested, item["n_bins"] * 0.02 - 30)),
    }


def predict_prefix(fitted, case, stop):
    features = {k: v if k == "channel_ids" else v[:stop] for k, v in case["features"].items()}
    segments = [(a, min(b, stop)) for a, b in case["segments"] if a < stop]
    return fitted.predict(features, segments)


class CaseReplay:
    def __init__(self, index_path, item, dev_items, layouts, statistics, prior_rows, output):
        self.index_path, self.item, self.output = index_path, item, output
        self.case = load_case(index_path, item)
        self.broker = PrefixLabels(index_path, item)
        self.candidates = model.candidate_grid()
        self.raw, self.priors, self.history = {}, {}, {}
        self.models, self.stages, self.saved_fits, self.timing = {}, {}, [], []
        for c in self.candidates:
            family = c["config"]["family"]
            if family in self.raw:
                continue
            raw = core.representation(self.case["features"], self.case["segments"], c["config"])
            self.raw[family] = raw
            self.priors[family], self.history[family] = old.prior_for(
                item, dev_items, layouts, statistics[family], raw.shape[1] + 1
            )
        probes = old.historical_evidence(item, dev_items, layouts, prior_rows)
        eligible = [
            p
            for p in probes
            if p["cases"] and any(h["disposition"] == "used" for h in self.history[p["family"]])
        ]
        self.prior_family = (
            min(eligible, key=lambda p: np.mean([r["summed_mse"] for r in p["cases"]]))["family"]
            if eligible
            else "tx45_sbp"
        )
        self.prior_selection = {"family": self.prior_family, "historical_probes": probes}

    def prior_decoder(self, family="tx45_sbp"):
        c = next(c for c in self.candidates if c["config"]["family"] == family)
        raw = self.raw[family]
        if len(raw) < 1500:
            _, mean, scale = core.design(raw, len(raw), self.item["n_channels"])
            return core.Decoder(c["config"], mean, scale, np.zeros_like(self.priors[family]))
        return model.prior_decoder(raw, self.item["n_channels"], c["config"], self.priors[family])

    def zero_decoder(self):
        fitted = self.prior_decoder()
        return core.Decoder(fitted.config, fitted.mean, fitted.scale, np.zeros_like(fitted.weights))

    def fit(self, candidate, stop, y, mask, phase, budget):
        family = candidate["config"]["family"]
        fitted, receipt = model.fit_candidate(
            self.raw[family],
            y,
            mask,
            self.case["segments"],
            stop,
            candidate,
            self.priors[family],
            self.item["n_channels"],
        )
        key = f"{self.item['session_id']}-{candidate['candidate_id']}-b{budget}-{phase}"
        record = save_fit(
            self.output,
            key,
            fitted,
            {
                "participant": self.item["participant_id"],
                "session_id": self.item["session_id"],
                "phase": phase,
                "training_seconds": budget,
                "candidate": candidate,
                "fit": receipt,
                "history": self.history[family],
            },
        )
        self.saved_fits.append(record)
        # Confirmation uses the reloaded, already-exported artifact.
        return core.Decoder.load(ROOT / record["model_path"]), record

    def get_stage(self, budget):
        if budget in self.stages:
            return self.stages[budget]
        if budget not in BUDGETS:
            raise ValueError("Unplanned training stage")
        began = time.perf_counter()
        stop, confirmation_stop = 1500 + 50 * budget, 4500 + 50 * budget
        evidence = {
            "training_seconds": budget,
            "requested_label_seconds": budget + 60,
            "available_label_seconds": max(0, min(budget + 60, self.item["n_bins"] * 0.02 - 30)),
            "decision_time_seconds": min(90 + budget, self.item["n_bins"] * 0.02),
        }
        if self.item["n_bins"] < confirmation_stop:
            self.broker.request(budget + 60)
            evidence["unavailable"] = "complete_fit_and_confirmation_not_observed"
        else:
            if budget == 0:
                fitted = self.prior_decoder(self.prior_family)
                record = save_fit(
                    self.output,
                    f"{self.item['session_id']}-prior-confirm",
                    fitted,
                    {
                        "participant": self.item["participant_id"],
                        "session_id": self.item["session_id"],
                        "phase": "prior_confirmation",
                        "training_seconds": 0,
                        "family": self.prior_family,
                        "history": self.history[self.prior_family],
                        "prior_selection": self.prior_selection,
                    },
                )
                self.saved_fits.append(record)
                fitted = core.Decoder.load(ROOT / record["model_path"])
                evidence["candidate_id"] = self.prior_family + "_prior0"
                evidence["inner_selection"] = {"kind": "strict_past_only", **self.prior_selection}
                training_mean = np.zeros(2)
            else:
                y, use = self.broker.request(budget)
                candidate = next(
                    c
                    for c in self.candidates
                    if c["candidate_id"] == "tx45_sbp_scratch1_fit_available"
                )
                fitted, record = self.fit(candidate, stop, y, use, "frozen_full", budget)
                train_mask = use.copy()
                train_mask[:1500] = False
                for left, right in self.case["segments"]:
                    train_mask[left : min(left + 62, right, stop)] = False
                training_mean = y[train_mask].mean(0) if train_mask.any() else np.zeros(2)
                evidence.update(
                    candidate_id=candidate["candidate_id"],
                    selection_rule="fixed_before_fresh_assessment",
                    fit=record["fit"],
                )
            evidence.update(
                model_sha256=record["model_sha256"],
                model_path=record["model_path"],
                fit_stop_bin=stop,
                confirmation_start_bin=stop + 100,
                confirmation_stop_bin=confirmation_stop,
            )
            # No alternative candidate can see these labels before selection/export.
            y, use = self.broker.request(budget + 60)
            prediction = predict_prefix(fitted, self.case, confirmation_stop)
            evidence["confirmation"] = policy.confirmation_evidence(
                y,
                use,
                self.case["segments"],
                stop + 100,
                confirmation_stop,
                prediction,
                training_mean,
            )
            self.models[budget] = fitted
        self.stages[budget] = evidence
        self.timing.append(
            {"training_seconds": budget, "stage_build_seconds": time.perf_counter() - began}
        )
        return evidence

    def decoder_for(self, decision):
        if decision["action"] == "abstain":
            return self.zero_decoder()
        b = decision["training_seconds"]
        stage = self.stages[b]
        if decision["model_sha256"] != stage["model_sha256"]:
            raise ValueError("Deployed model differs from confirmed snapshot")
        if digest(ROOT / stage["model_path"]) != stage["model_sha256"]:
            raise ValueError("Confirmed model changed")
        return core.Decoder.load(ROOT / stage["model_path"])

    def controls(self):
        values = {"zero": self.zero_decoder(), "history0": self.prior_decoder()}
        candidate = next(
            c
            for c in self.candidates
            if c["config"]["family"] == "tx45_sbp"
            and c["head"] == "scratch1"
            and c["normalization"] == "fit_available"
        )
        for budget in (120, 240, 360):
            if self.item["n_bins"] <= 1500:
                values[f"daily{budget}"] = self.zero_decoder()
                continue
            y, use = PrefixLabels(self.index_path, self.item).request(budget)
            stop = min(1500 + budget * 50, self.item["n_bins"])
            values[f"daily{budget}"] = self.fit(candidate, stop, y, use, "control", budget)[0]
        return values


def history_evidence(item, items, layouts, control_rows):
    curves = []
    for old_item in histories(item, items, layouts):
        budgets = {}
        for b in (120, 240, 360):
            r = control_rows[old_item["session_id"], f"daily{b}"]
            m = r["metrics"]
            budgets[str(b)] = {
                "r2": m["r2"],
                "n": m["n"],
                "coordinate_mse": m["summed_mse"] / 2 if m["summed_mse"] is not None else None,
                "model_sha256": r["model_sha256"],
                "trace_sha256": r["trace_sha256"],
            }
        curves.append(
            {
                "session_id": old_item["session_id"],
                "post_implant_day": old_item["post_implant_day"],
                "budgets": budgets,
            }
        )
    return {
        "current_session_id": item["session_id"],
        "current_post_implant_day": item["post_implant_day"],
        "past_learning_curves": curves,
    }


def old_comparator(
    index_path, item, dev_items, layouts, statistics, prior_rows, output, old_selection
):
    if item["n_bins"] < 1500:
        replay = CaseReplay(index_path, item, dev_items, layouts, statistics, prior_rows, output)
        return replay.zero_decoder(), {
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
    replay = old.CaseReplay(
        index_path, item, dev_items, layouts, statistics, output, "new_sessions"
    )
    evidence = old.quality_evidence(item, dev_items, layouts, prior_rows, replay)
    selected = next(
        c["policy_id"]
        for c in old_selection["choices"]
        if c["target_participant"] == item["participant_id"]
    )
    config = next(c for c in old_selection["policies"] if c["policy_id"] == selected)
    decision = old_policy.decide(evidence, replay.get_stage, config)
    fitted = replay.decoder_for(decision)
    return fitted, {
        "history_evidence": evidence,
        "stages": list(replay.stages.values()),
        "decision": decision,
        "policy_id": selected,
        "fits": replay.saved_fits,
        "label_requests": list(replay.broker.requests),
    }


def context(index_path, index, output):
    layouts = {
        s["session_id"]: tuple(load_case(index_path, s)["features"]["channel_ids"])
        for s in index["sessions"]
    }
    stats = {}
    for family in ("tx45", "tx45_sbp"):
        c = next(c for c in model.candidate_grid() if c["config"]["family"] == family)
        stats[family] = old.history_statistics(index_path, index["sessions"], c["config"])
    # Frozen V5 probes keep their original [272,600) reliability window.
    prior_rows = old.source_prior_rows(index_path, index, layouts, stats, output)
    return layouts, stats, prior_rows


def select(plan_path, index_path, output):
    began = time.perf_counter()
    ident = identity(plan_path, index_path)
    plan = json.loads(Path(plan_path).read_text())
    if ident["development_index"]["sha256"] != plan["development_index_sha256"]:
        raise ValueError("Development index differs from locked plan")
    index = data.load_index(index_path, expected_role="development")
    output = init_output(output)
    write_json(
        output / "manifest.json",
        {**ident, "stage": "source", "status": "running", "fresh_labels_opened": False},
    )
    layouts, statistics, prior_rows = context(index_path, index, output)
    old_selection = json.loads(OLD_SELECTION.read_text())
    control_lookup, evidence, fits = {}, [], []
    for item in index["sessions"]:
        replay = CaseReplay(
            index_path, item, index["sessions"], layouts, statistics, prior_rows, output
        )
        for b in BUDGETS:
            replay.get_stage(b)
        scored = load_case(index_path, item, True)
        controls = replay.controls()
        old_fitted, old_evidence = old_comparator(
            index_path,
            item,
            index["sessions"],
            layouts,
            statistics,
            prior_rows,
            output,
            old_selection,
        )
        controls["v5_adaptive"] = old_fitted
        case_controls = []
        for name, fitted in controls.items():
            b = {
                "zero": 0,
                "history0": 0,
                "daily120": 120,
                "daily240": 240,
                "daily360": 360,
                "v5_adaptive": old_evidence["decision"]["budget_seconds"],
            }[name]
            action = (
                old_evidence["decision"]["action"]
                if name == "v5_adaptive"
                else "abstain"
                if name == "zero"
                else "accept"
            )
            row = save_score(
                output,
                f"source-{item['session_id']}-{name}",
                fitted,
                scored,
                row_metadata(item, name, b, action, b),
            )
            control_lookup[item["session_id"], name] = row
            case_controls.append(row)
        evidence.append(
            {
                "participant": item["participant_id"],
                "session_id": item["session_id"],
                "stages": list(replay.stages.values()),
                "controls": case_controls,
                "v5_comparator": old_evidence,
                "timing": replay.timing,
                "zero_model_path": control_lookup[item["session_id"], "zero"]["model_path"],
            }
        )
        fits.extend(replay.saved_fits)
        print(f"source evidence {item['session_id']} complete", flush=True)
    rows = []
    configs = policy.policy_grid()
    for item, ev in zip(index["sessions"], evidence, strict=True):
        history = history_evidence(item, index["sessions"], layouts, control_lookup)
        ev["history_evidence"] = history
        stages = {r["training_seconds"]: r for r in ev["stages"]}
        scored = load_case(index_path, item, True)
        for config in configs:
            decision = policy.decide(history, stages.__getitem__, config)
            model_path = (
                ev["zero_model_path"]
                if decision["action"] == "abstain"
                else stages[decision["training_seconds"]]["model_path"]
            )
            fitted = core.Decoder.load(ROOT / model_path)
            zero = control_lookup[item["session_id"], "zero"]["metrics"]["summed_mse"]
            metadata = {
                **row_metadata(
                    item,
                    "confirmed",
                    decision["requested_label_seconds"],
                    decision["action"],
                    decision["training_seconds"],
                ),
                "decision": decision,
                "policy_id": config["policy_id"],
                "zero_mse": zero / 2 if zero is not None else None,
                "daily240_r2": control_lookup[item["session_id"], "daily240"]["metrics"]["r2"],
            }
            row = save_score(
                output,
                f"source-{item['session_id']}-{config['policy_id']}",
                fitted,
                scored,
                metadata,
            )
            if decision["action"] == "accept" and row["model_sha256"] != decision["model_sha256"]:
                raise ValueError("Source deployment changed confirmed model bytes")
            rows.append(row)
    artifact = {
        **ident,
        "stage": "source",
        "status": "complete",
        "fresh_labels_opened": False,
        "policies": configs,
        "candidates": model.candidate_grid(),
        "cases": rows,
        "evidence": evidence,
        "fits": fits,
        "history_probes": list(prior_rows.values()),
        "choices": policy.choose_policy(rows, configs, plan["participants"]),
        "elapsed_seconds": time.perf_counter() - began,
    }
    write_json(output / "selection.json", artifact)
    write_json(
        output / "manifest.json",
        {
            **ident,
            "stage": "source",
            "status": "complete",
            "selection_sha256": digest(output / "selection.json"),
            "fresh_labels_opened": False,
        },
    )
    return artifact


def assess(selection_path, audit_path, index_path, output):
    selection = json.loads(Path(selection_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("status") != "PASS" or audit.get("selection_sha256") != digest(selection_path):
        raise ValueError("Passed exact source audit required")
    for path, expected in selection["code_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen execution source changed: {path}")
    dev_path = ROOT / selection["development_index"]["path"]
    ident = identity(ROOT / selection["plan"]["path"], dev_path)
    if any(selection[k] != ident[k] for k in ident):
        raise ValueError("Source identity changed")
    dev = data.load_index(dev_path, expected_role="development")
    index = data.load_index(index_path, expected_role="new_sessions")
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
        "index": {"path": relative(index_path), "sha256": digest(index_path)},
        "cases": [],
        "evidence": [],
        "fits": [],
        "data_dispositions": index["participant_dispositions"],
    }
    write_json(output / "manifest.json", artifact)
    layouts = {
        s["session_id"]: tuple(load_case(dev_path, s)["features"]["channel_ids"])
        for s in dev["sessions"]
    }
    statistics = {}
    for family in ("tx45", "tx45_sbp"):
        c = next(c for c in model.candidate_grid() if c["config"]["family"] == family)
        statistics[family] = old.history_statistics(dev_path, dev["sessions"], c["config"])
    prior_rows = {(r["session_id"], r["family"]): r for r in selection["history_probes"]}
    control_lookup = {
        (r["session_id"], r["method"]): r for ev in selection["evidence"] for r in ev["controls"]
    }
    old_selection = json.loads(OLD_SELECTION.read_text())
    try:
        for item in index["sessions"]:
            layouts[item["session_id"]] = tuple(
                load_case(index_path, item)["features"]["channel_ids"]
            )
            replay = CaseReplay(
                index_path, item, dev["sessions"], layouts, statistics, prior_rows, output
            )
            history = history_evidence(item, dev["sessions"], layouts, control_lookup)
            selected = next(
                c["policy_id"]
                for c in selection["choices"]
                if c["target_participant"] == item["participant_id"]
            )
            config = next(c for c in selection["policies"] if c["policy_id"] == selected)
            artifact["fresh_labels_opened"] = True
            write_json(output / "manifest.json", artifact)
            decision = policy.decide(history, replay.get_stage, config)
            fitted = replay.decoder_for(decision)
            requests = list(replay.broker.requests)
            # All controller decisions finish before the scorer or comparator opens labels.
            scored = load_case(index_path, item, True)
            row = save_score(
                output,
                f"new-{item['session_id']}-confirmed",
                fitted,
                scored,
                {
                    **row_metadata(
                        item,
                        "confirmed",
                        decision["requested_label_seconds"],
                        decision["action"],
                        decision["training_seconds"],
                    ),
                    "decision": decision,
                    "policy_id": selected,
                    "label_requests": requests,
                },
            )
            if decision["action"] == "accept" and row["model_sha256"] != decision["model_sha256"]:
                raise ValueError("Assessed model differs from confirmed numeric export")
            artifact["cases"].append(row)
            controls = replay.controls()
            old_fitted, old_evidence = old_comparator(
                index_path,
                item,
                dev["sessions"],
                layouts,
                statistics,
                prior_rows,
                output,
                old_selection,
            )
            controls["v5_adaptive"] = old_fitted
            for name, control in controls.items():
                b = {
                    "zero": 0,
                    "history0": 0,
                    "daily120": 120,
                    "daily240": 240,
                    "daily360": 360,
                    "v5_adaptive": old_evidence["decision"]["budget_seconds"],
                }[name]
                action = (
                    old_evidence["decision"]["action"]
                    if name == "v5_adaptive"
                    else "abstain"
                    if name == "zero"
                    else "accept"
                )
                artifact["cases"].append(
                    save_score(
                        output,
                        f"new-{item['session_id']}-{name}",
                        control,
                        scored,
                        row_metadata(item, name, b, action, b),
                    )
                )
            artifact["evidence"].append(
                {
                    "participant": item["participant_id"],
                    "session_id": item["session_id"],
                    "history_evidence": history,
                    "stages": list(replay.stages.values()),
                    "v5_comparator": old_evidence,
                    "timing": replay.timing,
                }
            )
            artifact["fits"].extend(replay.saved_fits)
            write_json(output / "manifest.json", artifact)
            print(f"assessment {item['session_id']} complete", flush=True)
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
    p.add_argument("command", choices=("select", "assess"))
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument(
        "--development-index", type=Path, default=ROOT / "data/neural_v6/development/index.json"
    )
    p.add_argument("--selection", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--new-index", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.command == "select":
            select(a.plan, a.development_index, a.output)
        else:
            assess(a.selection, a.audit, a.new_index, a.output)


if __name__ == "__main__":
    main()
