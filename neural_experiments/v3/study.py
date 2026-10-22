"""Locked source-only selection, then one assessment on new session members.

The select command cannot open an assessment index. Assess requires a passed,
hash-bound independent source-selection audit before loading new labels.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_experiments.v3 import data, model

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def resolve(path):
    return ROOT / path


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score(y, prediction, mask):
    actual, estimated = np.asarray(y[mask], float), np.asarray(prediction[mask], float)
    if len(actual) < 50:
        return {
            "r2": None,
            "per_output_r2": [None, None],
            "n": len(actual),
            "angular_error_degrees": None,
            "zero_prediction_count": int(
                np.count_nonzero(np.linalg.norm(estimated, axis=1) <= 1e-12)
            ),
        }
    if not np.isfinite(actual).all() or not np.isfinite(estimated).all():
        raise ValueError("nonfinite score input")
    sse = np.square(actual - estimated).sum(0)
    sst = np.square(actual - actual.mean(0)).sum(0)
    norm = np.linalg.norm(estimated, axis=1) * np.linalg.norm(actual, axis=1)
    cosine = np.divide(
        np.sum(actual * estimated, axis=1), norm, out=np.zeros(len(actual)), where=norm > 1e-12
    )
    return {
        "r2": float(1 - sse.sum() / sst.sum()) if sst.sum() > 1e-12 else None,
        "per_output_r2": [
            float(1 - a / b) if b > 1e-12 else None for a, b in zip(sse, sst, strict=True)
        ],
        "n": len(actual),
        "sse": sse.tolist(),
        "sst": sst.tolist(),
        "angular_error_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()),
        "prediction_norm_rms": float(np.sqrt(np.square(estimated).sum(1).mean())),
        "zero_prediction_count": int(np.count_nonzero(np.linalg.norm(estimated, axis=1) <= 1e-12)),
    }


def load_case(index_path, item, labels=True):
    cache = Path(index_path).resolve().parent
    fp, tp = cache / item["feature_path"], cache / item["target_path"]
    if digest(fp) != item["feature_sha256"]:
        raise ValueError("changed feature artifact")
    features = data.load_features(fp)
    segments = [(int(s["start"]), int(s["end"])) for s in item["segments"] if s["end"] > s["start"]]
    case = {
        "metadata": item,
        "tx": features["x_tx"],
        "sbp": features["x_sbp"],
        "segments": segments,
    }
    if labels:
        if digest(tp) != item["target_sha256"]:
            raise ValueError("changed target artifact")
        with np.load(tp, allow_pickle=False) as z:
            case.update(y=z["y"].astype(float), mask=z["eval_mask"].astype(bool))
    return case


def prefix(case, stop):
    if stop > len(case["tx"]):
        raise ValueError("calibration budget exceeds recording duration")
    return {
        **case,
        **{k: case[k][:stop] for k in ("tx", "sbp", "y", "mask")},
        "segments": [(a, min(b, stop)) for a, b in case["segments"] if a < stop],
    }


def candidate_grid(plan):
    configs = []
    grid = plan["grid"]
    for family, tau, baseline, penalty in itertools.product(
        plan["features"]["families"],
        grid["tau_seconds"],
        grid["baseline_tau_seconds"],
        grid["ridge_lambda"],
    ):
        configs.append(
            {
                "candidate_id": f"c{len(configs):02d}",
                "config": {
                    "kind": "regularized",
                    "family": family,
                    "tau_seconds": tau,
                    "baseline_tau_seconds": baseline,
                    "ridge_lambda": penalty,
                },
            }
        )
    return configs


def choose(cases, candidates, plan):
    choices = []
    for person, budget in itertools.product(
        plan["participants"], plan["calibration_budgets_seconds"]
    ):
        source_people = [p for p in plan["participants"] if p != person]
        for method, family in (
            ("tx_daily", "tx"),
            ("tx_sbp_daily", "tx_sbp"),
            ("selected_daily", None),
        ):
            ranking = []
            for candidate in candidates:
                if family is not None and candidate["config"]["family"] != family:
                    continue
                per_person = []
                for source in source_people:
                    values = [
                        r["metrics"]["r2"]
                        for r in cases
                        if r["participant"] == source
                        and r["budget_seconds"] == budget
                        and r["candidate_id"] == candidate["candidate_id"]
                        and r["metrics"]["r2"] is not None
                    ]
                    if values:
                        per_person.append(float(np.mean(values)))
                if not per_person:
                    raise ValueError("no source validation evidence")
                ranking.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "objective": float(np.mean(per_person)),
                        "eligible_source_people": len(per_person),
                    }
                )
            best = max(ranking, key=lambda r: r["objective"])
            choices.append(
                {
                    "target_participant": person,
                    "budget_seconds": budget,
                    "method": method,
                    "source_participants": source_people,
                    **best,
                    "ranking": ranking,
                }
            )
    return choices


def snapshot(output, plan_path, index_path):
    sources = []
    for name in ("model.py", "study.py", "data.py"):
        source = Path(__file__).parent / name
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources.append(
            {"path": relative(source), "snapshot_path": relative(target), "sha256": digest(source)}
        )
    return {
        "schema_version": 1,
        "study": "causal-recalibration-heldout-sessions-v3",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_running",
        "plan": {"path": relative(plan_path), "sha256": digest(plan_path)},
        "source_files": sources,
        "development_index": {"path": relative(index_path), "sha256": digest(index_path)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "models": [],
        "unavailable": [],
    }


def select(plan_path, development_index, output):
    plan = json.loads(Path(plan_path).read_text())
    index = data.load_index(development_index, expected_role="development")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "development_traces").mkdir()
    manifest = snapshot(output, plan_path, development_index)
    write_json(output / "manifest.json", manifest)
    configs = candidate_grid(plan)
    cases, unavailable = [], []
    began = time.perf_counter()
    try:
        for item in index["sessions"]:
            case = load_case(development_index, item)
            stop = min(len(case["tx"]), 30000)
            start = 15100
            if stop <= start or case["mask"][start:stop].sum() < 50:
                unavailable.extend(
                    {
                        "session_id": item["session_id"],
                        "budget_seconds": budget,
                        "reason": "no development suffix",
                    }
                    for budget in plan["calibration_budgets_seconds"]
                )
                continue
            cubes, ledgers = {}, {}
            for budget in plan["calibration_budgets_seconds"]:
                n = round(budget / 0.02)
                if n > len(case["tx"]):
                    unavailable.append(
                        {
                            "session_id": item["session_id"],
                            "budget_seconds": budget,
                            "reason": "short prefix",
                        }
                    )
                    continue
                pre = prefix(case, n)
                if model.training_mask(pre["mask"], pre["segments"]).sum() < 50:
                    unavailable.append(
                        {
                            "session_id": item["session_id"],
                            "budget_seconds": budget,
                            "reason": "too few training labels",
                        }
                    )
                    continue
                cubes[budget] = np.empty((len(configs), stop - start, 2), dtype=np.float32)
                ledgers[budget] = []
            for base in range(0, len(configs), len(plan["grid"]["ridge_lambda"])):
                config = configs[base]["config"]
                raw = model.representation(case["tx"], case["sbp"], case["segments"], config)
                for budget, cube in cubes.items():
                    pre = prefix(case, round(budget / 0.02))
                    system = model.prepare_system(
                        raw[: len(pre["tx"])],
                        pre["y"],
                        pre["mask"],
                        pre["segments"],
                        pre["tx"].shape[1],
                    )
                    normalized = np.clip(
                        (raw[start:stop] - system["mean"]) / system["scale"], -8, 8
                    )
                    for offset in range(len(plan["grid"]["ridge_lambda"])):
                        ci = base + offset
                        fitted, receipt = model.solve_system(system, configs[ci]["config"])
                        prediction = model.project_direction(
                            normalized @ fitted.coefficients + fitted.intercept
                        ).astype(np.float32)
                        cube[ci] = prediction
                        ledgers[budget].append(
                            {
                                "participant": item["participant_id"],
                                "session_id": item["session_id"],
                                "budget_seconds": budget,
                                "candidate_id": configs[ci]["candidate_id"],
                                "candidate_index": ci,
                                "score_start_bin": start,
                                "score_stop_bin": stop,
                                "mask_sha256": mask_digest(case["mask"][start:stop]),
                                "metrics": score(
                                    case["y"][start:stop], prediction, case["mask"][start:stop]
                                ),
                                "feature_sha256": item["feature_sha256"],
                                "target_sha256": item["target_sha256"],
                                **receipt,
                            }
                        )
            for budget, cube in cubes.items():
                target = output / "development_traces" / f"{item['session_id']}-{budget}s.npz"
                np.savez_compressed(
                    target,
                    predictions=cube,
                    candidate_ids=np.array([c["candidate_id"] for c in configs]),
                    start_bin=np.asarray(start),
                    stop_bin=np.asarray(stop),
                )
                trace_hash = digest(target)
                for row in ledgers[budget]:
                    row.update(trace_path=relative(target), trace_sha256=trace_hash)
                cases.extend(ledgers[budget])
            print(f"source grid {item['session_id']} complete", flush=True)
        selection = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "plan": manifest["plan"],
            "source_files": manifest["source_files"],
            "development_index": manifest["development_index"],
            "candidates": configs,
            "development_cases": cases,
            "development_unavailable": unavailable,
            "choices": choose(cases, configs, plan),
            "elapsed_seconds": time.perf_counter() - began,
            "assessment_labels_opened": False,
        }
        path = output / "selection.json"
        write_json(path, selection)
        manifest.update(
            status="source_selection_complete",
            selection={"path": relative(path), "sha256": digest(path)},
        )
        write_json(output / "manifest.json", manifest)
        return selection
    except Exception as exc:
        write_json(
            output / "failure.json",
            {"phase": "source_selection", "error": str(exc), "assessment_labels_opened": False},
        )
        raise


def interval(values):
    values = np.asarray(values, float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(20261022)
    boot = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def summarize(identities, plan):
    groups = []
    for budget, method_name, partition in itertools.product(
        plan["calibration_budgets_seconds"],
        plan["assessment"]["methods"],
        ("primary", "common_suffix"),
    ):
        people = []
        for person in plan["participants"]:
            rows = [
                t["metrics"]
                for m in identities
                if m["target_participant"] == person
                and m["budget_seconds"] == budget
                and m["method"] == method_name
                for t in m["traces"]
                if t["partition"] == partition and t["metrics"]["r2"] is not None
            ]
            if rows:
                people.append(
                    {
                        "participant": person,
                        "r2": float(np.mean([r["r2"] for r in rows])),
                        "angular_error_degrees": float(
                            np.mean([r["angular_error_degrees"] for r in rows])
                        ),
                        "n_sessions": len(rows),
                    }
                )
        if people:
            values = [p["r2"] for p in people]
            groups.append(
                {
                    "budget_seconds": budget,
                    "method": method_name,
                    "partition": partition,
                    "people": people,
                    "mean_r2": float(np.mean(values)),
                    "r2_bootstrap_95": interval(values),
                    "positive_people": sum(v > 0 for v in values),
                    "mean_angular_error_degrees": float(
                        np.mean([p["angular_error_degrees"] for p in people])
                    ),
                }
            )
    primary = next(
        (
            g
            for g in groups
            if g["budget_seconds"] == 120
            and g["method"] == "selected_daily"
            and g["partition"] == "primary"
        ),
        None,
    )
    comparisons = []
    for method_name in ("zero", "prefix_mean", "v2_ridge_daily"):
        other = next(
            (
                g
                for g in groups
                if g["budget_seconds"] == 120
                and g["method"] == method_name
                and g["partition"] == "primary"
            ),
            None,
        )
        if primary and other:
            lookup = {p["participant"]: p["r2"] for p in other["people"]}
            effects = [
                {"participant": p["participant"], "difference": p["r2"] - lookup[p["participant"]]}
                for p in primary["people"]
                if p["participant"] in lookup
            ]
            values = [p["difference"] for p in effects]
            comparisons.append(
                {
                    "reference": method_name,
                    "effects": effects,
                    "mean_difference": float(np.mean(values)),
                    "bootstrap_95": interval(values),
                }
            )
    gate = plan["operational_usefulness_gate"]
    criteria = {
        "nine_eligible_people": bool(
            primary and len(primary["people"]) == gate["eligible_people_required"]
        ),
        "mean_r2_at_least_0_1": bool(primary and primary["mean_r2"] >= gate["mean_r2_at_least"]),
        "r2_interval_above_zero": bool(primary and primary["r2_bootstrap_95"][0] > 0),
        "seven_positive_people": bool(
            primary and primary["positive_people"] >= gate["positive_r2_people_at_least"]
        ),
        "angular_error_at_most_80": bool(
            primary
            and primary["mean_angular_error_degrees"] <= gate["angular_error_degrees_at_most"]
        ),
        "gain_at_least_0_05_over_each_null": len(
            [
                c
                for c in comparisons
                if c["reference"] in ("zero", "prefix_mean") and c["mean_difference"] >= 0.05
            ]
        )
        == 2,
    }
    return {
        "groups": groups,
        "primary_comparisons": comparisons,
        "operational_gate": {"criteria": criteria, "passed": all(criteria.values())},
    }


def assess(run, assessment_index, selection_audit):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "source_selection_complete":
        raise ValueError("assessment is a one-shot transition from completed source selection")
    selection_path = resolve(manifest["selection"]["path"])
    if digest(selection_path) != manifest["selection"]["sha256"]:
        raise ValueError("changed selection")
    audit = json.loads(Path(selection_audit).read_text())
    if audit.get("status") != "passed" or audit.get("selection_sha256") != digest(selection_path):
        raise ValueError("passed independent selection audit required")
    for source in manifest["source_files"]:
        if digest(resolve(source["path"])) != source["sha256"]:
            raise ValueError("source changed after source selection")
    if digest(resolve(manifest["plan"]["path"])) != manifest["plan"]["sha256"]:
        raise ValueError("plan changed")
    plan = json.loads(resolve(manifest["plan"]["path"]).read_text())
    selection = json.loads(selection_path.read_text())
    index = data.load_index(assessment_index, expected_role="assessment")
    development_path = resolve(manifest["development_index"]["path"])
    development = data.load_index(development_path, expected_role="development")
    manifest.update(
        status="assessment_running",
        assessment_started_at=datetime.now(UTC).isoformat(),
        assessment_index={"path": relative(assessment_index), "sha256": digest(assessment_index)},
        selection_audit={"path": relative(selection_audit), "sha256": digest(selection_audit)},
    )
    write_json(run / "manifest.json", manifest)
    candidates = {c["candidate_id"]: c["config"] for c in selection["candidates"]}
    identities = []
    began = time.perf_counter()
    try:
        for item in index["sessions"]:
            case = load_case(assessment_index, item)
            person = item["participant_id"]
            old_item = min(
                (s for s in development["sessions"] if s["participant_id"] == person),
                key=lambda s: (s["post_implant_day"], s["session_id"]),
            )
            old_case = load_case(development_path, old_item)
            for budget in plan["calibration_budgets_seconds"]:
                n = round(budget / 0.02)
                for method_name in plan["assessment"]["methods"]:
                    calibration = old_case if method_name == "stale_mapping" else case
                    calibration_index = (
                        development_path
                        if method_name == "stale_mapping"
                        else Path(assessment_index)
                    )
                    base = {
                        "target_participant": person,
                        "session_id": item["session_id"],
                        "method": method_name,
                        "budget_seconds": budget,
                    }
                    if n > len(calibration["tx"]):
                        manifest["unavailable"].append(
                            {**base, "reason": "calibration budget exceeds recording"}
                        )
                        continue
                    pre = prefix(calibration, n)
                    trainmask = model.training_mask(pre["mask"], pre["segments"])
                    if trainmask.sum() < 50:
                        manifest["unavailable"].append(
                            {**base, "reason": "fewer than 50 training labels after burn"}
                        )
                        continue
                    folder = run / "models" / f"{item['session_id']}-{method_name}-{budget}s"
                    folder.mkdir(parents=True, exist_ok=False)
                    source_choice = None
                    fitted_started = time.perf_counter()
                    if method_name in ("zero", "prefix_mean"):
                        value = (
                            np.zeros(2) if method_name == "zero" else pre["y"][trainmask].mean(0)
                        )
                        config = {"kind": "constant", "rule": method_name}
                        model_path = folder / "model.npz"
                        np.savez(model_path, constant=value)
                        with np.load(model_path, allow_pickle=False) as z:
                            prediction = np.broadcast_to(z["constant"], (len(case["tx"]), 2)).copy()
                        fit_receipt = {
                            "n_training_labels": int(trainmask.sum()),
                            "n_available_labels": int(pre["mask"].sum()),
                            "n_prefix_bins": n,
                        }
                    else:
                        if method_name == "v2_ridge_daily":
                            config = {"kind": "v2_ridge", "family": "tx"}
                        else:
                            chosen_method = (
                                "selected_daily" if method_name == "stale_mapping" else method_name
                            )
                            source_choice = next(
                                c
                                for c in selection["choices"]
                                if c["target_participant"] == person
                                and c["budget_seconds"] == budget
                                and c["method"] == chosen_method
                            )
                            config = candidates[source_choice["candidate_id"]]
                        fitted, fit_receipt = model.fit(
                            pre["tx"], pre["sbp"], pre["y"], pre["mask"], pre["segments"], config
                        )
                        model_path = folder / "model.npz"
                        fitted.save(model_path)
                        prediction = model.Decoder.load(model_path).predict(
                            case["tx"], case["sbp"], case["segments"]
                        )
                    fit_elapsed = time.perf_counter() - fitted_started
                    trace_path = folder / "prediction.npz"
                    np.savez_compressed(trace_path, prediction=prediction)
                    traces = []
                    for partition, start in (
                        ("primary", round((budget + 2) / 0.02)),
                        ("common_suffix", 15100),
                    ):
                        start = min(start, len(prediction))
                        mask = case["mask"].copy()
                        mask[:start] = False
                        traces.append(
                            {
                                "partition": partition,
                                "path": relative(trace_path),
                                "sha256": digest(trace_path),
                                "start_bin": start,
                                "stop_bin": len(prediction),
                                "mask_sha256": mask_digest(mask),
                                "metrics": score(case["y"], prediction, mask),
                                "feature_sha256": item["feature_sha256"],
                                "target_sha256": item["target_sha256"],
                            }
                        )
                    identity = {
                        **base,
                        "config": config,
                        "source_choice": source_choice,
                        "calibration": {
                            "role": "development"
                            if method_name == "stale_mapping"
                            else "assessment",
                            "session_id": calibration["metadata"]["session_id"],
                            "participant_id": person,
                            "start_bin": 0,
                            "stop_bin": n,
                            "feature_sha256": calibration["metadata"]["feature_sha256"],
                            "target_sha256": calibration["metadata"]["target_sha256"],
                            "mask_sha256": mask_digest(trainmask),
                            "source_index_path": relative(calibration_index),
                            "source_index_sha256": digest(calibration_index),
                            **fit_receipt,
                        },
                        "assessment_day_minus_calibration_day": item["post_implant_day"]
                        - calibration["metadata"]["post_implant_day"],
                        "model_path": relative(model_path),
                        "model_sha256": digest(model_path),
                        "fit_export_inference_seconds": fit_elapsed,
                        "traces": traces,
                    }
                    identity_path = folder / "identity.json"
                    write_json(identity_path, identity)
                    identities.append(identity)
                    manifest["models"].append(
                        {"identity_path": relative(identity_path), "sha256": digest(identity_path)}
                    )
                write_json(run / "manifest.json", manifest)
            print(f"assessment {item['session_id']} complete", flush=True)
        manifest.update(
            status="complete",
            assessment_elapsed_seconds=time.perf_counter() - began,
            assessment_completed_at=datetime.now(UTC).isoformat(),
        )
        summary = {"status": "complete", **summarize(identities, plan)}
        write_json(run / "summary.json", summary)
        manifest["summary"] = {
            "path": relative(run / "summary.json"),
            "sha256": digest(run / "summary.json"),
        }
        write_json(run / "manifest.json", manifest)
        return summary
    except Exception as exc:
        write_json(
            run / "failure.json",
            {
                "phase": "assessment",
                "error": str(exc),
                "completed_models": len(identities),
                "assessment_outcomes_exposed": bool(identities),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("select")
    dev.add_argument("--plan", type=Path, default=Path("neural_experiments/v3/plan.json"))
    dev.add_argument("--development-index", type=Path, required=True)
    dev.add_argument("--output", type=Path, required=True)
    assessment = sub.add_parser("assess")
    assessment.add_argument("--run", type=Path, required=True)
    assessment.add_argument("--assessment-index", type=Path, required=True)
    assessment.add_argument("--selection-audit", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "select":
            select(args.plan, args.development_index, args.output)
        else:
            assess(args.run, args.assessment_index, args.selection_audit)


if __name__ == "__main__":
    main()
