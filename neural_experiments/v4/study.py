"""Chronological source selection and locked, separate-domain V4 assessment.

Raw data, models, and sample traces remain in ignored local directories. Selection
cannot accept a fresh index; assessment requires a hash-bound independent audit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_experiments.v3.study import digest, mask_digest, relative, score, write_json
from neural_experiments.v4 import data, model

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "neural_experiments/v4/plan.json"
PLAN_SHA = "f00419baf098ab5e42d6757714bd43cc5107579edf2a7c915c24a4f6ff7d8aba"


def grids(plan):
    g = plan["global_grid"]
    global_grid = []
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
    t = plan["T3_targeted_grid"]
    targeted = []
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


def load_case(index_path, item, labels=True):
    base = Path(index_path).resolve().parent
    fp = base / item["feature_path"]
    if digest(fp) != item["feature_sha256"]:
        raise ValueError("feature hash mismatch")
    case = {
        "item": item,
        "features": data.load_features(fp),
        "segments": [(s["start"], s["end"]) for s in item["segments"] if s["end"] > s["start"]],
    }
    if labels:
        tp = base / item["target_path"]
        if digest(tp) != item["target_sha256"]:
            raise ValueError("target hash mismatch")
        case.update(data.load_targets(tp, item["n_bins"]))
    return case


def histories(query, items, layouts):
    """Choose three chronological identities before checking label availability."""
    eligible = [
        s
        for s in items
        if s["participant_id"] == query["participant_id"]
        and s["post_implant_day"] < query["post_implant_day"]
        and layouts[s["session_id"]] == layouts[query["session_id"]]
    ]
    return sorted(eligible, key=lambda s: (s["post_implant_day"], s["session_id"]))[-3:]


def fitting_stats(z, y, mask, segments, stop):
    stop = min(stop, len(z))
    eligible = model.training_mask(mask, segments, stop)
    return model.sufficient(z[:stop], y[:stop], eligible)


def history_stats(case, config):
    raw = model.representation(case["features"], case["segments"], config)
    z, _, _ = model.design(raw, len(raw), case["item"]["n_channels"])
    mask = model.training_mask(case["eval_mask"], case["segments"])
    mask &= np.arange(len(mask)) % 5 == 0
    return model.sufficient(z, case["y"], mask)


def prior_for(query, items, layouts, stats, dimension):
    selected = histories(query, items, layouts)
    receipt, valid = [], []
    for s in selected:
        st = stats[s["session_id"]]
        ok = st["n"] >= 50
        receipt.append(
            {
                "session_id": s["session_id"],
                "post_implant_day": s["post_implant_day"],
                "n": st["n"],
                "disposition": "used" if ok else "insufficient_labels",
            }
        )
        if ok:
            valid.append(st)
    return model.solve_history(valid, dimension), receipt


def save_prediction(output, key, fitted, case, eval_mask, row):
    mp, tp = output / "models" / f"{key}.npz", output / "traces" / f"{key}.npz"
    fitted.save(mp)
    prediction = model.Decoder.load(mp).predict(case["features"], case["segments"])
    np.savez_compressed(tp, prediction=prediction, eval_mask=eval_mask)
    metrics = score(case["y"], prediction, eval_mask)
    metrics["mse"] = (
        float(np.square(case["y"][eval_mask] - prediction[eval_mask]).mean())
        if eval_mask.any()
        else None
    )
    row.update(
        model_path=relative(mp),
        model_sha256=digest(mp),
        trace_path=relative(tp),
        trace_sha256=digest(tp),
        metrics=metrics,
        eval_mask_sha256=mask_digest(eval_mask),
    )
    return row


def choose(cases, candidates, plan):
    result = []
    for person in plan["participants"]:
        ranking = []
        for candidate in candidates:
            means = []
            for source in plan["participants"]:
                if source == person:
                    continue
                scores = [
                    r["metrics"]["r2"]
                    for r in cases
                    if r["kind"] == "global"
                    and r["participant"] == source
                    and r["candidate_id"] == candidate["candidate_id"]
                    and r["metrics"]["r2"] is not None
                ]
                if scores:
                    means.append(float(np.mean(scores)))
            if not means:
                raise ValueError("no source selection evidence")
            ranking.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "objective": float(np.mean(means)),
                    "eligible_source_people": len(means),
                }
            )
        result.append(
            {
                "target_participant": person,
                "candidate_id": max(ranking, key=lambda r: r["objective"])["candidate_id"],
                "ranking": ranking,
            }
        )
    return result


def choose_targeted(cases, candidates, items):
    result = []
    for query in items:
        if query["participant_id"] != "T3":
            continue
        source = [
            r
            for r in cases
            if r["kind"] == "targeted"
            and r["participant"] == "T3"
            and r["post_implant_day"] < query["post_implant_day"]
            and r["metrics"]["r2"] is not None
        ]
        scope = "earlier_T3" if source else "other_people"
        if not source:
            source = [
                r
                for r in cases
                if r["kind"] == "targeted"
                and r["participant"] != "T3"
                and r["metrics"]["r2"] is not None
            ]
        ranking = []
        for candidate in candidates:
            rows = [r for r in source if r["candidate_id"] == candidate["candidate_id"]]
            per_person = [
                np.mean([r["metrics"]["r2"] for r in rows if r["participant"] == p])
                for p in sorted({r["participant"] for r in rows})
            ]
            if not per_person:
                raise ValueError("no targeted selection evidence")
            ranking.append(
                {"candidate_id": candidate["candidate_id"], "objective": float(np.mean(per_person))}
            )
        result.append(
            {
                "target_session_id": query["session_id"],
                "post_implant_day": query["post_implant_day"],
                "selection_scope": scope,
                "source_session_ids": sorted({r["session_id"] for r in source}),
                "candidate_id": max(ranking, key=lambda r: r["objective"])["candidate_id"],
                "ranking": ranking,
            }
        )
    return result


def identity(plan_path, index_path):
    if digest(plan_path) != PLAN_SHA:
        raise ValueError("unrecognized locked study plan")
    paths = [
        "neural_experiments/v4/model.py",
        "neural_experiments/v4/data.py",
        "neural_experiments/v4/study.py",
        "neural_experiments/v3/study.py",
        "neural_experiments/v2/data.py",
    ]
    return {
        "schema_version": 1,
        "plan": {"path": relative(plan_path), "sha256": digest(plan_path)},
        "development_index": {"path": relative(index_path), "sha256": digest(index_path)},
        "code_sha256": {p: digest(ROOT / p) for p in paths},
    }


def init_output(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "traces").mkdir()
    return output


def row_identity(item, kind, candidate):
    return {
        "participant": item["participant_id"],
        "session_id": item["session_id"],
        "post_implant_day": item["post_implant_day"],
        "kind": kind,
        "candidate_id": candidate["candidate_id"],
    }


def select(plan_path, index_path, output):
    began = time.perf_counter()
    plan = json.loads(Path(plan_path).read_text())
    base = identity(plan_path, index_path)
    if base["development_index"]["sha256"] != plan["development_index_sha256"]:
        raise ValueError("development index differs from locked plan")
    index = data.load_index(index_path, expected_role="development")
    output = init_output(output)
    write_json(output / "manifest.json", {**base, "status": "source_running"})
    candidates, targeted = grids(plan)
    layouts = {
        s["session_id"]: tuple(load_case(index_path, s, False)["features"]["channel_ids"])
        for s in index["sessions"]
    }
    rows = []
    for start in range(0, len(candidates), 2):
        pair = candidates[start : start + 2]
        config = pair[0]["config"]
        stats = {
            s["session_id"]: history_stats(load_case(index_path, s), config)
            for s in index["sessions"]
        }
        for item in index["sessions"]:
            case = load_case(index_path, item)
            n = item["n_bins"]
            raw = model.representation(case["features"], case["segments"], config)
            z, mean, scale = model.design(raw, min(1500, n), item["n_channels"])
            target = fitting_stats(z, case["y"], case["eval_mask"], case["segments"], 1500)
            prior, receipt = prior_for(item, index["sessions"], layouts, stats, z.shape[1])
            evaluation = case["eval_mask"].copy()
            evaluation[:6100] = False
            for candidate in pair:
                weights = model.adapt(target, prior, candidate["config"]["adapt_gamma"])
                fitted = model.Decoder(candidate["config"], mean, scale, weights)
                row = {
                    **row_identity(item, "global", candidate),
                    "normalization_bins": min(1500, n),
                    "training_rows": target["n"],
                    "history": receipt,
                    "fallback": target["n"] < 50,
                }
                rows.append(
                    save_prediction(
                        output,
                        f"{item['session_id']}-{candidate['candidate_id']}",
                        fitted,
                        case,
                        evaluation,
                        row,
                    )
                )
        print(
            f"source global {pair[0]['candidate_id']}..{pair[-1]['candidate_id']} complete",
            flush=True,
        )
    for start in range(0, len(targeted), 4):
        group = targeted[start : start + 4]
        for item in index["sessions"]:
            if item["n_bins"] < 6000:
                for candidate in group:
                    rows.append(
                        {
                            **row_identity(item, "targeted", candidate),
                            "unavailable": "short_120s_prefix",
                            "metrics": {"r2": None},
                        }
                    )
                continue
            case = load_case(index_path, item)
            raw = model.representation(case["features"], case["segments"], group[0]["config"])
            z, mean, scale = model.design(raw, 6000, item["n_channels"])
            target = fitting_stats(z, case["y"], case["eval_mask"], case["segments"], 6000)
            evaluation = case["eval_mask"].copy()
            evaluation[:15100] = False
            for candidate in group:
                weights = model.solve_daily(target, candidate["penalty"], candidate["bias"])
                fitted = model.Decoder(candidate["config"], mean, scale, weights)
                row = {
                    **row_identity(item, "targeted", candidate),
                    "normalization_bins": 6000,
                    "training_rows": target["n"],
                    "history": [],
                    "fallback": target["n"] < 50,
                }
                rows.append(
                    save_prediction(
                        output,
                        f"{item['session_id']}-{candidate['candidate_id']}",
                        fitted,
                        case,
                        evaluation,
                        row,
                    )
                )
        print(
            f"source targeted {group[0]['candidate_id']}..{group[-1]['candidate_id']} complete",
            flush=True,
        )
    selection = {
        **base,
        "stage": "source",
        "candidates": candidates,
        "targeted_candidates": targeted,
        "cases": rows,
        "choices": choose(rows, candidates, plan),
        "targeted_choices": choose_targeted(rows, targeted, index["sessions"]),
        "elapsed_seconds": time.perf_counter() - began,
        "fresh_labels_opened": False,
    }
    write_json(output / "selection.json", selection)
    write_json(
        output / "manifest.json",
        {
            **base,
            "status": "source_complete",
            "selection_sha256": digest(output / "selection.json"),
        },
    )
    return selection


def interval(values):
    values = np.asarray(values, float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(20261023)
    return np.quantile(
        values[rng.integers(0, len(values), (10000, len(values)))].mean(1), [0.025, 0.975]
    ).tolist()


def summarize(rows, plan):
    domains = {}
    for domain in ("new_sessions", "t3_blocks"):
        methods = {}
        for method_name in plan["assessment"][domain]["methods"]:
            eligible = [
                r
                for r in rows
                if r["domain"] == domain
                and r["method"] == method_name
                and r["metrics"]["r2"] is not None
            ]
            unit_key = "participant" if domain == "new_sessions" else "session_id"
            units = []
            for unit in sorted({r[unit_key] for r in eligible}):
                rr = [r["metrics"] for r in eligible if r[unit_key] == unit]
                units.append(
                    {
                        "unit": unit,
                        "r2": float(np.mean([r["r2"] for r in rr])),
                        "angular_error_degrees": float(
                            np.mean([r["angular_error_degrees"] for r in rr])
                        ),
                        "n_sessions": len(rr),
                    }
                )
            values = [u["r2"] for u in units]
            methods[method_name] = {
                "units": units,
                "n_units": len(units),
                "n_sessions": len(eligible),
                "mean_r2": float(np.mean(values)) if values else None,
                "r2_bootstrap_95": interval(values),
                "positive_units": sum(v > 0 for v in values),
                "mean_angular_error_degrees": float(
                    np.mean([u["angular_error_degrees"] for u in units])
                )
                if units
                else None,
            }
        contrasts = []
        for arm in (
            "selected_prior0",
            "selected_prior30",
            "selected_prior120",
            "targeted_daily120",
        ):
            if arm not in methods:
                continue
            for control in ("zero", "prefix_mean30", "v3_daily120", "selected_scratch30"):
                left = {
                    r["session_id"]: r
                    for r in rows
                    if r["domain"] == domain
                    and r["method"] == arm
                    and r["metrics"]["r2"] is not None
                }
                right = {
                    r["session_id"]: r
                    for r in rows
                    if r["domain"] == domain
                    and r["method"] == control
                    and r["metrics"]["r2"] is not None
                }
                pairs = []
                for sid in sorted(left.keys() & right.keys()):
                    a, b = left[sid], right[sid]
                    if a["eval_mask_sha256"] != b["eval_mask_sha256"]:
                        raise ValueError("paired masks differ")
                    pairs.append(
                        {"unit": a[unit_key], "gain": a["metrics"]["r2"] - b["metrics"]["r2"]}
                    )
                values = [
                    float(np.mean([p["gain"] for p in pairs if p["unit"] == u]))
                    for u in sorted({p["unit"] for p in pairs})
                ]
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
        nulls = [
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
            and all(c["mean_gain"] >= 0.05 for c in nulls)
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
        and tc["zero"]["mean_gain"] >= 0.05
        and tc["v3_daily120"]["gain_bootstrap_95"][0] > 0
    )
    return {
        "domains": domains,
        "gates": {
            "reduced_label_useful": useful("selected_prior30"),
            "zero_fresh_label_useful": useful("selected_prior0"),
            "T3_recovered": recovered,
            "calibration_noninferiority": bool(
                useful("selected_prior30") and ni["gain_bootstrap_95"][0] > -0.05
            ),
        },
    }


def assess(selection_path, audit_path, indices, output):
    selection = json.loads(Path(selection_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("status") != "PASS" or audit.get("selection_sha256") != digest(selection_path):
        raise ValueError("passed hash-bound source audit required")
    for path, sha in selection["code_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"changed source: {path}")
    plan_path = ROOT / selection["plan"]["path"]
    dev_path = ROOT / selection["development_index"]["path"]
    if identity(plan_path, dev_path) != {k: selection[k] for k in identity(plan_path, dev_path)}:
        raise ValueError("source provenance changed")
    plan = json.loads(plan_path.read_text())
    dev = data.load_index(dev_path, expected_role="development")
    output = init_output(output)
    manifest = {
        "schema_version": 1,
        "stage": "assessment",
        "status": "assessment_started",
        "selection_path": relative(selection_path),
        "selection_sha256": digest(selection_path),
        "source_audit_path": relative(audit_path),
        "source_audit_sha256": digest(audit_path),
        "fresh_labels_opened": False,
        "indices": {},
        "cases": [],
    }
    write_json(output / "manifest.json", manifest)
    layouts = {
        s["session_id"]: tuple(load_case(dev_path, s, False)["features"]["channel_ids"])
        for s in dev["sessions"]
    }
    stats_cache = {}
    try:
        for domain, index_path in zip(("new_sessions", "t3_blocks"), indices, strict=True):
            index = data.load_index(index_path, expected_role=domain)
            manifest["indices"][domain] = {
                "path": relative(index_path),
                "sha256": digest(index_path),
            }
            for item in index["sessions"]:
                methods = plan["assessment"][domain]["methods"]
                if item["n_bins"] < 6100 and domain == "new_sessions":
                    manifest["cases"].extend(
                        {
                            "domain": domain,
                            "method": method_name,
                            "participant": item["participant_id"],
                            "session_id": item["session_id"],
                            "unavailable": "duration_before_common_evaluation",
                            "metrics": {"r2": None},
                        }
                        for method_name in methods
                    )
                    continue
                manifest["fresh_labels_opened"] = True
                write_json(output / "manifest.json", manifest)
                case = load_case(index_path, item)
                layouts[item["session_id"]] = tuple(case["features"]["channel_ids"])
                if domain == "t3_blocks":
                    prefix_path = Path(item["known_prefix_target_path"])
                    if digest(prefix_path) != item["known_prefix_target_sha256"]:
                        raise ValueError("known prefix hash changed")
                    labels = data.load_targets(prefix_path, item["known_prefix_n_bins"])
                    # Only allowed calibration labels are materialized in the fit buffer.
                    fit_y = np.zeros_like(case["y"])
                    fit_mask = np.zeros_like(case["eval_mask"])
                    fit_y[:6000], fit_mask[:6000] = labels["y"][:6000], labels["eval_mask"][:6000]
                else:
                    fit_y, fit_mask = case["y"], case["eval_mask"]
                evaluation = case["eval_mask"].copy()
                if domain == "new_sessions":
                    evaluation[:6100] = False
                chosen = next(
                    c["candidate_id"]
                    for c in selection["choices"]
                    if c["target_participant"] == item["participant_id"]
                )
                selected = next(c for c in selection["candidates"] if c["candidate_id"] == chosen)
                raw_cache = {}
                for method_name in methods:
                    config = dict(selected["config"])
                    candidate_id = chosen
                    budget = (
                        0
                        if method_name in ("zero", "selected_prior0")
                        else (120 if method_name.endswith("120") else 30)
                    )
                    normalizer = 30
                    daily = method_name.startswith(("v3_daily", "targeted_daily"))
                    penalty, bias = 1.0, 1
                    if method_name.startswith("v3_daily"):
                        config = {
                            "family": "tx45_sbp",
                            "tau_seconds": 0.2,
                            "baseline_tau_seconds": None,
                            "adapt_gamma": 1.0,
                        }
                        candidate_id, normalizer = "v3_fixed", budget
                    if method_name.startswith("targeted_daily"):
                        cid = next(
                            c["candidate_id"]
                            for c in selection["targeted_choices"]
                            if c["target_session_id"] == item["session_id"]
                        )
                        candidate = next(
                            c for c in selection["targeted_candidates"] if c["candidate_id"] == cid
                        )
                        config, penalty, bias = (
                            candidate["config"],
                            candidate["penalty"],
                            candidate["bias"],
                        )
                        candidate_id, normalizer = cid, budget
                    key = json.dumps(
                        {k: v for k, v in config.items() if k != "adapt_gamma"}, sort_keys=True
                    )
                    if key not in raw_cache:
                        raw_cache[key] = model.representation(
                            case["features"], case["segments"], config
                        )
                    z, mean, scale = model.design(
                        raw_cache[key], normalizer * 50, item["n_channels"]
                    )
                    target = fitting_stats(z, fit_y, fit_mask, case["segments"], budget * 50)
                    history, prior = [], np.zeros((z.shape[1], 2))
                    if method_name.startswith("selected_prior"):
                        if key not in stats_cache:
                            stats_cache[key] = {
                                s["session_id"]: history_stats(load_case(dev_path, s), config)
                                for s in dev["sessions"]
                            }
                        prior, history = prior_for(
                            item, dev["sessions"], layouts, stats_cache[key], z.shape[1]
                        )
                    if daily:
                        weights = model.solve_daily(target, penalty, bias)
                    elif method_name == "zero":
                        weights = prior
                    elif method_name == "prefix_mean30":
                        weights = prior
                        if target["n"] >= 50:
                            weights[-1] = target["rhs"][-1]
                    else:
                        weights = model.adapt(target, prior, config["adapt_gamma"])
                    fitted = model.Decoder(config, mean, scale, weights)
                    row = {
                        "domain": domain,
                        "method": method_name,
                        **row_identity(item, "assessment", {"candidate_id": candidate_id}),
                        "budget_seconds": budget,
                        "normalization_bins": normalizer * 50,
                        "training_rows": target["n"],
                        "history": history,
                        "fallback": budget > 0 and target["n"] < 50,
                        "penalty": penalty if daily else config["adapt_gamma"],
                        "bias": bias,
                    }
                    manifest["cases"].append(
                        save_prediction(
                            output,
                            f"{domain}-{item['session_id']}-{method_name}",
                            fitted,
                            case,
                            evaluation,
                            row,
                        )
                    )
                write_json(output / "manifest.json", manifest)
                print(f"assessment {domain} {item['session_id']} complete", flush=True)
        manifest.update(status="assessment_complete", summary=summarize(manifest["cases"], plan))
        write_json(output / "assessment.json", manifest)
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="assessment_failed", error=str(exc))
        write_json(output / "failure.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select")
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("assess")
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--new-index", type=Path, required=True)
    p.add_argument("--t3-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "select":
            select(args.plan, args.index, args.output)
        else:
            assess(args.selection, args.audit, (args.new_index, args.t3_index), args.output)


if __name__ == "__main__":
    main()
