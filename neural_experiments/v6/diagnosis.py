"""Exposed-only V6 T2 pilot: fixed exports, later confirmation, longer calibration.

All outcomes are exploratory. Inputs are exact V5 prepared indexes capped at600s;
no V6 fresh data are admitted. Candidate choice uses earlier inner validation,
never its confirmation or final scoring interval. All numeric fits are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from neural_experiments.v4 import model as core
from neural_experiments.v5 import data
from neural_experiments.v6 import model

ROOT = Path(__file__).resolve().parents[2]
INDEX_PINS = {
    "data/neural_v5/development/index.json": (
        "1fb2d9b3167e2c671e279dd89fe05ebb6f38633d4f3e6f3b56078a0c61621fb1"
    ),
    "data/neural_v5/new_sessions/index.json": (
        "6ac718377da28dcb43ca127c8da6e6a78b40703ffc4b81358b06bb56ada4c5a4"
    ),
}
BUDGETS = (60, 120, 240, 360)
SCORE_START = 22600
SOURCE_FILES = (
    "neural_experiments/v6/diagnosis.py",
    "neural_experiments/v6/model.py",
    "neural_experiments/v4/model.py",
    "neural_experiments/v5/data.py",
    "neural_experiments/v4/data.py",
    "neural_experiments/v2/data.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metric(y, prediction, mask):
    yy, pp = np.asarray(y[mask], float), np.asarray(prediction[mask], float)
    if len(yy) < 50:
        return {"n": len(yy), "r2": None, "mse": None}
    residual = pp - yy
    angle = np.arctan2(yy[:, 1], yy[:, 0])
    counts = np.bincount(np.floor((angle + np.pi) / (np.pi / 4)).astype(int) % 8, minlength=8)
    return {
        "n": len(yy),
        "r2": float(r2_score(yy, pp, multioutput="variance_weighted")),
        "mse": float(np.mean(residual**2)),
        "bias_mse": float(np.mean(residual.mean(0) ** 2)),
        "centered_mse": float(np.mean((residual - residual.mean(0)) ** 2)),
        "residual_mean": residual.mean(0).tolist(),
        "target_mean": yy.mean(0).tolist(),
        "prediction_mean": pp.mean(0).tolist(),
        "direction_sector_counts": counts.tolist(),
    }


def window(case, prediction, start, stop, training_mean=None, *, burned=True):
    mask = case["mask"].copy()
    if burned:
        mask = core.training_mask(mask, case["segments"])
    mask[:start], mask[stop:] = False, False
    measured = metric(case["y"], prediction, mask)
    if measured["mse"] is not None:
        measured["zero_mse"] = float(np.mean(np.asarray(case["y"][mask], float) ** 2))
        mean = np.zeros(2) if training_mean is None else training_mean
        measured["training_mean_mse"] = float(np.mean((case["y"][mask] - mean) ** 2))
        measured["gain_vs_zero"] = measured["zero_mse"] - measured["mse"]
        measured["gain_vs_training_mean"] = measured["training_mean_mse"] - measured["mse"]
    measured.update(start_bin=start, stop_bin=stop, mask_sha256=model.digest_array(mask))
    return measured


def confirmation(case, prediction, start, stop, mean):
    mid = (start + stop) // 2
    return {
        "pooled": window(case, prediction, start, stop, mean),
        "first_half": window(case, prediction, start, mid, mean),
        "last_half": window(case, prediction, mid, stop, mean),
    }


def passed(evidence, margin, halves):
    names = ("pooled", "first_half", "last_half") if halves else ("pooled",)
    return all(
        evidence[k]["n"] >= 100
        and evidence[k]["mse"] is not None
        and sum(n >= 10 for n in evidence[k]["direction_sector_counts"]) >= 4
        and evidence[k]["gain_vs_zero"] >= margin
        and evidence[k]["gain_vs_training_mean"] >= margin
        for k in names
    )


def prediction_from_export(path, raw):
    exported = model.Decoder.load(path)
    return core.project_direction(
        np.clip((raw - exported.mean) / exported.scale, -8, 8) @ exported.weights[:-1]
        + exported.weights[-1]
    )


def load_cases():
    cases, provenance = [], []
    for relative, expected in INDEX_PINS.items():
        index_path = ROOT / relative
        if sha(index_path) != expected:
            raise ValueError("unrecognized exposed input index")
        index = json.loads(index_path.read_text())
        for item in index["sessions"]:
            if item["participant_id"] != "T2":
                continue
            if not 22600 < item["n_bins"] <= 30000:
                raise ValueError("pilot requires the declared exposed 600s T2 cases")
            feature, target = (index_path.parent / item[k + "_path"] for k in ("feature", "target"))
            if sha(feature) != item["feature_sha256"] or sha(target) != item["target_sha256"]:
                raise ValueError("exposed arrays changed")
            features, labels = (
                data.load_features(feature),
                data.load_targets(target, item["n_bins"]),
            )
            cases.append(
                {
                    "item": item,
                    "features": features,
                    "y": labels["y"],
                    "mask": labels["eval_mask"],
                    "segments": [
                        (s["start"], s["end"]) for s in item["segments"] if s["end"] > s["start"]
                    ],
                }
            )
            provenance.append(
                {
                    "session_id": item["session_id"],
                    "n_bins": item["n_bins"],
                    "post_implant_day": item["post_implant_day"],
                    "index_sha256": expected,
                    "feature_sha256": item["feature_sha256"],
                    "target_sha256": item["target_sha256"],
                }
            )
    cases.sort(key=lambda c: c["item"]["post_implant_day"])
    if len(cases) != 15 or len({c["item"]["session_id"] for c in cases}) != 15:
        raise ValueError("exact15 exposed T2 cases required")
    return cases, provenance


def original_v5(cases):
    path = ROOT / "results/neural_v5/assessment/manifest.json"
    manifest = json.loads(path.read_text())
    rows = []
    for case in cases:
        sid = case["item"]["session_id"]
        selected = [
            r
            for r in manifest["cases"]
            if r["session_id"] == sid and r["method"] in ("adaptive", "daily120", "daily240")
        ]
        for row in selected:
            fitted = core.Decoder.load(ROOT / row["model_path"])
            prediction = fitted.predict(case["features"], case["segments"])
            with np.load(ROOT / row["trace_path"], allow_pickle=False) as z:
                err = float(np.max(np.abs(prediction - z["prediction"])))
            entry = {
                "session_id": sid,
                "method": row["method"],
                "candidate_id": row["candidate_id"],
                "model_sha256": row["model_sha256"],
                "trace_sha256": row["trace_sha256"],
                "trace_max_absolute_error": err,
                "old_endpoint": window(case, prediction, 13600, 30000, burned=False),
                "common_endpoint": window(case, prediction, SCORE_START, 30000, burned=False),
            }
            if row["method"] == "adaptive" and row["action"] == "accept":
                budget = row["label_seconds"]
                stop, half = 1500 + 50 * budget, 1500 + 25 * budget
                matched = [
                    f
                    for f in manifest["fits"]
                    if f["session_id"] == sid
                    and f["candidate_id"] == row["candidate_id"]
                    and f["budget_seconds"] == budget
                    and f["phase"] == "validation"
                ]
                if len(matched) != 1:
                    raise ValueError("unique original inner artifact required")
                old = matched[0]
                half_pred = core.Decoder.load(ROOT / old["model_path"]).predict(
                    case["features"], case["segments"]
                )
                entry["original_inner_model_sha256"] = old["model_sha256"]
                entry["artifact_comparison"] = {
                    name: {
                        "inner": window(case, half_pred, a, b),
                        "deployed_refit": window(case, prediction, a, b),
                    }
                    for name, a, b in (
                        ("old_validation", half + 100, stop),
                        ("next58s", stop + 100, stop + 3000),
                        ("later_old_endpoint", 13600, 30000),
                    )
                }
            rows.append(entry)
    return {"manifest_sha256": sha(path), "rows": rows}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    plan = {
        "scope": "post-hoc15 exposed T2 cases only; no fresh V6 inference",
        "input_index_sha256": INDEX_PINS,
        "source_sha256": sources,
        "fit_budgets_seconds": BUDGETS,
        "candidate_grid": model.candidate_grid(),
        "normalization": (
            "first30 or all features available at that fit; frozen throughout "
            "confirmation and deployment"
        ),
        "inner_selection": (
            "minimum coordinate MSE on [half training end+2s,full training end), >=100 "
            "rows; fixed grid-order tie"
        ),
        "confirm": (
            "exact full export, [training end+2s,training end+60s); no postconfirmation refit"
        ),
        "prior0": (
            "fixed combined family, first30 normalization, latest3 strictly earlier "
            "same-layout histories; confirm[30,90),60 label seconds"
        ),
        "common_score_seconds": [452, 600],
        "confirmation_margins": [0.025, 0.05],
        "confirmation_rules": ["pooled", "pooled_and_both_halves"],
        "label_cost": (
            "all elapsed requested labels from30 throughconfirmationend, includingpurge; "
            "prior0cost60"
        ),
        "all_fits_retained": True,
    }
    write(output / "plan.json", plan)
    started = time.perf_counter()
    cases, provenance = load_cases()
    result = {
        "status": "running",
        "plan_sha256": sha(output / "plan.json"),
        "source_sha256": sources,
        "inputs": provenance,
        "fits": [],
        "stages": [],
        "prior0": [],
        "sequential": [],
        "original_v5": original_v5(cases),
    }
    with threadpool_limits(limits=1):
        caches = {}
        for case in cases:
            sid = case["item"]["session_id"]
            caches[sid] = {}
            for candidate in model.candidate_grid()[::6]:
                config = candidate["config"]
                raw = core.representation(case["features"], case["segments"], config)
                stats = model.historical_stats(
                    raw, case["y"], case["mask"], case["segments"], case["item"]["n_channels"]
                )
                caches[sid][config["family"]] = {"raw": raw, "stats": stats}
        for case in cases:
            item, sid = case["item"], case["item"]["session_id"]
            past = [
                c
                for c in cases
                if c["item"]["post_implant_day"] < item["post_implant_day"]
                and np.array_equal(c["features"]["channel_ids"], case["features"]["channel_ids"])
            ][-3:]
            priors = {
                family: core.solve_history(
                    [
                        caches[c["item"]["session_id"]][family]["stats"]
                        for c in past
                        if caches[c["item"]["session_id"]][family]["stats"]["n"] >= 50
                    ],
                    caches[sid][family]["raw"].shape[1] + 1,
                )
                for family in ("tx45", "tx45_sbp")
            }
            config = model.candidate_grid()[6]["config"]
            prior_model = model.prior_decoder(
                caches[sid]["tx45_sbp"]["raw"], item["n_channels"], config, priors["tx45_sbp"]
            )
            prior_path = output / "models" / f"{sid}-prior0.npz"
            prior_model.save(prior_path)
            pp = prediction_from_export(prior_path, caches[sid]["tx45_sbp"]["raw"])
            prior_row = {
                "session_id": sid,
                "post_implant_day": item["post_implant_day"],
                "history_ids": [c["item"]["session_id"] for c in past],
                "model_sha256": sha(prior_path),
                "confirmation": confirmation(case, pp, 1500, 4500, np.zeros(2)),
                "common": window(case, pp, SCORE_START, 30000, burned=False),
                "label_seconds": 60,
            }
            result["prior0"].append(prior_row)
            chosen_stages = []
            for budget in BUDGETS:
                half, stop = 1500 + 25 * budget, 1500 + 50 * budget
                inner_rows = []
                for candidate in model.candidate_grid():
                    cid, family = candidate["candidate_id"], candidate["config"]["family"]
                    raw = caches[sid][family]["raw"]
                    fitted, receipt = model.fit(
                        raw,
                        case["y"][:half],
                        case["mask"][:half],
                        case["segments"],
                        item["n_channels"],
                        half,
                        candidate,
                        priors[family],
                    )
                    path = output / "models" / f"{sid}-{budget}-{cid}-inner.npz"
                    fitted.save(path)
                    pp = prediction_from_export(path, raw)
                    inner = window(case, pp, half + 100, stop)
                    row = {
                        "session_id": sid,
                        "budget": budget,
                        "phase": "inner",
                        "candidate_id": cid,
                        "model_sha256": sha(path),
                        "fit": receipt,
                        "inner_validation": inner,
                    }
                    result["fits"].append(row)
                    inner_rows.append(row)
                eligible = [r for r in inner_rows if r["inner_validation"]["n"] >= 100]
                chosen_id = (
                    min(eligible, key=lambda r: r["inner_validation"]["mse"])["candidate_id"]
                    if eligible
                    else None
                )
                # Candidate is now locked. No confirmation metric can change it.
                for candidate in model.candidate_grid():
                    cid, family = candidate["candidate_id"], candidate["config"]["family"]
                    raw = caches[sid][family]["raw"]
                    fitted, receipt = model.fit(
                        raw,
                        case["y"][:stop],
                        case["mask"][:stop],
                        case["segments"],
                        item["n_channels"],
                        stop,
                        candidate,
                        priors[family],
                    )
                    path = output / "models" / f"{sid}-{budget}-{cid}-full.npz"
                    fitted.save(path)
                    pp = prediction_from_export(path, raw)
                    trainmask = core.training_mask(case["mask"], case["segments"], stop)
                    trainmask[:1500] = False
                    mean = (
                        np.asarray(case["y"][:stop][trainmask], float).mean(0)
                        if trainmask.sum() >= 50
                        else np.zeros(2)
                    )
                    row = {
                        "session_id": sid,
                        "post_implant_day": item["post_implant_day"],
                        "budget": budget,
                        "phase": "full",
                        "candidate_id": cid,
                        "model_sha256": sha(path),
                        "fit": receipt,
                        "chosen_before_confirmation": cid == chosen_id,
                        "confirmation": confirmation(case, pp, stop + 100, stop + 3000, mean),
                        "common": window(case, pp, SCORE_START, 30000, burned=False),
                    }
                    result["fits"].append(row)
                    if cid == chosen_id:
                        choice = {
                            **row,
                            "selection_model_sha256": next(
                                r["model_sha256"] for r in inner_rows if r["candidate_id"] == cid
                            ),
                            "selection_metric": next(
                                r["inner_validation"]
                                for r in inner_rows
                                if r["candidate_id"] == cid
                            ),
                            "label_seconds": budget + 60,
                        }
                        result["stages"].append(choice)
                        chosen_stages.append(choice)
            zero = window(
                case, np.zeros_like(case["y"], dtype=float), SCORE_START, 30000, burned=False
            )
            for margin in (0.025, 0.05):
                for halves in (False, True):
                    selected = (
                        prior_row
                        if passed(prior_row["confirmation"], margin, halves)
                        else next(
                            (r for r in chosen_stages if passed(r["confirmation"], margin, halves)),
                            None,
                        )
                    )
                    performance = selected["common"] if selected else zero
                    result["sequential"].append(
                        {
                            "session_id": sid,
                            "post_implant_day": item["post_implant_day"],
                            "margin": margin,
                            "require_halves": halves,
                            "action": "accept" if selected else "abstain",
                            "candidate_id": (
                                selected.get("candidate_id", "prior0") if selected else "zero"
                            ),
                            "label_seconds": selected["label_seconds"] if selected else 420,
                            "training_budget": selected.get("budget", 0) if selected else None,
                            "common": performance,
                            "zero_common": zero,
                            "accepted_worse_than_zero": bool(
                                selected and performance["mse"] > zero["mse"]
                            ),
                            "accepted_negative_r2": bool(selected and performance["r2"] < 0),
                        }
                    )
            print(json.dumps({"session": sid, "fit_count": len(result["fits"])}), flush=True)
            write(output / "diagnosis.json", result)
    result.update(
        status="complete",
        elapsed_seconds=time.perf_counter() - started,
        claims=(
            "Exploratory exposed-data results; no new held-out evidence "
            "or person-specific deployed rule"
        ),
    )
    write(output / "diagnosis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/neural_v6/diagnosis-v1")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
