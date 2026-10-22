"""Post-hoc T2 diagnosis on hash-bound, already-exposed <=600s prepared windows."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from neural_experiments.v4 import model, study

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/neural_v5/diagnosis/t2_pilot.json"
BUDGETS = (30, 60, 120, 240)
FAMILIES = ("tx45_sbp", "tx45", "tx3")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metric(y, p, m):
    yy, pp = np.asarray(y[m], float), np.asarray(p[m], float)
    if len(yy) < 50:
        return {"n": len(yy), "r2": None}
    residual = pp - yy
    return {
        "n": len(yy),
        "r2": float(r2_score(yy, pp, multioutput="variance_weighted")),
        "mse": float(np.square(residual).mean()),
        "bias_mse": float(np.square(residual.mean(0)).mean()),
        "centered_mse": float(np.square(residual - residual.mean(0)).mean()),
        "prediction_mean": pp.mean(0).tolist(),
        "target_mean": yy.mean(0).tolist(),
        "prediction_rms": float(np.sqrt(np.square(pp).sum(1).mean())),
    }


def direction(y, mask, segments, stop):
    valid = mask[:stop]
    yy = np.asarray(y[:stop][valid], float)
    bins = np.floor((np.arctan2(yy[:, 1], yy[:, 0]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    hist = np.bincount(bins, minlength=8)
    prob = hist[hist > 0] / max(1, len(yy))
    runs = []
    for a, b in segments:
        b = min(b, stop)
        if a >= b:
            continue
        edges = np.diff(np.r_[False, mask[a:b], False].astype(int))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        runs.extend((ends - starts).tolist())
    return {
        "n": len(yy),
        "mean_norm": float(np.linalg.norm(yy.mean(0))) if len(yy) else None,
        "octant_counts": hist.tolist(),
        "octant_entropy": float(-(prob * np.log(prob)).sum() / np.log(8)),
        "go_runs": len(runs),
        "median_go_run_seconds": float(np.median(runs) * 0.02) if runs else None,
        "blocks_observed": sum(a < stop for a, b in segments),
    }


def normalized(raw, stop, channels, family):
    floor = np.full(raw.shape[1], 0.1)
    floor[channels:] = 0.01
    mu = raw[:stop].mean(0)
    sd = np.maximum(raw[:stop].std(0), floor)
    return np.column_stack((np.clip((raw - mu) / sd, -8, 8), np.ones(len(raw)))), mu, sd


def fitstats(z, case, stop, stride=1):
    m = model.training_mask(case["eval_mask"], case["segments"], stop)
    m &= np.arange(stop) % stride == 0
    return model.sufficient(z[:stop], case["y"][:stop], m)


def project(z, w):
    return model.project_direction(z @ w)


def fitarm(stats, prior, name):
    if name == "prior":
        return prior.copy()
    if name == "daily1":
        return model.solve_daily(stats, 1.0, 1)
    if name == "daily10":
        return model.solve_daily(stats, 10.0, 1)
    if name.startswith("adapt"):
        return model.adapt(stats, prior, float(name[5:]))
    if name == "scratch10":
        return model.adapt(stats, np.zeros_like(prior), 10.0)
    raise ValueError(name)


def main(output=OUT):
    output = Path(output)
    started = time.monotonic()
    indices = [
        ROOT / "data/neural_v4/development/index.json",
        ROOT / "data/neural_v4/new_sessions/index.json",
    ]
    expected = {
        "data/neural_v4/development/index.json": (
            "aa6bf303c746edde0fb4d0a3a937a0fb07afb849270f2c79093fc021a9c2c500"
        ),
        "data/neural_v4/new_sessions/index.json": (
            "76293562fd6e586e7d6f5506c41ebda97241c971e18984c22489b69ac29558cc"
        ),
    }
    for path in indices:
        if sha(path) != expected[str(path.relative_to(ROOT))]:
            raise ValueError("Exposed input identity differs")
    manifest_path = ROOT / "results/neural_v4/assessment/assessment.json"
    manifest = json.loads(manifest_path.read_text())
    cases = []
    provenance = []
    for ip in indices:
        ix = json.loads(ip.read_text())
        for item in ix["sessions"]:
            if item["participant_id"] != "T2":
                continue
            if item["n_bins"] > 30000:
                raise ValueError("Exposed first600 cap exceeded")
            c = study.load_case(ip, item)
            c["origin"] = ix["role"]
            cases.append(c)
            provenance.append(
                {
                    "session_id": item["session_id"],
                    "n_bins": item["n_bins"],
                    "feature_sha256": item["feature_sha256"],
                    "target_sha256": item["target_sha256"],
                    "index_sha256": sha(ip),
                }
            )
    cases.sort(key=lambda c: c["item"]["post_implant_day"])
    original = []
    coverage = []
    features = []
    priorweights = []
    pilots = []
    choices = []
    for c in cases:
        sid = c["item"]["session_id"]
        y = c["y"]
        m = c["eval_mask"]
        for budget in BUDGETS:
            coverage.append(
                {"session_id": sid, "budget": budget, **direction(y, m, c["segments"], budget * 50)}
            )
        for row in manifest["cases"]:
            if row["session_id"] != sid or row["domain"] != "new_sessions":
                continue
            tp = ROOT / row["trace_path"]
            mp = ROOT / row["model_path"]
            if sha(tp) != row["trace_sha256"] or sha(mp) != row["model_sha256"]:
                raise ValueError("Trace/model hash")
            with np.load(tp, allow_pickle=False) as a:
                p = a["prediction"]
                evaluated = a["eval_mask"]
            original.append(
                {
                    "session_id": sid,
                    "method": row["method"],
                    "trace_sha256": row["trace_sha256"],
                    "model_sha256": row["model_sha256"],
                    "primary": metric(y, p, evaluated),
                    "common302": metric(y, p, m & (np.arange(len(m)) >= 15100)),
                }
            )
            if row["method"] in (
                "selected_prior0",
                "selected_prior30",
                "selected_prior120",
                "v3_daily120",
            ):
                fitted = model.Decoder.load(mp)
                raw = model.representation(c["features"], c["segments"], fitted.config)
                z = np.clip((raw - fitted.mean) / fitted.scale, -8, 8)
                channels = c["item"]["n_channels"]
                w = fitted.weights
                evalmask = m & (np.arange(len(m)) >= 15100)
                contributions = {
                    "tx": z[:, :channels] @ w[:channels],
                    "sbp": z[:, channels:] @ w[channels:-1],
                }
                for family, part in contributions.items():
                    original.append(
                        {
                            "session_id": sid,
                            "method": row["method"] + "_" + family + "_only_plus_intercept",
                            "common302": metric(y, model.project_direction(part + w[-1]), evalmask),
                        }
                    )
                for a, b in c["segments"]:
                    mm = evalmask.copy()
                    mm[:a] = False
                    mm[b:] = False
                    if mm.any():
                        original.append(
                            {
                                "session_id": sid,
                                "method": row["method"] + "_block",
                                "segment": [a, b],
                                "common302": metric(y, p, mm),
                            }
                        )
    for family in FAMILIES:
        config = {
            "family": family,
            "tau_seconds": 0.2 if family.endswith("sbp") else 0.4,
            "baseline_tau_seconds": None,
            "adapt_gamma": 10.0,
        }
        prepared = []
        for c in cases:
            raw = model.representation(c["features"], c["segments"], config)
            z, mu, sd = normalized(raw, len(raw), c["item"]["n_channels"], family)
            stats = fitstats(z, c, len(raw), stride=5)
            w = (
                model.solve_history([stats], raw.shape[1] + 1)
                if stats["n"] >= 50
                else np.zeros((raw.shape[1] + 1, 2))
            )
            prepared.append({"raw": raw, "stats": stats, "weights": w, "mean": mu, "scale": sd})
            for budget in (30, 120):
                stop = budget * 50
                zz, mean, scale = normalized(raw, stop, c["item"]["n_channels"], family)
                suffix = raw[15100:]
                standard = (suffix - mean) / scale
                channels = c["item"]["n_channels"]
                features.append(
                    {
                        "session_id": c["item"]["session_id"],
                        "family": family,
                        "budget": budget,
                        "count_nonzero_fraction": float(
                            np.mean(
                                c["features"]["x_tx3" if family == "tx3" else "x_tx45"][:stop] > 0
                            )
                        ),
                        "count_floor_fraction": float(np.mean(raw[:stop, :channels].std(0) < 0.1)),
                        "suffix_mean_rms_z": float(np.sqrt(np.square(standard.mean(0)).mean())),
                        "suffix_clipped_fraction": float(np.mean(np.abs(standard) > 8)),
                        "prefix_feature_std_mean": float(raw[:stop].std(0).mean()),
                    }
                )
        for j, c in enumerate(cases):
            sid = c["item"]["session_id"]
            raw = prepared[j]["raw"]
            channels = c["item"]["n_channels"]
            n = len(raw)
            # For the old V4 assessment reproduction, histories remain limited to original
            # development.
            eligible = [k for k in range(j) if cases[k]["origin"] == "development"][-3:]
            evalmask = c["eval_mask"] & (np.arange(n) >= 15100)
            for recency in (1, 3):
                ids = eligible[-recency:]
                valid = [prepared[k]["stats"] for k in ids if prepared[k]["stats"]["n"] >= 50]
                prior = model.solve_history(valid, raw.shape[1] + 1)
                for k in ids:
                    w = prepared[k]["weights"][:-1].ravel()
                    q = prior[:-1].ravel()
                    den = np.linalg.norm(w) * np.linalg.norm(q)
                    priorweights.append(
                        {
                            "session_id": sid,
                            "family": family,
                            "recency": recency,
                            "history_id": cases[k]["item"]["session_id"],
                            "age_days": c["item"]["post_implant_day"]
                            - cases[k]["item"]["post_implant_day"],
                            "cosine_to_pooled": float(w @ q / den) if den else None,
                            "tx_weight_norm": float(
                                np.linalg.norm(prepared[k]["weights"][:channels])
                            ),
                            "sbp_weight_norm": float(
                                np.linalg.norm(prepared[k]["weights"][channels:-1])
                            ),
                            "intercept": prepared[k]["weights"][-1].tolist(),
                        }
                    )
                for budget in BUDGETS:
                    stop = budget * 50
                    # Fair candidate normalizers use all available unlabeled prefix, identically.
                    z, mean, scale = normalized(raw, stop, channels, family)
                    stats = fitstats(z, c, stop)
                    names = ("prior", "adapt1", "adapt10", "daily1", "daily10", "scratch10")
                    outcomes = {}
                    for name in names:
                        w = fitarm(stats, prior, name)
                        p = project(z, w)
                        result = {
                            "session_id": sid,
                            "family": family,
                            "recency": recency,
                            "budget": budget,
                            "arm": name,
                            "history_ids": [cases[k]["item"]["session_id"] for k in ids],
                            "training_n": stats["n"],
                            "common302": metric(c["y"], p, evalmask),
                        }
                        pilots.append(result)
                        outcomes[name] = result["common302"]
                    # Validate on final third of the budget;2s gap; avoid random correlated-bin
                    # split.
                    cut = stop * 2 // 3
                    train_stop = max(1, cut - 100)
                    zv, _, _ = normalized(raw, train_stop, channels, family)
                    st = fitstats(zv, c, train_stop)
                    vm = c["eval_mask"].copy()
                    vm[:cut] = False
                    vm[stop:] = False
                    validation = {
                        name: metric(c["y"], project(zv, fitarm(st, prior, name)), vm)
                        for name in names
                    }
                    eligible_names = [name for name in names if validation[name]["n"] >= 50]
                    chosen = (
                        min(eligible_names, key=lambda name: validation[name]["mse"])
                        if eligible_names
                        else "prior"
                    )
                    choices.append(
                        {
                            "session_id": sid,
                            "family": family,
                            "recency": recency,
                            "budget": budget,
                            "train_stop": train_stop,
                            "validation_start": cut,
                            "validation_stop": stop,
                            "validation": validation,
                            "selected": chosen,
                            "selected_common302": outcomes[chosen],
                            "oracle_common302_arm": max(
                                names, key=lambda name: outcomes[name]["r2"] or -np.inf
                            ),
                            "coverage": direction(c["y"], c["eval_mask"], c["segments"], stop),
                        }
                    )
            print(f"{family} {sid} complete", flush=True)
    result = {
        "scope": (
            "Post-hoc exposed-only diagnosis, not fresh validation; no partici"
            "pant-specific deployed rule."
        ),
        "feature_bound": (
            "Only12 T2 prepared recordings with n_bins<=30000, already exposed V2/V3/V4 first600s."
        ),
        "historical_bound": (
            "Strictly earlier days, original V4 development histories only; la"
            "st1/3 before label availability."
        ),
        "pilot_plan": {
            "families": FAMILIES,
            "budgets": BUDGETS,
            "recency": [1, 3],
            "arms": ["prior", "adapt1", "adapt10", "daily1", "daily10", "scratch10"],
            "selection": (
                "last third labeled prefix with2s purge; refit at full budget; fin"
                "al302:600 only afterward"
            ),
        },
        "source_sha256": sha(__file__),
        "model_source_sha256": sha(ROOT / "neural_experiments/v4/model.py"),
        "study_source_sha256": sha(ROOT / "neural_experiments/v4/study.py"),
        "assessment_sha256": sha(manifest_path),
        "inputs": provenance,
        "original": original,
        "coverage": coverage,
        "feature_drift": features,
        "history_weights": priorweights,
        "pilots": pilots,
        "prefix_choices": choices,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            {"output": str(output), "sha256": sha(output), "seconds": result["elapsed_seconds"]}
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
