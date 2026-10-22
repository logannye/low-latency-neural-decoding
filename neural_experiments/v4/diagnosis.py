"""POST HOC T3 pilots: only V3 prepared, previously exposed <=600 s files."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from neural_experiments.v3 import model

EXPOSED_INDEXES = {
    "v3_development": "10bf61377fc43b1b6e77a4ae7238d80cde8894bb69af6f68388f54b8a5ac98dd",
    "v3_assessment": "8850117ea21bfe939e869b53a49d5b8ccf7d40e905d51936d503759310f04c4d",
    "v4_development": "aa6bf303c746edde0fb4d0a3a937a0fb07afb849270f2c79093fc021a9c2c500",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_exposed_index(path, role):
    if role not in EXPOSED_INDEXES or sha(path) != EXPOSED_INDEXES[role]:
        raise ValueError("diagnosis requires an exact previously exposed index")
    return json.loads(Path(path).read_text())


def metric(y, p, mask):
    a, b = y[mask].astype(float), p[mask]
    if len(a) < 50:
        return {"n": len(a), "r2": None}
    residual = b - a
    sse = np.square(residual).sum()
    norm = np.linalg.norm(b, axis=1)
    cos = np.divide(
        (a * b).sum(1), np.linalg.norm(a, axis=1) * norm, out=np.zeros(len(a)), where=norm > 1e-12
    )
    centered = []
    for i in range(2):
        centered.append(
            float(np.corrcoef(a[:, i], b[:, i])[0, 1])
            if b[:, i].std() > 1e-12 and a[:, i].std() > 1e-12
            else None
        )
    return {
        "n": len(a),
        "r2": float(r2_score(a, b, multioutput="variance_weighted")),
        "angular_degrees": float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()),
        "prediction_mean": b.mean(0).tolist(),
        "target_mean": a.mean(0).tolist(),
        "mean_bias_sse_fraction": float(len(a) * np.square(residual.mean(0)).sum() / sse),
        "rms_norm": float(np.sqrt(np.square(b).sum(1).mean())),
        "centered_correlations": centered,
    }


def target_stats(y, mask, q, start, stop):
    yy, mm, qq = y[start:stop].astype(float), mask[start:stop], q[start:stop]
    a = yy[mm]
    if not len(a):
        return {"n": 0}
    hist = np.histogram(np.arctan2(a[:, 1], a[:, 0]), bins=np.linspace(-np.pi, np.pi, 9))[0]
    prob = hist / hist.sum()
    cs = np.r_[0, np.cumsum(~mm)]
    autocorrelation = []
    for lag in (1, 5, 25, 50, 250, 500):
        if lag >= len(yy):
            continue
        valid = (qq[lag:] == qq[:-lag]) & (cs[lag + 1 :] == cs[: -(lag + 1)])
        dots = (yy[lag:][valid] * yy[:-lag][valid]).sum(1)
        autocorrelation.append(
            {
                "lag_seconds": lag * 0.02,
                "n_contiguous_go_pairs": len(dots),
                "mean_direction_dot": float(dots.mean()) if len(dots) else None,
            }
        )
    return {
        "n": len(a),
        "observed_seconds": (stop - start) * 0.02,
        "go_fraction": float(mm.mean()),
        "mean_direction": a.mean(0).tolist(),
        "contiguous_go_runs": int(np.count_nonzero(mm & np.r_[True, ~mm[:-1]])),
        "mean_direction_norm": float(np.linalg.norm(a.mean(0))),
        "octant_counts": hist.tolist(),
        "normalized_octant_entropy": float(
            -(prob[prob > 0] * np.log(prob[prob > 0])).sum() / np.log(8)
        ),
        "autocorrelation": autocorrelation,
    }


def feature_stats(tx, sbp, filtered, mean, scale, start, stop):
    x, s = tx[start:stop], sbp[start:stop]
    z = (filtered[start:stop] - mean) / scale
    c = tx.shape[1]
    return {
        "bins": stop - start,
        "count_nonzero_fraction": float((x > 0).mean()),
        "count_zero_channels": int(np.count_nonzero(x.sum(0) == 0)),
        "mean_count_rate_hz": float(x.mean() / 0.02),
        "channel_rate_hz_quantiles": np.quantile(x.mean(0) / 0.02, [0, 0.5, 0.9, 1]).tolist(),
        "sbp_nonzero_fraction": float((s > 0).mean()),
        "sbp_quantiles": np.quantile(s, [0, 0.5, 0.99, 1]).tolist(),
        "tx_mean_displacement_rms_prefix_z": float(np.sqrt(np.square(z[:, :c].mean(0)).mean())),
        "sbp_mean_displacement_rms_prefix_z": float(np.sqrt(np.square(z[:, c:].mean(0)).mean())),
        "tx_fraction_abs_z_gt8": float((np.abs(z[:, :c]) > 8).mean()),
        "sbp_fraction_abs_z_gt8": float((np.abs(z[:, c:]) > 8).mean()),
    }


def configs():
    base = {
        "family": "combined",
        "tau": 0.2,
        "slow": None,
        "penalty": 1.0,
        "intercept_multiplier": 1.0,
        "sbp_common_mode": False,
        "sbp_past_center_seconds": None,
    }
    variants = [
        ("combined_v3", {}),
        ("tx_matched", {"family": "tx"}),
        ("sbp_only", {"family": "sbp"}),
        ("tx_v3", {"family": "tx", "tau": 0.4}),
        ("combined_lambda01", {"penalty": 0.1}),
        ("combined_lambda10", {"penalty": 10.0}),
        ("combined_lambda100", {"penalty": 100.0}),
        ("combined_tau04", {"tau": 0.4}),
        ("combined_slow30", {"slow": 30.0}),
        ("sbp_slow30", {"family": "sbp", "slow": 30.0}),
        ("combined_intercept025", {"intercept_multiplier": 0.25}),
        ("combined_intercept0", {"intercept_multiplier": 0.0}),
        ("tx_intercept0", {"family": "tx", "intercept_multiplier": 0.0}),
        ("sbp_commonmode", {"family": "sbp", "sbp_common_mode": True}),
        ("sbp_past5", {"family": "sbp", "sbp_past_center_seconds": 5.0}),
        (
            "sbp_commonmode_past5",
            {"family": "sbp", "sbp_common_mode": True, "sbp_past_center_seconds": 5.0},
        ),
        ("zero", {"family": "zero"}),
        ("prefix_mean", {"family": "prefix_mean"}),
    ]
    return [{"name": name, **base, **delta} for name, delta in variants]


def run(output, development_index, assessment_index, prior_results):
    if output.exists():
        raise FileExistsError(output)
    began = time.monotonic()
    result = {
        "scope": (
            "POST HOC exploratory T3 only: all eight previously exposed sessions; "
            "no confirmatory inference."
        ),
        "max_bins_opened_per_prepared_file": 30000,
        "raw_files_opened": 0,
        "budgets_seconds": [30, 120],
        "common_scoring_seconds": [302, 600],
        "fit": (
            "mean-loss ridge; prefix-only all-observation mean/std, floors TX .1 SBP .01; "
            "common 62-bin/block training burn; feature clip8; unit-disk output; "
            "no mask in filtering"
        ),
        "intercept_shrink_definition": (
            "Multiply the fitted standardized-feature intercept; post-hoc deployment-available "
            "rule, no evaluator labels used in transform."
        ),
        "sbp_recenter_definition": (
            "Optionally subtract each current logSBP vector's channel mean, then subtract a "
            "per-channel 5s EW mean from PREVIOUS bins (initialize to first observed vector at "
            "each block); update baseline with current bin only after subtraction; "
            "apply .2s fast filter afterward."
        ),
        "prior_public_results_sha256": sha(prior_results),
        "configs": configs(),
        "indexes": [],
        "sessions": [],
        "pilots": [],
    }
    for role in ("development", "assessment"):
        path = development_index if role == "development" else assessment_index
        index = read_exposed_index(path, f"v3_{role}")
        result["indexes"].append({"role": role, "sha256": sha(path)})
        for item in index["sessions"]:
            if item["participant_id"] != "T3":
                continue
            assert item["n_bins"] <= 30000
            fp, tp = path.parent / item["feature_path"], path.parent / item["target_path"]
            assert sha(fp) == item["feature_sha256"] and sha(tp) == item["target_sha256"]
            with np.load(fp, allow_pickle=False) as z:
                tx, sbp, q = z["x_tx"], z["x_sbp"], z["sequence_id"]
            with np.load(tp, allow_pickle=False) as z:
                y, mask = z["y"], z["eval_mask"]
            assert len(tx) == len(sbp) == len(y) == len(mask) == len(q) == item["n_bins"]
            segments = [(s["start"], s["end"]) for s in item["segments"] if s["end"] > s["start"]]
            assert np.array_equal(
                np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1, len(q)],
                [segments[0][0]] + [b for a, b in segments],
            )
            transformed = np.concatenate(
                (np.sqrt(tx.astype(float)), np.log1p(sbp.astype(float))), axis=1
            )
            c = tx.shape[1]
            representations = {}
            for cfg in configs():
                key = (
                    cfg["tau"],
                    cfg["slow"],
                    cfg["sbp_common_mode"],
                    cfg["sbp_past_center_seconds"],
                )
                if key not in representations:
                    transformed_now = transformed.copy()
                    if cfg["sbp_common_mode"]:
                        transformed_now[:, c:] -= transformed_now[:, c:].mean(1, keepdims=True)
                    if cfg["sbp_past_center_seconds"]:
                        baseline = model.exponential(
                            transformed_now[:, c:], segments, cfg["sbp_past_center_seconds"]
                        )
                        past = np.empty_like(baseline)
                        for a, b in segments:
                            past[a] = transformed_now[a, c:]
                            past[a + 1 : b] = baseline[a : b - 1]
                        transformed_now[:, c:] -= past
                    value = model.exponential(transformed_now, segments, cfg["tau"])
                    if cfg["slow"] is not None:
                        value -= model.exponential(transformed_now, segments, cfg["slow"])
                    representations[key] = value
            raw = representations[(0.2, None, False, None)]
            mean = raw[:6000].mean(0)
            std = raw[:6000].std(0)
            scale = np.maximum(std, np.r_[np.full(c, 0.1), np.full(c, 0.01)])
            session = {
                "session_id": item["session_id"],
                "formerly": role,
                "n_bins": len(tx),
                "channels": c,
                "feature_sha256": sha(fp),
                "target_sha256": sha(tp),
                "prefix120_tx_filtered_std_below_floor_fraction": float((std[:c] < 0.1).mean()),
                "prefix120_sbp_filtered_std_below_floor_fraction": float((std[c:] < 0.01).mean()),
                "prefix120_tx_raw_sqrt_std_below_floor_fraction": float(
                    (np.sqrt(tx[:6000]).std(0) < 0.1).mean()
                ),
                "windows": [],
                "blocks": [],
            }
            windows = [
                ("prefix30", 0, 1500),
                ("prefix120", 0, 6000),
                ("common_suffix", 15100, len(tx)),
            ]
            for name, a, b in windows:
                session["windows"].append(
                    {
                        "name": name,
                        "start_bin": a,
                        "stop_bin": b,
                        "targets": target_stats(y, mask, q, a, b),
                        "features": feature_stats(tx, sbp, raw, mean, scale, a, b),
                    }
                )
            for i, (a, b) in enumerate(segments):
                session["blocks"].append(
                    {
                        "sequence_id": int(q[a]),
                        "start_bin": a,
                        "stop_bin": b,
                        "targets": target_stats(y, mask, q, a, b),
                        "features": feature_stats(tx, sbp, raw, mean, scale, a, b),
                    }
                )
            result["sessions"].append(session)
            for budget in result["budgets_seconds"]:
                n = int(budget / 0.02)
                ps = [(a, min(b, n)) for a, b in segments if a < n]
                usable = model.training_mask(mask[:n], ps)
                for cfg in configs():
                    raw = representations[
                        (
                            cfg["tau"],
                            cfg["slow"],
                            cfg["sbp_common_mode"],
                            cfg["sbp_past_center_seconds"],
                        )
                    ]
                    ix = (
                        slice(None, c)
                        if cfg["family"] == "tx"
                        else slice(c, None)
                        if cfg["family"] == "sbp"
                        else slice(None)
                    )
                    raw = raw[:, ix]
                    floor = np.r_[np.full(c, 0.1), np.full(c, 0.01)][ix]
                    mean, scale = raw[:n].mean(0), np.maximum(raw[:n].std(0), floor)
                    z = np.clip((raw - mean) / scale, -8, 8)
                    xx, yy = z[:n][usable], y[:n][usable].astype(float)
                    xm, ym = xx.mean(0), yy.mean(0)
                    xc, yc = xx - xm, yy - ym
                    gram = xc.T @ xc / len(xx)
                    gram.flat[:: len(gram) + 1] += cfg["penalty"]
                    coef = np.linalg.solve(gram, xc.T @ yc / len(xx))
                    intercept = (ym - xm @ coef) * cfg["intercept_multiplier"]
                    pred = model.project_direction(z @ coef + intercept)
                    if cfg["family"] in ("zero", "prefix_mean"):
                        intercept = np.zeros(2) if cfg["family"] == "zero" else yy.mean(0)
                        pred = np.broadcast_to(intercept, (len(y), 2))
                    row = {
                        "session_id": item["session_id"],
                        "formerly": role,
                        "budget_seconds": budget,
                        "config": cfg["name"],
                        "training_bins": len(xx),
                        "fitted_intercept_norm": float(np.linalg.norm(intercept)),
                        "training": metric(y[:n], pred[:n], usable),
                        "blocks_after_prefix": [],
                    }
                    for name, a, b in [
                        ("primary", n + 100, len(tx)),
                        ("common_suffix", 15100, len(tx)),
                    ]:
                        scoring = mask & (np.arange(len(tx)) >= a) & (np.arange(len(tx)) < b)
                        row[name] = metric(y, pred, scoring)
                    for a, b in segments:
                        a = max(a, n + 100)
                        if a >= b:
                            continue
                        scoring = mask & (np.arange(len(tx)) >= a) & (np.arange(len(tx)) < b)
                        row["blocks_after_prefix"].append(
                            {
                                "sequence_id": int(q[a]),
                                "start_bin": a,
                                "stop_bin": b,
                                **metric(y, pred, scoring),
                            }
                        )
                    result["pilots"].append(row)
            print(item["session_id"], "done", flush=True)
    result["summary"] = []
    for cohort in ("all_exposed", "formerly_development", "formerly_assessment"):
        for budget in result["budgets_seconds"]:
            for cfg in configs():
                rows = [
                    r
                    for r in result["pilots"]
                    if r["budget_seconds"] == budget
                    and r["config"] == cfg["name"]
                    and (cohort == "all_exposed" or "formerly_" + r["formerly"] == cohort)
                ]
                result["summary"].append(
                    {
                        "cohort": cohort,
                        "budget_seconds": budget,
                        "config": cfg["name"],
                        "sessions": len(rows),
                        "mean_r2_common_suffix": float(
                            np.mean([r["common_suffix"]["r2"] for r in rows])
                        ),
                        "mean_r2_training": float(np.mean([r["training"]["r2"] for r in rows])),
                        "mean_angular_common_suffix": float(
                            np.mean([r["common_suffix"]["angular_degrees"] for r in rows])
                        ),
                    }
                )
    published = json.loads(prior_results.read_text())
    differences = []
    for identity in published["models"]:
        if (
            identity["target_participant"] != "T3"
            or identity["method"] != "selected_daily"
            or identity["budget_seconds"] != 120
        ):
            continue
        row = next(
            r
            for r in result["pilots"]
            if r["session_id"] == identity["session_id"]
            and r["budget_seconds"] == 120
            and r["config"] == "combined_v3"
        )
        for trace in identity["traces"]:
            differences.append(abs(trace["metrics"]["r2"] - row[trace["partition"]]["r2"]))
    assert len(differences) == 4 and max(differences) < 1e-10
    result["reproduction_check"] = {
        "original_v3_scores_checked": 4,
        "max_absolute_r2_error": max(differences),
    }
    result.update(
        source_sha256=sha(__file__),
        frozen_v3_model_source_sha256=sha(model.__file__),
        elapsed_seconds=time.monotonic() - began,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output, result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-index", type=Path, required=True)
    parser.add_argument("--assessment-index", type=Path, required=True)
    parser.add_argument("--prior-results", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output, args.development_index, args.assessment_index, args.prior_results)
