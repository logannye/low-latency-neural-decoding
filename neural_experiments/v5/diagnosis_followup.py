"""Post-hoc mapping/normalizer and prefix-validation diagnostics; exposed arrays only."""

import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_experiments.v4 import model, study
from neural_experiments.v5.diagnosis_pilot import (
    ROOT,
    fitarm,
    fitstats,
    metric,
    normalized,
    project,
    sha,
)


def balance_loss(y, p, mask, mode, segments):
    yy, pp = y[mask], p[mask]
    errors = np.square(yy - pp).mean(1)
    if not len(errors):
        return None
    if mode == "bins":
        return float(errors.mean())
    ids = np.floor((np.arctan2(yy[:, 1], yy[:, 0]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    if mode == "octants":
        return float(np.mean([errors[ids == k].mean() for k in range(8) if np.any(ids == k)]))
    losses = []
    for a, b in segments:
        edges = np.diff(np.r_[False, mask[a:b], False].astype(int))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            losses.append(float(np.square(y[a + s : a + e] - p[a + s : a + e]).mean()))
    return float(np.mean(losses)) if losses else None


def similarity(a, b):
    a, b = a.ravel(), b.ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    weighted = np.sqrt(np.abs(a * b))
    return {
        "cosine": float(a @ b / den) if den else None,
        "weighted_sign_agreement": float(np.sum(weighted * (a * b > 0)) / weighted.sum())
        if weighted.sum()
        else None,
    }


def main(
    output=ROOT / "results/neural_v5/diagnosis/t2_followup.json",
    parent=ROOT / "results/neural_v5/diagnosis/t2_pilot.json",
):
    start = time.monotonic()
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
    cases = []
    for path in indices:
        for item in json.loads(path.read_text())["sessions"]:
            if item["participant_id"] == "T2":
                if item["n_bins"] > 30000:
                    raise ValueError("Exposed cap")
                c = study.load_case(path, item)
                c["origin"] = "development" if path == indices[0] else "new_sessions"
                cases.append(c)
    cases.sort(key=lambda c: c["item"]["post_implant_day"])
    cfg = {
        "family": "tx45_sbp",
        "tau_seconds": 0.2,
        "baseline_tau_seconds": None,
        "adapt_gamma": 10.0,
    }
    prep = []
    for c in cases:
        raw = model.representation(c["features"], c["segments"], cfg)
        z, mu, sd = normalized(raw, len(raw), c["item"]["n_channels"], "tx45_sbp")
        st = fitstats(z, c, len(raw), 5)
        prep.append(
            {
                "raw": raw,
                "stats": st,
                "weights": model.solve_history([st], z.shape[1]),
                "mean": mu,
                "scale": sd,
            }
        )
    pairs = []
    for i in range(len(cases)):
        for j in range(i):
            ch = cases[i]["item"]["n_channels"]
            w, q = prep[i]["weights"], prep[j]["weights"]
            pairs.append(
                {
                    "earlier": cases[j]["item"]["session_id"],
                    "later": cases[i]["item"]["session_id"],
                    "tx": similarity(w[:ch], q[:ch]),
                    "sbp": similarity(w[ch:-1], q[ch:-1]),
                    "all": similarity(w[:-1], q[:-1]),
                }
            )
    normalization = []
    validation = []
    sourcefit = []
    names = ("prior", "adapt1", "adapt10", "daily1", "daily10", "scratch10", "zero")
    for j, c in enumerate(cases):
        raw = prep[j]["raw"]
        n = len(raw)
        ch = c["item"]["n_channels"]
        sid = c["item"]["session_id"]
        suffix = c["eval_mask"] & (np.arange(n) >= 15100)
        earlier = [k for k in range(j) if cases[k]["origin"] == "development"][-3:]
        prior = model.solve_history([prep[k]["stats"] for k in earlier], raw.shape[1] + 1)
        ownz, _, _ = normalized(raw, n, ch, "tx45_sbp")
        sourcefit.append(
            {
                "session_id": sid,
                "optimistic_full600_fit": metric(
                    c["y"], project(ownz, prep[j]["weights"]), c["eval_mask"]
                ),
            }
        )
        if earlier:
            # One shared, entirely historical transform tests the standardized-coordinate
            # explanation.
            allraw = np.concatenate([prep[k]["raw"] for k in earlier])
            _, mu, sd = normalized(allraw, len(allraw), ch, "tx45_sbp")
            stats = []
            for k in earlier:
                zz = np.column_stack(
                    (np.clip((prep[k]["raw"] - mu) / sd, -8, 8), np.ones(len(prep[k]["raw"])))
                )
                stats.append(fitstats(zz, cases[k], len(zz), 5))
            shared = model.solve_history(stats, raw.shape[1] + 1)
            for normname, mm, ss, ww in [
                ("shared_historical", mu, sd, shared),
                ("latest_historical", prep[earlier[-1]]["mean"], prep[earlier[-1]]["scale"], prior),
            ]:
                zz = np.column_stack((np.clip((raw - mm) / ss, -8, 8), np.ones(n)))
                normalization.append(
                    {
                        "session_id": sid,
                        "normalizer": normname,
                        "common302": metric(c["y"], project(zz, ww), suffix),
                    }
                )
        for budget in [30, 60, 120, 240]:
            stop = budget * 50
            z, mu, sd = normalized(raw, stop, ch, "tx45_sbp")
            stats = fitstats(z, c, stop)
            targetweights = {
                name: (np.zeros_like(prior) if name == "zero" else fitarm(stats, prior, name))
                for name in names
            }
            outcome = {
                name: metric(c["y"], project(z, w), suffix) for name, w in targetweights.items()
            }
            cut = 2 * stop // 3
            train_stop = cut - 100
            vz, _, _ = normalized(raw, train_stop, ch, "tx45_sbp")
            st = fitstats(vz, c, train_stop)
            vm = c["eval_mask"].copy()
            vm[:cut] = False
            vm[stop:] = False
            valpred = {
                name: project(
                    vz, np.zeros_like(prior) if name == "zero" else fitarm(st, prior, name)
                )
                for name in names
            }
            for mode in ["bins", "octants", "go_runs"]:
                loss = {
                    name: balance_loss(c["y"], p, vm, mode, c["segments"])
                    for name, p in valpred.items()
                }
                selected = min(names, key=lambda name: loss[name])
                validation.append(
                    {
                        "session_id": sid,
                        "budget": budget,
                        "validation_unit": mode,
                        "loss": loss,
                        "selected": selected,
                        "zero_validation_gain": 1 - loss[selected] / loss["zero"],
                        "selected_common302": outcome[selected],
                        "all_common302": outcome,
                    }
                )
        print(sid, flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": (
            "Post-hoc exposed-first600 T2 followup; no fresh claim. Full600 fi"
            "ts are explicitly optimistic diagnostic fits, never deployment ca"
            "ndidates."
        ),
        "source_sha256": sha(__file__),
        "helper_source_sha256": sha(Path(__file__).with_name("diagnosis_pilot.py")),
        "parent_pilot_sha256": sha(parent),
        "mapping_pairs": pairs,
        "normalizer_controls": normalization,
        "source_fit": sourcefit,
        "prefix_validation": validation,
        "elapsed_seconds": time.monotonic() - start,
    }
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"sha256": sha(output), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
