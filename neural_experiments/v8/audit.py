"""Independent V8 arithmetic, information-boundary and artifact verification.

No V8 producer/model/importer/scorer is imported. The independent earlier
sklearn score reference and causal recurrence are reused; this is a separate
implementation audit, not independent biological replication.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit import compare_structure, score_arrays
from neural_experiments.v3.audit import contained, numeric, read, segment_ranges, sha256, verify
from neural_experiments.v4.audit import reference_representation
from neural_experiments.v7.audit import average, burned_mask, unique

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = "01fc7756fe3110630a98f78152a08f85b2119d76c211196da93beb23f405a392"
ACQUISITION_SHA = "64ae9b11167d402fdc7e629cdd9b8c98df52c3fbf708e53111dfc3588fc383ac"
PARENTS = {
    "development": "4b2c0c8b47a2caf1a2135b28dcea197a26c8f6006c19f243ff2fed23c5e1362e",
    "fresh": "34f51825b1f1b4b7f8144fcfe93822f61295a2a013f0cd564fff610228d20e47",
}
PEOPLE = ["T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11"]
BASELINE = {
    "family": "tx45_sbp",
    "taus_seconds": [0.2],
    "ridge": 1.0,
    "phase": "all",
    "weighting": "uniform",
    "target": "instantaneous",
    "control": "neural",
}
FAR = {
    "T2": 0.2,
    "T3": 0.2,
    "T5": 160.0,
    "T6": 160.0,
    "T7": 0.2,
    "T8": 5.6,
    "T9": 0.2,
    "T10": 0.2,
    "T11": 0.2,
}
FAR_SPECIAL = {
    f"T5_day_{d}_decoding": 0.4 for d in (2107, 2126, 2133, 2149, 2595, 2609, 2637, 2667)
}


def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def artifact(path):
    value = Path(path)
    value = value.resolve() if value.is_absolute() else contained(ROOT, value)
    if not value.is_relative_to(ROOT):
        raise ValueError("Artifact outside repository")
    return value


def config_reference(config):
    if (
        set(config) != set(BASELINE)
        or config["family"] not in ("tx45_sbp", "tx3_sbp")
        or config["taus_seconds"] not in ([0.05], [0.2], [0.5], [0.05, 0.5])
        or type(config["ridge"]) not in (int, float)
        or config["ridge"] not in (1, 10)
        or config["phase"] not in ("all", "clean")
        or config["weighting"] not in ("uniform", "equal_trial")
        or config["target"] not in ("instantaneous", "onset")
        or config["control"] not in ("neural", "zero", "intercept", "trial_permutation")
        or config["target"] == "onset"
        and config["phase"] != "clean"
    ):
        raise ValueError("Unregistered model configuration")


def representation_reference(case, config):
    config_reference(config)
    return np.concatenate(
        [
            reference_representation(
                case["features"],
                case["segments"],
                {
                    "family": config["family"],
                    "tau_seconds": tau,
                    "baseline_tau_seconds": None,
                    "adapt_gamma": 10.0,
                },
            )
            for tau in config["taus_seconds"]
        ],
        axis=1,
    )


def masks_reference(case, config=None):
    c = case["context"]
    ids = c["trial_id"]
    valid = burned_mask(case["eval_mask"], case["segments"]) & (ids >= 0)
    clean = (
        c["reference40_available"]
        & (c["reference40_distance"] > case["item"]["far_distance"])
        & (c["trial_age_seconds"] >= 0.2)
        & (c["trial_age_seconds"] < 1)
        & (ids >= 0)
    )
    purchased = sorted(set(int(v) for v in ids[1500:7500] if v >= 0))
    primary = valid & ~np.isin(ids, purchased)
    primary[:7600] = False
    secondary = primary & clean
    train = valid.copy()
    train[:1500] = False
    train[7500:] = False
    if config is not None and config["phase"] == "clean":
        train &= clean & (c["trial_onset_seconds"] >= 30.0)
    return train, primary, secondary, purchased


def permutation_reference(y, ids):
    order = sorted(set(ids.tolist()), key=lambda t: int(np.flatnonzero(ids == t)[0]))
    out, mapping = y.copy(), []
    for i, recipient in enumerate(order):
        donor = order[(i + 1) % len(order)]
        a, b = np.flatnonzero(ids == recipient), np.flatnonzero(ids == donor)
        out[a] = y[b[(np.arange(len(a)) * len(b)) // len(a)]]
        mapping.append(
            {
                "recipient_trial": int(recipient),
                "donor_trial": int(donor),
                "recipient_rows": len(a),
                "donor_rows": len(b),
            }
        )
    return out, mapping


def fit_reference(case, config, raw):
    """Return declared normal equations and exact purchased-label fit evidence."""
    config_reference(config)
    train, _, _, _ = masks_reference(case, config)
    n = len(case["y"])
    ix = np.flatnonzero(train)
    ids = case["context"]["trial_id"][ix]
    yy = (
        case["y"]
        if config["target"] == "instantaneous"
        else case["context"]["reference40_direction"]
    )
    labels = np.asarray(yy[ix], np.float64)
    original = labels.copy()
    if not np.isfinite(labels).all():
        raise ValueError("Nonfinite purchased labels")
    trials, counts = np.unique(ids, return_counts=True)
    weights = np.ones(len(ix), np.float64)
    if config["weighting"] == "equal_trial" and len(ix):
        counts_by_id = dict(zip(trials, counts, strict=True))
        weights = np.asarray([1.0 / counts_by_id[t] for t in ids])
        weights *= len(ix) / weights.sum()
    mapping = []
    if config["control"] == "trial_permutation":
        labels, mapping = permutation_reference(labels, ids)
    c = case["features"]["x_tx45"].shape[1]
    floor = np.tile(np.r_[np.full(c, 0.1), np.full(c, 0.01)], len(config["taus_seconds"]))
    mean = raw[:1500].mean(0) if n >= 1500 else np.zeros(raw.shape[1])
    scale = np.maximum(raw[:1500].std(0), floor) if n >= 1500 else floor
    if not np.isfinite(raw[: min(7500, n)]).all():
        raise ValueError("Nonfinite observed input prefix")
    divisor = float(np.sqrt(len(config["taus_seconds"])))
    z = np.clip((raw[ix] - mean) / scale, -8, 8) / divisor
    design = np.c_[z, np.ones(len(ix))]
    total = float(weights.sum())
    fallback = (
        n < 7500 or len(ix) < 50 or config["control"] == "trial_permutation" and len(trials) < 2
    )
    diagonal = np.r_[np.full(raw.shape[1], config["ridge"]), 0.0]
    gram = (
        (design.T * weights) @ design / total if total else np.zeros((len(diagonal), len(diagonal)))
    )
    rhs = (design.T * weights) @ labels / total if total else np.zeros((len(diagonal), 2))
    trial_mass = np.asarray([weights[ids == t].sum() for t in trials])
    sectors = (
        np.floor((np.arctan2(original[:, 1], original[:, 0]) + np.pi) / (np.pi / 4)).astype(int) % 8
    )
    receipt = {
        "config": config,
        "normalization_start_bin": 0,
        "normalization_stop_bin": 1500,
        "label_start_bin": 1500,
        "label_stop_bin": 7500,
        "last_training_bin": int(ix[-1]) if len(ix) else None,
        "training_rows": len(ix),
        "training_trials": len(trials),
        "training_mask_sha256": ahash(train),
        "training_weights_sha256": ahash(weights),
        "training_labels_sha256": ahash(labels),
        "weight_sum": total,
        "effective_sample_size": total**2 / float(weights @ weights) if total else 0.0,
        "effective_trials": total**2 / float(trial_mass @ trial_mass) if total else 0.0,
        "direction_octant_rows": np.bincount(sectors, minlength=8).tolist(),
        "direction_octants_with_at_least50_rows": int(
            (np.bincount(sectors, minlength=8) >= 50).sum()
        ),
        "permutation": mapping,
        "fallback_zero": bool(fallback),
        "feature_penalty": config["ridge"],
        "free_intercept": True,
        "multiscale_design_divisor": divisor,
        "export_prediction_max_error": 0.0,
    }
    return {
        "mask": train,
        "sample_weights": weights,
        "mean": mean,
        "scale": scale,
        "gram": gram + np.diag(diagonal),
        "rhs": rhs,
        "labels": labels,
        "receipt": receipt,
    }


def validate_fit(arrays, receipt, case, config, raw):
    if (
        set(arrays) != {"version", "config", "mean", "scale", "weights"}
        or arrays["version"].dtype != np.int64
        or arrays["version"].shape != ()
        or arrays["version"].item() != 8
        or arrays["config"].dtype.kind != "U"
        or arrays["config"].shape != ()
        or json.loads(arrays["config"].item()) != config
    ):
        raise ValueError("Wrong portable model version/configuration")
    ref = fit_reference(case, config, raw)
    compare_structure(receipt, ref["receipt"], "purchased fit receipt")
    d = raw.shape[1]
    for field, shape in (("mean", (d,)), ("scale", (d,)), ("weights", (d + 1, 2))):
        if (
            arrays[field].shape != shape
            or arrays[field].dtype != np.float64
            or not np.isfinite(arrays[field]).all()
        ):
            raise ValueError("Invalid numeric model geometry/dtype")
    for field in ("mean", "scale"):
        if not np.allclose(arrays[field], ref[field], rtol=1e-12, atol=1e-12):
            raise ValueError("Wrong prefix-only normalization")
    w = arrays["weights"]
    if ref["receipt"]["fallback_zero"] or config["control"] == "zero":
        if np.any(w != 0):
            raise ValueError("Required zero fallback/control changed")
        residual = 0.0
    elif config["control"] == "intercept":
        if np.any(w[:-1] != 0) or not np.allclose(w[-1], ref["rhs"][-1], atol=1e-12, rtol=1e-12):
            raise ValueError("Not the purchased-label mean control")
        residual = 0.0
    else:
        residual = float(
            np.linalg.norm(ref["gram"] @ w - ref["rhs"]) / max(1.0, np.linalg.norm(ref["rhs"]))
        )
        if residual > 1e-9:
            raise ValueError("Not the declared weighted ridge optimum")
    return ref, residual


def replay_reference(arrays, raw, config):
    z = np.clip((raw - arrays["mean"]) / arrays["scale"], -8, 8) / np.sqrt(
        len(config["taus_seconds"])
    )
    prediction = z @ arrays["weights"][:-1] + arrays["weights"][-1]
    prediction /= np.maximum(1.0, np.linalg.norm(prediction, axis=1))[:, None]
    prediction[:7600] = 0
    return prediction


def score_reference(y, p, mask):
    r = score_arrays(y, p, mask)
    if r["n"] < 50:
        r.update(r2=None, per_output_r2=[None, None], angular_error_degrees=None)
    if r["n"] >= 50:
        yy, pp = np.asarray(y, float)[mask], np.asarray(p, float)[mask]
        r.update(
            sse=np.square(yy - pp).sum(0).tolist(),
            sst=np.square(yy - yy.mean(0)).sum(0).tolist(),
            prediction_norm_rms=float(np.sqrt(np.square(pp).sum(1).mean())),
        )
    r["summed_mse"] = (
        float(np.square(np.asarray(y, float)[mask] - np.asarray(p, float)[mask]).sum() / mask.sum())
        if r["r2"] is not None
        else None
    )
    return r


def trial_score_reference(y, p, mask, ids):
    trials = np.unique(ids[mask])
    if len(trials) < 2:
        return {
            "trials": len(trials),
            "r2": None,
            "summed_mse": None,
            "angular_error_degrees": None,
        }
    a = np.stack([np.asarray(y, float)[mask & (ids == t)].mean(0) for t in trials])
    b = np.stack([np.asarray(p, float)[mask & (ids == t)].mean(0) for t in trials])
    sse = float(np.square(a - b).sum())
    sst = float(np.square(a - a.mean(0)).sum())
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide(np.sum(a * b, axis=1), denom, out=np.zeros(len(a)), where=denom > 1e-12)
    return {
        "trials": len(trials),
        "r2": 1 - sse / sst if sst > 1e-12 else None,
        "summed_mse": sse / len(trials),
        "angular_error_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean()),
    }


def check_session(case, configurations, rows, session_receipt):
    table = unique(rows, lambda r: r["method"])
    if (
        set(table) != set(configurations)
        or set(session_receipt["fit_receipts"]) != set(table)
        or set(session_receipt["model_refs"]) != set(table)
    ):
        raise ValueError("Incomplete session/model inventory")
    trace_refs = {(r["trace"]["trace_path"], r["trace"]["trace_sha256"]) for r in rows}
    if len(trace_refs) != 1:
        raise ValueError("Session must bind one common trace")
    path, pin = next(iter(trace_refs))
    path = artifact(path)
    verify(path, pin)
    trace = numeric(path)
    expected_keys = {"y", "onset_y", "primary_mask", "secondary_mask"} | {
        m + "__" + f for m in table for f in ("prediction", "training_mask", "training_weights")
    }
    if set(trace) != expected_keys:
        raise ValueError("Unexpected/missing trace fields")
    _, primary, secondary, purchased = masks_reference(case)
    for name, expected in (
        ("y", case["y"]),
        ("onset_y", case["context"]["reference40_direction"]),
        ("primary_mask", primary),
        ("secondary_mask", secondary),
    ):
        if not np.array_equal(trace[name], expected):
            raise ValueError("Saved target/mask changed")
    cache, errors, residuals, rebuilt = {}, [], [], []
    for method, config in configurations.items():
        row = table[method]
        key = (config["family"], tuple(config["taus_seconds"]))
        if key not in cache:
            cache[key] = representation_reference(case, config)
        raw = cache[key]
        compare_structure(row["model"], session_receipt["model_refs"][method], "model ref")
        compare_structure(
            row["training"], session_receipt["fit_receipts"][method], "duplicate fit receipt"
        )
        path = artifact(row["model"]["model_path"])
        verify(path, row["model"]["model_sha256"])
        arrays = numeric(path)
        ref, residual = validate_fit(arrays, row["training"], case, config, raw)
        pred = replay_reference(arrays, raw, config)
        saved = trace[method + "__prediction"]
        if (
            saved.dtype != np.float64
            or saved.shape != pred.shape
            or not np.isfinite(saved).all()
            or not np.allclose(pred, saved, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("Saved prediction differs from independent full replay")
        for name, value in (
            ("training_mask", ref["mask"]),
            ("training_weights", ref["sample_weights"]),
        ):
            if not np.array_equal(trace[method + "__" + name], value):
                raise ValueError("Saved purchased training rows/weights changed")
        item = case["item"]
        expected = {
            "participant": item["participant_id"],
            "session_id": item["session_id"],
            "post_implant_day": item["post_implant_day"],
            "method": method,
            "n_bins": len(pred),
            "config": config,
            "target": config["target"],
            "metrics": score_reference(case["y"], pred, primary),
            "score_mask_sha256": ahash(primary),
            "valid_rows": int(primary.sum()),
            "secondary": {
                "instantaneous": score_reference(case["y"], pred, secondary),
                "onset": score_reference(case["context"]["reference40_direction"], pred, secondary),
                "score_mask_sha256": ahash(secondary),
                "valid_rows": int(secondary.sum()),
                "endpoint": "fixed40ms_trial_direction_on_early_far_trial_bins",
                "trial_averaged": {
                    "instantaneous": trial_score_reference(
                        case["y"], pred, secondary, case["context"]["trial_id"]
                    ),
                    "onset": trial_score_reference(
                        case["context"]["reference40_direction"],
                        pred,
                        secondary,
                        case["context"]["trial_id"],
                    ),
                },
            },
            "planned_label_seconds": 120.0,
            "available_label_seconds": max(0, min(len(pred), 7500) - 1500) * 0.02,
            "training": ref["receipt"],
            "excluded_purchased_trial_count": len(purchased),
            "model": row["model"],
            "trace": row["trace"],
        }
        compare_structure(row, expected, "model scores and geometry")
        rebuilt.append(expected)
        errors.append(float(np.max(np.abs(saved - pred))) if len(pred) else 0.0)
        residuals.append(residual)
    return rebuilt, errors, residuals


class Inputs:
    """Separate metadata/array reader; bind the independently rebuilt context receipt."""

    def __init__(self, path, data_audit):
        self.path = Path(path).resolve()
        self.index = read(self.path)
        self.checked = set()
        self.data_audit_path = Path(data_audit).resolve()
        receipt = read(self.data_audit_path)
        index = self.index
        role = index["role"]
        if (
            role not in ("development", "fresh")
            or index["schema_version"] != 1
            or index["provider"] != {"doi": "10.5061/dryad.x0k6djj1h", "version_id": 462702}
            or receipt["status"] != "PASS"
            or receipt["role"] != role
            or receipt["index_sha256"] != sha256(self.path)
            or receipt["auditor_sha256"] != sha256(ROOT / "neural_experiments/v8/audit_data.py")
        ):
            raise ValueError("Data audit must pass and bind exact role/index/reference")
        self.items = unique(index["sessions"], lambda r: r["session_id"])
        audited = unique(receipt["records"], lambda r: r["session_id"])
        if set(self.items) != set(audited) or receipt["sessions"] != len(self.items):
            raise ValueError("Data audit omitted/added sessions")
        parents = {}
        for name, pin in PARENTS.items():
            p = artifact(
                index["parents"][name]["path"]
                if role == "development"
                else index[f"v7_{name}_index_path"]
            )
            verify(p, pin)
            parent = read(p)
            if parent["role"] != name:
                raise ValueError("Wrong exposed ancestor role")
            for row in parent["sessions"]:
                if row["session_id"] in parents:
                    raise ValueError("Exposed ancestors overlap")
                parents[row["session_id"]] = (row, p, name, pin)
        self.parent_items = {s: r[0] for s, r in parents.items()}
        if len(parents) != 176:
            raise ValueError("Wrong exposed ancestry inventory")
        if role == "development":
            if set(self.items) != set(parents) or index["participants"] != PEOPLE:
                raise ValueError("Wrong development participant/session inventory")
            for sid, item in self.items.items():
                prior, path, name, pin = parents[sid]
                for key in prior.keys() - {
                    "role",
                    "feature_path",
                    "target_path",
                    "reused_parent_artifacts",
                }:
                    compare_structure(item[key], prior[key], "exposed ancestor " + key)
                compare_structure(
                    item["v8_parent"],
                    {"index_path": str(path), "index_sha256": pin, "role": name, "session_id": sid},
                    "parent binding",
                )
                for kind in ("feature", "target"):
                    compare_structure(
                        item["reused_parent_artifacts"][kind],
                        {
                            "path": str(contained(path.parent, prior[kind + "_path"])),
                            "sha256": prior[kind + "_sha256"],
                            "mode": "immutable_hard_link",
                        },
                        "hardlink identity",
                    )
        else:
            self._fresh(parents)
        for sid, item in self.items.items():
            record = audited[sid]
            if (
                item["role"] != role
                or item["bin_seconds"] != 0.02
                or type(item["n_bins"]) is not int
                or not 0 <= item["n_bins"] <= 30000
                or item["n_bins"] != min(30000, item["full_prepared_n_bins"])
                or item["far_distance"] != FAR_SPECIAL.get(sid, FAR[item["participant_id"]])
            ):
                raise ValueError("Prepared geometry/far threshold changed")
            segment_ranges(item["segments"], item["n_bins"])
            for key in (
                "participant_id",
                "session_id",
                "post_implant_day",
                "n_bins",
                "n_channels",
                "feature_sha256",
                "target_sha256",
                "context_sha256",
            ):
                if record[key] != item[key]:
                    raise ValueError("Data-audit record binding changed")
            if record["member_sha256"] != item["source"]["member_sha256"] or not all(
                record[k] is True
                for k in (
                    "context_matches_independent_rebuild",
                    "unchanged_target_and_mask_match",
                    "bounded_to_authenticated_parent_segments",
                )
            ):
                raise ValueError("Data reconstruction not verified")
            for kind in ("feature", "target", "context"):
                verify(contained(self.path.parent, item[kind + "_path"]), item[kind + "_sha256"])
        self.data_audit = receipt

    def _fresh(self, parents):
        index = self.index
        people = [p for p in PEOPLE if p != "T3"]
        if (
            index["participants"] != people
            or index["acquisition_plan_sha256"] != ACQUISITION_SHA
            or index["study_lock"]["sha256"] != PLAN_SHA
            or index["acquisition_lock"]["sha256"] != ACQUISITION_SHA
        ):
            raise ValueError("Fresh acquisition/study plans changed")
        verify(artifact(index["study_lock"]["path"]), PLAN_SHA)
        verify(artifact(index["acquisition_lock"]["path"]), ACQUISITION_SHA)
        refs = unique(index["acquisition_sources"], lambda r: r["path"])
        if set(refs) != {f"acquisition/{p}.json" for p in people}:
            raise ValueError("Missing requested participant acquisition")
        expected = {}
        dispositions = []
        for person in people:
            ref = refs[f"acquisition/{person}.json"]
            path = contained(self.path.parent, ref["path"])
            verify(path, ref["sha256"])
            r = read(path)
            prior = sorted(
                [v[0]["source"] for v in parents.values() if v[0]["participant_id"] == person],
                key=lambda s: s["archive_ordinal"],
            )
            if (
                len(prior) != 21
                or r["status"] != "complete"
                or r["participant_id"] != person
                or r["plan_sha256"] != ACQUISITION_SHA
                or r["selection"]["regular_session_ranks_one_based"] != [22, 23, 24]
                or r["selection"]["scores_or_duration_used_for_selection"] is not False
                or r["full_archive_sha256_verified"] is not False
                or r["archive"]["file_id"] != prior[0]["archive_file_id"]
                or r["archive"]["provider_sha256"] != prior[0]["archive_sha256_published"]
            ):
                raise ValueError("Wrong pinned archive/selection")
            for name, pin in PARENTS.items():
                if r[f"v7_{name}_index_sha256"] != pin:
                    raise ValueError("Wrong archive ancestry")
            pattern = rf"(?:\./)?decoding/{person}/{person}_day_(\d+)_decoding\.mat"
            seen = [
                s
                for s in r["seen_members"]
                if s["regular_file"] and re.fullmatch(pattern, s["name"])
            ]
            old, new = r["verified_prior_members"], r["selected"]
            if (
                len(old) != 21
                or len(new) > 3
                or len(seen) != 21 + len(new)
                or len(new) < 3
                and not r["tar_end_observed"]
                or r["missing_requested_ranks"] != list(range(22 + len(new), 25))
            ):
                raise ValueError("Wrong complete member/shortage inventory")
            for rank, (m, s) in enumerate(zip(old + new, seen, strict=True), 1):
                match = re.fullmatch(pattern, m["member_name"])
                if (
                    match is None
                    or m["regular_session_rank"] != rank
                    or m["participant_id"] != person
                    or m["post_implant_day"] != int(match.group(1))
                    or m["session_id"] != Path(m["member_name"]).stem
                    or m["member_name"] != s["name"]
                    or m["member_size"] != s["size"]
                    or m["archive_ordinal"] != s["ordinal"]
                ):
                    raise ValueError("Archive-order identity changed")
                if rank <= 21:
                    for key in (
                        "member_name",
                        "member_size",
                        "member_sha256",
                        "archive_ordinal",
                        "session_id",
                    ):
                        if m[key] != prior[rank - 1][key]:
                            raise ValueError("Prior twenty-one member authentication changed")
                else:
                    if m["session_id"] in expected:
                        raise ValueError("Duplicate new session")
                    expected[m["session_id"]] = {
                        **m,
                        "archive_file_id": r["archive"]["file_id"],
                        "archive_sha256_published": r["archive"]["provider_sha256"],
                        "full_archive_sha256_verified": False,
                    }
            dispositions.append(
                {
                    "participant_id": person,
                    "selected_sessions": len(new),
                    "missing_requested_ranks": r["missing_requested_ranks"],
                }
            )
        compare_structure(index["participant_dispositions"], dispositions, "fresh shortages")
        if set(self.items) != set(expected):
            raise ValueError("Selected member missing/replaced")
        for sid, item in self.items.items():
            compare_structure(item["source"], expected[sid], "selected raw member")
            if (
                item["participant_id"] != expected[sid]["participant_id"]
                or item["post_implant_day"] != expected[sid]["post_implant_day"]
            ):
                raise ValueError("Wrong selected session identity")

    def case(self, sid):
        item = self.items[sid]
        bundles = {
            k: numeric(contained(self.path.parent, item[k + "_path"]))
            for k in ("feature", "target", "context")
        }
        features, targets, ctx = bundles["feature"], bundles["target"], bundles["context"]
        n, c = item["n_bins"], item["n_channels"]
        if set(features) != {
            "x_tx3",
            "x_tx45",
            "x_sbp",
            "timestamps",
            "sequence_id",
            "channel_ids",
        } or set(targets) != {"y", "eval_mask"}:
            raise ValueError("Unknown feature/target fields")
        for key in ("x_tx3", "x_tx45", "x_sbp"):
            a = features[key]
            if (
                a.shape != (n, c)
                or a.dtype != np.float32
                or not np.isfinite(a).all()
                or np.any(a < 0)
            ):
                raise ValueError("Invalid nonnegative neural arrays")
        if (
            features["channel_ids"].shape != (c,)
            or features["channel_ids"].dtype != np.int64
            or len(set(features["channel_ids"].tolist())) != c
        ):
            raise ValueError("Channel identities changed")
        if (
            targets["y"].shape != (n, 2)
            or targets["y"].dtype != np.float32
            or targets["eval_mask"].shape != (n,)
            or targets["eval_mask"].dtype != bool
        ):
            raise ValueError("Invalid target arrays")
        for s in item["segments"]:
            a, b = s["start"], s["end"]
            if not np.all(features["sequence_id"][a:b] == s["sequence_id"]) or not np.allclose(
                np.diff(features["timestamps"][a:b]), 0.02, rtol=0, atol=1e-9
            ):
                raise ValueError("Feature reset/timestamp geometry mismatch")
        delta = ctx["target_position"] - ctx["cursor_position"]
        distance = np.linalg.norm(delta, axis=1)
        good = np.isfinite(delta).all(1) & np.isfinite(distance) & (distance > 1e-12)
        y = np.zeros((n, 2), np.float32)
        y[good] = (delta[good] / distance[good, None]).astype(np.float32)
        if not np.array_equal(y, targets["y"]) or not np.array_equal(
            good & (ctx["trial_id"] >= 0), targets["eval_mask"]
        ):
            raise ValueError("Context changed original target/mask")
        return {
            "item": item,
            "features": features,
            **targets,
            "context": ctx,
            "segments": [(s["start"], s["end"]) for s in item["segments"] if s["end"] > s["start"]],
        }


def representations_reference(plan):
    return {
        f"representation_{i:02d}": {
            **BASELINE,
            **{k: r[k] for k in ("family", "taus_seconds", "ridge")},
        }
        for i, r in enumerate(plan["representation_grid"])
    }


def factorial_reference(representation):
    out = {}
    for name, rep in (("baseline", BASELINE), ("selected", representation)):
        for phase, weighting in itertools.product(("all", "clean"), ("uniform", "equal_trial")):
            method = (
                "baseline"
                if (name, phase, weighting) == ("baseline", "all", "uniform")
                else f"{name}_{phase}_{weighting}"
            )
            out[method] = {
                **BASELINE,
                **{k: rep[k] for k in ("family", "taus_seconds", "ridge")},
                "phase": phase,
                "weighting": weighting,
            }
    return out


def configurations_reference(factorials, primary):
    return {
        **factorials,
        "zero": {**BASELINE, "control": "zero"},
        "constant": {**BASELINE, "control": "intercept"},
        "trialshuffle": {**factorials[primary], "control": "trial_permutation"},
        "primary_onset": {**factorials[primary], "phase": "clean", "target": "onset"},
    }


def choose_reference(rows, configs):
    ids = sorted({r["session_id"] for r in rows})
    table = unique(rows, lambda r: (r["session_id"], r["method"]))
    if not ids or set(table) != {(s, m) for s in ids for m in configs}:
        raise ValueError("Incomplete selection inventory")
    scorable = []
    for sid in ids:
        values = [table[sid, m] for m in configs]
        if (
            len({v["score_mask_sha256"] for v in values}) != 1
            or len({v["metrics"]["r2"] is None for v in values}) != 1
        ):
            raise ValueError("Unpaired candidate comparison")
        if values[0]["metrics"]["r2"] is not None:
            scorable.append(sid)
    ranking = [
        {
            "method": m,
            "mean_r2": average([table[s, m]["metrics"]["r2"] for s in scorable]),
            "scored_sessions": len(scorable),
            "selected_sessions": len(ids),
            "session_r2": [{"session_id": s, "r2": table[s, m]["metrics"]["r2"]} for s in ids],
        }
        for m in configs
    ]
    best = max(range(len(ranking)), key=lambda i: ranking[i]["mean_r2"]) if scorable else 0
    return {
        "selected_method": ranking[best]["method"],
        "ranking": ranking,
        "no_scorable_sessions_fallback": not scorable,
        "tie_break": "declared_order",
    }


def audit_run(run, output, *, data_audit, source_run=None, source_audit=None):
    run = Path(run).resolve()
    mpath = run / "manifest.json"
    m = read(mpath)
    auditor_sha = sha256(__file__)
    if m["status"] != "complete" or m["study"] != "V8" or m["schema_version"] != 1:
        raise ValueError("Completed V8 manifest required")
    if m["plan"]["sha256"] != PLAN_SHA:
        raise ValueError("Wrong prespecified plan")
    plan_path = artifact(m["plan"]["path"])
    verify(plan_path, PLAN_SHA)
    plan = read(plan_path)
    expected_sources = {
        f"neural_experiments/{version}/{name}.py"
        for version, names in {
            "v2": ("data",),
            "v3": ("data", "model", "study"),
            "v4": ("data", "model"),
            "v5": ("data",),
            "v6": ("data",),
            "v7": ("data", "fresh_data"),
            "v8": (
                "data",
                "fresh_data",
                "model",
                "development",
                "assessment",
                "metrics",
                "diagnostics",
                "errors",
            ),
        }.items()
        for name in names
    }
    if set(m["code_sha256"]) != expected_sources:
        raise ValueError("Incomplete frozen scientific source inventory")
    for path, pin in m["code_sha256"].items():
        verify(artifact(path), pin)
        verify(contained(run / "source_snapshot", path), pin)
    index_path = artifact(m["index"]["path"])
    verify(index_path, m["index"]["sha256"])
    inputs = Inputs(index_path, data_audit)
    if inputs.index["role"] != ("development" if m["phase"] == "development" else "fresh"):
        raise ValueError("Role boundary changed")
    checked_rows = []
    errors = []
    residuals = []
    models = 0
    traces = 0
    session_refs = unique(m["sessions"], lambda r: (r.get("stage", "assessment"), r["session_id"]))
    visited = set()

    def check_group(stage, items, configs, rows):
        nonlocal models, traces
        if {(r["session_id"], r["method"]) for r in rows} != {
            (i["session_id"], c) for i in items for c in configs
        } or len(rows) != len(items) * len(configs):
            raise ValueError("Incomplete stage cases")
        rebuilt = []
        for item in items:
            sid = item["session_id"]
            key = (stage, sid)
            if key not in session_refs:
                raise ValueError("Missing session export receipt")
            visited.add(key)
            case = inputs.case(sid)
            rr, ee, ss = check_session(
                case, configs, [r for r in rows if r["session_id"] == sid], session_refs[key]
            )
            rebuilt.extend(rr)
            errors.extend(ee)
            residuals.extend(ss)
            models += len(rr)
            traces += 1
            print(json.dumps({"audit_stage": stage, "session": sid, "models": len(rr)}), flush=True)
        checked_rows.extend(rebuilt)
        return rebuilt

    source_binding = {}
    if m["phase"] == "development":
        items = sorted(
            [r for r in inputs.items.values() if r["participant_id"] == "T2"],
            key=lambda r: r["source"]["archive_ordinal"],
        )
        if len(items) != 21:
            raise ValueError("Wrong T2 development inventory")
        configs = representations_reference(plan)
        compare_structure(m["representation_configurations"], configs, "representation grid")
        candidate_ids = {
            f"representation_{i:02d}": r["candidate_id"]
            for i, r in enumerate(plan["representation_grid"])
        }
        compare_structure(m["representation_candidate_ids"], candidate_ids, "candidate identifiers")
        rows = check_group("representation", items[:15], configs, m["representation_rows"])
        rep_choice = choose_reference(rows, configs)
        compare_structure(m["representation_selection"], rep_choice, "representation ranking")
        chosen = rep_choice["selected_method"]
        factorials = factorial_reference(configs[chosen])
        compare_structure(m["factorial_configurations"], factorials, "fixed factorial grid")
        rows = check_group("factorial", items[15:], factorials, m["factorial_rows"])
        choice = choose_reference(rows, factorials)
        compare_structure(m["factorial_selection"], choice, "factorial ranking")
        primary = choice["selected_method"]
        allconfigs = configurations_reference(factorials, primary)
        diagnostics = {k: v for k, v in allconfigs.items() if k not in factorials}
        check_group("diagnostic", items[15:], diagnostics, m["diagnostic_rows"])
        selection = {
            "schema_version": 1,
            "study": "V8",
            "source_audit_status": "PENDING",
            "plan_sha256": PLAN_SHA,
            "index_sha256": sha256(index_path),
            "scientific_code_sha256": m["code_sha256"],
            "code_sha256": m["code_sha256"],
            "configurations": allconfigs,
            "primary_method": primary,
            "representation_choice": {
                "method": chosen,
                "candidate_id": candidate_ids[chosen],
                "config": configs[chosen],
            },
            "representation_selection": rep_choice,
            "factorial_selection": choice,
            "onset_matched_instantaneous_method": next(
                k
                for k, v in factorials.items()
                if v == {**allconfigs["primary_onset"], "target": "instantaneous"}
            ),
        }
        compare_structure(m["selection"], selection, "complete selected configuration")
        summary = {
            "representation_selection": rep_choice,
            "factorial_selection": choice,
            "primary_method": primary,
        }
    elif m["phase"] == "assessment":
        if source_run is None or source_audit is None:
            raise ValueError("Fresh audit needs explicit retained source and passed audit")
        source_path = Path(source_run) / "manifest.json"
        original = read(source_path)
        passed = read(source_audit)
        if (
            passed["status"] != "PASS"
            or passed["phase"] != "development"
            or passed["manifest_sha256"] != sha256(source_path)
        ):
            raise ValueError("Source audit not bound to retained source")
        selection_path = artifact(m["source_selection"]["path"])
        verify(selection_path, m["source_selection"]["sha256"])
        published = read(selection_path)
        expected = {
            **original["selection"],
            "source_manifest_sha256": sha256(source_path),
            "source_audit_status": "PASS",
            "source_audit_sha256": sha256(source_audit),
        }
        compare_structure(published, expected, "audited published selection")
        compare_structure(m["code_sha256"], original["code_sha256"], "frozen scientific source")
        compare_structure(
            m["configurations"], published["configurations"], "fresh fixed configurations"
        )
        if m["primary_method"] != published["primary_method"]:
            raise ValueError("Fresh primary substitution")
        rows = check_group(
            "assessment", list(inputs.items.values()), m["configurations"], m["rows"]
        )
        summary = summary_reference(rows, m["primary_method"])
        compare_structure(m["summary"], summary, "fresh aggregate and gate")
        source_binding = {
            "source_manifest_sha256": sha256(source_path),
            "source_audit_sha256": sha256(source_audit),
            "selection_sha256": sha256(selection_path),
        }
    else:
        raise ValueError("Unknown run phase")
    if set(session_refs) != visited:
        raise ValueError("Extra unassessed session artifacts")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "phase": m["phase"],
        "auditor_sha256": auditor_sha,
        "numeric_reference_sha256": {
            f"neural_experiments/{version}/audit.py": sha256(
                ROOT / f"neural_experiments/{version}/audit.py"
            )
            for version in ("v2", "v3", "v4", "v7")
        },
        "manifest_sha256": sha256(mpath),
        "plan_sha256": PLAN_SHA,
        "index_sha256": sha256(index_path),
        "data_audit_sha256": sha256(inputs.data_audit_path),
        "data_audit_sessions": inputs.data_audit["sessions"],
        "models_checked": models,
        "traces_checked": traces,
        "rows_checked": len(checked_rows),
        "max_prediction_error": max(errors, default=0.0),
        "max_normal_equation_residual": max(residuals, default=0.0),
        "summary": summary,
        **source_binding,
        "scope": (
            "Separate numeric reference verifies every fit and full trace, selection and scores. "
            "Separate bound data audit independently rebuilds raw behavior context; this audit "
            "cross-checks target parity, geometry and ancestry. No biological or closed-loop "
            "efficacy proof."
        ),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve existing audit receipt; select a fresh output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def interval_reference(values):
    if not len(values):
        return [None, None]
    a = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(20261023).integers(0, len(a), size=(10000, len(a)))
    return np.quantile(np.mean(a[draws], axis=1), [0.025, 0.975]).tolist()


def effects_reference(rows):
    configurations = factorial_reference(BASELINE)
    order = list(configurations)
    table = unique(rows, lambda r: (r["session_id"], r["method"]))
    if not set(order) <= {r["method"] for r in rows}:
        return {}
    coordinates = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    effects = {}
    for name, axes in (
        ("representation", (0,)),
        ("clean_phase", (1,)),
        ("equal_trial", (2,)),
        ("representation_x_phase", (0, 1)),
        ("representation_x_weighting", (0, 2)),
        ("phase_x_weighting", (1, 2)),
        ("three_way", (0, 1, 2)),
    ):
        signed = np.prod(coordinates[:, axes], axis=1) / (2 ** (3 - len(axes)))
        sessions = []
        for sid in sorted({r["session_id"] for r in rows}):
            case = [table[sid, m] for m in order]
            if any(r["metrics"]["r2"] is None for r in case):
                continue
            sessions.append(
                {
                    "session_id": sid,
                    "participant": case[0]["participant"],
                    "effect": float(signed @ np.asarray([r["metrics"]["r2"] for r in case])),
                }
            )
        people = sorted({s["participant"] for s in sessions}, key=lambda p: int(p[1:]))
        units = [
            {
                "participant": p,
                "mean_effect": average([s["effect"] for s in sessions if s["participant"] == p]),
            }
            for p in people
        ]
        values = [u["mean_effect"] for u in units]
        t2 = [s["effect"] for s in sessions if s["participant"] == "T2"]
        effects[name] = {
            "participants": units,
            "sessions": sessions,
            "mean_effect": average(values),
            "paired_participant_95": interval_reference(values),
            "T2_mean_effect": average(t2),
            "T2_session_95": interval_reference(t2),
            "role": "Prespecified descriptive attribution; no multiplicity-adjusted claim",
        }
    return effects


def summary_reference(rows, primary):
    table = unique(rows, lambda r: (r["session_id"], r["method"]))
    methods = sorted({r["method"] for r in rows})
    ids = sorted({r["session_id"] for r in rows})
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    if not {primary, "baseline", "zero"} <= set(methods) or set(table) != {
        (s, m) for s in ids for m in methods
    }:
        raise ValueError("Incomplete aggregate inventory")
    for sid in ids:
        group = [table[sid, m] for m in methods]
        if (
            len({r["score_mask_sha256"] for r in group}) != 1
            or len({r["metrics"]["r2"] is None for r in group}) != 1
        ):
            raise ValueError("Aggregate methods are not paired")
    summaries = {}
    for method in methods:
        units = []
        for person in people:
            selected = [r for r in rows if r["method"] == method and r["participant"] == person]
            scored = [r for r in selected if r["metrics"]["r2"] is not None]
            values = [r["metrics"]["r2"] for r in scored]
            units.append(
                {
                    "participant": person,
                    "selected_sessions": len(selected),
                    "scored_sessions": len(scored),
                    "mean_r2": average(values),
                    "positive_sessions": sum(v > 0 for v in values),
                    "session_bootstrap_95": interval_reference(values),
                    "mean_summed_mse": average([r["metrics"]["summed_mse"] for r in scored]),
                    "sessions": [
                        {k: r[k] for k in ("session_id", "post_implant_day", "metrics")}
                        for r in selected
                    ],
                }
            )
        usable = [u for u in units if u["mean_r2"] is not None]
        values = [u["mean_r2"] for u in usable]
        other = [u["mean_r2"] for u in usable if u["participant"] != "T2"]
        summaries[method] = {
            "participants": units,
            "mean_r2": average(values),
            "participant_bootstrap_95": interval_reference(values),
            "scored_sessions": sum(u["scored_sessions"] for u in units),
            "n_participants": len(usable),
            "non_T2_mean_r2": average(other),
            "non_T2_participant_bootstrap_95": interval_reference(other),
        }
    contrasts = {}
    for comparator in ("baseline", "zero"):
        units = []
        for person in people:
            delta = [
                table[s, primary]["metrics"]["r2"] - table[s, comparator]["metrics"]["r2"]
                for s in ids
                if table[s, primary]["participant"] == person
                and table[s, primary]["metrics"]["r2"] is not None
            ]
            units.append(
                {
                    "participant": person,
                    "mean_gain": average(delta),
                    "session_gains": delta,
                    "session_bootstrap_95": interval_reference(delta),
                }
            )
        gains = [u["mean_gain"] for u in units if u["mean_gain"] is not None]
        contrasts[f"primary_minus_{comparator}"] = {
            "participants": units,
            "mean_gain": average(gains),
            "paired_participant_95": interval_reference(gains),
        }
    t2 = next((u for u in summaries[primary]["participants"] if u["participant"] == "T2"), None)
    gain2 = [
        next((u for u in c["participants"] if u["participant"] == "T2"), None)
        for c in contrasts.values()
    ]
    gate = bool(
        t2
        and t2["selected_sessions"] == t2["scored_sessions"] == 3
        and t2["mean_r2"] >= 0.05
        and t2["positive_sessions"] >= 2
        and all(
            u
            and u["mean_gain"] is not None
            and u["mean_gain"] > 0
            and u["session_bootstrap_95"][0] > 0
            for u in gain2
        )
    )
    return {
        "selected_sessions": len(ids),
        "primary_method": primary,
        "methods": summaries,
        "contrasts": contrasts,
        "factorial_effects": effects_reference(rows),
        "gates": {"useful_T2_primary_recovery": gate},
        "interval_limit": (
            "Three T2 sessions give descriptive session uncertainty; cohort intervals "
            "resample people. Neither establishes untouched-participant transfer."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-audit", type=Path)
    args = parser.parse_args()
    result = audit_run(
        args.run,
        args.output,
        data_audit=args.data_audit,
        source_run=args.source_run,
        source_audit=args.source_audit,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "phase", "models_checked", "max_prediction_error")}
        )
    )


if __name__ == "__main__":
    main()
