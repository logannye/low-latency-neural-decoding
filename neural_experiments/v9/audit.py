"""Independent V9 trial quota, causal numerical and artifact verification.

No V9 producer, model, importer, selector or scorer is imported. Earlier
independent recurrence/sklearn references are reused. This verifies numerical
implementation and provenance, not biological intention or clinical utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit import compare_structure
from neural_experiments.v3.audit import contained, numeric, read, segment_ranges, sha256, verify
from neural_experiments.v4.audit import reference_representation
from neural_experiments.v7.audit import average, burned_mask, unique
from neural_experiments.v8.audit import FAR, FAR_SPECIAL, score_reference
from neural_experiments.v8.audit import Inputs as PreviousInputs

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = "aea60063bf6aa027ea54ce05dfb52e83ff00c61f8728f3a8f59f4f5394431dbf"
ACQUISITION_SHA = "56e697e30821105ef5689a29b92e63c4f214f113583028798f2d366501ec9271"
PARENTS = {
    "development": "5c4b5ab0b0fa0ec28c2c5a1b746ca8e8ceee35b4797e2f7e2f66cc9d9dfdbf84",
    "fresh": "0ea2078c184e58014c9fc68d5fd5efa3855e93a12a82db6f2627174ae894c4b8",
}
PEOPLE = ["T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11"]
FRESH_PEOPLE = ["T2", "T7", "T9", "T10"]


def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def artifact(path):
    value = Path(path)
    value = value.resolve() if value.is_absolute() else contained(ROOT, value)
    if not value.is_relative_to(ROOT):
        raise ValueError("Artifact outside repository")
    return value


class Inputs(PreviousInputs):
    """Independent parent/receipt walk; reuse audited same-schema array reader."""

    def __init__(self, path, data_audit):
        self.path = Path(path).resolve()
        self.index = read(self.path)
        self.checked = set()
        self.data_audit_path = Path(data_audit).resolve()
        receipt = read(self.data_audit_path)
        index, role = self.index, self.index["role"]
        if (
            role not in ("development", "fresh")
            or index["schema_version"] != 1
            or index["provider"] != {"doi": "10.5061/dryad.x0k6djj1h", "version_id": 462702}
            or receipt["status"] != "PASS"
            or receipt["role"] != role
            or receipt["index_sha256"] != sha256(self.path)
            or receipt["auditor_sha256"] != sha256(ROOT / "neural_experiments/v9/audit_data.py")
        ):
            raise ValueError("Data audit must bind exact role/index/reference")
        self.items = unique(index["sessions"], lambda r: r["session_id"])
        audited = unique(receipt["records"], lambda r: r["session_id"])
        if set(self.items) != set(audited) or receipt["sessions"] != len(self.items):
            raise ValueError("Data audit omitted/added sessions")
        parents = {}
        for name, pin in PARENTS.items():
            p = artifact(
                index["parents"][name]["path"]
                if role == "development"
                else index[f"v8_{name}_index_path"]
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
        if len(parents) != 200:
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
                    "context_path",
                    "reused_parent_artifacts",
                }:
                    compare_structure(item[key], prior[key], "exposed ancestor " + key)
                compare_structure(
                    item["v9_parent"],
                    {"index_path": str(path), "index_sha256": pin, "role": name, "session_id": sid},
                    "parent binding",
                )
                for kind in ("feature", "target", "context"):
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
        people = FRESH_PEOPLE
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
                len(prior) != 24
                or r["status"] != "complete"
                or r["participant_id"] != person
                or r["plan_sha256"] != ACQUISITION_SHA
                or r["selection"]["regular_session_ranks_one_based"] != [25, 26, 27, 28, 29, 30]
                or r["selection"]["scores_or_duration_used_for_selection"] is not False
                or r["full_archive_sha256_verified"] is not False
                or r["archive"]["file_id"] != prior[0]["archive_file_id"]
                or r["archive"]["provider_sha256"] != prior[0]["archive_sha256_published"]
            ):
                raise ValueError("Wrong pinned archive/selection")
            for name, pin in PARENTS.items():
                if r[f"v8_{name}_index_sha256"] != pin:
                    raise ValueError("Wrong archive ancestry")
            pattern = rf"(?:\./)?decoding/{person}/{person}_day_(\d+)_decoding\.mat"
            seen = [
                s
                for s in r["seen_members"]
                if s["regular_file"] and re.fullmatch(pattern, s["name"])
            ]
            old, new = r["verified_prior_members"], r["selected"]
            if (
                len(old) != 24
                or len(new) > 6
                or len(seen) != 24 + len(new)
                or len(new) < 6
                and not r["tar_end_observed"]
                or r["missing_requested_ranks"] != list(range(25 + len(new), 31))
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
                if rank <= 24:
                    for key in (
                        "member_name",
                        "member_size",
                        "member_sha256",
                        "archive_ordinal",
                        "session_id",
                    ):
                        if m[key] != prior[rank - 1][key]:
                            raise ValueError("Prior twenty-four member authentication changed")
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


def masks_reference(case):
    """Fixed common suffix, including every trial touched by the maximum purchase."""
    ids = np.asarray(case["context"]["trial_id"])
    valid = burned_mask(case["eval_mask"], case["segments"]) & (ids >= 0)
    c = case["context"]
    overlap = (
        (ids >= 0)
        & (np.asarray(c["trial_onset_seconds"]) < 390)
        & (np.asarray(c["trial_stop_bin"]) + 1 > 1500)
    )
    touched = sorted(set(int(v) for v in ids[overlap]))
    primary = valid & ~np.isin(ids, touched)
    primary[:19600] = False
    return primary, touched


def sector_reference(vector):
    """Four 90-degree sectors centered on cardinal directions."""
    v = np.asarray(vector, dtype=np.float64)
    return int(np.floor((np.arctan2(v[1], v[0]) + np.pi / 4) / (np.pi / 2))) % 4


def boxcar_reference(raw, segments, width=15):
    """Independent causal finite-window mean with per-block startup denominator."""
    raw = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(raw)
    for begin, end in segments:
        for t in range(begin, end):
            out[t] = np.mean(raw[max(begin, t - width + 1) : t + 1], axis=0)
    return out


def ridge_reference(design, labels, weights, penalty):
    """Weighted mean loss; free intercept; return optimum equations, not producer fit."""
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (design, labels, weights))
    if x.ndim != 2 or y.shape != (len(x), 2) or w.shape != (len(x),):
        raise ValueError("Invalid weighted fit shapes")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(w).all()
        or np.any(w < 0)
    ):
        raise ValueError("Nonfinite/negative fit arrays")
    total = float(np.sum(w))
    gram, rhs = np.zeros((x.shape[1], x.shape[1])), np.zeros((x.shape[1], 2))
    if total:
        gram = (x.T * w) @ x / total
        rhs = (x.T * w) @ y / total
    gram += np.diag(np.r_[np.full(x.shape[1] - 1, penalty), 0.0])
    return gram, rhs


def choose_reference(rows, configurations, session_ids):
    """Nine declared late exposed T2 sessions; no fresh or alternative endpoint."""
    ids = sorted(session_ids)
    table = unique(rows, lambda r: (r["session_id"], r["method"]))
    if (
        not ids
        or len(set(ids)) != len(ids)
        or set(table) != {(s, c) for s in ids for c in configurations}
    ):
        raise ValueError("Incomplete source selection inventory")
    available = []
    for sid in ids:
        values = [table[sid, c] for c in configurations]
        if (
            len({v["score_mask_sha256"] for v in values}) != 1
            or len({v["metrics"]["r2"] is None for v in values}) != 1
        ):
            raise ValueError("Unpaired source candidate comparison")
        if values[0]["metrics"]["r2"] is not None:
            available.append(sid)
    ranking = [
        {
            "method": c,
            "mean_r2": average([table[s, c]["metrics"]["r2"] for s in available]),
            "selected_sessions": len(ids),
            "scored_sessions": len(available),
            "session_r2": [{"session_id": s, "r2": table[s, c]["metrics"]["r2"]} for s in ids],
        }
        for c in configurations
    ]
    best = max(range(len(ranking)), key=lambda i: ranking[i]["mean_r2"]) if available else 0
    return {
        "selected_method": ranking[best]["method"],
        "ranking": ranking,
        "no_scorable_sessions_fallback": not available,
        "tie_break": "declared_order",
    }


def trial_records_reference(case):
    """Reconstruct common RT eligibility independently from paired context.

    The common45-bin span holds trial membership fixed across both RT choices.
    Trial completion is separate: paired stop+one bin is the conservative
    observable completion time, including an otherwise hidden trailing10ms.
    """
    c = case["context"]
    ids = np.asarray(c["trial_id"])
    n = len(ids)
    valid = burned_mask(case["eval_mask"], case["segments"])
    segment_id = np.full(n, -1, dtype=np.int64)
    for j, (a, b) in enumerate(case["segments"]):
        segment_id[a:b] = j
    records = []
    for trial in sorted(
        set(int(v) for v in ids if v >= 0), key=lambda v: int(np.flatnonzero(ids == v)[0])
    ):
        ix = np.flatnonzero(ids == trial)
        first = int(ix[0])
        record = {
            "trial_id": trial,
            "start_bin": int(c["trial_start_bin"][first]),
            "stop_bin": int(c["trial_stop_bin"][first]),
            "completion_bin": int(c["trial_stop_bin"][first]) + 1,
            "onset_seconds": float(c["trial_onset_seconds"][first]),
            "right_censored": bool(c["trial_right_censored"][first]),
            "eligible": False,
        }
        records.append(record)
        if not np.array_equal(ix, np.arange(record["start_bin"], record["stop_bin"])):
            continue
        windows = {}
        for rt in (0.2, 0.8):
            age_cs = np.rint(np.asarray(c["trial_age_seconds"])[ix] * 100)
            if not np.allclose(
                age_cs, np.asarray(c["trial_age_seconds"])[ix] * 100, rtol=0, atol=1e-8
            ):
                raise ValueError("Trial ages must bind exact raw10ms increments")
            possible = ix[age_cs - 1 >= round(rt * 100)]
            if not len(possible):
                break
            a = int(possible[0])
            b = a + 15
            if b > record["stop_bin"] or b > n or not np.all(ids[a:b] == trial):
                break
            windows[rt] = (a, b)
        if len(windows) != 2:
            continue
        a, b = windows[0.2][0], windows[0.8][1]
        if (
            b - a != 45
            or not np.all(ids[a:b] == trial)
            or not np.all(valid[a:b])
            or len(set(segment_id[a:b].tolist())) != 1
            or not bool(c["reference40_available"][first])
        ):
            continue
        ref_distance = float(c["reference40_distance"][first])
        ref = np.asarray(c["reference40_direction"][first], dtype=np.float64)
        if (
            not np.isfinite(ref_distance)
            or ref_distance <= case["item"]["far_distance"]
            or not np.isfinite(ref).all()
            or abs(float(np.linalg.norm(ref)) - 1.0) > 1e-6
        ):
            continue
        target = np.asarray(c["target_position"][a:b], dtype=np.float64)
        delta = target - np.asarray(c["cursor_position"][a:b], dtype=np.float64)
        length = np.linalg.norm(delta, axis=1)
        if not np.isfinite(target).all() or not np.isfinite(delta).all() or np.any(length <= 1e-12):
            continue
        stationarity = float(np.linalg.norm(target - target[0], axis=1).max() / ref_distance)
        coherence = float(np.linalg.norm((delta / length[:, None]).mean(0)))
        if (
            float(np.linalg.norm(target - target[0], axis=1).max()) > 0.01 * ref_distance
            or coherence < 0.95
        ):
            continue
        targets = {}
        for rt, (left, right) in windows.items():
            displacement = np.asarray(
                c["target_position"][left:right], dtype=np.float64
            ) - np.asarray(c["cursor_position"][left:right], dtype=np.float64)
            mean = displacement.mean(0)
            norm = float(np.linalg.norm(mean))
            if not np.isfinite(mean).all() or norm <= 1e-12:
                break
            targets[rt] = mean / norm
        if len(targets) != 2:
            continue
        record.update(
            {
                "eligible": True,
                "windows": windows,
                "window_targets": targets,
                "sector": sector_reference(c["reference40_direction"][first]),
                "coherence": coherence,
                "target_excursion_ratio": stationarity,
                "target_excursion": float(np.linalg.norm(target - target[0], axis=1).max()),
                "reference40_distance": ref_distance,
                "common_start_bin": a,
                "common_stop_bin": b,
            }
        )
    return records


def quota_reference(records, n):
    """Prospective complete-trial quota, with every timeout retained."""
    deadline = min(19500, n)
    counts = np.zeros(4, dtype=np.int64)
    selected = []
    attained = False
    stop = 19500
    for record in sorted(records, key=lambda r: (r["completion_bin"], r["trial_id"])):
        if (
            not record["eligible"]
            or record["right_censored"]
            or record["onset_seconds"] < 30
            or record["completion_bin"] > deadline
        ):
            continue
        selected.append(record)
        counts[record["sector"]] += 1
        if len(selected) >= 24 and np.all(counts >= 4):
            attained = True
            stop = record["completion_bin"]
            break
    return {
        "stop_bin": stop,
        "attained": attained,
        "trial_ids": [r["trial_id"] for r in selected],
        "sector_counts": counts.tolist(),
        "observed_label_seconds": max(0, min(n, stop) - 1500) * 0.02,
    }


def representations_reference(case):
    """Frozen continuous control and separate trial-window representation."""
    features = case["features"]
    transformed = np.column_stack(
        (
            np.sqrt(features["x_tx45"].astype(np.float64)),
            np.log1p(features["x_sbp"].astype(np.float64)),
        )
    )
    return {
        "ewma200": reference_representation(
            features,
            case["segments"],
            {
                "family": "tx45_sbp",
                "tau_seconds": 0.2,
                "baseline_tau_seconds": None,
                "adapt_gamma": 10.0,
            },
        ),
        "boxcar300": boxcar_reference(transformed, case["segments"]),
    }


def normalization_reference(raw, channels):
    floors = np.r_[np.full(channels, 0.1), np.full(channels, 0.01)]
    if len(raw) < 1500:
        return np.zeros(2 * channels), floors
    prefix = np.asarray(raw[:1500], dtype=np.float64)
    if prefix.shape != (1500, 2 * channels) or not np.isfinite(prefix).all():
        raise ValueError("Invalid first30s normalization prefix")
    return np.mean(prefix, axis=0), np.maximum(np.std(prefix, axis=0), floors)


def replay_reference(raw, mean, scale, weights):
    """Neural-only full trace, clipping predictions only after the linear head."""
    z = np.clip((raw - mean) / scale, -8, 8)
    prediction = z @ weights[:-1] + weights[-1]
    magnitude = np.linalg.norm(prediction, axis=1)
    return prediction / np.maximum(1.0, magnitude)[:, None]


def training_arrays_reference(case, records, raw, *, stop, training, target, weighting, control):
    """Purchased-only fit operands, independent of model/producer helpers."""
    n = len(case["y"])
    if training == "continuous":
        mask = burned_mask(case["eval_mask"], case["segments"])
        mask[:1500] = False
        mask[min(stop, n) :] = False
        indices = np.flatnonzero(mask)
        labels = np.asarray(case["y"][indices], dtype=np.float64)
        ids = np.asarray(case["context"]["trial_id"])[indices]
        weights = np.ones(len(indices), dtype=np.float64)
    else:
        rt = target[1]
        selected = [
            r
            for r in records
            if r["eligible"]
            and not r["right_censored"]
            and r["onset_seconds"] >= 30
            and r["completion_bin"] <= min(stop, n)
        ]
        selected.sort(key=lambda r: (r["completion_bin"], r["trial_id"]))
        indices = np.asarray([r["windows"][rt][1] - 1 for r in selected], dtype=np.int64)
        ids = np.asarray([r["trial_id"] for r in selected], dtype=np.int64)
        labels = np.asarray(
            [
                r["window_targets"][rt]
                if target[0] == "window_mean"
                else case["y"][r["windows"][rt][1] - 1]
                for r in selected
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        sectors = np.asarray([r["sector"] for r in selected], dtype=np.int64)
        weights = np.ones(len(indices), dtype=np.float64)
        if weighting == "sector" and len(indices):
            counts = np.bincount(sectors, minlength=4)
            weights = 1.0 / counts[sectors]
            weights *= len(weights) / weights.sum()
    if control == "zero":
        indices, ids, labels, weights = (
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            np.empty((0, 2)),
            np.empty(0),
        )
    elif control == "trial_permutation" and len(labels):
        labels = np.roll(labels, -1, axis=0)
    channels = case["features"]["x_tx45"].shape[1]
    mean, scale = normalization_reference(raw, channels)
    design = np.c_[np.clip((raw[indices] - mean) / scale, -8, 8), np.ones(len(indices))]
    return {
        "indices": indices,
        "trial_ids": ids,
        "labels": labels,
        "sample_weights": weights,
        "mean": mean,
        "scale": scale,
        "design": design,
    }


METHODS = (
    "baseline",
    "duration",
    "quota_continuous",
    "trial_startup_sector_mean",
    "trial_duration_sector_mean",
    "quota_uniform_endpoint",
    "quota_sector_endpoint",
    "quota_uniform_mean",
    "quota_sector_mean",
    "zero",
    "constant",
    "trialshuffle",
)
PRIMARY = "quota_sector_mean"
CONTRASTS = {
    "primary_minus_baseline": (PRIMARY, "baseline"),
    "primary_minus_zero": (PRIMARY, "zero"),
    "primary_minus_duration": (PRIMARY, "duration"),
    "duration_minus_baseline": ("duration", "baseline"),
    "primary_minus_quota_continuous": (PRIMARY, "quota_continuous"),
    "sector_weighting": (PRIMARY, "quota_uniform_mean"),
    "window_target": (PRIMARY, "quota_sector_endpoint"),
    "full_budget_minus_quota": ("trial_duration_sector_mean", PRIMARY),
    "quota_minus_startup_trial": (PRIMARY, "trial_startup_sector_mean"),
}


def interval_reference(values):
    if not len(values):
        return [None, None]
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap units")
    choices = np.random.default_rng(20261023).integers(0, len(values), (10000, len(values)))
    return np.quantile(np.mean(values[choices], axis=1), [0.025, 0.975]).tolist()


def summary_reference(rows):
    """Reconstruct session pairing, equal-person weighting and both fixed gates."""
    table = unique(rows, lambda r: (r["session_id"], r["method"]))
    ids = sorted({r["session_id"] for r in rows})
    if not ids or set(table) != {(s, m) for s in ids for m in METHODS}:
        raise ValueError("Complete twelve-method inventory required")
    people = sorted({r["participant"] for r in rows}, key=lambda p: int(p[1:]))
    for s in ids:
        group = [table[s, m] for m in METHODS]
        for key in ("participant", "score_mask_sha256"):
            if len({r[key] for r in group}) != 1:
                raise ValueError("Unpaired session identity or mask")
        if len({r["metrics"]["r2"] is None for r in group}) != 1:
            raise ValueError("Method-dependent missing outcome")
    methods, contrasts = {}, {}
    for m in METHODS:
        person_rows = []
        for p in people:
            selected = [r for r in rows if r["method"] == m and r["participant"] == p]
            usable = [r for r in selected if r["metrics"]["r2"] is not None]
            scores = [r["metrics"]["r2"] for r in usable]
            person_rows.append(
                {
                    "participant": p,
                    "selected_sessions": len(selected),
                    "scored_sessions": len(usable),
                    "mean_r2": average(scores),
                    "positive_sessions": sum(x > 0 for x in scores),
                    "session_bootstrap_95": interval_reference(scores),
                    "mean_summed_mse": average([r["metrics"]["summed_mse"] for r in usable]),
                    "sessions": [
                        {k: r[k] for k in ("session_id", "post_implant_day", "metrics")}
                        for r in selected
                    ],
                }
            )
        defined = [u for u in person_rows if u["mean_r2"] is not None]
        values = [u["mean_r2"] for u in defined]
        methods[m] = {
            "participants": person_rows,
            "mean_r2": average(values),
            "participant_bootstrap_95": interval_reference(values),
            "scored_sessions": sum(u["scored_sessions"] for u in person_rows),
            "n_participants": len(defined),
            "non_T2_mean_r2": average([u["mean_r2"] for u in defined if u["participant"] != "T2"]),
        }
    for name, (first, second) in CONTRASTS.items():
        units = []
        for p in people:
            pairs = [
                {
                    "session_id": s,
                    "gain": table[s, first]["metrics"]["r2"] - table[s, second]["metrics"]["r2"],
                }
                for s in ids
                if table[s, first]["participant"] == p
                and table[s, first]["metrics"]["r2"] is not None
            ]
            values = [x["gain"] for x in pairs]
            units.append(
                {
                    "participant": p,
                    "paired_sessions": len(pairs),
                    "mean_gain": average(values),
                    "session_bootstrap_95": interval_reference(values),
                    "session_gains": pairs,
                }
            )
        values = [u["mean_gain"] for u in units if u["mean_gain"] is not None]
        contrasts[name] = {
            "first": first,
            "second": second,
            "participants": units,
            "mean_gain": average(values),
            "paired_participant_95": interval_reference(values),
        }
    t2 = next((u for u in methods[PRIMARY]["participants"] if u["participant"] == "T2"), None)
    gains = {}
    for name in ("primary_minus_baseline", "primary_minus_zero", "primary_minus_duration"):
        person = next(
            (u for u in contrasts[name]["participants"] if u["participant"] == "T2"), None
        )
        gains[name] = bool(
            person
            and person["paired_sessions"] == 6
            and person["mean_gain"] is not None
            and person["mean_gain"] > 0
            and person["session_bootstrap_95"][0] > 0
        )
    useful = bool(
        t2
        and t2["selected_sessions"] == 6
        and t2["scored_sessions"] == 6
        and t2["mean_r2"] >= 0.05
        and t2["positive_sessions"] >= 4
        and gains["primary_minus_baseline"]
        and gains["primary_minus_zero"]
    )
    return {
        "selected_sessions": len(ids),
        "primary_method": PRIMARY,
        "methods": methods,
        "contrasts": contrasts,
        "gates": {
            "useful_T2_recovery": useful,
            "incremental_benefit_over_more_labels": useful and gains["primary_minus_duration"],
        },
        "interval_limit": (
            "T2 intervals resample available sessions from one person; "
            "descriptive, not independent participant replication"
        ),
    }


BASELINE = {
    "representation": "ewma200",
    "policy": "startup120",
    "training": "continuous",
    "target": "instantaneous",
    "weighting": "uniform",
    "rt_seconds": 0.2,
    "ridge": 1.0,
    "control": "neural",
}


def config_reference(config):
    if (
        set(config) != set(BASELINE)
        or config["representation"] not in ("ewma200", "boxcar300")
        or config["policy"] not in ("startup120", "duration360", "quota24")
        or config["training"] not in ("continuous", "trial")
        or config["target"] not in ("instantaneous", "window_mean", "endpoint")
        or config["weighting"] not in ("uniform", "sector")
        or type(config["rt_seconds"]) not in (int, float)
        or config["rt_seconds"] not in (0.2, 0.8)
        or type(config["ridge"]) not in (int, float)
        or config["ridge"] not in (0.1, 1)
        or config["control"] not in ("neural", "zero", "intercept", "trial_permutation")
    ):
        raise ValueError("Unregistered V9 configuration")
    if config["training"] == "continuous":
        if (
            config["representation"] != "ewma200"
            or config["target"] != "instantaneous"
            or config["weighting"] != "uniform"
            or config["ridge"] != 1
            or config["control"] == "trial_permutation"
        ):
            raise ValueError("Changed continuous reference recipe")
    elif (
        config["representation"] != "boxcar300"
        or config["target"] == "instantaneous"
        or config["control"] in ("zero", "intercept")
    ):
        raise ValueError("Changed trial readout recipe")


def configurations_reference(rt, ridge):
    baseline = {**BASELINE, "rt_seconds": rt}
    trial = {
        **baseline,
        "representation": "boxcar300",
        "policy": "quota24",
        "training": "trial",
        "target": "window_mean",
        "weighting": "sector",
        "ridge": ridge,
    }
    return {
        "baseline": baseline,
        "duration": {**baseline, "policy": "duration360"},
        "quota_continuous": {**baseline, "policy": "quota24"},
        "trial_startup_sector_mean": {**trial, "policy": "startup120"},
        "trial_duration_sector_mean": {**trial, "policy": "duration360"},
        "quota_uniform_endpoint": {**trial, "weighting": "uniform", "target": "endpoint"},
        "quota_sector_endpoint": {**trial, "target": "endpoint"},
        "quota_uniform_mean": {**trial, "weighting": "uniform"},
        PRIMARY: trial,
        "zero": {**baseline, "policy": "duration360", "control": "zero"},
        "constant": {**baseline, "policy": "duration360", "control": "intercept"},
        "trialshuffle": {**trial, "control": "trial_permutation"},
    }


def private_trial_reference(record, case):
    c = case["context"]
    windows = []
    for rt in (0.2, 0.8):
        a, b = record["windows"][rt]
        windows.append(
            {
                "rt_seconds": rt,
                "start_bin": a,
                "stop_bin": b,
                "window_mean": record["window_targets"][rt].tolist(),
                "endpoint": np.asarray(case["y"][b - 1], dtype=float).tolist(),
                "first_raw_age_seconds": float(c["trial_age_seconds"][a] - 0.01),
            }
        )
    return {
        "trial_id": record["trial_id"],
        "completion_bin": record["completion_bin"],
        "sector": record["sector"],
        "common_start_bin": record["common_start_bin"],
        "common_stop_bin": record["common_stop_bin"],
        "directional_coherence": record["coherence"],
        "target_excursion": record["target_excursion"],
        "reference40_distance": record["reference40_distance"],
        "windows": windows,
    }


def calibration_reference(case, records, policy):
    n = len(case["y"])
    stop = 7500 if policy == "startup120" else 19500
    selected, counts, attained = [], np.zeros(4, dtype=np.int64), False
    for r in sorted(records, key=lambda v: (v["completion_bin"], v["start_bin"], v["trial_id"])):
        if (
            not r["eligible"]
            or r["right_censored"]
            or r["onset_seconds"] < 30
            or r["completion_bin"] > min(n, stop)
        ):
            continue
        selected.append(r)
        counts[r["sector"]] += 1
        if policy == "quota24" and len(selected) >= 24 and np.all(counts >= 4):
            stop, attained = r["completion_bin"], True
            break
    reason = (
        "quota_met"
        if attained
        else "recording_ended"
        if n < stop
        else "deadline"
        if policy == "quota24"
        else "fixed_endpoint"
    )
    return selected, {
        "policy": policy,
        "label_start_bin": 1500,
        "label_stop_bin": stop,
        "deployment_bin": stop + 100,
        "quota_met": attained if policy == "quota24" else None,
        "stop_reason": reason,
        "eligible_trials": len(selected),
        "sector_trial_counts": counts.tolist(),
        "trial_deficit": max(0, 24 - len(selected)),
        "sector_deficits": np.maximum(0, 4 - counts).tolist(),
        "planned_label_seconds": 120.0 if policy == "startup120" else 360.0,
        "requested_label_seconds": max(0, stop - 1500) * 0.02,
        "available_label_seconds": max(0, min(n, stop) - 1500) * 0.02,
    }


def fit_reference(case, config, raw, records=None):
    config_reference(config)
    records = trial_records_reference(case) if records is None else records
    if config["control"] == "zero":
        selected = []
        purchase = {
            "policy": config["policy"],
            "label_start_bin": 1500,
            "label_stop_bin": 19500,
            "deployment_bin": 19600,
            "quota_met": None,
            "stop_reason": "zero_control",
            "eligible_trials": 0,
            "sector_trial_counts": [0, 0, 0, 0],
            "trial_deficit": 24,
            "sector_deficits": [4, 4, 4, 4],
            "planned_label_seconds": 0.0,
            "requested_label_seconds": 0.0,
            "available_label_seconds": 0.0,
        }
    else:
        selected, purchase = calibration_reference(case, records, config["policy"])
    stop, n = purchase["label_stop_bin"], len(case["y"])
    operands = training_arrays_reference(
        case,
        records,
        raw,
        stop=stop,
        training=config["training"],
        target=(config["target"], config["rt_seconds"]),
        weighting=config["weighting"],
        control=config["control"],
    )
    ix, y, w, ids = (operands[k] for k in ("indices", "labels", "sample_weights", "trial_ids"))
    mask = np.zeros(n, dtype=bool)
    mask[ix] = True
    sectors = (
        np.full(len(ix), -1, np.int64)
        if config["training"] == "continuous"
        else np.asarray([r["sector"] for r in selected], np.int64)
    )
    minimum = 50 if config["training"] == "continuous" else 1
    fallback = bool(
        n < stop or len(ix) < minimum or config["control"] == "trial_permutation" and len(ix) < 2
    )
    gram, rhs = ridge_reference(operands["design"], y, w, config["ridge"])
    total = float(w.sum())
    permutation = []
    if config["control"] == "trial_permutation" and len(ids):
        permutation = [
            {"recipient_trial": int(ids[i]), "donor_trial": int(ids[(i + 1) % len(ids)])}
            for i in range(len(ids))
        ]
    receipt = {
        "config": config,
        "normalization_start_bin": 0,
        "normalization_stop_bin": 1500,
        **purchase,
        "training_rows": len(ix),
        "training_trials": len(np.unique(ids)),
        "last_training_bin": int(ix[-1]) if len(ix) else None,
        "training_mask_sha256": ahash(mask),
        "training_indices_sha256": ahash(ix),
        "training_weights_sha256": ahash(w),
        "training_labels_sha256": ahash(y),
        "weight_sum": total,
        "effective_sample_size": total**2 / float(w @ w) if total else 0.0,
        "fallback_zero": fallback,
        "feature_penalty": config["ridge"],
        "free_intercept": True,
        "private_trials": [private_trial_reference(r, case) for r in selected],
        "private_permutation": permutation,
        "export_prediction_max_error": 0.0,
    }
    return {
        **operands,
        "gram": gram,
        "rhs": rhs,
        "receipt": receipt,
        "arrays": {
            "training_mask": mask,
            "training_indices": ix,
            "training_weights": w,
            "training_labels": y,
            "training_trial_ids": ids,
            "training_sectors": sectors,
        },
    }


def validate_fit(arrays, receipt, case, config, raw, records):
    if (
        set(arrays) != {"version", "config", "mean", "scale", "weights"}
        or arrays["version"].shape != ()
        or arrays["version"].dtype != np.int64
        or arrays["version"].item() != 9
        or arrays["config"].shape != ()
        or arrays["config"].dtype.kind != "U"
        or json.loads(arrays["config"].item()) != config
    ):
        raise ValueError("Wrong portable model version/configuration")
    ref = fit_reference(case, config, raw, records)
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
            raise ValueError("Wrong purchased-label mean control")
        residual = 0.0
    else:
        residual = float(
            np.linalg.norm(ref["gram"] @ w - ref["rhs"]) / max(1.0, np.linalg.norm(ref["rhs"]))
        )
        if residual > 1e-9:
            raise ValueError("Not the declared weighted ridge optimum")
    return ref, residual


def secondary_score_reference(actual, predicted):
    y, p = (
        np.asarray(actual, dtype=np.float64).reshape(-1, 2),
        np.asarray(predicted, dtype=np.float64).reshape(-1, 2),
    )
    if len(y) < 2:
        return {"trials": len(y), "r2": None, "summed_mse": None, "angular_error_degrees": None}
    error = float(np.sum(np.square(y - p)))
    variance = float(np.sum(np.square(y - np.mean(y, axis=0))))
    denom = np.linalg.norm(y, axis=1) * np.linalg.norm(p, axis=1)
    cosine = np.divide(np.sum(y * p, axis=1), denom, out=np.zeros(len(y)), where=denom > 1e-12)
    return {
        "trials": len(y),
        "r2": 1 - error / variance if variance > 1e-12 else None,
        "summed_mse": error / len(y),
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
    refs = {(r["trace"]["trace_path"], r["trace"]["trace_sha256"]) for r in rows}
    if len(refs) != 1:
        raise ValueError("Session requires one bound common trace")
    path, pin = next(iter(refs))
    verify(artifact(path), pin)
    trace = numeric(artifact(path))
    per_method = (
        "prediction",
        "training_mask",
        "training_indices",
        "training_weights",
        "training_labels",
        "training_trial_ids",
        "training_sectors",
        "secondary_indices",
        "secondary_trial_ids",
        "secondary_window_mean",
        "secondary_endpoint",
    )
    if set(trace) != {"y", "primary_mask"} | {m + "__" + f for m in table for f in per_method}:
        raise ValueError("Unexpected/missing trace fields")
    primary, purchased = masks_reference(case)
    for name, expected in (("y", case["y"]), ("primary_mask", primary)):
        if trace[name].dtype != expected.dtype or not np.array_equal(trace[name], expected):
            raise ValueError("Saved target/mask changed")
    records = trial_records_reference(case)
    secondary = [
        r
        for r in records
        if r["eligible"]
        and not r["right_censored"]
        and r["completion_bin"] <= len(primary)
        and np.all(primary[r["common_start_bin"] : r["common_stop_bin"]])
    ]
    secondary.sort(key=lambda r: (r["completion_bin"], r["start_bin"], r["trial_id"]))
    if "secondary_trials" in session_receipt:
        compare_structure(
            session_receipt["secondary_trials"],
            [private_trial_reference(r, case) for r in secondary],
            "heldout trial selection",
        )
    cache = representations_reference(case)
    errors, residuals, rebuilt = [], [], []
    for method, config in configurations.items():
        config_reference(config)
        row = table[method]
        compare_structure(row["model"], session_receipt["model_refs"][method], "model reference")
        compare_structure(
            row["training"], session_receipt["fit_receipts"][method], "duplicate fit receipt"
        )
        path = artifact(row["model"]["model_path"])
        verify(path, row["model"]["model_sha256"])
        arrays = numeric(path)
        raw = cache[config["representation"]]
        ref, residual = validate_fit(arrays, row["training"], case, config, raw, records)
        pred = replay_reference(raw, arrays["mean"], arrays["scale"], arrays["weights"])
        pred[: ref["receipt"]["deployment_bin"]] = 0
        saved = trace[method + "__prediction"]
        if (
            saved.shape != pred.shape
            or saved.dtype != np.float64
            or not np.isfinite(saved).all()
            or not np.allclose(saved, pred, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("Saved prediction differs from independent full replay")
        for name, expected in ref["arrays"].items():
            actual = trace[method + "__" + name]
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError("Saved purchased training arrays changed: " + name)
        rt = config["rt_seconds"]
        ends = np.asarray([r["windows"][rt][1] - 1 for r in secondary], np.int64)
        ids = np.asarray([r["trial_id"] for r in secondary], np.int64)
        targets = {
            "window_mean": np.asarray(
                [r["window_targets"][rt] for r in secondary], dtype=float
            ).reshape(-1, 2),
            "endpoint": np.asarray(case["y"][ends], dtype=float).reshape(-1, 2),
        }
        expected_secondary = {
            "secondary_indices": ends,
            "secondary_trial_ids": ids,
            **{"secondary_" + k: v for k, v in targets.items()},
        }
        for name, expected in expected_secondary.items():
            actual = trace[method + "__" + name]
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError("Saved secondary target/geometry changed: " + name)
        secondary_mask = np.zeros(len(primary), bool)
        secondary_mask[ends] = True
        item = case["item"]
        expected = {
            "participant": item["participant_id"],
            "session_id": item["session_id"],
            "post_implant_day": item["post_implant_day"],
            "method": method,
            "config": config,
            "n_bins": len(pred),
            "metrics": score_reference(case["y"], pred, primary),
            "score_mask_sha256": ahash(primary),
            "valid_rows": int(primary.sum()),
            "secondary": {
                **{k: secondary_score_reference(y, pred[ends]) for k, y in targets.items()},
                "score_mask_sha256": ahash(secondary_mask),
                "trials": len(secondary),
                "prediction_definition": (
                    "one_causal_prediction_at_each_complete_eligible_trial_window_end"
                ),
            },
            **{
                k: ref["receipt"][k]
                for k in (
                    "planned_label_seconds",
                    "requested_label_seconds",
                    "available_label_seconds",
                    "quota_met",
                    "stop_reason",
                    "eligible_trials",
                    "sector_trial_counts",
                    "trial_deficit",
                    "sector_deficits",
                    "label_stop_bin",
                    "deployment_bin",
                )
            },
            "training": ref["receipt"],
            "excluded_purchased_trial_count": len(purchased),
            "model": row["model"],
            "trace": row["trace"],
        }
        receipt = ref["receipt"]
        expected["calibration"] = {
            "policy": receipt["policy"],
            "planned_label_seconds": receipt["planned_label_seconds"],
            "requested_label_seconds": receipt["requested_label_seconds"],
            "observed_label_seconds": receipt["available_label_seconds"],
            "stop_bin": receipt["label_stop_bin"],
            "quota_attained": receipt["quota_met"],
            "timed_out": receipt["stop_reason"] == "deadline",
            "eligible_trials": receipt["eligible_trials"],
            "sector_counts": receipt["sector_trial_counts"],
            "sector_deficits": receipt["sector_deficits"],
        }
        compare_structure(row, expected, "model scores and geometry")
        rebuilt.append(expected)
        errors.append(float(np.max(np.abs(saved - pred))) if len(pred) else 0.0)
        residuals.append(residual)
    return rebuilt, errors, residuals


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def grid_reference():
    return {
        f"grid_{i:02d}": configurations_reference(rt, ridge)[PRIMARY]
        for i, (rt, ridge) in enumerate(((0.2, 0.1), (0.2, 1.0), (0.8, 0.1), (0.8, 1.0)))
    }


def expected_sources():
    return {
        f"neural_experiments/{v}/{name}.py"
        for v, names in {
            "v2": ("data",),
            "v3": ("data", "model", "study"),
            "v4": ("data", "model"),
            "v5": ("data",),
            "v6": ("data",),
            "v7": ("data", "fresh_data"),
            "v8": ("data",),
            "v9": (
                "data",
                "fresh_data",
                "model",
                "development",
                "assessment",
                "metrics",
                "prepare_fresh",
            ),
        }.items()
        for name in names
    }


def source_binding_reference(manifest, source_run, source_audit):
    if source_run is None or source_audit is None:
        raise ValueError("Fresh audit requires explicit retained source and passed source audit")
    mpath = Path(source_run).resolve() / "manifest.json"
    original, passed = read(mpath), read(source_audit)
    if (
        original["status"] != "complete"
        or original["phase"] != "development"
        or passed["status"] != "PASS"
        or passed["phase"] != "development"
        or passed["manifest_sha256"] != sha256(mpath)
        or passed["selection_sha256"] != canonical_sha256(original["selection"])
        or passed["plan_sha256"] != PLAN_SHA
    ):
        raise ValueError("Source audit does not bind exact completed source/recipe")
    published_path = artifact(manifest["source_selection"]["path"])
    verify(published_path, manifest["source_selection"]["sha256"])
    published = read(published_path)
    audit_path = Path(source_audit).resolve()
    if not audit_path.is_relative_to(ROOT):
        raise ValueError("Passed source audit must remain inside the repository")
    expected = {
        **original["selection"],
        "source_audit_status": "PASS",
        "source_manifest_sha256": sha256(mpath),
        "source_audit_sha256": sha256(audit_path),
        "source_audit_path": str(audit_path.relative_to(ROOT)),
    }
    compare_structure(published, expected, "exact audited published selection")
    compare_structure(manifest["code_sha256"], original["code_sha256"], "frozen scientific code")
    compare_structure(
        manifest["configurations"], published["configurations"], "fixed fresh methods"
    )
    if manifest["primary_method"] != PRIMARY or published["primary_method"] != PRIMARY:
        raise ValueError("Fresh primary substitution")
    choice = published["recipe_choice"]
    grid = grid_reference()
    if choice["method"] not in grid:
        raise ValueError("Unknown selected source candidate")
    compare_structure(choice["config"], grid[choice["method"]], "selected source recipe")
    compare_structure(
        published["configurations"],
        configurations_reference(choice["config"]["rt_seconds"], choice["config"]["ridge"]),
        "registered fresh configuration family",
    )
    return {
        "source_manifest_sha256": sha256(mpath),
        "source_audit_sha256": sha256(audit_path),
        "selection_sha256": sha256(published_path),
    }


def audit_run(run, output, *, data_audit, source_run=None, source_audit=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve existing audit receipt; use a fresh output")
    auditor_sha = sha256(__file__)
    run = Path(run).resolve()
    mpath = run / "manifest.json"
    manifest_pin = sha256(mpath)
    m = read(mpath)
    if (
        m["status"] != "complete"
        or m["schema_version"] != 1
        or m["study"] != "V9"
        or m["phase"] not in ("development", "assessment")
    ):
        raise ValueError("Completed registered V9 manifest required")
    if m["plan"]["sha256"] != PLAN_SHA:
        raise ValueError("Wrong locked study plan")
    plan_path = artifact(m["plan"]["path"])
    verify(plan_path, PLAN_SHA)
    plan = read(plan_path)
    if set(m["code_sha256"]) != expected_sources():
        raise ValueError("Incomplete frozen scientific source inventory")
    for path, pin in m["code_sha256"].items():
        verify(artifact(path), pin)
        verify(contained(run / "source_snapshot", path), pin)
    index_path = artifact(m["index"]["path"])
    verify(index_path, m["index"]["sha256"])
    inputs = Inputs(index_path, data_audit)
    if inputs.index["role"] != ("development" if m["phase"] == "development" else "fresh"):
        raise ValueError("Run/input role mismatch")
    refs = unique(m["sessions"], lambda r: (r.get("stage", "assessment"), r["session_id"]))
    visited = set()
    all_rows, errors, residuals = [], [], []
    trace_count = 0

    def check_group(stage, items, configs, rows):
        nonlocal trace_count
        expected = {(i["session_id"], name) for i in items for name in configs}
        table = unique(rows, lambda r: (r["session_id"], r["method"]))
        if set(table) != expected:
            raise ValueError("Incomplete stage case inventory")
        rebuilt = []
        for item in items:
            sid = item["session_id"]
            key = stage, sid
            if key not in refs:
                raise ValueError("Missing session export receipt")
            visited.add(key)
            case = inputs.case(sid)
            rr, ee, ss = check_session(
                case, configs, [table[sid, name] for name in configs], refs[key]
            )
            rebuilt.extend(rr)
            errors.extend(ee)
            residuals.extend(ss)
            trace_count += 1
            print(json.dumps({"audit_stage": stage, "session": sid, "models": len(rr)}), flush=True)
        all_rows.extend(rebuilt)
        return rebuilt

    binding = {}
    if m["phase"] == "development":
        if sha256(index_path) != plan["development"]["index_sha256"]:
            raise ValueError("Wrong exposed development index")
        verify(run / "source_snapshot" / "study-plan.json", PLAN_SHA)
        selected = [r for r in inputs.items.values() if r["participant_id"] in FRESH_PEOPLE]
        selected.sort(
            key=lambda r: (FRESH_PEOPLE.index(r["participant_id"]), r["source"]["archive_ordinal"])
        )
        if len(selected) != 96:
            raise ValueError("Exactly96 exposed model sessions required")
        for person in FRESH_PEOPLE:
            if [
                r["source"]["archive_ordinal"] for r in selected if r["participant_id"] == person
            ] != list(range(1, 25)):
                raise ValueError("Wrong locked source archive rank inventory")
        selection_items = [
            r
            for r in selected
            if r["participant_id"] == "T2" and 16 <= r["source"]["archive_ordinal"] <= 24
        ]
        if len(selection_items) != 9:
            raise ValueError("Exactlynine late exposed T2 selection sessions required")
        grid = grid_reference()
        compare_structure(m["grid_configurations"], grid, "four locked source candidates")
        grid_rows = check_group("grid", selection_items, grid, m["grid_rows"])
        choice = choose_reference(grid_rows, grid, [r["session_id"] for r in selection_items])
        compare_structure(m["grid_selection"], choice, "source ranking and tie rule")
        chosen = grid[choice["selected_method"]]
        configs = configurations_reference(chosen["rt_seconds"], chosen["ridge"])
        compare_structure(m["configurations"], configs, "selected fixed twelve methods")
        if m["primary_method"] != PRIMARY:
            raise ValueError("Source primary substitution")
        final_rows = check_group("final", selected, configs, m["final_rows"])
        compare_structure(
            m["rows"], m["grid_rows"] + m["final_rows"], "complete source row inventory"
        )
        selection = {
            "schema_version": 1,
            "study": "V9",
            "source_audit_status": "PENDING",
            "plan_sha256": PLAN_SHA,
            "index_sha256": sha256(index_path),
            "scientific_code_sha256": m["code_sha256"],
            "code_sha256": m["code_sha256"],
            "primary_method": PRIMARY,
            "configurations": configs,
            "recipe_choice": {"method": choice["selected_method"], "config": chosen},
            "grid_selection": choice,
        }
        compare_structure(m["selection"], selection, "complete source selection payload")
        binding["selection_sha256"] = canonical_sha256(selection)
        summary = summary_reference(final_rows)
    else:
        binding = source_binding_reference(m, source_run, source_audit)
        final_rows = check_group(
            "assessment", list(inputs.items.values()), m["configurations"], m["rows"]
        )
        summary = summary_reference(final_rows)
    compare_structure(m["summary"], summary, "aggregate paired uncertainty and gates")
    if set(refs) != visited:
        raise ValueError("Unverified extra session exports")
    verify(mpath, manifest_pin)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "phase": m["phase"],
        "auditor_sha256": auditor_sha,
        "numeric_reference_sha256": {
            f"neural_experiments/{v}/audit.py": sha256(ROOT / f"neural_experiments/{v}/audit.py")
            for v in ("v2", "v3", "v4", "v7", "v8")
        },
        "manifest_sha256": manifest_pin,
        "plan_sha256": PLAN_SHA,
        "index_sha256": sha256(index_path),
        "data_audit_sha256": sha256(inputs.data_audit_path),
        "data_audit_sessions": inputs.data_audit["sessions"],
        "models_checked": len(all_rows),
        "traces_checked": trace_count,
        "rows_checked": len(all_rows),
        "max_prediction_error": max(errors, default=0.0),
        "max_normal_equation_residual": max(residuals, default=0.0),
        "summary": summary,
        **binding,
        "scope": (
            "Independent numeric implementation checks every purchased fit, causal full "
            "trace, quota, shared trial purge, both endpoints, source choice, paired "
            "summary and gates. A separately bound raw-context audit verifies geometric "
            "sidecars. These audits do not establish biological intention or closed-loop "
            "efficacy."
        ),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


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
            {
                k: result[k]
                for k in (
                    "status",
                    "phase",
                    "models_checked",
                    "traces_checked",
                    "max_prediction_error",
                    "max_normal_equation_residual",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
