"""Independent participant-held-out protocol and saved-prediction audit.

No experiment model, importer, predictor, or scorer is imported. Only numeric
arrays, JSON receipts, and sklearn's independent R2 implementation are used.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

PARTICIPANTS = ("T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11")
METHODS = ("ridge", "scratch", "pretrained_frozen_core", "pretrained_finetune")
ARCHIVE_DIGESTS = {
    "T2": "892d860cb690b59368c0aa8a550e9ac691e3be95af33b55eb60609351df5d3a5",
    "T3": "bf0424759beda78a13c3ecb45bb5dcc48cbb23c867d1dc3f4f8866d023844230",
    "T5": "9bdd17c23734d0a16de5a8ae51fa100a7a957af7f80e31eda91b9b910dd4eab3",
    "T6": "4096e48994457e30571c6ce3e29b344abd27cbd9b805ff6c417355515312be61",
    "T7": "5887e32005441b04f37ece85381621247abb361f1168f73178163125c6bf6e4d",
    "T8": "e3c75e6117163bf1e71d93fd8d96d79b1447dc1f0014a557af2d074e24d95276",
    "T9": "253a4928670087366b4fbf0641b40eca36452e0fcac9e661630948adb64513ff",
    "T10": "4f35f977ddd43fc024bf2aa5f46c411b2314a526592232da6390cf20d937d7a8",
    "T11": "74f9735ff07ac7e47f8cfad16bf1070c5f3895ff279697d8bb8e96b6c6143c61",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_hash(mask):
    if np.asarray(mask).dtype != np.bool_:
        raise ValueError("Scoring mask must be boolean")
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def contained(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact path must be relative to its declared root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes its declared root")
    return path


def numeric(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def bins(seconds, bin_seconds):
    result = int(round(seconds / bin_seconds))
    if not np.isclose(result * bin_seconds, seconds, rtol=0, atol=1e-10):
        raise ValueError("Time budget does not contain an integer number of bins")
    return result


def validate_plan(config):
    if (
        config["schema_version"] != 1
        or config["provider_version"] != 462702
        or tuple(config["participants"]) != PARTICIPANTS
        or tuple(config["methods"]) != METHODS
        or config["seeds"] != [0, 1, 2]
        or config["calibration_budgets_seconds"] != [30, 120, 300]
        or config["bin_seconds"] != 0.02
        or config["common_suffix_start_seconds"] != 300
        or config["source_session_cap_seconds"] != 600
        or config["evaluation_session_cap_seconds"] != 600
        or config["minimum_suffix_scored_bins"] != 50
        or config["sessions_per_participant"] != 6
    ):
        raise ValueError("Plan differs from the prespecified nine-person protocol")
    if config["model"] != {
        "width": 32,
        "dilations": [1, 2, 4, 8, 16],
        "kernel_size": 3,
        "channel_dropout": 0.1,
        "residual_core": True,
    }:
        raise ValueError("Unregistered model architecture")
    if config["ridge"] != {"alpha": 10.0, "fir_taps": 12, "fir_tau_bins": 12}:
        raise ValueError("Unregistered ridge settings")
    if config["source_training"] != {
        "steps": 2400,
        "checkpoints": [600, 1200, 2400],
        "batch_chunks": 8,
        "chunk_bins": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }:
        raise ValueError("Unregistered source training schedule")
    if config["adaptation"] != {
        "epochs": 40,
        "batch_chunks": 8,
        "chunk_bins": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }:
        raise ValueError("Unregistered target adaptation schedule")


def sessions_by_person(index, config):
    sessions = index["sessions"]
    ids = [row["session_id"] for row in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate session identity")
    grouped = {person: [] for person in config["participants"]}
    for row in sessions:
        if row["participant_id"] not in grouped:
            raise ValueError("Unexpected participant in prepared index")
        if row["bin_seconds"] != config["bin_seconds"]:
            raise ValueError("Prepared bin duration differs from plan")
        grouped[row["participant_id"]].append(row)
    for person, rows in grouped.items():
        rows.sort(key=lambda row: (int(row["post_implant_day"]), row["session_id"]))
        if not rows or len(rows) > config["sessions_per_participant"]:
            raise ValueError(f"Missing participant or excess selected sessions: {person}")
        days = [int(row["post_implant_day"]) for row in rows]
        if len(days) != len(set(days)):
            raise ValueError("Multiple selected sessions share an ambiguous implant day")
    return grouped


def validate_segments(row, features):
    n = row["n_bins"]
    if type(n) is not int or n < 0 or row["n_channels"] not in (96, 192):
        raise ValueError("Invalid prepared session geometry")
    if set(features) != {"x", "timestamps", "sequence_id", "channel_ids"}:
        raise ValueError("Feature file contains undeclared fields")
    x = features["x"]
    if x.shape != (n, row["n_channels"]) or x.dtype != np.float32:
        raise ValueError("Feature shape or dtype mismatch")
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Counts must be finite and nonnegative")
    if features["timestamps"].shape != (n,) or features["sequence_id"].shape != (n,):
        raise ValueError("Feature time/reset metadata shape mismatch")
    if (
        features["timestamps"].dtype != np.float64
        or features["sequence_id"].dtype != np.int64
        or features["channel_ids"].dtype != np.int64
        or np.any(np.diff(features["timestamps"]) <= 0)
    ):
        raise ValueError("Feature metadata dtype or chronology mismatch")
    if features["channel_ids"].shape != (row["n_channels"],):
        raise ValueError("Channel identity shape mismatch")
    if len(np.unique(features["channel_ids"])) != row["n_channels"]:
        raise ValueError("Duplicated channel identity")
    cursor, sequence_ids = 0, []
    for segment in row["segments"]:
        start, end = segment["start"], segment["end"]
        if start != cursor or not start <= end <= n:
            raise ValueError("Segments must partition the complete observation stream")
        sequence_ids.append(segment["sequence_id"])
        if not np.all(features["sequence_id"][start:end] == segment["sequence_id"]):
            raise ValueError("Reset identifiers disagree with segment receipt")
        timestamp = features["timestamps"][start:end]
        if not np.isfinite(timestamp).all() or not np.allclose(
            np.diff(timestamp), row["bin_seconds"], rtol=0, atol=1e-8
        ):
            raise ValueError("Acquisition discontinuity inside a declared segment")
        cursor = end
    if cursor != n or len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("Missing bins or reused reset identifier")


def source_roles(grouped, held_out):
    train, validation = [], []
    for person, rows in grouped.items():
        if person == held_out:
            continue
        if len(rows) == 1:
            train.append(rows[0]["session_id"])
        else:
            train.extend(row["session_id"] for row in rows[:-1])
            validation.append(rows[-1]["session_id"])
    return sorted(train), sorted(validation)


def validate_provenance(index, cache, config):
    """Bind selected member bytes to the pinned, outcome-independent archive walk."""
    provider = index["provider"]
    if (
        index["schema_version"] != 1
        or index["participants"] != list(PARTICIPANTS)
        or provider["doi"] != "10.5061/dryad.x0k6djj1h"
        or provider["version_id"] != 462702
        or provider["license"] != "CC0-1.0"
    ):
        raise ValueError("Prepared index provider/cohort identity mismatch")
    policy = index["selection"]
    expected_policy = {
        "selection": "first_n_complete_session_mat_members_in_provider_archive_order",
        "scores_used_for_selection": False,
        "ordering_after_selection": "numeric_post_implant_day_then_session_id",
        "neural_feature": "tx_4_5",
        "fallback": None,
        "count_conversion": "sum_two_consecutive_10ms_bins_within_each_block",
        "bin_seconds": 0.02,
        "normalization": "none",
        "upstream_feature_causality": "unverified",
        "max_sessions_per_participant": config["sessions_per_participant"],
    }
    if any(policy.get(k) != v for k, v in expected_policy.items()):
        raise ValueError("Outcome-independent acquisition/preparation policy mismatch")
    manifest_path = contained(cache, provider["manifest_path"])
    verify(manifest_path, provider["manifest_sha256"])
    provider_manifest = read_json(manifest_path)
    for entry in provider_manifest["metadata_sources"]:
        verify(contained(cache, entry["path"]), entry["sha256"])
    archives = {a["participant_id"]: a for a in provider_manifest["archives"]}
    receipts = {a["path"]: a["sha256"] for a in index["acquisition_sources"]}
    if len(receipts) != len(index["acquisition_sources"]) or set(receipts) != {
        f"acquisition/{p}.json" for p in PARTICIPANTS
    }:
        raise ValueError("Missing or duplicated acquisition receipt")
    checked = 0
    for number, person in enumerate(PARTICIPANTS):
        archive = archives[person]
        if (
            archive["file_id"] != 4952113 + number
            or archive["provider_sha256"] != ARCHIVE_DIGESTS[person]
            or archive["filename"] != f"decoding_{person}.tar.gz"
        ):
            raise ValueError("Pinned provider archive identity mismatch")
        receipt_path = contained(cache, f"acquisition/{person}.json")
        verify(receipt_path, receipts[f"acquisition/{person}.json"])
        receipt = read_json(receipt_path)
        if (
            receipt["status"] != "complete"
            or receipt["participant_id"] != person
            or receipt["requested_sessions"] != config["sessions_per_participant"]
            or any(
                receipt["archive"][k] != archive[k] for k in ("file_id", "size", "provider_sha256")
            )
        ):
            raise ValueError("Acquisition receipt identity or selection budget mismatch")
        if any(
            receipt["policy"].get(k) != v
            for k, v in expected_policy.items()
            if k != "max_sessions_per_participant"
        ):
            raise ValueError("Acquisition receipt selection policy mismatch")
        if receipt["full_archive_sha256_verified"] is not False:
            raise ValueError("Prefix acquisition cannot assert full archive verification")
        pattern = rf"(?:\./)?decoding/{person}/{person}_day_(\d+)_decoding\.mat"
        walk = receipt["seen_members"]
        if [r["ordinal"] for r in walk] != list(range(len(walk))):
            raise ValueError("Archive walk is incomplete or reordered")
        chosen = [r for r in walk if r["regular_file"] and re.fullmatch(pattern, r["name"])][
            : config["sessions_per_participant"]
        ]
        selected = receipt["selected"]
        if len(chosen) != len(selected) or not selected:
            raise ValueError("Selected members differ from the initial eligible archive members")
        if len(selected) < config["sessions_per_participant"] and not receipt["tar_end_observed"]:
            raise ValueError("Unfinished selection cannot be reported complete")
        rows = {r["session_id"]: r for r in index["sessions"] if r["participant_id"] == person}
        if set(rows) != {r["session_id"] for r in selected}:
            raise ValueError("Prepared index omitted or replaced a selected session")
        for member, seen in zip(selected, chosen, strict=True):
            row = rows[member["session_id"]]
            match = re.fullmatch(pattern, member["member_name"])
            if (
                match is None
                or member["participant_id"] != person
                or member["post_implant_day"] != int(match.group(1))
                or member["member_name"] != seen["name"]
                or member["archive_ordinal"] != seen["ordinal"]
                or member["member_size"] != seen["size"]
                or any(row["source"].get(k) != v for k, v in member.items())
                or row["source"]["archive_sha256_published"] != ARCHIVE_DIGESTS[person]
                or row["post_implant_day"] != member["post_implant_day"]
            ):
                raise ValueError("Selected-member provenance disagrees with prepared session")
            raw = contained(cache, member["raw_path"])
            if raw.stat().st_size != member["member_size"]:
                raise ValueError("Selected raw member length mismatch")
            verify(raw, member["member_sha256"])
            checked += 1
    return checked


def effective_training_mask(mask, segments, stop, burn):
    result = np.asarray(mask[:stop]).copy()
    for segment in segments:
        result[segment["start"] : min(segment["start"] + burn, segment["end"], stop)] = False
    return result


def normalization(x):
    # sqrt is intentionally float32, matching the declared feature transform.
    values = np.sqrt(x).astype(np.float64)
    if not len(values):
        raise ValueError("Empty normalization prefix")
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 0.1).astype(np.float32)


def validate_model_export(identity, arrays, prefix_x, config, core=None):
    channels, width = prefix_x.shape[1], config["model"]["width"]
    stats = normalization(prefix_x)
    for name, expected in zip(("input_mean", "input_scale"), stats, strict=True):
        if (
            arrays[name].dtype != np.float32
            or arrays[name].shape != (channels,)
            or not np.array_equal(arrays[name], expected)
        ):
            raise ValueError("Exported normalization differs from exact allowed feature prefix")
    if identity["method"] == "ridge":
        shapes = {"coefficient": (2, channels), "intercept": (2,)}
        expected_keys = {*shapes, "input_mean", "input_scale"}
    else:
        if (
            arrays["format_version"].shape != ()
            or arrays["format_version"] != 2
            or not np.array_equal(arrays["dilations"], config["model"]["dilations"])
        ):
            raise ValueError("Unregistered exported model format or dilations")
        shapes = {
            "adapter.weight": (width, channels),
            "adapter.bias": (width,),
            "head.weight": (2, width),
            "head.bias": (2,),
        }
        for layer in range(len(config["model"]["dilations"])):
            shapes[f"core.{layer}.weight"] = (width, width, 3)
            shapes[f"core.{layer}.bias"] = (width,)
        expected_keys = {*shapes, "input_mean", "input_scale", "format_version", "dilations"}
    if set(arrays) != expected_keys:
        raise ValueError("Model export contains missing or undeclared parameters")
    for name, shape in shapes.items():
        if (
            arrays[name].shape != shape
            or arrays[name].dtype.kind != "f"
            or not np.isfinite(arrays[name]).all()
        ):
            raise ValueError("Model parameter shape or numeric validity mismatch")
        if identity["method"] == "pretrained_frozen_core" and name.startswith("core."):
            if core is None or not np.array_equal(arrays[name], core[name.removeprefix("core.")]):
                raise ValueError("Frozen core changed after source training")
    if identity["parameter_count"] != sum(arrays[k].size for k in shapes):
        raise ValueError("Reported model parameter count differs from numeric export")


def validate_model_roles(identity, grouped, config):
    person, method = identity["target_participant"], identity["method"]
    if person not in grouped or method not in config["methods"]:
        raise ValueError("Unexpected model participant/method")
    if (
        identity["seed"] not in config["seeds"]
        or identity["budget_seconds"] not in config["calibration_budgets_seconds"]
    ):
        raise ValueError("Unexpected seed or calibration budget")
    train, validation = source_roles(grouped, person)
    participants = sorted(set(grouped) - {person})
    if method in {"ridge", "scratch"}:
        train, validation, participants = [], [], []
    for key, expected in (
        ("source_train_session_ids", train),
        ("source_validation_session_ids", validation),
        ("source_participants", participants),
    ):
        if sorted(identity[key]) != expected:
            raise ValueError(f"Source role overlap, omission, or wrong method role: {key}")
    first = grouped[person][0]
    stop = bins(identity["budget_seconds"], config["bin_seconds"])
    if stop > first["n_bins"]:
        raise ValueError("Calibration exceeds earliest target session duration")
    expected = {"session_id": first["session_id"], "start_bin": 0, "stop_bin": stop}
    for name in ("calibration", "normalization"):
        if any(identity[name].get(key) != value for key, value in expected.items()):
            raise ValueError(f"{name} uses data outside the exact target prefix")


def score_arrays(target, prediction, mask):
    """Independent exact pooled-SSE R2 plus angular accounting.

    Centering each coordinate then flattening lets sklearn reconstruct the
    declared sum-SSE/sum-SST metric even when one coordinate has zero variance.
    Such a coordinate's prediction errors must not disappear from pooled SSE.
    """
    target, prediction, mask = map(np.asarray, (target, prediction, mask))
    if target.ndim != 2 or target.shape[1] != 2 or prediction.shape != target.shape:
        raise ValueError("Targets/predictions require matching N by 2 shape")
    if mask.dtype != np.bool_ or mask.shape != (len(target),):
        raise ValueError("Scoring mask shape or dtype mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite prediction or scored label")
    y, p = target[mask].astype(float), prediction[mask].astype(float)
    n = len(y)
    result = {
        "r2": None,
        "per_output_r2": [None, None],
        "angular_error_degrees": None,
        "zero_prediction_count": 0,
        "n": n,
    }
    if not n:
        return result
    target_norm, prediction_norm = np.linalg.norm(y, axis=1), np.linalg.norm(p, axis=1)
    if np.any(target_norm <= 0):
        raise ValueError("Zero-direction target included in scoring mask")
    zero = prediction_norm <= 1e-12
    cosine = np.divide(
        np.sum(y * p, axis=1),
        target_norm * prediction_norm,
        out=np.zeros(n),
        where=target_norm * prediction_norm > 1e-12,
    )
    result["angular_error_degrees"] = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean())
    result["zero_prediction_count"] = int(zero.sum())
    if n < 2:
        return result
    centered = y - y.mean(axis=0)
    variance = np.sum(centered**2, axis=0)
    if variance.sum() > 1e-12:
        result["r2"] = float(r2_score(centered.ravel(), (p - y.mean(axis=0)).ravel()))
    for output in range(2):
        if variance[output] > 1e-12:
            result["per_output_r2"][output] = float(r2_score(y[:, output], p[:, output]))
    return result


def compare_metrics(recorded, expected):
    for key, value in expected.items():
        if key not in recorded:
            raise ValueError(f"Missing metric: {key}")
        actual = recorded[key]
        if value is None:
            if actual is not None:
                raise ValueError(f"Undefined metric was replaced: {key}")
        elif isinstance(value, list):
            if not isinstance(actual, list) or len(actual) != len(value):
                raise ValueError("Per-output score shape mismatch")
            for a, e in zip(actual, value, strict=True):
                if e is None:
                    if a is not None:
                        raise ValueError("Undefined per-output score was replaced")
                elif not np.isclose(a, e, rtol=1e-9, atol=1e-10):
                    raise ValueError("Per-output score mismatch")
        elif isinstance(value, int) and (type(actual) is not int or actual != value):
            raise ValueError(f"Metric count mismatch: {key}")
        elif not np.isclose(actual, value, rtol=1e-9, atol=1e-10):
            raise ValueError(f"Independent metric mismatch: {key}")


def compare_structure(actual, expected, path="summary"):
    """Exact identities/counts, tolerance only for independently computed floats."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Missing or additional aggregate fields: {path}")
        for key, value in expected.items():
            compare_structure(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Aggregate inventory mismatch: {path}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare_structure(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not np.isclose(
            actual, expected, rtol=1e-9, atol=1e-10
        ):
            raise ValueError(f"Independent aggregate mismatch: {path}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Aggregate identity/count mismatch: {path}")


def reconstruct_summary(scored, config):
    """Independent day -> seed -> person aggregation and paired-person inference."""
    cases = {}
    for row in scored:
        key = tuple(row[k] for k in ("target_participant", "method", "seed", "budget_seconds"))
        cases.setdefault(key, []).append(row)
    rows = []
    for (person, method, seed, budget), traces in cases.items():
        for partition in ("same_session_suffix", "future_day"):
            values = [
                r["metrics"]["r2"]
                for r in traces
                if r["partition"] == partition and r["metrics"]["r2"] is not None
            ]
            if values:
                rows.append(
                    {
                        "participant": person,
                        "method": method,
                        "seed": seed,
                        "budget_seconds": budget,
                        "partition": partition,
                        "mean_day_r2": float(np.mean(values)),
                        "n_days": len(values),
                    }
                )
    groups, by_person = [], {}
    for partition, budget, method in itertools.product(
        ("same_session_suffix", "future_day"),
        config["calibration_budgets_seconds"],
        config["methods"],
    ):
        people = []
        for person in config["participants"]:
            values = [
                r["mean_day_r2"]
                for r in rows
                if (r["participant"], r["method"], r["budget_seconds"], r["partition"])
                == (person, method, budget, partition)
            ]
            if values:
                person_score = {
                    "participant": person,
                    "r2": float(np.mean(values)),
                    "n_seeds": len(values),
                }
                people.append(person_score)
                by_person[(person, method, budget, partition)] = person_score
        groups.append(
            {
                "partition": partition,
                "budget_seconds": budget,
                "method": method,
                "people": people,
                "mean_person_r2": float(np.mean([p["r2"] for p in people])) if people else None,
            }
        )
    effects = []
    for person in config["participants"]:
        frozen = by_person.get((person, "pretrained_frozen_core", 120, "future_day"))
        scratch = by_person.get((person, "scratch", 120, "future_day"))
        if (
            frozen is not None
            and scratch is not None
            and frozen["n_seeds"] == scratch["n_seeds"] == len(config["seeds"])
        ):
            effects.append({"participant": person, "difference": frozen["r2"] - scratch["r2"]})
    inference = {"effects": effects, "n_people": len(effects), "mean_difference": None}
    if effects:
        delta = np.array([r["difference"] for r in effects])
        resampled = np.random.default_rng(20261022).choice(delta, (10000, len(delta)), replace=True)
        signed_means = [
            float(np.mean(delta * signs)) for signs in itertools.product((-1, 1), repeat=len(delta))
        ]
        inference.update(
            mean_difference=float(delta.mean()),
            bootstrap_percentile_95=np.quantile(resampled.mean(1), [0.025, 0.975]).tolist(),
            exact_two_sided_sign_flip_p=float(
                np.mean(np.abs(signed_means) >= abs(delta.mean()) - 1e-12)
            ),
            positive_people=int(np.count_nonzero(delta > 0)),
            target_gain_met=bool(delta.mean() >= config["success_target"]["absolute_r2_gain"]),
        )
    return {"rows": rows, "groups": groups, "primary_contrast": inference}


def validate_pretraining(manifest, grouped, config, root, source_stats):
    cores = {}
    for reference in manifest["pretraining"]:
        key = (reference["target_participant"], reference["seed"])
        if key in cores or key[0] not in grouped or key[1] not in config["seeds"]:
            raise ValueError("Unexpected or repeated source-training case")
        path = contained(root, reference["receipt_path"])
        verify(path, reference["receipt_sha256"])
        receipt = read_json(path)
        people = sorted(set(grouped) - {key[0]})
        if receipt["seed"] != key[1] or receipt["source_people"] != people:
            raise ValueError("Held-out participant entered source checkpoint training")
        train, validation = source_roles(grouped, key[0])
        if (
            sorted(receipt["source_train_session_ids"]) != train
            or sorted(receipt["source_validation_session_ids"]) != validation
        ):
            raise ValueError("Source checkpoint training/validation session roles mismatch")
        cap = bins(config["source_session_cap_seconds"], config["bin_seconds"])
        expected_ranges = {
            person: [
                {
                    "session_id": row["session_id"],
                    "start_bin": 0,
                    "stop_bin": min(cap, row["n_bins"]),
                }
                for row in (grouped[person][:-1] or grouped[person])
            ]
            for person in people
        }
        compare_structure(receipt["source_normalization"], expected_ranges, "source.normalization")
        norm_path = contained(root, receipt["source_normalization_path"])
        verify(norm_path, receipt["source_normalization_sha256"])
        stats = numeric(norm_path)
        if set(stats) != {f"{p}__{name}" for p in people for name in ("input_mean", "input_scale")}:
            raise ValueError("Source normalization participant/field inventory mismatch")
        for person in people:
            for name, expected in zip(
                ("input_mean", "input_scale"), source_stats[person], strict=True
            ):
                if not np.array_equal(stats[f"{person}__{name}"], expected):
                    raise ValueError(
                        "Exported source normalization consumed nontraining observations"
                    )
        losses = receipt["losses"]
        if [r["step"] for r in losses] != list(range(1, config["source_training"]["steps"] + 1)):
            raise ValueError("Source training schedule differs from locked plan")
        for step, row in enumerate(losses):
            if (
                row["participant"] != people[step % len(people)]
                or row["n"] < 1
                or not np.isfinite(row["loss"])
                or row["loss"] < 0
            ):
                raise ValueError("Invalid source-only training loss receipt")
        checkpoints = receipt["checkpoints"]
        if [r["step"] for r in checkpoints] != config["source_training"]["checkpoints"]:
            raise ValueError("Unregistered source checkpoint selection")
        if any(
            not np.isfinite(r["validation_equal_person_mse"])
            or r["validation_equal_person_mse"] < 0
            or not 0 <= r["validation_people"] <= len(people)
            for r in checkpoints
        ):
            raise ValueError("Invalid source validation receipt")
        chosen = min(checkpoints, key=lambda r: r["validation_equal_person_mse"])["step"]
        if receipt["selected_step"] != chosen:
            raise ValueError("Selected checkpoint differs from source-only minimum")
        core_path = path.parent / "core.npz"
        verify(core_path, receipt["core_sha256"])
        core = numeric(core_path)
        width = config["model"]["width"]
        shapes = {
            f"{i}.{kind}": (width, width, 3) if kind == "weight" else (width,)
            for i in range(len(config["model"]["dilations"]))
            for kind in ("weight", "bias")
        }
        if set(core) != set(shapes) or any(
            core[k].shape != shape or core[k].dtype != np.float32 or not np.isfinite(core[k]).all()
            for k, shape in shapes.items()
        ):
            raise ValueError("Source core shape/numeric validity mismatch")
        cores[key] = (receipt["core_sha256"], core)
    if set(cores) != set(itertools.product(config["participants"], config["seeds"])):
        raise ValueError("Missing source-pretraining cases")
    return cores


def audit(run, index_path, root):
    root, run, index_path = map(lambda p: Path(p).resolve(), (root, run, index_path))
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only a complete run may pass the audit")
    verify(index_path, manifest["index_sha256"])
    if contained(root, manifest["index_path"]) != index_path:
        raise ValueError("Run consumes a different index path")
    config_path = contained(root, manifest["config_path"])
    verify(config_path, manifest["config_sha256"])
    config = read_json(config_path)
    validate_plan(config)
    if set(manifest["source"]) != {
        "neural_experiments/v2/transfer.py",
        "neural_experiments/v2/data.py",
    } or set(manifest["source"]) != set(manifest["source_snapshots"]):
        raise ValueError("Consumed source snapshot inventory mismatch")
    for relative, digest in manifest["source"].items():
        verify(contained(root, manifest["source_snapshots"][relative]), digest)
    compare_structure(read_json(run / "plan.json"), config, "snapshot.plan")
    index = read_json(index_path)
    compare_structure(read_json(run / "input_index.json"), index, "snapshot.index")
    members_checked = validate_provenance(index, index_path.parent, config)
    grouped = sessions_by_person(index, config)
    rows = {row["session_id"]: row for values in grouped.values() for row in values}
    labels, prefixes, source_stats, source_chunks = {}, {}, {}, {}
    longest_prefix = bins(max(config["calibration_budgets_seconds"]), config["bin_seconds"])
    source_cap = bins(config["source_session_cap_seconds"], config["bin_seconds"])
    first_ids = {rr[0]["session_id"] for rr in grouped.values()}
    source_ids = {r["session_id"] for rr in grouped.values() for r in (rr[:-1] or rr)}
    for session_id, row in rows.items():
        feature_path = contained(index_path.parent, row["feature_path"])
        target_path = contained(index_path.parent, row["target_path"])
        verify(feature_path, row["feature_sha256"])
        verify(target_path, row["target_sha256"])
        features = numeric(feature_path)
        validate_segments(row, features)
        if session_id in first_ids:
            prefixes[session_id] = features["x"][:longest_prefix].copy()
        person = row["participant_id"]
        if session_id in source_ids:
            source_chunks.setdefault(person, []).append(features["x"][:source_cap].copy())
        if session_id == grouped[person][-1]["session_id"]:
            source_stats[person] = normalization(np.concatenate(source_chunks.pop(person)))
        target = numeric(target_path)
        if set(target) != {"y", "eval_mask"} or target["y"].shape != (row["n_bins"], 2):
            raise ValueError("Target array contract mismatch")
        if target["eval_mask"].shape != (row["n_bins"],) or target["eval_mask"].dtype != np.bool_:
            raise ValueError("Target mask contract mismatch")
        if (
            target["y"].dtype != np.float32
            or not np.isfinite(target["y"]).all()
            or not np.allclose(
                np.linalg.norm(target["y"][target["eval_mask"]], axis=1), 1, rtol=0, atol=1e-6
            )
        ):
            raise ValueError("Prepared target is not a finite, scored unit direction")
        labels[session_id] = target
    cores = validate_pretraining(manifest, grouped, config, root, source_stats)
    actual, scored = set(), []
    for entry in manifest["models"]:
        identity_path = contained(root, entry["identity_path"])
        verify(identity_path, entry["identity_sha256"])
        identity = read_json(identity_path)
        validate_model_roles(identity, grouped, config)
        case = (
            identity["target_participant"],
            identity["method"],
            identity["seed"],
            identity["budget_seconds"],
        )
        if case in actual:
            raise ValueError("Duplicate target model case")
        actual.add(case)
        model_path = contained(root, identity["model_path"])
        verify(model_path, identity["model_sha256"])
        first = grouped[case[0]][0]
        stop = identity["calibration"]["stop_bin"]
        prefix_mask = labels[first["session_id"]]["eval_mask"][:stop]
        if (
            identity["calibration"]["eligible_count"] != int(prefix_mask.sum())
            or identity["calibration"]["mask_sha256"] != mask_hash(prefix_mask)
            or not prefix_mask.any()
        ):
            raise ValueError("Calibration labels/count/mask differ from allowed prefix")
        effective = effective_training_mask(
            prefix_mask, first["segments"], stop, 2 * sum(config["model"]["dilations"])
        )
        if (
            identity["calibration"]["training_count"] != int(effective.sum())
            or identity["calibration"]["training_mask_sha256"] != mask_hash(effective)
            or effective.sum() < 50
        ):
            raise ValueError("Consumed calibration labels differ from common burn-in budget")
        core_hash, core = cores[(case[0], case[2])]
        expected_core_hash = core_hash if case[1].startswith("pretrained") else None
        if identity["core_sha256"] != expected_core_hash:
            raise ValueError("Target model points to wrong held-out-person source core")
        validate_model_export(
            identity, numeric(model_path), prefixes[first["session_id"]][:stop], config, core
        )
        cap = bins(config["evaluation_session_cap_seconds"], config["bin_seconds"])
        suffix = bins(config["common_suffix_start_seconds"], config["bin_seconds"])
        expected_sessions = {row["session_id"] for row in grouped[case[0]]}
        seen = set()
        for trace in identity["traces"]:
            session_id = trace["session_id"]
            if session_id in seen or session_id not in expected_sessions:
                raise ValueError("Duplicate or unexpected target evaluation session")
            seen.add(session_id)
            row, target = rows[session_id], labels[session_id]
            end = min(row["n_bins"], cap)
            partition = "same_session_suffix" if session_id == first["session_id"] else "future_day"
            start = min(suffix, end) if partition == "same_session_suffix" else 0
            if (trace["start_bin"], trace["stop_bin"], trace["partition"]) != (
                start,
                end,
                partition,
            ):
                raise ValueError("Evaluation interval or partition differs from locked protocol")
            mask = target["eval_mask"][:end].copy()
            mask[:start] = False
            if mask_hash(mask) != trace["score_mask_sha256"]:
                raise ValueError("Evaluation scoring rows differ across methods")
            prediction_path = contained(root, trace["prediction_path"])
            verify(prediction_path, trace["prediction_sha256"])
            prediction = np.load(prediction_path, allow_pickle=False)
            expected = score_arrays(target["y"][:end], prediction, mask)
            if (
                partition == "same_session_suffix"
                and mask.sum() < config["minimum_suffix_scored_bins"]
            ):
                expected = {
                    "r2": None,
                    "per_output_r2": [None, None],
                    "angular_error_degrees": None,
                    "zero_prediction_count": 0,
                    "n": int(mask.sum()),
                    "unavailable_reason": "fewer than minimum scored evaluation bins",
                }
            compare_structure(trace["metrics"], expected, "trace.metrics")
            scored.append(
                {
                    "target_participant": case[0],
                    "method": case[1],
                    "seed": case[2],
                    "budget_seconds": case[3],
                    "session_id": session_id,
                    "partition": partition,
                    "metrics": expected,
                }
            )
        if seen != expected_sessions:
            raise ValueError("Missing target evaluation session")
    expected_cases = {
        (person, method, seed, budget)
        for person, method, budget in itertools.product(
            config["participants"], config["methods"], config["calibration_budgets_seconds"]
        )
        for seed in ([0] if method == "ridge" else config["seeds"])
    }
    unavailable = set()
    for item in manifest["unavailable"]:
        case = (item["target_participant"], item["method"], item["seed"], item["budget_seconds"])
        if case in unavailable or case in actual or not item.get("reason"):
            raise ValueError("Duplicate, completed, or unexplained unavailable case")
        if case not in expected_cases:
            raise ValueError("Unavailable case is not part of registered experiment inventory")
        first = grouped[case[0]][0]
        stop = bins(case[3], config["bin_seconds"])
        if stop > first["n_bins"]:
            reason = "earliest session shorter than prefix"
        elif (
            effective_training_mask(
                labels[first["session_id"]]["eval_mask"],
                first["segments"],
                stop,
                2 * sum(config["model"]["dilations"]),
            ).sum()
            < 50
        ):
            reason = "fewer than50 scored prefix bins after common burn-in"
        else:
            raise ValueError("Feasible target case was omitted without a prespecified reason")
        if item["reason"] != reason:
            raise ValueError("Unavailable case reason differs from actual calibration eligibility")
        unavailable.add(case)
    if actual | unavailable != expected_cases:
        raise ValueError("Run does not account for the full prespecified model inventory")
    summary = read_json(run / "summary.json")
    reconstructed = reconstruct_summary(scored, config)
    for key, value in reconstructed.items():
        compare_structure(summary[key], value, f"summary.{key}")
    if (
        summary["manifest_sha256"] != sha256(run / "manifest.json")
        or summary["status"] != "complete"
        or summary["unavailable"] != manifest["unavailable"]
    ):
        raise ValueError("Summary is not bound to the complete audited run")
    return {
        "status": "passed",
        "models_checked": len(actual),
        "traces_checked": len(scored),
        "unavailable_models": len(unavailable),
        "selected_members_checked": members_checked,
        "source_cores_checked": len(cores),
        "participants": list(grouped),
        "manifest_sha256": sha256(run / "manifest.json"),
        "index_sha256": sha256(index_path),
        "config_sha256": sha256(config_path),
        "audit_source_sha256": sha256(__file__),
        "summary_sha256": sha256(run / "summary.json"),
        "independent_summary": reconstructed,
        "independent_scores": scored,
        "reference": "sklearn centered-flattened pooled-SSE R2; NumPy angular error",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output must be fresh")
    result = audit(args.run, args.index, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "independent_scores"}))


if __name__ == "__main__":
    main()
