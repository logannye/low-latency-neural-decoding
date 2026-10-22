"""Independent V3 arithmetic and protocol verification.

No experiment model, predictor, importer, or scorer is imported. Numerical
references use NumPy recurrences, ridge optimality conditions, and sklearn R².
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from neural_experiments.v2.audit import compare_structure, score_arrays

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA256 = "6d0f188bba2c9719bbb8dffece385d48a6b70e07b22835ee6413c89eee063392"
V2_INDEX_SHA256 = "91de850304c4fb96096daa3f9ec19edab0372675c236a899005cf4308ec6c523"
SOURCE_PINS = {
    "neural_experiments/v3/model.py": (
        "56f2bfabf8383c65ec621407e9bccd026b295e27912011c4b9cf97173d4fe2c2"
    ),
    "neural_experiments/v3/study.py": (
        "8a193512fd5bc4d3b1447382de5ae9b56f5dc87c51bccdd393ad81f44e336d43"
    ),
    "neural_experiments/v3/data.py": (
        "260442fe492e1c01cf19e888edf18eb625c120cf865dde30c46ec85f3ccaa390"
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def mask_hash(mask):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise ValueError("Expected boolean mask")
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def numeric(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def contained(root, value):
    value = Path(value)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Artifact path must be relative to declared root")
    path = (Path(root).resolve() / value).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes declared root")
    return path


def grid(plan):
    return [
        {
            "kind": "regularized",
            "family": family,
            "tau_seconds": tau,
            "baseline_tau_seconds": baseline,
            "ridge_lambda": penalty,
        }
        for family, tau, baseline, penalty in itertools.product(
            plan["features"]["families"],
            plan["grid"]["tau_seconds"],
            plan["grid"]["baseline_tau_seconds"],
            plan["grid"]["ridge_lambda"],
        )
    ]


def segment_ranges(segments, n):
    """Preserve all declared bins, allowing empty provenance blocks."""
    ranges = [
        (int(s["start"]), int(s["end"])) if isinstance(s, dict) else (int(s[0]), int(s[1]))
        for s in segments
    ]
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != n:
        raise ValueError("Segments do not cover the recording")
    if any(a > b for a, b in ranges) or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Segments overlap or omit recording bins")
    return [(a, b) for a, b in ranges if a < b]


def clipped_segments(segments, stop):
    return [(a, min(b, stop)) for a, b in segments if a < stop]


def training_mask(mask, segments, burn=62):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError("Invalid calibration mask")
    result = mask.copy()
    for start, end in segment_ranges(segments, len(mask)):
        result[start : min(start + burn, end)] = False
    return result


def transformed(tx, sbp, family):
    tx, sbp = np.asarray(tx, float), np.asarray(sbp, float)
    if (
        tx.ndim != 2
        or tx.shape != sbp.shape
        or not np.isfinite(tx).all()
        or not np.isfinite(sbp).all()
        or np.any(tx < 0)
        or np.any(sbp < 0)
    ):
        raise ValueError("Invalid nonnegative feature arrays")
    if family == "tx":
        return np.sqrt(tx)
    if family == "tx_sbp":
        return np.concatenate((np.sqrt(tx), np.log1p(sbp)), axis=1)
    raise ValueError("Unknown feature family")


def reference_representation(tx, sbp, segments, config):
    """Direct recurrence independent of production scipy.signal.lfilter."""
    values = transformed(tx, sbp, config["family"])
    output = np.empty_like(values)
    decay = np.exp(-0.02 / config["tau_seconds"])
    tau = config["baseline_tau_seconds"]
    slow_decay = None if tau is None else np.exp(-0.02 / tau)
    for start, end in segment_ranges(segments, len(values)):
        fast = values[start].copy()
        slow = values[start].copy()
        output[start] = fast if tau is None else 0.0
        for index in range(start + 1, end):
            fast = decay * fast + (1 - decay) * values[index]
            if slow_decay is not None:
                slow = slow_decay * slow + (1 - slow_decay) * values[index]
            output[index] = fast if tau is None else fast - slow
    return output


def reference_design(tx, sbp, segments, config):
    if config["kind"] == "v2_ridge":
        raw = transformed(tx, sbp, "tx")
        mean, scale = raw.mean(0), np.maximum(raw.std(0), 0.1)
        z = (raw - mean) / scale
        values = np.zeros_like(z)
        weights = np.exp(-np.arange(12) / 12)
        weights /= weights.sum()
        for start, end in segment_ranges(segments, len(z)):
            for lag, weight in enumerate(weights):
                if start + lag < end:
                    values[start + lag : end] += weight * z[start : end - lag]
    elif config["kind"] == "regularized":
        raw = reference_representation(tx, sbp, segments, config)
        floor = np.full(raw.shape[1], 0.1)
        if config["family"] == "tx_sbp":
            floor[tx.shape[1] :] = 0.01
        mean, scale = raw.mean(0), np.maximum(raw.std(0), floor)
        values = np.clip((raw - mean) / scale, -8.0, 8.0)
    else:
        raise ValueError("Unknown model kind")
    return values, mean, scale


def validate_fitted_model(arrays, tx, sbp, y, mask, segments, config):
    """Strictly convex ridge optimum verified without rerunning production fit."""
    expected_keys = {"format_version", "config", "mean", "scale", "coefficients", "intercept"}
    if (
        set(arrays) != expected_keys
        or arrays["format_version"].shape != ()
        or arrays["format_version"].item() != 3
    ):
        raise ValueError("Invalid model artifact fields or version")
    if json.loads(arrays["config"].item()) != config:
        raise ValueError("Model config differs from source-selected recipe")
    design, mean, scale = reference_design(tx, sbp, segments, config)
    d = design.shape[1]
    for key, shape in (
        ("mean", (d,)),
        ("scale", (d,)),
        ("coefficients", (d, 2)),
        ("intercept", (2,)),
    ):
        value = arrays[key]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("Model numeric geometry or dtype mismatch")
    if not np.allclose(arrays["mean"], mean, rtol=1e-10, atol=1e-10) or not np.allclose(
        arrays["scale"], scale, rtol=1e-10, atol=1e-10
    ):
        raise ValueError("Model normalization differs from exact calibration prefix")
    usable = training_mask(mask, segments)
    if usable.sum() < 50 or np.asarray(y).shape != (len(tx), 2) or not np.isfinite(y[usable]).all():
        raise ValueError("Insufficient or invalid training labels")
    x, y = design[usable], np.asarray(y, float)[usable]
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    penalty = 10.0 if config["kind"] == "v2_ridge" else config["ridge_lambda"] * len(x)
    w, b = arrays["coefficients"], arrays["intercept"]
    gradient = xc.T @ (xc @ w - yc) + penalty * w
    residual = float(np.linalg.norm(gradient) / max(1.0, np.linalg.norm(xc.T @ yc)))
    if residual > 1e-8 or not np.allclose(b, ym - xm @ w, rtol=1e-9, atol=1e-9):
        raise ValueError("Model is not the declared prefix-only ridge optimum")
    return {
        "normal_equation_relative_residual": residual,
        "n_training_labels": int(usable.sum()),
        "training_mask_sha256": mask_hash(usable),
        "sum_loss_ridge_penalty": float(penalty),
    }


def replay_prediction(arrays, representation, segments, config):
    """Apply exported parameters using independently constructed causal features."""
    z = (representation - arrays["mean"]) / arrays["scale"]
    if config["kind"] == "v2_ridge":
        values = np.zeros_like(z)
        weights = np.exp(-np.arange(12) / 12)
        weights /= weights.sum()
        for start, end in segment_ranges(segments, len(z)):
            for lag, weight in enumerate(weights):
                if start + lag < end:
                    values[start + lag : end] += weight * z[start : end - lag]
    else:
        values = np.clip(z, -8.0, 8.0)
    prediction = values @ arrays["coefficients"] + arrays["intercept"]
    if config["kind"] == "regularized":
        prediction /= np.maximum(1.0, np.linalg.norm(prediction, axis=1))[:, None]
    return prediction


def checked_reference(root, reference):
    path = contained(root, reference["path"])
    verify(path, reference["sha256"])
    return path


def verify_sources(sources, root):
    if len(sources) != 3 or {s["path"]: s["sha256"] for s in sources} != SOURCE_PINS:
        raise ValueError("Executed source differs from pre-assessment frozen implementation")
    for source in sources:
        verify(contained(root, source["snapshot_path"]), source["sha256"])


def index_metadata(path, role):
    path = Path(path).resolve()
    index = read(path)
    if (
        index["role"] != role
        or index["plan_sha256"] != PLAN_SHA256
        or index["v2_index_sha256"] != V2_INDEX_SHA256
    ):
        raise ValueError("Development/assessment index role or locked identities mismatch")
    if index["preparer"] != {
        "v3_sha256": SOURCE_PINS["neural_experiments/v3/data.py"],
        "v2_dependency_sha256": "e75e64c08a4e7041f086b52893dcd17212e4e211094b184abae6ea6d23998a11",
    }:
        raise ValueError("Prepared arrays use an unpinned importer implementation")
    old_path = Path(index["v2_index_path"])
    verify(old_path, V2_INDEX_SHA256)
    old = read(old_path)
    if index["participants"] != old["participants"] or index["provider"] != {
        "doi": "10.5061/dryad.x0k6djj1h",
        "version_id": 462702,
    }:
        raise ValueError("Index provider or participant cohort changed")
    old_rows = {row["session_id"]: row for row in old["sessions"]}
    expected = {}
    if role == "development":
        if index["selection"] != {"rule": "exact_frozen_V2_sessions_first_min600s"}:
            raise ValueError("Development selection is not the exact exposed cohort")
        expected = {sid: row["source"] for sid, row in old_rows.items()}
    else:
        if (
            index["selection"].get("regular_session_ranks_one_based") != [7, 8, 9]
            or index["selection"].get("scores_or_duration_used_for_selection") is not False
        ):
            raise ValueError("Assessment selection is not outcome-independent ranks7–9")
        refs = index["acquisition_sources"]
        if len(refs) != len(index["participants"]):
            raise ValueError("Missing assessment acquisition receipts")
        for person in index["participants"]:
            matches = [r for r in refs if r["path"] == f"acquisition/{person}.json"]
            if len(matches) != 1:
                raise ValueError("Duplicate or absent participant acquisition")
            receipt = read(checked_reference(path.parent, matches[0]))
            prior = sorted(
                [r["source"] for r in old_rows.values() if r["participant_id"] == person],
                key=lambda r: r["archive_ordinal"],
            )
            if (
                receipt["status"] != "complete"
                or receipt["participant_id"] != person
                or receipt["plan_sha256"] != PLAN_SHA256
                or receipt["v2_index_sha256"] != V2_INDEX_SHA256
            ):
                raise ValueError("Assessment receipt identity mismatch")
            if (
                receipt["archive"]["provider_sha256"] != prior[0]["archive_sha256_published"]
                or receipt["archive"]["file_id"] != prior[0]["archive_file_id"]
            ):
                raise ValueError("Assessment archive is not pinned V2 provider object")
            pattern = rf"(?:\./)?decoding/{person}/{person}_day_(\d+)_decoding\.mat"
            walked = receipt["seen_members"]
            if [r["ordinal"] for r in walked] != list(range(len(walked))):
                raise ValueError("Acquisition archive walk omitted entries")
            regular = [r for r in walked if r["regular_file"] and re.fullmatch(pattern, r["name"])]
            selected = receipt["selected"]
            verified = receipt["verified_prior_members"]
            if (
                len(verified) != 6
                or len(selected) != len(regular[6:9])
                or not 6 <= len(regular) <= 9
            ):
                raise ValueError("Acquisition omitted selected or prior members")
            if len(selected) < 3 and not receipt["tar_end_observed"]:
                raise ValueError("Shortage without observed archive end")
            if receipt["missing_requested_ranks"] != list(range(7 + len(selected), 10)):
                raise ValueError("Assessment shortage ledger mismatch")
            for rank, (member, item) in enumerate(
                zip(verified + selected, regular, strict=True), 1
            ):
                match = re.fullmatch(pattern, member["member_name"])
                if (
                    not match
                    or member["regular_session_rank"] != rank
                    or member["member_name"] != item["name"]
                    or member["member_size"] != item["size"]
                    or member["archive_ordinal"] != item["ordinal"]
                    or member["post_implant_day"] != int(match.group(1))
                ):
                    raise ValueError("Assessment member rank or provenance mismatch")
                if rank <= 6:
                    if any(
                        member[k] != prior[rank - 1][k]
                        for k in (
                            "session_id",
                            "member_name",
                            "member_size",
                            "member_sha256",
                            "archive_ordinal",
                        )
                    ):
                        raise ValueError("Authenticated prior member differs from V2")
                else:
                    if member["session_id"] in old_rows or member["session_id"] in expected:
                        raise ValueError("Assessment duplicates exposed or selected session")
                    verify(contained(path.parent, member["raw_path"]), member["member_sha256"])
                    expected[member["session_id"]] = member
    rows = {row["session_id"]: row for row in index["sessions"]}
    if len(rows) != len(index["sessions"]) or set(rows) != set(expected):
        raise ValueError("Index omitted or substituted selected sessions")
    for sid, row in rows.items():
        if (
            row["role"] != role
            or row["participant_id"] not in index["participants"]
            or any(row["source"].get(k) != v for k, v in expected[sid].items())
        ):
            raise ValueError("Prepared session role or source identity mismatch")
        if (
            row["session_id"] != row["source"]["session_id"]
            or row["participant_id"] != row["source"]["participant_id"]
            or row["post_implant_day"] != row["source"]["post_implant_day"]
        ):
            raise ValueError("Prepared session identity disagreement")
        if (
            not 0 <= row["n_bins"] <= 30000
            or row["n_bins"] != min(row["full_prepared_n_bins"], 30000)
            or row["bin_seconds"] != 0.02
        ):
            raise ValueError("Prepared export exceeds fixed600s cap or has wrong duration")
        segment_ranges(row["segments"], row["n_bins"])
        for name in ("feature", "target"):
            verify(contained(path.parent, row[f"{name}_path"]), row[f"{name}_sha256"])
    return index, rows


def labels_for(index_path, row):
    result = numeric(contained(Path(index_path).resolve().parent, row["target_path"]))
    if (
        set(result) != {"y", "eval_mask"}
        or result["y"].shape != (row["n_bins"], 2)
        or result["eval_mask"].shape != (row["n_bins"],)
        or result["eval_mask"].dtype != bool
    ):
        raise ValueError("Prepared labels have invalid geometry or mask")
    valid = result["y"][result["eval_mask"]]
    if not np.isfinite(valid).all() or not np.allclose(
        np.linalg.norm(valid, axis=1), 1, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("Scored labels are not finite unit directions")
    return result


def validate_feature_metadata(arrays, row):
    """Bind batch/index resets to the sequence IDs used by streaming replay."""
    n, channels = row["n_bins"], row["n_channels"]
    time, sequence, channel_ids = (
        arrays["timestamps"],
        arrays["sequence_id"],
        arrays["channel_ids"],
    )
    if (
        time.shape != (n,)
        or time.dtype != np.float64
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
        or sequence.shape != (n,)
        or not np.issubdtype(sequence.dtype, np.integer)
        or np.any(sequence < 0)
        or np.any(np.diff(sequence) < 0)
        or channel_ids.shape != (channels,)
        or not np.issubdtype(channel_ids.dtype, np.integer)
        or np.any(np.diff(channel_ids) <= 0)
    ):
        raise ValueError("Feature timestamps/sequence/channel metadata invalid")
    within = np.diff(sequence) == 0
    if not np.allclose(np.diff(time)[within], 0.02, rtol=0, atol=1e-9):
        raise ValueError("Feature timestamps omit an unmarked within-block gap")
    ranges = segment_ranges(row["segments"], n)
    starts = [a for a, _ in ranges]
    observed_starts = ([0] + (np.flatnonzero(np.diff(sequence) != 0) + 1).tolist()) if n else []
    if observed_starts != starts:
        raise ValueError("Feature sequence resets do not match index block boundaries")
    for segment in row["segments"]:
        if not np.all(sequence[segment["start"] : segment["end"]] == segment["sequence_id"]):
            raise ValueError("Feature sequence reset differs from index acquisition block")


def score_reference(y, prediction, mask):
    result = score_arrays(y, prediction, mask)
    if result["n"] < 50:
        limited = {
            "r2": None,
            "per_output_r2": [None, None],
            "n": result["n"],
            "angular_error_degrees": None,
            "zero_prediction_count": result["zero_prediction_count"],
        }
        return limited, limited.copy()
    actual, estimated = np.asarray(y, float)[mask], np.asarray(prediction, float)[mask]
    runner = {
        "r2": result["r2"],
        "per_output_r2": result["per_output_r2"],
        "n": result["n"],
        "sse": np.square(actual - estimated).sum(0).tolist(),
        "sst": np.square(actual - actual.mean(0)).sum(0).tolist(),
        "angular_error_degrees": result["angular_error_degrees"],
        "prediction_norm_rms": float(np.sqrt(np.square(estimated).sum(1).mean())),
        "zero_prediction_count": result["zero_prediction_count"],
    }
    return runner, runner.copy()


def reconstruct_choices(cases, candidates, plan):
    result = []
    for target, budget in itertools.product(
        plan["participants"], plan["calibration_budgets_seconds"]
    ):
        sources = [p for p in plan["participants"] if p != target]
        for method, family in (
            ("tx_daily", "tx"),
            ("tx_sbp_daily", "tx_sbp"),
            ("selected_daily", None),
        ):
            ranking = []
            for candidate in candidates:
                if family and candidate["config"]["family"] != family:
                    continue
                means = []
                for source in sources:
                    values = [
                        row["metrics"]["r2"]
                        for row in cases
                        if row["participant"] == source
                        and row["budget_seconds"] == budget
                        and row["candidate_id"] == candidate["candidate_id"]
                        and row["metrics"]["r2"] is not None
                    ]
                    if values:
                        means.append(sum(values) / len(values))
                if not means:
                    raise ValueError("No eligible source people for candidate ranking")
                ranking.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "objective": sum(means) / len(means),
                        "eligible_source_people": len(means),
                    }
                )
            best_index = int(np.argmax([r["objective"] for r in ranking]))
            result.append(
                {
                    "target_participant": target,
                    "budget_seconds": budget,
                    "method": method,
                    "source_participants": sources,
                    **ranking[best_index],
                    "ranking": ranking,
                }
            )
    return result


def audit_selection(selection_path, development_index, root=ROOT):
    selection_path, development_index = (
        Path(selection_path).resolve(),
        Path(development_index).resolve(),
    )
    selected = read(selection_path)
    plan_path = checked_reference(root, selected["plan"])
    verify(plan_path, PLAN_SHA256)
    plan = read(plan_path)
    if (
        checked_reference(root, selected["development_index"]) != development_index
        or selected["assessment_labels_opened"] is not False
    ):
        raise ValueError("Source selection input or exposure boundary mismatch")
    verify_sources(selected["source_files"], root)
    index, rows = index_metadata(development_index, "development")
    expected_candidates = [
        {"candidate_id": f"c{i:02d}", "config": config} for i, config in enumerate(grid(plan))
    ]
    if selected["candidates"] != expected_candidates:
        raise ValueError("Source grid order/settings differ from locked plan")
    candidate_ids = [c["candidate_id"] for c in expected_candidates]
    ledgers = {}
    for row in selected["development_cases"]:
        key = row["session_id"], row["budget_seconds"], row["candidate_id"]
        if key in ledgers:
            raise ValueError("Duplicate source candidate case")
        ledgers[key] = row
    unavailable = {
        (r["session_id"], r["budget_seconds"]): r["reason"]
        for r in selected["development_unavailable"]
    }
    if len(unavailable) != len(selected["development_unavailable"]):
        raise ValueError("Duplicate unavailable development case")
    expected_cases, expected_unavailable, scored = set(), {}, []
    for sid, item in rows.items():
        labels = labels_for(development_index, item)
        n = item["n_bins"]
        start, stop = 15100, n
        segments = segment_ranges(item["segments"], n)
        for budget in plan["calibration_budgets_seconds"]:
            prefix_stop = round(budget / 0.02)
            reason = None
            if stop <= start or labels["eval_mask"][start:stop].sum() < 50:
                reason = "no development suffix"
            elif prefix_stop > n:
                reason = "short prefix"
            else:
                usable = training_mask(
                    labels["eval_mask"][:prefix_stop], clipped_segments(segments, prefix_stop)
                )
                if usable.sum() < 50:
                    reason = "too few training labels"
            if reason:
                expected_unavailable[sid, budget] = reason
                continue
            keys = [(sid, budget, c) for c in candidate_ids]
            if any(key not in ledgers for key in keys):
                raise ValueError("Missing feasible source-grid candidate")
            first = ledgers[keys[0]]
            path = contained(root, first["trace_path"])
            verify(path, first["trace_sha256"])
            arrays = numeric(path)
            if (
                set(arrays) != {"predictions", "candidate_ids", "start_bin", "stop_bin"}
                or arrays["candidate_ids"].tolist() != candidate_ids
                or arrays["start_bin"].item() != start
                or arrays["stop_bin"].item() != stop
            ):
                raise ValueError("Source-grid trace inventory or interval changed")
            predictions = arrays["predictions"]
            if (
                predictions.shape != (36, stop - start, 2)
                or predictions.dtype != np.float32
                or not np.isfinite(predictions).all()
            ):
                raise ValueError("Source-grid prediction geometry/dtype invalid")
            mask = labels["eval_mask"][start:stop]
            for i, key in enumerate(keys):
                row = ledgers[key]
                expected_cases.add(key)
                if (
                    row["participant"] != item["participant_id"]
                    or row["candidate_index"] != i
                    or row["trace_path"] != first["trace_path"]
                    or row["trace_sha256"] != first["trace_sha256"]
                    or (row["score_start_bin"], row["score_stop_bin"]) != (start, stop)
                ):
                    raise ValueError("Source case participant/grid/trace interval mismatch")
                if (
                    row["feature_sha256"] != item["feature_sha256"]
                    or row["target_sha256"] != item["target_sha256"]
                    or row["mask_sha256"] != mask_hash(mask)
                ):
                    raise ValueError("Source case input or score-mask identity mismatch")
                if (
                    row["n_prefix_bins"] != prefix_stop
                    or row["n_available_labels"] != int(labels["eval_mask"][:prefix_stop].sum())
                    or row["n_training_labels"] != int(usable.sum())
                ):
                    raise ValueError("Source calibration labels do not match fixed prefix")
                penalty = expected_candidates[i]["config"]["ridge_lambda"] * int(usable.sum())
                if row["sum_loss_ridge_penalty"] != penalty:
                    raise ValueError("Source mean-loss ridge convention changed")
                metrics, independent = score_reference(
                    labels["y"][start:stop], predictions[i], mask
                )
                compare_structure(row["metrics"], metrics, "source.metrics")
                scored.append({**row, "metrics": metrics, "independent_metrics": independent})
    if set(ledgers) != expected_cases or unavailable != expected_unavailable:
        raise ValueError("Source grid inventory has unexplained or omitted cases")
    reconstructed = reconstruct_choices(scored, expected_candidates, plan)
    compare_structure(selected["choices"], reconstructed, "source.choices")
    return {
        "status": "passed",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_sha256": sha256(selection_path),
        "development_index_sha256": sha256(development_index),
        "plan_sha256": PLAN_SHA256,
        "auditor_source_sha256": sha256(__file__),
        "source_prediction_cases_checked": len(scored),
        "source_unavailable_cases_checked": len(unavailable),
        "outer_choices_checked": len(reconstructed),
        "assessment_labels_opened": False,
        "independent_choices": reconstructed,
        "scope": (
            "Independent saved source-grid score reconstruction and whole-person-excluded "
            "ranking; source fits are not retrained. Fresh assessment labels are not opened."
        ),
    }


def bootstrap(values, plan):
    values = np.asarray(values, float)
    if not len(values):
        return [None, None]
    cfg = plan["uncertainty"]
    rng = np.random.default_rng(cfg["seed"])
    sampled = rng.choice(values, size=(cfg["bootstrap_draws"], len(values)), replace=True)
    return np.quantile(sampled.mean(1), [0.025, 0.975]).tolist()


def reconstruct_summary(models, plan):
    groups = []
    for budget, method, partition in itertools.product(
        plan["calibration_budgets_seconds"],
        plan["assessment"]["methods"],
        ("primary", "common_suffix"),
    ):
        people = []
        for person in plan["participants"]:
            rows = [
                trace["metrics"]
                for model in models
                if (model["target_participant"], model["method"], model["budget_seconds"])
                == (person, method, budget)
                for trace in model["traces"]
                if trace["partition"] == partition and trace["metrics"]["r2"] is not None
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
                    "method": method,
                    "partition": partition,
                    "people": people,
                    "mean_r2": float(np.mean(values)),
                    "r2_bootstrap_95": bootstrap(values, plan),
                    "positive_people": sum(v > 0 for v in values),
                    "mean_angular_error_degrees": float(
                        np.mean([p["angular_error_degrees"] for p in people])
                    ),
                }
            )
    lookup = {(g["method"], g["budget_seconds"], g["partition"]): g for g in groups}
    primary = lookup.get(("selected_daily", 120, "primary"))
    comparisons = []
    for reference in ("zero", "prefix_mean", "v2_ridge_daily"):
        baseline = lookup.get((reference, 120, "primary"))
        if not primary or not baseline:
            continue
        controls = {p["participant"]: p["r2"] for p in baseline["people"]}
        effects = [
            {"participant": p["participant"], "difference": p["r2"] - controls[p["participant"]]}
            for p in primary["people"]
            if p["participant"] in controls
        ]
        values = [e["difference"] for e in effects]
        comparisons.append(
            {
                "reference": reference,
                "effects": effects,
                "mean_difference": float(np.mean(values)),
                "bootstrap_95": bootstrap(values, plan),
            }
        )
    gate = plan["operational_usefulness_gate"]
    criteria = {
        "nine_eligible_people": bool(
            primary and len(primary["people"]) == gate["eligible_people_required"]
        ),
        "mean_r2_at_least_0_1": bool(primary and primary["mean_r2"] >= gate["mean_r2_at_least"]),
        "r2_interval_above_zero": bool(
            primary and primary["r2_bootstrap_95"][0] > gate["absolute_r2_bootstrap_lower_above"]
        ),
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
                if c["reference"] in ("zero", "prefix_mean")
                and c["mean_difference"] >= gate["mean_gain_over_zero_and_prefix_mean_at_least"]
            ]
        )
        == 2,
    }
    return {
        "groups": groups,
        "primary_comparisons": comparisons,
        "operational_gate": {"criteria": criteria, "passed": all(criteria.values())},
    }


def audit_run(run, development_index, assessment_index, root=ROOT):
    run = Path(run).resolve()
    manifest = read(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only a complete prospective assessment can pass")
    plan_path = checked_reference(root, manifest["plan"])
    verify(plan_path, PLAN_SHA256)
    plan = read(plan_path)
    verify_sources(manifest["source_files"], root)
    if (
        checked_reference(root, manifest["development_index"]) != Path(development_index).resolve()
        or checked_reference(root, manifest["assessment_index"]) != Path(assessment_index).resolve()
    ):
        raise ValueError("Run input indexes differ from audited inputs")
    selection_path = checked_reference(root, manifest["selection"])
    selection = read(selection_path)
    if selection["source_files"] != manifest["source_files"]:
        raise ValueError("Source changed between selection and assessment")
    source_audit_path = checked_reference(root, manifest["selection_audit"])
    source_receipt = read(source_audit_path)
    if (
        source_receipt["status"] != "passed"
        or source_receipt["selection_sha256"] != sha256(selection_path)
        or source_receipt["assessment_labels_opened"] is not False
        or source_receipt["development_index_sha256"] != sha256(development_index)
    ):
        raise ValueError("Assessment has no exact passed source-selection gate")
    if (
        not selection["created_at"]
        <= source_receipt["created_at"]
        <= manifest["assessment_started_at"]
        <= manifest["assessment_completed_at"]
    ):
        raise ValueError("Source audit did not precede assessment-label exposure")
    source_check = audit_selection(selection_path, development_index, root)
    development, dev_rows = index_metadata(development_index, "development")
    assessment, new_rows = index_metadata(assessment_index, "assessment")
    if set(dev_rows) & set(new_rows):
        raise ValueError("Assessment contains an exposed development session")
    labels = {
        ("development", sid): labels_for(development_index, row) for sid, row in dev_rows.items()
    }
    labels.update(
        {("assessment", sid): labels_for(assessment_index, row) for sid, row in new_rows.items()}
    )

    @lru_cache(maxsize=2)
    def features(role, sid):
        path = development_index if role == "development" else assessment_index
        row = (dev_rows if role == "development" else new_rows)[sid]
        arrays = numeric(contained(Path(path).resolve().parent, row["feature_path"]))
        if set(arrays) != {"x_tx", "x_sbp", "timestamps", "sequence_id", "channel_ids"}:
            raise ValueError("Unexpected feature file fields")
        tx, sbp = arrays["x_tx"], arrays["x_sbp"]
        if (
            tx.shape != (row["n_bins"], row["n_channels"])
            or sbp.shape != tx.shape
            or tx.dtype != np.float32
            or sbp.dtype != np.float32
            or not np.isfinite(tx).all()
            or not np.isfinite(sbp).all()
            or np.any(tx < 0)
            or np.any(sbp < 0)
        ):
            raise ValueError("Feature arrays have invalid counts/power geometry")
        validate_feature_metadata(arrays, row)
        return tx, sbp

    choices = {
        (r["target_participant"], r["budget_seconds"], r["method"]): r
        for r in source_check["independent_choices"]
    }
    configs = {r["candidate_id"]: r["config"] for r in selection["candidates"]}
    identities, independent_scores, optimality, replay_checks = [], [], [], []
    completed, unavailable = {}, {}
    for ref in manifest["models"]:
        identity = read(
            checked_reference(root, {"path": ref["identity_path"], "sha256": ref["sha256"]})
        )
        key = identity["session_id"], identity["method"], identity["budget_seconds"]
        if key in completed:
            raise ValueError("Duplicate assessment model case")
        completed[key] = identity
    for row in manifest["unavailable"]:
        key = row["session_id"], row["method"], row["budget_seconds"]
        if key in unavailable or key in completed:
            raise ValueError("Duplicate or completed unavailable case")
        unavailable[key] = row
    expected_keys = set()
    for sid, item in new_rows.items():
        # Bound memory to the representations for this one assessment session.
        representations = {}
        person = item["participant_id"]
        earliest = min(
            [r for r in dev_rows.values() if r["participant_id"] == person],
            key=lambda r: (r["post_implant_day"], r["session_id"]),
        )
        for method, budget in itertools.product(
            plan["assessment"]["methods"], plan["calibration_budgets_seconds"]
        ):
            key = sid, method, budget
            expected_keys.add(key)
            role = "development" if method == "stale_mapping" else "assessment"
            cal_row = earliest if role == "development" else item
            cal_index = Path(
                development_index if role == "development" else assessment_index
            ).resolve()
            cal_labels = labels[role, cal_row["session_id"]]
            stop = round(budget / 0.02)
            segments = segment_ranges(cal_row["segments"], cal_row["n_bins"])
            reason = None
            if stop > cal_row["n_bins"]:
                reason = "calibration budget exceeds recording"
            else:
                prefix_segments = clipped_segments(segments, stop)
                usable = training_mask(cal_labels["eval_mask"][:stop], prefix_segments)
                if usable.sum() < 50:
                    reason = "fewer than 50 training labels after burn"
            if reason:
                expected = {
                    "target_participant": person,
                    "session_id": sid,
                    "method": method,
                    "budget_seconds": budget,
                    "reason": reason,
                }
                if unavailable.get(key) != expected:
                    raise ValueError(
                        "Unavailable assessment reason/inventory differs from actual prefix"
                    )
                continue
            if key not in completed:
                raise ValueError("Feasible assessment case omitted")
            identity = completed[key]
            if identity["target_participant"] != person:
                raise ValueError("Assessment model belongs to wrong participant")
            source_choice = None
            if method in ("zero", "prefix_mean"):
                expected_config = {"kind": "constant", "rule": method}
            elif method == "v2_ridge_daily":
                expected_config = {"kind": "v2_ridge", "family": "tx"}
            else:
                choice_method = "selected_daily" if method == "stale_mapping" else method
                source_choice = choices[person, budget, choice_method]
                expected_config = configs[source_choice["candidate_id"]]
            if identity["config"] != expected_config:
                raise ValueError("Assessment model uses unselected or changed configuration")
            compare_structure(identity["source_choice"], source_choice, "model.source_choice")
            expected_calibration = {
                "role": role,
                "session_id": cal_row["session_id"],
                "participant_id": person,
                "start_bin": 0,
                "stop_bin": stop,
                "feature_sha256": cal_row["feature_sha256"],
                "target_sha256": cal_row["target_sha256"],
                "mask_sha256": mask_hash(usable),
                "source_index_path": str(cal_index.relative_to(Path(root).resolve())),
                "source_index_sha256": sha256(cal_index),
                "n_prefix_bins": stop,
                "n_available_labels": int(cal_labels["eval_mask"][:stop].sum()),
                "n_training_labels": int(usable.sum()),
            }
            model_path = contained(root, identity["model_path"])
            verify(model_path, identity["model_sha256"])
            arrays = numeric(model_path)
            if method in ("zero", "prefix_mean"):
                value = (
                    np.zeros(2)
                    if method == "zero"
                    else np.asarray(cal_labels["y"][:stop], float)[usable].mean(0)
                )
                if (
                    set(arrays) != {"constant"}
                    or arrays["constant"].dtype != np.float64
                    or arrays["constant"].shape != (2,)
                    or not np.array_equal(arrays["constant"], value)
                ):
                    raise ValueError("Constant baseline differs from declared calibration labels")
            else:
                tx, sbp = features(role, cal_row["session_id"])
                result = validate_fitted_model(
                    arrays,
                    tx[:stop],
                    sbp[:stop],
                    cal_labels["y"][:stop],
                    cal_labels["eval_mask"][:stop],
                    prefix_segments,
                    expected_config,
                )
                expected_calibration["sum_loss_ridge_penalty"] = result["sum_loss_ridge_penalty"]
                optimality.append(
                    {"session_id": sid, "method": method, "budget_seconds": budget, **result}
                )
                representation_key = tuple(
                    expected_config.get(k)
                    for k in ("kind", "family", "tau_seconds", "baseline_tau_seconds")
                )
                if representation_key not in representations:
                    target_tx, target_sbp = features("assessment", sid)
                    target_segments = segment_ranges(item["segments"], item["n_bins"])
                    representations[representation_key] = (
                        transformed(target_tx, target_sbp, "tx")
                        if expected_config["kind"] == "v2_ridge"
                        else reference_representation(
                            target_tx, target_sbp, target_segments, expected_config
                        )
                    )
                replay = replay_prediction(
                    arrays, representations[representation_key], item["segments"], expected_config
                )
            compare_structure(identity["calibration"], expected_calibration, "model.calibration")
            if (
                identity["assessment_day_minus_calibration_day"]
                != item["post_implant_day"] - cal_row["post_implant_day"]
            ):
                raise ValueError("Reported assessment/calibration chronology is wrong")
            target = labels["assessment", sid]
            if len(identity["traces"]) != 2 or {t["partition"] for t in identity["traces"]} != {
                "primary",
                "common_suffix",
            }:
                raise ValueError("Missing primary or matched common-suffix trace")
            for trace in identity["traces"]:
                requested = round((budget + 2) / 0.02) if trace["partition"] == "primary" else 15100
                start, end = min(requested, item["n_bins"]), item["n_bins"]
                mask = target["eval_mask"].copy()
                mask[:start] = False
                if (
                    (trace["start_bin"], trace["stop_bin"]) != (start, end)
                    or trace["mask_sha256"] != mask_hash(mask)
                    or trace["feature_sha256"] != item["feature_sha256"]
                    or trace["target_sha256"] != item["target_sha256"]
                ):
                    raise ValueError("Assessment interval/mask/input differs across methods")
                predictions = numeric(checked_reference(root, trace))
                if (
                    set(predictions) != {"prediction"}
                    or predictions["prediction"].dtype != np.float64
                ):
                    raise ValueError("Prediction artifact contains undeclared fields or dtype")
                prediction = predictions["prediction"]
                if method not in ("zero", "prefix_mean"):
                    if prediction.shape != replay.shape or not np.allclose(
                        prediction, replay, rtol=1e-9, atol=1e-9
                    ):
                        raise ValueError("Saved predictions do not match independent model replay")
                    if trace["partition"] == "primary":
                        replay_checks.append(
                            {
                                "session_id": sid,
                                "method": method,
                                "budget_seconds": budget,
                                "max_absolute_error": float(np.max(np.abs(prediction - replay))),
                            }
                        )
                metrics, detailed = score_reference(target["y"], prediction, mask)
                compare_structure(trace["metrics"], metrics, "assessment.metrics")
                if method in ("zero", "prefix_mean") and not np.array_equal(
                    prediction, np.broadcast_to(value, (end, 2))
                ):
                    raise ValueError("Constant-model predictions differ from exported constant")
                trace["metrics"] = metrics
                independent_scores.append(
                    {
                        "target_participant": person,
                        "session_id": sid,
                        "method": method,
                        "budget_seconds": budget,
                        "partition": trace["partition"],
                        "metrics": detailed,
                    }
                )
            identities.append(identity)
    if set(completed) | set(unavailable) != expected_keys:
        raise ValueError("Assessment contains unregistered or missing cases")
    summary_path = checked_reference(root, manifest["summary"])
    expected_summary = {"status": "complete", **reconstruct_summary(identities, plan)}
    compare_structure(read(summary_path), expected_summary, "assessment.summary")
    return {
        "status": "passed",
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": sha256(run / "manifest.json"),
        "summary_sha256": sha256(summary_path),
        "selection_sha256": sha256(selection_path),
        "plan_sha256": PLAN_SHA256,
        "development_index_sha256": sha256(development_index),
        "assessment_index_sha256": sha256(assessment_index),
        "auditor_source_sha256": sha256(__file__),
        "models_checked": len(identities),
        "traces_checked": len(independent_scores),
        "unavailable_models": len(unavailable),
        "source_prediction_cases_checked": source_check["source_prediction_cases_checked"],
        "independent_summary": expected_summary,
        "independent_scores": independent_scores,
        "ridge_optimality": optimality,
        "nonconstant_models_replayed": len(replay_checks),
        "prediction_replay": replay_checks,
        "prediction_replay_max_absolute_error": max(
            (r["max_absolute_error"] for r in replay_checks), default=0.0
        ),
        "scope": (
            "Independent source-grid and assessment scoring, whole-person source selection, "
            "calibration normalization and ridge convex optimality, artifact/input/inventory "
            "verification, and independent full-session assessment decoder replay. Source-grid "
            "fits are not retrained; upstream feature causality remains unverified."
        ),
    }


def main():
    import argparse

    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selection", type=Path)
    mode.add_argument("--run", type=Path)
    parser.add_argument("--development-index", type=Path, required=True)
    parser.add_argument("--assessment-index", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output must be fresh")
    with threadpool_limits(limits=1):
        if args.selection:
            if args.assessment_index:
                parser.error("Source-only audit must not receive an assessment index")
            result = audit_selection(args.selection, args.development_index, args.root)
        else:
            if not args.assessment_index:
                parser.error("Complete run audit requires --assessment-index")
            result = audit_run(args.run, args.development_index, args.assessment_index, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "independent_choices",
                    "independent_summary",
                    "independent_scores",
                    "ridge_optimality",
                    "prediction_replay",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
