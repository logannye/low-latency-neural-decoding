"""Frozen assessment release, with metadata-only validation before NWB access.

A hash establishes byte identity, not authorization. The caller must receive the
release decision and its expected freeze digest independently before export.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    UPSTREAM_PIN,
    _metadata,
    _official_loader,
    bin_spikes,
    load_feature_file,
    load_target_file,
    session_mapping,
    sha256_file,
)
from .splits import day_number

PRIMARY_METRICS = {
    "decoding": "mean_of_day_variance_weighted_r2",
    "runtime": "p99_complete_step_service_ms",
}
STAGES = {
    "early": {
        "training_days": [f"S{i}" for i in range(5)],
        "evaluation_days": ["S5"],
        "provider_split": "held-in-calib",
    },
    "later": {
        "training_days": [f"S{i}" for i in range(6)],
        "evaluation_days": [f"S{i}" for i in range(6, 13)],
        "provider_split": "held-out-calib",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    obj = json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 identity")
    return value


def _artifact(root: Path, record: dict[str, Any]) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Artifact path must remain inside the repository")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact symlink escapes repository")
    if not path.is_file() or sha256_file(path) != _digest(record["sha256"]):
        raise ValueError(f"Frozen artifact identity changed: {relative}")
    return path


def _verify_git_source(root: Path, revision: str, hashes: dict[str, str], production: set[str]):
    try:
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=root, stderr=subprocess.STDOUT
        )
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision, "--", "src/neural_decoder"],
            cwd=root,
            text=True,
        ).splitlines()
        if {name for name in names if name.endswith(".py")} != production:
            raise ValueError("Current decoder file inventory differs from frozen commit")
        for relative, digest in hashes.items():
            content = subprocess.check_output(
                ["git", "show", f"{revision}:{relative}"], cwd=root, stderr=subprocess.STDOUT
            )
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Frozen source digest differs from committed source: {relative}")
    except subprocess.CalledProcessError as error:
        raise ValueError("Frozen source revision or committed file is unavailable") from error


def select_assessment_assets(cache: str | Path, stage: str) -> list[dict[str, Any]]:
    """Read public metadata only; audit UUID, mapped day, provider role and shape."""
    if stage not in STAGES:
        raise ValueError("Unknown assessment stage")
    cache = Path(cache)
    manifest = _read_json(cache / "manifest.json")
    schema = _read_json(cache / "schema_audit.json")
    mapping = session_mapping(cache / "upstream")
    if manifest.get("dandiset") != "000954" or manifest.get("version") != "draft":
        raise ValueError("Only pinned FALCON H1 assets are permitted")
    if schema.get("source_manifest_sha256") != sha256_file(cache / "manifest.json"):
        raise ValueError("Schema audit does not identify the selected source manifest")
    rows = {row["asset_id"]: row for row in schema["files"]}
    if len(rows) != len(schema["files"]):
        raise ValueError("Duplicate schema asset identity")
    rule = STAGES[stage]
    selected, seen_assets, seen_streams = [], set(), set()
    for asset in manifest["assets"]:
        asset_id = asset["asset_id"]
        if asset_id in seen_assets:
            raise ValueError("Duplicate source asset UUID")
        seen_assets.add(asset_id)
        filename = Path(asset["path"]).stem
        stamp = filename.split("_ses-")[-1]
        if stamp not in mapping:
            raise ValueError("Source asset lacks pinned session mapping")
        file_id = mapping[stamp]
        day = file_id.split("_")[0]
        provider = filename.split("sub-HumanPitt-")[-1].split("_ses-")[0]
        if day not in rule["evaluation_days"] or provider != rule["provider_split"]:
            continue
        if file_id in seen_streams:
            raise ValueError("Duplicate assessment stream")
        seen_streams.add(file_id)
        row = rows[asset_id]
        expected = {
            "file_id": file_id,
            "day": day,
            "provider_split": provider,
            "raw_sha256": asset["sha256"],
            "path": asset["path"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Schema role/identity differs from pinned provider mapping")
        if asset["sha256"] != asset["provider_digest"]["dandi:sha2-256"]:
            raise ValueError("Provider and local raw identities disagree")
        relative = Path(asset["local_path"])
        if relative != Path("raw") / asset["path"] or ".." in relative.parts:
            raise ValueError("Raw asset location disagrees with provider path")
        if (row["n_channels"], row["n_outputs"]) != (176, 7):
            raise ValueError("Assessment schema differs from H1 shape")
        selected.append({**row, "local_path": str(relative), "size": asset["size"]})
    expected_streams = {f"{day}_set_{i}" for day in rule["evaluation_days"] for i in (1, 2)}
    if seen_streams != expected_streams:
        raise ValueError("Assessment must contain exactly the expected provider streams")
    return sorted(selected, key=lambda row: (day_number(row["day"]), row["file_id"]))


def _validate_freeze(path, expected_sha256, root, cache):
    if sha256_file(path) != _digest(expected_sha256):
        raise ValueError("Freeze bytes differ from the independently supplied digest")
    freeze = _read_json(path)
    if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen":
        raise ValueError("Assessment requires a completed schema-v1 freeze")
    stage = freeze.get("stage")
    if stage not in STAGES:
        raise ValueError("Unknown assessment stage")
    rules = STAGES[stage]
    for key in ("evaluation_days", "provider_split"):
        if freeze.get(key) != rules[key]:
            raise ValueError(f"Frozen {key} disagree with stage roles")
    if dt.datetime.fromisoformat(freeze["created_at"]).tzinfo is None:
        raise ValueError("Freeze creation time needs an explicit timezone")
    if freeze.get("primary_metrics") != PRIMARY_METRICS:
        raise ValueError("Primary metrics differ from the declared assessment contract")
    source = freeze["source"]
    revision = source["git_revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Freeze must identify a complete committed source revision")
    hashes = source["files_sha256"]
    production = {str(p.relative_to(root)) for p in (root / "src/neural_decoder").rglob("*.py")}
    if not production or not production.issubset(hashes):
        raise ValueError("Freeze source identities omit decoder implementation files")
    for relative, digest in hashes.items():
        _artifact(root, {"path": relative, "sha256": digest})
    _verify_git_source(root, revision, hashes, production)
    _artifact(root, freeze["recipe"])
    models = freeze["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("Freeze requires fitted model artifact identities")
    names, paths = set(), set()
    for model in models:
        if not isinstance(model.get("name"), str) or not model["name"]:
            raise ValueError("Model requires a nonempty name")
        if model["name"] in names or model["path"] in paths:
            raise ValueError("Duplicate frozen model identity")
        names.add(model["name"])
        paths.add(model["path"])
        if model.get("training_days") != rules["training_days"]:
            raise ValueError("Model training days violate frozen assessment roles")
        if model.get("regime") not in {"frozen_zero_calibration", "label_free_online_adaptation"}:
            raise ValueError("Primary models cannot use new-day supervised calibration")
        _artifact(root, model)
    identities = freeze["data"]
    if identities.get("upstream_pin") != UPSTREAM_PIN:
        raise ValueError("Freeze changed the upstream task definition")
    for filename, key in (
        ("manifest.json", "manifest_sha256"),
        ("schema_audit.json", "schema_audit_sha256"),
        ("split_manifest.json", "split_manifest_sha256"),
    ):
        if sha256_file(cache / filename) != _digest(identities[key]):
            raise ValueError(f"Frozen data metadata identity changed: {filename}")
    select_assessment_assets(cache, stage)
    if stage == "early":
        if "parent_freeze" in freeze:
            raise ValueError("Early assessment must not silently inherit another freeze")
    else:
        parent_record = freeze["parent_freeze"]
        parent_path = _artifact(root, parent_record)
        if _read_json(parent_path).get("stage") != "early":
            raise ValueError("Later assessment must descend from an early freeze")
        parent = _validate_freeze(parent_path, parent_record["sha256"], root, cache)
        for key in ("recipe", "primary_metrics", "data"):
            if freeze[key] != parent[key]:
                raise ValueError(f"Later assessment changed the frozen {key}")
        if source["files_sha256"] != parent["source"]["files_sha256"]:
            raise ValueError("Decoder source changed between early and later assessment")
        if dt.datetime.fromisoformat(freeze["created_at"]) < dt.datetime.fromisoformat(
            parent["created_at"]
        ):
            raise ValueError("Later freeze predates its parent")
    return freeze


def validate_freeze(freeze_path, expected_sha256, *, root, cache) -> dict[str, Any]:
    """Metadata/artifact validation only. expected_sha256 must come from the release decision."""
    try:
        return _validate_freeze(
            Path(freeze_path), expected_sha256, Path(root).resolve(), Path(cache)
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Incomplete or malformed freeze: {error}") from error


def calibration_feasibility(cache: str | Path) -> dict[str, Any]:
    """Optional 0/5/15/30s label budgets, assessed solely from existing metadata."""
    files = []
    for row in select_assessment_assets(cache, "later"):
        n = row["n_bins"]
        if row["bin_size_s"] != 0.02 or n <= 1500:
            raise ValueError("Every file needs a nonempty common suffix after 30s")
        files.append(
            {
                "file_id": row["file_id"],
                "day": row["day"],
                "n_bins": n,
                "prefix_budget_bins": [0, 250, 750, 1500],
                "prefix_budget_nominal_seconds": [0, 5, 15, 30],
                "common_score_interval_bins_half_open": [1500, n],
                "common_suffix_nominal_seconds": round((n - 1500) * 0.02, 2),
                "actual_valid_prefix_labels": "unknown until authorized scorer inspection",
            }
        )
    return {
        "status": "metadata_feasible_not_executed",
        "files": files,
        "nominal_label_seconds_per_day": [0, 10, 30, 60],
        "prefix_state_rule": "Every arm ingests all preceding neural bins; reset per file",
        "label_rule": "Separate calibration job uses only its own preceding file prefix",
        "scoring_rule": "Every budget scores the same suffix and official mask",
        "primary_assessment": "Separate, with no new-day supervised labels",
    }


def _read_arrays(raw: Path, expected: dict[str, Any], official):
    import h5py

    with h5py.File(raw, "r") as file:
        metadata = _metadata(file, expected)
        if metadata["n_bins"] != expected["n_bins"]:
            raise ValueError("Actual NWB duration differs from frozen metadata")
        ids = np.array(metadata["channel_ids"], dtype=np.int64)
        if not np.array_equal(ids, np.arange(176)):
            raise ValueError("Actual channel order differs from frozen H1 contract")
        n = metadata["n_bins"]
        timestamps = metadata["nwb_data_offset"] + np.arange(n) * metadata["nwb_rate_field"]
        flat = file["units/spike_times"][:]
        ends = file["units/spike_times_index"][:]
        starts = np.r_[0, ends[:-1]]
        counts = bin_spikes(
            [flat[int(a) : int(b)] for a, b in zip(starts, ends, strict=True)], timestamps
        )
        labels = file["acquisition/OpenLoopKinematicsVelocity/data"][:]
        mask = file[f"acquisition/{metadata['mask_name']}/data"][:].astype(bool)
        if metadata["mask_name"] == "Blacklist":
            mask = ~mask
        trials = np.r_[False, np.diff(file["acquisition/TrialNum/data"][:]) > 0]
        if not np.isfinite(labels).all() or mask.shape != (n,) or trials.shape != (n,):
            raise ValueError("Invalid assessment targets/masks/trials")
    for actual, reference in zip((counts, labels, trials, mask), official(raw), strict=True):
        if not np.array_equal(actual, reference):
            raise ValueError("Assessment arrays differ from pinned official loader")
    return (
        {
            "counts": counts,
            "timestamps": timestamps,
            "sequence_id": np.arange(n),
            "channel_ids": ids,
        },
        {"targets": labels, "eval_mask": mask, "trial_change": trials},
        {
            "count_max": int(counts.max()),
            "scored_bins": int(mask.sum()),
            "official_loader_exact_array_parity": True,
        },
    )


def export_assessment(freeze_path, expected_sha256, *, root, cache) -> Path:
    """Authorized array release into a fresh local cache; never overwrite old evidence."""
    root, cache = Path(root).resolve(), Path(cache)
    freeze = validate_freeze(freeze_path, expected_sha256, root=root, cache=cache)
    rows = select_assessment_assets(cache, freeze["stage"])
    destination = cache / "prepared_assessment" / f"{freeze['stage']}-{expected_sha256[:16]}"
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Assessment output must be local to the current repository")
    for path in [destination, *destination.parents]:
        if path == root:
            break
        if path.is_symlink():
            raise ValueError("Assessment output path cannot use a symlink")
    if destination.exists():
        raise FileExistsError("Assessment export already exists; preserve its evidence")
    for row in rows:
        raw = cache / row["local_path"]
        if raw.stat().st_size != row["size"] or sha256_file(raw) != row["raw_sha256"]:
            raise ValueError("Raw assessment asset differs from pinned identity")
    destination.mkdir(parents=True)
    records = []
    try:
        official = _official_loader(cache / "upstream")
        for row in rows:
            features, targets, diagnostics = _read_arrays(cache / row["local_path"], row, official)
            record = {
                key: row[key]
                for key in (
                    "file_id",
                    "day",
                    "provider_split",
                    "asset_id",
                    "raw_sha256",
                    "n_bins",
                    "n_channels",
                    "n_outputs",
                    "bin_size_s",
                    "channel_ids",
                )
            }
            record.update(diagnostics)
            for kind, arrays in (("features", features), ("targets", targets)):
                path = destination / kind / f"{row['file_id']}.npz"
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, **arrays)
                record[kind] = str(path.relative_to(destination))
                record[f"{kind}_sha256"] = sha256_file(path)
            records.append(record)
        validate_freeze(freeze_path, expected_sha256, root=root, cache=cache)
        index = {
            "schema_version": 1,
            "status": "complete",
            "role": "sealed_local_assessment",
            "stage": freeze["stage"],
            "freeze_sha256": expected_sha256,
            "source_manifest_sha256": freeze["data"]["manifest_sha256"],
            "upstream_pin": UPSTREAM_PIN,
            "files": records,
            "label_access": "scorer only; S5 may later enter fixed-recipe S0-S5 refit",
        }
        path = destination / "index.json"
        path.write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
        (destination / "receipt.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "freeze_sha256": expected_sha256,
                    "index_sha256": sha256_file(path),
                    "exported_at": dt.datetime.now(dt.UTC).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        return path
    except BaseException as error:
        (destination / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "freeze_sha256": expected_sha256,
                    "completed_files": records,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
            )
            + "\n"
        )
        raise


def load_assessment_index(index_path, freeze_path, expected_sha256, *, root, cache):
    """Validate complete export roles and hashes before any sample-array loader runs."""
    freeze = validate_freeze(freeze_path, expected_sha256, root=root, cache=cache)
    index_path = Path(index_path)
    expected_path = (
        Path(cache)
        / "prepared_assessment"
        / f"{freeze['stage']}-{expected_sha256[:16]}"
        / "index.json"
    )
    if index_path.resolve() != expected_path.resolve():
        raise ValueError("Assessment index is outside its frozen export location")
    index = _read_json(index_path)
    receipt = _read_json(index_path.parent / "receipt.json")
    if (
        index.get("schema_version") != 1
        or index.get("status") != "complete"
        or index.get("role") != "sealed_local_assessment"
        or index.get("stage") != freeze["stage"]
        or index.get("freeze_sha256") != expected_sha256
        or index.get("source_manifest_sha256") != freeze["data"]["manifest_sha256"]
        or index.get("upstream_pin") != UPSTREAM_PIN
    ):
        raise ValueError("Assessment index identity or role disagrees with freeze")
    if (
        receipt.get("status") != "complete"
        or receipt.get("freeze_sha256") != expected_sha256
        or receipt.get("index_sha256") != sha256_file(index_path)
        or (index_path.parent / "failure.json").exists()
    ):
        raise ValueError("Assessment receipt is incomplete or changed")
    expected_rows = {
        row["asset_id"]: row for row in select_assessment_assets(cache, freeze["stage"])
    }
    seen = set()
    file_paths = []
    for row in index["files"]:
        asset_id = row["asset_id"]
        if asset_id not in expected_rows or asset_id in seen:
            raise ValueError("Unauthorized or duplicate assessment asset")
        seen.add(asset_id)
        expected = expected_rows[asset_id]
        for key in (
            "file_id",
            "day",
            "provider_split",
            "raw_sha256",
            "n_bins",
            "n_channels",
            "n_outputs",
            "bin_size_s",
            "channel_ids",
        ):
            if row.get(key) != expected[key]:
                raise ValueError(f"Assessment row disagrees with frozen provider identity: {key}")
        for kind in ("features", "targets"):
            if row[kind] != f"{kind}/{row['file_id']}.npz":
                raise ValueError("Assessment array path does not match its authorized stream")
            path = index_path.parent / row[kind]
            if not path.resolve().is_relative_to(index_path.parent.resolve()):
                raise ValueError("Assessment array symlink escapes export")
            file_paths.append((path, _digest(row[f"{kind}_sha256"])))
    if seen != set(expected_rows):
        raise ValueError("Assessment index omits an authorized stream")
    # All role/identity checks above precede the first sample-file byte read.
    for path, digest in file_paths:
        if sha256_file(path) != digest:
            raise ValueError("Assessment sample artifact identity changed")
    return index


def load_assessment_files(index_path, freeze_path, expected_sha256, *, root, cache):
    """Authorized scorer/refit loader; predictor entry points still receive features only."""
    index = load_assessment_index(index_path, freeze_path, expected_sha256, root=root, cache=cache)
    files = []
    for row in index["files"]:
        features = load_feature_file(
            Path(index_path).parent / row["features"], row["features_sha256"]
        )
        targets = load_target_file(Path(index_path).parent / row["targets"], row["targets_sha256"])
        x, y, mask = features["counts"], targets["targets"], targets["eval_mask"]
        if (
            x.shape != (row["n_bins"], 176)
            or y.shape != (row["n_bins"], 7)
            or not np.array_equal(features["channel_ids"], np.arange(176))
        ):
            raise ValueError("Assessment payload shape/channel order differs from its metadata")
        files.append((row, x, y, mask))
    return index, files
