"""Pinned H1 acquisition, schema audit, and label-separated development export.

The binning algorithm below is adapted from FALCON's MIT-licensed dataloaders.py
at 770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f. We preserve its bin endpoints/gap
behavior and use uint32 counts so the upstream uint8 assignment cannot overflow.
No sealed day is materialized as arrays for modeling by this module.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

from .splits import DEVELOPMENT_DAYS, assert_allowed, day_number, write_split_manifest

UPSTREAM_PIN = "770c9b9597b8f58bf67b5a9ec6401ef705a2cb0f"
SOURCE_FILES = (
    "LICENSE",
    "setup.py",
    "falcon_challenge/config.py",
    "falcon_challenge/evaluator.py",
    "falcon_challenge/dataloaders.py",
    "falcon_challenge/interface.py",
    "decoder_demos/sklearn_decoder.py",
    "decoder_demos/filtering.py",
)
SOURCE_SHA256 = {
    "falcon_challenge/config.py": (
        "3565e4449b7fc1f55a57e7c202083bfc6094ba55e37df2956851b7865a1c6ca9"
    ),
    "falcon_challenge/dataloaders.py": (
        "7638d839ec2085c7dde71ec9ca9676cbf1256c79fdcb327c4ac6487bfe5c9b05"
    ),
}
API = "https://api.dandiarchive.org/api"
FEATURE_KEYS = frozenset({"counts", "timestamps", "sequence_id", "channel_ids"})
TARGET_KEYS = frozenset({"targets", "eval_mask", "trial_change"})


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _download(url: str, path: Path, limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    size = 0
    with urlopen(url, timeout=90) as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > limit:
                raise ValueError(f"Download exceeds declared size limit: {url}")
            stream.write(block)
    partial.replace(path)


def acquire_h1(
    cache: str | Path,
    max_bytes: int = 5 * 1024**3,
    pinned_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Download only public H1 and retain source/provider/local identities.

    DANDI metadata includes CC-BY-4.0 and additional no-redistribution terms.
    Calling this function is appropriate only for authorized local research.
    Completed existing assets are hash checked and reused.
    """
    cache = Path(cache)
    if pinned_manifest is not None or (cache / "manifest.json").exists():
        return _acquire_pinned_h1(
            cache,
            Path(pinned_manifest) if pinned_manifest is not None else cache / "manifest.json",
            max_bytes,
        )
    (cache / "metadata").mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    failures: list[dict[str, str]] = []
    try:
        for name, url in (
            ("versions", f"{API}/dandisets/000954/versions/"),
            ("info", f"{API}/dandisets/000954/versions/draft/info/"),
            ("assets", f"{API}/dandisets/000954/versions/draft/assets/?page_size=100"),
        ):
            _download(url, cache / "metadata" / f"{name}.json", 10 * 1024**2)
        info = json.loads((cache / "metadata/info.json").read_text())
        listing = json.loads((cache / "metadata/assets.json").read_text())
        if listing.get("next"):
            raise ValueError(
                "Inventory pagination changed: audit complete inventory before download"
            )
        assets = listing["results"]
        total = sum(a["size"] for a in assets)
        if total > max_bytes or total != info["size"]:
            raise ValueError("Inventory exceeds storage budget or disagrees with dataset metadata")
        if info["metadata"].get("license") != ["spdx:CC-BY-4.0"]:
            raise ValueError("Dataset license changed; inspect terms before downloading")
        rows = []
        for asset in assets:
            asset_id = asset["asset_id"]
            meta_path = cache / "metadata" / f"{asset_id}.json"
            _download(f"{API}/assets/{asset_id}/", meta_path, 10 * 1024**2)
            metadata = json.loads(meta_path.read_text())
            rel = Path(asset["path"])
            if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".nwb":
                raise ValueError("Unsafe or unexpected provider path")
            output = cache / "raw" / rel
            if not output.exists():
                _download(metadata["contentUrl"][0], output, asset["size"])
            digest = sha256_file(output)
            if output.stat().st_size != asset["size"]:
                raise ValueError(f"Size mismatch: {asset_id}")
            if digest != metadata["digest"]["dandi:sha2-256"]:
                raise ValueError(f"Provider SHA-256 mismatch: {asset_id}")
            rows.append(
                {
                    **asset,
                    "provider_digest": metadata["digest"],
                    "sha256": digest,
                    "retrieved_at": _now(),
                    "local_path": str(output.relative_to(cache)),
                }
            )
        manifest = {
            "schema_version": 1,
            "dandiset": "000954",
            "version": "draft",
            "retrieved_at": _now(),
            "asset_count": len(rows),
            "total_bytes": total,
            "license": info["metadata"]["license"],
            "dua": "Metadata includes additional no-redistribution terms; local cache only",
            "assets": rows,
            "failures": failures,
        }
        _write_json(manifest_path, manifest)
        pin_upstream(cache)
        return manifest
    except Exception as error:
        failures.append({"time": _now(), "type": type(error).__name__, "message": str(error)})
        _write_json(cache / "acquisition_failure.json", {"status": "failed", "failures": failures})
        raise


def _acquire_pinned_h1(cache: Path, source: Path, max_bytes: int) -> dict[str, Any]:
    """Reacquire exact asset UUIDs, never substitute the current mutable draft."""
    manifest_bytes = source.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("dandiset") != "000954" or manifest.get("version") != "draft":
        raise ValueError("Only pinned H1 draft manifests are supported")
    assets = manifest["assets"]
    if sum(asset["size"] for asset in assets) > max_bytes:
        raise ValueError("Pinned inventory exceeds storage budget")
    receipts = []
    try:
        for asset in assets:
            relative = Path(asset["local_path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "raw":
                raise ValueError("Unsafe pinned asset path")
            output = cache / relative
            if not output.exists():
                metadata_path = cache / "metadata" / f"{asset['asset_id']}.json"
                _download(f"{API}/assets/{asset['asset_id']}/", metadata_path, 10 * 1024**2)
                metadata = json.loads(metadata_path.read_text())
                if metadata["digest"] != asset["provider_digest"]:
                    raise ValueError("Provider digest changed for pinned asset")
                _download(metadata["contentUrl"][0], output, asset["size"])
            if (
                output.stat().st_size != asset["size"]
                or sha256_file(output) != asset["sha256"]
                or asset["sha256"] != asset["provider_digest"]["dandi:sha2-256"]
            ):
                raise ValueError("Pinned asset content identity mismatch")
            receipts.append(
                {"asset_id": asset["asset_id"], "verified_at": _now(), "sha256": asset["sha256"]}
            )
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "manifest.json").write_bytes(manifest_bytes)
        _write_json(
            cache / "acquisition_receipt.json",
            {
                "status": "complete",
                "pin_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "verified_assets": receipts,
            },
        )
        pin_upstream(cache)
        return manifest
    except Exception as error:
        _write_json(
            cache / "acquisition_failure.json",
            {
                "status": "failed",
                "verified_assets": receipts,
                "time": _now(),
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        raise


def pin_upstream(cache: Path) -> None:
    rows = []
    for relative in SOURCE_FILES:
        url = f"https://raw.githubusercontent.com/snel-repo/falcon-challenge/{UPSTREAM_PIN}/{relative}"
        path = cache / "upstream" / relative
        if not path.exists():
            _download(url, path, 2 * 1024**2)
        if relative in SOURCE_SHA256 and sha256_file(path) != SOURCE_SHA256[relative]:
            raise ValueError(f"Pinned upstream source changed: {relative}")
        rows.append(
            {
                "path": relative,
                "url": url,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if (cache / "upstream/manifest.json").exists():
        previous = json.loads((cache / "upstream/manifest.json").read_text())
        known = {r["path"]: r for r in previous["files"] if "sha256" in r}
        if previous.get("source_pin") != UPSTREAM_PIN or any(
            known.get(row["path"], {}).get("sha256") != row["sha256"] for row in rows
        ):
            raise ValueError("Upstream cache differs from its pinned manifest")
        return
    _write_json(
        cache / "upstream/manifest.json",
        {"source_pin": UPSTREAM_PIN, "retrieved_at": _now(), "files": rows},
    )


def _verify_source(upstream: Path, relative: str) -> Path:
    path = upstream / relative
    if sha256_file(path) != SOURCE_SHA256[relative]:
        raise ValueError(f"Pinned upstream source changed: {relative}")
    return path


def session_mapping(upstream_dir: str | Path) -> dict[str, str]:
    """Read the literal official mapping without importing its Hydra dependency."""
    source = _verify_source(Path(upstream_dir), "falcon_challenge/config.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "H1_NEW_TO_OLD" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Pinned upstream H1 day mapping missing")


def _text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def bin_spikes(
    spike_times: list[np.ndarray], timestamps: np.ndarray, bin_size_s: float = 0.02
) -> np.ndarray:
    """Match pinned histogram convention, including the closed final right edge.

    Internal boundaries belong to the next bin. Gaps larger than one nominal bin
    discard their earlier interval and retain the proximal interval; short bins
    retain their actual width. No missing bins are synthesized for inference.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all():
        raise ValueError("Timestamps must be a nonempty finite vector")
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        raise ValueError("Bin size must be finite and positive")
    gaps = np.diff(timestamps)
    if (gaps <= 0).any():
        raise ValueError("Bin timestamps must be strictly increasing")
    ends = [timestamps[0]]
    keep = [True]
    for i, gap in enumerate(gaps):
        if not np.isclose(gap, bin_size_s) and gap > bin_size_s:
            ends.extend([timestamps[i + 1] - bin_size_s, timestamps[i + 1]])
            keep.extend([False, True])
        else:
            ends.append(timestamps[i + 1])
            keep.append(True)
    edges = np.r_[ends[0] - bin_size_s, ends]
    output = np.empty((len(timestamps), len(spike_times)), dtype=np.uint32)
    for channel, spikes in enumerate(spike_times):
        spikes = np.asarray(spikes)
        if spikes.ndim != 1 or not np.isfinite(spikes).all():
            raise ValueError("Invalid spike time vector")
        counts = np.histogram(spikes, bins=edges)[0][keep]
        if counts.max(initial=0) > np.iinfo(np.uint32).max:
            raise ValueError("Count exceeds uint32 range")
        output[:, channel] = counts
    return output


def _identity(asset: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    filename = Path(asset["path"]).stem
    session = filename.split("_ses-")[-1]
    file_id = mapping[session]
    provider_split = filename.split("sub-HumanPitt-")[-1].split("_ses-")[0]
    return {
        "file_id": file_id,
        "day": file_id.split("_")[0],
        "provider_split": provider_split,
        "asset_id": asset["asset_id"],
        "raw_sha256": asset["sha256"],
        "path": asset["path"],
    }


def _metadata(file: Any, identity: dict[str, Any]) -> dict[str, Any]:
    kin = file["acquisition/OpenLoopKinematics"]
    velocity = file["acquisition/OpenLoopKinematicsVelocity"]
    n = velocity["data"].shape[0]
    channel_ids = file["units/id"][:].astype(int).tolist()
    if len(set(channel_ids)) != 176 or velocity["data"].shape != (n, 7):
        raise ValueError(f"Unexpected H1 channel/target shape: {identity['path']}")
    rate = float(kin["starting_time"].attrs["rate"])
    offset = float(kin["data"].attrs["offset"])
    mask_name = "eval_mask" if "eval_mask" in file["acquisition"] else "Blacklist"
    return {
        **identity,
        "access_role": "development"
        if identity["day"] in DEVELOPMENT_DAYS
        else "sealed_metadata_only",
        "n_bins": n,
        "n_channels": 176,
        "n_outputs": 7,
        "channel_ids": channel_ids,
        "channel_order": "NWB units table order",
        "channel_physical_mapping": "Not provided by units table; IDs are row identifiers",
        "nwb_identifier": _text(file["identifier"][()]),
        "session_start_time": _text(file["session_start_time"][()]),
        "nwb_starting_time": float(kin["starting_time"][()]),
        "nwb_rate_field": rate,
        "nwb_data_offset": offset,
        "loader_timestamp_rule": "OpenLoopKinematics.offset + arange(T) * rate",
        "loader_first_timestamp": offset,
        "loader_last_timestamp": offset + (n - 1) * rate,
        "bin_size_s": 0.02,
        "nominal_duration_s": n * 0.02,
        "loader_spacing_s": rate,
        "loader_nonuniform_intervals": 0,
        "loader_gap_intervals": (n - 1) if rate > 0.02 and not np.isclose(rate, 0.02) else 0,
        "target_shape": list(velocity["data"].shape),
        "target_dtype": str(velocity["data"].dtype),
        "target_description": _text(velocity.attrs.get("description", "")),
        "target_units": _text(velocity["data"].attrs.get("unit", "")),
        "mask_name": mask_name,
        "mask_shape": list(file[f"acquisition/{mask_name}/data"].shape),
        "mask_dtype": str(file[f"acquisition/{mask_name}/data"].dtype),
        "trial_shape": list(file["acquisition/TrialNum/data"].shape),
        "nwb_has_explicit_kinematic_timestamps": "timestamps" in kin,
        "spike_storage_count": int(file["units/spike_times"].shape[0]),
        "spike_storage_dtype": str(file["units/spike_times"].dtype),
    }


def _development_arrays(file: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_allowed([row], DEVELOPMENT_DAYS)
    timestamps = row["nwb_data_offset"] + np.arange(row["n_bins"]) * row["nwb_rate_field"]
    spike_flat = file["units/spike_times"][:]
    indices = file["units/spike_times_index"][:]
    starts = np.r_[0, indices[:-1]]
    spikes = [spike_flat[int(a) : int(b)] for a, b in zip(starts, indices, strict=True)]
    counts = bin_spikes(spikes, timestamps)
    targets = file["acquisition/OpenLoopKinematicsVelocity/data"][:]
    eval_mask = file[f"acquisition/{row['mask_name']}/data"][:].astype(bool)
    if row["mask_name"] == "Blacklist":
        eval_mask = ~eval_mask
    trial_nums = file["acquisition/TrialNum/data"][:]
    trials = np.r_[False, np.diff(trial_nums) > 0]
    features = {
        "counts": counts,
        "timestamps": timestamps,
        "sequence_id": np.arange(row["n_bins"], dtype=np.int64),
        "channel_ids": np.array(row["channel_ids"], dtype=np.int64),
    }
    scoring = {"targets": targets, "eval_mask": eval_mask, "trial_change": trials}
    return features, scoring


def prepare_h1(cache: str | Path, *, compare_official: bool = True) -> dict[str, Any]:
    """Audit all files, but export only S0-S4 held-in calibration arrays."""
    import h5py

    cache = Path(cache)
    raw_manifest = cache / "manifest.json"
    manifest = json.loads(raw_manifest.read_text())
    mapping = session_mapping(cache / "upstream")
    prepared = cache / "prepared"
    audit_rows = []
    records = []
    official = _official_loader(cache / "upstream") if compare_official else None
    for asset in manifest["assets"]:
        raw = cache / asset["local_path"]
        if sha256_file(raw) != asset["sha256"]:
            raise ValueError(f"Raw identity changed: {asset['path']}")
        identity = _identity(asset, mapping)
        with h5py.File(raw, "r") as file:
            row = _metadata(file, identity)
            if (
                identity["day"] in DEVELOPMENT_DAYS
                and identity["provider_split"] == "held-in-calib"
            ):
                features, scoring = _development_arrays(file, row)
                if official:
                    reference = official(raw)
                    for actual, expected in zip(
                        [
                            features["counts"],
                            scoring["targets"],
                            scoring["trial_change"],
                            scoring["eval_mask"],
                        ],
                        reference,
                        strict=True,
                    ):
                        if not np.array_equal(actual, expected):
                            raise ValueError(f"Pinned loader parity failed: {asset['path']}")
                row["count_min"] = int(features["counts"].min())
                row["count_max"] = int(features["counts"].max())
                row["uint8_overflow_possible_in_this_file"] = row["count_max"] > 255
                row["scored_bins"] = int(scoring["eval_mask"].sum())
                row["trial_changes"] = int(scoring["trial_change"].sum())
                row["official_loader_exact_array_parity"] = bool(official)
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
                        "scored_bins",
                    )
                }
                for kind, arrays in [("features", features), ("targets", scoring)]:
                    out = prepared / kind / f"{row['file_id']}.npz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out, **arrays)
                    record[kind] = str(out.relative_to(prepared))
                    record[f"{kind}_sha256"] = sha256_file(out)
                records.append(record)
            audit_rows.append(row)
    records.sort(key=lambda x: (day_number(x["day"]), x["file_id"]))
    index = {
        "schema_version": 1,
        "role": "development",
        "upstream_pin": UPSTREAM_PIN,
        "source_manifest_sha256": sha256_file(raw_manifest),
        "files": records,
    }
    assert_allowed(records, DEVELOPMENT_DAYS)
    _write_json(prepared / "index.json", index)
    _write_json(
        cache / "schema_audit.json",
        {
            "schema_version": 1,
            "source_manifest_sha256": sha256_file(raw_manifest),
            "upstream_pin": UPSTREAM_PIN,
            "sealed_behavioral_arrays_read": False,
            "files": audit_rows,
        },
    )
    write_split_manifest(cache / "split_manifest.json")
    audit_minival_overlap(cache)
    return index


def audit_minival_overlap(cache: str | Path) -> dict[str, Any]:
    """Compare development prefixes only; sealed overlaps remain metadata-derived."""
    import h5py

    cache = Path(cache)
    rows = json.loads((cache / "schema_audit.json").read_text())["files"]
    lookup = {(row["file_id"], row["provider_split"]): row for row in rows}
    evidence = []
    for row in rows:
        if row["provider_split"] != "held-in-minival":
            continue
        parent = lookup[(row["file_id"], "held-in-calib")]
        item = {
            "file_id": row["file_id"],
            "day": row["day"],
            "minival_asset_id": row["asset_id"],
            "calibration_asset_id": parent["asset_id"],
            "same_session_start_time": row["session_start_time"] == parent["session_start_time"],
            "minival_n_bins": row["n_bins"],
            "calibration_n_bins": parent["n_bins"],
            "provider_declares_minival_subset": True,
            "use": "interface checks only; not independent evidence",
        }
        if row["day"] in DEVELOPMENT_DAYS:
            with (
                h5py.File(cache / "raw" / row["path"], "r") as mini,
                h5py.File(cache / "raw" / parent["path"], "r") as calibration,
            ):
                n = row["n_bins"]
                comparisons = {}
                for field in ("OpenLoopKinematicsVelocity", "eval_mask", "TrialNum"):
                    comparisons[field] = bool(
                        np.array_equal(
                            mini[f"acquisition/{field}/data"][:],
                            calibration[f"acquisition/{field}/data"][:n],
                        )
                    )
                if not all(comparisons.values()):
                    raise ValueError("Minival prefix changed; investigate overlap before use")
                item["exact_development_prefix_fields"] = comparisons
                item["calibration_interval_bins_half_open"] = [0, n]
                item["behavioral_arrays_inspected"] = True
        else:
            item["behavioral_arrays_inspected"] = False
            item["exact_interval"] = "not inspected; sealed"
        evidence.append(item)
    result = {"schema_version": 1, "upstream_pin": UPSTREAM_PIN, "pairs": evidence}
    _write_json(cache / "minival_overlap.json", result)
    return result


def _official_loader(upstream: Path) -> Any:
    """Execute only the pinned loader functions, with an exact minimal task enum.

    The official config imports Hydra and registers global state. AST extraction
    leaves the loader source unchanged while avoiding installation of the whole
    benchmark package. Source hashes remain in upstream/manifest.json.
    """
    import enum
    from typing import Optional, Union

    import pandas as pd

    # Keep the library schema cache beside the single study cache, within budget.
    previous_cache = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str((upstream.parent / "library_cache").resolve())
    try:
        from pynwb import NWBHDF5IO
    finally:
        if previous_cache is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_cache

    class FalconTask(enum.Enum):
        h1 = "falcon_h1_7d"
        h2 = "falcon_h2_writing"
        m1 = "falcon_m1_finger"
        m2 = "falcon_m2_reach"
        b1 = "falcon_b1_vocal"

    tree = ast.parse(_verify_source(upstream, "falcon_challenge/dataloaders.py").read_text())
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"bin_units", "load_nwb"}
    ]
    namespace = {
        "np": np,
        "pd": pd,
        "NWBHDF5IO": NWBHDF5IO,
        "Path": Path,
        "Optional": Optional,
        "Union": Union,
        "Tuple": tuple,
        "FalconTask": FalconTask,
    }
    exec(
        compile(ast.Module(body=funcs, type_ignores=[]), "pinned_dataloaders.py", "exec"), namespace
    )
    return namespace["load_nwb"]


def load_index(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    index = json.loads(path.read_text())
    if index.get("schema_version") != 1 or index.get("role") != "development":
        raise ValueError("Unknown or nondevelopment prepared index")
    assert_allowed(index["files"], DEVELOPMENT_DAYS)
    if index.get("upstream_pin") != UPSTREAM_PIN:
        raise ValueError("Prepared index upstream pin changed")
    cache = path.parent.parent
    manifest_path = cache / "manifest.json"
    if sha256_file(manifest_path) != index["source_manifest_sha256"]:
        raise ValueError("Source manifest identity changed")
    assets = {a["asset_id"]: a for a in json.loads(manifest_path.read_text())["assets"]}
    mapping = session_mapping(cache / "upstream")
    for record in index["files"]:
        if record["asset_id"] not in assets:
            raise ValueError("Unknown source asset")
        identity = _identity(assets[record["asset_id"]], mapping)
        for key in ("file_id", "day", "provider_split", "raw_sha256"):
            if record[key] != identity[key]:
                raise ValueError(f"Source role/identity mismatch: {key}")
        for kind in ("features", "targets"):
            relative = Path(record[kind])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Prepared path must stay inside prepared cache")
            if relative != Path(kind) / f"{record['file_id']}.npz":
                raise ValueError("Prepared stream path does not match file identity")
    return index


def _load_arrays(
    path: str | Path, expected_keys: frozenset[str], expected_sha256: str | None = None
) -> dict[str, np.ndarray]:
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError("Prepared array identity changed")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise ValueError("Prepared array keys violate information boundary")
        return {key: arrays[key] for key in arrays.files}


def load_feature_file(
    path: str | Path, expected_sha256: str | None = None
) -> dict[str, np.ndarray]:
    arrays = _load_arrays(path, FEATURE_KEYS, expected_sha256)
    counts = arrays["counts"]
    n = len(counts)
    if counts.ndim != 2 or counts.shape[1] != len(arrays["channel_ids"]):
        raise ValueError("Counts/channel shape mismatch")
    if not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError("Counts must be nonnegative integers")
    if not np.array_equal(arrays["sequence_id"], np.arange(n)):
        raise ValueError("Missing, reordered, or duplicate sequence ID")
    stamps = arrays["timestamps"]
    if stamps.shape != (n,) or not np.isfinite(stamps).all() or (np.diff(stamps) <= 0).any():
        raise ValueError("Invalid timestamps")
    if len(set(arrays["channel_ids"].tolist())) != counts.shape[1]:
        raise ValueError("Duplicate channel IDs")
    return arrays


def load_target_file(path: str | Path, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    arrays = _load_arrays(path, TARGET_KEYS, expected_sha256)
    targets, mask, changes = (arrays[key] for key in ("targets", "eval_mask", "trial_change"))
    if targets.ndim != 2 or mask.shape != (len(targets),) or changes.shape != mask.shape:
        raise ValueError("Target/mask/trial shape mismatch")
    if mask.dtype != bool or changes.dtype != bool or not np.isfinite(targets).all():
        raise ValueError("Invalid targets, mask, or trial changes")
    return arrays


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "prepare"])
    parser.add_argument("--cache", type=Path, default=Path("data/neural_h1"))
    parser.add_argument("--manifest", type=Path, help="Reacquire exact pinned H1 asset UUIDs")
    args = parser.parse_args(argv)
    result = (
        acquire_h1(args.cache, pinned_manifest=args.manifest)
        if args.command == "acquire"
        else prepare_h1(args.cache)
    )
    print(
        json.dumps(
            {"command": args.command, "files": len(result.get("files", result.get("assets", [])))},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
