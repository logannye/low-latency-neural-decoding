"""Bounded, pinned nine-human Dryad acquisition and label-separated conversion.

Run from the repository root with ``PYTHONPATH=src python -m
neural_experiments.v2.data --help``. This module never normalizes by session or
block statistics. Causality of the provider's voltage-to-count pipeline remains
unverified; only the supplied-feature conversion is causal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import parse_qs, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
import scipy
from scipy.io import loadmat

DOI = "10.5061/dryad.x0k6djj1h"
VERSION_ID = 462702
PARTICIPANTS = ("T2", "T3", "T5", "T6", "T7", "T8", "T9", "T10", "T11")
BASE = "https://datadryad.org"
PINS = {
    "T2": (4952113, 4373873076, "892d860cb690b59368c0aa8a550e9ac691e3be95af33b55eb60609351df5d3a5"),
    "T3": (4952114, 616043613, "bf0424759beda78a13c3ecb45bb5dcc48cbb23c867d1dc3f4f8866d023844230"),
    "T5": (
        4952115,
        11335444469,
        "9bdd17c23734d0a16de5a8ae51fa100a7a957af7f80e31eda91b9b910dd4eab3",
    ),
    "T6": (4952116, 6918399242, "4096e48994457e30571c6ce3e29b344abd27cbd9b805ff6c417355515312be61"),
    "T7": (4952117, 4419103236, "5887e32005441b04f37ece85381621247abb361f1168f73178163125c6bf6e4d"),
    "T8": (
        4952118,
        14175870111,
        "e3c75e6117163bf1e71d93fd8d96d79b1447dc1f0014a557af2d074e24d95276",
    ),
    "T9": (
        4952119,
        11725366996,
        "253a4928670087366b4fbf0641b40eca36452e0fcac9e661630948adb64513ff",
    ),
    "T10": (
        4952120,
        4621774785,
        "4f35f977ddd43fc024bf2aa5f46c411b2314a526592232da6390cf20d937d7a8",
    ),
    "T11": (
        4952121,
        21242829481,
        "74f9735ff07ac7e47f8cfad16bf1070c5f3895ff279697d8bb8e96b6c6143c61",
    ),
}
FEATURE_KEYS = frozenset({"x", "timestamps", "sequence_id", "channel_ids"})
TARGET_KEYS = frozenset({"y", "eval_mask"})
BIN_SECONDS = 0.02
TARGET_EPSILON = 1e-12
TARGET_DEFINITION = (
    "At the second 10-ms sample of each within-block pair, y=(target_position-"
    "cursor_position)/norm(target_position-cursor_position); zero if nonfinite "
    "or norm<=1e-12. eval_mask requires finite nonzero direction and both samples "
    "inside one provider go interval. Target is inferred intended direction, "
    "not measured limb velocity."
)
POLICY = {
    "selection": "first_n_complete_session_mat_members_in_provider_archive_order",
    "scores_used_for_selection": False,
    "ordering_after_selection": "numeric_post_implant_day_then_session_id",
    "neural_feature": "tx_4_5",
    "fallback": None,
    "count_conversion": "sum_two_consecutive_10ms_bins_within_each_block",
    "target_definition": TARGET_DEFINITION,
    "target_epsilon": TARGET_EPSILON,
    "bin_seconds": BIN_SECONDS,
    "normalization": "none",
    "upstream_feature_causality": "unverified",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Unsafe artifact path")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes cache")
    return result


def _cache_size(cache: Path) -> int:
    return sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())


def pin_metadata(cache: str | Path) -> dict[str, Any]:
    """Read public metadata only and verify immutable archive identities."""
    cache = Path(cache)
    metadata = cache / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    objects = {}
    endpoints = {
        "dataset": f"{BASE}/api/v2/datasets/doi%3A10.5061%2Fdryad.x0k6djj1h",
        "version": f"{BASE}/api/v2/versions/{VERSION_ID}",
        "files-page1": f"{BASE}/api/v2/versions/{VERSION_ID}/files",
        "files-page2": f"{BASE}/api/v2/versions/{VERSION_ID}/files?page=2",
    }
    sources = []
    for name, url in endpoints.items():
        path = metadata / f"{name}.json"
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError("Metadata response exceeded limit")
            path.write_bytes(body)
        objects[name] = json.loads(path.read_bytes())
        sources.append(
            {"url": url, "path": str(path.relative_to(cache)), "sha256": sha256_file(path)}
        )
    if objects["dataset"].get("license") != "https://spdx.org/licenses/CC0-1.0.html":
        raise ValueError("Unexpected provider license; inspect before acquisition")
    if objects["version"]["_links"]["self"]["href"] != f"/api/v2/versions/{VERSION_ID}":
        raise ValueError("Provider version mismatch")
    rows = objects["files-page1"]["_embedded"]["stash:files"]
    rows += objects["files-page2"]["_embedded"]["stash:files"]
    if len(rows) != objects["files-page1"]["total"]:
        raise ValueError("Incomplete provider metadata pagination")
    archives = []
    for participant in PARTICIPANTS:
        candidates = [r for r in rows if r["path"] == f"decoding_{participant}.tar.gz"]
        if len(candidates) != 1:
            raise ValueError(f"Expected one decoding archive for {participant}")
        row = candidates[0]
        file_id, size, digest = PINS[participant]
        if (
            row["size"] != size
            or row["digest"] != digest
            or row["digestType"] != "sha-256"
            or row["_links"]["self"]["href"] != f"/api/v2/files/{file_id}"
        ):
            raise ValueError(f"Provider archive identity mismatch: {participant}")
        archives.append(
            {
                "participant_id": participant,
                "file_id": file_id,
                "filename": row["path"],
                "size": size,
                "provider_sha256": digest,
                "download_url": f"{BASE}/downloads/file_stream/{file_id}",
            }
        )
    manifest = {
        "schema_version": 1,
        "doi": DOI,
        "version_id": VERSION_ID,
        "license": "CC0-1.0",
        "metadata_sources": sources,
        "archives": archives,
    }
    _json(cache / "provider-manifest.json", manifest)
    return manifest


class BoundedReader:
    """Count and hash consumed compressed prefix, without claiming full verification."""

    def __init__(self, stream: BinaryIO, limit: int):
        self.stream, self.limit = stream, limit
        self.count = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        # tar streaming asks for fixed chunks. A read-all call is never allowed.
        if size < 0:
            raise ValueError("Unbounded stream read is prohibited")
        if self.count + size > self.limit:
            size = self.limit - self.count
        if size <= 0:
            raise ValueError("Compressed acquisition prefix exceeded byte limit")
        block = self.stream.read(size)
        self.count += len(block)
        self.digest.update(block)
        return block


def extract_first_sessions(
    stream: BinaryIO,
    *,
    cache: Path,
    participant: str,
    sessions: int,
    max_stream_bytes: int = 4 * 1024**3,
    max_cache_bytes: int = 20 * 1024**3,
    archive: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select exact first N session members, including short/zero-go sessions.

    Never extract tar paths directly. Only regular, strictly named session MAT
    files are copied; links/path traversal are rejected. A failed member keeps
    its .part file and a failure receipt. Selection is not changed after failure.
    """
    if participant not in PARTICIPANTS or not 1 <= sessions <= 6:
        raise ValueError("Use a pinned participant and 1..6 sessions")
    cache = Path(cache)
    raw_dir = cache / "raw" / participant
    raw_dir.mkdir(parents=True, exist_ok=True)
    reader = BoundedReader(stream, max_stream_bytes)
    selected: list[dict[str, Any]] = []
    seen = []
    receipt = {
        "schema_version": 1,
        "participant_id": participant,
        "requested_sessions": sessions,
        "policy": POLICY,
        "archive": archive,
        "started_at": _now(),
        "status": "running",
    }
    failure = None
    consumed_end = False
    try:
        with tarfile.open(fileobj=reader, mode="r|gz", bufsize=64 * 1024) as tar:
            for ordinal, member in enumerate(tar):
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                    raise ValueError("Unsafe archive member path")
                if member.issym() or member.islnk():
                    raise ValueError("Archive links are not accepted")
                seen.append(
                    {
                        "ordinal": ordinal,
                        "name": member.name,
                        "size": member.size,
                        "regular_file": member.isfile(),
                    }
                )
                if not member.isfile():
                    continue
                match = re.fullmatch(
                    rf"(?:\./)?decoding/{participant}/{participant}_day_(\d+)_decoding\.mat",
                    member.name,
                )
                if not match:
                    if member.name.endswith(".mat") and not name.name.startswith("._"):
                        raise ValueError(f"Unexpected session member: {member.name}")
                    continue
                target = raw_dir / name.name
                if any(row["member_name"] == member.name for row in selected):
                    raise ValueError("Duplicate session member")
                existing_bytes = target.stat().st_size if target.exists() else 0
                if member.size > 2 * 1024**3:
                    raise ValueError("Selected session exceeds 2 GiB safety budget")
                if _cache_size(cache) + member.size - existing_bytes > max_cache_bytes:
                    raise ValueError("Selected session would exceed cache disk budget")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError("Regular session member has no payload")
                temporary = target.with_suffix(".mat.part")
                digest = hashlib.sha256()
                written = 0
                with temporary.open("wb") as output:
                    while chunk := extracted.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if written != member.size:
                    raise ValueError("Truncated session member")
                if target.exists() and sha256_file(target) != digest.hexdigest():
                    raise ValueError("Existing selected member identity changed")
                temporary.replace(target)
                selected.append(
                    {
                        "participant_id": participant,
                        "session_id": name.stem,
                        "post_implant_day": int(match.group(1)),
                        "archive_ordinal": ordinal,
                        "member_name": member.name,
                        "member_size": member.size,
                        "member_sha256": digest.hexdigest(),
                        "raw_path": str(target.relative_to(cache)),
                    }
                )
                if len(selected) == sessions:
                    break
            else:
                consumed_end = True
        if not selected:
            raise ValueError("No valid session members in archive")
        receipt["status"] = "complete"
    except Exception as error:
        receipt["status"] = "failed"
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt.update(
            {
                "finished_at": _now(),
                "selected": selected,
                "seen_members": seen,
                "failure": failure,
                "compressed_prefix_bytes": reader.count,
                "compressed_prefix_sha256": reader.digest.hexdigest(),
                "tar_end_observed": consumed_end,
                "full_archive_sha256_verified": False,
                "verification_limit": "Selected complete member bytes hashed locally; "
                "provider checksum covers full archive, not this consumed prefix.",
            }
        )
        _json(cache / "acquisition" / f"{participant}.json", receipt)
    return receipt


def validate_receipt(receipt: dict[str, Any], participant: str) -> None:
    """Check score-independent member selection against the saved archive walk."""
    if (
        receipt.get("status") != "complete"
        or receipt.get("participant_id") != participant
        or receipt.get("policy") != POLICY
    ):
        raise ValueError("Invalid acquisition receipt identity/policy")
    budget = receipt.get("requested_sessions")
    selected = receipt.get("selected", [])
    if type(budget) is not int or not 1 <= budget <= 6 or not 1 <= len(selected) <= budget:
        raise ValueError("Invalid acquisition selection budget")
    if len(selected) < budget and not receipt.get("tar_end_observed"):
        raise ValueError("Incomplete acquisition must not count as complete selection")
    file_id, size, digest = PINS[participant]
    archive = receipt.get("archive") or {}
    if (
        archive.get("file_id") != file_id
        or archive.get("size") != size
        or archive.get("provider_sha256") != digest
    ):
        raise ValueError("Acquisition provider archive identity mismatch")
    pattern = rf"(?:\./)?decoding/{participant}/{participant}_day_(\d+)_decoding\.mat"
    expected = [
        row
        for row in receipt["seen_members"]
        if row["regular_file"] and re.fullmatch(pattern, row["name"])
    ][:budget]
    if len(expected) != len(selected):
        raise ValueError("Selected sessions differ from first archive members")
    for row, seen in zip(selected, expected, strict=True):
        match = re.fullmatch(pattern, row["member_name"])
        if match is None:
            raise ValueError("Invalid selected member name")
        session_id = PurePosixPath(row["member_name"]).stem
        if (
            row["participant_id"] != participant
            or row["session_id"] != session_id
            or row["post_implant_day"] != int(match.group(1))
            or row["archive_ordinal"] != seen["ordinal"]
            or row["member_name"] != seen["name"]
            or row["member_size"] != seen["size"]
            or row["raw_path"] != f"raw/{participant}/{session_id}.mat"
            or re.fullmatch(r"[0-9a-f]{64}", row["member_sha256"]) is None
        ):
            raise ValueError("Selected member provenance differs from archive walk")


def acquire(
    cache: str | Path,
    *,
    sessions: int = 6,
    participants: tuple[str, ...] = PARTICIPANTS,
    archive_directory: str | Path | None = None,
    url_file: str | Path | None = None,
    max_stream_bytes: int = 4 * 1024**3,
    max_cache_bytes: int = 20 * 1024**3,
) -> dict[str, Any]:
    cache = Path(cache)
    manifest = pin_metadata(cache)
    if archive_directory is not None and url_file is not None:
        raise ValueError("Choose local archives or observed redirects, not both")
    observed_urls = json.loads(Path(url_file).read_text()) if url_file is not None else {}
    receipts = []
    for participant in participants:
        if participant not in PARTICIPANTS:
            raise ValueError("Only pinned decoding participants may be acquired")
        path = cache / "acquisition" / f"{participant}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["status"] == "complete" and previous["requested_sessions"] == sessions:
                validate_receipt(previous, participant)
                for item in previous["selected"]:
                    if sha256_file(_safe_path(cache, item["raw_path"])) != item["member_sha256"]:
                        raise ValueError("Cached raw member hash mismatch")
                receipts.append(previous)
                continue
        archive = next(a for a in manifest["archives"] if a["participant_id"] == participant)
        try:
            if archive_directory is not None:
                local = Path(archive_directory) / archive["filename"]
                if local.stat().st_size != archive["size"]:
                    raise ValueError("Local archive size differs from pinned provider metadata")
                if sha256_file(local) != archive["provider_sha256"]:
                    raise ValueError("Local archive SHA256 differs from pinned provider metadata")
                source = local.open("rb")
            else:
                download_url = archive["download_url"]
                if url_file is not None:
                    download_url = observed_urls[participant]
                    parsed = urlsplit(download_url)
                    disposition = parse_qs(parsed.query).get("response-content-disposition", [])
                    if (
                        parsed.scheme != "https"
                        or parsed.username
                        or parsed.password
                        or parsed.hostname
                        != "dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com"
                        or disposition != [f"attachment; filename={archive['filename']}"]
                    ):
                        raise ValueError(
                            "Redirect must be an observed matching public Dryad object"
                        )
                    archive = {
                        **archive,
                        "transport": {
                            "mode": "browser_observed_public_redirect",
                            "object_url": urlunsplit(
                                (parsed.scheme, parsed.netloc, parsed.path, "", "")
                            ),
                            "private_url_file_sha256": sha256_file(url_file),
                        },
                    }
                request = Request(
                    download_url,
                    headers={"User-Agent": "low-latency-neural-decoding/research", "Accept": "*/*"},
                )
                source = urlopen(request, timeout=90)
                if source.headers.get("Content-Length") != str(archive["size"]):
                    source.close()
                    raise ValueError("Provider response length differs from pinned archive size")
            with source:
                receipt = extract_first_sessions(
                    source,
                    cache=cache,
                    participant=participant,
                    sessions=sessions,
                    max_stream_bytes=max_stream_bytes,
                    max_cache_bytes=max_cache_bytes,
                    archive=archive,
                )
            if archive_directory is not None:
                receipt["full_archive_sha256_verified"] = True
                receipt["verification_limit"] = "Full local archive and selected members verified"
                _json(path, receipt)
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "participant": participant,
                        "sessions": len(receipt["selected"]),
                        "compressed_prefix_bytes": receipt["compressed_prefix_bytes"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")
            _json(
                cache / "failures" / f"{participant}-{stamp}.json",
                {
                    "participant_id": participant,
                    "time": _now(),
                    "archive": archive,
                    "type": type(error).__name__,
                    "message": str(error),
                    "status": "failed_no_substitution",
                },
            )
            raise
    return {"provider": manifest, "receipts": receipts}


def _integers(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value).reshape(-1)
    if not np.issubdtype(vector.dtype, np.number) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite numeric integers")
    if not np.equal(vector, np.floor(vector)).all():
        raise ValueError(f"{name} must be integer-valued")
    return vector.astype(np.int64)


def _struct_vectors(value: Any, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not isinstance(value, dict) or not all(f in value for f in fields):
        raise ValueError(f"Missing provider struct fields: {fields}")
    rows = {f: _integers(value[f], f) for f in fields}
    if len({len(v) for v in rows.values()}) != 1:
        raise ValueError("Struct fields have inconsistent lengths")
    return rows


def convert_session(
    raw: dict[str, Any], participant: str, day: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Convert a complete session without reading another session or fitting stats.

    Pair within each provider block, discard at most one trailing 10-ms bin per
    block, and reset at each boundary. Gaps between blocks remain visible in
    source-relative timestamps. No wall-clock time across blocks is inferred.
    """
    if str(raw.get("participant_id")) != participant:
        raise ValueError("Provider participant does not match archive")
    raw_day = _integers(raw.get("post_implant_day"), "post_implant_day")
    if raw_day.size != 1 or raw_day[0] != day:
        raise ValueError("Provider day does not match member name")
    if float(raw["bin_size"]) != 0.01:
        raise ValueError("Expected provider 10-ms bins")
    neural = raw["neural"]
    if not isinstance(neural, dict) or "tx_4_5" not in neural:
        raise ValueError("Required prespecified tx_4_5 counts unavailable; no fallback")
    counts = np.asarray(neural["tx_4_5"])
    if counts.ndim != 2 or counts.shape[1] not in (96, 192):
        raise ValueError("Expected time x 96/192 provider threshold counts")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Neural counts must be finite and nonnegative")
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Threshold counts must be integer-valued")
    n_source, channels = counts.shape
    electrode = raw["electrodes"]
    channel_ids = _integers(electrode["electrode_id"], "electrode_id")
    if (
        len(channel_ids) != channels
        or len(set(channel_ids.tolist())) != channels
        or np.any(np.diff(channel_ids) <= 0)
    ):
        raise ValueError("Electrodes must be unique and in strictly increasing provider order")
    cursor = np.asarray(raw["cursor_position"], dtype=np.float64)
    target = np.asarray(raw["target_position"], dtype=np.float64)
    if cursor.shape != (n_source, 2) or target.shape != cursor.shape:
        raise ValueError("Cursor/target shape differs from aligned neural stream")
    blocks = _struct_vectors(raw["blocks"], ("block_id", "sample_start_index", "sample_end_index"))
    starts, ends, ids = (blocks[k] for k in ("sample_start_index", "sample_end_index", "block_id"))
    if (
        len(starts) == 0
        or np.any(starts < 0)
        or np.any(ends > n_source)
        or np.any(ends <= starts)
        or np.any(starts[1:] < ends[:-1])
        or len(set(ids.tolist())) != len(ids)
    ):
        raise ValueError("Provider blocks must be unique, ordered, nonoverlapping intervals")
    trials = _struct_vectors(raw["trials"], ("trial_start_index", "trial_end_index", "block_id"))
    go_id = np.full(n_source, -1, dtype=np.int64)
    block_lookup = {int(b): (int(s), int(e)) for b, s, e in zip(ids, starts, ends, strict=True)}
    for number, (start, end, block) in enumerate(
        zip(trials["trial_start_index"], trials["trial_end_index"], trials["block_id"], strict=True)
    ):
        if int(block) not in block_lookup:
            raise ValueError("Trial references unknown block")
        block_start, block_end = block_lookup[int(block)]
        if not block_start <= start < end <= block_end:
            raise ValueError("Go interval crosses its provider block")
        if np.any(go_id[start:end] != -1):
            raise ValueError("Go intervals overlap")
        go_id[start:end] = number
    x_parts, t_parts, y_parts, masks, sequence_parts = [], [], [], [], []
    segments = []
    output_start = 0
    for sequence, (block_id, start, end) in enumerate(zip(ids, starts, ends, strict=True)):
        start, end = int(start), int(end)
        usable = (end - start) // 2 * 2
        n_pairs = usable // 2
        second = np.arange(start + 1, start + usable, 2)
        paired = counts[start : start + usable].astype(np.float64).reshape(n_pairs, 2, channels)
        x = paired.sum(axis=1).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("20-ms counts overflow float32")
        delta = target[second] - cursor[second]
        norm = np.linalg.norm(delta, axis=1)
        valid = np.isfinite(delta).all(axis=1) & np.isfinite(norm) & (norm > TARGET_EPSILON)
        y = np.zeros((n_pairs, 2), dtype=np.float32)
        y[valid] = (delta[valid] / norm[valid, None]).astype(np.float32)
        go = (go_id[second] >= 0) & (go_id[second] == go_id[second - 1])
        x_parts.append(x)
        # Endpoint convention: first output represents completed source bins 0 and 1 at .02s.
        t_parts.append((second + 1).astype(np.float64) * 0.01)
        y_parts.append(y)
        masks.append(valid & go)
        sequence_parts.append(np.full(n_pairs, sequence, dtype=np.int64))
        segments.append(
            {
                "sequence_id": sequence,
                "block_id": int(block_id),
                "start": output_start,
                "end": output_start + n_pairs,
                "source_start": start,
                "source_end": end,
                "source_n_bins": end - start,
                "paired_source_bins": usable,
                "discarded_trailing_10ms_bins": (end - start) - usable,
                "source_gap_before_bins": start - (int(ends[sequence - 1]) if sequence else 0),
            }
        )
        output_start += n_pairs
    features = {
        "x": np.concatenate(x_parts),
        "timestamps": np.concatenate(t_parts),
        "sequence_id": np.concatenate(sequence_parts),
        "channel_ids": channel_ids,
    }
    targets = {"y": np.concatenate(y_parts), "eval_mask": np.concatenate(masks)}
    metadata = {
        "n_source_bins": n_source,
        "n_bins": output_start,
        "n_channels": channels,
        "bin_seconds": BIN_SECONDS,
        "duration_seconds": output_start * BIN_SECONDS,
        "source_duration_seconds": n_source * 0.01,
        "n_eval_bins": int(targets["eval_mask"].sum()),
        "segments": segments,
        "discarded_unblocked_10ms_bins": int(n_source - np.sum(ends - starts)),
        "discarded_trailing_10ms_bins": int(
            sum(s["discarded_trailing_10ms_bins"] for s in segments)
        ),
        "target_definition": TARGET_DEFINITION,
        "target_epsilon": TARGET_EPSILON,
        "timestamp_definition": "source-bin right endpoint; between-block wall time unknown",
        "channel_identity": "provider-local electrode IDs; not cross-human homology",
        "neural_fields_available": sorted(neural),
        "normalization": "none",
        "upstream_feature_causality": "unverified",
    }
    return features, targets, metadata


def prepare(cache: str | Path, *, max_cache_bytes: int = 20 * 1024**3) -> dict[str, Any]:
    cache = Path(cache)
    manifest_path = cache / "provider-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("doi") != DOI or manifest.get("version_id") != VERSION_ID:
        raise ValueError("Unexpected provider manifest identity")
    sessions = []
    acquisition_sources = []
    requested_counts = set()
    for participant in PARTICIPANTS:
        receipt_path = cache / "acquisition" / f"{participant}.json"
        receipt = json.loads(receipt_path.read_text())
        validate_receipt(receipt, participant)
        requested_counts.add(receipt["requested_sessions"])
        acquisition_sources.append(
            {"path": str(receipt_path.relative_to(cache)), "sha256": sha256_file(receipt_path)}
        )
        for selected in receipt["selected"]:
            raw_path = _safe_path(cache, selected["raw_path"])
            if sha256_file(raw_path) != selected["member_sha256"]:
                raise ValueError("Raw selected-member hash mismatch")
            raw = loadmat(raw_path, simplify_cells=True)
            features, targets, metadata = convert_session(
                raw, participant, selected["post_implant_day"]
            )
            del raw  # Release unused provider feature arrays before loading the next MAT.
            output = cache / "prepared" / participant
            output.mkdir(parents=True, exist_ok=True)
            feature_path = output / f"{selected['session_id']}.features.npz"
            target_path = output / f"{selected['session_id']}.targets.npz"
            estimate = sum(v.nbytes for v in [*features.values(), *targets.values()]) + 65536
            replace_bytes = sum(p.stat().st_size for p in (feature_path, target_path) if p.exists())
            if _cache_size(cache) + estimate - replace_bytes > max_cache_bytes:
                raise ValueError("Prepared output would exceed cache disk budget")
            np.savez_compressed(feature_path, **features)
            np.savez_compressed(target_path, **targets)
            source = {
                **selected,
                "archive_file_id": receipt["archive"]["file_id"],
                "archive_sha256_published": receipt["archive"]["provider_sha256"],
                "full_archive_sha256_verified": receipt["full_archive_sha256_verified"],
            }
            sessions.append(
                {
                    "participant_id": participant,
                    "session_id": selected["session_id"],
                    "post_implant_day": selected["post_implant_day"],
                    "feature_path": str(feature_path.relative_to(cache)),
                    "target_path": str(target_path.relative_to(cache)),
                    "feature_sha256": sha256_file(feature_path),
                    "target_sha256": sha256_file(target_path),
                    "source": source,
                    **metadata,
                }
            )
    if len(requested_counts) != 1:
        raise ValueError("Participant selection budgets differ")
    order = {p: i for i, p in enumerate(PARTICIPANTS)}
    sessions.sort(
        key=lambda s: (order[s["participant_id"]], s["post_implant_day"], s["session_id"])
    )
    index = {
        "schema_version": 1,
        "created_at": _now(),
        "preparer": {
            "path": "neural_experiments/v2/data.py",
            "sha256": sha256_file(__file__),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "provider": {
            "doi": DOI,
            "version_id": VERSION_ID,
            "license": "CC0-1.0",
            "manifest_path": str(manifest_path.relative_to(cache)),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "selection": {**POLICY, "max_sessions_per_participant": requested_counts.pop()},
        "participants": list(PARTICIPANTS),
        "sessions": sessions,
        "acquisition_sources": acquisition_sources,
    }
    _json(cache / "index.json", index)
    return index


def load_index(path: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Validate all cohort identities and artifacts before opening any sample arrays."""
    path = Path(path)
    cache = path.parent
    index = json.loads(path.read_text())
    if index.get("schema_version") != 1 or index.get("participants") != list(PARTICIPANTS):
        raise ValueError("Index must contain all nine pinned participants")
    provider = index["provider"]
    if provider["doi"] != DOI or provider["version_id"] != VERSION_ID:
        raise ValueError("Unexpected provider identity")
    if any(index["selection"].get(k) != v for k, v in POLICY.items()):
        raise ValueError("Preparation policy differs from declared protocol")
    budget = index["selection"]["max_sessions_per_participant"]
    if type(budget) is not int or not 1 <= budget <= 6:
        raise ValueError("Unexpected per-participant selection budget")
    receipts = {}
    expected_paths = {f"acquisition/{p}.json" for p in PARTICIPANTS}
    if {r["path"] for r in index["acquisition_sources"]} != expected_paths:
        raise ValueError("Expected all nine acquisition provenance files")
    for participant in PARTICIPANTS:
        receipt = json.loads(_safe_path(cache, f"acquisition/{participant}.json").read_text())
        validate_receipt(receipt, participant)
        if receipt["requested_sessions"] != budget:
            raise ValueError("Index/receipt selection budget mismatch")
        receipts[participant] = receipt
    identities = set()
    for participant in PARTICIPANTS:
        records = [s for s in index["sessions"] if s["participant_id"] == participant]
        if not 1 <= len(records) <= budget:
            raise ValueError("Missing/over-budget participant sessions")
        selected_by_id = {r["session_id"]: r for r in receipts[participant]["selected"]}
        if {r["session_id"] for r in records} != set(selected_by_id):
            raise ValueError("Index omitted or replaced an archive-selected session")
        if records != sorted(records, key=lambda s: (s["post_implant_day"], s["session_id"])):
            raise ValueError("Sessions must be chronological after archive-order selection")
        for record in records:
            session_id = record["session_id"]
            if session_id in identities or not re.fullmatch(
                rf"{participant}_day_{record['post_implant_day']}_decoding", session_id
            ):
                raise ValueError("Duplicate or mismapped session identity")
            identities.add(session_id)
            source = record["source"]
            if (
                source["archive_file_id"] != PINS[participant][0]
                or source["archive_sha256_published"] != PINS[participant][2]
                or source["participant_id"] != participant
                or source["post_implant_day"] != record["post_implant_day"]
                or source["session_id"] != session_id
            ):
                raise ValueError("Session provider identity mismatch")
            if any(source.get(k) != v for k, v in selected_by_id[session_id].items()):
                raise ValueError("Index source differs from acquisition receipt")
            if (
                source["full_archive_sha256_verified"]
                != receipts[participant]["full_archive_sha256_verified"]
            ):
                raise ValueError("Index/archive verification status mismatch")
            if (
                type(record["post_implant_day"]) is not int
                or type(record["n_bins"]) is not int
                or record["n_bins"] < 0
            ):
                raise ValueError("Invalid integer session geometry")
            if record["bin_seconds"] != BIN_SECONDS or record["n_channels"] not in (96, 192):
                raise ValueError("Invalid feature geometry")
            boundary = 0
            for number, segment in enumerate(record["segments"]):
                if (
                    segment["start"] != boundary
                    or segment["end"] < boundary
                    or segment["sequence_id"] != number
                ):
                    raise ValueError("Segments must partition the prepared chronology")
                boundary = segment["end"]
            if boundary != record["n_bins"]:
                raise ValueError("Segments do not cover prepared stream")
            if verify_hashes:
                for key in ("feature", "target"):
                    artifact = _safe_path(cache, record[f"{key}_path"])
                    if sha256_file(artifact) != record[f"{key}_sha256"]:
                        raise ValueError(f"{key} artifact hash mismatch")
                if sha256_file(_safe_path(cache, source["raw_path"])) != source["member_sha256"]:
                    raise ValueError("Raw member hash mismatch")
    if len(identities) != len(index["sessions"]):
        raise ValueError("Unexpected participant records")
    if verify_hashes:
        if sha256_file(_safe_path(cache, provider["manifest_path"])) != provider["manifest_sha256"]:
            raise ValueError("Provider manifest hash mismatch")
        for receipt in index["acquisition_sources"]:
            if sha256_file(_safe_path(cache, receipt["path"])) != receipt["sha256"]:
                raise ValueError("Acquisition receipt hash mismatch")
    return index


def load_features(path: str | Path) -> dict[str, np.ndarray]:
    """The inference-side loader opens no target/go/trial side file."""
    with np.load(path, allow_pickle=False) as data:
        if frozenset(data.files) != FEATURE_KEYS:
            raise ValueError("Feature file contains missing or forbidden fields")
        arrays = {k: data[k] for k in data.files}
    x, timestamp, sequence, channels = (
        arrays[k] for k in ("x", "timestamps", "sequence_id", "channel_ids")
    )
    if (
        x.dtype != np.float32
        or x.ndim != 2
        or x.shape[1] not in (96, 192)
        or not np.isfinite(x).all()
        or np.any(x < 0)
    ):
        raise ValueError("Invalid prepared counts")
    if (
        timestamp.shape != (len(x),)
        or not np.isfinite(timestamp).all()
        or np.any(np.diff(timestamp) <= 0)
    ):
        raise ValueError("Invalid timestamp chronology")
    if (
        sequence.shape != (len(x),)
        or not np.issubdtype(sequence.dtype, np.integer)
        or np.any(sequence < 0)
        or np.any(np.diff(sequence) < 0)
    ):
        raise ValueError("Invalid acquisition sequence IDs")
    if (
        channels.shape != (x.shape[1],)
        or not np.issubdtype(channels.dtype, np.integer)
        or np.any(np.diff(channels) <= 0)
    ):
        raise ValueError("Invalid channel order")
    same = np.diff(sequence) == 0
    if np.any(~np.isclose(np.diff(timestamp)[same], BIN_SECONDS, rtol=0, atol=1e-9)):
        raise ValueError("Unmarked time gap within acquisition sequence")
    return arrays


def load_targets(path: str | Path, n_bins: int | None = None) -> dict[str, np.ndarray]:
    """Training/scorer-only loader; never call this from a prediction function."""
    with np.load(path, allow_pickle=False) as data:
        if frozenset(data.files) != TARGET_KEYS:
            raise ValueError("Unexpected target fields")
        arrays = {k: data[k] for k in data.files}
    y, mask = arrays["y"], arrays["eval_mask"]
    if (
        y.dtype != np.float32
        or y.ndim != 2
        or y.shape[1] != 2
        or not np.isfinite(y).all()
        or mask.dtype != np.bool_
        or mask.shape != (len(y),)
        or (n_bins is not None and len(y) != n_bins)
    ):
        raise ValueError("Invalid scorer target geometry")
    if mask.any() and not np.allclose(np.linalg.norm(y[mask], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Scored target is not unit intended direction")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["metadata", "acquire", "prepare", "validate"])
    parser.add_argument("--cache", type=Path, default=Path("data/neural_v2"))
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--participants", nargs="+", choices=PARTICIPANTS, default=PARTICIPANTS)
    parser.add_argument("--archive-directory", type=Path)
    parser.add_argument(
        "--url-file",
        type=Path,
        help="Private JSON participant-to-URL map observed from normal public redirects",
    )
    parser.add_argument("--max-stream-gib", type=float, default=4)
    parser.add_argument("--max-cache-gib", type=float, default=20)
    args = parser.parse_args()
    if args.command == "metadata":
        result = pin_metadata(args.cache)
        print(json.dumps({"version_id": result["version_id"], "archives": len(result["archives"])}))
    elif args.command == "acquire":
        acquire(
            args.cache,
            sessions=args.sessions,
            participants=tuple(args.participants),
            archive_directory=args.archive_directory,
            url_file=args.url_file,
            max_stream_bytes=int(args.max_stream_gib * 1024**3),
            max_cache_bytes=int(args.max_cache_gib * 1024**3),
        )
    else:
        result = (
            prepare(args.cache, max_cache_bytes=int(args.max_cache_gib * 1024**3))
            if args.command == "prepare"
            else load_index(args.cache / "index.json")
        )
        print(
            json.dumps(
                {
                    "participants": result["participants"],
                    "sessions": len(result["sessions"]),
                    "n_bins": sum(s["n_bins"] for s in result["sessions"]),
                }
            )
        )


if __name__ == "__main__":
    main()
