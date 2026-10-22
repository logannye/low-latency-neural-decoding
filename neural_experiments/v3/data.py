"""Separate V3 TX/SBP exports and deterministic next-three-session acquisition.

All neural samples remain private. The existing V2 cache is read-only. Neither
conversion nor metadata acquisition computes model outcomes or normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import parse_qs, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat

from neural_experiments.v2 import data as v2

PARTICIPANTS = v2.PARTICIPANTS
V2_INDEX_SHA256 = "91de850304c4fb96096daa3f9ec19edab0372675c236a899005cf4308ec6c523"
FEATURE_KEYS = frozenset({"x_tx", "x_sbp", "timestamps", "sequence_id", "channel_ids"})
MAX_BINS = 30000
PLAN_SHA256 = "6d0f188bba2c9719bbb8dffece385d48a6b70e07b22835ee6413c89eee063392"
MAX_COMPRESSED_BYTES = 14_000_000_000
MAX_CACHE_BYTES = 18_000_000_000
SELECTION = {
    "rule": "regular_decoding_session_members_7_through_9_in_pinned_archive_order",
    "regular_session_ranks_one_based": [7, 8, 9],
    "scores_or_duration_used_for_selection": False,
    "shortages": "retain_every_selected_session_and_record_archive_end_no_substitution",
    "ordering_after_selection": "numeric_post_implant_day_then_session_id",
}
FEATURE_DEFINITION = {
    "tx": "tx_4_5: sum two consecutive 10ms count bins within each block",
    "sbp": "sbp: arithmetic mean of the same two 10ms power bins, units microvolt_squared",
    "bin_seconds": 0.02,
    "normalization": "none; prefix-only transforms belong to downstream training",
    "upstream_feature_causality": "unverified",
    "export_cap_bins": MAX_BINS,
    "target_definition": v2.TARGET_DEFINITION,
}


def _write(path: Path, value: Any) -> None:
    v2._json(path, value)


def _read_v2(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if v2.sha256_file(path) != V2_INDEX_SHA256:
        raise ValueError("V2 development index identity differs from locked cohort")
    return v2.load_index(path, verify_hashes=False)


def _prior_members(index: dict[str, Any], participant: str) -> list[dict[str, Any]]:
    return sorted(
        [s["source"] for s in index["sessions"] if s["participant_id"] == participant],
        key=lambda s: s["archive_ordinal"],
    )


def _pattern(participant: str) -> str:
    return rf"(?:\./)?decoding/{participant}/{participant}_day_(\d+)_decoding\.mat"


def extract_next_sessions(
    stream: BinaryIO,
    *,
    cache: Path,
    participant: str,
    prior_members: list[dict[str, Any]],
    archive: dict[str, Any],
    max_stream_bytes: int = 4 * 1024**3,
    max_cache_bytes: int = MAX_CACHE_BYTES,
) -> dict[str, Any]:
    """Stream ranks1..9, authenticate old1..6, retain only new7..9."""
    if participant not in PARTICIPANTS or len(prior_members) != 6:
        raise ValueError("Require a pinned participant and exactly six prior members")
    cache = Path(cache)
    raw_dir = cache / "raw" / participant
    raw_dir.mkdir(parents=True, exist_ok=True)
    reader = v2.BoundedReader(stream, max_stream_bytes)
    receipt = {
        "schema_version": 1,
        "participant_id": participant,
        "archive": archive,
        "selection": SELECTION,
        "v2_index_sha256": V2_INDEX_SHA256,
        "plan_sha256": PLAN_SHA256,
        "started_at": v2._now(),
        "status": "running",
        "selected": [],
        "seen_members": [],
        "verified_prior_members": [],
        "tar_end_observed": False,
        "full_archive_sha256_verified": False,
    }
    regular_rank = 0
    known_names = set()
    try:
        with tarfile.open(fileobj=reader, mode="r|gz", bufsize=64 * 1024) as tar:
            for ordinal, member in enumerate(tar):
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                    raise ValueError("Unsafe archive member path")
                if member.issym() or member.islnk():
                    raise ValueError("Archive links are not accepted")
                receipt["seen_members"].append(
                    {
                        "ordinal": ordinal,
                        "name": member.name,
                        "size": member.size,
                        "regular_file": member.isfile(),
                    }
                )
                if not member.isfile():
                    continue
                match = re.fullmatch(_pattern(participant), member.name)
                if not match:
                    if member.name.endswith(".mat") and not name.name.startswith("._"):
                        raise ValueError("Unexpected regular MAT member")
                    continue
                if name.stem in known_names:
                    raise ValueError("Duplicate regular session identity")
                known_names.add(name.stem)
                regular_rank += 1
                previous = prior_members[regular_rank - 1] if regular_rank <= 6 else None
                if previous is not None and (
                    previous["member_name"] != member.name
                    or previous["member_size"] != member.size
                    or previous["archive_ordinal"] != ordinal
                ):
                    raise ValueError("First six archive members differ from locked V2 selection")
                if member.size > 2 * 1024**3:
                    raise ValueError("Selected member exceeds 2GiB safety bound")
                target = raw_dir / name.name
                temporary = target.with_suffix(".mat.part")
                if previous is None:
                    existing = target.stat().st_size if target.exists() else 0
                    if v2._cache_size(cache) + member.size - existing > max_cache_bytes:
                        raise ValueError("V3 cache disk budget exceeded")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError("Missing regular member payload")
                output = temporary.open("wb") if previous is None else None
                digest = hashlib.sha256()
                n_bytes = 0
                try:
                    while chunk := extracted.read(1024 * 1024):
                        digest.update(chunk)
                        n_bytes += len(chunk)
                        if output is not None:
                            output.write(chunk)
                finally:
                    if output is not None:
                        output.close()
                if n_bytes != member.size:
                    raise ValueError("Truncated complete member")
                identity = {
                    "participant_id": participant,
                    "session_id": name.stem,
                    "post_implant_day": int(match.group(1)),
                    "archive_ordinal": ordinal,
                    "regular_session_rank": regular_rank,
                    "member_name": member.name,
                    "member_size": member.size,
                    "member_sha256": digest.hexdigest(),
                }
                if previous is not None:
                    if digest.hexdigest() != previous["member_sha256"]:
                        raise ValueError("Previously acquired V2 member hash changed")
                    receipt["verified_prior_members"].append(identity)
                else:
                    if target.exists() and v2.sha256_file(target) != digest.hexdigest():
                        raise ValueError("Existing V3 raw member identity changed")
                    temporary.replace(target)
                    identity["raw_path"] = str(target.relative_to(cache))
                    receipt["selected"].append(identity)
                if regular_rank == 9:
                    break
            else:
                receipt["tar_end_observed"] = True
        if regular_rank < 6:
            raise ValueError("Archive ended before the authenticated V2 prefix")
        receipt["status"] = "complete"
        receipt["missing_requested_ranks"] = list(range(max(7, regular_rank + 1), 10))
    except Exception as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt.update(
            {
                "finished_at": v2._now(),
                "compressed_prefix_bytes": reader.count,
                "compressed_prefix_sha256": reader.digest.hexdigest(),
                "verification_limit": (
                    "Local hashes authenticate retained new members and old prefix members; "
                    "the provider full-archive digest is not verified by partial streaming."
                ),
            }
        )
        _write(cache / "acquisition" / f"{participant}.json", receipt)
    return receipt


def validate_receipt(receipt: dict[str, Any], participant: str, prior_members=None) -> None:
    if (
        receipt.get("status") != "complete"
        or receipt.get("participant_id") != participant
        or receipt.get("selection") != SELECTION
        or receipt.get("v2_index_sha256") != V2_INDEX_SHA256
        or receipt.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("Invalid V3 receipt role/selection")
    file_id, size, digest = v2.PINS[participant]
    archive = receipt["archive"]
    if (archive.get("file_id"), archive.get("size"), archive.get("provider_sha256")) != (
        file_id,
        size,
        digest,
    ):
        raise ValueError("Provider archive identity mismatch")
    regular = [
        r
        for r in receipt["seen_members"]
        if r["regular_file"] and re.fullmatch(_pattern(participant), r["name"])
    ]
    if len(regular) < 6 or len(regular) > 9:
        raise ValueError("Invalid archive regular-member inventory")
    selected = receipt["selected"]
    prior = receipt["verified_prior_members"]
    if len(prior) != 6 or len(selected) != len(regular[6:9]):
        raise ValueError("Missing authenticated prefix or selected rank")
    if len(selected) < 3 and not receipt["tar_end_observed"]:
        raise ValueError("Short acquisition requires observed archive end")
    identities = set()
    for rank, (row, seen) in enumerate(zip(prior + selected, regular, strict=True), 1):
        match = re.fullmatch(_pattern(participant), row["member_name"])
        if (
            match is None
            or row["participant_id"] != participant
            or row["regular_session_rank"] != rank
            or row["post_implant_day"] != int(match.group(1))
            or row["session_id"] != PurePosixPath(row["member_name"]).stem
            or row["member_name"] != seen["name"]
            or row["member_size"] != seen["size"]
            or row["archive_ordinal"] != seen["ordinal"]
            or re.fullmatch(r"[0-9a-f]{64}", row["member_sha256"]) is None
            or row["session_id"] in identities
        ):
            raise ValueError("Member provenance/rank mismatch")
        identities.add(row["session_id"])
        if rank > 6 and row["raw_path"] != f"raw/{participant}/{row['session_id']}.mat":
            raise ValueError("Unsafe raw-member path")
        if rank <= 6 and prior_members is not None:
            expected = prior_members[rank - 1]
            if any(
                row[k] != expected[k]
                for k in [
                    "member_name",
                    "member_size",
                    "member_sha256",
                    "archive_ordinal",
                    "session_id",
                ]
            ):
                raise ValueError("Receipt authenticated prefix differs from V2")
    expected_missing = list(range(7 + len(selected), 10))
    if receipt.get("missing_requested_ranks") != expected_missing:
        raise ValueError("Incorrect shortage accounting")


def acquire(
    cache,
    v2_index,
    *,
    participants=PARTICIPANTS,
    url_file=None,
    max_stream_bytes=4 * 1024**3,
    max_total_stream_bytes=MAX_COMPRESSED_BYTES,
    max_cache_bytes=MAX_CACHE_BYTES,
):
    cache = Path(cache)
    old = _read_v2(v2_index)
    plan_path = Path(__file__).with_name("plan.json")
    if v2.sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("Committed assessment selection plan changed")
    if max_total_stream_bytes > MAX_COMPRESSED_BYTES or max_cache_bytes > MAX_CACHE_BYTES:
        raise ValueError("Acquisition exceeds locked network/storage bound")
    manifest = v2.pin_metadata(cache)
    observed = json.loads(Path(url_file).read_text()) if url_file is not None else {}
    for participant in participants:
        if participant not in PARTICIPANTS:
            raise ValueError("Only pinned participants are allowed")
        path = cache / "acquisition" / f"{participant}.json"
        prior = _prior_members(old, participant)
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt["status"] == "complete":
                validate_receipt(receipt, participant, prior)
                for s in receipt["selected"]:
                    if v2.sha256_file(v2._safe_path(cache, s["raw_path"])) != s["member_sha256"]:
                        raise ValueError("Cached V3 member hash changed")
                continue
            _write(cache / "failures" / f"{participant}-{v2._now().replace(':', '')}.json", receipt)
        consumed = sum(
            json.loads(p.read_text()).get("compressed_prefix_bytes", 0)
            for p in (cache / "acquisition").glob("*.json")
        )
        remaining = min(max_stream_bytes, max_total_stream_bytes - consumed)
        if remaining <= 0:
            raise ValueError("Aggregate V3 network budget exhausted")
        archive = next(a for a in manifest["archives"] if a["participant_id"] == participant)
        url = archive["download_url"]
        if url_file is not None:
            url = observed[participant]
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.username
                or parsed.password
                or parsed.hostname != "dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com"
                or parse_qs(parsed.query).get("response-content-disposition")
                != [f"attachment; filename={archive['filename']}"]
            ):
                raise ValueError("Expected matching observed public Dryad redirect")
            archive = {
                **archive,
                "transport": {
                    "mode": "browser_observed_public_redirect",
                    "object_url": urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
                    "private_url_file_sha256": v2.sha256_file(url_file),
                },
            }
        try:
            request = Request(
                url, headers={"User-Agent": "low-latency-neural-decoding/research", "Accept": "*/*"}
            )
            with urlopen(request, timeout=90) as response:
                if response.headers.get("Content-Length") != str(archive["size"]):
                    raise ValueError("Public response length differs from pinned archive")
                receipt = extract_next_sessions(
                    response,
                    cache=cache,
                    participant=participant,
                    prior_members=prior,
                    archive=archive,
                    max_stream_bytes=remaining,
                    max_cache_bytes=max_cache_bytes,
                )
            validate_receipt(receipt, participant, prior)
            print(
                json.dumps(
                    {
                        "participant": participant,
                        "selected": len(receipt["selected"]),
                        "compressed_prefix_bytes": receipt["compressed_prefix_bytes"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            _write(
                cache / "failures" / f"{participant}-{v2._now().replace(':', '')}.json",
                {
                    "participant_id": participant,
                    "archive": archive,
                    "type": type(error).__name__,
                    "message": str(error),
                    "status": "failed_no_substitution",
                },
            )
            raise


def _cap_raw(raw: dict[str, Any], cap: int) -> tuple[dict[str, Any], dict[str, Any]]:
    blocks = v2._struct_vectors(
        raw["blocks"], ("block_id", "sample_start_index", "sample_end_index")
    )
    tx = np.asarray(raw["neural"]["tx_4_5"])
    sbp = np.asarray(raw["neural"]["sbp"])
    if (
        tx.ndim != 2
        or sbp.shape != tx.shape
        or np.asarray(raw["cursor_position"]).shape != (len(tx), 2)
        or np.asarray(raw["target_position"]).shape != (len(tx), 2)
    ):
        raise ValueError("Full raw feature/behavior geometry mismatch")
    starts, ends = blocks["sample_start_index"], blocks["sample_end_index"]
    if (
        len(starts) == 0
        or np.any(starts < 0)
        or np.any(ends > len(tx))
        or np.any(ends <= starts)
        or np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError("Full raw block geometry invalid")
    count = 0
    source_stop = 0
    keep = 0
    for start, stop in zip(blocks["sample_start_index"], blocks["sample_end_index"], strict=True):
        pairs = (int(stop) - int(start)) // 2
        take = min(pairs, cap - count)
        if take <= 0 and count >= cap:
            break
        source_stop = int(stop) if take == pairs else int(start) + 2 * take
        count += take
        keep += 1
        if count >= cap:
            break
    result = dict(raw)
    result["blocks"] = {k: v[:keep].copy() for k, v in blocks.items()}
    if keep:
        result["blocks"]["sample_end_index"][-1] = source_stop
    result["cursor_position"] = np.asarray(raw["cursor_position"])[:source_stop]
    result["target_position"] = np.asarray(raw["target_position"])[:source_stop]
    result["neural"] = {k: np.asarray(raw["neural"][k])[:source_stop] for k in ["tx_4_5", "sbp"]}
    trials = v2._struct_vectors(raw["trials"], ("trial_start_index", "trial_end_index", "block_id"))
    valid = trials["trial_start_index"] < source_stop
    result["trials"] = {k: v[valid].copy() for k, v in trials.items()}
    result["trials"]["trial_end_index"] = np.minimum(
        result["trials"]["trial_end_index"], source_stop
    )
    full = {
        "full_raw_n_source_bins": int(np.asarray(raw["neural"]["tx_4_5"]).shape[0]),
        "full_raw_n_blocks": len(blocks["block_id"]),
        "full_prepared_n_bins": int(
            sum(
                (b - a) // 2
                for a, b in zip(
                    blocks["sample_start_index"], blocks["sample_end_index"], strict=True
                )
            )
        ),
        "full_raw_bin_seconds": float(raw["bin_size"]),
    }
    return result, full


def convert_session(raw, participant, day, *, cap_bins=MAX_BINS):
    if type(cap_bins) is not int or not 1 <= cap_bins <= MAX_BINS:
        raise ValueError("V3 export cap must be in1..30000 bins")
    raw, full = _cap_raw(raw, cap_bins)
    features, targets, metadata = v2.convert_session(raw, participant, day)
    sbp = np.asarray(raw["neural"]["sbp"])
    if (
        sbp.shape != np.asarray(raw["neural"]["tx_4_5"]).shape
        or not np.isfinite(sbp).all()
        or np.any(sbp < 0)
    ):
        raise ValueError("SBP must be aligned finite nonnegative power")
    powers = []
    for segment in metadata["segments"]:
        start, pairs = segment["source_start"], segment["end"] - segment["start"]
        paired = sbp[start : start + 2 * pairs].astype(np.float64).reshape(pairs, 2, sbp.shape[1])
        powers.append(paired.mean(axis=1).astype(np.float32))
    features["x_tx"] = features.pop("x")
    features["x_sbp"] = np.concatenate(powers)
    if not np.isfinite(features["x_sbp"]).all():
        raise ValueError("SBP overflows float32")
    metadata.update(full)
    metadata["feature_definition"] = FEATURE_DEFINITION
    metadata["export_cap_bins"] = cap_bins
    return features, targets, metadata


def prepare(cache, v2_index, *, role, max_cache_bytes=MAX_CACHE_BYTES):
    if role not in {"development", "assessment"}:
        raise ValueError("Explicit development/assessment role required")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "index.json").exists():
        raise ValueError("Prepared V3 index already exists; do not overwrite an active input")
    old = _read_v2(v2_index)
    sources = []
    acquisition = []
    dispositions = []
    if role == "development":
        for session in old["sessions"]:
            sources.append((session["source"], Path(v2_index).resolve().parent))
    else:
        for participant in PARTICIPANTS:
            path = cache / "acquisition" / f"{participant}.json"
            receipt = json.loads(path.read_text())
            validate_receipt(receipt, participant, _prior_members(old, participant))
            acquisition.append(
                {"path": str(path.relative_to(cache)), "sha256": v2.sha256_file(path)}
            )
            dispositions.append(
                {
                    "participant_id": participant,
                    "selected_sessions": len(receipt["selected"]),
                    "missing_requested_ranks": receipt["missing_requested_ranks"],
                }
            )
            for source in receipt["selected"]:
                sources.append(
                    (
                        {
                            **source,
                            "archive_file_id": receipt["archive"]["file_id"],
                            "archive_sha256_published": receipt["archive"]["provider_sha256"],
                            "full_archive_sha256_verified": False,
                        },
                        cache,
                    )
                )
    sessions = []
    for source, raw_root in sources:
        raw_path = v2._safe_path(raw_root, source["raw_path"])
        if v2.sha256_file(raw_path) != source["member_sha256"]:
            raise ValueError("Raw source member hash mismatch")
        raw = loadmat(raw_path, simplify_cells=True)
        features, targets, metadata = convert_session(
            raw, source["participant_id"], source["post_implant_day"]
        )
        del raw
        folder = cache / "prepared" / source["participant_id"]
        folder.mkdir(parents=True, exist_ok=True)
        feature = folder / f"{source['session_id']}.features.npz"
        target = folder / f"{source['session_id']}.targets.npz"
        estimate = sum(v.nbytes for v in [*features.values(), *targets.values()]) + 65536
        if v2._cache_size(cache) + estimate > max_cache_bytes:
            raise ValueError("Prepared V3 data would exceed cache bound")
        np.savez_compressed(feature, **features)
        np.savez_compressed(target, **targets)
        sessions.append(
            {
                "participant_id": source["participant_id"],
                "session_id": source["session_id"],
                "post_implant_day": source["post_implant_day"],
                "role": role,
                "source": source,
                "raw_source_root": str(raw_root.resolve()),
                "feature_path": str(feature.relative_to(cache)),
                "target_path": str(target.relative_to(cache)),
                "feature_sha256": v2.sha256_file(feature),
                "target_sha256": v2.sha256_file(target),
                **metadata,
            }
        )
        print(
            json.dumps(
                {"prepared": source["session_id"], "role": role, "n_bins": metadata["n_bins"]}
            ),
            flush=True,
        )
    order = {p: i for i, p in enumerate(PARTICIPANTS)}
    sessions.sort(
        key=lambda s: (order[s["participant_id"]], s["post_implant_day"], s["session_id"])
    )
    result = {
        "schema_version": 1,
        "role": role,
        "created_at": v2._now(),
        "participants": list(PARTICIPANTS),
        "selection": SELECTION
        if role == "assessment"
        else {"rule": "exact_frozen_V2_sessions_first_min600s"},
        "v2_index_path": str(Path(v2_index).resolve()),
        "v2_index_sha256": V2_INDEX_SHA256,
        "plan_sha256": PLAN_SHA256,
        "provider": {"doi": v2.DOI, "version_id": v2.VERSION_ID},
        "feature_definition": FEATURE_DEFINITION,
        "preparer": {
            "v3_sha256": v2.sha256_file(__file__),
            "v2_dependency_sha256": v2.sha256_file(v2.__file__),
        },
        "acquisition_sources": acquisition,
        "participant_dispositions": dispositions,
        "sessions": sessions,
    }
    _write(cache / "index.json", result)
    return result


def load_features(path):
    with np.load(path, allow_pickle=False) as z:
        if frozenset(z.files) != FEATURE_KEYS:
            raise ValueError("Feature file contains missing or forbidden fields")
        a = {k: z[k] for k in z.files}
    x, sbp = a["x_tx"], a["x_sbp"]
    if (
        x.dtype != np.float32
        or x.ndim != 2
        or x.shape[1] not in (96, 192)
        or sbp.dtype != np.float32
        or sbp.shape != x.shape
        or not np.isfinite(x).all()
        or not np.isfinite(sbp).all()
        or np.any(x < 0)
        or np.any(sbp < 0)
        or not np.equal(x, np.floor(x)).all()
    ):
        raise ValueError("Invalid V3 neural count/power arrays")
    t, q, c = a["timestamps"], a["sequence_id"], a["channel_ids"]
    if (
        t.dtype != np.float64
        or t.shape != (len(x),)
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
        or q.shape != (len(x),)
        or not np.issubdtype(q.dtype, np.integer)
        or np.any(q < 0)
        or np.any(np.diff(q) < 0)
        or c.shape != (x.shape[1],)
        or not np.issubdtype(c.dtype, np.integer)
        or np.any(np.diff(c) <= 0)
    ):
        raise ValueError("Invalid V3 timestamp/sequence/channel metadata")
    if np.any(~np.isclose(np.diff(t)[np.diff(q) == 0], 0.02, rtol=0, atol=1e-9)):
        raise ValueError("Unmarked within-block time gap")
    return a


def load_index(path, *, expected_role=None, verify_hashes=True):
    path = Path(path)
    cache = path.resolve().parent
    index = json.loads(path.read_text())
    role = index.get("role")
    if role not in {"development", "assessment"} or (
        expected_role is not None and role != expected_role
    ):
        raise ValueError("V3 index role mismatch")
    if (
        index.get("schema_version") != 1
        or index.get("participants") != list(PARTICIPANTS)
        or index.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
        or index.get("feature_definition") != FEATURE_DEFINITION
        or index.get("v2_index_sha256") != V2_INDEX_SHA256
        or index.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("V3 index provider/feature identity mismatch")
    old = _read_v2(index["v2_index_path"])
    expected = {}
    if role == "development":
        if index.get("selection") != {"rule": "exact_frozen_V2_sessions_first_min600s"}:
            raise ValueError("Development selection mismatch")
        expected = {s["session_id"]: s["source"] for s in old["sessions"]}
    else:
        if index.get("selection") != SELECTION:
            raise ValueError("Assessment selection mismatch")
        if len(index.get("acquisition_sources", [])) != len(PARTICIPANTS):
            raise ValueError("Missing participant acquisition receipts")
        for participant in PARTICIPANTS:
            relative = f"acquisition/{participant}.json"
            refs = [r for r in index["acquisition_sources"] if r["path"] == relative]
            if len(refs) != 1:
                raise ValueError("Missing or duplicate participant receipt")
            receipt_path = v2._safe_path(cache, relative)
            if verify_hashes and v2.sha256_file(receipt_path) != refs[0]["sha256"]:
                raise ValueError("Acquisition receipt hash mismatch")
            receipt = json.loads(receipt_path.read_text())
            validate_receipt(receipt, participant, _prior_members(old, participant))
            for s in receipt["selected"]:
                expected[s["session_id"]] = s
    if len(index["sessions"]) != len(expected) or {
        s["session_id"] for s in index["sessions"]
    } != set(expected):
        raise ValueError("V3 omitted/replaced/duplicated selected session")
    previous = {}
    for s in index["sessions"]:
        source = s["source"]
        p = s["participant_id"]
        if s.get("role") != role or any(
            source.get(k) != v for k, v in expected[s["session_id"]].items()
        ):
            raise ValueError("V3 session source/role mismatch")
        if (
            s["session_id"] != source["session_id"]
            or p != source["participant_id"]
            or s["post_implant_day"] != source["post_implant_day"]
            or type(s["n_bins"]) is not int
            or not 0 <= s["n_bins"] <= MAX_BINS
            or s["n_bins"] != min(s["full_prepared_n_bins"], MAX_BINS)
            or s["n_channels"] not in (96, 192)
            or s["bin_seconds"] != 0.02
        ):
            raise ValueError("V3 session geometry/identity mismatch")
        key = (s["post_implant_day"], s["session_id"])
        if p in previous and key <= previous[p]:
            raise ValueError("V3 sessions not chronological")
        previous[p] = key
        end = 0
        for number, segment in enumerate(s["segments"]):
            if segment["start"] != end or segment["end"] < end or segment["sequence_id"] != number:
                raise ValueError("V3 segments do not partition observations")
            end = segment["end"]
        if end != s["n_bins"]:
            raise ValueError("V3 segments do not cover observations")
        raw_root = Path(index["v2_index_path"]).resolve().parent if role == "development" else cache
        if Path(s["raw_source_root"]).resolve() != raw_root:
            raise ValueError("Unexpected raw source root")
        if verify_hashes:
            for k in ["feature", "target"]:
                if v2.sha256_file(v2._safe_path(cache, s[f"{k}_path"])) != s[f"{k}_sha256"]:
                    raise ValueError(f"V3 {k} artifact hash mismatch")
            if (
                v2.sha256_file(v2._safe_path(raw_root, source["raw_path"]))
                != source["member_sha256"]
            ):
                raise ValueError("V3 raw member hash mismatch")
    return index


load_targets = v2.load_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["metadata", "acquire", "prepare-development", "prepare-assessment", "validate"],
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--v2-index", type=Path, default=Path("data/neural_v2/index.json"))
    parser.add_argument("--participants", nargs="+", choices=PARTICIPANTS, default=PARTICIPANTS)
    parser.add_argument("--url-file", type=Path)
    parser.add_argument("--max-stream-gib", type=float, default=4)
    parser.add_argument(
        "--max-total-stream-gib", type=float, default=MAX_COMPRESSED_BYTES / 1024**3
    )
    parser.add_argument("--max-cache-gib", type=float, default=MAX_CACHE_BYTES / 1024**3)
    parser.add_argument("--expected-role", choices=["development", "assessment"])
    args = parser.parse_args()
    if args.command == "metadata":
        result = v2.pin_metadata(args.cache)
        print(json.dumps({"version_id": result["version_id"], "archives": len(result["archives"])}))
    elif args.command == "acquire":
        acquire(
            args.cache,
            args.v2_index,
            participants=tuple(args.participants),
            url_file=args.url_file,
            max_stream_bytes=int(args.max_stream_gib * 1024**3),
            max_total_stream_bytes=int(args.max_total_stream_gib * 1024**3),
            max_cache_bytes=int(args.max_cache_gib * 1024**3),
        )
    elif args.command.startswith("prepare-"):
        result = prepare(
            args.cache,
            args.v2_index,
            role=args.command.removeprefix("prepare-"),
            max_cache_bytes=int(args.max_cache_gib * 1024**3),
        )
        print(json.dumps({"role": result["role"], "sessions": len(result["sessions"])}))
    else:
        result = load_index(args.cache / "index.json", expected_role=args.expected_role)
        for s in result["sessions"]:
            a = load_features(args.cache / s["feature_path"])
            load_targets(args.cache / s["target_path"], len(a["x_tx"]))
        print(
            json.dumps(
                {"validated": True, "role": result["role"], "sessions": len(result["sessions"])}
            )
        )


if __name__ == "__main__":
    main()
