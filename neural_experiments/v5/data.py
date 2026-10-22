"""V5 bounded three-feature exports with separate new-session and T3 continuation roles.

All neural samples remain private. The existing V2 cache is read-only. Neither
conversion nor metadata acquisition computes model outcomes or normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import parse_qs, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat

from neural_experiments.v2 import data as v2
from neural_experiments.v4 import data as v4

ALL_PARTICIPANTS = v2.PARTICIPANTS
PARTICIPANTS = tuple(p for p in ALL_PARTICIPANTS if p != "T3")
V4_NEW_INDEX_SHA256 = "76293562fd6e586e7d6f5506c41ebda97241c971e18984c22489b69ac29558cc"
V4_DEVELOPMENT_INDEX_SHA256 = "aa6bf303c746edde0fb4d0a3a937a0fb07afb849270f2c79093fc021a9c2c500"
FEATURE_KEYS = frozenset({"x_tx3", "x_tx45", "x_sbp", "timestamps", "sequence_id", "channel_ids"})
MAX_BINS = 30000
PLAN_SHA256 = "43a55022369f44d76659d453e20880009692bfa80591aa49d0b54677cb8c1f47"
MAX_COMPRESSED_BYTES = 20_000_000_000
MAX_CACHE_BYTES = 12_000_000_000
SELECTION = {
    "rule": "regular_decoding_session_members_13_through_15_in_pinned_archive_order",
    "regular_session_ranks_one_based": [13, 14, 15],
    "scores_or_duration_used_for_selection": False,
    "shortages": "retain_every_selected_session_and_record_archive_end_no_substitution",
    "ordering_after_selection": "numeric_post_implant_day_then_session_id",
}
FEATURE_DEFINITION = {
    "tx3": "tx_3: sum two consecutive 10ms count bins within each block",
    "tx45": "tx_4_5: sum two consecutive 10ms count bins within each block",
    "sbp": "sbp: arithmetic mean of the same two 10ms power bins, units microvolt_squared",
    "bin_seconds": 0.02,
    "normalization": "none; prefix-only transforms belong to downstream training",
    "upstream_feature_causality": "unverified",
    "export_cap_bins": MAX_BINS,
    "target_definition": v2.TARGET_DEFINITION,
}


def _write(path: Path, value: Any) -> None:
    v2._json(path, value)


V4_T3_INDEX_SHA256 = "7f8908c54312814ce41b7e743042e85fb890a443ee267f8fed937ae66b27de69"
V4_T3_INDEX = Path("data/neural_v4/t3_blocks/index.json")


def _read_parents(v4_development_index, v4_new_index):
    records = []
    for path, digest, role in [
        (v4_development_index, V4_DEVELOPMENT_INDEX_SHA256, "development"),
        (v4_new_index, V4_NEW_INDEX_SHA256, "new_sessions"),
    ]:
        path = Path(path)
        if v2.sha256_file(path) != digest:
            raise ValueError("V4 parent index differs from the locked exposed cohort")
        index = v4.load_index(path, expected_role=role, verify_hashes=False)
        for record in index["sessions"]:
            records.append(
                {
                    **record,
                    "parent_cache": record["raw_source_root"],
                    "prepared_cache": str(path.resolve().parent),
                }
            )
    if len(records) != 104 or len({s["session_id"] for s in records}) != 104:
        raise ValueError("Expected 104 unique previously exposed sessions")
    return {"participants": list(ALL_PARTICIPANTS), "sessions": records}


def _t3_selection(raw, parent_ids=None):
    blocks, lengths, offsets = _block_geometry(raw)
    eligible = np.flatnonzero((offsets >= MAX_BINS) & (lengths > 0))
    if parent_ids is not None and [int(blocks["block_id"][i]) for i in eligible[:2]] != parent_ids:
        raise ValueError("V4 selected T3 blocks differ from raw metadata")
    return eligible[2:4]


def _t3_inventory(old, parent_path):
    parent_path = Path(parent_path)
    if v2.sha256_file(parent_path) != V4_T3_INDEX_SHA256:
        raise ValueError("V4 continuation parent identity mismatch")
    parent = json.loads(parent_path.read_text())
    rows = {s["session_id"]: s for s in parent["sessions"]}
    retained, unavailable = {}, []
    for s in old["sessions"]:
        if s["participant_id"] != "T3":
            continue
        raw = loadmat(
            v2._safe_path(Path(s["parent_cache"]), s["source"]["raw_path"]),
            simplify_cells=True,
            variable_names=["blocks"],
        )
        chosen = _t3_selection(raw, rows[s["session_id"]]["selected_evaluation_block_ids"])
        if len(chosen):
            retained[s["session_id"]] = s
        else:
            unavailable.append(
                {
                    "participant_id": "T3",
                    "session_id": s["session_id"],
                    "reason": "no_remaining_complete_block_after_V4_selection",
                    "requested_continuation_blocks": 2,
                    "available_continuation_blocks": 0,
                }
            )
    return retained, unavailable


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
    """Stream ranks1..15, authenticate old1..12, retain only new13..15."""
    if participant not in PARTICIPANTS or len(prior_members) != 12:
        raise ValueError("Require a pinned participant and exactly twelve prior members")
    cache = Path(cache)
    raw_dir = cache / "raw" / participant
    raw_dir.mkdir(parents=True, exist_ok=True)
    reader = v2.BoundedReader(stream, max_stream_bytes)
    receipt = {
        "schema_version": 1,
        "participant_id": participant,
        "archive": archive,
        "selection": SELECTION,
        "v4_development_index_sha256": V4_DEVELOPMENT_INDEX_SHA256,
        "v4_new_index_sha256": V4_NEW_INDEX_SHA256,
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
                previous = prior_members[regular_rank - 1] if regular_rank <= 12 else None
                if previous is not None and (
                    previous["member_name"] != member.name
                    or previous["member_size"] != member.size
                    or previous["archive_ordinal"] != ordinal
                ):
                    raise ValueError("First twelve archive members differ from locked V4 selection")
                if member.size > 2 * 1024**3:
                    raise ValueError("Selected member exceeds 2GiB safety bound")
                target = raw_dir / name.name
                temporary = target.with_suffix(".mat.part")
                if previous is None:
                    existing = target.stat().st_size if target.exists() else 0
                    if v2._cache_size(cache) + member.size - existing > max_cache_bytes:
                        raise ValueError("V5 cache disk budget exceeded")
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
                        raise ValueError("Existing V5 raw member identity changed")
                    temporary.replace(target)
                    identity["raw_path"] = str(target.relative_to(cache))
                    receipt["selected"].append(identity)
                if regular_rank == 15:
                    break
            else:
                receipt["tar_end_observed"] = True
        if regular_rank < 12:
            raise ValueError("Archive ended before the authenticated V2 prefix")
        receipt["status"] = "complete"
        receipt["missing_requested_ranks"] = list(range(max(13, regular_rank + 1), 16))
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
        or receipt.get("v4_development_index_sha256") != V4_DEVELOPMENT_INDEX_SHA256
        or receipt.get("v4_new_index_sha256") != V4_NEW_INDEX_SHA256
        or receipt.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("Invalid V5 receipt role/selection")
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
    if len(regular) < 12 or len(regular) > 15:
        raise ValueError("Invalid archive regular-member inventory")
    selected = receipt["selected"]
    prior = receipt["verified_prior_members"]
    if len(prior) != 12 or len(selected) != len(regular[12:15]):
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
        if rank > 12 and row["raw_path"] != f"raw/{participant}/{row['session_id']}.mat":
            raise ValueError("Unsafe raw-member path")
        if rank <= 12 and prior_members is not None:
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
    expected_missing = list(range(13 + len(selected), 16))
    if receipt.get("missing_requested_ranks") != expected_missing:
        raise ValueError("Incorrect shortage accounting")


def acquire(
    cache,
    v4_development_index,
    v4_new_index,
    *,
    participants=PARTICIPANTS,
    url_file=None,
    max_stream_bytes=4 * 1024**3,
    max_total_stream_bytes=MAX_COMPRESSED_BYTES,
    max_cache_bytes=MAX_CACHE_BYTES,
):
    cache = Path(cache)
    old = _read_parents(v4_development_index, v4_new_index)
    plan_path = Path(__file__).with_name("acquisition-plan.json")
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
                        raise ValueError("Cached V5 member hash changed")
                continue
            _write(cache / "failures" / f"{participant}-{v2._now().replace(':', '')}.json", receipt)
        attempts = {}
        for receipt_file in [
            *(cache / "acquisition").glob("*.json"),
            *(cache / "failures").glob("*.json"),
        ]:
            previous = json.loads(receipt_file.read_text())
            if "compressed_prefix_bytes" in previous:
                identity = (previous["participant_id"], previous["started_at"])
                attempts[identity] = max(
                    attempts.get(identity, 0), previous["compressed_prefix_bytes"]
                )
        consumed = sum(attempts.values())
        remaining = min(max_stream_bytes, max_total_stream_bytes - consumed)
        if remaining <= 0:
            raise ValueError("Aggregate V5 network budget exhausted")
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


ROLES = frozenset({"development", "new_sessions", "t3_blocks"})


def _block_geometry(raw):
    blocks = v2._struct_vectors(
        raw["blocks"], ("block_id", "sample_start_index", "sample_end_index")
    )
    a, b, ids = (blocks[k] for k in ("sample_start_index", "sample_end_index", "block_id"))
    if (
        not len(a)
        or np.any(a < 0)
        or np.any(b <= a)
        or np.any(a[1:] < b[:-1])
        or len(set(ids.tolist())) != len(ids)
    ):
        raise ValueError("Invalid full block geometry")
    lengths = (b - a) // 2
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    return blocks, lengths, offsets


def convert_session(raw, participant, day, *, role="development", cap_bins=MAX_BINS):
    """Export bounded observations; T3 labels exist only in selected fresh blocks.

    T3 features include the causal history from session start. Its known-prefix
    labels are referenced separately from the development export by prepare().
    """
    if role not in ROLES or type(cap_bins) is not int or not 1 <= cap_bins <= MAX_BINS:
        raise ValueError("Invalid role or export cap")
    if role == "t3_blocks" and participant != "T3":
        raise ValueError("Continuation blocks are restricted to T3")
    blocks, lengths, offsets = _block_geometry(raw)
    required = ("tx_3", "tx_4_5", "sbp")
    neural = {k: np.asarray(raw["neural"][k]) for k in required}
    shape = neural["tx_4_5"].shape
    if (
        len(shape) != 2
        or any(x.shape != shape for x in neural.values())
        or np.asarray(raw["cursor_position"]).shape != (shape[0], 2)
        or np.asarray(raw["target_position"]).shape != (shape[0], 2)
        or np.any(blocks["sample_end_index"] > shape[0])
    ):
        raise ValueError("Full feature and behavior geometry mismatch")
    if role == "t3_blocks":
        score_blocks = _t3_selection(raw).tolist()
        if not score_blocks:
            raise ValueError("No complete T3 continuation block; retain unavailable disposition")
        takes = [(i, int(blocks["sample_end_index"][i])) for i in range(score_blocks[-1] + 1)]
    else:
        score_blocks = []
        takes = []
        used = 0
        for i, pairs in enumerate(lengths):
            take = min(int(pairs), cap_bins - used)
            if used >= cap_bins:
                break
            stop = (
                int(blocks["sample_end_index"][i])
                if take == pairs
                else int(blocks["sample_start_index"][i]) + 2 * take
            )
            takes.append((i, stop))
            used += take
        score_blocks = [i for i, _ in takes]
    compact = {k: raw[k] for k in ("participant_id", "post_implant_day", "bin_size", "electrodes")}
    compact["neural"] = {k: [] for k in required}
    compact["blocks"] = {k: [] for k in ("block_id", "sample_start_index", "sample_end_index")}
    compact["trials"] = {k: [] for k in ("block_id", "trial_start_index", "trial_end_index")}
    trials = v2._struct_vectors(raw["trials"], ("trial_start_index", "trial_end_index", "block_id"))
    cursors, targets, original = [], [], []
    position = 0
    for i, stop in takes:
        start = int(blocks["sample_start_index"][i])
        block_id = int(blocks["block_id"][i])
        n = stop - start
        selected = i in score_blocks
        for k in required:
            compact["neural"][k].append(neural[k][start:stop])
        compact["blocks"]["block_id"].append(block_id)
        compact["blocks"]["sample_start_index"].append(position)
        compact["blocks"]["sample_end_index"].append(position + n)
        if selected:
            cursors.append(np.asarray(raw["cursor_position"])[start:stop])
            targets.append(np.asarray(raw["target_position"])[start:stop])
            for a, b, trial_block in zip(
                trials["trial_start_index"],
                trials["trial_end_index"],
                trials["block_id"],
                strict=True,
            ):
                if int(trial_block) == block_id and int(a) < stop:
                    if not start <= a < b <= int(blocks["sample_end_index"][i]):
                        raise ValueError("Trial crosses its original provider block")
                    compact["trials"]["block_id"].append(block_id)
                    compact["trials"]["trial_start_index"].append(position + int(a) - start)
                    compact["trials"]["trial_end_index"].append(
                        position + min(int(b), stop) - start
                    )
        else:
            # No labels outside the fixed fresh scorer blocks are derived for T3.
            cursors.append(np.zeros((n, 2), dtype=np.float64))
            targets.append(np.zeros((n, 2), dtype=np.float64))
        original.append((i, start, stop, selected))
        position += n
    compact["neural"] = {k: np.concatenate(v) for k, v in compact["neural"].items()}
    compact["cursor_position"] = np.concatenate(cursors)
    compact["target_position"] = np.concatenate(targets)
    for key in ("tx_3", "sbp"):
        x = compact["neural"][key]
        if not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError(f"Invalid {key}: finite nonnegative values required")
        if key == "tx_3" and not np.equal(x, np.floor(x)).all():
            raise ValueError("TX3 counts must be integer-valued")
    features, target, metadata = v2.convert_session(compact, participant, day)
    features["x_tx45"] = features.pop("x")
    for raw_key, output_key, reducer in [("tx_3", "x_tx3", "sum"), ("sbp", "x_sbp", "mean")]:
        parts = []
        for segment in metadata["segments"]:
            start = segment["source_start"]
            pairs = segment["end"] - segment["start"]
            a = (
                compact["neural"][raw_key][start : start + 2 * pairs]
                .astype(np.float64)
                .reshape(pairs, 2, shape[1])
            )
            parts.append(getattr(a, reducer)(axis=1).astype(np.float32))
        features[output_key] = np.concatenate(parts)
        if not np.isfinite(features[output_key]).all():
            raise ValueError("Paired feature overflows float32")
    times = []
    for segment, (i, start, stop, selected) in zip(metadata["segments"], original, strict=True):
        pairs = segment["end"] - segment["start"]
        times.append((np.arange(start + 1, start + 2 * pairs, 2, dtype=np.float64) + 1) * 0.01)
        segment.update(
            source_start=start,
            source_end=stop,
            source_n_bins=stop - start,
            source_gap_before_bins=start - (int(blocks["sample_end_index"][i - 1]) if i else 0),
            global_start=int(offsets[i]),
            global_end=int(offsets[i]) + pairs,
            original_block_index=i,
            selected_for_evaluation=bool(selected),
        )
    features["timestamps"] = np.concatenate(times)
    metadata.update(
        role=role,
        feature_definition=FEATURE_DEFINITION,
        full_raw_n_source_bins=shape[0],
        full_raw_n_blocks=len(lengths),
        full_prepared_n_bins=int(lengths.sum()),
        full_raw_bin_seconds=float(raw["bin_size"]),
        export_cap_bins=None if role == "t3_blocks" else cap_bins,
        selected_evaluation_block_ids=[int(blocks["block_id"][i]) for i in score_blocks],
        requested_continuation_blocks=2 if role == "t3_blocks" else None,
        unavailable_continuation_blocks=max(0, 2 - len(score_blocks)) if role == "t3_blocks" else 0,
        label_scope="selected_fresh_blocks_only" if role == "t3_blocks" else "exported_prefix",
    )
    return features, target, metadata


def load_features(path):
    with np.load(path, allow_pickle=False) as z:
        if frozenset(z.files) != FEATURE_KEYS:
            raise ValueError("Missing or forbidden feature fields")
        arrays = {k: z[k] for k in z.files}
    x = arrays["x_tx45"]
    if x.ndim != 2 or x.shape[1] not in (96, 192):
        raise ValueError("Invalid feature geometry")
    for name in ("x_tx3", "x_tx45", "x_sbp"):
        a = arrays[name]
        if a.dtype != np.float32 or a.shape != x.shape or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError("Invalid neural values or dtype")
        if name != "x_sbp" and not np.equal(a, np.floor(a)).all():
            raise ValueError("Counts must be integer-valued")
    t, q, c = (arrays[k] for k in ("timestamps", "sequence_id", "channel_ids"))
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
        raise ValueError("Invalid timestamp, sequence or channel metadata")
    if np.any(~np.isclose(np.diff(t)[np.diff(q) == 0], 0.02, rtol=0, atol=1e-9)):
        raise ValueError("Unmarked within-block time gap")
    return arrays


load_targets = v2.load_targets


def _study_lock(path, expected_sha256):
    if (
        path is None
        or expected_sha256 is None
        or re.fullmatch("[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("Fresh export requires an explicit locked study-plan path and SHA256")
    path = Path(path).resolve()
    if v2.sha256_file(path) != expected_sha256:
        raise ValueError("Study-plan identity mismatch")
    root = Path(__file__).resolve().parents[2]
    try:
        relative = path.relative_to(root).as_posix()
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        committed = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"], stderr=subprocess.DEVNULL
        )
    except (ValueError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "Fresh export requires a study plan committed in this repository"
        ) from error
    if committed != path.read_bytes():
        raise ValueError("Study-plan bytes differ from the committed lock")
    return {"path": str(path), "sha256": expected_sha256, "git_revision": revision}


def prepare(
    cache,
    v4_development_index,
    v4_new_index,
    *,
    role,
    development_index=None,
    study_plan=None,
    study_plan_sha256=None,
    v4_t3_index=V4_T3_INDEX,
):
    if role not in ROLES:
        raise ValueError("Explicit V5 export role required")
    lock = None if role == "development" else _study_lock(study_plan, study_plan_sha256)
    if v2.sha256_file(Path(__file__).with_name("acquisition-plan.json")) != PLAN_SHA256:
        raise ValueError("Locked acquisition plan changed")
    cache = Path(cache)
    if (cache / "index.json").exists():
        raise ValueError("V5 prepared indexes are immutable; choose a new output directory")
    old = _read_parents(v4_development_index, v4_new_index)
    sources, receipts, dispositions = [], [], []
    t3_allowed = None
    if role == "t3_blocks":
        t3_allowed, dispositions = _t3_inventory(old, v4_t3_index)
    if role in ("development", "t3_blocks"):
        for s in old["sessions"]:
            if role == "development" or s["session_id"] in t3_allowed:
                sources.append((s["source"], Path(s["parent_cache"])))
    else:
        for participant in PARTICIPANTS:
            path = cache / "acquisition" / f"{participant}.json"
            r = json.loads(path.read_text())
            validate_receipt(r, participant, _prior_members(old, participant))
            receipts.append({"path": str(path.relative_to(cache)), "sha256": v2.sha256_file(path)})
            dispositions.append(
                {
                    "participant_id": participant,
                    "selected_sessions": len(r["selected"]),
                    "missing_requested_ranks": r["missing_requested_ranks"],
                }
            )
            for source in r["selected"]:
                sources.append(
                    (
                        {
                            **source,
                            "archive_file_id": r["archive"]["file_id"],
                            "archive_sha256_published": r["archive"]["provider_sha256"],
                            "full_archive_sha256_verified": False,
                        },
                        cache,
                    )
                )
    # Existing exposed T3 observations are prepared first for diagnostics.
    sources.sort(
        key=lambda item: (
            item[0]["participant_id"] != "T3",
            ALL_PARTICIPANTS.index(item[0]["participant_id"]),
            item[0]["post_implant_day"],
            item[0]["session_id"],
        )
    )
    known = None
    if role == "t3_blocks":
        if development_index is None:
            raise ValueError("T3 continuation requires separate known-prefix development labels")
        known = load_index(development_index, expected_role="development")
    sessions = []
    for source, raw_root in sources:
        raw_path = v2._safe_path(raw_root, source["raw_path"])
        if v2.sha256_file(raw_path) != source["member_sha256"]:
            raise ValueError("Raw source hash differs from pinned retained member")
        if role == "development":
            parent = next(s for s in old["sessions"] if s["session_id"] == source["session_id"])
            folder = cache / "prepared" / source["participant_id"]
            folder.mkdir(parents=True, exist_ok=True)
            record = {
                k: value
                for k, value in parent.items()
                if k not in {"parent_cache", "prepared_cache"}
            }
            record["role"] = role
            record["reused_parent_artifacts"] = {}
            for kind in ("feature", "target"):
                origin = v2._safe_path(Path(parent["prepared_cache"]), parent[kind + "_path"])
                if v2.sha256_file(origin) != parent[kind + "_sha256"]:
                    raise ValueError("Exposed V4 prepared artifact changed")
                target_path = folder / origin.name
                if target_path.exists():
                    raise ValueError("Partial prepared output exists; do not silently overwrite")
                os.link(origin, target_path)
                record[kind + "_path"] = str(target_path.relative_to(cache))
                record["reused_parent_artifacts"][kind] = {
                    "path": str(origin),
                    "sha256": parent[kind + "_sha256"],
                    "mode": "immutable_hard_link",
                }
            sessions.append(record)
            print(
                json.dumps(
                    {
                        "prepared": source["session_id"],
                        "role": role,
                        "n_bins": record["n_bins"],
                        "reused": True,
                    }
                ),
                flush=True,
            )
            continue
        raw = loadmat(raw_path, simplify_cells=True)
        features, targets, metadata = convert_session(
            raw, source["participant_id"], source["post_implant_day"], role=role
        )
        del raw
        folder = cache / "prepared" / source["participant_id"]
        folder.mkdir(parents=True, exist_ok=True)
        feature = folder / f"{source['session_id']}.features.npz"
        target = folder / f"{source['session_id']}.targets.npz"
        if feature.exists() or target.exists():
            raise ValueError("Partial prepared output exists; do not silently overwrite")
        np.savez_compressed(feature, **features)
        np.savez_compressed(target, **targets)
        record = {
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
        if known is not None:
            prefix = next(s for s in known["sessions"] if s["session_id"] == source["session_id"])
            record.update(
                known_prefix_target_path=str(
                    Path(development_index).resolve().parent / prefix["target_path"]
                ),
                known_prefix_target_sha256=prefix["target_sha256"],
                known_prefix_n_bins=prefix["n_bins"],
            )
        sessions.append(record)
        print(
            json.dumps(
                {"prepared": source["session_id"], "role": role, "n_bins": metadata["n_bins"]}
            ),
            flush=True,
        )
    sessions.sort(
        key=lambda s: (
            ALL_PARTICIPANTS.index(s["participant_id"]),
            s["post_implant_day"],
            s["session_id"],
        )
    )
    index = {
        "schema_version": 1,
        "role": role,
        "created_at": v2._now(),
        "participants": list(
            ALL_PARTICIPANTS
            if role == "development"
            else PARTICIPANTS
            if role == "new_sessions"
            else ["T3"]
        ),
        "provider": {"doi": v2.DOI, "version_id": v2.VERSION_ID},
        "acquisition_plan_sha256": PLAN_SHA256,
        "v4_development_index_path": str(Path(v4_development_index).resolve()),
        "v4_development_index_sha256": V4_DEVELOPMENT_INDEX_SHA256,
        "v4_new_index_path": str(Path(v4_new_index).resolve()),
        "v4_new_index_sha256": V4_NEW_INDEX_SHA256,
        "v4_t3_index_path": str(Path(v4_t3_index).resolve()),
        "v4_t3_index_sha256": V4_T3_INDEX_SHA256,
        "feature_definition": FEATURE_DEFINITION,
        "study_lock": lock,
        "preparer": {
            "v5_sha256": v2.sha256_file(__file__),
            "v2_dependency_sha256": v2.sha256_file(v2.__file__),
            "v4_dependency_sha256": v2.sha256_file(v4.__file__),
        },
        "acquisition_sources": receipts,
        "participant_dispositions": dispositions,
        "sessions": sessions,
    }
    if known is not None:
        index["development_index_path"] = str(Path(development_index).resolve())
        index["development_index_sha256"] = v2.sha256_file(development_index)
    _write(cache / "index.json", index)
    return index


def load_index(path, *, expected_role=None, verify_hashes=True):
    path = Path(path)
    cache = path.resolve().parent
    index = json.loads(path.read_text())
    role = index.get("role")
    if role not in ROLES or (expected_role is not None and role != expected_role):
        raise ValueError("V5 index role mismatch")
    participants = list(
        ALL_PARTICIPANTS
        if role == "development"
        else PARTICIPANTS
        if role == "new_sessions"
        else ["T3"]
    )
    if (
        index.get("schema_version") != 1
        or index.get("participants") != participants
        or index.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
        or index.get("acquisition_plan_sha256") != PLAN_SHA256
        or index.get("v4_development_index_sha256") != V4_DEVELOPMENT_INDEX_SHA256
        or index.get("v4_new_index_sha256") != V4_NEW_INDEX_SHA256
        or index.get("v4_t3_index_sha256") != V4_T3_INDEX_SHA256
        or index.get("feature_definition") != FEATURE_DEFINITION
    ):
        raise ValueError("V5 index provider, parent or feature identity mismatch")
    if role != "development":
        _study_lock(index["study_lock"]["path"], index["study_lock"]["sha256"])
    old = _read_parents(index["v4_development_index_path"], index["v4_new_index_path"])
    expected = {}
    if role in ("development", "t3_blocks"):
        expected = {
            s["session_id"]: (s["source"], Path(s["parent_cache"]))
            for s in old["sessions"]
            if role == "development" or s["participant_id"] == "T3"
        }
    else:
        if len(index.get("acquisition_sources", [])) != len(PARTICIPANTS):
            raise ValueError("Missing V5 acquisition receipts")
        for participant in PARTICIPANTS:
            relative = f"acquisition/{participant}.json"
            refs = [r for r in index["acquisition_sources"] if r["path"] == relative]
            if len(refs) != 1:
                raise ValueError("Missing or duplicated acquisition receipt")
            receipt_path = v2._safe_path(cache, relative)
            if verify_hashes and v2.sha256_file(receipt_path) != refs[0]["sha256"]:
                raise ValueError("Acquisition receipt hash mismatch")
            r = json.loads(receipt_path.read_text())
            validate_receipt(r, participant, _prior_members(old, participant))
            for source in r["selected"]:
                expected[source["session_id"]] = (source, cache)
    if role == "t3_blocks":
        retained, dispositions = _t3_inventory(old, index["v4_t3_index_path"])
        expected = {key: value for key, value in expected.items() if key in retained}
        if index["participant_dispositions"] != dispositions:
            raise ValueError("T3 unavailable-session dispositions changed")
    if len(index["sessions"]) != len(expected) or {
        s["session_id"] for s in index["sessions"]
    } != set(expected):
        raise ValueError("Selected session inventory changed")
    known = None
    if role == "t3_blocks":
        dev_path = Path(index["development_index_path"])
        if v2.sha256_file(dev_path) != index["development_index_sha256"]:
            raise ValueError("Known-prefix index identity mismatch")
        known = load_index(dev_path, expected_role="development", verify_hashes=False)
    previous = {}
    for s in index["sessions"]:
        source, root = expected[s["session_id"]]
        if (
            s["role"] != role
            or any(s["source"].get(k) != v for k, v in source.items())
            or Path(s["raw_source_root"]).resolve() != root.resolve()
            or s["participant_id"] != source["participant_id"]
            or s["post_implant_day"] != source["post_implant_day"]
            or type(s["n_bins"]) is not int
            or s["n_bins"] < 0
            or s["n_channels"] not in (96, 192)
            or s["bin_seconds"] != 0.02
        ):
            raise ValueError("Session role, source or geometry mismatch")
        if role == "development":
            parent = next(a for a in old["sessions"] if a["session_id"] == s["session_id"])
            if any(
                s[k] != parent[k]
                for k in ("feature_sha256", "target_sha256", "n_bins", "n_channels", "segments")
            ):
                raise ValueError("Exposed development artifact differs from its V4 parent")
        key = (s["post_implant_day"], s["session_id"])
        p = s["participant_id"]
        if p in previous and key <= previous[p]:
            raise ValueError("Sessions must be chronological within participant")
        previous[p] = key
        end = 0
        selected = []
        for number, segment in enumerate(s["segments"]):
            if (
                segment["start"] != end
                or segment["end"] < end
                or segment["sequence_id"] != number
                or segment["global_start"] != segment["start"]
                or segment["global_end"] != segment["end"]
            ):
                raise ValueError("Segments do not partition original causal history")
            if segment["selected_for_evaluation"]:
                selected.append(segment)
            end = segment["end"]
        if end != s["n_bins"]:
            raise ValueError("Incomplete segment coverage")
        if role != "t3_blocks" and s["n_bins"] != min(s["full_prepared_n_bins"], MAX_BINS):
            raise ValueError("Export exceeds fixed development/new-session prefix")
        if role == "t3_blocks":
            # Independently re-read only block metadata to authenticate the selected boundary.
            rawmeta = loadmat(
                v2._safe_path(root, source["raw_path"]),
                simplify_cells=True,
                variable_names=["blocks"],
            )
            blocks, lengths, offsets = _block_geometry(rawmeta)
            chosen = _t3_selection(rawmeta)
            expected_ids = [int(blocks["block_id"][i]) for i in chosen]
            if (
                not len(chosen)
                or s["selected_evaluation_block_ids"] != expected_ids
                or [a["block_id"] for a in selected] != expected_ids
                or s["n_bins"] != int(offsets[chosen[-1]] + lengths[chosen[-1]])
            ):
                raise ValueError("T3 selected block boundaries changed")
            prefix = next(a for a in known["sessions"] if a["session_id"] == s["session_id"])
            prefix_path = (
                Path(index["development_index_path"]).resolve().parent / prefix["target_path"]
            )
            if (
                Path(s["known_prefix_target_path"]).resolve() != prefix_path.resolve()
                or s["known_prefix_target_sha256"] != prefix["target_sha256"]
                or s["known_prefix_n_bins"] != prefix["n_bins"]
            ):
                raise ValueError("Known-prefix target binding mismatch")
            if verify_hashes and v2.sha256_file(prefix_path) != prefix["target_sha256"]:
                raise ValueError("Known-prefix target hash mismatch")
        if verify_hashes:
            for k in ("feature", "target"):
                if v2.sha256_file(v2._safe_path(cache, s[k + "_path"])) != s[k + "_sha256"]:
                    raise ValueError("Prepared artifact hash mismatch")
            if v2.sha256_file(v2._safe_path(root, source["raw_path"])) != source["member_sha256"]:
                raise ValueError("Retained raw member hash mismatch")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "metadata",
            "acquire",
            "prepare-development",
            "prepare-new_sessions",
            "prepare-t3_blocks",
            "validate",
        ],
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--v4-development-index", type=Path, default=Path("data/neural_v4/development/index.json")
    )
    parser.add_argument(
        "--v4-new-index", type=Path, default=Path("data/neural_v4/new_sessions/index.json")
    )
    parser.add_argument("--participants", nargs="+", choices=PARTICIPANTS, default=PARTICIPANTS)
    parser.add_argument("--url-file", type=Path)
    parser.add_argument("--development-index", type=Path)
    parser.add_argument("--v4-t3-index", type=Path, default=V4_T3_INDEX)
    parser.add_argument("--study-plan", type=Path)
    parser.add_argument("--study-plan-sha256")
    parser.add_argument("--expected-role", choices=sorted(ROLES))
    args = parser.parse_args()
    if args.command == "metadata":
        print(json.dumps({"version_id": v2.pin_metadata(args.cache)["version_id"]}))
    elif args.command == "acquire":
        acquire(
            args.cache,
            args.v4_development_index,
            args.v4_new_index,
            participants=tuple(args.participants),
            url_file=args.url_file,
        )
    elif args.command.startswith("prepare-"):
        result = prepare(
            args.cache,
            args.v4_development_index,
            args.v4_new_index,
            role=args.command.removeprefix("prepare-"),
            development_index=args.development_index,
            study_plan=args.study_plan,
            study_plan_sha256=args.study_plan_sha256,
            v4_t3_index=args.v4_t3_index,
        )
        print(json.dumps({"role": result["role"], "sessions": len(result["sessions"])}))
    else:
        result = load_index(args.cache / "index.json", expected_role=args.expected_role)
        for s in result["sessions"]:
            f = load_features(args.cache / s["feature_path"])
            t = load_targets(args.cache / s["target_path"], len(f["x_tx45"]))
            if result["role"] == "t3_blocks":
                allowed = np.zeros(s["n_bins"], dtype=bool)
                for segment in s["segments"]:
                    if segment["selected_for_evaluation"]:
                        allowed[segment["start"] : segment["end"]] = True
                if np.any(t["eval_mask"][~allowed]) or np.any(t["y"][~allowed]):
                    raise ValueError("Unselected continuation labels leaked into scorer sidefile")
        print(
            json.dumps(
                {"validated": True, "role": result["role"], "sessions": len(result["sessions"])}
            )
        )


if __name__ == "__main__":
    main()
