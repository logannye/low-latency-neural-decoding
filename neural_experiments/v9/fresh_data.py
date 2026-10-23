"""V9 fresh ranks25..30 with authenticated prior24 and bounded private exports.

Acquisition requires the exact committed acquisition lock; conversion additionally
requires a committed V9 study plan. No old raw tails are prepared or inspected.
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
from neural_experiments.v6 import data as v6
from neural_experiments.v9 import data as development

ALL_PARTICIPANTS = v2.PARTICIPANTS
PARTICIPANTS = ("T2", "T7", "T9", "T10")
V8_FRESH_INDEX_SHA256 = development.V8_FRESH_INDEX_SHA256
V8_DEVELOPMENT_INDEX_SHA256 = development.V8_DEVELOPMENT_INDEX_SHA256
FEATURE_KEYS = frozenset({"x_tx3", "x_tx45", "x_sbp", "timestamps", "sequence_id", "channel_ids"})
MAX_BINS = 30000
PLAN_SHA256 = "56e697e30821105ef5689a29b92e63c4f214f113583028798f2d366501ec9271"
MAX_COMPRESSED_BYTES = 32000000000
MAX_CACHE_BYTES = 12000000000
SELECTION = {
    "rule": "regular_decoding_session_members_25_through_30_in_pinned_archive_order",
    "regular_session_ranks_one_based": [25, 26, 27, 28, 29, 30],
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


_study_lock = v6._study_lock


def _write(path: Path, value: Any) -> None:
    v2._json(path, value)


def _read_parents(v8_development_index, v8_fresh_index):
    _, records = development._parents(v8_development_index, v8_fresh_index)
    return {"participants": list(ALL_PARTICIPANTS), "sessions": records}


def _prior_members(index: dict[str, Any], participant: str) -> list[dict[str, Any]]:
    return sorted(
        [s["source"] for s in index["sessions"] if s["participant_id"] == participant],
        key=lambda s: s["archive_ordinal"],
    )


def _pattern(participant: str) -> str:
    return f"(?:\\./)?decoding/{participant}/{participant}_day_(\\d+)_decoding\\.mat"


def extract_next_sessions(
    stream: BinaryIO,
    *,
    cache: Path,
    participant: str,
    prior_members: list[dict[str, Any]],
    archive: dict[str, Any],
    max_stream_bytes: int = MAX_COMPRESSED_BYTES,
    max_cache_bytes: int = MAX_CACHE_BYTES,
    reserve_member=None,
) -> dict[str, Any]:
    """Stream ranks1..30, authenticate old1..24, retain only new25..30."""
    if participant not in PARTICIPANTS or len(prior_members) != 24:
        raise ValueError("Require a pinned participant and exactly twenty-four prior members")
    cache = Path(cache)
    raw_dir = cache / "raw" / participant
    raw_dir.mkdir(parents=True, exist_ok=True)
    reader = v2.BoundedReader(stream, max_stream_bytes)
    receipt = {
        "schema_version": 1,
        "participant_id": participant,
        "archive": archive,
        "selection": SELECTION,
        "v8_development_index_sha256": V8_DEVELOPMENT_INDEX_SHA256,
        "v8_fresh_index_sha256": V8_FRESH_INDEX_SHA256,
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
                    if member.name.endswith(".mat") and (not name.name.startswith("._")):
                        raise ValueError("Unexpected regular MAT member")
                    continue
                if name.stem in known_names:
                    raise ValueError("Duplicate regular session identity")
                known_names.add(name.stem)
                regular_rank += 1
                previous = prior_members[regular_rank - 1] if regular_rank <= 24 else None
                if previous is not None and (
                    previous["member_name"] != member.name
                    or previous["member_size"] != member.size
                    or previous["archive_ordinal"] != ordinal
                ):
                    raise ValueError(
                        "First twenty-four archive members differ from locked V8 selection"
                    )
                if member.size > 2 * 1024**3:
                    raise ValueError("Selected member exceeds 2GiB safety bound")
                target = raw_dir / name.name
                temporary = target.with_suffix(".mat.part")
                if previous is None:
                    if reserve_member is not None:
                        reserve_member(target, member.size)
                    existing = target.stat().st_size if target.exists() else 0
                    if v2._cache_size(cache) + member.size - existing > max_cache_bytes:
                        raise ValueError("V9 cache disk budget exceeded")
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
                        raise ValueError("Previously acquired V8 member hash changed")
                    receipt["verified_prior_members"].append(identity)
                else:
                    if target.exists() and v2.sha256_file(target) != digest.hexdigest():
                        raise ValueError("Existing V9 raw member identity changed")
                    temporary.replace(target)
                    identity["raw_path"] = str(target.relative_to(cache))
                    receipt["selected"].append(identity)
                if regular_rank == 30:
                    break
            else:
                receipt["tar_end_observed"] = True
        if regular_rank < 24:
            raise ValueError("Archive ended before the authenticated V8 prefix")
        receipt["status"] = "complete"
        receipt["missing_requested_ranks"] = list(range(max(25, regular_rank + 1), 31))
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
        or (receipt.get("v8_development_index_sha256") != V8_DEVELOPMENT_INDEX_SHA256)
        or (receipt.get("v8_fresh_index_sha256") != V8_FRESH_INDEX_SHA256)
        or (receipt.get("plan_sha256") != PLAN_SHA256)
    ):
        raise ValueError("Invalid V9 receipt role/selection")
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
    if len(regular) < 24 or len(regular) > 30:
        raise ValueError("Invalid archive regular-member inventory")
    selected = receipt["selected"]
    prior = receipt["verified_prior_members"]
    if len(prior) != 24 or len(selected) != len(regular[24:30]):
        raise ValueError("Missing authenticated prefix or selected rank")
    if len(selected) < 6 and (not receipt["tar_end_observed"]):
        raise ValueError("Short acquisition requires observed archive end")
    identities = set()
    for rank, (row, seen) in enumerate(zip(prior + selected, regular, strict=True), 1):
        match = re.fullmatch(_pattern(participant), row["member_name"])
        if (
            match is None
            or row["participant_id"] != participant
            or row["regular_session_rank"] != rank
            or (row["post_implant_day"] != int(match.group(1)))
            or (row["session_id"] != PurePosixPath(row["member_name"]).stem)
            or (row["member_name"] != seen["name"])
            or (row["member_size"] != seen["size"])
            or (row["archive_ordinal"] != seen["ordinal"])
            or (re.fullmatch("[0-9a-f]{64}", row["member_sha256"]) is None)
            or (row["session_id"] in identities)
        ):
            raise ValueError("Member provenance/rank mismatch")
        identities.add(row["session_id"])
        if rank > 24 and row["raw_path"] != f"raw/{participant}/{row['session_id']}.mat":
            raise ValueError("Unsafe raw-member path")
        if rank <= 24 and prior_members is not None:
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
                raise ValueError("Receipt authenticated prefix differs from V8")
    expected_missing = list(range(25 + len(selected), 31))
    if receipt.get("missing_requested_ranks") != expected_missing:
        raise ValueError("Incorrect shortage accounting")


def acquire(
    cache,
    v8_development_index,
    v8_fresh_index,
    *,
    participants=PARTICIPANTS,
    url_file=None,
    max_stream_bytes=MAX_COMPRESSED_BYTES,
    max_total_stream_bytes=MAX_COMPRESSED_BYTES,
    max_cache_bytes=MAX_CACHE_BYTES,
    manifest=None,
    opener=None,
    reserve_member=None,
):
    cache = Path(cache)
    old = _read_parents(v8_development_index, v8_fresh_index)
    plan_path = Path(__file__).with_name("acquisition-plan.json")
    if v2.sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("Committed assessment selection plan changed")
    if max_total_stream_bytes > MAX_COMPRESSED_BYTES or max_cache_bytes > MAX_CACHE_BYTES:
        raise ValueError("Acquisition exceeds locked network/storage bound")
    _study_lock(plan_path, PLAN_SHA256)
    manifest = v2.pin_metadata(cache) if manifest is None else manifest
    opener = urlopen if opener is None else opener
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
                        raise ValueError("Cached V9 member hash changed")
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
            raise ValueError("Aggregate V9 network budget exhausted")
        archive = next(a for a in manifest["archives"] if a["participant_id"] == participant)
        url = archive["download_url"]
        if url_file is not None:
            url = observed[participant]
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.username
                or parsed.password
                or (parsed.hostname != "dryad-assetstore-merritt-west.s3.us-west-2.amazonaws.com")
                or (
                    parse_qs(parsed.query).get("response-content-disposition")
                    != [f"attachment; filename={archive['filename']}"]
                )
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
            with opener(request, timeout=90) as response:
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
                    reserve_member=reserve_member,
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


def convert_session(raw, participant, day, *, role="fresh", cap_bins=MAX_BINS):
    if role != "fresh":
        raise ValueError("V9 only permits the fresh role")
    return v6.convert_session(raw, participant, day, role="new_sessions", cap_bins=cap_bins)


load_features = v6.load_features
load_targets = v6.load_targets
load_context = development.load_context
load_case = development.load_case


def prepare(
    cache,
    v8_development_index,
    v8_fresh_index,
    *,
    role="fresh",
    study_plan=None,
    study_plan_sha256=None,
):
    """Export only selected new sessions after both committed plans authenticate."""
    if role != "fresh":
        raise ValueError("V9 export requires role=fresh")
    lock = _study_lock(study_plan, study_plan_sha256)
    acquisition_lock = _study_lock(Path(__file__).with_name("acquisition-plan.json"), PLAN_SHA256)
    cache = Path(cache).resolve()
    if (cache / "index.json").exists():
        raise ValueError("V9 prepared indexes are immutable; choose a new output directory")
    old = _read_parents(v8_development_index, v8_fresh_index)
    sources, receipts, dispositions = [], [], []
    for participant in PARTICIPANTS:
        path = cache / "acquisition" / f"{participant}.json"
        receipt = json.loads(path.read_text())
        validate_receipt(receipt, participant, _prior_members(old, participant))
        receipts.append({"path": str(path.relative_to(cache)), "sha256": v2.sha256_file(path)})
        dispositions.append(
            {
                "participant_id": participant,
                "selected_sessions": len(receipt["selected"]),
                "missing_requested_ranks": receipt["missing_requested_ranks"],
            }
        )
        for source in receipt["selected"]:
            sources.append(
                {
                    **source,
                    "archive_file_id": receipt["archive"]["file_id"],
                    "archive_sha256_published": receipt["archive"]["provider_sha256"],
                    "full_archive_sha256_verified": False,
                }
            )
    sources.sort(
        key=lambda s: (
            PARTICIPANTS.index(s["participant_id"]),
            s["post_implant_day"],
            s["session_id"],
        )
    )
    sessions = []
    for source in sources:
        raw_path = v2._safe_path(cache, source["raw_path"])
        if v2.sha256_file(raw_path) != source["member_sha256"]:
            raise ValueError("Raw source differs from authenticated selected member")
        raw = loadmat(raw_path, simplify_cells=True)
        features, targets, metadata = convert_session(
            raw, source["participant_id"], source["post_implant_day"]
        )
        context = development.make_context(raw, metadata)
        del raw
        development.validate_context(context, metadata["n_bins"])
        development.validate_context_targets(context, targets)
        folder = cache / "prepared" / source["participant_id"]
        folder.mkdir(parents=True, exist_ok=True)
        feature = folder / f"{source['session_id']}.features.npz"
        target = folder / f"{source['session_id']}.targets.npz"
        context_path = folder / f"{source['session_id']}.contexts.npz"
        if feature.exists() or target.exists() or context_path.exists():
            raise ValueError("Partial prepared output exists; do not silently overwrite")
        np.savez_compressed(feature, **features)
        np.savez_compressed(target, **targets)
        np.savez_compressed(context_path, **context)
        sessions.append(
            {
                **metadata,
                "participant_id": source["participant_id"],
                "session_id": source["session_id"],
                "post_implant_day": source["post_implant_day"],
                "role": "fresh",
                "source": source,
                "raw_source_root": str(cache),
                "feature_path": str(feature.relative_to(cache)),
                "target_path": str(target.relative_to(cache)),
                "feature_sha256": v2.sha256_file(feature),
                "target_sha256": v2.sha256_file(target),
                "context_path": str(context_path.relative_to(cache)),
                "context_sha256": v2.sha256_file(context_path),
                "far_distance": development.FAR_DISTANCE_BY_SESSION.get(
                    source["session_id"],
                    development.FAR_DISTANCE_BY_PARTICIPANT[source["participant_id"]],
                ),
            }
        )
        print(
            json.dumps({"prepared": source["session_id"], "n_bins": metadata["n_bins"]}), flush=True
        )
    index = {
        "schema_version": 1,
        "role": "fresh",
        "created_at": v2._now(),
        "participants": list(PARTICIPANTS),
        "provider": {"doi": v2.DOI, "version_id": v2.VERSION_ID},
        "feature_definition": FEATURE_DEFINITION,
        "context_definition": development.CONTEXT_DEFINITION,
        "acquisition_plan_sha256": PLAN_SHA256,
        "acquisition_lock": acquisition_lock,
        "study_lock": lock,
        "v8_development_index_path": str(Path(v8_development_index).resolve()),
        "v8_development_index_sha256": V8_DEVELOPMENT_INDEX_SHA256,
        "v8_fresh_index_path": str(Path(v8_fresh_index).resolve()),
        "v8_fresh_index_sha256": V8_FRESH_INDEX_SHA256,
        "acquisition_sources": receipts,
        "participant_dispositions": dispositions,
        "preparer": {
            "v9_fresh_sha256": v2.sha256_file(__file__),
            "v6_dependency_sha256": v2.sha256_file(v6.__file__),
            "v9_development_dependency_sha256": v2.sha256_file(development.__file__),
        },
        "sessions": sessions,
    }
    _write(cache / "index.json", index)
    return index


def load_index(path, *, expected_role=None, verify_hashes=True):
    """Authenticate the fixed selected cohort and all separate prepared artifacts."""
    path = Path(path).resolve()
    cache = path.parent
    index = json.loads(path.read_text())
    if index.get("role") != "fresh" or expected_role not in (None, "fresh"):
        raise ValueError("V9 index role mismatch")
    if (
        index.get("schema_version") != 1
        or index.get("participants") != list(PARTICIPANTS)
        or index.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
        or index.get("feature_definition") != FEATURE_DEFINITION
        or index.get("context_definition") != development.CONTEXT_DEFINITION
        or index.get("acquisition_plan_sha256") != PLAN_SHA256
        or index.get("v8_development_index_sha256") != V8_DEVELOPMENT_INDEX_SHA256
        or index.get("v8_fresh_index_sha256") != V8_FRESH_INDEX_SHA256
    ):
        raise ValueError("V9 provider, parent, plan or feature identity mismatch")
    _study_lock(index["study_lock"]["path"], index["study_lock"]["sha256"])
    _study_lock(Path(__file__).with_name("acquisition-plan.json"), PLAN_SHA256)
    old = _read_parents(index["v8_development_index_path"], index["v8_fresh_index_path"])
    if len(index.get("acquisition_sources", [])) != len(PARTICIPANTS):
        raise ValueError("Missing acquisition receipts")
    expected, dispositions = {}, []
    for participant in PARTICIPANTS:
        relative = f"acquisition/{participant}.json"
        refs = [r for r in index["acquisition_sources"] if r["path"] == relative]
        if len(refs) != 1:
            raise ValueError("Missing or duplicated acquisition receipt")
        receipt_path = v2._safe_path(cache, relative)
        if v2.sha256_file(receipt_path) != refs[0]["sha256"]:
            raise ValueError("Acquisition receipt hash mismatch")
        receipt = json.loads(receipt_path.read_text())
        validate_receipt(receipt, participant, _prior_members(old, participant))
        dispositions.append(
            {
                "participant_id": participant,
                "selected_sessions": len(receipt["selected"]),
                "missing_requested_ranks": receipt["missing_requested_ranks"],
            }
        )
        for source in receipt["selected"]:
            expected[source["session_id"]] = source
    if index.get("participant_dispositions") != dispositions:
        raise ValueError("Shortage dispositions differ from acquisition")
    if len(index["sessions"]) != len(expected) or {
        s["session_id"] for s in index["sessions"]
    } != set(expected):
        raise ValueError("Selected session inventory changed")
    previous = None
    for item in index["sessions"]:
        source = expected[item["session_id"]]
        key = (
            PARTICIPANTS.index(item["participant_id"]),
            item["post_implant_day"],
            item["session_id"],
        )
        if previous is not None and key <= previous:
            raise ValueError("Sessions must be chronological within ordered participants")
        previous = key
        if (
            item.get("role") != "fresh"
            or any(item["source"].get(k) != value for k, value in source.items())
            or Path(item["raw_source_root"]).resolve() != cache
            or item["participant_id"] != source["participant_id"]
            or item["post_implant_day"] != source["post_implant_day"]
            or type(item["n_bins"]) is not int
            or not 0 <= item["n_bins"] <= MAX_BINS
            or item["n_bins"] != min(item["full_prepared_n_bins"], MAX_BINS)
            or item["n_channels"] not in (96, 192)
            or item["bin_seconds"] != 0.02
            or item.get("far_distance")
            != development.FAR_DISTANCE_BY_SESSION.get(
                item["session_id"], development.FAR_DISTANCE_BY_PARTICIPANT[item["participant_id"]]
            )
        ):
            raise ValueError("Fresh record source, geometry or exposure changed")
        end = 0
        for number, segment in enumerate(item["segments"]):
            if (
                segment["sequence_id"] != number
                or segment["start"] != end
                or not end <= segment["end"] <= item["n_bins"]
                or segment["global_start"] != segment["start"]
                or segment["global_end"] != segment["end"]
            ):
                raise ValueError("Segments do not partition the selected prefix")
            end = segment["end"]
        if end != item["n_bins"]:
            raise ValueError("Incomplete selected prefix")
        if verify_hashes:
            for kind in ("feature", "target", "context"):
                if (
                    v2.sha256_file(v2._safe_path(cache, item[kind + "_path"]))
                    != item[kind + "_sha256"]
                ):
                    raise ValueError("Prepared artifact hash mismatch")
            if v2.sha256_file(v2._safe_path(cache, source["raw_path"])) != source["member_sha256"]:
                raise ValueError("Retained fresh raw hash mismatch")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("metadata", "acquire", "prepare", "validate"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--v8-development-index", type=Path, default=Path("data/neural_v8/development/index.json")
    )
    parser.add_argument(
        "--v8-fresh-index", type=Path, default=Path("data/neural_v8/fresh/index.json")
    )
    parser.add_argument("--participants", nargs="+", choices=PARTICIPANTS, default=PARTICIPANTS)
    parser.add_argument("--url-file", type=Path)
    parser.add_argument("--study-plan", type=Path)
    parser.add_argument("--study-plan-sha256")
    args = parser.parse_args()
    if args.command == "metadata":
        _study_lock(Path(__file__).with_name("acquisition-plan.json"), PLAN_SHA256)
        print(json.dumps({"version_id": v2.pin_metadata(args.cache)["version_id"]}))
    elif args.command == "acquire":
        acquire(
            args.cache,
            args.v8_development_index,
            args.v8_fresh_index,
            participants=tuple(args.participants),
            url_file=args.url_file,
        )
    elif args.command == "prepare":
        index = prepare(
            args.cache,
            args.v8_development_index,
            args.v8_fresh_index,
            study_plan=args.study_plan,
            study_plan_sha256=args.study_plan_sha256,
        )
        print(json.dumps({"role": index["role"], "sessions": len(index["sessions"])}))
    else:
        index = load_index(args.cache / "index.json", expected_role="fresh")
        print(json.dumps({"role": index["role"], "sessions": len(index["sessions"])}))


if __name__ == "__main__":
    main()
