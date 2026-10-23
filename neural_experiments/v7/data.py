"""Authenticated V7 development prefixes with separate neural and label artifacts.

V7 promotes exactly the 152 previously exposed V6 prefixes to development. This
module never opens retained MAT sources or prepares any continuation samples.
Feature-only callers never open targets. Hash validation is an integrity boundary,
not operating-system isolation of the local private data cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from neural_experiments.v2 import data as v2
from neural_experiments.v6 import data as v6

ALL_PARTICIPANTS = v6.ALL_PARTICIPANTS
PARTICIPANTS = v6.PARTICIPANTS
FEATURE_KEYS = v6.FEATURE_KEYS
MAX_BINS = v6.MAX_BINS
V6_DEVELOPMENT_INDEX_SHA256 = "357f8b917abec17f6354dac3cd93b4d87d24df0db4bf60afeee8bdbefbf84678"
V6_NEW_INDEX_SHA256 = "dfe5ed95bf7f43b0bb51f3a11d163505e99641a4affc86daf38591fbdc5af534"
PARENT_COUNTS = {"development": 128, "new_sessions": 24}
EXPECTED_COUNTS = {
    participant: 8 if participant == "T3" else 18 for participant in ALL_PARTICIPANTS
}
EXPOSURE = {
    "rule": "exact_union_of_exposed_v6_development_and_new_session_prefixes",
    "sessions": 152,
    "maximum_bins_per_session": MAX_BINS,
    "continuation_tails_included": False,
    "fresh_data_included": False,
}

load_features = v6.load_features
load_targets = v6.load_targets


def _parents(v6_development_index, v6_new_index):
    """Authenticate metadata without parsing features, labels, or retained raw MATs."""
    parents, records = {}, []
    for location, role, expected_sha in (
        (v6_development_index, "development", V6_DEVELOPMENT_INDEX_SHA256),
        (v6_new_index, "new_sessions", V6_NEW_INDEX_SHA256),
    ):
        path = Path(location).resolve()
        if v2.sha256_file(path) != expected_sha:
            raise ValueError("V6 parent index differs from the locked exposed cohort")
        parent = json.loads(path.read_text())
        if (
            parent.get("role") != role
            or parent.get("schema_version") != 1
            or parent.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
            or parent.get("feature_definition") != v6.FEATURE_DEFINITION
            or len(parent.get("sessions", [])) != PARENT_COUNTS[role]
        ):
            raise ValueError("V6 parent role, count, provider or feature identity mismatch")
        parents[role] = {"path": str(path), "sha256": expected_sha}
        for item in parent["sessions"]:
            if (
                item.get("role") != role
                or item.get("participant_id") not in ALL_PARTICIPANTS
                or type(item.get("n_bins")) is not int
                or not 0 <= item["n_bins"] <= MAX_BINS
                or item["n_bins"] != min(item["full_prepared_n_bins"], MAX_BINS)
                or item.get("bin_seconds") != 0.02
                or item.get("n_channels") not in (96, 192)
            ):
                raise ValueError("Parent record exceeds the exposed prefix or geometry")
            previous_end = 0
            for number, segment in enumerate(item["segments"]):
                if (
                    segment["sequence_id"] != number
                    or segment["start"] != previous_end
                    or not previous_end <= segment["end"] <= item["n_bins"]
                    or segment["global_start"] != segment["start"]
                    or segment["global_end"] != segment["end"]
                ):
                    raise ValueError("Parent segments do not partition the exposed prefix")
                previous_end = segment["end"]
            if previous_end != item["n_bins"]:
                raise ValueError("Incomplete parent segment coverage")
            record = copy.deepcopy(item)
            record["role"] = "development"
            record["v7_parent"] = {
                "index_path": str(path),
                "index_sha256": expected_sha,
                "role": role,
                "session_id": item["session_id"],
            }
            record["reused_parent_artifacts"] = {}
            for kind in ("feature", "target"):
                origin = v2._safe_path(path.parent, item[kind + "_path"])
                record[kind + "_path"] = (
                    f"prepared/{item['participant_id']}/{item['session_id']}.{kind}s.npz"
                )
                record["reused_parent_artifacts"][kind] = {
                    "path": str(origin),
                    "sha256": item[kind + "_sha256"],
                    "mode": "immutable_hard_link",
                }
            records.append(record)
    if len(records) != EXPOSURE["sessions"] or len({r["session_id"] for r in records}) != len(
        records
    ):
        raise ValueError("Expected exactly 152 unique previously exposed sessions")
    if {
        p: sum(r["participant_id"] == p for r in records) for p in ALL_PARTICIPANTS
    } != EXPECTED_COUNTS:
        raise ValueError("Exposed participant inventory changed")
    records.sort(
        key=lambda s: (
            ALL_PARTICIPANTS.index(s["participant_id"]),
            s["post_implant_day"],
            s["session_id"],
        )
    )
    return parents, records


def prepare_development(cache, v6_development_index, v6_new_index):
    """Reuse only authenticated, already exposed feature/target bytes by hard link."""
    cache = Path(cache).resolve()
    if (cache / "index.json").exists():
        raise ValueError("V7 prepared indexes are immutable; choose a new output directory")
    parents, records = _parents(v6_development_index, v6_new_index)
    for item in records:
        for kind in ("feature", "target"):
            reference = item["reused_parent_artifacts"][kind]
            origin = Path(reference["path"])
            if v2.sha256_file(origin) != reference["sha256"]:
                raise ValueError("Exposed V6 prepared artifact changed")
            destination = v2._safe_path(cache, item[kind + "_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError("Partial prepared output exists; do not silently overwrite")
            os.link(origin, destination)
    index = {
        "schema_version": 1,
        "role": "development",
        "created_at": v2._now(),
        "participants": list(ALL_PARTICIPANTS),
        "provider": {"doi": v2.DOI, "version_id": v2.VERSION_ID},
        "feature_definition": v6.FEATURE_DEFINITION,
        "exposure": EXPOSURE,
        "parents": parents,
        "preparer": {"v7_sha256": v2.sha256_file(__file__)},
        "sessions": records,
    }
    v2._json(cache / "index.json", index)
    return index


def load_index(path, *, expected_role=None, verify_hashes=True):
    """Validate provenance; optional hash checks touch only exposed prepared files."""
    path = Path(path).resolve()
    index = json.loads(path.read_text())
    if index.get("role") != "development" or expected_role not in (None, "development"):
        raise ValueError("V7 development index role mismatch")
    if (
        index.get("schema_version") != 1
        or index.get("participants") != list(ALL_PARTICIPANTS)
        or index.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
        or index.get("feature_definition") != v6.FEATURE_DEFINITION
        or index.get("exposure") != EXPOSURE
    ):
        raise ValueError("V7 development provider, exposure or feature identity mismatch")
    parents, expected = _parents(
        index["parents"]["development"]["path"], index["parents"]["new_sessions"]["path"]
    )
    if index["parents"] != parents or index.get("sessions") != expected:
        raise ValueError("V7 development record differs from its immutable V6 parent")
    if verify_hashes:
        for item in expected:
            for kind in ("feature", "target"):
                artifact = v2._safe_path(path.parent, item[kind + "_path"])
                if v2.sha256_file(artifact) != item[kind + "_sha256"]:
                    raise ValueError("Prepared artifact hash mismatch")
    return index


def load_case(index_path, item, *, labels=False):
    """Return pure neural features by default; labels require an explicit request.

    The caller obtains ``item`` from ``load_index``. The separate target loader is
    intended for the scorer or a chronological label broker, not policy features.
    """
    root = Path(index_path).resolve().parent
    feature = v2._safe_path(root, item["feature_path"])
    if v2.sha256_file(feature) != item["feature_sha256"]:
        raise ValueError("Feature identity mismatch")
    features = load_features(feature)
    if len(features["x_tx45"]) != item["n_bins"]:
        raise ValueError("Feature geometry differs from authenticated record")
    result = {
        "item": item,
        "features": features,
        "segments": [(s["start"], s["end"]) for s in item["segments"] if s["end"] > s["start"]],
    }
    if labels:
        target = v2._safe_path(root, item["target_path"])
        if v2.sha256_file(target) != item["target_sha256"]:
            raise ValueError("Target identity mismatch")
        result.update(load_targets(target, item["n_bins"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-development", "validate-development"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--v6-development-index", type=Path, default=Path("data/neural_v6/development/index.json")
    )
    parser.add_argument(
        "--v6-new-index", type=Path, default=Path("data/neural_v6/new_sessions/index.json")
    )
    args = parser.parse_args()
    if args.command == "prepare-development":
        index = prepare_development(args.cache, args.v6_development_index, args.v6_new_index)
    else:
        index = load_index(args.cache / "index.json", expected_role="development")
    print(json.dumps({"role": index["role"], "sessions": len(index["sessions"])}))


if __name__ == "__main__":
    main()
