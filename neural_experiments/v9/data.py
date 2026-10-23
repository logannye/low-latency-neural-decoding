"""V9 exact union of 200 exposed V8 prefixes, reusing all sidecars without raw reads."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from neural_experiments.v2 import data as v2
from neural_experiments.v6 import data as v6
from neural_experiments.v8 import data as v8

ALL_PARTICIPANTS = v2.PARTICIPANTS
PARTICIPANTS = tuple(p for p in ALL_PARTICIPANTS if p != "T3")
MAX_BINS = 30000
FEATURE_KEYS = v6.FEATURE_KEYS
FEATURE_DEFINITION = v6.FEATURE_DEFINITION
V8_DEVELOPMENT_INDEX_SHA256 = "5c4b5ab0b0fa0ec28c2c5a1b746ca8e8ceee35b4797e2f7e2f66cc9d9dfdbf84"
V8_FRESH_INDEX_SHA256 = "0ea2078c184e58014c9fc68d5fd5efa3855e93a12a82db6f2627174ae894c4b8"
PARENT_COUNTS = {"development": 176, "fresh": 24}
EXPECTED_COUNTS = {p: 8 if p == "T3" else 24 for p in ALL_PARTICIPANTS}
CONTEXT_KEYS = v8.CONTEXT_KEYS
FAR_DISTANCE_BY_PARTICIPANT = v8.FAR_DISTANCE_BY_PARTICIPANT
FAR_DISTANCE_BY_SESSION = v8.FAR_DISTANCE_BY_SESSION
CONTEXT_DEFINITION = v8.CONTEXT_DEFINITION
EXPOSURE = {
    "rule": "exact_union_of_exposed_v8_development176_and_fresh24_prefixes",
    "sessions": 200,
    "maximum_bins_per_session": MAX_BINS,
    "continuation_tails_included": False,
    "development_reuse": (
        "immutable hardlinks of existing feature, target and context sidecars; no raw reads"
    ),
}

load_features = v6.load_features
load_targets = v6.load_targets


def _geometry(item):
    if (
        type(item.get("n_bins")) is not int
        or not 0 <= item["n_bins"] <= MAX_BINS
        or item["n_bins"] != min(item["full_prepared_n_bins"], MAX_BINS)
        or item.get("n_channels") not in (96, 192)
        or item.get("bin_seconds") != 0.02
    ):
        raise ValueError("Record exceeds the exposed prefix or geometry")
    end = 0
    for number, s in enumerate(item["segments"]):
        if (
            s["sequence_id"] != number
            or s["start"] != end
            or not end <= s["end"] <= item["n_bins"]
            or s["global_start"] != s["start"]
            or s["global_end"] != s["end"]
            or (s["source_end"] - s["source_start"]) // 2 != s["end"] - s["start"]
        ):
            raise ValueError("Segments do not partition the exposed prefix")
        end = s["end"]
    if end != item["n_bins"]:
        raise ValueError("Incomplete exposed segment coverage")


def _parents(v8_development_index, v8_fresh_index):
    parents, records = {}, []
    for location, role, digest in (
        (v8_development_index, "development", V8_DEVELOPMENT_INDEX_SHA256),
        (v8_fresh_index, "fresh", V8_FRESH_INDEX_SHA256),
    ):
        path = Path(location).resolve()
        if v2.sha256_file(path) != digest:
            raise ValueError("V8 parent index differs from the locked exposed cohort")
        parent = json.loads(path.read_text())
        if (
            parent.get("role") != role
            or parent.get("schema_version") != 1
            or parent.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
            or parent.get("feature_definition") != FEATURE_DEFINITION
            or parent.get("context_definition") != CONTEXT_DEFINITION
            or len(parent.get("sessions", [])) != PARENT_COUNTS[role]
        ):
            raise ValueError("V8 parent role, count, provider or feature identity mismatch")
        parents[role] = {"path": str(path), "sha256": digest}
        for item in parent["sessions"]:
            if item.get("role") != role or item["participant_id"] not in ALL_PARTICIPANTS:
                raise ValueError("Parent record role or participant mismatch")
            _geometry(item)
            record = copy.deepcopy(item)
            record["role"] = "development"
            record["v9_parent"] = {
                "index_path": str(path),
                "index_sha256": digest,
                "role": role,
                "session_id": item["session_id"],
            }
            record["reused_parent_artifacts"] = {}
            for kind in ("feature", "target", "context"):
                origin = v2._safe_path(path.parent, item[kind + "_path"])
                record["reused_parent_artifacts"][kind] = {
                    "path": str(origin),
                    "sha256": item[kind + "_sha256"],
                    "mode": "immutable_hard_link",
                }
                record[kind + "_path"] = (
                    f"prepared/{item['participant_id']}/{item['session_id']}.{kind}s.npz"
                )
            record["context_path"] = (
                f"prepared/{item['participant_id']}/{item['session_id']}.contexts.npz"
            )
            record["far_distance"] = FAR_DISTANCE_BY_SESSION.get(
                item["session_id"], FAR_DISTANCE_BY_PARTICIPANT[item["participant_id"]]
            )
            records.append(record)
    if len(records) != EXPOSURE["sessions"] or len({r["session_id"] for r in records}) != len(
        records
    ):
        raise ValueError("Expected exactly 200 unique exposed sessions")
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


make_context = v8.make_context
validate_context = v8.validate_context
validate_context_targets = v8.validate_context_targets
load_context = v8.load_context


def prepare_development(cache, v8_development_index, v8_fresh_index):
    cache = Path(cache).resolve()
    if (cache / "index.json").exists():
        raise ValueError("V9 indexes are immutable; choose a new output directory")
    parents, records = _parents(v8_development_index, v8_fresh_index)
    for item in records:
        for kind in ("feature", "target", "context"):
            ref = item["reused_parent_artifacts"][kind]
            if v2.sha256_file(ref["path"]) != ref["sha256"]:
                raise ValueError("Exposed V8 prepared artifact changed")
            dest = v2._safe_path(cache, item[kind + "_path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise ValueError("Partial prepared output exists; do not overwrite")
            os.link(ref["path"], dest)
        print(json.dumps({"prepared": item["session_id"], "n_bins": item["n_bins"]}), flush=True)
    index = {
        "schema_version": 1,
        "role": "development",
        "created_at": v2._now(),
        "participants": list(ALL_PARTICIPANTS),
        "provider": {"doi": v2.DOI, "version_id": v2.VERSION_ID},
        "feature_definition": FEATURE_DEFINITION,
        "context_definition": CONTEXT_DEFINITION,
        "exposure": EXPOSURE,
        "parents": parents,
        "preparer": {"v9_data_sha256": v2.sha256_file(__file__)},
        "sessions": records,
    }
    v2._json(cache / "index.json", index)
    return index


def load_index(path, *, expected_role=None, verify_hashes=True):
    path = Path(path).resolve()
    index = json.loads(path.read_text())
    if index.get("role") == "fresh" and expected_role in (None, "fresh"):
        from neural_experiments.v9 import fresh_data

        return fresh_data.load_index(path, expected_role="fresh", verify_hashes=verify_hashes)
    if index.get("role") != "development" or expected_role not in (None, "development"):
        raise ValueError("V9 development index role mismatch")
    if (
        index.get("schema_version") != 1
        or index.get("participants") != list(ALL_PARTICIPANTS)
        or index.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
        or index.get("feature_definition") != FEATURE_DEFINITION
        or index.get("context_definition") != CONTEXT_DEFINITION
        or index.get("exposure") != EXPOSURE
    ):
        raise ValueError("V9 development contract mismatch")
    parents, records = _parents(
        index["parents"]["development"]["path"], index["parents"]["fresh"]["path"]
    )
    actual = index["sessions"]
    if index["parents"] != parents or actual != records:
        raise ValueError("V9 development record differs from its immutable V8 parent")
    if verify_hashes:
        for item in index["sessions"]:
            for kind in ("feature", "target", "context"):
                if (
                    v2.sha256_file(v2._safe_path(path.parent, item[kind + "_path"]))
                    != item[kind + "_sha256"]
                ):
                    raise ValueError("Prepared artifact hash mismatch")
    return index


def load_case(index_path, item, *, labels=False, context=False):
    root = Path(index_path).resolve().parent

    def artifact(kind):
        path = v2._safe_path(root, item[kind + "_path"])
        if v2.sha256_file(path) != item[kind + "_sha256"]:
            raise ValueError(f"{kind.capitalize()} identity mismatch")
        return path

    features = load_features(artifact("feature"))
    if len(features["x_tx45"]) != item["n_bins"]:
        raise ValueError("Feature geometry differs from authenticated record")
    result = {
        "item": item,
        "features": features,
        "segments": [(s["start"], s["end"]) for s in item["segments"] if s["end"] > s["start"]],
    }
    if labels:
        result.update(load_targets(artifact("target"), item["n_bins"]))
    if context:
        result["context"] = load_context(artifact("context"), item["n_bins"])
        if labels:
            validate_context_targets(result["context"], result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-development", "validate-development"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--v8-development-index", type=Path, default=Path("data/neural_v8/development/index.json")
    )
    parser.add_argument(
        "--v8-fresh-index", type=Path, default=Path("data/neural_v8/fresh/index.json")
    )
    args = parser.parse_args()
    index = (
        prepare_development(args.cache, args.v8_development_index, args.v8_fresh_index)
        if args.command == "prepare-development"
        else load_index(args.cache / "index.json", expected_role="development")
    )
    print(json.dumps({"role": index["role"], "sessions": len(index["sessions"])}))


if __name__ == "__main__":
    main()
