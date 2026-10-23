"""V8's exact 176 exposed prefixes, with behavior isolated from neural features.

Compressed MAT variables may be materialized mechanically by scipy. Every
sample-valued operation below first slices to an authenticated exposed segment;
no statistics, labels or context are derived from retained recording tails.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from neural_experiments.v2 import data as v2
from neural_experiments.v6 import data as v6

ALL_PARTICIPANTS = v2.PARTICIPANTS
PARTICIPANTS = tuple(p for p in ALL_PARTICIPANTS if p != "T3")
MAX_BINS = 30000
FEATURE_KEYS = v6.FEATURE_KEYS
FEATURE_DEFINITION = v6.FEATURE_DEFINITION
V7_DEVELOPMENT_INDEX_SHA256 = "4b2c0c8b47a2caf1a2135b28dcea197a26c8f6006c19f243ff2fed23c5e1362e"
V7_FRESH_INDEX_SHA256 = "34f51825b1f1b4b7f8144fcfe93822f61295a2a013f0cd564fff610228d20e47"
PARENT_COUNTS = {"development": 152, "fresh": 24}
EXPECTED_COUNTS = {p: 8 if p == "T3" else 21 for p in ALL_PARTICIPANTS}
CONTEXT_KEYS = frozenset(
    {
        "cursor_position",
        "target_position",
        "distance",
        "trial_id",
        "trial_age_seconds",
        "trial_onset_seconds",
        "trial_start_bin",
        "trial_stop_bin",
        "trial_right_censored",
        "onset_direction",
        "onset_distance",
        "reference40_direction",
        "reference40_distance",
        "reference40_available",
    }
)
FAR_DISTANCE_BY_PARTICIPANT = {
    "T2": 0.2,
    "T3": 0.2,
    "T5": 160.0,
    "T6": 160.0,
    "T7": 0.2,
    "T8": 5.6,
    "T9": 0.2,
    "T10": 0.2,
    "T11": 0.2,
}
FAR_DISTANCE_BY_SESSION = {
    f"T5_day_{day}_decoding": 0.4 for day in (2107, 2126, 2133, 2149, 2595, 2609, 2637, 2667)
}
CONTEXT_DEFINITION = {
    "alignment": "same second 10ms source sample and within-block pairs as V7",
    "behavior_access": "separate context sidecar; never returned by feature-only loading",
    "trial_id": "provider go interval ordinal; -1 outside a paired single go interval",
    "trial_age_seconds": "(second source-bin index - provider go start index) * 0.01",
    "trial_onset_seconds": "exact 10ms go onset on the accumulated exposed 20ms clock",
    "trial_bounds": "first and exclusive last complete 20ms pair within go, clipped to exposure",
    "trial_right_censored": "provider go end extends beyond the selected exposed segment",
    "onset": "direction and distance at provider go start; zero direction if invalid",
    "reference40": (
        "direction and distance at go start+4 source bins; unavailable outside exposed go"
    ),
    "target_meaning": (
        "cursor-to-target geometry, not measured limb velocity or independent intention"
    ),
    "provider_reference_commit": "93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834",
    "far_distance_by_participant": FAR_DISTANCE_BY_PARTICIPANT,
    "far_distance_by_session": FAR_DISTANCE_BY_SESSION,
    "no_task_type_or_assistance_inference": True,
}
EXPOSURE = {
    "rule": "exact_union_of_exposed_v7_development152_and_fresh24_prefixes",
    "sessions": 176,
    "maximum_bins_per_session": MAX_BINS,
    "continuation_tails_included": False,
    "raw_decompression": (
        "mechanical materialization only; sample analysis restricted to parent segments"
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


def _parents(v7_development_index, v7_fresh_index):
    parents, records = {}, []
    for location, role, digest in (
        (v7_development_index, "development", V7_DEVELOPMENT_INDEX_SHA256),
        (v7_fresh_index, "fresh", V7_FRESH_INDEX_SHA256),
    ):
        path = Path(location).resolve()
        if v2.sha256_file(path) != digest:
            raise ValueError("V7 parent index differs from the locked exposed cohort")
        parent = json.loads(path.read_text())
        if (
            parent.get("role") != role
            or parent.get("schema_version") != 1
            or parent.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
            or parent.get("feature_definition") != FEATURE_DEFINITION
            or len(parent.get("sessions", [])) != PARENT_COUNTS[role]
        ):
            raise ValueError("V7 parent role, count, provider or feature identity mismatch")
        parents[role] = {"path": str(path), "sha256": digest}
        for item in parent["sessions"]:
            if item.get("role") != role or item["participant_id"] not in ALL_PARTICIPANTS:
                raise ValueError("Parent record role or participant mismatch")
            _geometry(item)
            record = copy.deepcopy(item)
            record["role"] = "development"
            record["v8_parent"] = {
                "index_path": str(path),
                "index_sha256": digest,
                "role": role,
                "session_id": item["session_id"],
            }
            record["reused_parent_artifacts"] = {}
            for kind in ("feature", "target"):
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
        raise ValueError("Expected exactly 176 unique exposed sessions")
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


def _direction(delta):
    distance = float(np.linalg.norm(delta))
    if np.isfinite(delta).all() and np.isfinite(distance) and distance > v2.TARGET_EPSILON:
        return delta / distance, distance
    return np.zeros(2), distance


def make_context(raw, metadata):
    """Slice selected samples first; derive no values from unexposed raw tails."""
    n = metadata["n_bins"]
    _geometry(metadata)
    context = {
        k: np.zeros((n, 2), dtype=np.float64)
        for k in ("cursor_position", "target_position", "onset_direction", "reference40_direction")
    }
    context.update(
        {
            k: np.full(n, np.nan, dtype=np.float64)
            for k in ("distance", "onset_distance", "reference40_distance")
        }
    )
    context.update(
        {
            k: np.full(n, -1, dtype=np.int64)
            for k in ("trial_id", "trial_start_bin", "trial_stop_bin")
        }
    )
    context["trial_age_seconds"] = np.full(n, -1.0)
    context["trial_onset_seconds"] = np.full(n, -1.0)
    context["trial_right_censored"] = np.zeros(n, dtype=bool)
    context["reference40_available"] = np.zeros(n, dtype=bool)
    trials = v2._struct_vectors(raw["trials"], ("trial_start_index", "trial_end_index", "block_id"))
    for segment in metadata["segments"]:
        start, stop = segment["source_start"], segment["source_end"]
        left, right = segment["start"], segment["end"]
        # The copies sever any view of a mechanically decompressed tail.
        cursor = np.asarray(raw["cursor_position"])[start:stop].astype(np.float64, copy=True)
        target = np.asarray(raw["target_position"])[start:stop].astype(np.float64, copy=True)
        if cursor.shape != (stop - start, 2) or target.shape != cursor.shape:
            raise ValueError("Selected cursor/target geometry mismatch")
        second = np.arange(1, 2 * (right - left), 2)
        context["cursor_position"][left:right] = cursor[second]
        context["target_position"][left:right] = target[second]
        context["distance"][left:right] = np.linalg.norm(target[second] - cursor[second], axis=1)
        for ordinal, (a, b, block) in enumerate(
            zip(
                trials["trial_start_index"],
                trials["trial_end_index"],
                trials["block_id"],
                strict=True,
            )
        ):
            a, b = int(a), int(b)
            if int(block) != segment["block_id"] or a >= stop or b <= start:
                continue
            if a < start or b <= a:
                raise ValueError("Go metadata crosses selected block start")
            first_pair = max(0, (a - start + 1) // 2)
            last_pair = min(right - left, (min(b, stop) - start) // 2)
            indexes = np.arange(left + first_pair, left + last_pair)
            if not len(indexes):
                continue
            if np.any(context["trial_id"][indexes] != -1):
                raise ValueError("Overlapping provider go intervals")
            context["trial_id"][indexes] = ordinal
            context["trial_start_bin"][indexes] = left + first_pair
            context["trial_stop_bin"][indexes] = left + last_pair
            context["trial_age_seconds"][indexes] = (start + 2 * (indexes - left) + 1 - a) * 0.01
            context["trial_onset_seconds"][indexes] = left * 0.02 + (a - start) * 0.01
            context["trial_right_censored"][indexes] = b > stop
            direction, distance = _direction(target[a - start] - cursor[a - start])
            context["onset_direction"][indexes] = direction
            context["onset_distance"][indexes] = distance
            reference = a + 4
            if reference < min(b, stop):
                direction, distance = _direction(
                    target[reference - start] - cursor[reference - start]
                )
                available = np.isfinite(distance) and distance > v2.TARGET_EPSILON
                context["reference40_direction"][indexes] = direction
                context["reference40_distance"][indexes] = distance
                context["reference40_available"][indexes] = available
    return context


def validate_context(context, n_bins):
    if set(context) != CONTEXT_KEYS:
        raise ValueError("Unexpected or missing behavior context fields")
    vector = {"cursor_position", "target_position", "onset_direction", "reference40_direction"}
    integers = {"trial_id", "trial_start_bin", "trial_stop_bin"}
    booleans = {"trial_right_censored", "reference40_available"}
    for key, array in context.items():
        shape = (n_bins, 2) if key in vector else (n_bins,)
        dtype = np.int64 if key in integers else bool if key in booleans else np.float64
        if array.shape != shape or array.dtype != dtype:
            raise ValueError("Invalid behavior context geometry or dtype")
    has_trial = context["trial_id"] >= 0
    indices = np.arange(n_bins)
    if (
        np.any(context["trial_id"] < -1)
        or np.any(
            has_trial
            & ((context["trial_start_bin"] > indices) | (context["trial_stop_bin"] <= indices))
        )
        or np.any(context["trial_stop_bin"] > n_bins)
    ):
        raise ValueError("Invalid paired go bounds")
    if np.any(
        has_trial & ((context["trial_age_seconds"] < 0) | (context["trial_onset_seconds"] < 0))
    ):
        raise ValueError("Negative go age")
    available = context["reference40_available"]
    if np.any(available & ~has_trial) or not np.allclose(
        np.linalg.norm(context["reference40_direction"][available], axis=1), 1
    ):
        raise ValueError("Invalid observed reference40 direction")
    return context


def load_context(path, n_bins=None):
    with np.load(path, allow_pickle=False) as archive:
        context = {k: archive[k] for k in archive.files}
    return validate_context(context, len(context["trial_id"]) if n_bins is None else n_bins)


def validate_context_targets(context, targets):
    delta = context["target_position"] - context["cursor_position"]
    distance = np.linalg.norm(delta, axis=1)
    good = np.isfinite(delta).all(axis=1) & np.isfinite(distance) & (distance > v2.TARGET_EPSILON)
    y = np.zeros(delta.shape, dtype=np.float32)
    y[good] = (delta[good] / distance[good, None]).astype(np.float32)
    if not np.array_equal(y, targets["y"]) or not np.array_equal(
        good & (context["trial_id"] >= 0), targets["eval_mask"]
    ):
        raise ValueError("Enriched context changed the authenticated V7 target or go mask")


def prepare_development(cache, v7_development_index, v7_fresh_index):
    cache = Path(cache).resolve()
    if (cache / "index.json").exists():
        raise ValueError("V8 indexes are immutable; choose a new output directory")
    parents, records = _parents(v7_development_index, v7_fresh_index)
    for item in records:
        raw_path = v2._safe_path(Path(item["raw_source_root"]), item["source"]["raw_path"])
        if v2.sha256_file(raw_path) != item["source"]["member_sha256"]:
            raise ValueError("Raw source changed from its exposed parent")
        raw = loadmat(
            raw_path,
            variable_names=["cursor_position", "target_position", "trials"],
            simplify_cells=True,
        )
        context = make_context(raw, item)
        del raw
        validate_context(context, item["n_bins"])
        for kind in ("feature", "target"):
            ref = item["reused_parent_artifacts"][kind]
            if v2.sha256_file(ref["path"]) != ref["sha256"]:
                raise ValueError("Exposed V7 prepared artifact changed")
        validate_context_targets(
            context, load_targets(item["reused_parent_artifacts"]["target"]["path"], item["n_bins"])
        )
        for kind in ("feature", "target"):
            dest = v2._safe_path(cache, item[kind + "_path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise ValueError("Partial prepared output exists; do not overwrite")
            os.link(item["reused_parent_artifacts"][kind]["path"], dest)
        dest = v2._safe_path(cache, item["context_path"])
        if dest.exists():
            raise ValueError("Partial context output exists; do not overwrite")
        np.savez_compressed(dest, **context)
        item["context_sha256"] = v2.sha256_file(dest)
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
        "preparer": {"v8_data_sha256": v2.sha256_file(__file__)},
        "sessions": records,
    }
    v2._json(cache / "index.json", index)
    return index


def load_index(path, *, expected_role=None, verify_hashes=True):
    path = Path(path).resolve()
    index = json.loads(path.read_text())
    if index.get("role") == "fresh" and expected_role in (None, "fresh"):
        from neural_experiments.v8 import fresh_data

        return fresh_data.load_index(path, expected_role="fresh", verify_hashes=verify_hashes)
    if index.get("role") != "development" or expected_role not in (None, "development"):
        raise ValueError("V8 development index role mismatch")
    if (
        index.get("schema_version") != 1
        or index.get("participants") != list(ALL_PARTICIPANTS)
        or index.get("provider") != {"doi": v2.DOI, "version_id": v2.VERSION_ID}
        or index.get("feature_definition") != FEATURE_DEFINITION
        or index.get("context_definition") != CONTEXT_DEFINITION
        or index.get("exposure") != EXPOSURE
    ):
        raise ValueError("V8 development contract mismatch")
    parents, records = _parents(
        index["parents"]["development"]["path"], index["parents"]["fresh"]["path"]
    )
    actual = [
        {k: v for k, v in item.items() if k != "context_sha256"} for item in index["sessions"]
    ]
    if index["parents"] != parents or actual != records:
        raise ValueError("V8 development record differs from its immutable V7 parent")
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
        "--v7-development-index", type=Path, default=Path("data/neural_v7/development/index.json")
    )
    parser.add_argument(
        "--v7-fresh-index", type=Path, default=Path("data/neural_v7/fresh/index.json")
    )
    args = parser.parse_args()
    index = (
        prepare_development(args.cache, args.v7_development_index, args.v7_fresh_index)
        if args.command == "prepare-development"
        else load_index(args.cache / "index.json", expected_role="development")
    )
    print(json.dumps({"role": index["role"], "sessions": len(index["sessions"])}))


if __name__ == "__main__":
    main()
