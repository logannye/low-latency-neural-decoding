"""Independently rebuild selected behavior context; publish allowlisted hashes/counts.

This audit never invokes make_context. Raw MAT variables may be mechanically
materialized; all numerical operations first slice authenticated exposed segments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from neural_experiments.v2 import data as v2
from neural_experiments.v9 import data


def independently_rebuild(raw, item):
    n = item["n_bins"]
    result = {
        k: np.zeros((n, 2), dtype=np.float64)
        for k in ("cursor_position", "target_position", "onset_direction", "reference40_direction")
    }
    for key in ("distance", "onset_distance", "reference40_distance"):
        result[key] = np.full(n, np.nan)
    for key in ("trial_id", "trial_start_bin", "trial_stop_bin"):
        result[key] = np.full(n, -1, dtype=np.int64)
    for key in ("trial_age_seconds", "trial_onset_seconds"):
        result[key] = np.full(n, -1.0)
    for key in ("trial_right_censored", "reference40_available"):
        result[key] = np.zeros(n, dtype=bool)
    trials = {
        k: np.atleast_1d(raw["trials"][k]).astype(np.int64)
        for k in ("trial_start_index", "trial_end_index", "block_id")
    }
    for segment in item["segments"]:
        a, b = segment["source_start"], segment["source_end"]
        left, right = segment["start"], segment["end"]
        cursor = np.array(raw["cursor_position"][a:b], dtype=float, copy=True)
        target = np.array(raw["target_position"][a:b], dtype=float, copy=True)
        length = b - a
        pair_right = np.arange(1, 2 * (right - left), 2)
        result["cursor_position"][left:right] = cursor[pair_right]
        result["target_position"][left:right] = target[pair_right]
        result["distance"][left:right] = np.sqrt(
            np.sum((target[pair_right] - cursor[pair_right]) ** 2, axis=1)
        )
        go = np.full(length, -1, dtype=np.int64)
        selected = np.flatnonzero(
            (trials["block_id"] == segment["block_id"])
            & (trials["trial_start_index"] < b)
            & (trials["trial_end_index"] > a)
        )
        for ordinal in selected:
            begin, end = (
                int(trials["trial_start_index"][ordinal]),
                int(trials["trial_end_index"][ordinal]),
            )
            if begin < a or end <= begin:
                raise ValueError("Invalid raw go interval")
            local_start, local_stop = begin - a, min(end, b) - a
            if np.any(go[local_start:local_stop] != -1):
                raise ValueError("Overlapping go intervals")
            go[local_start:local_stop] = ordinal
        paired = (go[pair_right] >= 0) & (go[pair_right] == go[pair_right - 1])
        result["trial_id"][left:right] = np.where(paired, go[pair_right], -1)
        for ordinal in selected:
            rows = np.flatnonzero(paired & (go[pair_right] == ordinal))
            if not len(rows):
                continue
            begin, end = (
                int(trials["trial_start_index"][ordinal]),
                int(trials["trial_end_index"][ordinal]),
            )
            destination = left + rows
            result["trial_start_bin"][destination] = int(destination[0])
            result["trial_stop_bin"][destination] = int(destination[-1]) + 1
            result["trial_age_seconds"][destination] = (a + pair_right[rows] - begin) * 0.01
            result["trial_onset_seconds"][destination] = left * 0.02 + (begin - a) * 0.01
            result["trial_right_censored"][destination] = end > b
            for offset, name in ((0, "onset"), (4, "reference40")):
                source = begin + offset
                if source >= min(end, b):
                    continue
                delta = target[source - a] - cursor[source - a]
                distance = float(np.sqrt(np.sum(delta**2)))
                result[name + "_distance"][destination] = distance
                valid = (
                    np.isfinite(delta).all()
                    and np.isfinite(distance)
                    and distance > v2.TARGET_EPSILON
                )
                if valid:
                    result[name + "_direction"][destination] = delta / distance
                if name == "reference40":
                    result["reference40_available"][destination] = valid
    return result


def audit(index_path, output):
    index_path = Path(index_path).resolve()
    index = data.load_index(index_path)
    rows = []
    for item in index["sessions"]:
        source = v2._safe_path(Path(item["raw_source_root"]), item["source"]["raw_path"])
        if v2.sha256_file(source) != item["source"]["member_sha256"]:
            raise ValueError("Raw member identity mismatch")
        raw = loadmat(
            source,
            variable_names=["cursor_position", "target_position", "trials"],
            simplify_cells=True,
        )
        rebuilt = independently_rebuild(raw, item)
        del raw
        context = data.load_context(
            v2._safe_path(index_path.parent, item["context_path"]), item["n_bins"]
        )
        for key in data.CONTEXT_KEYS:
            if not np.array_equal(context[key], rebuilt[key], equal_nan=True):
                raise ValueError(f"Independent context mismatch: {item['session_id']} {key}")
        targets = data.load_targets(
            v2._safe_path(index_path.parent, item["target_path"]), item["n_bins"]
        )
        delta = rebuilt["target_position"] - rebuilt["cursor_position"]
        distance = np.sqrt(np.sum(delta**2, axis=1))
        valid = (
            np.isfinite(delta).all(axis=1) & np.isfinite(distance) & (distance > v2.TARGET_EPSILON)
        )
        y = np.zeros(delta.shape, dtype=np.float32)
        y[valid] = (delta[valid] / distance[valid, None]).astype(np.float32)
        mask = valid & (rebuilt["trial_id"] >= 0)
        if not np.array_equal(y, targets["y"]) or not np.array_equal(mask, targets["eval_mask"]):
            raise ValueError("Independent unchanged target reconstruction mismatch")
        rows.append(
            {
                "participant_id": item["participant_id"],
                "session_id": item["session_id"],
                "post_implant_day": item["post_implant_day"],
                "n_bins": item["n_bins"],
                "n_channels": item["n_channels"],
                "n_valid_go_bins": int(mask.sum()),
                "n_observed_go_intervals": len(
                    set(rebuilt["trial_id"][rebuilt["trial_id"] >= 0].tolist())
                ),
                "member_sha256": item["source"]["member_sha256"],
                "feature_sha256": item["feature_sha256"],
                "target_sha256": item["target_sha256"],
                "context_sha256": item["context_sha256"],
                "context_matches_independent_rebuild": True,
                "unchanged_target_and_mask_match": True,
                "bounded_to_authenticated_parent_segments": True,
            }
        )
        print(item["session_id"], flush=True)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "role": index["role"],
        "index_sha256": v2.sha256_file(index_path),
        "sessions": len(rows),
        "participants": len({r["participant_id"] for r in rows}),
        "n_bins": sum(r["n_bins"] for r in rows),
        "parent_index_sha256": {
            "v8_development": data.V8_DEVELOPMENT_INDEX_SHA256,
            "v8_fresh": data.V8_FRESH_INDEX_SHA256,
        },
        "auditor_sha256": v2.sha256_file(__file__),
        "preparer": index["preparer"],
        "context_definition": data.CONTEXT_DEFINITION,
        "sample_level_values_published": False,
        "records": rows,
    }
    v2._json(Path(output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.index, args.output)


if __name__ == "__main__":
    main()
