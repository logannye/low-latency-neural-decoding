"""Allowlisted V8 cohort metadata; reads JSON metadata only, never sample arrays."""

from __future__ import annotations

import json
from pathlib import Path

from neural_experiments.v2.data import _json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/neural_v8/fresh"
DEST = ROOT / "neural_experiments/results/v8-cohort.json"
PARTICIPANTS = ("T2", "T5", "T6", "T7", "T8", "T9", "T10", "T11")


def main():
    index_path = CACHE / "index.json"
    index = json.loads(index_path.read_text())
    ledger_path = CACHE / "parallel-acquisition.json"
    ledger = json.loads(ledger_path.read_text())
    if ledger["status"] != "complete" or index["role"] != "fresh":
        raise ValueError(
            "Only completed fixed acquisition and prepared fresh index are publishable"
        )
    archives = []
    receipts = {}
    for participant in PARTICIPANTS:
        path = CACHE / "acquisition" / f"{participant}.json"
        r = json.loads(path.read_text())
        receipts[participant] = r
        archives.append(
            {
                "participant_id": participant,
                "receipt_sha256": sha256_file(path),
                "archive": {
                    k: r["archive"][k] for k in ("file_id", "filename", "size", "provider_sha256")
                },
                "status": r["status"],
                "compressed_prefix_bytes": r["compressed_prefix_bytes"],
                "compressed_prefix_sha256": r["compressed_prefix_sha256"],
                "full_archive_sha256_verified": r["full_archive_sha256_verified"],
                "verified_prior_member_count": len(r["verified_prior_members"]),
                "selected_ranks": [s["regular_session_rank"] for s in r["selected"]],
                "missing_requested_ranks": r["missing_requested_ranks"],
                "tar_end_observed": r["tar_end_observed"],
                "transport_map_sha256": r["archive"]
                .get("transport", {})
                .get("private_url_file_sha256"),
            }
        )
    sessions = []
    for s in index["sessions"]:
        potential = sum(
            max(0, min(segment["end"], 30000) - max(segment["start"] + 62, 7600))
            for segment in s["segments"]
        )
        source = s["source"]
        sessions.append(
            {
                "participant_id": s["participant_id"],
                "session_id": s["session_id"],
                "post_implant_day": s["post_implant_day"],
                "regular_archive_rank": source["regular_session_rank"],
                "archive_ordinal": source["archive_ordinal"],
                "member_size_bytes": source["member_size"],
                "member_sha256": source["member_sha256"],
                "feature_sha256": s["feature_sha256"],
                "target_sha256": s["target_sha256"],
                "context_sha256": s["context_sha256"],
                "n_bins": s["n_bins"],
                "n_channels": s["n_channels"],
                "prepared_observation_seconds": s["n_bins"] * 0.02,
                "n_eval_bins_full_prefix": s["n_eval_bins"],
                "full_prepared_n_bins_from_metadata": s["full_prepared_n_bins"],
                "maximum_potential_shared_scoring_bins_from_geometry": potential,
                "potentially_scorable_by_geometry": potential >= 50,
                "scoring_potential_is_not_valid_go_confirmation": True,
                "segment_count": len(s["segments"]),
                "prepared_role": s["role"],
            }
        )
    failures = []
    for path in sorted((CACHE / "failures").glob("T*.json")):
        r = json.loads(path.read_text())
        failures.append(
            {
                "participant_id": r["participant_id"],
                "error_type": r.get("type", r.get("failure", {}).get("type")),
                "status": r["status"],
                "compressed_prefix_bytes": r.get("compressed_prefix_bytes", 0),
                "receipt_sha256": sha256_file(path),
            }
        )
    orchestrations = []
    for path in [*sorted((CACHE / "failures").glob("parallel-*.json")), ledger_path]:
        r = json.loads(path.read_text())
        orchestrations.append(
            {
                k: r.get(k)
                for k in (
                    "status",
                    "started_at",
                    "finished_at",
                    "compressed_bytes_before_orchestration",
                    "compressed_bytes_actual_total",
                    "compressed_bytes_claimed_pending",
                    "orchestrator_sha256",
                    "importer_sha256",
                    "maximum_parallel_workers",
                )
            }
        )
    if any(r["compressed_bytes_claimed_pending"] for r in orchestrations):
        raise ValueError("Cannot publish incomplete transfer accounting")
    if {r["participant_id"] for r in archives} != set(PARTICIPANTS):
        raise ValueError("Missing fixed participant")
    if any(r["verified_prior_member_count"] != 21 for r in archives):
        raise ValueError("Incomplete authenticated prior21")
    if any(r["regular_archive_rank"] not in (22, 23, 24) for r in sessions):
        raise ValueError("Unplanned rank")
    expected = {s["session_id"]: s for receipt in receipts.values() for s in receipt["selected"]}
    if len(index["sessions"]) != len(expected) or {
        s["session_id"] for s in index["sessions"]
    } != set(expected):
        raise ValueError("Prepared inventory differs from fixed acquired members")
    for session in index["sessions"]:
        if any(
            session["source"].get(key) != value
            for key, value in expected[session["session_id"]].items()
        ):
            raise ValueError("Prepared source differs from acquired member")
    dispositions = [
        {
            "participant_id": participant,
            "selected_sessions": len(receipts[participant]["selected"]),
            "missing_requested_ranks": receipts[participant]["missing_requested_ranks"],
        }
        for participant in PARTICIPANTS
    ]
    if index["participant_dispositions"] != dispositions:
        raise ValueError("Participant dispositions differ from fixed receipt inventory")
    for receipt in receipts.values():
        selected = [s["regular_session_rank"] for s in receipt["selected"]]
        if (
            receipt["status"] != "complete"
            or selected != [22, 23, 24][: len(selected)]
            or receipt["missing_requested_ranks"] != [22, 23, 24][len(selected) :]
            or (len(selected) < 3 and not receipt["tar_end_observed"])
        ):
            raise ValueError("Invalid fixed-rank shortage disposition")
    if ledger["compressed_bytes_actual_total"] != sum(
        r["compressed_prefix_bytes"] for r in archives
    ) + sum(r["compressed_prefix_bytes"] for r in failures):
        raise ValueError("Archive attempt accounting requires investigation")
    observation_path = CACHE / "transport-observations.json"
    observations = json.loads(observation_path.read_text()) if observation_path.exists() else None
    result = {
        "schema_version": 1,
        "study": "V8 target, coverage and representation",
        "scope": (
            "Metadata only. Fixed new archive sessions from eight known participants; "
            "no raw samples, model outcomes, coefficients, transport URLs or local absolute paths."
        ),
        "provider": {"doi": "10.5061/dryad.x0k6djj1h", "version_id": 462702, "license": "CC0-1.0"},
        "selection": {
            "participants": list(PARTICIPANTS),
            "regular_archive_ranks": [22, 23, 24],
            "requested_session_count": 24,
            "duration_or_outcome_used_for_selection": False,
            "missing_cases": "retain explicit archive shortage; no substitution",
            "maximum_prefix_seconds": 600,
            "T3_continuations_included": False,
        },
        "acquisition_plan_sha256": index["acquisition_plan_sha256"],
        "study_plan_sha256": index["study_lock"]["sha256"],
        "fresh_index_sha256": sha256_file(index_path),
        "exposed_development_index_sha256": sha256_file(
            ROOT / "data/neural_v8/development/index.json"
        ),
        "parent_v7_development_index_sha256": index["v7_development_index_sha256"],
        "parent_v7_fresh_index_sha256": index["v7_fresh_index_sha256"],
        "metadata_exporter_sha256": sha256_file(__file__),
        "acquisition_ledger_sha256": sha256_file(ledger_path),
        "acquisition": {
            k: ledger[k]
            for k in (
                "status",
                "started_at",
                "finished_at",
                "maximum_compressed_bytes_total",
                "maximum_retained_raw_bytes_total",
                "maximum_parallel_workers",
                "compressed_bytes_actual_total",
                "raw_bytes_actual_final",
                "orchestrator_sha256",
                "importer_sha256",
            )
        },
        "orchestration_attempts": orchestrations,
        "failed_archive_attempts": failures,
        "separate_public_link_resolution_attempts": {
            **{
                key: observations[key]
                for key in (
                    "count",
                    "http_status",
                    "archive_body_bytes",
                    "ordinary_browser_validation",
                )
            },
            "receipt_sha256": sha256_file(observation_path),
        }
        if observations is not None
        else {"disposition": "not_recorded"},
        "archives": archives,
        "participant_dispositions": index["participant_dispositions"],
        "selected_session_count": len(sessions),
        "recorded_bins": sum(s["n_bins"] for s in sessions),
        "sessions": sessions,
        "scoring_potential_definition": (
            "Only observed-bin geometry and 62-bin acquisition-block burn are considered. "
            "Trial overlap purges and phase/target validity are not included in this upper bound. "
            "Actual valid-go rows after 152s and model results belong to "
            "the separately audited study."
        ),
        "source_identity": {
            "acquisition_source_git_revision": "755fe45",
            "scientific_producer_freeze_git_revision": "f669eed",
            "importer_bytes_unchanged_since_acquisition": sha256_file(
                ROOT / "neural_experiments/v8/fresh_data.py"
            )
            == ledger["importer_sha256"],
            "orchestrator_bytes_unchanged_since_acquisition": sha256_file(
                ROOT / "neural_experiments/v8/acquire_parallel.py"
            )
            == ledger["orchestrator_sha256"],
        },
        "limits": [
            "Fresh sessions from known participants, not untouched-person validation.",
            "Partial streaming authenticates member hashes and prior prefix; "
            "full provider archive hashes are not verified.",
            "Exposure capped at first-min(600 seconds,duration); "
            "original retained raw archive members can be longer.",
        ],
    }
    text = json.dumps(result, indent=2, allow_nan=False)
    for marker in (
        "/Users/",
        "/home/",
        "/private/",
        "https://",
        "http://",
        "X-Amz-",
        "Signature=",
        "Credential=",
        "Security-Token=",
    ):
        if marker in text:
            raise ValueError("Forbidden private path, URL or transport field")

    def scan(value):
        if isinstance(value, dict):
            for key, entry in value.items():
                if key in (
                    "y",
                    "prediction",
                    "weights",
                    "raw_path",
                    "feature_path",
                    "target_path",
                    "download_url",
                    "object_url",
                    "eval_mask",
                    "context_path",
                    "cursor_position",
                    "target_position",
                    "trial_id",
                    "reference40_direction",
                ):
                    raise ValueError("Sample-bearing or private field")
                scan(entry)
        elif isinstance(value, list):
            for entry in value:
                scan(entry)

    scan(result)
    _json(DEST, result)
    print(
        json.dumps(
            {
                "public_cohort": str(DEST.relative_to(ROOT)),
                "sha256": sha256_file(DEST),
                "sessions": len(sessions),
                "arrays_read": False,
                "sanitization": "explicit allowlists plus recursive forbidden-key and value checks",
            }
        )
    )
    return result


if __name__ == "__main__":
    main()
