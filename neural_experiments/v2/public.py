"""Export audited V2 aggregate evidence without recording arrays or trained weights.

Both commands require a completed manifest, its exact summary, and a passed
independent audit bound to both files. Output paths must be fresh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE_KEYS = ("target_participant", "method", "seed", "budget_seconds")
IDENTITY_KEYS = (
    "model_id",
    *CASE_KEYS,
    "source_participants",
    "source_train_session_ids",
    "source_validation_session_ids",
    "calibration",
    "normalization",
    "model_sha256",
    "core_sha256",
    "training_seconds",
    "parameter_count",
)
METRIC_KEYS = ("r2", "per_output_r2", "angular_error_degrees", "zero_prediction_count", "n")
TRANSFER_AUDIT_KEYS = (
    "status",
    "models_checked",
    "traces_checked",
    "unavailable_models",
    "selected_members_checked",
    "source_cores_checked",
    "participants",
    "manifest_sha256",
    "index_sha256",
    "config_sha256",
    "audit_source_sha256",
    "summary_sha256",
    "reference",
)
RUNTIME_AUDIT_KEYS = (
    "status",
    "auditor_source_sha256",
    "manifest_sha256",
    "summary_sha256",
    "model_sha256",
    "feature_sha256",
    "source_snapshots_checked",
    "blocks_checked",
    "prediction_correctness_independently_verified",
    "prediction_check_scope",
    "clock_check_scope",
    "memory_check_scope",
)
PRIVATE_KEYS = {
    "x",
    "y",
    "timestamps",
    "predictions",
    "input_mean",
    "input_scale",
    "coefficient",
    "intercept",
    "state_dict",
    "weights",
    "source_snapshots",
    "model_path",
    "feature_path",
    "prediction_path",
    "artifact_path",
    "raw_path",
    "identity_path",
    "filepath",
    "access_token",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, expected):
    require(
        isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
        "Invalid artifact digest",
    )
    require(digest(path) == expected, f"Artifact hash mismatch: {path.name}")


def contained(root, relative):
    relative = Path(relative)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Artifact references must remain inside the repository",
    )
    result = (root / relative).resolve()
    require(result.is_relative_to(root), "Artifact reference escapes the repository")
    return result


def pick(value, keys):
    return {key: value[key] for key in keys}


def same_aggregate(first, second):
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(same_aggregate(first[k], second[k]) for k in first)
        )
    if isinstance(first, list):
        return (
            isinstance(second, list)
            and len(first) == len(second)
            and all(same_aggregate(a, b) for a, b in zip(first, second, strict=True))
        )
    if isinstance(first, float):
        return isinstance(second, (int, float)) and math.isclose(
            first, second, rel_tol=1e-9, abs_tol=1e-10
        )
    return type(first) is type(second) and first == second


def environment(value):
    """Keep runtime/library facts; never publish installed-library filesystem paths."""
    result = {
        key: value[key]
        for key in ("python", "platform", "torch", "numpy", "threads")
        if key in value
    }
    if "threadpools" in value:
        allowed = {
            "user_api",
            "internal_api",
            "num_threads",
            "prefix",
            "version",
            "threading_layer",
            "architecture",
        }
        result["threadpools"] = [
            {k: v for k, v in pool.items() if k in allowed} for pool in value["threadpools"]
        ]
    return result


def ensure_public(value):
    if isinstance(value, dict):
        require(
            not PRIVATE_KEYS.intersection(value),
            "Private array, model, path, or auth field in export",
        )
        for key, item in value.items():
            ensure_public(key)
            ensure_public(item)
    elif isinstance(value, list):
        for item in value:
            ensure_public(item)
    elif isinstance(value, str):
        require(
            not value.startswith(("/", "~", "file://"))
            and re.match(r"^[A-Za-z]:[\\/]", value) is None
            and "/Users/" not in value
            and "/home/" not in value,
            "Absolute/private filesystem path in export",
        )
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite aggregate value")


def checked_run(run, audit_path, status):
    manifest_path, summary_path = run / "manifest.json", run / "summary.json"
    manifest, summary, audit = read(manifest_path), read(summary_path), read(audit_path)
    require(
        manifest["status"] == summary["status"] == "complete", "Cannot export an incomplete run"
    )
    require(audit["status"] == status, "Independent audit did not pass")
    verify(manifest_path, audit["manifest_sha256"])
    verify(summary_path, audit["summary_sha256"])
    require(
        summary["manifest_sha256"] == audit["manifest_sha256"],
        "Summary/audit manifest identity mismatch",
    )
    return manifest, summary, audit


def common_fields(manifest, audit, run, audit_path):
    source = manifest["source"]
    for name, value in source.items():
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Private or invalid source identity",
        )
        require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid source digest")
    public_environment = environment(manifest["environment"])
    return {
        "schema_version": 1,
        "status": "complete",
        "source": source,
        "manifest_sha256": audit["manifest_sha256"],
        "summary_sha256": audit["summary_sha256"],
        "audit_sha256": digest(audit_path),
        "export_source_sha256": digest(Path(__file__)),
        "environment": public_environment,
        "environment_sha256": canonical_digest(manifest["environment"]),
        "public_environment_sha256": canonical_digest(public_environment),
        "environment_hash_definition": "SHA256 of sorted-key compact JSON; "
        "original and sanitized separately",
        "run_name": run.name,
    }


def export_transfer(run, audit_path, root):
    manifest, summary, audit = checked_run(run, audit_path, "passed")
    for key in ("index_sha256", "config_sha256"):
        require(manifest[key] == audit[key], f"Audit {key} differs from run")
    for key in ("rows", "groups", "primary_contrast"):
        require(
            same_aggregate(summary[key], audit["independent_summary"][key]),
            f"Public {key} differs from independently reconstructed summary",
        )
    scores = {}
    for row in audit["independent_scores"]:
        key = tuple(row[k] for k in (*CASE_KEYS, "session_id", "partition"))
        require(key not in scores, "Duplicate independently scored trace")
        scores[key] = row["metrics"]
    models, used = [], set()
    for reference in manifest["models"]:
        path = contained(root, reference["identity_path"])
        verify(path, reference["identity_sha256"])
        identity = read(path)
        model = pick(identity, IDENTITY_KEYS)
        model["calibration"] = pick(
            identity["calibration"],
            (
                "session_id",
                "start_bin",
                "stop_bin",
                "eligible_count",
                "mask_sha256",
                "training_count",
                "training_mask_sha256",
            ),
        )
        model["normalization"] = pick(
            identity["normalization"], ("session_id", "start_bin", "stop_bin")
        )
        model["identity_sha256"] = reference["identity_sha256"]
        model["traces"] = []
        for trace in identity["traces"]:
            key = tuple(identity[k] for k in CASE_KEYS) + (trace["session_id"], trace["partition"])
            require(
                key in scores and key not in used, "Missing or repeated independently scored trace"
            )
            used.add(key)
            metrics = pick(scores[key], METRIC_KEYS)
            if "unavailable_reason" in scores[key]:
                metrics["unavailable_reason"] = scores[key]["unavailable_reason"]
            model["traces"].append(
                {
                    **pick(
                        trace,
                        (
                            "session_id",
                            "partition",
                            "start_bin",
                            "stop_bin",
                            "prediction_sha256",
                            "score_mask_sha256",
                        ),
                    ),
                    "metrics": metrics,
                }
            )
        models.append(model)
    require(used == set(scores), "Unmatched independent score rows")
    require(
        len(models) == audit["models_checked"] and len(used) == audit["traces_checked"],
        "Audit/model/trace inventory mismatch",
    )
    require(
        len(manifest["unavailable"]) == audit["unavailable_models"],
        "Unavailable-case audit inventory mismatch",
    )
    pretraining = []
    for reference in manifest["pretraining"]:
        path = contained(root, reference["receipt_path"])
        verify(path, reference["receipt_sha256"])
        receipt = read(path)
        pretraining.append(
            {
                **pick(
                    reference, ("target_participant", "seed", "training_seconds", "receipt_sha256")
                ),
                **pick(
                    receipt,
                    (
                        "source_people",
                        "source_train_session_ids",
                        "source_validation_session_ids",
                        "source_normalization",
                        "source_normalization_sha256",
                        "selected_step",
                        "checkpoints",
                        "core_sha256",
                    ),
                ),
            }
        )
    result = {
        **common_fields(manifest, audit, run, audit_path),
        "kind": "participant_transfer",
        "config_sha256": manifest["config_sha256"],
        "index_sha256": manifest["index_sha256"],
        **pick(summary, ("rows", "groups", "primary_contrast", "unavailable")),
        "models": models,
        "pretraining": pretraining,
        "independent_audit": pick(audit, TRANSFER_AUDIT_KEYS),
        "scope": "Nine-person bounded offline cursor-intention study "
        "with target-prefix calibration; "
        "not zero-shot transfer, live control, or population-level clinical evidence",
    }
    ensure_public(result)
    return result


def export_runtime(run, audit_path, root):
    manifest, summary, audit = checked_run(run, audit_path, "pass")
    require(
        {k: v for k, v in summary.items() if k != "manifest_sha256"} == manifest,
        "Runtime summary differs from completed manifest",
    )
    require(
        audit["model_sha256"] == manifest["model_sha256"]
        and audit["feature_sha256"] == manifest["feature_sha256"],
        "Runtime audit model/feature identity mismatch",
    )
    require(
        len(audit["blocks"]) == len(manifest["blocks"]) == audit["blocks_checked"],
        "Runtime audit block inventory mismatch",
    )
    blocks = []
    for block, verified in zip(manifest["blocks"], audit["blocks"], strict=True):
        for key, value in verified.items():
            if isinstance(value, float):
                require(
                    math.isclose(block[key], value, rel_tol=0, abs_tol=1e-12),
                    f"Runtime aggregate differs from independent audit: {key}",
                )
            else:
                require(block[key] == value, f"Runtime audit block identity mismatch: {key}")
        blocks.append(
            {
                **verified,
                **pick(
                    block,
                    (
                        "artifact_sha256",
                        "initialization_ms",
                        "persistent_numeric_state_bytes",
                        "state_measurement",
                    ),
                ),
            }
        )
    result = {
        **common_fields(manifest, audit, run, audit_path),
        **pick(
            manifest,
            (
                "kind",
                "model_sha256",
                "feature_sha256",
                "runtime_source_sha256",
                "protocol",
                "parameter_count",
                "numeric_weight_bytes",
                "full_trace_parity",
            ),
        ),
        "blocks": blocks,
        "independent_audit": pick(audit, RUNTIME_AUDIT_KEYS),
        "scope": "Feature-ready measured replay. Service time, scheduler queue, and completion "
        "latency are distinct; upstream acquisition causality "
        "and hard real-time guarantees unverified",
    }
    ensure_public(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("transfer", "runtime"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    require(not args.output.exists(), "Public output must be fresh; preserve prior receipts")
    function = export_transfer if args.kind == "transfer" else export_runtime
    result = function(args.run.resolve(), args.audit.resolve(), args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest(args.output),
                "kind": result["kind"],
                "status": result["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
