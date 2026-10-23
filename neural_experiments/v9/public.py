"""Allowlisted aggregate publication. Never copy trace arrays or model coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from neural_experiments.v3.study import digest, write_json

ROW_FIELDS = (
    "participant",
    "session_id",
    "post_implant_day",
    "method",
    "config",
    "metrics",
    "secondary",
    "score_mask_sha256",
    "n_bins",
    "valid_rows",
    "calibration",
    "planned_label_seconds",
    "requested_label_seconds",
    "available_label_seconds",
    "quota_met",
    "stop_reason",
    "eligible_trials",
    "sector_trial_counts",
    "trial_deficit",
    "sector_deficits",
    "label_stop_bin",
    "deployment_bin",
    "excluded_purchased_trial_count",
)
TRAINING_FIELDS = (
    "normalization_start_bin",
    "normalization_stop_bin",
    "last_training_bin",
    "training_rows",
    "training_trials",
    "training_mask_sha256",
    "training_indices_sha256",
    "training_weights_sha256",
    "training_labels_sha256",
    "weight_sum",
    "effective_sample_size",
    "fallback_zero",
    "feature_penalty",
    "free_intercept",
    "export_prediction_max_error",
)


def payload_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _scalar(value):
    if value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("Public aggregate scalar cannot contain sample arrays or objects")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Public numeric values must be finite or null")
    if isinstance(value, str) and (
        value.startswith("/")
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or any(
            marker in value
            for marker in (
                "/Users/",
                "/home/",
                "/private/",
                "https://",
                "http://",
                "file://",
                "X-Amz-",
                "Signature=",
                "Credential=",
                "Security-Token=",
                "\\",
            )
        )
    ):
        raise ValueError("Private path or transport URL in public field")
    return value


def _array(value, size):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError("Public aggregate vector has undeclared shape")
    return [_scalar(item) for item in value]


def _object(value, fields):
    if not isinstance(value, dict) or set(value) - set(fields):
        raise ValueError("Unknown field in nested public aggregate schema")
    return {key: fields[key](entry) for key, entry in value.items()}


def _fields(names, **structured):
    return {**dict.fromkeys(names.split(), _scalar), **structured}


def _records(value, transform):
    if not isinstance(value, list):
        raise ValueError("Public aggregate records must be a list")
    return [transform(row) for row in value]


def _hashes(value):
    if not isinstance(value, dict) or any(
        re.fullmatch(r"neural_experiments/v[0-9]+/[a-z_]+\.py", key) is None
        or not isinstance(sha, str)
        or re.fullmatch("[0-9a-f]{64}", sha) is None
        for key, sha in value.items()
    ):
        raise ValueError("Only relative scientific source names and SHA256 identities are public")
    return dict(value)


def _config(value):
    return _object(
        value, _fields("representation policy training target weighting rt_seconds ridge control")
    )


def _configurations(value):
    if not isinstance(value, dict) or any(
        re.fullmatch(
            r"(?:grid_[0-9]{2}|baseline|duration|quota_continuous|"
            r"trial_startup_sector_mean|trial_duration_sector_mean|"
            r"quota_uniform_endpoint|quota_sector_endpoint|quota_uniform_mean|"
            r"quota_sector_mean|zero|constant|trialshuffle)",
            key,
        )
        is None
        for key in value
    ):
        raise ValueError("Unknown public method")
    return {key: _config(config) for key, config in value.items()}


def _metrics(value):
    return _object(
        value,
        _fields(
            "r2 n summed_mse angular_error_degrees prediction_norm_rms "
            "zero_prediction_count trials",
            per_output_r2=lambda v: _array(v, 2),
            sse=lambda v: _array(v, 2),
            sst=lambda v: _array(v, 2),
        ),
    )


def _secondary(value):
    return _object(
        value,
        _fields(
            "score_mask_sha256 trials prediction_definition",
            window_mean=_metrics,
            endpoint=_metrics,
        ),
    )


def _calibration(value):
    return _object(
        value,
        _fields(
            "policy planned_label_seconds requested_label_seconds observed_label_seconds "
            "stop_bin quota_attained timed_out eligible_trials",
            sector_counts=lambda v: _array(v, 4),
            sector_deficits=lambda v: _array(v, 4),
        ),
    )


def _row(value):
    fields = _fields(
        " ".join(ROW_FIELDS),
        config=_config,
        metrics=_metrics,
        secondary=_secondary,
        calibration=_calibration,
        sector_trial_counts=lambda v: _array(v, 4),
        sector_deficits=lambda v: _array(v, 4),
    )
    result = _object({k: value[k] for k in ROW_FIELDS if k in value}, fields)
    result["training"] = _object(
        {k: value["training"][k] for k in TRAINING_FIELDS if k in value.get("training", {})},
        dict.fromkeys(TRAINING_FIELDS, _scalar),
    )
    return result


def _grid_selection(value):
    rank = _fields(
        "method mean_r2 selected_sessions scored_sessions",
        session_r2=lambda v: _records(v, lambda r: _object(r, _fields("session_id r2"))),
    )
    return _object(
        value,
        _fields(
            "selected_method no_scorable_sessions_fallback tie_break",
            ranking=lambda v: _records(v, lambda r: _object(r, rank)),
        ),
    )


def _selection(value):
    if value is None:
        return None
    return _object(
        value,
        _fields(
            "schema_version study source_audit_status plan_sha256 index_sha256 primary_method "
            "source_manifest_sha256 source_audit_sha256",
            scientific_code_sha256=_hashes,
            code_sha256=_hashes,
            configurations=_configurations,
            recipe_choice=lambda v: _object(v, _fields("method", config=_config)),
            grid_selection=_grid_selection,
        ),
    )


def _summary(value):
    if value is None:
        return None
    from neural_experiments.v9.metrics import CONTRASTS, METHODS

    participant = _fields(
        "participant selected_sessions scored_sessions mean_r2 positive_sessions mean_summed_mse",
        session_bootstrap_95=lambda v: _array(v, 2),
        sessions=lambda v: _records(
            v, lambda r: _object(r, _fields("session_id post_implant_day", metrics=_metrics))
        ),
    )
    method = _fields(
        "mean_r2 scored_sessions n_participants non_T2_mean_r2",
        participant_bootstrap_95=lambda v: _array(v, 2),
        participants=lambda v: _records(v, lambda r: _object(r, participant)),
    )
    comparison_person = _fields(
        "participant paired_sessions mean_gain",
        session_bootstrap_95=lambda v: _array(v, 2),
        session_gains=lambda v: _records(v, lambda r: _object(r, _fields("session_id gain"))),
    )
    comparison = _fields(
        "first second mean_gain",
        paired_participant_95=lambda v: _array(v, 2),
        participants=lambda v: _records(v, lambda r: _object(r, comparison_person)),
    )
    return _object(
        value,
        _fields(
            "selected_sessions primary_method interval_limit",
            methods=lambda v: _object(v, {name: lambda r: _object(r, method) for name in METHODS}),
            contrasts=lambda v: _object(
                v, {name: lambda r: _object(r, comparison) for name in CONTRASTS}
            ),
            gates=lambda v: _object(
                v, _fields("useful_T2_recovery incremental_benefit_over_more_labels")
            ),
        ),
    )


def aggregate(manifest, audit):
    if manifest.get("study") != "V9" or manifest.get("status") != "complete":
        raise ValueError("complete V9 study required")
    if manifest.get("phase") not in ("development", "assessment"):
        raise ValueError("explicit development or assessment phase required")
    if audit.get("status") != "PASS" or audit.get("phase") != manifest["phase"]:
        raise ValueError("independent audit must pass for the same phase")
    result = {
        "schema_version": 1,
        "study": "V9",
        "phase": manifest["phase"],
        "plan_sha256": _scalar(manifest["plan"]["sha256"]),
        "index_sha256": _scalar(manifest["index"]["sha256"]),
        "scientific_code_sha256": _hashes(manifest["code_sha256"]),
        "primary_method": _scalar(manifest.get("primary_method")),
        "configurations": _configurations(manifest.get("configurations", {})),
        "summary": _summary(manifest.get("summary")),
        "selection": _selection(manifest.get("selection")),
        "rows": [_row(row) for row in manifest.get("final_rows", manifest["rows"])],
        "grid_configurations": _configurations(manifest.get("grid_configurations", {})),
        "grid_selection": _grid_selection(manifest["grid_selection"])
        if "grid_selection" in manifest
        else None,
        "grid_rows": [_row(row) for row in manifest.get("grid_rows", [])],
        "evidence_role": (
            "Previously exposed sessions; source development only"
            if manifest["phase"] == "development"
            else "Fresh fixed archive ranks from known participants; T2-selected recipe"
        ),
    }
    return result


def publish(manifest_path, audit_path, output):
    manifest = json.loads(Path(manifest_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("audit must bind this exact manifest")
    result = aggregate(manifest, audit)
    result["private_manifest_sha256"] = digest(manifest_path)
    result["audit_sha256"] = digest(audit_path)
    write_json(output, result)
    return result


def seal_selection(manifest_path, audit_path, output):
    """Bind audited source choices before committing them for fresh execution."""
    manifest = json.loads(Path(manifest_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if manifest.get("phase") != "development":
        raise ValueError("only development produces a selection")
    aggregate(manifest, audit)
    if audit.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("selection audit must bind this exact manifest")
    selection = dict(manifest["selection"])
    if audit.get("selection_sha256") != payload_sha256(selection):
        raise ValueError("source audit must bind the exact selected recipe payload")
    selection.update(
        source_audit_status="PASS",
        source_manifest_sha256=digest(manifest_path),
        source_audit_sha256=digest(audit_path),
        source_audit_path=str(
            Path(audit_path).resolve().relative_to(Path(__file__).resolve().parents[2])
        ),
    )
    write_json(output, selection)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()
    publish(args.manifest, args.audit, args.output)
    if args.selection_output:
        seal_selection(args.manifest, args.audit, args.selection_output)
