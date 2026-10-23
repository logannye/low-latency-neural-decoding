"""Four locked V9 source candidates, then all fixed methods on exposed sessions."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from neural_experiments.v3.study import digest, relative, write_json
from neural_experiments.v6.data import _study_lock
from neural_experiments.v9 import data, model

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = "quota_sector_mean"


def trial_config(rt_seconds, ridge, *, policy="quota24", target="window_mean", weighting="sector"):
    return {
        "representation": "boxcar300",
        "policy": policy,
        "training": "trial",
        "target": target,
        "weighting": weighting,
        "rt_seconds": rt_seconds,
        "ridge": ridge,
        "control": "neural",
    }


def source_grid(plan):
    configs = {
        f"grid_{i:02d}": trial_config(entry["rt_seconds"], entry["ridge"])
        for i, entry in enumerate(plan["source_grid"])
    }
    if len(configs) != 4:
        raise ValueError("four prespecified candidates required")
    for config in configs.values():
        model.validate_config(config)
    return configs


def assessment_configs(choice):
    rt, ridge = choice["rt_seconds"], choice["ridge"]
    baseline = {**model.BASELINE, "rt_seconds": rt}
    primary = trial_config(rt, ridge)
    return {
        "baseline": baseline,
        "duration": {**baseline, "policy": "duration360"},
        "quota_continuous": {**baseline, "policy": "quota24"},
        "trial_startup_sector_mean": {**primary, "policy": "startup120"},
        "trial_duration_sector_mean": {**primary, "policy": "duration360"},
        "quota_uniform_endpoint": {**primary, "target": "endpoint", "weighting": "uniform"},
        "quota_sector_endpoint": {**primary, "target": "endpoint"},
        "quota_uniform_mean": {**primary, "weighting": "uniform"},
        PRIMARY: primary,
        "zero": {**baseline, "policy": "duration360", "control": "zero"},
        "constant": {**baseline, "policy": "duration360", "control": "intercept"},
        "trialshuffle": {**primary, "control": "trial_permutation"},
    }


def choose(rows, configurations):
    methods = list(configurations)
    sessions = sorted({row["session_id"] for row in rows})
    lookup = {(row["session_id"], row["method"]): row for row in rows}
    if not sessions or len(lookup) != len(rows) or len(rows) != len(sessions) * len(methods):
        raise ValueError("complete unique candidate-session grid required")
    scored = []
    for sid in sessions:
        group = [lookup[sid, method] for method in methods]
        if len({row["score_mask_sha256"] for row in group}) != 1:
            raise ValueError("source candidates must have identical primary masks")
        available = [row["metrics"]["r2"] is not None for row in group]
        if len(set(available)) != 1:
            raise ValueError("candidate-dependent score availability is forbidden")
        if available[0]:
            scored.append(sid)
    ranking = [
        {
            "method": method,
            "mean_r2": float(np.mean([lookup[sid, method]["metrics"]["r2"] for sid in scored]))
            if scored
            else None,
            "selected_sessions": len(sessions),
            "scored_sessions": len(scored),
            "session_r2": [
                {"session_id": sid, "r2": lookup[sid, method]["metrics"]["r2"]} for sid in sessions
            ],
        }
        for method in methods
    ]
    if scored and not np.isfinite([r["mean_r2"] for r in ranking]).all():
        raise ValueError("finite comparable development scores required")
    selected = methods[int(np.argmax([r["mean_r2"] for r in ranking]))] if scored else methods[0]
    return {
        "selected_method": selected,
        "ranking": ranking,
        "no_scorable_sessions_fallback": not scored,
        "tie_break": "declared_order",
    }


def run(index_path, output, *, plan_path=None, plan_sha256=None):
    from neural_experiments.v9.assessment import scientific_sources, source_hashes
    from neural_experiments.v9.metrics import summarize

    index_path, output = Path(index_path), Path(output)
    plan_path = Path(plan_path) if plan_path else Path(__file__).with_name("study-plan.json")
    expected = plan_sha256 or digest(plan_path)
    _study_lock(plan_path, expected)
    plan = json.loads(plan_path.read_text())
    if digest(index_path) != plan["development"]["index_sha256"]:
        raise ValueError("exact exposed development index required")
    index = data.load_index(index_path, expected_role="development", verify_hashes=True)
    participants = plan["development"]["study_participants"]
    selected = [item for item in index["sessions"] if item["participant_id"] in participants]
    selected.sort(
        key=lambda item: (
            participants.index(item["participant_id"]),
            item["source"]["archive_ordinal"],
        )
    )
    if len(index["sessions"]) != 200 or len(selected) != 96:
        raise ValueError("exact200 exposed records and96 selected model sessions required")
    for participant in participants:
        ranks = [
            item["source"]["archive_ordinal"]
            for item in selected
            if item["participant_id"] == participant
        ]
        if ranks != list(range(1, 25)):
            raise ValueError("each source participant requires archive ranks1..24")
    grid_cases = [
        item
        for item in selected
        if item["participant_id"] == "T2"
        and item["source"]["archive_ordinal"]
        in plan["development"]["source_selection_regular_ranks"]
    ]
    if len(grid_cases) != 9:
        raise ValueError("nine exposed T2 selection sessions required")
    configs = source_grid(plan)
    hashes = source_hashes()
    for source, expected_source in hashes.items():
        _study_lock(ROOT / source, expected_source)
    output.mkdir(parents=True, exist_ok=False)
    for source in scientific_sources():
        destination = output / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    shutil.copyfile(plan_path, output / "source_snapshot" / "study-plan.json")
    manifest = {
        "schema_version": 1,
        "study": "V9",
        "phase": "development",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "plan": {"path": relative(plan_path), "sha256": expected},
        "index": {"path": relative(index_path), "sha256": digest(index_path)},
        "code_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threadpool_limit": 1,
        },
        "grid_configurations": configs,
        "grid_rows": [],
        "final_rows": [],
        "sessions": [],
        "primary_method": PRIMARY,
    }
    write_json(output / "manifest.json", manifest)

    def evaluate(items, configurations, stage):
        rows = []
        for item in items:
            case = data.load_case(index_path, item, labels=True, context=True)
            artifact_dir = output / stage / item["session_id"]
            result = model.evaluate_session(case, configurations, artifact_dir=artifact_dir)
            rows.extend(result["rows"])
            manifest["sessions"].append(
                {
                    "stage": stage,
                    "session_id": item["session_id"],
                    "artifact_dir": str(artifact_dir.resolve()),
                    "fit_receipts": result["fit_receipts"],
                    "model_refs": result["model_refs"],
                    "secondary_trials": result["secondary_trials"],
                }
            )
            manifest[stage + "_rows"] = rows
            write_json(output / "manifest.json", manifest)
            print(json.dumps({"stage": stage, "session": item["session_id"]}), flush=True)
        return rows

    with threadpool_limits(limits=1):
        rows = evaluate(grid_cases, configs, "grid")
        choice = choose(rows, configs)
        chosen = configs[choice["selected_method"]]
        configurations = assessment_configs(chosen)
        if list(configurations) != plan["methods"]:
            raise ValueError("final methods differ from locked protocol order")
        manifest["grid_selection"] = choice
        manifest["configurations"] = configurations
        final_rows = evaluate(selected, configurations, "final")
        manifest["summary"] = summarize(final_rows)
    if source_hashes() != hashes or digest(plan_path) != expected:
        raise ValueError("scientific source or protocol changed during source execution")
    selection = {
        "schema_version": 1,
        "study": "V9",
        "source_audit_status": "PENDING",
        "plan_sha256": expected,
        "index_sha256": digest(index_path),
        "scientific_code_sha256": hashes,
        "code_sha256": hashes,
        "primary_method": PRIMARY,
        "configurations": configurations,
        "recipe_choice": {"method": choice["selected_method"], "config": chosen},
        "grid_selection": choice,
    }
    manifest.update(
        status="complete",
        finished_at=datetime.now(UTC).isoformat(),
        selection=selection,
        rows=manifest["grid_rows"] + manifest["final_rows"],
    )
    write_json(output / "manifest.json", manifest)
    selection["source_manifest_sha256"] = digest(output / "manifest.json")
    write_json(output / "selection-provisional.json", selection)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    run(args.index, args.output, plan_path=args.plan, plan_sha256=args.plan_sha256)
