"""Two-stage, exposed-data V8 selection; never opens a fresh-role index.

T2 archive ranks1–15 choose one representation. Ranks16–21 choose one of eight
fixed factorial cells. All are previously exposed development data. An audited,
published selection is required separately before the one fresh assessment.
"""

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
from neural_experiments.v8 import data, model

ROOT = Path(__file__).resolve().parents[2]


def representation_grid(plan):
    """Exact protocol order controls deterministic ties, including fallback."""
    configs, identities = {}, {}
    for position, entry in enumerate(plan["representation_grid"]):
        method = f"representation_{position:02d}"
        config = {
            **model.BASELINE,
            **{key: entry[key] for key in ("family", "taus_seconds", "ridge")},
        }
        model.validate_config(config)
        configs[method], identities[method] = config, entry["candidate_id"]
    if len(configs) != 8 or len(set(identities.values())) != 8:
        raise ValueError("eight uniquely identified fixed representation choices required")
    return configs, identities


def factorial_configs(representation_choice):
    """Eight semantic cells remain even if selected representation is baseline."""
    configs = {}
    for representation_name, representation in (
        ("baseline", model.BASELINE),
        ("selected", representation_choice),
    ):
        for phase in ("all", "clean"):
            for weighting in ("uniform", "equal_trial"):
                method = f"{representation_name}_{phase}_{weighting}"
                if representation_name == "baseline" and phase == "all" and weighting == "uniform":
                    method = "baseline"
                configs[method] = {
                    **model.BASELINE,
                    **{key: representation[key] for key in ("family", "taus_seconds", "ridge")},
                    "phase": phase,
                    "weighting": weighting,
                }
    return configs


def assessment_configs(factorials, primary_method):
    primary = factorials[primary_method]
    return {
        **factorials,
        "zero": {**model.BASELINE, "control": "zero"},
        "constant": {**model.BASELINE, "control": "intercept"},
        "trialshuffle": {**primary, "control": "trial_permutation"},
        "primary_onset": {**primary, "phase": "clean", "target": "onset"},
    }


def choose(rows, configurations):
    """Matched scorable sessions, equal session weight, first exact maximum."""
    methods = list(configurations)
    session_ids = sorted({row["session_id"] for row in rows})
    lookup = {(row["session_id"], row["method"]): row for row in rows}
    if not session_ids or len(lookup) != len(rows) or len(rows) != len(session_ids) * len(methods):
        raise ValueError("complete unique candidate-session grid required")
    scored = []
    for sid in session_ids:
        group = [lookup[sid, method] for method in methods]
        if len({r["score_mask_sha256"] for r in group}) != 1:
            raise ValueError("representation and factorial comparisons require identical masks")
        availability = [r["metrics"]["r2"] is not None for r in group]
        if len(set(availability)) != 1:
            raise ValueError("candidate-dependent score availability is forbidden")
        if availability[0]:
            scored.append(sid)
    ranking = [
        {
            "method": method,
            "mean_r2": float(np.mean([lookup[sid, method]["metrics"]["r2"] for sid in scored]))
            if scored
            else None,
            "scored_sessions": len(scored),
            "selected_sessions": len(session_ids),
            "session_r2": [
                {"session_id": sid, "r2": lookup[sid, method]["metrics"]["r2"]}
                for sid in session_ids
            ],
        }
        for method in methods
    ]
    if not scored:
        selected = methods[0]
    else:
        values = [r["mean_r2"] for r in ranking]
        if not np.isfinite(values).all():
            raise ValueError("finite comparable development scores required")
        selected = methods[int(np.argmax(values))]
    return {
        "selected_method": selected,
        "ranking": ranking,
        "no_scorable_sessions_fallback": not scored,
        "tie_break": "declared_order",
    }


def run(index_path, output, *, plan_path=None, plan_sha256=None):
    # Delayed import avoids an assessment/development import cycle while sharing
    # one exact transitive source inventory for source and fresh execution.
    from neural_experiments.v8.assessment import scientific_sources, source_hashes

    index_path, output = Path(index_path), Path(output)
    plan_path = Path(plan_path) if plan_path else Path(__file__).with_name("study-plan.json")
    expected = plan_sha256 or digest(plan_path)
    _study_lock(plan_path, expected)
    plan = json.loads(plan_path.read_text())
    index = data.load_index(index_path, expected_role="development", verify_hashes=True)
    t2 = sorted(
        [item for item in index["sessions"] if item["participant_id"] == "T2"],
        key=lambda item: item["source"]["archive_ordinal"],
    )
    if len(index["sessions"]) != 176 or len(t2) != 21:
        raise ValueError("exact exposed176-session cohort and21 T2 sessions required")
    if len({item["source"]["archive_ordinal"] for item in t2}) != 21:
        raise ValueError("unique ordered archive members required")
    representation_configs, representation_ids = representation_grid(plan)
    hashes = source_hashes()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_snapshot").mkdir()
    for source in scientific_sources():
        destination = output / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    shutil.copyfile(plan_path, output / "source_snapshot" / "study-plan.json")
    manifest = {
        "schema_version": 1,
        "study": "V8",
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
        "representation_configurations": representation_configs,
        "representation_candidate_ids": representation_ids,
        "representation_rows": [],
        "factorial_rows": [],
        "diagnostic_rows": [],
        "sessions": [],
    }
    write_json(output / "manifest.json", manifest)

    def evaluate(items, configs, phase):
        rows = []
        for item in items:
            case = data.load_case(index_path, item, labels=True, context=True)
            result = model.evaluate_session(
                case, configs, artifact_dir=output / phase / item["session_id"]
            )
            rows.extend(result["rows"])
            manifest["sessions"].append(
                {
                    "stage": phase,
                    "session_id": item["session_id"],
                    "fit_receipts": result["fit_receipts"],
                    "model_refs": result["model_refs"],
                }
            )
            manifest[phase + "_rows"] = rows
            write_json(output / "manifest.json", manifest)
            print(json.dumps({"stage": phase, "session": item["session_id"]}), flush=True)
        return rows

    with threadpool_limits(limits=1):
        rows = evaluate(t2[:15], representation_configs, "representation")
        representation_selection = choose(rows, representation_configs)
        representation_method = representation_selection["selected_method"]
        selected_representation = representation_configs[representation_method]
        manifest["representation_selection"] = representation_selection
        factorials = factorial_configs(selected_representation)
        manifest["factorial_configurations"] = factorials
        rows = evaluate(t2[15:], factorials, "factorial")
        factorial_selection = choose(rows, factorials)
        manifest["factorial_selection"] = factorial_selection
        primary = factorial_selection["selected_method"]
        configurations = assessment_configs(factorials, primary)
        # Controls and alternative target are diagnostic only; these outcomes
        # cannot change either previously selected identity.
        controls = {key: value for key, value in configurations.items() if key not in factorials}
        evaluate(t2[15:], controls, "diagnostic")
    if source_hashes() != hashes or digest(plan_path) != expected:
        raise ValueError("scientific source or protocol changed during development")
    selection = {
        "schema_version": 1,
        "study": "V8",
        "source_audit_status": "PENDING",
        "plan_sha256": expected,
        "index_sha256": digest(index_path),
        "scientific_code_sha256": hashes,
        "code_sha256": hashes,
        "configurations": configurations,
        "primary_method": primary,
        "representation_choice": {
            "method": representation_method,
            "candidate_id": representation_ids[representation_method],
            "config": selected_representation,
        },
        "representation_selection": representation_selection,
        "factorial_selection": factorial_selection,
        "onset_matched_instantaneous_method": next(
            method
            for method, config in factorials.items()
            if config == {**configurations["primary_onset"], "target": "instantaneous"}
        ),
    }
    manifest["rows"] = (
        manifest["representation_rows"] + manifest["factorial_rows"] + manifest["diagnostic_rows"]
    )
    manifest.update(
        status="complete", finished_at=datetime.now(UTC).isoformat(), selection=selection
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
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    result = run(args.index, args.output, plan_path=args.plan, plan_sha256=args.plan_sha256)
    print(json.dumps(result["selection"]["primary_method"]))
