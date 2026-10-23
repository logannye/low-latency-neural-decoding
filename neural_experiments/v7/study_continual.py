"""Frozen V7B exposed selection and one-shot fresh-session assessment."""

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
from neural_experiments.v7 import continual, continual_metrics, data, fresh_data, model

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = "4f88a6cf98dd7147691f4eaf64143784736863e1cfa4fd5fbd7b3fadd1eaea23"
SOURCES = tuple(
    f"neural_experiments/{version}/{name}.py"
    for version, names in (
        (
            "v7",
            ("continual", "continual_metrics", "study_continual", "model", "data", "fresh_data"),
        ),
        ("v6", ("data",)),
        ("v5", ("data",)),
        ("v4", ("model", "data")),
        ("v3", ("study", "model", "data")),
        ("v2", ("data",)),
    )
    for name in names
)


def run(phase, index_path, output, *, selection_path=None, selection_sha=None):
    plan_path = Path(__file__).with_name("continual-plan.json")
    if phase not in ("source", "assessment") or digest(plan_path) != PLAN_SHA:
        raise ValueError("declared phase and exact continual plan required")
    plan = json.loads(plan_path.read_text())
    if phase == "source":
        if digest(index_path) != plan["development_index_sha256"]:
            raise ValueError("exposed union changed")
        index = data.load_index(index_path, expected_role="development", verify_hashes=True)
        choices, selection = None, None
    else:
        index = fresh_data.load_index(index_path, expected_role="fresh", verify_hashes=True)
        if not selection_path or not selection_sha or digest(selection_path) != selection_sha:
            raise ValueError("published source selection digest required before assessment")
        selection = json.loads(Path(selection_path).read_text())
        if selection["plan_sha256"] != PLAN_SHA or selection["source_audit_status"] != "PASS":
            raise ValueError("audited source selection from this plan required")
        choices = {r["participant"]: r["threshold"] for r in selection["selection"]["choices"]}
        current = {p: digest(ROOT / p) for p in SOURCES}
        if current != selection["scientific_code_sha256"]:
            raise ValueError("scientific execution source changed after source selection")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "traces"):
        (output / name).mkdir()
    hashes = {p: digest(ROOT / p) for p in SOURCES}
    for path in SOURCES:
        destination = output / "source_snapshot" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    manifest = {
        "schema_version": 1,
        "phase": phase,
        "study": "V7B",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "plan": {"path": relative(plan_path), "sha256": PLAN_SHA},
        "index": {"path": relative(index_path), "sha256": digest(index_path)},
        "code_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "threadpool_limit": 1,
        },
        "source_selection": {"path": relative(selection_path), "sha256": selection_sha}
        if selection
        else None,
        "rows": [],
        "events": [],
    }
    write_json(output / "manifest.json", manifest)
    with threadpool_limits(limits=1):
        for item in index["sessions"]:
            case = (
                data.load_case(index_path, item, labels=True)
                if phase == "source"
                else _fresh_case(index_path, item)
            )
            thresholds = (
                continual_metrics.THRESHOLDS
                if phase == "source"
                else (choices[item["participant_id"]],)
            )
            for threshold in thresholds:

                def save(kind, method, ordinal, payload):
                    key = f"{item['session_id']}-{threshold:g}-{method}-{ordinal}"
                    if kind == "model":
                        path = output / "models" / f"{key}.npz"
                        payload["decoder"].save(path)
                        restored = model.Decoder.load(path)
                        for field in ("weights", "mean", "scale"):
                            np.testing.assert_array_equal(
                                getattr(restored, field), getattr(payload["decoder"], field)
                            )
                        if restored.config != payload["decoder"].config:
                            raise ValueError("export config changed")
                        return {
                            "decoder": restored,
                            "record": {"model_path": relative(path), "model_sha256": digest(path)},
                        }
                    if kind != "trace":
                        raise ValueError("unknown artifact kind")
                    path = output / "traces" / f"{key}.npz"
                    np.savez_compressed(path, **payload)
                    return {"trace_path": relative(path), "trace_sha256": digest(path)}

                rows, events = continual.evaluate_session(case, threshold, save_artifact=save)
                manifest["rows"].extend({**r, "threshold": threshold} for r in rows)
                manifest["events"].extend({**e, "threshold": threshold} for e in events)
            write_json(output / "manifest.json", manifest)
            print(json.dumps({"phase": phase, "session": item["session_id"]}), flush=True)
    if phase == "source":
        manifest["selection"] = continual_metrics.select_source(manifest["rows"])
        choices = {r["participant"]: r["threshold"] for r in manifest["selection"]["choices"]}
        selected = [r for r in manifest["rows"] if r["threshold"] == choices[r["participant"]]]
        manifest["summary"] = continual_metrics.summarize(selected)
    else:
        manifest["summary"] = continual_metrics.summarize(manifest["rows"])
    if any(digest(ROOT / p) != expected for p, expected in hashes.items()):
        raise ValueError("scientific code changed during run")
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    write_json(output / "manifest.json", manifest)
    return manifest


def _fresh_case(index_path, item):
    root = Path(index_path).resolve().parent
    return {
        "item": item,
        "features": fresh_data.load_features(root / item["feature_path"]),
        "segments": [
            (int(s["start"]), int(s["end"])) for s in item["segments"] if s["end"] > s["start"]
        ],
        **fresh_data.load_targets(root / item["target_path"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("source", "assessment"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    args = parser.parse_args()
    result = run(
        args.phase,
        args.index,
        args.output,
        selection_path=args.selection,
        selection_sha=args.selection_sha256,
    )
    print(json.dumps(result.get("selection", result["summary"]), indent=2))
