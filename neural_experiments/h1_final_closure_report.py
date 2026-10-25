"""Create-new, aggregate-only final H1 evidence export from authenticated receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_decoder.artifacts import sha256, write_json
from neural_experiments.audit_h1_final_closure import validate_audit_receipt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "A": "PyTorch window f32",
    "B": "NumPy incremental f32",
    "C": "ONNX original f32",
    "D": "ONNX rewritten f32",
    "E": "ONNX rewritten int8",
}


def read(path):
    return json.loads(Path(path).read_text())


def runtime_evidence(run, audit):
    receipt = read(audit)
    manifest = read(Path(run) / "manifest.json")
    if (
        receipt.get("kind") != "h1_final_closure_runtime_audit"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "passed"
        or manifest.get("status") != "complete"
        or receipt["run"] != str(Path(run).resolve())
        or receipt["manifest_sha256"] != sha256(Path(run) / "manifest.json")
        or receipt["summary_sha256"] != sha256(Path(run) / "summary.json")
    ):
        raise ValueError("runtime audit does not bind a completed run")
    expected_sources = {
        "neural_experiments/audit_h1_final_closure_runtime.py",
        "neural_experiments/audit_runtime.py",
    }
    if set(receipt["auditor_source"]) != expected_sources:
        raise ValueError("runtime audit source inventory changed")
    if sha256(manifest["quality_audit"]) != manifest["quality_audit_sha256"]:
        raise ValueError("runtime quality receipt bytes changed")
    for name, digest in receipt["auditor_source"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("runtime auditor source changed")
    summary = read(Path(run) / "summary.json")
    if receipt["quality_audit_sha256"] != manifest["quality_audit_sha256"]:
        raise ValueError("runtime quality receipt differs")
    return manifest, summary, receipt


def _export_report(
    quality, quality_audit, runtime, runtime_audit, soak, soak_audit, lifecycle, output
):
    output = Path(output)
    qa = validate_audit_receipt(quality, quality_audit, ROOT)
    qm = read(Path(quality) / "manifest.json")
    q = read(Path(quality) / "summary.json")
    rm, r, ra = runtime_evidence(runtime, runtime_audit)
    sm, s, sa = runtime_evidence(soak, soak_audit)
    from neural_experiments.h1_final_closure_recovery import validate_lifecycle_receipt

    recovery = validate_lifecycle_receipt(lifecycle, quality, quality_audit, ROOT)
    if "E" in sm["eligible_arms"] and sm["lifecycle"]["sha256"] != sha256(lifecycle):
        raise ValueError("soak lifecycle receipt differs")
    if (
        rm["quality_audit_sha256"] != sha256(quality_audit)
        or sm["quality_audit_sha256"] != sha256(quality_audit)
        or recovery["quality_audit_sha256"] != sha256(quality_audit)
    ):
        raise ValueError("evidence combines different quality runs")
    public_quality = {
        key: q[key]
        for key in (
            "gates",
            "seed_day_rows",
            "mean_delta_r2",
            "worst_seed_day_delta",
            "timing_seed",
            "exposure",
        )
    }
    public_quality["models"] = []
    for model in q["models"]:
        public_quality["models"].append(
            {
                "seed": model["seed"],
                "model": model["model"],
                "fingerprint": model["fingerprint"],
                "parameter_count": model["parameter_count"],
                "export_wall_seconds": model["export_wall_seconds"],
                "reference_scores": model["reference"]["scores"],
                "arms": {
                    arm: {
                        key: value
                        for key, value in row.items()
                        if key not in {"predictions", "trace_checks", "traceback"}
                    }
                    for arm, row in model["arms"].items()
                },
            }
        )
    write_json(output / "quality.json", public_quality)
    write_json(output / "runtime.json", r)
    write_json(output / "soak.json", s)
    write_json(
        output / "lifecycle.json",
        {
            key: recovery[key]
            for key in ("status", "lifecycle", "overhead", "faults", "qualified_arms")
        },
    )
    bindings = {
        "quality": {
            "run": str(Path(quality).resolve()),
            "manifest_sha256": sha256(Path(quality) / "manifest.json"),
            "audit_sha256": sha256(quality_audit),
            "source": qm["source"],
        },
        "runtime": {
            "run": str(Path(runtime).resolve()),
            "manifest_sha256": sha256(Path(runtime) / "manifest.json"),
            "audit_sha256": sha256(runtime_audit),
            "source": rm["source"],
        },
        "soak": {
            "run": str(Path(soak).resolve()),
            "manifest_sha256": sha256(Path(soak) / "manifest.json"),
            "audit_sha256": sha256(soak_audit),
            "source": sm["source"],
        },
        "lifecycle_sha256": sha256(lifecycle),
        "publication": "aggregate statistics and identity bindings only",
    }
    write_json(output / "bindings.json", bindings)
    for name, receipt in (("quality-audit", qa), ("runtime-audit", ra), ("soak-audit", sa)):
        write_json(output / f"{name}.json", receipt)
    rows = [
        "# Final H1 deployment measurements",
        "",
        "All timing uses predetermined final seed 0. Quality covers all three seeds and seven "
        "previously assessed days from one participant. This is offline counts-ready replay, "
        "not fresh generalization or closed-loop control.",
        "",
        "| Arm | Seed-0 R² | 3-seed mean R² | Worst day ΔR² | Graph bytes | State bytes | "
        "Fast p99 ms, range | Paced service p99 ms, range | Delivery p99 ms, range | "
        "Paced misses / bins | Soak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def number(value, signed=False):
        if value is None or not np.isfinite(value):
            return "not measured"
        return f"{value:+.6f}" if signed else f"{value:.6f}"

    def span(values):
        return "not measured" if not values else f"{min(values):.6f}–{max(values):.6f}"

    for arm in "ABCDE":
        models = [m["arms"][arm] for m in q["models"]]
        seed0 = next(m for m in q["models"] if m["seed"] == 0)["arms"][arm]
        fast = [x for x in r["runs"] if x["arm"] == arm and x["mode"] == "fast"]
        paced = [x for x in r["runs"] if x["arm"] == arm and x["mode"] == "paced"]
        soaks = [x for x in s["runs"] if x["arm"] == arm]
        score = seed0.get("scores", {}).get("r2_mean")
        scores = [m["scores"]["r2_mean"] for m in models if "scores" in m]
        deltas = [
            v["r2_variance_weighted"] for m in models for v in m.get("delta_per_day", {}).values()
        ]
        graph = seed0.get("graph")
        graph_bytes = (
            graph.get("bytes", graph.get("onnx_bytes", "see quality.json"))
            if graph
            else ("NPZ" if arm in "AB" else "not measured")
        )
        count = sum(x["bins"] for x in paced)
        misses = sum(x["missed_deadlines"] for x in paced)
        soak_status = (
            (
                f"{sum(x['missed_deadlines'] for x in soaks)} misses / "
                f"{sum(x['bins'] for x in soaks)} bins"
            )
            if soaks
            else "not included"
        )
        rows.append(
            f"| {arm}: {LABELS[arm]} | {number(score)} | "
            f"{number(float(np.mean(scores)) if len(scores) == 3 else None)} | "
            f"{number(min(deltas) if deltas else None, signed=True)} | {graph_bytes} | "
            f"{seed0.get('numeric_state_bytes', 'not measured')} | "
            f"{span([x['step_service']['p99_ms'] for x in fast])} | "
            f"{span([x['step_service']['p99_ms'] for x in paced])} | "
            f"{span([x['release_to_completion']['p99_ms'] for x in paced])} | "
            f"{f'{misses} / {count}' if count else 'not measured'} | {soak_status} |"
        )
    rows += [
        "",
        "## Every runtime repeat",
        "",
        "| Mode | Repeat | Position | Arm | Service p99 ms | Queue p99 ms | "
        "Delivery p99 ms | Misses | Bins |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for x in [*r["runs"], *s["runs"]]:
        rows.append(
            f"| {x['mode']} | {x['repeat']} | {x['position']} | {x['arm']} | "
            f"{x['step_service']['p99_ms']:.6f} | {x['queue_delay']['p99_ms']:.6f} | "
            f"{x['release_to_completion']['p99_ms']:.6f} | {x['missed_deadlines']} | "
            f"{x['bins']} |"
        )
    rows += [
        "",
        "All per-output/day scores, constructor/reset/resource accounting and "
        "audit bindings accompany this table. Graph size is distinct from process RSS. "
        "Repeated recordings add no biological evidence. The finite deadline counts are "
        "observations, not hard real-time guarantees.",
        "",
    ]
    (output / "tables.md").write_text("\n".join(rows))
    return bindings


def export_report(*args, **kwargs):
    output = Path(kwargs["output"] if "output" in kwargs else args[-1])
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "kind": "h1_final_public_report"}
    write_json(output / "start.json", state)
    try:
        result = _export_report(*args, **kwargs)
        state["status"] = "complete"
        return result
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["artifacts"] = {
            path.name: sha256(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(output / "manifest.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "quality",
        "quality-audit",
        "runtime",
        "runtime-audit",
        "soak",
        "soak-audit",
        "lifecycle",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    export_report(**vars(args))
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
