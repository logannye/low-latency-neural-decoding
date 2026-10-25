"""Independent final-runtime audit of identities, clocks, resets and measured predictions.

No runtime producer is imported. Arithmetic references use independent whole-file
convolutions, and int8 predictions are replayed directly through ORT with explicit
history feeds. Auditing runs only after the sole measurement process has stopped.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from neural_experiments.audit_runtime import (
    compare_summary,
    digest,
    independently_summarize,
    read_json,
    reference_trace,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]
STAMP_NAMES = ("release_ns", "start_ns", "prediction_ready_ns", "complete_ns", "deadline_ns")


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def contained(base, relative):
    result = (Path(base) / relative).resolve()
    if not result.is_relative_to(Path(base).resolve()):
        raise ValueError("evidence path escapes its bound directory")
    return result


def verify_committed_source(run, manifest, config, root):
    source = manifest["source"]
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", source["head"]], cwd=root, text=True
    ).splitlines()
    patterns = [
        *config["runtime"]["source_dependencies"],
        "neural_experiments/audit_runtime.py",
        "tests/neural_decoder/test_h1_final_closure*.py",
    ]
    required = {name for name in tracked if any(fnmatch.fnmatchcase(name, p) for p in patterns)}
    if set(source["files"]) != required or not required:
        raise ValueError("committed runtime source dependency inventory changed")
    for name, checksum in source["files"].items():
        verify(contained(run / "source", name), checksum)
        committed = subprocess.check_output(["git", "show", f"{source['head']}:{name}"], cwd=root)
        if hashlib.sha256(committed).hexdigest() != checksum:
            raise ValueError("runtime source snapshot is not its claimed committed revision")
    if (
        manifest["harness_sha256"]
        != source["files"]["neural_experiments/h1_final_closure_runtime.py"]
    ):
        raise ValueError("runtime harness identity differs from retained source")


def verify_attempt_times(start, final):
    if (
        start["status"] != "running"
        or datetime.fromisoformat(final["finished_utc"])
        < datetime.fromisoformat(start["started_utc"])
        or final["started_utc"] != start["started_utc"]
    ):
        raise ValueError("attempt start/final status or chronology changed")
    for key, value in start.items():
        if key not in {"status", "statuses"} and final.get(key) != value:
            raise ValueError(f"attempt start/final identity changed: {key}")


def independent_int8_trace(graph, counts):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(
        str(graph), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_modelmeta().custom_metadata_map.get("precision") != "dynamic_int8":
        raise ValueError("int8 audit received a different precision")
    state = [np.zeros(shape, dtype=np.float32) for shape in ((2, 176), (4, 32), (8, 32))]
    predictions = np.empty((len(counts), 7), dtype=np.float32)
    for i, row in enumerate(counts):
        result = session.run(
            ["prediction", "next_history0", "next_history1", "next_history2"],
            {
                "counts": row.astype(np.float32),
                "history0": state[0],
                "history1": state[1],
                "history2": state[2],
            },
        )
        predictions[i], state = result[0], result[1:]
    return predictions


def expected_segments(files, mode, bins):
    """Independently enumerate file boundaries without importing producer scheduling."""
    result, start, cycle, index = [], 0, 0, 0
    while start < bins:
        length = min(files[index]["n_bins"], bins - start)
        result.append(
            {
                "global_id": start,
                "cycle": cycle,
                "file_index": index,
                "file_id": files[index]["file_id"],
                "bins": length,
                "partial": length != files[index]["n_bins"],
            }
        )
        start += length
        index += 1
        if index == len(files):
            index, cycle = 0, cycle + 1
        if start < bins and mode == "paced":
            raise ValueError("paced workload cannot stitch files")
        if start < bins and mode == "fast" and cycle:
            raise ValueError("fast workload cannot repeat files")
    return result


def audit_records(
    records,
    summary,
    resets,
    files,
    timestamps,
    references,
    *,
    mode,
    arm,
    period_ns,
    expected_bins,
    rtol=3e-5,
    atol=1e-7,
):
    """Validate raw logs against independently reconstructed boundaries and predictions."""
    if len(records) != expected_bins or not np.array_equal(
        records["global_id"], np.arange(expected_bins)
    ):
        raise ValueError("missing, duplicate or out-of-order global bin")
    for name in (*STAMP_NAMES, "global_id", "cycle", "file_index", "sequence_id"):
        if records.dtype[name].kind != "i":
            raise ValueError("raw timestamps and identities must be signed integers")
    if (
        records["prediction"].shape != (expected_bins, 7)
        or not np.isfinite(records["prediction"]).all()
    ):
        raise ValueError("invalid measured prediction payload")
    stamps = np.column_stack([records[name] for name in STAMP_NAMES])
    metrics = independently_summarize(stamps, period_ns)
    if not np.array_equal(stamps[:, 2], stamps[:, 3]):
        raise ValueError("frozen inference ready/complete event definition changed")
    if mode == "fast":
        if not np.array_equal(stamps[:, 0], stamps[:, 1]):
            raise ValueError("fast release must equal processing start")
    elif not np.array_equal(
        stamps[:, 0], summary["replay"]["release_origin_ns"] + np.arange(expected_bins) * period_ns
    ):
        raise ValueError("global release origin changed, including a file boundary")
    segments = expected_segments(files, mode, expected_bins)
    if len(resets) != len(segments):
        raise ValueError("wrong number of file resets")
    if summary.get("file_reset_ns", [row["complete_ns"] - row["start_ns"] for row in resets]) != [
        row["complete_ns"] - row["start_ns"] for row in resets
    ]:
        raise ValueError("reset duration summary differs from reset timestamps")
    if summary["warmup_bins"] != 500 or summary["warmup_complete_ns"] > resets[0]["start_ns"]:
        raise ValueError("warmup is missing or included in measurement")
    if summary["replay"]["release_origin_ns"] > resets[0]["start_ns"]:
        raise ValueError("release origin must precede the first measured reset")
    maximum_error = 0.0
    for expected, reset in zip(segments, resets, strict=True):
        if any(reset.get(key) != value for key, value in expected.items()):
            raise ValueError("file reset schedule or final partial-file identity changed")
        start, length, index = expected["global_id"], expected["bins"], expected["file_index"]
        block = records[start : start + length]
        if (
            reset["complete_ns"] < reset["start_ns"]
            or reset["complete_ns"] > block["start_ns"][0]
            or (start and reset["start_ns"] < records["complete_ns"][start - 1])
        ):
            raise ValueError("file reset overlaps measured processing")
        if (
            not np.all(block["file_index"] == index)
            or not np.all(block["cycle"] == expected["cycle"])
            or not np.array_equal(block["sequence_id"], np.arange(length))
        ):
            raise ValueError("file sequence or composite cycle/file identity changed")
        if not np.array_equal(block["source_timestamp"], timestamps[index][:length]):
            raise ValueError("source-bin timestamps changed")
        expected_prediction = references[index][:length]
        # E consistency is int8-to-int8, independent of its approximation-quality budget.
        np.testing.assert_allclose(
            block["prediction"],
            expected_prediction,
            rtol=0 if arm == "E" else rtol,
            atol=0 if arm == "E" else atol,
        )
        maximum_error = max(
            maximum_error, float(np.max(np.abs(block["prediction"] - expected_prediction)))
        )
    compare_summary(metrics, summary)
    if summary["deadline_miss_rate"] != metrics["missed_deadlines"] / expected_bins:
        raise ValueError("deadline miss rate differs from integer stamps")
    if summary["service_goal_passed"] != (metrics["step_service"]["p99_ms"] < 2):
        raise ValueError("service goal decision differs from measured p99")
    logger = summary["logger"]
    if (
        logger["capacity_records"] != 4096
        or logger["high_water_records"] != min(4096, expected_bins)
        or logger["written_records"] != expected_bins
        or logger["allocated_records"] != expected_bins
        or logger["dropped_records"] != 0
        or logger["flushes"] != (expected_bins + 4095) // 4096
    ):
        raise ValueError("bounded logger accounting changed")
    return metrics, maximum_error


def audit(run, *, root=ROOT):
    from neural_experiments.audit_h1_final_closure import validate_audit_receipt

    run = Path(run).resolve()
    manifest = read_json(run / "manifest.json")
    if (
        manifest["status"] != "complete"
        or manifest["statuses"]["implementation_execution"] != "complete"
        or manifest["statuses"]["artifact_completion"] != "complete"
    ):
        raise ValueError("failed or incomplete execution cannot pass runtime audit")
    verify(run / "summary.json", manifest["summary_sha256"])
    verify(run / "config.json", manifest["config_sha256"])
    config = read_json(run / "config.json")
    if config["quality"] != {
        "parity_rtol": 3e-5,
        "parity_atol": 1e-7,
        "score_atol": 1e-5,
        "max_int8_r2_loss": 0.005,
    }:
        raise ValueError("fixed float32/approximation tolerances changed")
    required_schedule = {
        "timing_seed": 0,
        "period_ns": 20_000_000,
        "warmup_bins": 500,
        "paced_bins": 3001,
        "soak_bins": 90000,
        "fast_bins": 112488,
        "logger_buffer_records": 4096,
        "rss_sample_interval_ns": 1_000_000_000,
    }
    if any(config["runtime"].get(key) != value for key, value in required_schedule.items()):
        raise ValueError("independently fixed timing workload or schedule changed")
    verify_committed_source(run, manifest, config, root)
    verify_attempt_times(read_json(run / "start.json"), manifest)
    if (
        len(manifest["command"]) < 2
        or manifest["command"][1] != manifest["mode"]
        or Path(manifest["command"][0]).name != "h1_final_closure_runtime.py"
    ):
        raise ValueError("campaign command does not identify its measurement mode")
    quality_run, quality_audit = Path(manifest["quality_run"]), Path(manifest["quality_audit"])
    verify(quality_audit, manifest["quality_audit_sha256"])
    verify(quality_run / "manifest.json", manifest["quality_manifest_sha256"])
    verify(quality_run / "summary.json", manifest["quality_summary_sha256"])
    receipt = validate_audit_receipt(quality_run, quality_audit, root)
    if (
        manifest["statuses"]["float32_equivalence"] != receipt["gates"]["float32_equivalence"]
        or manifest["statuses"]["compression_quality"] != receipt["gates"]["compression_quality"]
    ):
        raise ValueError("runtime inherited quality status changed")
    if (
        receipt["qualified_arms"] != manifest["qualified_arms"]
        or receipt["config_sha256"] != manifest["config_sha256"]
    ):
        raise ValueError("quality gate or configuration binding changed")
    eligible = [arm for arm in "ABCDE" if receipt["qualified_arms"][arm]]
    if manifest["mode"] == "soak":
        eligible = ["B"] if "B" in eligible else []
        if receipt["qualified_arms"]["E"]:
            from neural_experiments.h1_final_closure_recovery import validate_lifecycle_receipt

            verify(manifest["lifecycle"]["path"], manifest["lifecycle"]["sha256"])
            lifecycle = validate_lifecycle_receipt(
                manifest["lifecycle"]["path"], quality_run, quality_audit, root
            )
            if lifecycle["qualified_arms"]["E"]:
                eligible.append("E")
        orders = [eligible]
        expected_runs = [("soak", 0, position, arm) for position, arm in enumerate(eligible)]
    else:
        orders = [eligible[offset:] + eligible[:offset] for offset in range(len(eligible))]
        expected_runs = [
            (mode, repeat, position, arm)
            for mode in ("fast", "paced")
            for repeat, order in enumerate(orders)
            for position, arm in enumerate(order)
        ]
    if not eligible or manifest["orders"] != orders or manifest["eligible_arms"] != eligible:
        raise ValueError("fallback order or lifecycle inclusion changed")
    workload = config["runtime"]["workload_index"]
    if manifest["workload_index"] != workload:
        raise ValueError("runtime input binding changed")
    # The authenticated pre-existing development cache may be a symlink outside
    # this checkout. Only this exact frozen index path/hash is authorized here.
    index_path = root / workload["path"]
    verify(index_path, workload["sha256"])
    index = read_json(index_path)
    files = sorted(index["files"], key=lambda row: (int(row["day"][1:]), row["file_id"]))
    if (
        index["role"] != "development"
        or len(files) != 11
        or [row["file_id"] for row in files] != config["runtime"]["fast_file_ids"]
        or any(row["provider_split"] != "held-in-calib" for row in files)
        or {row["day"] for row in files} != {f"S{i}" for i in range(5)}
    ):
        raise ValueError("not the fixed eleven-file development workload")
    model_record = next(row for row in config["models"] if row["seed"] == 0)
    model_path = contained(root, model_record["path"])
    verify(model_path, model_record["sha256"])
    if manifest["model"]["sha256"] != model_record["sha256"]:
        raise ValueError("timing model is not the predetermined final seed0")
    quality_model = next(
        row for row in read_json(quality_run / "summary.json")["models"] if row["seed"] == 0
    )
    with np.load(model_path, allow_pickle=False) as bundle:
        weights = {name: bundle[name] for name in bundle.files}
    references, int8_references, timestamps = [], [], []
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for row in files:
            feature_path = contained(index_path.parent, row["features"])
            verify(feature_path, row["features_sha256"])
            # Authenticate scoring payload too: the index binds the entire prior input inventory.
            verify(contained(index_path.parent, row["targets"]), row["targets_sha256"])
            with np.load(feature_path, allow_pickle=False) as bundle:
                if (
                    set(bundle.files) != {"counts", "timestamps", "sequence_id", "channel_ids"}
                    or not np.array_equal(bundle["sequence_id"], np.arange(row["n_bins"]))
                    or not np.array_equal(bundle["channel_ids"], np.arange(176))
                ):
                    raise ValueError("source stream identity changed")
                counts = bundle["counts"]
                timestamps.append(bundle["timestamps"])
                references.append(reference_trace(counts, weights))
                if "E" in eligible:
                    graph = quality_model["arms"]["E"]["graph"]
                    graph_path = contained(quality_run, graph["path"])
                    verify(graph_path, graph["sha256"])
                    int8_references.append(independent_int8_trace(graph_path, counts))
    summary = read_json(run / "summary.json")
    if len(manifest["blocks"]) != len(expected_runs) or len(summary["runs"]) != len(expected_runs):
        raise ValueError("missing, additional or repeated timing block")
    reports, max_errors = [], {}
    for entry, row, expected in zip(
        manifest["blocks"], summary["runs"], expected_runs, strict=True
    ):
        mode, repeat, position, arm = expected
        expected_folder = f"{mode}-{repeat}-{position}-{arm}"
        if (
            entry["folder"] != expected_folder
            or entry["returncode"] != 0
            or (row["mode"], row["repeat"], row["position"], row["arm"]) != expected
        ):
            raise ValueError("counterbalanced run order or execution status changed")
        folder = run / expected_folder
        for name in ("start", "request", "status", "summary"):
            verify(folder / f"{name}.json", entry[f"{name}_sha256"])
        status, request = read_json(folder / "status.json"), read_json(folder / "request.json")
        if status["status"] != "complete" or status["summary_sha256"] != entry["summary_sha256"]:
            raise ValueError("block terminal execution did not complete")
        verify_attempt_times(read_json(folder / "start.json"), status)
        for key, expected_path in {
            "root": root.resolve(),
            "campaign": run,
            "output": folder,
            "quality_run": quality_run.resolve(),
        }.items():
            if request[key] != str(expected_path):
                raise ValueError(f"worker {key} points outside the bound campaign")
        expected_command = [
            str(root / "neural_experiments/h1_final_closure_runtime.py"),
            "_worker",
            "--request",
            str(folder / "request.json"),
        ]
        if entry["command"][1:] != expected_command:
            raise ValueError("worker command differs from its bound request")
        if read_json(folder / "summary.json") != row:
            raise ValueError("aggregate and per-block summaries disagree")
        graph = quality_model["arms"][arm]["graph"]
        if (
            request["source"] != manifest["source"]
            or request["model"]["sha256"] != model_record["sha256"]
            or request["graph"] != graph
            or request["arm"] != arm
            or row["model_sha256"] != model_record["sha256"]
            or row["graph_sha256"] != (None if graph is None else graph["sha256"])
        ):
            raise ValueError("block model/graph/source identity changed")
        if status["request_sha256"] != entry["request_sha256"]:
            raise ValueError("worker status belongs to another request")
        if (request["mode"], request["repeat"], request["position"], request["arm"]) != expected:
            raise ValueError("worker request changed its scheduled position")
        if row["source_index_sha256"] != workload["sha256"]:
            raise ValueError("worker source index binding changed")
        if graph is not None:
            verify(contained(quality_run, graph["path"]), graph["sha256"])
        if set(row["raw_sha256"]) != {"records.npy", "resets.jsonl", "rss.jsonl"}:
            raise ValueError("raw evidence hash inventory changed")
        for filename, checksum in row["raw_sha256"].items():
            verify(contained(folder, filename), checksum)
        records = np.load(folder / "records.npy", mmap_mode="r", allow_pickle=False)
        bins = config["runtime"][
            {"fast": "fast_bins", "paced": "paced_bins", "soak": "soak_bins"}[mode]
        ]
        metrics, error = audit_records(
            records,
            row,
            read_lines(folder / "resets.jsonl"),
            files,
            timestamps,
            int8_references if arm == "E" else references,
            mode=mode,
            arm=arm,
            period_ns=config["runtime"]["period_ns"],
            expected_bins=bins,
            rtol=config["quality"]["parity_rtol"],
            atol=config["quality"]["parity_atol"],
        )
        rss = read_lines(folder / "rss.jsonl")
        if (
            not rss
            or max(sample["rss_bytes"] for sample in rss) != row["replay"]["sampled_max_rss_bytes"]
            or any(sample["rss_bytes"] <= 0 for sample in rss)
            or any(
                right["clock_ns"] - left["clock_ns"] < 1_000_000_000
                for left, right in zip(rss, rss[1:])
            )
        ):
            raise ValueError("RSS sampling/accounting changed")
        for sample in rss:
            index = sample["global_id"]
            if (
                not 0 <= index < bins
                or sample["clock_ns"] < records["complete_ns"][index]
                or (index + 1 < bins and sample["clock_ns"] > records["start_ns"][index + 1])
            ):
                raise ValueError("RSS observation lies outside its recorded processing boundary")
        max_errors[arm] = max(max_errors.get(arm, 0), error)
        reports.append(
            {
                "mode": mode,
                "repeat": repeat,
                "position": position,
                "arm": arm,
                "metrics": metrics,
                "measured_prediction_parity": "passed",
            }
        )
    goal = "passed" if all(row["service_goal_passed"] for row in summary["runs"]) else "failed"
    if manifest["statuses"]["runtime_goals"] != goal:
        raise ValueError("campaign runtime goal status differs from audited blocks")
    return {
        "schema_version": 1,
        "kind": "h1_final_closure_runtime_audit",
        "status": "passed",
        "audited_utc": datetime.now(UTC).isoformat(),
        "run": str(run),
        "manifest_sha256": digest(run / "manifest.json"),
        "summary_sha256": digest(run / "summary.json"),
        "auditor_source": {
            str(Path(__file__).relative_to(root)): digest(__file__),
            "neural_experiments/audit_runtime.py": digest(
                root / "neural_experiments/audit_runtime.py"
            ),
        },
        "quality_audit_sha256": digest(quality_audit),
        "runs": reports,
        "max_prediction_absolute_difference": max_errors,
        "scope": "identities, admission, raw clocks, order, resets, every measured prediction",
        "limitations": (
            "constructor/CPU/RSS are retained observations, not reconstructible from bin timestamps"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(UTC).isoformat()
    with args.output.open("x") as handle:
        handle.write(
            json.dumps(
                {
                    "status": "running",
                    "started_utc": started,
                    "run": str(args.run.resolve()),
                    "auditor_sha256": digest(__file__),
                }
            )
            + "\n"
        )
    try:
        result = audit(args.run)
        result["started_utc"] = started
    except BaseException as error:
        result = {
            "status": "failed",
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "run": str(args.run.resolve()),
            "auditor_sha256": digest(__file__),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "runs": len(result["runs"])}), flush=True)


if __name__ == "__main__":
    main()
