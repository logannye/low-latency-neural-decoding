"""Final-weight, quality-gated counts-ready timing with bounded durable logging.

The parent launches one fresh measurement subprocess at a time. No scoring or
prediction audit runs inside a timed block. Raw traces stay in ignored results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from neural_decoder.artifacts import environment, sha256, write_json
from neural_decoder.data import load_feature_file
from neural_decoder.optimized import ONNXTCNStream, model_fingerprint
from neural_decoder.runtime import timing_summary
from neural_decoder.temporal import TCNStream, load_tcn
from neural_decoder.train import checked_development_index

ROOT = Path(__file__).resolve().parents[1]
STAMP_FIELDS = ("release_ns", "start_ns", "prediction_ready_ns", "complete_ns", "deadline_ns")
RECORD_DTYPE = np.dtype(
    [
        ("global_id", "<i8"),
        ("cycle", "<i4"),
        ("file_index", "<i4"),
        ("sequence_id", "<i8"),
        ("source_timestamp", "<f8"),
    ]
    + [(name, "<i8") for name in STAMP_FIELDS]
    + [("prediction", "<f4", (7,))]
)


def read_json(path):
    return json.loads(Path(path).read_text())


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"changed evidence: {path}")


def cyclic_orders(eligible):
    """Fixed fallback: alphabetical eligible paths, every cyclic rotation."""
    arms = sorted(eligible)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set("ABCDE"):
        raise ValueError("expected distinct eligible A-E arms")
    return [arms[i:] + arms[:i] for i in range(len(arms))]


def validate_runtime_config(config):
    runtime = config["runtime"]
    fixed = {
        "timing_seed": 0,
        "warmup_bins": 500,
        "period_ns": 20_000_000,
        "paced_bins": 3001,
        "soak_bins": 90_000,
        "threads": 1,
        "logger_buffer_records": 4096,
        "rss_sample_interval_ns": 1_000_000_000,
        "eligible_order": "ABCDE",
    }
    if any(runtime.get(key) != value for key, value in fixed.items()):
        raise ValueError("prespecified runtime schedule or resource limit changed")
    if runtime["orders"] != cyclic_orders("ABCDE") or len(runtime["fast_file_ids"]) != 11:
        raise ValueError("runtime orders or eleven-file workload changed")
    if runtime["workload_index"] != config["inputs"]["development_index"]:
        raise ValueError("runtime index differs from authenticated development index")
    return runtime


def source_snapshot(root, destination, config):
    """Snapshot the declared committed dependency inventory, including helper imports."""
    paths = set()
    for pattern in config["runtime"]["source_dependencies"]:
        matches = list(root.glob(pattern))
        if not matches:
            raise ValueError(f"missing declared source dependency: {pattern}")
        paths.update(path for path in matches if path.is_file())
    paths.update((root / "neural_experiments/audit_runtime.py",))
    paths.update(root.glob("tests/neural_decoder/test_h1_final_closure*.py"))
    relative = sorted(str(path.relative_to(root)) for path in paths)
    tracked = set(
        subprocess.check_output(
            ["git", "ls-files", "--", *relative], cwd=root, text=True
        ).splitlines()
    )
    if set(relative) != tracked:
        raise ValueError("commit every runtime dependency before measurement")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative], cwd=root, text=True
    )
    if dirty.strip():
        raise ValueError("runtime dependencies must be committed and clean")
    hashes = {}
    for name in relative:
        target = destination / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
        hashes[name] = sha256(target)
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files": hashes,
    }


def verify_sources(root, source):
    for name, expected in source["files"].items():
        verify(root / name, expected)


def admit_quality(config_path, quality_run, quality_audit, root=ROOT):
    """A receipt is authority only after its evidence and audit-source bindings pass."""
    from neural_experiments.audit_h1_final_closure import validate_audit_receipt

    config = read_json(config_path)
    validate_runtime_config(config)
    receipt = validate_audit_receipt(quality_run, quality_audit, root)
    if receipt["config_sha256"] != sha256(config_path):
        raise ValueError("quality audit belongs to a different configuration")
    qualified = receipt["qualified_arms"]
    if set(qualified) != set("ABCDE") or any(type(x) is not bool for x in qualified.values()):
        raise ValueError("invalid quality eligibility mapping")
    summary = read_json(Path(quality_run) / "summary.json")
    models = [row for row in summary["models"] if row["seed"] == 0]
    if len(models) != 1:
        raise ValueError("missing predetermined seed0")
    model = models[0]
    expected = next(row for row in config["models"] if row["seed"] == 0)
    if model["model"]["sha256"] != expected["sha256"]:
        raise ValueError("quality model differs from predetermined seed0")
    verify(root / expected["path"], expected["sha256"])
    return config, receipt, model


def make_decoder(arm, model_path, graph_path=None):
    frozen = load_tcn(model_path)
    if arm in "AB":
        return TCNStream(frozen, "window" if arm == "A" else "incremental")
    return ONNXTCNStream(
        graph_path,
        expected_model_identity=model_fingerprint(frozen),
        precision="dynamic_int8" if arm == "E" else "float32",
    )


class BoundedWriter:
    """Fixed-size user-space buffer backed by a preallocated, durable NPY mapping."""

    def __init__(self, path, records, capacity=4096):
        if records < 1 or capacity < 1:
            raise ValueError("positive log extent and capacity required")
        if Path(path).exists():
            raise FileExistsError(path)
        self.mapping = np.lib.format.open_memmap(
            path, mode="w+", dtype=RECORD_DTYPE, shape=(records,)
        )
        self.buffer = np.empty(capacity, dtype=RECORD_DTYPE)
        self.used = self.written = self.high_water = self.flushes = 0
        self.flush_ns = 0

    def append(self, record):
        if self.written + self.used >= len(self.mapping):
            raise ValueError("log extent exceeded; no sample can be dropped")
        self.buffer[self.used] = record
        self.used += 1
        self.high_water = max(self.high_water, self.used)
        if self.used == len(self.buffer):
            self.flush()

    def flush(self):
        if self.used:
            started = time.perf_counter_ns()
            self.mapping[self.written : self.written + self.used] = self.buffer[: self.used]
            self.mapping.flush()
            self.written += self.used
            self.used = 0
            self.flushes += 1
            self.flush_ns += time.perf_counter_ns() - started

    def stats(self):
        return {
            "capacity_records": len(self.buffer),
            "buffer_nbytes": self.buffer.nbytes,
            "allocated_records": len(self.mapping),
            "written_records": self.written,
            "high_water_records": self.high_water,
            "flushes": self.flushes,
            "flush_ns": self.flush_ns,
            "dropped_records": 0,
            "durable_bytes": self.mapping.nbytes,
            "writer_queue": "synchronous; no queue",
        }


def file_segments(files, mode, total_bins):
    """Yield real file prefixes with composite cycle/file identity; never pad bins."""
    remaining, cycle = total_bins, 0
    while remaining:
        for index, item in enumerate(files):
            bins = min(item["n_bins"], remaining)
            yield cycle, index, bins
            remaining -= bins
            if not remaining:
                return
            if mode == "paced":
                raise ValueError("short paced workload exceeds its real first file")
        if mode == "fast":
            raise ValueError("fast workload exceeds one complete canonical cycle")
        cycle += 1


def replay(
    decoder,
    files,
    arrays,
    *,
    mode,
    total_bins,
    period_ns,
    writer,
    resets,
    rss_log,
    clock=time.perf_counter_ns,
    sleep=time.sleep,
    process=None,
    rss_interval_ns=1_000_000_000,
):
    """One release origin spans all file resets and log flushes, including soak cycles."""
    process = process or psutil.Process()
    origin, global_id = clock(), 0
    next_rss = origin
    max_rss = 0
    for cycle, file_index, bins in file_segments(files, mode, total_bins):
        reset_start = clock()
        decoder.reset()
        reset_complete = clock()
        resets.write(
            json.dumps(
                {
                    "global_id": global_id,
                    "cycle": cycle,
                    "file_index": file_index,
                    "file_id": files[file_index]["file_id"],
                    "start_ns": reset_start,
                    "complete_ns": reset_complete,
                    "bins": bins,
                    "partial": bins < files[file_index]["n_bins"],
                }
            )
            + "\n"
        )
        counts, timestamps = arrays[file_index]
        for sequence in range(bins):
            if mode != "fast":
                release = origin + global_id * period_ns
                while (remaining := release - clock()) > 0:
                    sleep(remaining / 1e9)
                start = clock()
            else:
                start = clock()
                release = start
            prediction = decoder.step(counts[sequence])
            complete = clock()
            writer.append(
                (
                    global_id,
                    cycle,
                    file_index,
                    sequence,
                    timestamps[sequence],
                    release,
                    start,
                    complete,
                    complete,
                    release + period_ns,
                    prediction,
                )
            )
            global_id += 1
            if complete >= next_rss:
                observed = clock()
                rss = process.memory_info().rss
                max_rss = max(max_rss, rss)
                rss_log.write(
                    json.dumps({"global_id": global_id - 1, "clock_ns": observed, "rss_bytes": rss})
                    + "\n"
                )
                next_rss = observed + rss_interval_ns
    writer.flush()
    return {"release_origin_ns": origin, "records": global_id, "sampled_max_rss_bytes": max_rss}


def load_workload(index_path, config):
    index = checked_development_index(index_path)
    files = sorted(index["files"], key=lambda row: (int(row["day"][1:]), row["file_id"]))
    if [row["file_id"] for row in files] != config["runtime"]["fast_file_ids"]:
        raise ValueError("development file identities or canonical order changed")
    if sum(row["n_bins"] for row in files) != config["runtime"]["fast_bins"]:
        raise ValueError("development bin inventory changed")
    arrays = []
    for row in files:
        bundle = load_feature_file(index_path.parent / row["features"], row["features_sha256"])
        if not np.array_equal(bundle["channel_ids"], np.arange(176)) or bundle["counts"].shape != (
            row["n_bins"],
            176,
        ):
            raise ValueError("feature-only stream metadata changed")
        arrays.append((bundle["counts"], bundle["timestamps"]))
    return index, files, arrays


def run_worker(request_path):
    request = read_json(request_path)
    root, output = Path(request["root"]), Path(request["output"])
    config = read_json(Path(request["campaign"]) / "config.json")
    runtime = validate_runtime_config(config)
    status = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "request_sha256": sha256(request_path),
        "pid": os.getpid(),
    }
    write_json(output / "start.json", status)
    write_json(output / "status.json", status)
    writer = None
    try:
        verify_sources(root, request["source"])
        model = root / request["model"]["path"]
        verify(model, request["model"]["sha256"])
        graph = request.get("graph")
        graph_path = None if graph is None else Path(request["quality_run"]) / graph["path"]
        if graph is not None:
            verify(graph_path, graph["sha256"])
        index_path = root / runtime["workload_index"]["path"]
        verify(index_path, runtime["workload_index"]["sha256"])
        index, files, arrays = load_workload(index_path, config)
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            constructor_start = time.perf_counter_ns()
            decoder = make_decoder(request["arm"], model, graph_path)
            constructor_ns = time.perf_counter_ns() - constructor_start
            for row in arrays[0][0][: runtime["warmup_bins"]]:
                decoder.step(row)
            warmup_complete_ns = time.perf_counter_ns()
            process = psutil.Process()
            rss_post_warmup = process.memory_info().rss
            mode = request["mode"]
            bins = runtime[{"fast": "fast_bins", "paced": "paced_bins", "soak": "soak_bins"}[mode]]
            writer = BoundedWriter(output / "records.npy", bins, runtime["logger_buffer_records"])
            cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
            with (
                (output / "resets.jsonl").open("x") as resets,
                (output / "rss.jsonl").open("x") as rss,
            ):
                replay_result = replay(
                    decoder,
                    files,
                    arrays,
                    mode=mode,
                    total_bins=bins,
                    period_ns=runtime["period_ns"],
                    writer=writer,
                    resets=resets,
                    rss_log=rss,
                    process=process,
                    rss_interval_ns=runtime["rss_sample_interval_ns"],
                )
            wall_ns = time.perf_counter_ns() - wall_start
            cpu_ns = time.process_time_ns() - cpu_start
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_after_replay = process.memory_info().rss
            # Arithmetic summarization and file hashing occur after CPU/wall accounting stops.
            records = writer.mapping
            stamps = np.column_stack([records[name] for name in STAMP_FIELDS])
            summary = timing_summary(stamps, period_ns=runtime["period_ns"])
            summary["deadline_miss_rate"] = summary["missed_deadlines"] / bins
            summary["service_goal_passed"] = summary["step_service"]["p99_ms"] < 2
            reset_records = [
                json.loads(line) for line in (output / "resets.jsonl").read_text().splitlines()
            ]
            summary.update(
                {
                    "arm": request["arm"],
                    "mode": mode,
                    "repeat": request["repeat"],
                    "position": request["position"],
                    "constructor_ns": constructor_ns,
                    "file_reset_ns": [
                        row["complete_ns"] - row["start_ns"] for row in reset_records
                    ],
                    "warmup_bins": runtime["warmup_bins"],
                    "warmup_complete_ns": warmup_complete_ns,
                    "numeric_state_bytes": decoder.state_nbytes,
                    "model_bytes": model.stat().st_size,
                    "graph_bytes": None if graph_path is None else graph_path.stat().st_size,
                    "rss_post_warmup_bytes": rss_post_warmup,
                    "rss_after_replay_bytes": rss_after_replay,
                    "process_peak_rss_bytes": int(
                        peak * (1 if platform.system() == "Darwin" else 1024)
                    ),
                    "input_cache_bytes": sum(x.nbytes + t.nbytes for x, t in arrays),
                    "cpu_ns": cpu_ns,
                    "wall_ns": wall_ns,
                    "logger": writer.stats(),
                    "environment": environment(),
                    "threadpools": threadpool_info(),
                    "load_average_end": os.getloadavg(),
                    "replay": replay_result,
                    "file_ids": [row["file_id"] for row in files],
                    "source_index_sha256": sha256(index_path),
                    "model_sha256": sha256(model),
                    "graph_sha256": None if graph_path is None else sha256(graph_path),
                    "raw_sha256": {
                        name: sha256(output / name)
                        for name in ("records.npy", "resets.jsonl", "rss.jsonl")
                    },
                }
            )
            verify_sources(root, request["source"])
            write_json(output / "summary.json", summary)
            status.update(status="complete", summary_sha256=sha256(output / "summary.json"))
    except BaseException as error:
        if writer is not None:
            writer.flush()
            status["partial_logger"] = writer.stats()
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        status["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(output / "status.json", status)


def run_campaign(
    config_path, quality_run, quality_audit, output, *, mode="benchmark", lifecycle=None, root=ROOT
):
    config_path, quality_run, quality_audit, output = map(
        lambda value: Path(value).resolve(), (config_path, quality_run, quality_audit, output)
    )
    root = Path(root).resolve()
    if not output.is_relative_to((root / "results/neural-h1").resolve()):
        raise ValueError("private runtime evidence must remain under results/neural-h1")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "kind": "h1_final_closure_runtime",
        "status": "running",
        "command": sys.argv,
        "started_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "environment": environment(),
        "harness_sha256": sha256(__file__),
        "quality_run": str(quality_run),
        "quality_audit": str(quality_audit),
        "statuses": {
            "implementation_execution": "running",
            "artifact_completion": "partial",
            "independent_verification": "not_run",
            "float32_equivalence": "not_verified",
            "compression_quality": "not_verified",
            "runtime_goals": "not_evaluated",
        },
    }
    write_json(output / "start.json", manifest)
    write_json(output / "manifest.json", manifest)
    try:
        (output / "config.json").write_bytes(config_path.read_bytes())
        manifest["config_sha256"] = sha256(config_path)
        write_json(output / "manifest.json", manifest)
        config, receipt, model = admit_quality(config_path, quality_run, quality_audit, root)
        source = source_snapshot(root, output, config)
        manifest["statuses"].update(
            float32_equivalence=receipt["gates"]["float32_equivalence"],
            compression_quality=receipt["gates"]["compression_quality"],
        )
        qualified = receipt["qualified_arms"]
        eligible = [arm for arm in "ABCDE" if qualified[arm]]
        exclusions = {
            arm: "quality-ineligible or unsupported; see authenticated quality audit"
            for arm in "ABCDE"
            if arm not in eligible
        }
        if mode == "soak":
            if "B" not in eligible:
                raise ValueError("mandatory B soak failed its quality admission")
            eligible = ["B"]
            if qualified["E"]:
                if lifecycle is None:
                    raise ValueError("qualified E soak requires an authenticated lifecycle receipt")
                from neural_experiments.h1_final_closure_recovery import validate_lifecycle_receipt

                lifecycle_receipt = validate_lifecycle_receipt(
                    lifecycle, quality_run, quality_audit, root
                )
                if lifecycle_receipt["qualified_arms"].get("E"):
                    eligible.append("E")
                else:
                    exclusions["E"] = "lifecycle failed; retained separate recovery result"
                manifest["lifecycle"] = {
                    "path": str(Path(lifecycle).resolve()),
                    "sha256": sha256(lifecycle),
                }
            jobs = [("soak", 0, i, arm) for i, arm in enumerate(eligible)]
        else:
            jobs = [
                (run_mode, repeat, position, arm)
                for run_mode in ("fast", "paced")
                for repeat, order in enumerate(cyclic_orders(eligible))
                for position, arm in enumerate(order)
            ]
        estimate = (
            sum(
                config["runtime"][
                    {"fast": "fast_bins", "paced": "paced_bins", "soak": "soak_bins"}[job[0]]
                ]
                * RECORD_DTYPE.itemsize
                for job in jobs
            )
            + 20_000_000
        )
        free = shutil.disk_usage(output).free
        if estimate > config["resources"]["max_new_output_bytes"] or free < estimate:
            raise ValueError("runtime log estimate exceeds storage budget or available disk")
        power = "unavailable on this platform"
        chip = platform.processor() or "unavailable"
        if platform.system() == "Darwin":
            observed = subprocess.run(
                ["/usr/bin/pmset", "-g"], capture_output=True, text=True, check=False
            )
            power = observed.stdout if observed.returncode == 0 else observed.stderr
            observed_chip = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            chip = observed_chip.stdout.strip() if observed_chip.returncode == 0 else "unavailable"
        manifest.update(
            config_sha256=sha256(config_path),
            source=source,
            quality_run=str(quality_run),
            quality_audit=str(quality_audit),
            quality_audit_sha256=sha256(quality_audit),
            quality_manifest_sha256=sha256(quality_run / "manifest.json"),
            quality_summary_sha256=sha256(quality_run / "summary.json"),
            qualified_arms=qualified,
            exclusions=exclusions,
            eligible_arms=eligible,
            orders=cyclic_orders(eligible) if mode == "benchmark" else [eligible],
            workload_index=config["runtime"]["workload_index"],
            model=model["model"],
            hardware={
                "processor": chip,
                "logical_cpus": os.cpu_count(),
                "physical_cpus": psutil.cpu_count(logical=False),
                "ram_bytes": psutil.virtual_memory().total,
                "power_settings_observation": power,
                "background_workload": "user workload uncontrolled; process inventory unavailable",
            },
            storage={"estimated_new_bytes": estimate, "free_bytes_before": free},
            boundary="counts-ready numeric engine; ready equals complete; logging outside service",
            process_memory_scope="fresh process incl. imports, model, input cache and mapped log",
            percentile="numpy.percentile method=linear",
            host_clock={
                "function": "time.perf_counter_ns",
                **vars(time.get_clock_info("perf_counter")),
            },
            blocks=[],
        )
        write_json(output / "manifest.json", manifest)
        failures = []
        for run_mode, repeat, position, arm in jobs:
            verify_sources(root, source)
            folder = output / f"{run_mode}-{repeat}-{position}-{arm}"
            folder.mkdir()
            request = {
                "root": str(root),
                "campaign": str(output),
                "output": str(folder),
                "quality_run": str(quality_run),
                "source": source,
                "model": model["model"],
                "graph": model["arms"][arm]["graph"],
                "arm": arm,
                "mode": run_mode,
                "repeat": repeat,
                "position": position,
            }
            write_json(folder / "request.json", request)
            command = [
                sys.executable,
                str(root / "neural_experiments/h1_final_closure_runtime.py"),
                "_worker",
                "--request",
                str(folder / "request.json"),
            ]
            env = dict(os.environ)
            env.update(
                {
                    key: "1"
                    for key in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "PYTHONDONTWRITEBYTECODE",
                    )
                }
            )
            with (folder / "process.log").open("x") as log:
                result = subprocess.run(
                    command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            entry = {
                "folder": folder.name,
                "command": command,
                "returncode": result.returncode,
                "request_sha256": sha256(folder / "request.json"),
                "start_sha256": sha256(folder / "start.json")
                if (folder / "start.json").exists()
                else None,
                "status_sha256": sha256(folder / "status.json")
                if (folder / "status.json").exists()
                else None,
            }
            if result.returncode or not (folder / "summary.json").exists():
                failures.append(folder.name)
            else:
                entry["summary_sha256"] = sha256(folder / "summary.json")
            manifest["blocks"].append(entry)
            write_json(output / "manifest.json", manifest)
            print(
                json.dumps(
                    {
                        "completed_block": folder.name,
                        "returncode": result.returncode,
                        "blocks_done": len(manifest["blocks"]),
                        "blocks_total": len(jobs),
                    }
                ),
                flush=True,
            )
        if failures:
            raise RuntimeError(f"failed blocks retained: {failures}")
        summaries = [
            read_json(output / row["folder"] / "summary.json") for row in manifest["blocks"]
        ]
        write_json(output / "summary.json", {"runs": summaries, "exclusions": exclusions})
        manifest.update(status="complete", summary_sha256=sha256(output / "summary.json"))
        manifest["statuses"].update(
            implementation_execution="complete",
            artifact_completion="complete",
            runtime_goals="passed"
            if all(row["service_goal_passed"] for row in summaries)
            else "failed",
        )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        manifest["statuses"]["implementation_execution"] = "failed"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("benchmark", "soak"):
        subparser = subparsers.add_parser(command)
        for argument in ("config", "quality-run", "quality-audit", "output"):
            subparser.add_argument(f"--{argument}", type=Path, required=True)
        subparser.add_argument("--lifecycle", type=Path)
    subparsers.add_parser("_worker").add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "_worker":
        run_worker(args.request)
    else:
        run_campaign(
            args.config,
            args.quality_run,
            args.quality_audit,
            args.output,
            mode=args.command,
            lifecycle=args.lifecycle,
        )


if __name__ == "__main__":
    main()
